"""User-triggered alpha actions: submission check, submit (always user-confirmed), properties, PnL, correlations."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from ..brain import brain
from ..brain.errors import BrainError
from ..brain.quota import EASTERN
from ..config import settings
from ..db import db
from ..db.models import Alpha, Submission
from ..events import bus
from ..jobs.queue import JobContext, jobs
from .evaluator import blocking_failures, check_state_of, pending_checks, reverdict
from .ingest import upsert_alpha

log = logging.getLogger(__name__)

PENDING_POLLS = 3            # /check answers still PENDING: ask this many more times before giving up for now
PENDING_WAIT_S = 20.0
MAX_CHECK_TRIES = 3          # unanswered /check results re-asked before the alpha waits for you
RECHECK_AFTER = timedelta(minutes=30)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def refresh_alpha(alpha_id: str) -> Alpha:
    data = await brain.alphas.get(alpha_id)
    return await upsert_alpha(data, origin="synced")


async def run_check(alpha_id: str) -> tuple[str, list[dict[str, Any]]]:
    """Run BRAIN's /check. Returns (check_state, checks). Marks the alpha 'ready' when everything passes.

    An unusable reply (no checks) changes nothing but the try count: an empty body must never read as a pass, and it
    must not wipe an earlier result. Checks still PENDING are asked again a few times before returning."""
    bus.publish("check", f"Running submission check for {alpha_id}…")
    checks = await brain.alphas.check(alpha_id)
    for _ in range(PENDING_POLLS):
        if not checks or blocking_failures(checks) or not pending_checks(checks):
            break
        await asyncio.sleep(PENDING_WAIT_S)
        checks = await brain.alphas.check(alpha_id)
    if not checks:
        async with db.write() as s:
            row = await s.get(Alpha, alpha_id)
            if row is not None:
                row.check_tries = (row.check_tries or 0) + 1
                row.checked_at = row.checked_at or utcnow()
                row.check_state = row.check_state or "pending"
                reverdict(row)
        bus.publish("check", f"{alpha_id}: BRAIN's check returned no results — nothing changed; it will be asked "
                             "again.", "warning")
        return "pending", []
    # one rule for "blocking" (evaluator.NON_BLOCKING): an informational FAIL such as POWER_POOL_CORRELATION used to
    # mark the check failed while the verdict ignored it, so such an alpha could never become "ready"
    failing = blocking_failures(checks)
    state = check_state_of(checks)
    self_corr = next((c.get("value") for c in checks if c.get("name") == "SELF_CORRELATION"), None)
    prod_corr = next((c.get("value") for c in checks if c.get("name") == "PROD_CORRELATION"), None)
    async with db.write() as s:
        row = await s.get(Alpha, alpha_id)
        if row is None:
            row = Alpha(id=alpha_id)
            s.add(row)
        row.checks = checks
        row.check_state = state
        row.checked_at = utcnow()
        row.check_tries = (row.check_tries or 0) + 1 if state == "pending" else 0
        if self_corr is not None:
            row.self_corr = self_corr
        if prod_corr is not None:
            row.prod_corr = prod_corr
        reverdict(row)
        estimate, closest = row.est_self_corr, row.est_corr_with
    if self_corr is not None and estimate is not None:
        await _remember_calibration(alpha_id, estimate, self_corr, closest)
    try:                                   # repairs aimed at correlation are judged by this check
        from .refine import resolve_attempts
        await resolve_attempts(child_alpha_id=alpha_id)
    except Exception:
        log.exception("could not settle repair attempts for %s", alpha_id)
    if state == "pass":
        bus.publish("check", f"{alpha_id} passed every submission check — ready for your review.", "success",
                    alpha_id=alpha_id, kind="ready")
    else:
        names = ", ".join(failing) or "pending checks"
        bus.publish("check", f"{alpha_id} check: {state.upper()} ({names}).", "warning" if state == "fail" else "info")
    return state, checks


CALIBRATION_KEY = "selfcorr_calibration"


async def _remember_calibration(alpha_id: str, estimate: float, real: Any, closest: str | None) -> None:
    """Keep (local estimate, BRAIN's SELF_CORRELATION) pairs so the estimate's error can be shown."""
    try:
        real_v = float(real)
    except (TypeError, ValueError):
        return
    structural, struct_with = None, None
    try:                                   # the pre-simulation similarity score, to calibrate the duplicate filter
        from .similarity import _operators, best_structural_match, submitted_shapes
        async with db.session() as s:
            a = await s.get(Alpha, alpha_id)
        if a is not None and a.region:
            shapes = [x for x in await submitted_shapes(a.region, a.delay) if x.alpha_id != alpha_id]
            structural, struct_with = best_structural_match(a.skeleton_hash, set(a.fields or []),
                                                            _operators(a.code), shapes)
    except Exception:
        log.exception("structural similarity failed for %s", alpha_id)
    rows = [r for r in (await db.kv_get(CALIBRATION_KEY) or []) if r.get("alpha_id") != alpha_id]
    rows.append({"alpha_id": alpha_id, "estimate": round(float(estimate), 4), "real": round(real_v, 4),
                 "closest": closest, "structural": structural, "structural_with": struct_with,
                 "ts": utcnow().isoformat(timespec="seconds")})
    await db.kv_set(CALIBRATION_KEY, rows[-500:])


async def mark_stale_after_submission(alpha_id: str) -> int:
    """A new submission changes every other alpha's self-correlation: earlier "pass" results in the same
    region/delay are no longer trustworthy until BRAIN checks them again."""
    async with db.session() as s:
        row = await s.get(Alpha, alpha_id)
    if row is None or row.region is None:
        return 0
    async with db.write() as s:
        stale = list(await s.scalars(select(Alpha).where(
            Alpha.id != alpha_id, Alpha.region == row.region, Alpha.delay == row.delay,
            Alpha.check_state == "pass", Alpha.status == "UNSUBMITTED")))
        for a in stale:
            a.check_state = "stale"
            reverdict(a)
    if stale:
        bus.publish("check", f"{len(stale)} alpha(s) in {row.region} D{row.delay} need a fresh BRAIN check after "
                             f"submitting {alpha_id} (self-correlation changes with every submission).", "info")
    return len(stale)


async def rederive_check_states() -> int:
    """Re-read stored /check results with the current rules, so alphas an older rule left stuck come free. Runs at
    startup; touches only alphas whose stored result now reads differently.

    - an informational FAIL (POWER_POOL_CORRELATION) or a daily-limit check no longer fails the whole check;
    - an ERROR is unanswered, not failed, so it goes back to pending;
    - a "pass" stored from an empty /check reply was never a pass: the alpha goes back to its simulation's checks.
    It never turns "pending" into "pass": a pending row can hold an older result, and only BRAIN can pass an alpha."""
    changed = 0
    async with db.write() as s:
        rows = list(await s.scalars(select(Alpha).where(Alpha.check_state.in_(["fail", "pending", "pass"]),
                                                        Alpha.checked_at.is_not(None))))
        for row in rows:
            if row.check_state == "pass" and row.checks:
                continue
            if row.check_state == "pass":
                row.checks, row.check_state, row.checked_at = row.is_checks, None, None
                reverdict(row)
                changed += 1
                continue
            state = check_state_of(row.checks)
            if state != row.check_state and not (state == "pass" and row.check_state == "pending"):
                row.check_state = state
                reverdict(row)
                changed += 1
    if changed:
        bus.publish("check", f"{changed} alpha(s) re-read with the corrected check rules (informational checks and "
                             "BRAIN errors no longer fail a check; empty results never pass).", "info")
    return changed


async def recheck_due(limit: int = 5) -> list[str]:
    """Alphas whose earlier /check is still unanswered and worth asking again: strong, not asked in the last half hour,
    fewer than MAX_CHECK_TRIES unanswered replies. Only ever finishes a check that already started."""
    cutoff = utcnow() - RECHECK_AFTER
    async with db.session() as s:
        return list(await s.scalars(select(Alpha.id).where(
            Alpha.check_state == "pending", Alpha.verdict == "strong", Alpha.checked_at < cutoff,
            (Alpha.check_tries.is_(None)) | (Alpha.check_tries < MAX_CHECK_TRIES))
            .order_by(Alpha.checked_at).limit(limit)))


async def recheck_loop(interval_s: float = 300.0) -> None:
    """Background: finish checks BRAIN left unanswered. Costs no simulation quota, one call at a time, and sits out
    while signed out or after Stop all. Separate from the daily scheduler, because one /check can take minutes."""
    from .audit import ACTOR
    from .simulator import simulator
    ACTOR.set("recheck")
    await asyncio.sleep(60)
    while True:
        try:
            if brain.client.logged_in and not simulator.paused:
                for alpha_id in await recheck_due():
                    if not brain.client.logged_in or simulator.paused:
                        break
                    try:
                        await run_check(alpha_id)
                    except BrainError as exc:      # counts as a try, so a broken alpha cannot be asked forever
                        log.info("re-check of %s failed: %s", alpha_id, exc.message)
                        await _count_try(alpha_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("re-check sweep failed")
        await asyncio.sleep(interval_s)


async def _count_try(alpha_id: str) -> None:
    async with db.write() as s:
        row = await s.get(Alpha, alpha_id)
        if row is not None:
            row.check_tries = (row.check_tries or 0) + 1
            row.checked_at = utcnow()


async def submissions_today(kind: str = "REGULAR") -> int:
    start = datetime.now(EASTERN).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    async with db.session() as s:
        return await s.scalar(select(func.count(Submission.id)).where(
            Submission.kind == kind, Submission.requested_at >= start,
            Submission.status.in_(["accepted", "active", "requested"]))) or 0


async def submit_alpha(alpha_id: str, *, confirmed: bool) -> str:
    """Submit to BRAIN. `confirmed` must come from an explicit user click in a confirmation dialog."""
    if not confirmed:
        raise PermissionError("Submission requires explicit user confirmation.")
    async with db.session() as s:
        row = await s.get(Alpha, alpha_id)
    kind = (row.type if row else None) or "REGULAR"
    limit = settings.programs.daily_super_submissions if kind == "SUPER" else settings.programs.daily_regular_submissions
    if await submissions_today(kind) >= limit:
        raise RuntimeError(f"Daily {kind.lower()} submission limit ({limit}) reached — BRAIN would only count the first "
                           f"{limit} today.")
    async with db.write() as s:
        sub = Submission(alpha_id=alpha_id, kind=kind)
        s.add(sub)
        await s.flush()
        sub_id = sub.id
    bus.publish("submit", f"Submitting {alpha_id} to BRAIN…")
    result = await brain.alphas.submit(alpha_id)
    async with db.write() as s:
        sub = await s.get(Submission, sub_id)
        sub.status = result.status if result.status in ("active", "accepted", "rejected") else "error"
        sub.result = {"message": result.message, "checks": result.checks}
        sub.finished_at = utcnow()
    if result.status in ("active", "accepted"):
        await refresh_alpha(alpha_id)
        await mark_stale_after_submission(alpha_id)
        bus.publish("submit", f"{alpha_id} submitted ({result.status}).", "success")
    else:
        bus.publish("submit", f"{alpha_id} not submitted: {result.status} — {result.message}", "error")
    return f"{result.status}: {result.message}"


async def set_properties(alpha_id: str, **props: Any) -> None:
    data = await brain.alphas.patch(alpha_id, **props)
    if isinstance(data, dict) and data.get("id"):
        await upsert_alpha(data, origin="synced")
    else:
        await refresh_alpha(alpha_id)


async def pnl_series(alpha_id: str) -> list[dict[str, Any]]:
    cached = await db.kv_get(f"pnl:{alpha_id}")
    if cached:
        return cached
    records = await brain.alphas.recordset(alpha_id, "pnl")
    await db.kv_set(f"pnl:{alpha_id}", records)
    return records


async def yearly_stats(alpha_id: str) -> list[dict[str, Any]]:
    cached = await db.kv_get(f"yearly:{alpha_id}")
    if cached:
        return cached
    records = await brain.alphas.recordset(alpha_id, "yearly-stats")
    await db.kv_set(f"yearly:{alpha_id}", records)
    return records


async def correlations(alpha_id: str, kind: str = "self") -> dict[str, Any]:
    result = await brain.alphas.correlations(alpha_id, kind)
    async with db.write() as s:
        row = await s.get(Alpha, alpha_id)
        if row is not None and result.get("max") is not None:
            if kind == "self":
                row.self_corr = result["max"]
            elif kind == "prod":
                row.prod_corr = result["max"]
    return result


# ---------------------------------------------------------------- jobs

@jobs.register("check_alpha")
async def _job_check(ctx: JobContext, params: dict) -> str:
    state, _ = await run_check(params["alpha_id"])
    return state


@jobs.register("check_many")
async def _job_check_many(ctx: JobContext, params: dict) -> str:
    ids = list(params.get("alpha_ids") or [])
    passed = 0
    for i, aid in enumerate(ids):
        try:
            state, _ = await run_check(aid)
            passed += state == "pass"
        except Exception as exc:  # keep going
            log.warning("check %s failed: %s", aid, exc)
        await ctx.update((i + 1) / max(1, len(ids)), f"{i + 1}/{len(ids)} checked, {passed} passed")
    return f"{passed}/{len(ids)} passed"


@jobs.register("submit_alpha")
async def _job_submit(ctx: JobContext, params: dict) -> str:
    return await submit_alpha(params["alpha_id"], confirmed=bool(params.get("confirmed")))


@jobs.register("refresh_alphas")
async def _job_refresh(ctx: JobContext, params: dict) -> str:
    ids = list(params.get("alpha_ids") or [])
    for i, aid in enumerate(ids):
        await refresh_alpha(aid)
        await ctx.update((i + 1) / max(1, len(ids)))
    return f"{len(ids)} refreshed"
