"""BRAIN programs: Power Pool, pyramids, Genius, Osmosis allocation, out-of-sample tracking.

Rules come from BRAIN Learn-page copies (2025–26) and are editable where the platform may have changed them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import func, select

from ..brain import brain
from ..brain.quota import EASTERN
from ..config import settings
from ..db import db
from ..db.models import Alpha, ConsultantSnapshot, Dataset, OsmosisAllocation, Pyramid, Submission
from ..events import bus
from ..fastexpr import compute_metrics, parse
from ..jobs.queue import JobContext, jobs
from . import actions
from .sync import current_quarter, quarter_bounds

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- Power Pool

GROUPING_FIELDS = {"country", "industry", "subindustry", "currency", "market", "sector", "exchange"}
RISK_HANDLED = {"STATISTICAL", "CROWDING", "FAST", "SLOW", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM"}
PP_REGIONS = {"USA", "EUR", "ASI", "GLB", "JPN", "CHN", "HKG", "TWN", "KOR", "AMR"}
PP_RISK_REQUIRED = {"USA", "EUR", "ASI", "GLB", "CHN"}
PP_TAG = "PowerPoolSelected"


def power_pool_shape(expr: str) -> tuple[int, set[str]]:
    m = compute_metrics(parse(expr))
    fields = {f for f in m.non_group_fields if f not in GROUPING_FIELDS}
    return m.operator_count, fields


def power_pool_shape_ok(expr: str) -> bool:
    try:
        ops, fields = power_pool_shape(expr)
    except Exception:
        return False
    return ops <= 8 and len(fields) <= 3


@dataclass
class PPResult:
    alpha_id: str
    eligible: bool
    reasons: list[str]
    ops: int | None
    fields: int | None
    tagged: bool
    description_ok: bool


def power_pool_eligibility(a: Alpha) -> PPResult:
    reasons: list[str] = []
    ops = nf = None
    if not a.code or a.type != "REGULAR":
        return PPResult(a.id, False, ["not a regular alpha"], None, None, False, False)
    try:
        ops, fields = power_pool_shape(a.code)
        nf = len(fields)
        if ops > 8:
            reasons.append(f"{ops} operators > 8")
        if nf > 3:
            reasons.append(f"{nf} data fields > 3")
    except Exception:
        reasons.append("unparsable expression")
    if (a.sharpe or 0) < 1.0:
        reasons.append(f"Sharpe {a.sharpe or 0:.2f} < 1.0")
    t = a.turnover or 0
    if not 0.01 <= t <= 0.7:
        reasons.append(f"turnover {t:.1%} outside 1–70%")
    sub = next((c for c in a.checks or [] if c.get("name") == "LOW_SUB_UNIVERSE_SHARPE"), None)
    if sub and str(sub.get("result")).upper() == "FAIL":
        reasons.append("fails sub-universe Sharpe")
    if (a.region or "") not in PP_REGIONS:
        reasons.append(f"region {a.region} not in Power Pool")
    if (a.region or "") in PP_RISK_REQUIRED and (a.neutralization or "") not in RISK_HANDLED:
        reasons.append(f"{a.region} needs risk-handled neutralization ({a.neutralization})")
    desc_ok = len(a.description or "") >= 100
    if not desc_ok:
        reasons.append("description < 100 characters")
    tagged = PP_TAG in (a.tags or [])
    return PPResult(a.id, not reasons, reasons, ops, nf, tagged, desc_ok)


async def set_power_pool_tag(alpha_id: str, on: bool) -> None:
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
    tags = [t for t in (a.tags or []) if t != PP_TAG] if a else []
    if on:
        tags.append(PP_TAG)
    await actions.set_properties(alpha_id, tags=tags)


@jobs.register("pp_tags")
async def _job_pp_tags(ctx: JobContext, params: dict) -> str:
    """Bulk PowerPoolSelected tag changes (only ever started from a confirmation dialog)."""
    tag, untag = list(params.get("tag") or []), list(params.get("untag") or [])
    work = [(a, True) for a in tag] + [(a, False) for a in untag]
    errors: list[str] = []
    for i, (aid, on) in enumerate(work, 1):
        try:
            await set_power_pool_tag(aid, on)
        except Exception as exc:
            errors.append(f"{aid}: {getattr(exc, 'message', exc)}")
        await ctx.update(i / max(1, len(work)), f"{i}/{len(work)} tag changes")
    bus.publish("sync", f"Power Pool tags: {len(tag)} tagged, {len(untag)} untagged ({len(errors)} errors).",
                "success" if not errors else "warning")
    if errors and len(errors) == len(work):
        raise RuntimeError("; ".join(errors[:5]))
    return f"{len(tag)} tagged, {len(untag)} untagged" + (f", {len(errors)} errors" if errors else "")


async def power_pool_monthly() -> dict[str, int]:
    start = date.today().replace(day=1)
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE", Alpha.date_submitted >= datetime(
            start.year, start.month, 1, tzinfo=timezone.utc))))
    counts: dict[str, int] = {}
    for a in rows:
        if PP_TAG in (a.tags or []):
            key = f"{a.region}/D{a.delay}"
            counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------- pyramids

async def pyramid_cells(quarter: str | None = None) -> list[Pyramid]:
    async with db.session() as s:
        return list(await s.scalars(select(Pyramid).where(Pyramid.quarter == (quarter or current_quarter()))
                                    .order_by(Pyramid.region, Pyramid.delay, Pyramid.category_name)))


async def alpha_category(a: Alpha) -> tuple[str | None, str | None]:
    if not a.dataset_ids:
        return None, None
    ds_id = str(a.dataset_ids[0]).split(",")[0]
    async with db.session() as s:
        ds = await s.scalar(select(Dataset).where(Dataset.id == ds_id, Dataset.region == a.region,
                                                  Dataset.delay == a.delay))
    return (ds.category_id, ds.category_name) if ds else (None, None)


async def pyramid_candidates(limit: int = 50) -> list[dict[str, Any]]:
    """Unsubmitted ready/strong alphas whose pyramid (region × delay × category) is not yet lit (< 3 this quarter)."""
    cells = {(p.category_id, p.region, p.delay): p for p in await pyramid_cells()}
    async with db.session() as s:
        alphas = list(await s.scalars(select(Alpha).where(Alpha.status == "UNSUBMITTED",
                                                          Alpha.verdict.in_(["ready", "strong"]))
                                      .order_by(Alpha.score.desc().nulls_last()).limit(300)))
    out = []
    for a in alphas:
        cat_id, cat_name = await alpha_category(a)
        if not cat_id:
            continue
        cell = cells.get((cat_id, a.region, a.delay))
        count = cell.alpha_count if cell else 0
        if count < 3:
            out.append({"alpha_id": a.id, "pyramid": f"{a.region}/D{a.delay}/{cat_name}", "current": count,
                        "multiplier": cell.multiplier if cell else None, "sharpe": a.sharpe, "verdict": a.verdict})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- Genius

LEVELS = ["GOLD", "EXPERT", "MASTER", "GRANDMASTER"]


async def latest_summary() -> dict[str, Any] | None:
    async with db.session() as s:
        row = await s.scalar(select(ConsultantSnapshot).order_by(ConsultantSnapshot.id.desc()).limit(1))
    return row.payload if row else None


async def quarter_signals() -> int:
    start, _ = quarter_bounds()
    start_dt = datetime(start.year, start.month, start.day, tzinfo=EASTERN)
    async with db.session() as s:
        n = await s.scalar(select(func.count(Alpha.id)).where(Alpha.status == "ACTIVE",
                                                              Alpha.date_submitted >= start_dt))
    return n or 0


def _norm_level(level: Any) -> str:
    return str(level or "GOLD").upper().replace(" ", "").replace("_", "")


async def genius_status() -> dict[str, Any]:
    summary = await latest_summary() or {}
    level = _norm_level(summary.get("geniusLevel"))
    level = level if level in LEVELS else "GOLD"
    idx = LEVELS.index(level)
    target = LEVELS[min(idx + 1, len(LEVELS) - 1)]
    thresholds = settings.programs.genius_thresholds
    t_signals, t_pyr, t_perf = thresholds.get(target, [0, 0, 0])
    signals = await quarter_signals()
    cells = await pyramid_cells()
    lit = sum(1 for c in cells if c.alpha_count >= 3) or int(summary.get("pyramidCount") or 0)
    perf = summary.get("combinedAlphaPerformance")
    try:
        perf_v = float(perf) if perf is not None else None
    except (TypeError, ValueError):
        perf_v = None
    _, end = quarter_bounds()
    return {
        "level": level, "target": target, "signals": signals, "signals_target": int(t_signals),
        "signals_remaining": max(0, int(t_signals) - signals), "pyramids": lit, "pyramids_target": int(t_pyr),
        "performance": perf_v, "performance_target": t_perf, "days_left": (end - date.today()).days,
        "tiebreak": {k: summary.get(k) for k in ("operatorAvg", "fieldAvg", "operatorCount", "fieldCount",
                                                 "maxSimulationStreak", "communityActivity")},
        "summary": summary,
    }


async def genius_plan() -> dict[str, Any]:
    """Ask the LLM to turn the Genius gaps into concrete, prioritised research goals ("Genius signals")."""
    from ..llm import llm
    from ..llm.prompts import render
    from ..llm.schemas import GeniusPlan
    st = await genius_status()
    cells = await pyramid_cells()
    pyr = "; ".join(f"{c.region}/D{c.delay}/{c.category_name}: {c.alpha_count} (x{c.multiplier})" for c in cells[:80])
    gaps = (f"signals {st['signals']}/{st['signals_target']}, pyramids {st['pyramids']}/{st['pyramids_target']}, "
            f"combined performance {st['performance']} vs {st['performance_target']}")
    prompt = await render("genius", level=st["level"], target=st["target"], gaps=gaps, tiebreak=st["tiebreak"],
                          pyramids=pyr or "not synced", days_left=st["days_left"])
    plan = await llm.generate_json(prompt, GeniusPlan, role="refine", purpose="genius")
    data = {"created": datetime.now(timezone.utc).isoformat(), "summary": plan.summary,
            "signals": [{**s.model_dump(), "done": False} for s in plan.signals]}
    await db.kv_set("genius_plan", data)
    bus.publish("analyse", f"Genius plan: {len(plan.signals)} signals toward {st['target'].title()}.", "success")
    return data


# ---------------------------------------------------------------- Osmosis

def next_osmosis_lock() -> datetime:
    now = datetime.now(EASTERN)
    target_wd = settings.programs.osmosis_lock_weekday
    days = (target_wd - now.weekday()) % 7
    lock = (now + timedelta(days=days)).replace(hour=settings.programs.osmosis_lock_hour_et,
                                                 minute=settings.programs.osmosis_lock_minute, second=0, microsecond=0)
    if lock <= now:
        lock += timedelta(days=7)
    return lock


def iso_week(d: date | None = None) -> str:
    y, w, _ = (d or date.today()).isocalendar()
    return f"{y}-W{w:02d}"


async def osmosis_pool(region: str, delay: int) -> list[Alpha]:
    async with db.session() as s:
        return list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE", Alpha.region == region,
                                                        Alpha.delay == delay, Alpha.type == "REGULAR")
                                    .order_by(Alpha.score.desc().nulls_last())))


def alpha_quality(a: Alpha) -> float:
    """Shrink OS Sharpe toward IS Sharpe (OS history is short); penalise high self-correlation."""
    is_s = a.sharpe or 0.0
    os_s = a.os_sharpe if a.os_sharpe is not None else None
    q = is_s if os_s is None else 0.4 * is_s + 0.6 * os_s
    if a.self_corr is not None:
        q *= max(0.3, 1.0 - max(0.0, a.self_corr - 0.3))
    return max(0.0, q)


async def pnl_returns(alpha_ids: list[str], days: int = 750) -> tuple[list[str], np.ndarray | None]:
    """Daily PnL increments (columns = alphas) over the common recent window."""
    from . import pnl
    rows = await pnl.stored(alpha_ids)
    for aid in alpha_ids:
        if aid not in rows:
            try:
                got = await pnl.fetch(aid)
            except Exception as exc:
                log.warning("pnl for %s unavailable: %s", aid, exc)
                continue
            if got is not None:
                rows[aid] = got
    df = pnl.frame(rows).dropna()
    ids = list(df.columns)
    if len(ids) < 2 or len(df) < 60:
        return ids, None
    return ids, df.tail(days).to_numpy()


def _constrain(weights: dict[str, float], total: int, lo: int, hi: int) -> dict[str, int]:
    """Scale to `total` points, clip to [lo, hi] iteratively, round to integers that sum exactly to total."""
    w = {k: max(0.0, v) for k, v in weights.items()}
    if not w or sum(w.values()) <= 0:
        w = {k: 1.0 for k in weights}
    fixed: dict[str, float] = {}
    for _ in range(50):
        free = {k: v for k, v in w.items() if k not in fixed}
        remaining = total - sum(fixed.values())
        s = sum(free.values()) or 1.0
        alloc = {k: remaining * v / s for k, v in free.items()}
        changed = False
        for k, v in alloc.items():
            if v > hi:
                fixed[k], changed = float(hi), True
            elif v < lo:
                fixed[k], changed = float(lo), True
        if not changed:
            fixed.update(alloc)
            break
    pts = {k: int(math.floor(v)) for k, v in fixed.items()}
    diff = total - sum(pts.values())
    for k in sorted(fixed, key=lambda k: fixed[k] - pts[k], reverse=True)[:abs(diff)]:
        pts[k] += 1 if diff > 0 else -1
    return pts


async def allocate_osmosis(alphas: list[Alpha], method: str = "score") -> dict[str, int]:
    p = settings.programs
    pool = sorted(alphas, key=alpha_quality, reverse=True)[: p.osmosis_max_alphas]
    if not pool:
        return {}
    if method == "equal":
        weights = {a.id: 1.0 for a in pool}
    elif method in ("kelly", "inverse_corr"):
        ids, rets = await pnl_returns([a.id for a in pool])
        if rets is None:
            weights = {a.id: alpha_quality(a) for a in pool}
        else:
            cov = np.cov(rets, rowvar=False) + np.eye(len(ids)) * 1e-9
            if method == "kelly":
                mean = rets.mean(axis=0)
                f = np.linalg.solve(cov, mean) * 0.5     # half-Kelly (Chan, ch. 6), long-only clip below
                weights = {aid: float(max(0.0, x)) for aid, x in zip(ids, f)}
            else:
                corr = np.corrcoef(rets, rowvar=False)
                avg_corr = (corr.sum(axis=0) - 1) / max(1, len(ids) - 1)
                weights = {aid: float(max(0.05, 1 - c)) for aid, c in zip(ids, avg_corr)}
            for a in pool:
                weights.setdefault(a.id, 0.0)
    else:
        weights = {a.id: alpha_quality(a) for a in pool}
    positive = {k: v for k, v in weights.items() if v > 0}
    if len(positive) < p.osmosis_min_alphas:
        ranked = sorted(weights, key=lambda k: weights[k], reverse=True)
        positive = {k: max(weights[k], 1e-6) for k in ranked[: max(p.osmosis_min_alphas, len(positive))]}
    return _constrain(positive, p.osmosis_points_total, p.osmosis_min_points, p.osmosis_max_points)


async def push_osmosis(region: str, delay: int, allocation: dict[str, int], method: str) -> tuple[int, list[str]]:
    """PATCH osmosisPoints for every alpha in the region/delay (others in the pool are cleared)."""
    pool = await osmosis_pool(region, delay)
    errors: list[str] = []
    ok = 0
    week = iso_week()
    for a in pool:
        pts = allocation.get(a.id)
        if pts is None and not a.osmosis_points:
            continue
        try:
            if pts is None:
                await brain.alphas.patch(a.id, clear_osmosis=True)
            else:
                await brain.alphas.patch(a.id, osmosis_points=pts)
            ok += 1
        except Exception as exc:
            errors.append(f"{a.id}: {getattr(exc, 'message', exc)}")
    async with db.write() as s:
        for aid, pts in allocation.items():
            s.add(OsmosisAllocation(week=week, region=region, delay=delay, alpha_id=aid, points=pts, method=method,
                                    status="pushed" if not errors else "failed",
                                    pushed_at=datetime.now(timezone.utc)))
        for a in pool:
            if a.id in allocation:
                row = await s.get(Alpha, a.id)
                if row:
                    row.osmosis_points = allocation[a.id]
    bus.publish("sync", f"Osmosis {region}/D{delay}: {len(allocation)} alphas, "
                        f"{sum(allocation.values()):,} points sent ({len(errors)} errors).",
                "success" if not errors else "warning")
    return ok, errors


@jobs.register("osmosis_push")
async def _job_osmosis(ctx: JobContext, params: dict) -> str:
    ok, errors = await push_osmosis(params["region"], int(params["delay"]),
                                    {k: int(v) for k, v in params["allocation"].items()}, params.get("method", ""))
    if errors:
        raise RuntimeError("; ".join(errors[:5]))
    return f"{ok} alphas updated"


# ---------------------------------------------------------------- out-of-sample

def os_metric(a: Alpha, *keys: str) -> float | None:
    os_ = a.os or {}
    for k in keys:
        v = os_.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


async def oos_rows() -> list[dict[str, Any]]:
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(Alpha.status.in_(["ACTIVE", "DECOMMISSIONED"]))
                                    .order_by(Alpha.date_submitted.desc().nulls_last())))
    out = []
    today = datetime.now(timezone.utc)
    for a in rows:
        os_sharpe = a.os_sharpe if a.os_sharpe is not None else os_metric(a, "sharpe", "osISSharpeRatio")
        ratio = (os_sharpe / a.sharpe) if (os_sharpe is not None and a.sharpe) else None
        age = (today - a.date_submitted).days if a.date_submitted else None
        flag = ""
        if ratio is not None and age and age > 60:
            flag = "decaying" if ratio < 0.3 else ("healthy" if ratio > 0.7 else "watch")
        out.append({"id": a.id, "region": a.region, "delay": a.delay, "is_sharpe": a.sharpe, "os_sharpe": os_sharpe,
                    "ratio": ratio, "os_fitness": os_metric(a, "fitness"), "os_turnover": os_metric(a, "turnover"),
                    "os_returns": os_metric(a, "returns"), "age_days": age, "flag": flag,
                    "osmosis": a.osmosis_points, "status": a.status,
                    "submitted": a.date_submitted.strftime("%Y-%m-%d") if a.date_submitted else ""})
    return out


async def submissions_by_day(days: int = 30) -> list[tuple[str, int]]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    async with db.session() as s:
        rows = list(await s.scalars(select(Submission.requested_at).where(Submission.requested_at >= since)))
    counts: dict[str, int] = {}
    for ts in rows:
        k = ts.astimezone(EASTERN).strftime("%Y-%m-%d")
        counts[k] = counts.get(k, 0) + 1
    return sorted(counts.items())
