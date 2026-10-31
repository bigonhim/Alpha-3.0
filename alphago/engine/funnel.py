"""Pipeline report: where your alphas stand on the way to Ready, and what is holding them back.

Everything here reads the local database and costs no quota. `shareable()` keeps ids out and counts in, so the report can
be sent to someone tuning the repair engine without giving away a single expression.
"""

from __future__ import annotations

import asyncio
import statistics
from collections import Counter
from typing import Any

from sqlalchemy import func, select

from ..db import db
from ..db.models import Alpha, OptimizationAttempt
from .evaluator import CORRELATION_CHECKS, blocking_failures, pending_checks
from .playbook import check_family


def _near_misses(alphas: list[Alpha]) -> tuple[dict[str, list[float]], Counter]:
    """Gap per failing check and root causes across near misses. CPU work: the page runs it off the UI thread."""
    from .refine import diagnose
    gaps: dict[str, list[float]] = {}
    causes: Counter = Counter()
    for a in alphas:
        if a.verdict not in ("promising", "weak"):
            continue
        d = diagnose(a)
        causes.update(d.profile.causes.values())
        for name in d.failing:
            gap = d.gaps.get(name)
            if gap is not None:
                gaps.setdefault(name, []).append(gap)
    return gaps, causes


async def pipeline(region: str | None = None, delay: int | None = None) -> dict[str, Any]:
    """Counts at each stage, what stops strong alphas, which checks fail most and by how much, and which moves worked."""
    async with db.session() as s:
        stmt = select(Alpha).where(Alpha.type == "REGULAR", (Alpha.status.is_(None)) | (Alpha.status == "UNSUBMITTED"))
        if region:
            stmt = stmt.where(Alpha.region == region)
        if delay is not None:
            stmt = stmt.where(Alpha.delay == delay)
        alphas = list(await s.scalars(stmt))
        moves = (await s.execute(select(OptimizationAttempt.fix, OptimizationAttempt.target_check,
                                        OptimizationAttempt.outcome, func.count())
                                 .where(OptimizationAttempt.fix.is_not(None), OptimizationAttempt.outcome.is_not(None))
                                 .group_by(OptimizationAttempt.fix, OptimizationAttempt.target_check,
                                           OptimizationAttempt.outcome))).all()
    verdicts = Counter(a.verdict or "unknown" for a in alphas)
    strong = [a for a in alphas if a.verdict == "strong"]
    never = sum(a.checked_at is None for a in strong)
    stale = sum(a.check_state == "stale" for a in strong)
    unanswered = [a for a in strong if a.check_state == "pending"]
    corr_only, other_fail = 0, 0
    for a in alphas:
        if a.check_state == "fail":
            failing = set(blocking_failures(a.checks))
            if failing and failing <= CORRELATION_CHECKS:
                corr_only += 1
            elif failing:
                other_fail += 1
    gaps, causes = await asyncio.to_thread(_near_misses, alphas)
    failing_rows = sorted(({"check": k, "alphas": len(v), "median_gap": round(statistics.median(v), 3),
                            "family": check_family(k)} for k, v in gaps.items()), key=lambda r: -r["alphas"])
    tally: dict[tuple[str, str], list[int]] = {}
    for fix, target, outcome, n in moves:
        row = tally.setdefault((fix, target or ""), [0, 0])
        row[0 if outcome == "success" else 1] += n if outcome in ("success", "failure") else 0
    move_rows = sorted(({"fix": f, "target": t, "worked": w, "tried": w + x, "rate": round(w / (w + x), 2)}
                        for (f, t), (w, x) in tally.items() if w + x), key=lambda r: (-r["tried"], r["fix"]))
    out = {"total": len(alphas), "verdicts": dict(verdicts), "strong": len(strong), "never_checked": never,
           "stale": stale, "unanswered": len(unanswered), "correlation_failed": corr_only,
           "other_check_failed": other_fail, "ready": verdicts.get("ready", 0),
           "exhausted": sum(a.repair_state == "exhausted" for a in alphas),
           "waiting_on_brain": [c for a in unanswered for c in pending_checks(a.checks)][:20],
           "failing": failing_rows, "causes": dict(causes.most_common()), "moves": move_rows[:30]}
    out["blocker"] = blocker(out)
    return out


def blocker(p: dict[str, Any]) -> str:
    """The single most useful thing to do next, in a sentence."""
    if not p["total"]:
        return "No unsubmitted alphas yet: run a research task first."
    if p["ready"]:
        return f"{p['ready']} alpha(s) passed BRAIN's check and are ready for your review on the Alphas page."
    if p["never_checked"]:
        return f"{p['never_checked']} strong alpha(s) have never had BRAIN's check: press 'Check all strong' on Alphas."
    if p["unanswered"]:
        return f"BRAIN has not finished checking {p['unanswered']} strong alpha(s); they are re-asked automatically."
    if p["correlation_failed"]:
        return (f"{p['correlation_failed']} alpha(s) pass every metric but fail correlation: a 'Repair near misses' "
                "task runs the decorrelation route on them.")
    if p["other_check_failed"]:
        return f"{p['other_check_failed']} alpha(s) failed BRAIN's check on a metric: a repair task picks them up."
    near = p["verdicts"].get("promising", 0) + p["verdicts"].get("weak", 0)
    if near:
        top = p["failing"][0]["check"] if p["failing"] else "their failing checks"
        return f"{near} near miss(es), most often failing {top}: a 'Repair near misses' task works on them."
    return "Nothing is close to passing: new ideas or datasets will do more than repairs."


def shareable(p: dict[str, Any]) -> dict[str, Any]:
    """The report with counts only: no alpha ids, expressions or data-field names."""
    return {k: v for k, v in p.items() if k != "blocker"}
