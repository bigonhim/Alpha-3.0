"""Aggregate numbers for the Dashboard and Analytics pages."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import case, func, select

from ..brain import brain
from ..brain.quota import EASTERN
from ..db import db
from ..db.models import (Alpha, Candidate, DataField, Dataset, DatasetInsight, JournalEntry, LLMCall,
                         OptimizationAttempt, Submission)
from .simulator import GENERATED_ORIGINS

GOOD = ["promising", "strong", "ready", "submitted"]


def _today_utc() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def brain_day_start() -> datetime:
    """Start of the current BRAIN day (midnight New York), when the simulation quota resets."""
    return datetime.now(EASTERN).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


async def pipeline_today() -> dict[str, int]:
    """Where today's candidates are: Generated → Simulating/Optimizing → verdicts, plus submissions and failures."""
    start = brain_day_start()
    async with db.session() as s:
        rows = (await s.execute(select(Candidate.status, Candidate.origin, Alpha.verdict)
                                .outerjoin(Alpha, Alpha.id == Candidate.alpha_id)
                                .where(Candidate.created_at >= start))).all()
        submitted = await s.scalar(select(func.count(Submission.id)).where(
            Submission.requested_at >= start, Submission.status.in_(["accepted", "active"])))
    out = {"Generated": 0, "Simulating": 0, "Optimizing": 0, "Promising": 0, "Strong Candidate": 0,
           "Submission Ready": 0, "Submitted": submitted or 0, "Rejected": 0, "Failed": 0}
    for status, origin, verdict in rows:
        if status in ("new", "approved"):
            out["Generated"] += 1
        elif status in ("queued", "simulating"):
            out["Optimizing" if origin == "optimize" else "Simulating"] += 1
        elif status in ("error", "invalid"):
            out["Failed"] += 1
        elif status == "done":
            key = {"promising": "Promising", "strong": "Strong Candidate", "ready": "Submission Ready",
                   "rejected": "Rejected", "weak": "Rejected"}.get(verdict or "")
            if key:
                out[key] += 1
    return out


async def dashboard_numbers() -> dict[str, Any]:
    start = brain_day_start()
    async with db.session() as s:
        count = lambda stmt: s.scalar(stmt)  # noqa: E731
        n = {
            "submittable": await count(select(func.count(Alpha.id)).where(Alpha.verdict == "ready",
                                                                          Alpha.status == "UNSUBMITTED")),
            "strong": await count(select(func.count(Alpha.id)).where(Alpha.verdict == "strong",
                                                                     Alpha.status != "ACTIVE")),
            "submitted": await count(select(func.count(Alpha.id)).where(Alpha.status == "ACTIVE")),
            "datasets": await count(select(func.count(func.distinct(Dataset.id)))),
            "dataset_scopes": await count(select(func.count(Dataset.pk))),
            "insights": await count(select(func.count(DatasetInsight.pk))),
            "fields": await count(select(func.count(DataField.pk))),
            "generated_today": await count(select(func.count(Candidate.id)).where(
                Candidate.created_at >= start, Candidate.origin.in_(GENERATED_ORIGINS))),
            "avg_score": await count(select(func.avg(Alpha.score)).where(
                Alpha.origin == "app", Alpha.date_created >= datetime.now(timezone.utc) - timedelta(days=7))),
            "simulated_total": await count(select(func.count(Candidate.id)).where(Candidate.status == "done")),
            "simulated_today": await count(select(func.count(Candidate.id)).where(
                Candidate.status == "done", Candidate.finished_at >= start)),
            "optimizations": await count(select(func.count(OptimizationAttempt.id))),
            "app_alphas": await count(select(func.count(Alpha.id)).where(Alpha.origin == "app")),
            "app_good": await count(select(func.count(Alpha.id)).where(Alpha.origin == "app",
                                                                       Alpha.verdict.in_(GOOD))),
            "llm_today": await count(select(func.count(LLMCall.id)).where(LLMCall.ts >= start)),
        }
    n = {k: (v or 0) for k, v in n.items()}
    n["avg_score"] = round(float(n["avg_score"]), 1) if n["avg_score"] else None
    q = brain.client.quota
    n["sims_left"] = q.effective_remaining
    n["sims_limit"] = q.effective_limit
    n["sent_today"] = (q.limit - q.remaining) if (q.limit is not None and q.remaining is not None) else q.sent_today
    n["success_rate"] = (n["app_good"] / n["app_alphas"]) if n["app_alphas"] else 0.0
    return n


async def daily_series(days: int = 30) -> dict[str, list]:
    since = _today_utc() - timedelta(days=days - 1)
    async with db.session() as s:
        cands = list((await s.execute(select(Candidate.finished_at, Candidate.status).where(
            Candidate.finished_at >= since))).all())
        alphas = list((await s.execute(select(Alpha.date_created, Alpha.verdict).where(
            Alpha.origin == "app", Alpha.date_created >= since))).all())
        llm = list(await s.scalars(select(LLMCall.ts).where(LLMCall.ts >= since)))
    labels = [(since + timedelta(days=i)).strftime("%m-%d") for i in range(days)]
    idx = {lab: i for i, lab in enumerate(labels)}
    sims, good, total, calls = [0] * days, [0] * days, [0] * days, [0] * days
    for ts, status in cands:
        if ts and status == "done":
            k = idx.get(ts.astimezone(timezone.utc).strftime("%m-%d"))
            if k is not None:
                sims[k] += 1
    for ts, verdict in alphas:
        if ts:
            k = idx.get(ts.astimezone(timezone.utc).strftime("%m-%d"))
            if k is not None:
                total[k] += 1
                good[k] += verdict in GOOD
    for ts in llm:
        k = idx.get(ts.astimezone(timezone.utc).strftime("%m-%d"))
        if k is not None:
            calls[k] += 1
    rate = [round(g / t * 100, 1) if t else None for g, t in zip(good, total)]
    return {"labels": labels, "simulations": sims, "success_rate": rate, "llm_calls": calls}


async def check_failure_counts(limit: int = 2000) -> dict[str, int]:
    async with db.session() as s:
        rows = list(await s.scalars(select(JournalEntry.failing_checks).order_by(JournalEntry.id.desc()).limit(limit)))
    counts: dict[str, int] = {}
    for fails in rows:
        for name in fails or []:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


async def yield_by(column: str, limit: int = 20) -> list[dict[str, Any]]:
    col = {"origin": Candidate.origin, "dataset": Candidate.dataset_id, "region": Candidate.region}[column]
    async with db.session() as s:
        rows = list((await s.execute(
            select(col, func.count(Candidate.id), func.sum(case((Alpha.verdict.in_(GOOD), 1), else_=0)),
                   func.avg(Alpha.sharpe))
            .join(Alpha, Alpha.id == Candidate.alpha_id).where(Candidate.status == "done")
            .group_by(col).order_by(func.count(Candidate.id).desc()).limit(limit))).all())
    return [{"key": k or "—", "n": n, "good": g or 0, "rate": (g or 0) / n if n else 0, "avg_sharpe": avg}
            for k, n, g, avg in rows]


async def sharpe_histogram(bins: int = 30) -> tuple[list[str], list[int]]:
    async with db.session() as s:
        vals = [v for v in await s.scalars(select(Alpha.sharpe).where(Alpha.origin == "app",
                                                                      Alpha.sharpe.is_not(None)))]
    if not vals:
        return [], []
    lo, hi = max(min(vals), -3.0), min(max(vals), 4.0)
    width = (hi - lo) / bins or 1
    counts = [0] * bins
    for v in vals:
        i = min(bins - 1, max(0, int((min(max(v, lo), hi) - lo) / width)))
        counts[i] += 1
    return [f"{lo + i * width:.2f}" for i in range(bins)], counts
