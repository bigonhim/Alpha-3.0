"""Submittable alphas and the shortlist worth a consultant's time (ported from alpha-harness vault/yields.py).

Every unsubmitted alpha whose stored checks BRAIN reads as submittable, best Sharpe first. Best Sharpe first is also
most overfit first, so the shortlist ignores it: only alphas that held up in the test years their search never saw,
ranked by stability (test Sharpe and sub-universe margin percentile ranks), each kept unless it moves with a
better-ranked pick (|correlation| ≥ 0.7 in a region both trade). At most five: more is a list to read, not a choice.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from itertools import accumulate
from typing import Any

from sqlalchemy import func, not_, select

from ..brain.platform import MIN_OVERLAP, is_implausible, is_near_miss
from ..db import db
from ..db.models import Alpha
from ..engine import metrics, series
from ..jobs.queue import jobs
from . import store

SHORTLIST = 5
SHORTLIST_MAX_CORRELATION = 0.7
HELD_OUT_KEEP = 0.5            # the test-years Sharpe must keep at least half the train-years Sharpe
SPARK_POINTS = 120
_REQUESTED: set[str] = set()   # PnL downloads already asked for in this session, so a refresh does not repeat them


def held_out_ok(test_sharpe: float | None, train_sharpe: float | None) -> bool:
    """Whether an alpha held up in the held-out test years. False without a test Sharpe: never tested is not held."""
    if test_sharpe is None or test_sharpe <= 0:
        return False
    return train_sharpe is None or test_sharpe >= HELD_OUT_KEEP * train_sharpe


def sub_universe_margin(checks: list[dict[str, Any]] | None) -> float | None:
    """How far LOW_SUB_UNIVERSE_SHARPE cleared its limit, as value over limit."""
    for c in checks or []:
        if str(c.get("name", "")).upper() != "LOW_SUB_UNIVERSE_SHARPE":
            continue
        value, limit = c.get("value"), c.get("limit")
        if isinstance(value, (int, float)) and isinstance(limit, (int, float)) and limit > 0:
            return value / limit
    return None


def stability_order(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Alpha ids, most stable first: the mean percentile rank of test Sharpe and sub-universe margin. Ranks rather
    than a weighted sum, so no measure needs a scale; a missing measure ranks last on that measure.

    Each row: {alpha_id, test_sharpe, checks}."""
    ids = [str(r["alpha_id"]) for r in rows]
    measures = [{str(r["alpha_id"]): r.get("test_sharpe") for r in rows},
                {str(r["alpha_id"]): sub_universe_margin(r.get("checks")) for r in rows}]
    score = dict.fromkeys(ids, 0.0)
    for measure in measures:
        known = sorted((v, a) for a in ids if (v := measure.get(a)) is not None)
        for place, (_, alpha_id) in enumerate(known, start=1):
            score[alpha_id] += place / len(ids)
    return sorted(ids, key=lambda a: -score[a])


def _pair(a: tuple[list[str], list[float]], b: tuple[list[str], list[float]]) -> float | None:
    """Pearson over the days both have (≥ 250), through `metrics.pair_correlation` on the aligned values."""
    pos = {d: i for i, d in enumerate(b[0])}
    shared = [(i, pos[d]) for i, d in enumerate(a[0]) if d in pos]
    if len(shared) < MIN_OVERLAP:
        return None
    x = [a[1][i] for i, _ in shared]
    y = [b[1][j] for _, j in shared]
    return metrics.pair_correlation(x, y)


def independent(order: Sequence[str], profiles: dict[str, dict[str, tuple[list[str], list[float]]]],
                limit: int = SHORTLIST, threshold: float = SHORTLIST_MAX_CORRELATION) -> list[str]:
    """Best first, each kept unless it moves with one already kept (|ρ| ≥ threshold in a region both trade). An
    alpha without a usable series counts as independent. `profiles`: alpha -> {region: (dates, pnl)}."""
    kept: list[str] = []
    for alpha_id in order:
        if len(kept) >= limit:
            break
        own = profiles.get(alpha_id)
        if own:
            clash = False
            for other in kept:
                theirs = profiles.get(other) or {}
                for region in own.keys() & theirs.keys():
                    rho = _pair(own[region], theirs[region])
                    if rho is not None and abs(rho) >= threshold:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                continue
        kept.append(alpha_id)
    return kept


def sparkline(values: Sequence[float]) -> list[float]:
    """Daily PnL as a cumulative curve, thinned to 120 points; the last point is always kept."""
    cumulative = [round(t, 2) for t in accumulate(values)]
    if len(cumulative) <= SPARK_POINTS:
        return cumulative
    step = len(cumulative) / SPARK_POINTS
    thinned = [cumulative[int(i * step)] for i in range(SPARK_POINTS)]
    thinned[-1] = cumulative[-1]
    return thinned


async def submittable(*, region: str | None = None, delay: int | None = None, universe: str | None = None,
                      limit: int = 50, show_implausible: bool = False, schedule: bool = True) -> dict[str, Any]:
    """Every unsubmitted alpha BRAIN's checks read as submittable, the counts that explain an empty list, and the
    shortlist. Held-out alphas with no stored PnL are queued for download (`fetch_series`) when `schedule`."""
    async with db.session() as s:
        stmt = select(Alpha).where(not_(store.submitted_clause()), Alpha.checks.is_not(None))
        for column, value in ((Alpha.region, region), (Alpha.delay, delay), (Alpha.universe, universe)):
            if value is not None:
                stmt = stmt.where(column == value)
        every = list(await s.scalars(stmt.order_by(Alpha.sharpe.desc().nulls_last(), Alpha.id)))
        count_all = int(await s.scalar(select(func.count()).select_from(Alpha)) or 0)
    ready, pending, near = [], 0, 0
    for a in every:
        found = store.verdict_of(a)
        if found == "submittable":
            ready.append(a)
        elif found == "pending":
            pending += 1
        elif found == "refused" and is_near_miss(a.checks):
            near += 1
    flagged = {a.id for a in ready if is_implausible(a.train_sharpe, a.test_sharpe, a.sharpe)}
    hidden = 0 if show_implausible else len(flagged)
    if hidden:
        ready = [a for a in ready if a.id not in flagged]
    validated = [a for a in ready if a.test_sharpe is not None]
    held = [a for a in validated if held_out_ok(a.test_sharpe, a.train_sharpe)]
    order = stability_order([{"alpha_id": a.id, "test_sharpe": a.test_sharpe, "checks": a.checks} for a in held])
    chosen = ready[:limit]
    wanted = list(dict.fromkeys([*order, *(a.id for a in chosen)]))
    loaded = await series.load(wanted)
    by_id = {a.id: a for a in ready}
    profiles = {a: {str(by_id[a].region or ""): (loaded[a]["dates"], loaded[a]["pnl"])}
                for a in order if a in loaded and len(loaded[a]["dates"]) >= MIN_OVERLAP}
    picks = await asyncio.to_thread(independent, order, profiles)
    walked = order.index(picks[-1]) + 1 if picks else 0
    missing = [a for a in order if a not in loaded]
    if schedule:
        await request_series(missing)
    spark = {a: sparkline(v["pnl"]) for a, v in loaded.items()}
    return {
        "shortlist": [by_id[a] for a in picks], "alphas": chosen, "total": len(ready),
        "markets": store.counts_by_market(ready), "pending": pending, "implausible_hidden": hidden,
        "near_misses": near, "held_out_failed": len(validated) - len(held), "unvalidated": len(ready) - len(validated),
        "correlated_pruned": walked - len(picks), "pnl_missing": missing, "spark": spark, "stored": count_all,
    }


async def request_series(alpha_ids: Sequence[str]) -> int | None:
    """Queue one `fetch_series` job for the alphas not yet asked for in this session; returns the job id."""
    todo = [a for a in dict.fromkeys(alpha_ids) if a not in _REQUESTED]
    if not todo:
        return None
    _REQUESTED.update(todo)
    return await jobs.submit("fetch_series", f"Daily PnL for {len(todo)} alphas", {"alpha_ids": todo,
                                                                                   "turnover": True})


async def families(limit: int = 50) -> list[dict[str, Any]]:
    """Unsubmitted region-agnostic parents with the MIN_SUBMITTABLE_CHILDREN rule and the standing child."""
    out = []
    for parent, children in await store.ra_families(limit=limit):
        fam = store.family_row(parent, children)
        out.append({"parent": parent, "children": children, "check": store.children_check(children),
                    "standing": fam["standing"] if fam else None, "regions": fam["regions"] if fam else []})
    return out

