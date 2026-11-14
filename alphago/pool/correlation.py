"""Power Pool and Self-Correlation measured locally, as BRAIN measures them (ported from alpha-harness
tools/power_pool.py, where each rule was measured against BRAIN's own answers).

- Pearson of daily PnL on the days both alphas have a row, over BRAIN's window: from Jan 1 of the 4th calendar year
  back from the later of the candidate's last day and its pool's last day (closing days move where the window ends
  but are never correlated), on at least 250 shared days.
- Scope is the region and only the region: delay and universe are not part of the bucket.
- Power Pool Correlation must be below 0.5; above it the alpha needs a Sharpe of at least 1.1 × that of *every*
  pool alpha it collides with (stricter than BRAIN's wording, and what BRAIN enforces).
- Self-Correlation compares with every submitted alpha of the region except those classed Power Pool alone.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from ..brain.platform import POWER_POOL_CORRELATION_CEILING, POWER_POOL_ESCAPE, is_power_pool
from ..db import db
from ..db.models import Alpha
from ..engine import metrics, series
from . import store

#: Verdicts an alpha can be submitted on.
PASSING = frozenset({"clear", "beats"})
#: Candidates measured against a pool in one matrix: bounds the memory of a sweep of thousands.
CHUNK = 400


@dataclass
class Grid:
    """Days × alphas of exported daily PnL (NaN where an alpha has no row, and on its closing days), from `dates[0]`,
    with each alpha's last counted day (closing days included)."""

    ids: list[str]
    dates: list[str]
    pnl: np.ndarray
    ends: dict[str, date]

    def __post_init__(self) -> None:
        self.at = {a: i for i, a in enumerate(self.ids)}


def build_grid(loaded: dict[str, dict[str, Any]], since: date | None = None) -> Grid:
    """A `Grid` from `series.load` output ({dates, pnl, closing}), keeping only days from `since` on."""
    ids = [a for a, v in loaded.items() if v.get("dates")]
    floor = since.isoformat() if since else ""
    dates = sorted({d for a in ids for d in loaded[a]["dates"] if d >= floor})
    pos = {d: i for i, d in enumerate(dates)}
    pnl = np.full((len(dates), len(ids)), np.nan)
    ends: dict[str, date] = {}
    for j, a in enumerate(ids):
        v = loaded[a]
        closing = v.get("closing") or [False] * len(v["dates"])
        for d, p, c in zip(v["dates"], v["pnl"], closing):
            if not c and d >= floor:
                pnl[pos[d], j] = p
        ends[a] = date.fromisoformat(v["dates"][-1])
    return Grid(ids, dates, pnl, ends)


async def load_grid(alpha_ids: Sequence[str]) -> Grid:
    """The stored series of these alphas as a `Grid`, cut to the earliest window any of them can need."""
    loaded = await series.load(list(dict.fromkeys(alpha_ids)))
    if not loaded:
        return Grid([], [], np.empty((0, 0)), {})
    last = min(date.fromisoformat(v["dates"][-1]) for v in loaded.values())
    return build_grid(loaded, metrics.window_start(last))


def _rho(grid: Grid, left: list[str], right: list[str], pool_end: date | None) -> np.ndarray:
    """`left` against `right`, each candidate over its own window: ending with its series or the pool's, whichever
    is later. One end for the whole batch would make a figure depend on what else was asked with it."""
    rho = np.full((len(left), len(right)), np.nan)
    if not (left and right):
        return rho
    by_start: dict[date, list[int]] = {}
    for k, a in enumerate(left):
        end = grid.ends[a] if pool_end is None else max(grid.ends[a], pool_end)
        by_start.setdefault(metrics.window_start(end), []).append(k)
    right_cols = [grid.at[p] for p in right]
    for start, picked in by_start.items():
        since = bisect_left(grid.dates, start.isoformat())
        for i in range(0, len(picked), CHUNK):
            ks = picked[i:i + CHUNK]
            cols = [grid.at[left[k]] for k in ks] + right_cols
            corr = metrics.correlation_matrix(grid.pnl[since:, cols])
            rho[ks] = corr[:len(ks), len(ks):]
    return rho


def _sharpe(row: dict[str, Any] | None) -> float | None:
    value = (row or {}).get("sharpe")
    return float(value) if value is not None else None


def _row(alpha_id: str, meta: dict[str, dict[str, Any]], sharpe: float | None) -> dict[str, Any]:
    row = meta.get(alpha_id) or {}
    return {"alpha_id": alpha_id, "sharpe": sharpe, "region": row.get("region"), "universe": row.get("universe"),
            "delay": row.get("delay")}


def verdict(alpha_id: str, meta: dict[str, dict[str, Any]], measured: list[tuple[str, float]], against: int,
            ceiling: float = POWER_POOL_CORRELATION_CEILING) -> dict[str, Any]:
    """One candidate's answer: its highest correlation, and whether that refuses it."""
    sharpe = _sharpe(meta.get(alpha_id))
    if not measured:
        # With an empty pool (nothing submitted in the region yet) the ceiling cannot be crossed, so the first alpha
        # of a region is eligible on this rule; only a pool that shares too few days is genuinely unmeasured.
        return _row(alpha_id, meta, sharpe) | {"correlation": None, "against": against, "closest": None,
                                               "closest_sharpe": None, "needed": None,
                                               "verdict": "unmeasured" if against else "clear"}
    closest, highest = max(measured, key=lambda pair: pair[1])
    # Every collision sets a bar; the alpha has to clear the tallest, not just the closest one's.
    bars = [found * POWER_POOL_ESCAPE for other, value in measured
            if value >= ceiling and (found := _sharpe(meta.get(other))) is not None]
    # Rounded: Sharpes are two-decimal figures, and 1.10 × 1.50 = 1.6500000000000001 must admit a 1.65.
    needed = round(max(bars), 4) if bars else None
    if highest < ceiling:
        found_verdict = "clear"
    elif needed is not None and sharpe is not None and sharpe >= needed:
        found_verdict = "beats"
    else:
        found_verdict = "blocked"
    return _row(alpha_id, meta, sharpe) | {
        "correlation": round(highest, 4), "against": against, "closest": closest,
        "closest_sharpe": _sharpe(meta.get(closest)), "needed": needed if highest >= ceiling else None,
        "verdict": found_verdict}


def _judge(alpha_id: str, meta: dict[str, dict[str, Any]], against: list[str], values: dict[str, float],
           stored: set[str] | dict[str, int], ceiling: float) -> dict[str, Any]:
    """The verdict from the pairs measured. A pool member without a stored series still counts in `against`, and
    while one has none the answer is unmeasured unless a collision already refuses it: it could hide another."""
    usable = [(p, v) for p in against if (v := values.get(p)) is not None and not math.isnan(v)]
    found = verdict(alpha_id, meta, usable, len(against), ceiling)
    if found["verdict"] != "blocked" and any(p not in stored for p in against):
        found["verdict"] = "unmeasured"
    return found


def correlate(grid: Grid, meta: dict[str, dict[str, Any]], candidates: Sequence[str], pool: Sequence[str],
              ceiling: float = POWER_POOL_CORRELATION_CEILING) -> list[dict[str, Any]]:
    """Each candidate against every member of one region's pool, with the verdict BRAIN would reach. A candidate in
    the pool is never measured against itself."""
    left = [a for a in dict.fromkeys(candidates) if a in grid.at]
    right = [p for p in dict.fromkeys(pool) if p in grid.at]
    pool_end = max((grid.ends[p] for p in right), default=None)
    rho = _rho(grid, left, right, pool_end)
    row_of = {a: i for i, a in enumerate(left)}
    out = []
    for alpha_id in candidates:
        values = ({p: float(rho[row_of[alpha_id], j]) for j, p in enumerate(right)} if alpha_id in row_of else {})
        out.append(_judge(alpha_id, meta, [p for p in pool if p != alpha_id], values, grid.at, ceiling))
    return out


def submission_order(grid: Grid, meta: dict[str, dict[str, Any]], candidates: Sequence[str],
                     pools: dict[str, list[str]], ceiling: float = POWER_POOL_CORRELATION_CEILING,
                     ) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    """Submit greedily, best After-Cost Sharpe first, without submitting anything: each pick joins its region's pool
    and that region's other candidates are judged again against it.

    Returns the picks in order (each with its row as judged at its turn), the candidates an earlier pick blocked
    (with that pick), and those that could not be measured against a pick (no stored series)."""

    def rank(alpha_id: str) -> float:
        value = (meta.get(alpha_id) or {}).get("after_cost_5bps")
        return float(value) if value is not None else float("-inf")

    remaining = sorted(dict.fromkeys(candidates), key=lambda a: (-rank(a), a))
    region_of = {a: (meta.get(a) or {}).get("region") for a in remaining}
    pool = {region: list(ids) for region, ids in pools.items()}
    pool_end: dict[str, date] = {}
    values: dict[str, dict[str, float]] = {a: {} for a in remaining}

    def measure(region: str, rivals: list[str], members: list[str]) -> None:
        left = [a for a in rivals if a in grid.at]
        right = [p for p in members if p in grid.at]
        rho = _rho(grid, left, right, pool_end.get(region))
        for i, a in enumerate(left):
            values[a].update((p, float(rho[i, j])) for j, p in enumerate(right))

    def judge(alpha_id: str, region: str) -> dict[str, Any]:
        against = [p for p in pool.get(region, []) if p != alpha_id]
        return _judge(alpha_id, meta, against, values[alpha_id], grid.at, ceiling)

    for region in {r for r in region_of.values() if r}:
        members = [p for p in pool.get(region, []) if p in grid.at]
        if members:
            pool_end[region] = max(grid.ends[p] for p in members)
            measure(region, [a for a in remaining if region_of[a] == region], members)
    judged = {a: judge(a, region) for a in remaining if (region := region_of[a])}
    remaining = [a for a in remaining if judged.get(a, {}).get("verdict") in PASSING]
    picks: list[dict[str, Any]] = []
    blocked: dict[str, str] = {}
    unmeasured: list[str] = []
    while remaining:
        pick = remaining.pop(0)
        region = region_of[pick]
        picks.append(judged.get(pick) or _row(pick, meta, _sharpe(meta.get(pick))) | {"verdict": "clear"})
        if region is None:
            continue
        pool.setdefault(region, []).append(pick)
        rivals = [a for a in remaining if region_of[a] == region]
        if pick in grid.at:
            if region not in pool_end or grid.ends[pick] > pool_end[region]:
                # a later end moves every window: the region is measured afresh
                pool_end[region] = grid.ends[pick]
                measure(region, rivals, pool[region])
            else:
                measure(region, rivals, [pick])
        for alpha_id in rivals:
            row = judged[alpha_id] = judge(alpha_id, region)
            if row["verdict"] == "blocked":
                blocked[alpha_id] = pick
            elif row["verdict"] not in PASSING:
                unmeasured.append(alpha_id)
        out = set(blocked) | set(unmeasured)
        remaining = [a for a in remaining if a not in out]
    return picks, blocked, unmeasured


# ---------------------------------------------------------------- reading the pool

def meta_of(a: Alpha) -> dict[str, Any]:
    return {"sharpe": a.sharpe, "region": a.region, "universe": a.universe, "delay": a.delay,
            "after_cost_5bps": a.after_cost_5bps}


def self_pools(members: Sequence[Alpha]) -> dict[str, list[str]]:
    """By region, the submitted alphas BRAIN's SELF_CORRELATION compares with: all but those classed Power Pool
    alone, whether or not the alpha compared is a Power Pool one."""
    pools: dict[str, list[str]] = {}
    for a in members:
        if a.region and not store.power_pool_only(a):
            pools.setdefault(str(a.region), []).append(a.id)
    return pools


def power_pools(members: Sequence[Alpha]) -> dict[str, list[str]]:
    """By region, the submitted alphas BRAIN judges as Power Pool (from their POWER_POOL_CORRELATION check)."""
    pools: dict[str, list[str]] = {}
    for a in members:
        if a.region and is_power_pool(a.checks):
            pools.setdefault(str(a.region), []).append(a.id)
    return pools


async def _by_region(alpha_ids: list[str], pools: dict[str, list[str]],
                     members: list[Alpha]) -> dict[str, dict[str, Any]]:
    alphas = await store.by_ids(alpha_ids)
    alphas.update({m.id: m for m in members})
    wanted: dict[str, list[str]] = {}
    for aid in dict.fromkeys(alpha_ids):
        a = alphas.get(aid)
        if a is not None and a.region:
            wanted.setdefault(str(a.region), []).append(aid)
    needed = list(alpha_ids) + [p for r in wanted for p in pools.get(r, [])]
    grid = await load_grid(needed)
    meta = {a.id: meta_of(a) for a in alphas.values()}
    out: dict[str, dict[str, Any]] = {}
    for region, ids in wanted.items():
        for row in correlate(grid, meta, ids, pools.get(region, [])):
            out[row["alpha_id"]] = row
    return out


async def local_self_correlation(alpha_ids: Sequence[str]) -> dict[str, tuple[float | None, int]]:
    """Each alpha's Self-Correlation as BRAIN's alpha page shows it today, worked out from stored PnL: its highest
    correlation with the submitted alphas of its region (Power-Pool-only ones aside), and how many it was compared
    against. (None, n) when nothing shared 250 days in the window."""
    ids = list(dict.fromkeys(alpha_ids))
    if not ids:
        return {}
    members = await store.submitted_members()
    found = await _by_region(ids, self_pools(members), members)
    return {a: (row["correlation"], row["against"]) for a, row in found.items()}


async def power_pool_correlations(alpha_ids: Sequence[str]) -> dict[str, tuple[float | None, int]]:
    """Each alpha's Power Pool Correlation against the submitted Power Pool of its region, (value, against)."""
    ids = list(dict.fromkeys(alpha_ids))
    if not ids:
        return {}
    members = await store.submitted_members()
    found = await _by_region(ids, power_pools(members), members)
    return {a: (row["correlation"], row["against"]) for a, row in found.items()}


async def power_pool_analysis(alpha_ids: Sequence[str], *, save: bool = True) -> dict[str, Any]:
    """These alphas against the submitted Power Pool, region by region (alpha-harness `_power_pool`).

    Only alphas BRAIN judges as Power Pool are measured; nothing is downloaded. An alpha without a series is listed
    in `without_pnl`; a region with nothing submitted is clear on this rule. With `save`, each measured verdict is
    written to Alpha.pp_corr / pp_verdict."""
    members = await store.submitted_members()
    pools = power_pools(members)
    member_ids = {m.id for m in members}
    stored = await store.by_ids(list(alpha_ids))
    candidates = {a: r for a, r in stored.items() if is_power_pool(r.checks)}
    everyone = {**{m.id: m for m in members if m.id in {p for ids in pools.values() for p in ids}}, **candidates}
    meta = {a: meta_of(r) for a, r in everyone.items()}
    grid = await load_grid([*candidates, *(p for ids in pools.values() for p in ids)])
    rows: list[dict[str, Any]] = []
    without: list[str] = []
    wanted: dict[str, list[str]] = {}
    unopposed: dict[str, list[str]] = {}
    for aid in dict.fromkeys(alpha_ids):
        row = candidates.get(aid)
        if row is None or not row.region:
            continue
        region = str(row.region)
        if region not in pools:
            unopposed.setdefault(region, []).append(aid)
            continue
        measurable = wanted.setdefault(region, [])
        (measurable if aid in grid.at else without).append(aid)
    for ids in unopposed.values():
        rows.extend(correlate(grid, meta, ids, []))
    scopes = [{"region": r, "pool": [], "pool_without_pnl": []} for r in unopposed]
    for region in sorted(wanted, key=lambda r: (-len(wanted[r]), r)):
        members_r = pools[region]
        rows.extend(correlate(grid, meta, wanted[region], members_r))
        scopes.append({"region": region, "pool": members_r,
                       "pool_without_pnl": [m for m in members_r if m not in grid.at]})
    if save and rows:
        async with db.write() as s:
            for r in rows:
                a = await s.get(Alpha, r["alpha_id"])
                if a is not None:
                    a.pp_corr, a.pp_verdict = r["correlation"], r["verdict"]
    return {"ceiling": POWER_POOL_CORRELATION_CEILING, "sharpe_edge": POWER_POOL_ESCAPE,
            "pool_size": sum(len(v) for v in pools.values()), "power_pool_alphas": len(candidates),
            "other_alphas": len(stored) - len(candidates), "without_pnl": without, "rows": rows, "scopes": scopes,
            "submitted": [r["alpha_id"] for r in rows if r["alpha_id"] in member_ids]}


async def order_for(alpha_ids: Sequence[str]) -> dict[str, Any]:
    """How many of these alphas could be submitted one after another (alpha-harness /power-pool-order). Nothing is
    submitted and nothing is downloaded."""
    members = await store.submitted_members()
    pools = power_pools(members)
    stored = await store.by_ids(list(alpha_ids))
    candidates = {a: r for a, r in stored.items() if is_power_pool(r.checks) and not store.is_submitted(r)}
    pool_meta = {m.id: meta_of(m) for m in members if m.id in {p for ids in pools.values() for p in ids}}
    meta = {**pool_meta, **{a: meta_of(r) for a, r in candidates.items()}}
    grid = await load_grid([*candidates, *pool_meta])
    picks, blocked, unmeasured = submission_order(grid, meta, list(candidates), pools)
    blocks: dict[str, int] = {}
    for by in blocked.values():
        blocks[by] = blocks.get(by, 0) + 1
    return {"candidates": len(candidates),
            "picks": [{**row, "after_cost_sharpe": meta[row["alpha_id"]].get("after_cost_5bps"),
                       "blocks": blocks.get(row["alpha_id"], 0)} for row in picks],
            "blocked": [{"alpha_id": a, "by": by} for a, by in blocked.items()], "unmeasured": unmeasured}
