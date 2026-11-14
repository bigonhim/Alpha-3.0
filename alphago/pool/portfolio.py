"""Portfolio: alphas combined at equal weight, as BRAIN combines its own pool (alpha-harness tools/portfolio.py).

The arithmetic is `engine.metrics.portfolio` (equal weight over the alphas already trading each day, BRAIN's 250-day
annualisation on a $20M book, correlations over its four-year window on ≥ 250 shared days). This module picks the
members, filters them (Selection Expression and facet chips) and lays out what the page shows.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..brain.platform import check_limits
from ..engine import metrics, series
from . import store
from .selection import compile_selection

REGION_AGNOSTIC_CLASS = "Region Agnostic Alpha"

#: A Selection Expression's metrics: the name it reads, how to read it off an alpha. BRAIN's own figures first.
METRICS: dict[str, Any] = {
    "sharpe": lambda a: a.sharpe, "fitness": lambda a: a.fitness, "returns": lambda a: a.returns,
    "turnover": lambda a: a.turnover, "drawdown": lambda a: a.drawdown, "margin": lambda a: a.margin,
    "pnl": lambda a: a.is_pnl, "long_count": lambda a: a.long_count, "short_count": lambda a: a.short_count,
    "prod_correlation": lambda a: a.prod_corr, "self_correlation": lambda a: a.self_corr,
    "train_sharpe": lambda a: a.train_sharpe, "test_sharpe": lambda a: a.test_sharpe,
    "test_fitness": lambda a: a.test_fitness, "after_cost_sharpe": lambda a: a.after_cost_5bps,
    "k_ratio": lambda a: a.k_ratio, "power_pool_correlation": lambda a: a.pp_corr,
    "operator_count": lambda a: a.op_count, "decay": lambda a: a.decay, "truncation": lambda a: a.truncation,
}
EXAMPLE = "sharpe >= 1.58;\nreturns / turnover >= 1;\nreturns / drawdown >= 1"
FACETS = ("classifications", "investability", "region", "delay", "pyramids")


def metrics_of(a: Any) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, read in METRICS.items():
        value = read(a)
        out[name] = None if value is None else float(value)
    return out


def classes_of(a: Any) -> list[str]:
    classes = store.classifications_of(a)
    # BRAIN gives a region-agnostic child no class of its own; named here so it can be filtered on like any other
    return [*classes, REGION_AGNOSTIC_CLASS] if a.sim_type == "RA_CHILD" else classes


def facet_values(a: Any) -> dict[str, list[str]]:
    return {"classifications": classes_of(a), "investability": [store.INVESTABILITY_LABEL[store.investability(a)]],
            "region": [str(a.region)] if a.region else [], "delay": [f"D{a.delay}"] if a.delay is not None else [],
            "pyramids": store.pyramids_of(a)}


def facet_counts(alphas: Sequence[Any]) -> dict[str, list[tuple[str, int]]]:
    out: dict[str, Counter] = {f: Counter() for f in FACETS}
    for a in alphas:
        for facet, values in facet_values(a).items():
            out[facet].update(set(values))
    return {f: sorted(c.items(), key=lambda kv: (-kv[1], kv[0])) for f, c in out.items()}


def select_members(alphas: Sequence[Any], expression: str, chosen: dict[str, set[str]]) -> list[Any]:
    """The alphas meeting every condition of `expression` and, for each facet with chips on, at least one of them
    (OR within a facet, AND across facets). Raises SelectionError for a bad expression."""
    test = compile_selection(expression or "", METRICS)
    out = []
    for a in alphas:
        values = facet_values(a)
        if any(on and not (on & set(values[f])) for f, on in chosen.items()):
            continue
        if test.test(metrics_of(a)):
            out.append(a)
    return out


def pyramid_grid(alphas: Sequence[Any]) -> tuple[list[str], list[str], dict[tuple[str, str], tuple[str, int]]]:
    """Pyramids as BRAIN lays them out: (categories, markets "REGION Dn", {(category, market): (pyramid, count)}),
    most-held first so the busiest rows and columns sit top left."""
    held: Counter = Counter(p for a in alphas for p in set(store.pyramids_of(a)))
    cells: dict[tuple[str, str], tuple[str, int]] = {}
    markets: Counter = Counter()
    categories: Counter = Counter()
    for pyramid, count in held.items():
        parts = pyramid.split("/")
        region, delay, category = (parts + ["", "", ""])[:3]
        market = f"{region} {delay}"
        cells[(category, market)] = (pyramid, count)
        markets[market] += count
        categories[category] += count

    def by_count(c: Counter) -> list[str]:
        return [k for k, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]
    return by_count(categories), by_count(markets), cells


def split_of(alphas: Sequence[Any]) -> str | None:
    """The test start most members share; a mixed pool takes the majority rather than failing."""
    starts = Counter(a.test_start for a in alphas if a.test_start)
    return starts.most_common(1)[0][0] if starts else None


async def compute(alpha_ids: Sequence[str], bps: float = 5.0) -> dict[str, Any]:
    """These alphas combined at equal weight, with every member's own figures. Ids without a stored PnL and turnover
    are listed in `missing` (a Sync downloads them)."""
    alphas = await store.by_ids(list(alpha_ids))
    ids = [a for a in dict.fromkeys(alpha_ids) if a in alphas]
    loaded = await series.load(ids)
    have = [a for a in ids if a in loaded and loaded[a]["has_turnover"]]
    # most recently submitted first, so the correlation grid reads newest to oldest
    have.sort(key=lambda a: str(alphas[a].date_submitted or ""), reverse=True)
    missing = [a for a in ids if a not in have]
    ser = {a: (loaded[a]["dates"], loaded[a]["pnl"], loaded[a]["turnover"]) for a in have}
    closing = {a: loaded[a]["closing"] for a in have}
    split = split_of([alphas[a] for a in have])
    result = await asyncio.to_thread(metrics.portfolio, ser, closing=closing, split=split, bps=bps)
    members = []
    for a in have:
        al = alphas[a]
        fig = metrics.figures(*ser[a], split=al.test_start, bps=bps)
        net = fig["after_cost"]
        members.append({
            "alpha_id": a, "region": al.region, "delay": al.delay, "universe": al.universe,
            "sharpe": al.sharpe, "fitness": al.fitness, "turnover": al.turnover, "returns": al.returns,
            "train_sharpe": al.train_sharpe, "test_sharpe": al.test_sharpe, "k_ratio": fig["k_ratio"],
            "net_sharpe": net.get("sharpe"), "net_fitness": net.get("fitness"), "net_returns": net.get("returns"),
            "net_train_sharpe": net.get("train_sharpe"), "net_test_sharpe": net.get("test_sharpe"),
            "max_pairwise": result.get("max_pairwise", {}).get(a),
            "portfolio_corr": result.get("portfolio_corr", {}).get(a),
            "incremental_sharpe": result.get("incremental_sharpe", {}).get(a),
            "limits": check_limits(al.checks),
        })
    return {**result, "missing": missing, "members": members, "bps": bps, "asked": len(ids)}
