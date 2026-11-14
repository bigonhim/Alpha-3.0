"""The local alpha pool as alpha-harness' Stored Alphas table reads it: filtered, sorted and paged in SQL.

Every figure shown is BRAIN's own as stored on `Alpha`; the few worked out locally (after-cost, K-ratio,
correlations) come from `engine.metrics` over the series `engine.series` stores. Region-agnostic families are
judged as BRAIN judges them: a parent submits only when two of its children pass every check.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import and_, case, exists, func, not_, or_, select

from ..brain.platform import (
    POWER_POOL_CLASS,
    RA_MIN_PASSING_CHILDREN,
    REGULAR_CLASS,
    brain_verdict,
    check_limits,
    failed_checks,
    investability_of,
)
from ..brain.quota import EASTERN
from ..db import db
from ..db.models import Alpha, AlphaPnL

SUBMITTED_STATUSES = ("ACTIVE", "DECOMMISSIONED")
PAGE_SIZES = (100, 250, 500)
GROUP_LIMIT = 2000                       # the most alphas one analysis reads, as alpha-harness' portfolio takes

INVESTABILITY_LABEL = {"max_trade": "Max Trade", "max_position": "Max Position", "none": "None"}


# ---------------------------------------------------------------- small readers

def names(value: Any, key: str = "name") -> list[str]:
    """BRAIN's label lists as plain names: classifications and pyramids come as [{id, name}] or strings."""
    out: list[str] = []
    for item in value or []:
        if isinstance(item, dict):
            text = item.get(key) or item.get("id")
            if text:
                out.append(str(text))
        elif item is not None:
            out.append(str(item))
    return out


def classifications_of(a: Alpha) -> list[str]:
    return names(a.classifications)


def pyramids_of(a: Alpha) -> list[str]:
    return names(a.pyramids)


def tags_of(a: Alpha) -> list[str]:
    return [str(t) for t in a.tags or [] if isinstance(t, str)]


def expression_of(a: Alpha) -> str:
    return a.code or a.selection or a.combo or ""


def is_submitted(a: Alpha) -> bool:
    """Anything past UNSUBMITTED has been submitted (ACTIVE, DECOMMISSIONED, ...)."""
    return bool(a.status) and str(a.status).upper() != "UNSUBMITTED"


def investability(a: Alpha) -> str:
    return investability_of({"maxTrade": a.max_trade, "maxPosition": a.max_position})


def imbalance(long: float | None, short: float | None) -> float | None:
    """(Long − Short) ÷ (Long + Short) in stocks held; None without both counts."""
    if long is None or short is None or not (long + short):
        return None
    return (long - short) / (long + short)


def calmar(a: Alpha) -> float | None:
    return a.returns / a.drawdown if a.returns is not None and a.drawdown else None


def verdict_of(a: Alpha) -> str | None:
    """BRAIN's reading of the stored checks, worked out again so an old row never carries a stale verdict."""
    return brain_verdict(a.checks, a.simulation_mode)


def power_pool_only(a: Alpha) -> bool:
    """Classed Power Pool alone: BRAIN keeps such an alpha out of the Self-Correlation pool."""
    classes = classifications_of(a)
    return POWER_POOL_CLASS in classes and REGULAR_CLASS not in classes


# ---------------------------------------------------------------- filters

#: Sort keys of the Stored table and the column each sorts by.
SORT_KEYS: dict[str, Any] = {
    "sharpe": Alpha.sharpe, "fitness": Alpha.fitness, "turnover": Alpha.turnover, "returns": Alpha.returns,
    "drawdown": Alpha.drawdown, "margin": Alpha.margin, "operator_count": Alpha.op_count,
    "calmar": case((Alpha.drawdown > 0, Alpha.returns / Alpha.drawdown)),
    "date_created": Alpha.date_created, "date_submitted": Alpha.date_submitted,
    "train_sharpe": Alpha.train_sharpe, "test_sharpe": Alpha.test_sharpe, "test_fitness": Alpha.test_fitness,
    "after_cost": Alpha.after_cost_5bps, "k_ratio": Alpha.k_ratio, "prod_corr": Alpha.prod_corr,
    "pp_corr": Alpha.pp_corr, "long_count": Alpha.long_count, "short_count": Alpha.short_count,
}
SORT_LABELS = {
    "sharpe": "Sharpe", "fitness": "Fitness", "turnover": "Turnover", "returns": "Returns", "drawdown": "Drawdown",
    "margin": "Margin", "operator_count": "Operators", "calmar": "Calmar", "date_created": "Date Created",
    "date_submitted": "Date Submitted", "train_sharpe": "Train Sharpe", "test_sharpe": "Test Sharpe",
    "test_fitness": "Test Fitness", "after_cost": "After-Cost Sharpe", "k_ratio": "K-Ratio",
    "prod_corr": "Production Correlation", "pp_corr": "Power Pool Correlation", "long_count": "Long Count",
    "short_count": "Short Count",
}

#: Bounds as typed (label, column, divisor from the shown unit to BRAIN's fraction).
BOUNDS: dict[str, tuple[str, Any, float]] = {
    "sharpe": ("Sharpe", Alpha.sharpe, 1.0),
    "fitness": ("Fitness", Alpha.fitness, 1.0),
    "turnover": ("Turnover %", Alpha.turnover, 100.0),
    "returns": ("Returns %", Alpha.returns, 100.0),
    "drawdown": ("Drawdown %", Alpha.drawdown, 100.0),
    "margin": ("Margin bps", Alpha.margin, 10_000.0),
    "operator_count": ("Operators", Alpha.op_count, 1.0),
}


@dataclass
class StoredFilter:
    """What the Stored Alphas table filters on. Bounds are in the units shown (%, bps), as typed."""

    submitted: bool = False
    regions: list[str] = field(default_factory=list)
    delays: list[int] = field(default_factory=list)
    universes: list[str] = field(default_factory=list)
    minimum: dict[str, float] = field(default_factory=dict)
    maximum: dict[str, float] = field(default_factory=dict)
    search: str = ""
    tags: list[str] = field(default_factory=list)
    tags_all: bool = False
    created_after: date | None = None
    created_before: date | None = None
    sort_by: str = "sharpe"
    sort_desc: bool = True
    limit: int = 100
    offset: int = 0

    def active(self) -> int:
        return (len(self.regions) + len(self.delays) + len(self.universes) + len(self.minimum) + len(self.maximum)
                + len(self.tags) + (self.created_after is not None) + (self.created_before is not None))


def day_starts(day: date) -> datetime:
    """When `day` starts on BRAIN's New York clock, in UTC: BRAIN's own date filter counts days there."""
    return datetime.combine(day, time(0), EASTERN).astimezone(timezone.utc)


def submitted_clause() -> Any:
    return and_(Alpha.status.is_not(None), Alpha.status != "UNSUBMITTED")


def tags_clause(tags: list[str], every: bool) -> Any:
    """Alphas carrying any of `tags`, or all of them with `every` (read from the JSON list in SQLite)."""
    if every:
        parts = []
        for tag in tags:
            each = func.json_each(Alpha.tags).table_valued("value").alias()
            parts.append(exists(select(1).select_from(each).where(each.c.value == tag)))
        return and_(*parts)
    each = func.json_each(Alpha.tags).table_valued("value").alias()
    return exists(select(1).select_from(each).where(each.c.value.in_(tags)))


def clauses(f: StoredFilter) -> list[Any]:
    out: list[Any] = [submitted_clause() if f.submitted else not_(submitted_clause())]
    if f.regions:
        out.append(Alpha.region.in_(f.regions))
    if f.delays:
        out.append(Alpha.delay.in_([int(d) for d in f.delays]))
    if f.universes:
        out.append(Alpha.universe.in_(f.universes))
    for bounds, op in ((f.minimum, "min"), (f.maximum, "max")):
        for key, value in bounds.items():
            if key not in BOUNDS or value is None:
                continue
            column, div = BOUNDS[key][1], BOUNDS[key][2]
            limit = float(value) / div
            out.append(column >= limit if op == "min" else column <= limit)
    if f.created_after:
        out.append(Alpha.date_created >= day_starts(f.created_after))
    if f.created_before:
        out.append(Alpha.date_created < day_starts(f.created_before + timedelta(days=1)))
    if f.tags:
        out.append(tags_clause(f.tags, f.tags_all))
    if f.search.strip():
        needle = f"%{f.search.strip().lower()}%"
        out.append(or_(func.lower(Alpha.id).like(needle), func.lower(func.coalesce(Alpha.code, "")).like(needle),
                       func.lower(func.coalesce(Alpha.name, "")).like(needle)))
    return out


async def page(f: StoredFilter) -> tuple[int, list[Alpha], set[str]]:
    """One page of the Stored table: total matching, the alphas, and which of them have a daily series stored."""
    where = clauses(f)
    key = SORT_KEYS.get(f.sort_by, Alpha.sharpe)
    order = (key.desc() if f.sort_desc else key.asc()).nulls_last()
    async with db.session() as s:
        total = int(await s.scalar(select(func.count()).select_from(Alpha).where(*where)) or 0)
        rows = list(await s.scalars(select(Alpha).where(*where).order_by(order, Alpha.id)
                                    .limit(max(1, f.limit)).offset(max(0, f.offset))))
        ids = [a.id for a in rows]
        stored = set(await s.scalars(select(AlphaPnL.alpha_id).where(AlphaPnL.alpha_id.in_(ids),
                                                                    AlphaPnL.s_dates.is_not(None)))) if ids else set()
    return total, rows, stored


async def matching_ids(f: StoredFilter, limit: int = GROUP_LIMIT + 1) -> list[str]:
    """Every alpha the filters keep, newest first, up to `limit` (one more than an analysis reads, to say "more")."""
    async with db.session() as s:
        return list(await s.scalars(select(Alpha.id).where(*clauses(f))
                                    .order_by(Alpha.date_created.desc().nulls_last(), Alpha.id).limit(limit)))


async def by_ids(alpha_ids: list[str]) -> dict[str, Alpha]:
    ids = list(dict.fromkeys(alpha_ids))
    out: dict[str, Alpha] = {}
    async with db.session() as s:
        for i in range(0, len(ids), 500):
            for a in await s.scalars(select(Alpha).where(Alpha.id.in_(ids[i:i + 500]))):
                out[a.id] = a
    return out


async def submitted_members() -> list[Alpha]:
    """Every submitted alpha but region-agnostic parents: BRAIN submits a parent's passing children, which carry the
    PnL and count in its pools; the parent has no figures of its own."""
    async with db.session() as s:
        return list(await s.scalars(
            select(Alpha).where(submitted_clause(), func.coalesce(Alpha.sim_type, "REGULAR") != "RA_PARENT")
            .order_by(Alpha.date_submitted.desc().nulls_last(), Alpha.id)))


async def overview() -> dict[str, Any]:
    """Counts for the pool's header, the scopes present (for the chips) and every tag stored alphas carry."""
    async with db.session() as s:
        alphas = int(await s.scalar(select(func.count()).select_from(Alpha)) or 0)
        with_pnl = int(await s.scalar(select(func.count()).select_from(AlphaPnL)
                                      .where(AlphaPnL.s_dates.is_not(None))) or 0)
        submitted = int(await s.scalar(select(func.count()).select_from(Alpha).where(submitted_clause())) or 0)
        scope_rows = (await s.execute(select(Alpha.region, Alpha.delay, Alpha.universe, func.count())
                                      .where(Alpha.region.is_not(None))
                                      .group_by(Alpha.region, Alpha.delay, Alpha.universe)
                                      .order_by(func.count().desc()))).all()
        each = func.json_each(Alpha.tags).table_valued("value").alias()
        tag_rows = (await s.execute(select(each.c.value, func.count()).select_from(Alpha).join(each, Alpha.tags.is_not(None))
                                    .group_by(each.c.value).order_by(each.c.value))).all()
    scopes = [{"region": r, "delay": d, "universe": u, "alphas": int(n)} for r, d, u, n in scope_rows]
    return {
        "alphas": alphas, "with_pnl": with_pnl, "submitted": submitted, "scopes": scopes,
        "regions": sorted({x["region"] for x in scopes if x["region"]}),
        "delays": sorted({int(x["delay"]) for x in scopes if x["delay"] is not None}),
        "universes": sorted({x["universe"] for x in scopes if x["universe"]}),
        "tags": [str(t) for t, _ in tag_rows if isinstance(t, str)],
        "tag_counts": {str(t): int(n) for t, n in tag_rows if isinstance(t, str)},
    }


async def save_tags(tags: dict[str, list[str]]) -> int:
    """Replace stored alphas' tags and touch nothing else; an alpha not stored yet arrives whole with the next sync."""
    done = 0
    async with db.write() as s:
        for alpha_id, kept in tags.items():
            row = await s.get(Alpha, alpha_id)
            if row is not None:
                row.tags = list(kept)
                done += 1
    return done


# ---------------------------------------------------------------- region-agnostic families

def _merit(a: Alpha) -> float | None:
    """What a family is ranked by: Train Fitness, or whole-period Fitness without a test split."""
    value = a.train_fitness if a.train_fitness is not None else a.fitness
    return float(value) if value is not None else None


def is_child_of(parent: Alpha, child: Alpha) -> bool:
    """BRAIN writes a family in one second from one expression and one set of settings, and rows carry no parent id,
    so a child is told by those (as alpha-harness `family_of`)."""
    return (child.sim_type == "RA_CHILD" and child.code == parent.code and child.date_created == parent.date_created
            and child.delay == parent.delay and child.neutralization == parent.neutralization
            and child.decay == parent.decay and child.truncation == parent.truncation)


def family_row(parent: Alpha, children: list[Alpha]) -> dict[str, Any] | None:
    """A parent as its family stands: its second-best region's best child, or None below two scored regions.

    BRAIN submits a region-agnostic alpha only when two regions pass, so the family is as good as that child."""
    ranked = sorted(((m, c) for c in children if (m := _merit(c)) is not None), key=lambda p: (-p[0], p[1].id))
    best: dict[str, tuple[float, Alpha]] = {}
    for merit, child in ranked:
        best.setdefault(str(child.region), (merit, child))
    if len(best) < 2:
        return None
    scored = [c for _, c in sorted(best.values(), key=lambda p: (-p[0], p[1].id))]
    return {"standing": scored[1], "regions": [c.region for c in scored],
            "region_series": {str(c.region): c.id for c in scored}}


def children_check(children: list[Alpha]) -> dict[str, Any]:
    """The parent judged as BRAIN judges it, as one check: PASS once two children pass every check, PENDING while
    two still could, FAIL otherwise."""
    verdicts = [verdict_of(c) for c in children]
    passing = sum(v == "submittable" for v in verdicts)
    possible = passing + sum(v == "pending" for v in verdicts)
    result = ("PASS" if passing >= RA_MIN_PASSING_CHILDREN
              else "PENDING" if possible >= RA_MIN_PASSING_CHILDREN else "FAIL")
    return {"name": "MIN_SUBMITTABLE_CHILDREN", "result": result, "value": passing, "limit": RA_MIN_PASSING_CHILDREN}


async def ra_families(parent_ids: list[str] | None = None, limit: int = 200,
                      unsubmitted: bool = True) -> list[tuple[Alpha, list[Alpha]]]:
    """Region-agnostic parents (these, or the newest `limit`) with their children."""
    async with db.session() as s:
        stmt = select(Alpha).where(Alpha.sim_type == "RA_PARENT")
        if parent_ids is not None:
            stmt = stmt.where(Alpha.id.in_(parent_ids))
        if unsubmitted:
            stmt = stmt.where(not_(submitted_clause()))
        parents = list(await s.scalars(stmt.order_by(Alpha.date_created.desc().nulls_last(), Alpha.id).limit(limit)))
        if not parents:
            return []
        stamps = {p.date_created for p in parents if p.date_created is not None}
        children = list(await s.scalars(select(Alpha).where(Alpha.sim_type == "RA_CHILD",
                                                            Alpha.date_created.in_(stamps)))) if stamps else []
    return [(p, sorted((c for c in children if is_child_of(p, c)), key=lambda c: str(c.region))) for p in parents]


# ---------------------------------------------------------------- table rows

def figure_tone(key: str, value: float | None, limits: dict[str, float]) -> str:
    """alpha-harness `figureTone`: green only when a figure meets the limit the alpha's own check names, red when it
    is negative (or turnover leaves its band); plain otherwise. Returns 'pos' | 'neg' | ''."""
    if value is None:
        return ""
    if key == "returns":
        return "pos" if value > 0 else "neg" if value < 0 else ""
    if key in ("sharpe", "fitness"):
        floor = limits.get("LOW_SHARPE" if key == "sharpe" else "LOW_FITNESS")
        return "neg" if value < 0 else "pos" if floor is not None and value >= floor else ""
    if key == "turnover":
        low, high = limits.get("LOW_TURNOVER"), limits.get("HIGH_TURNOVER")
        if (low is not None and value < low) or (high is not None and value > high):
            return "neg"
        return "pos" if low is not None and high is not None else ""
    if key == "margin":
        return "neg" if value < 0 else ""
    return ""


def ceiling_tone(value: float | None, ceiling: float | None) -> str:
    """A correlation: green below the ceiling BRAIN holds it to, amber at or above."""
    if value is None or ceiling is None:
        return ""
    return "warn" if value >= ceiling else "pos"


def counts_by_market(alphas: list[Alpha]) -> list[dict[str, Any]]:
    found = Counter((a.instrument_type or "EQUITY", a.region, a.delay, a.universe) for a in alphas)
    return [{"instrument_type": i, "region": r, "delay": d, "universe": u, "count": n}
            for (i, r, d, u), n in found.most_common()]


def limits_of(a: Alpha) -> dict[str, float]:
    return check_limits(a.checks)


def failed_of(a: Alpha) -> list[str]:
    return failed_checks(a.checks)
