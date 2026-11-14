"""Evolution Lab: breed new Alphas from seed Alphas, scored on the train years only (alpha-harness labs/ga.py and
api/ga.py).

A hybrid of a gene GA and tree GP, because seeds are arbitrary expressions:

* **Genes** are what a mutation may change without changing the shape: an operator for a peer taking the same
  inputs, a lookback for a neighbouring economic window, a field for another in its dataset, a group for another
  group, an arithmetic operator for another, and the neutralization and decay.
* **Crossover** is uniform, gene by gene, when two parents share a shape; otherwise a call of one is grafted into
  the other, never growing past the larger parent.

Every child is simulated holding the last two years out as a test and scored on Train Fitness. Parents are the best
of everything scored so far, kept apart by the correlation of their daily PnL over the train years (downloaded in
the background where missing). A task stops early once 1,000 children found no new best.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Sequence

from sqlalchemy import select

from ..brain.platform import IGNORED_CHECKS, IMPLAUSIBLE_SHARPE, MIN_OVERLAP, TEST_PERIOD, UNCOUNTED_OPERATORS
from ..db import db
from ..db.models import Alpha, DataField
from ..db.models_labs import LabTask, LabTrial
from . import launch
from .launch import NO_NEUTRALIZATION, OPERATORS_UNREAD
from .runner import FAILURE, FREE_MESSAGE, Lab, RefillContext, TrialSpec, trial_key
from .search import DECAYS, EXCLUDED, GROUPS, MAX_CORES, TRUNCATION
from .tree import (
    Node,
    OperatorInfo,
    ParseError,
    Path,
    graft,
    graft_sites,
    grafts,
    guarded,
    node_at,
    operator_count,
    operator_table,
    parse,
    relabel,
    render,
    replace_at,
    validate,
    walk_with_parents,
    written_out,
)

log = logging.getLogger(__name__)

WINDOWS = (5, 10, 21, 42, 63, 126, 252)
ARITHMETIC = ("+", "-", "*", "/")
MUTATION_RATE = 0.05
MUTATION_RATES = (0.03, 0.05, 0.08)
POPULATIONS = (50, 100, 200)
MAX_SEEDS = 100
#: Children spent without a new best before a task stops early.
PATIENCE = 1_000
#: Candidates Auto Select examines at most, so a weak market cannot cost hundreds of downloads.
MAX_EXAMINED = 200
STALLED = "Stopped early: no new best Train Fitness in the last 1,000 simulations."
NO_CHILD = ("No new child could be bred: every child of these parents was already simulated or is not a valid "
            "Alpha here.")
NOT_SWAPPED = frozenset({"ts_step"})
NOT_SWAPPED_CATEGORIES = frozenset({"Vector", "Special", "Reduce"})
#: Two Alphas moving together at |correlation| of this or more are one idea, not two.
DEFAULT_MAX_CORRELATION = 0.5
SUBMITTED = ("ACTIVE", "DECOMMISSIONED")


# --- seed scores ------------------------------------------------------------

_CURVE = 2.42           # the Sharpe curve ln(x - 0.58) from 1.58 to 4.0: a reward phase 2.42 wide
_SLOPE = 27.5 / 3       # the turnover curve falls 0.5 a point past 40%, against a best of 1.5 over 27.5 points


def more(x: float, low: float, high: float) -> float:
    """Punished below `low`, a log reward to `high`, then flat."""
    t = (x - low) / (high - low)
    if t < 0:
        return -0.5 * (_CURVE * t) ** 2 / math.log1p(_CURVE)
    return math.log1p(_CURVE * min(t, 1.0)) / math.log1p(_CURVE)


def less(x: float, best: float, worst: float) -> float:
    """Full reward to `best`, a quadratic fall to `worst`, then a steep line."""
    t = (x - best) / (worst - best)
    if t <= 0:
        return 1.0
    if t <= 1:
        return 1.0 - t * t
    return -_SLOPE * (t - 1.0)


#: Weight, curve and anchors per stored metric. Returns plateau at 15% so a volatile Alpha cannot buy its rank.
UTILITY: dict[str, tuple[float, Callable[[float, float, float], float], float, float]] = {
    "fitness": (0.30, more, 1.0, 2.5),
    "sharpe": (0.25, more, 1.58, 4.0),
    "turnover": (0.15, less, 0.125, 0.40),
    "margin": (0.10, more, 0.0005, 0.0020),
    "drawdown": (0.10, less, 0.05, 0.20),
    "returns": (0.10, more, 0.05, 0.15),
}
#: The check whose limit is BRAIN's own cutoff for a metric: it replaces the curve's low anchor.
CUTOFFS = {"sharpe": "LOW_SHARPE", "fitness": "LOW_FITNESS"}
#: Checks measuring the last two years, which the train/test split holds out.
TEST_YEAR_CHECKS = frozenset({"LOW_2Y_SHARPE", "IS_LADDER_SHARPE"})


def limits_of(checks: list[dict[str, Any]] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in checks or []:
        if isinstance(c, dict) and c.get("limit") is not None:
            try:
                out[str(c.get("name"))] = float(c["limit"])
            except (TypeError, ValueError):
                continue
    return out


def train_figure(row: dict[str, Any], key: str) -> Any:
    """`key` over the train years where the Alpha has it, so the held-out years pick nothing."""
    train = row.get(f"train_{key}")
    return row.get(key) if train is None else train


def utility(row: dict[str, Any]) -> float:
    """The weighted utility of an Alpha's stored metrics. A missing metric leaves the weights."""
    total = weight = 0.0
    limits = limits_of(row.get("checks"))
    for key, (share, curve, low, high) in UTILITY.items():
        cutoff = limits.get(CUTOFFS.get(key, ""))
        if cutoff is not None:
            low, high = cutoff, high + cutoff - low         # the whole curve moves, the reward phase keeps its width
        value = train_figure(row, key)
        if value is not None:
            total += share * curve(float(value), low, high)
            weight += share
    return total / weight if weight else -math.inf


def pass_rate(checks: list[Any] | None) -> float | None:
    """The share of judged checks passed; pending, labelling and test-year checks are not judged."""
    skipped = IGNORED_CHECKS | TEST_YEAR_CHECKS
    judged = [str(c.get("result", "")).upper() for c in checks or []
              if isinstance(c, dict) and str(c.get("name", "")).upper() not in skipped]
    judged = [r for r in judged if r in ("PASS", "FAIL", "WARNING", "ERROR")]
    return sum(r == "PASS" for r in judged) / len(judged) if judged else None


def seed_score(row: dict[str, Any]) -> float:
    """Utility scaled by the checks passed, S x (1 + passed / total); a negative score is divided instead, so
    passing more checks always ranks higher."""
    score = utility(row)
    rate = pass_rate(row.get("checks"))
    if rate is None or not math.isfinite(score):
        return score
    return score * (1 + rate) if score >= 0 else score / (1 + rate)


def population_for(simulations: int, chosen: int | None = None) -> int:
    """A hundred, smaller for a small task so it breeds about twenty generations, and never below twenty."""
    return chosen or min(100, max(20, 10 * (simulations // 200)))


def stalled(values: list[float | None]) -> bool:
    """True when the last PATIENCE spent children found nothing better than those before (None = failed)."""
    if len(values) <= PATIENCE:
        return False
    before = [v for v in values[:-PATIENCE] if v is not None]
    recent = [v for v in values[-PATIENCE:] if v is not None]
    return bool(before) and (not recent or max(recent) <= max(before))


# --- the market ---------------------------------------------------------------

def _inputs(info: OperatorInfo, count: int) -> tuple[str, ...]:
    return info.inputs[:count]


@dataclass(frozen=True)
class Market:
    """What a child in one market may be made of."""

    table: dict[str, OperatorInfo]
    swappable: dict[str, tuple[OperatorInfo, ...]]       # operators a mutation may swap in, by category
    dataset_of: dict[str, str]                            # matrix fields and their dataset
    by_dataset: dict[str, tuple[str, ...]]
    names: frozenset[str]                                 # every field of the market and the grouping fields
    neutralizations: tuple[str, ...]

    def peers(self, node: Node) -> list[str]:
        """Operators that can stand in for this call: same category, the same inputs."""
        info = self.table.get(node.value)
        if info is None or node.kwargs or node.value in UNCOUNTED_OPERATORS:
            return []
        count = len(node.args)
        inputs = _inputs(info, count)
        return [o.name for o in self.swappable.get(info.category, ()) if o.name != info.name
                and o.required <= count and (o.maximum is None or count <= o.maximum) and _inputs(o, count) == inputs]

    def genes(self, tree: Node) -> list[tuple[Path, str]]:
        """Where a mutation may change this tree, and what kind of change each is."""
        found: list[tuple[Path, str]] = []
        for path, node, parent in walk_with_parents(tree):
            if node.kind == "binary" and node.value in ARITHMETIC:
                found.append((path, "op"))
            elif node.kind == "call":
                if self.peers(node):
                    found.append((path, "op"))
                info = self.table.get(node.value)
                if info is None or node.value in UNCOUNTED_OPERATORS:
                    continue
                for index, (arg, name) in enumerate(zip(node.args, _inputs(info, len(node.args)))):
                    if name == "d" and arg.kind == "num" and arg.value.isdigit():
                        found.append(((*path, index), "window"))
            elif node.kind == "name":
                grouping = (parent is not None and parent.kind == "call" and parent.value.startswith("group_")
                            and 1 <= path[-1] < len(parent.args))
                dataset = self.dataset_of.get(node.value)
                if grouping and node.value in GROUPS:
                    found.append((path, "group"))
                elif dataset and not guarded(parent, path) and len(self.by_dataset[dataset]) > 1:
                    found.append((path, "field"))
        return found


def build_market(operators: list[dict[str, Any]], fields: list[dict[str, Any]], neutralizations: list[str]) -> Market:
    table = operator_table(operators)
    levels = {o.get("name"): o.get("level") for o in operators}
    # Only ALL-level operators are swapped in; one listed without a level is never swapped in (a seed may keep it).
    leveled = any(level is not None for level in levels.values())
    swappable: dict[str, list[OperatorInfo]] = defaultdict(list)
    for info in table.values():
        if (info.name in UNCOUNTED_OPERATORS or info.name in EXCLUDED or info.name in NOT_SWAPPED
                or info.category in NOT_SWAPPED_CATEGORIES or (leveled and levels.get(info.name) != "ALL")):
            continue
        swappable[info.category].append(info)
    dataset_of = {str(f["field_id"]): str(f["dataset_id"]) for f in fields
                  if f.get("field_type") == "MATRIX" and f.get("dataset_id")}
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for fid, dataset in dataset_of.items():
        by_dataset[dataset].append(fid)
    return Market(table=table, swappable={k: tuple(v) for k, v in swappable.items()}, dataset_of=dataset_of,
                  by_dataset={k: tuple(v) for k, v in by_dataset.items()},
                  names=frozenset(str(f["field_id"]) for f in fields) | frozenset(GROUPS),
                  neutralizations=tuple(neutralizations))


_markets: dict[tuple[Any, ...], Market] = {}


async def market_for(region: str, delay: int, universe: str, neutralizations: list[str],
                     operators: list[dict[str, Any]] | None = None) -> Market | None:
    """The market's fields with the account's operators, or None when either is missing (kept until restart)."""
    operators = await launch.account_operators() if operators is None else operators
    key = (region, int(delay), universe, len(operators))
    if key not in _markets:
        async with db.session() as s:
            rows = (await s.execute(select(DataField.id, DataField.dataset_id, DataField.type).where(
                DataField.region == region, DataField.delay == int(delay), DataField.universe == universe))).all()
        if not rows or not operators:
            return None
        fields = [{"field_id": f, "dataset_id": d, "field_type": t or "MATRIX"} for f, d, t in rows]
        _markets[key] = build_market(operators, fields, neutralizations)
    return replace(_markets[key], neutralizations=tuple(neutralizations))


def unreadable(expression: str | None, market: Market) -> str | None:
    """Why an expression cannot breed in this market, or None."""
    try:
        tree = parse(expression or "")
    except ParseError as exc:
        return f"Its expression could not be read: {exc}"
    problems = validate(tree, market.table, market.names)
    return problems[0] if problems else None


# --- breeding ---------------------------------------------------------------

@dataclass(frozen=True)
class Parent:
    alpha_id: str | None
    tree: Node
    neutralization: str
    decay: int
    truncation: float

    @classmethod
    def of(cls, alpha_id: str | None, expression: str | None, settings: dict[str, Any]) -> Parent | None:
        try:
            tree = parse(expression or "")
        except ParseError:
            return None
        return cls(alpha_id, tree, str(settings.get("neutralization") or "NONE"), int(settings.get("decay") or 0),
                   float(settings.get("truncation") or TRUNCATION))

    @property
    def key(self) -> tuple[Node, str, int, float]:
        return self.tree, self.neutralization, self.decay, self.truncation


def skeleton(tree: Node, genes: list[tuple[Path, str]]) -> Node:
    """The tree with its genes blanked: parents sharing one cross gene by gene."""
    return relabel(tree, {path: f"?{kind}" for path, kind in genes})


def _field_groups(tree: Node, genes: list[tuple[Path, str]]) -> dict[Path, str]:
    """Each field gene's path, by the field it reads: a field read twice is one input."""
    return {path: node_at(tree, path).value for path, kind in genes if kind == "field"}


def uniform(rng: random.Random, a: Node, b: Node, genes: list[tuple[Path, str]]) -> Node:
    """Each gene from either parent with equal chance; every reading of one field comes from the same parent, so
    ``x / ts_mean(x, 20)`` stays a ratio of one thing."""
    fields = _field_groups(a, genes)
    from_b = {name: rng.random() < 0.5 for name in dict.fromkeys(fields.values())}
    taken = {path: node_at(b, path).value for path, _ in genes
             if (from_b[fields[path]] if path in fields else rng.random() < 0.5)}
    return relabel(a, taken)


def _mutated(rng: random.Random, node: Node, kind: str, market: Market) -> str:
    if kind == "window":
        current = int(node.value)
        if current not in WINDOWS:
            return str(min(WINDOWS, key=lambda w: abs(w - current)))
        at = WINDOWS.index(current)
        return str(rng.choice([WINDOWS[i] for i in (at - 1, at + 1) if 0 <= i < len(WINDOWS)]))
    if kind == "field":
        dataset = market.dataset_of[node.value]
        return rng.choice([f for f in market.by_dataset[dataset] if f != node.value])
    if kind == "group":
        return rng.choice([g for g in GROUPS if g != node.value])
    if node.kind == "binary":
        return rng.choice([o for o in ARITHMETIC if o != node.value])
    return rng.choice(market.peers(node))


def mutate(rng: random.Random, tree: Node, settings: dict[str, Any], market: Market, rate: float, *,
           force: bool = False, keep_decay: bool = False) -> tuple[Node, dict[str, Any]]:
    """Change each gene with probability `rate`; with `force`, at least one where any can change. `keep_decay`
    leaves the decay as it was inherited."""
    moves: list[tuple[Path, str]] = market.genes(tree)
    if any(n != settings["neutralization"] for n in market.neutralizations):
        moves.append(((), "neutralization"))
    if not keep_decay:
        moves.append(((), "decay"))
    chosen = [move for move in moves if rng.random() < rate]
    if force and not chosen and moves:
        chosen = [rng.choice(moves)]
    settings = dict(settings)
    values: dict[Path, str] = {}
    for path, kind in chosen:
        if kind == "neutralization":
            settings[kind] = rng.choice([n for n in market.neutralizations if n != settings["neutralization"]])
        elif kind == "decay":
            settings[kind] = rng.choice([d for d in DECAYS if d != settings["decay"]])
        else:
            values[path] = _mutated(rng, node_at(tree, path), kind, market)
    # A field read twice is swapped everywhere it is read, to the first value drawn for it.
    fields = _field_groups(tree, moves)
    swapped: dict[str, str] = {}
    for path, value in values.items():
        if path in fields:
            swapped.setdefault(fields[path], value)
    values |= {path: swapped[name] for path, name in fields.items() if name in swapped}
    child = relabel(tree, values)
    for path, value in values.items():
        # An operator swapped in gets the options BRAIN will not default; the parent's own calls stay as accepted.
        node = node_at(child, path)
        if node.kind == "call" and (info := market.table.get(value)) is not None:
            child = replace_at(child, path, written_out(node, info))
    return child, settings


@dataclass(frozen=True)
class Shape:
    """What breeding reads of a parent on every attempt, worked out once per parent."""

    genes: list[tuple[Path, str]]
    skeleton: Node
    operators: int
    sites: list[Path]
    grafts: list[Node]

    @classmethod
    def of(cls, tree: Node, market: Market) -> Shape:
        genes = market.genes(tree)
        return cls(genes, skeleton(tree, genes), operator_count(tree), graft_sites(tree), grafts(tree))


def child(rng: random.Random, a: Parent, b: Parent, market: Market, rate: float, shapes: dict[Node, Shape], *,
          keep_decay: bool = False) -> tuple[Node, dict[str, Any]] | None:
    """A valid child of `a` and `b` that is neither of them (a clone is rejected), or None after ten tries."""
    shape_a, shape_b = shapes[a.tree], shapes[b.tree]
    genes = shape_a.genes
    same = shape_a.skeleton == shape_b.skeleton
    limit = max(shape_a.operators, shape_b.operators)
    parents = {a.key, b.key}

    def clone(tree: Node, settings: dict[str, Any]) -> bool:
        return (tree, settings["neutralization"], settings["decay"], settings["truncation"]) in parents

    for _ in range(10):
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        if same:
            tree = uniform(rng, a.tree, b.tree, genes)
        else:
            tree = graft(rng, left.tree, shapes[left.tree].sites, shapes[right.tree].grafts) or left.tree
            if operator_count(tree) > limit:
                continue
        settings = {"neutralization": rng.choice((a.neutralization, b.neutralization)),
                    "decay": rng.choice((a.decay, b.decay)), "truncation": left.truncation}
        tree, settings = mutate(rng, tree, settings, market, rate, keep_decay=keep_decay)
        if clone(tree, settings):
            tree, settings = mutate(rng, tree, settings, market, 0.0, force=True, keep_decay=keep_decay)
        if clone(tree, settings) or validate(tree, market.table, market.names):
            continue
        return tree, settings
    return None


def settings_for(settings: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    return {"instrumentType": "EQUITY", "region": run["region"], "delay": int(run["delay"]),
            "universe": run["universe"], "neutralization": settings["neutralization"],
            "decay": int(settings["decay"]), "truncation": float(settings["truncation"]), "testPeriod": TEST_PERIOD}


def ga_key(expression: str | None, settings: dict[str, Any] | None) -> str:
    """One child whatever its test period: a seed is that child holding nothing out."""
    s = {k: v for k, v in (settings or {}).items() if k != "testPeriod"}
    return trial_key(expression or "", s)


# --- keeping parents apart ----------------------------------------------------

#: An Alpha's train-year daily PnL: (dates, values).
Profile = tuple[list[str], list[float]]


def kinship(a: Profile, b: Profile) -> float | None:
    """Pearson correlation of two Alphas' daily PnL over the days both have (None below 250 shared days)."""
    from ..engine.metrics import pair_correlation
    at = {d: i for i, d in enumerate(a[0])}
    pairs = [(a[1][at[d]], v) for d, v in zip(b[0], b[1]) if d in at]
    if len(pairs) < MIN_OVERLAP:
        return None
    return pair_correlation([x for x, _ in pairs], [y for _, y in pairs])


def collision(candidate: Profile, kept: dict[str, Profile],
              threshold: float = DEFAULT_MAX_CORRELATION) -> tuple[str, float] | None:
    """The first kept Alpha this one moves with, at |correlation| of `threshold` or more."""
    for alpha_id, other in kept.items():
        rho = kinship(candidate, other)
        if rho is not None and abs(rho) >= threshold:
            return alpha_id, rho
    return None


def independent(order: list[str], series: dict[str, Profile | None], limit: int,
                threshold: float = DEFAULT_MAX_CORRELATION) -> list[str]:
    """Best first, each kept unless it moves with one already kept. An Alpha without a usable series counts as
    independent."""
    kept: list[str] = []
    profiles: dict[str, Profile] = {}
    for alpha_id in order:
        if len(kept) >= limit:
            break
        own = series.get(alpha_id)
        if own is not None:
            if collision(own, profiles, threshold):
                continue
            profiles[alpha_id] = own
        kept.append(alpha_id)
    return kept


async def train_profiles(alpha_ids: Sequence[str]) -> dict[str, Profile]:
    """Each Alpha's stored daily PnL over its train years (before the test period starts)."""
    from ..engine import series
    loaded = await series.load(alpha_ids)
    out: dict[str, Profile] = {}
    for aid, ser in loaded.items():
        alpha = ser.get("alpha")
        start = getattr(alpha, "test_start", None)
        pairs = [(d, p) for d, p in zip(ser["dates"], ser["pnl"]) if p is not None and (not start or d < start)]
        if len(pairs) >= MIN_OVERLAP:
            out[aid] = ([d for d, _ in pairs], [float(p) for _, p in pairs])
    return out


# --- seeds -------------------------------------------------------------------

def seed_row(a: Alpha) -> dict[str, Any]:
    checks = a.is_checks if a.is_checks is not None else a.checks
    return {"alpha_id": a.id, "expression": a.code or "", "sharpe": a.sharpe, "fitness": a.fitness,
            "turnover": a.turnover, "returns": a.returns, "drawdown": a.drawdown, "margin": a.margin,
            "train_sharpe": a.train_sharpe, "train_fitness": a.train_fitness, "test_sharpe": a.test_sharpe,
            "test_fitness": a.test_fitness, "checks": checks or [], "region": a.region, "delay": a.delay,
            "universe": a.universe, "status": a.status, "sim_type": a.sim_type or a.type, "type": a.type,
            "neutralization": a.neutralization or (a.settings or {}).get("neutralization"),
            "decay": a.decay if a.decay is not None else (a.settings or {}).get("decay"),
            "truncation": a.truncation if a.truncation is not None else (a.settings or {}).get("truncation"),
            "submitted": (a.status or "").upper() in SUBMITTED or a.date_submitted is not None}


def seed_problem(row: dict[str, Any] | None, region: str, delay: int, universe: str, market: Market | None) -> str | None:
    """Why a stored Alpha cannot be a seed in this market, or None."""
    if row is None or not row.get("expression"):
        return "Not stored here. Sync from BRAIN first."
    if row.get("submitted"):
        return "Already submitted."
    kind = str(row.get("sim_type") or "REGULAR").upper()
    if kind == "RA_PARENT":
        return "A region-agnostic Alpha breeds in All Regions. Here, seed from one of its regions."
    if kind not in ("REGULAR", "RA_CHILD"):
        return f"A {kind} Alpha cannot be a seed: its expression is not a regular expression."
    if (row.get("region"), row.get("delay"), row.get("universe")) != (region, int(delay), universe):
        return f"It ran in {row.get('region')} D{row.get('delay')} {row.get('universe')}."
    if row.get("fitness") is None:
        return "It has no Fitness yet."
    return unreadable(row["expression"], market) if market else None


def seed_view(row: dict[str, Any]) -> dict[str, Any]:
    score = seed_score(row)
    return {"alpha_id": row["alpha_id"], "expression": row.get("expression") or "",
            **{k: row.get(k) for k in ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin",
                                       "train_sharpe", "train_fitness", "test_sharpe", "test_fitness")},
            "score": round(score, 4) if math.isfinite(score) else None, "limits": limits_of(row.get("checks"))}


async def stored_seeds(ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    ids = list(dict.fromkeys(ids))
    if not ids:
        return {}
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(Alpha.id.in_(ids))))
    return {a.id: seed_row(a) for a in rows}


async def evolvable_markets() -> list[dict[str, Any]]:
    """Markets holding unsubmitted regular Alphas, and how many."""
    from sqlalchemy import func
    async with db.session() as s:
        rows = (await s.execute(select(Alpha.region, Alpha.delay, Alpha.universe, func.count(Alpha.id)).where(
            Alpha.code.is_not(None), Alpha.type == "REGULAR", Alpha.region.is_not(None),
            Alpha.universe.is_not(None), ~Alpha.status.in_(SUBMITTED)).group_by(
            Alpha.region, Alpha.delay, Alpha.universe))).all()
    return sorted(({"region": r, "delay": int(d or 0), "universe": u, "alphas": int(n)} for r, d, u, n in rows),
                  key=lambda m: -m["alphas"])


async def candidates(region: str, delay: int, universe: str, limit: int = 500) -> list[dict[str, Any]]:
    """Unsubmitted Alphas of a market, best seed score first: what the seed picker lists."""
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(
            Alpha.region == region, Alpha.delay == int(delay), Alpha.universe == universe, Alpha.code.is_not(None),
            Alpha.type == "REGULAR", ~Alpha.status.in_(SUBMITTED)).limit(5000)))
    seeds = [seed_row(a) for a in rows if a.sim_type != "RA_PARENT" and a.fitness is not None]
    seeds.sort(key=seed_score, reverse=True)
    return seeds[:limit]


async def auto_seeds(market: Market, region: str, delay: int, universe: str, want: int,
                     report: Callable[[int, int], Awaitable[None]] | None = None) -> dict[str, Any]:
    """The best, mutually uncorrelated unsubmitted Alphas of a market, and why the rest were not chosen. Downloads
    daily PnL where missing; simulates nothing. At most MAX_EXAMINED are examined."""
    from ..brain.platform import is_implausible
    from ..engine import series
    pool = await candidates(region, delay, universe, limit=5000)
    rows = [r for r in pool if (train_figure(r, "sharpe") or 0) > 0 and train_figure(r, "fitness") is not None
            and not is_implausible(train_figure(r, "sharpe")) and not r.get("submitted")]
    kept: list[dict[str, Any]] = []
    kept_series: dict[str, Profile] = {}
    reasons: list[dict[str, str]] = []
    examined = errors = 0
    first_error: Exception | None = None
    for row in rows:
        if len(kept) >= want or examined >= MAX_EXAMINED:
            break
        examined += 1
        aid = str(row["alpha_id"])
        problem: str | None
        own: Profile | None = None
        try:
            problem = unreadable(row["expression"], market)
            if not problem:
                profiles = await train_profiles([aid])
                if aid not in profiles:
                    await series.fetch_series(aid, turnover=False)
                    profiles = await train_profiles([aid])
                own = profiles.get(aid)
                if own is None:
                    problem = "No usable daily PnL."
                elif hit := collision(own, kept_series):
                    problem = f"Moves with {hit[0]}: correlation {hit[1]:.2f}."
        except Exception as exc:          # one unreadable Alpha costs one seed, not the whole selection
            log.warning("auto seed %s failed", aid, exc_info=True)
            first_error = first_error or exc
            errors += 1
            problem = f"Could not be read: {exc}"
        if problem or own is None:
            reasons.append({"alpha_id": aid, "reason": problem or "No usable daily PnL."})
        else:
            kept.append(row)
            kept_series[aid] = own
        if report:
            await report(len(kept), examined)
    if first_error is not None and errors == examined:
        raise first_error
    return {"seeds": [seed_view(r) for r in kept], "wanted": want, "pool": len(rows), "examined": examined,
            "reasons": reasons}


AUTO_KEY = "labs:auto_seeds:{region}:{delay}:{universe}"


async def _job_auto_seeds(ctx: Any, params: dict[str, Any]) -> str:
    region, delay, universe = str(params["region"]), int(params["delay"]), str(params["universe"])
    want = max(2, min(int(params.get("count") or 50), MAX_SEEDS))
    market = await market_for(region, delay, universe, launch.neutralizations_for(region))
    if market is None:
        raise RuntimeError(f"{region} delay {delay} {universe} is not downloaded. Sync it in the Data Explorer.")

    async def report(kept: int, examined: int) -> None:
        await ctx.update(kept / want, f"{kept} of {want} seeds · {examined} examined")

    found = await auto_seeds(market, region, delay, universe, want, report)
    await db.kv_set(AUTO_KEY.format(region=region, delay=delay, universe=universe), found)
    return f"Chose {len(found['seeds'])} of {want} seeds ({found['examined']} examined, {found['pool']} in the market)."


# --- planning -----------------------------------------------------------------

async def plan(body: dict[str, Any]) -> dict[str, Any]:
    """Everything a task would breed from, checked, without queueing anything.

    body: region, delay, universe, alpha_ids, neutralizations, population, mutation_rate, keep_decay, simulations."""
    region, delay, universe = str(body.get("region") or "USA"), int(body.get("delay") or 1), str(body.get("universe")
                                                                                                  or "TOP3000")
    problems: list[str] = []
    operators = await launch.account_operators()
    if not operators:
        problems.append(OPERATORS_UNREAD)
    chosen = list(body.get("neutralizations") or [])
    neutralizations = launch.neutralizations_for(region, chosen)
    if not chosen:
        problems.append(NO_NEUTRALIZATION)
    elif not neutralizations:
        problems.append(f"BRAIN offers none of the chosen neutralizations in {region} delay {delay}.")
    market = await market_for(region, delay, universe, neutralizations, operators) if operators else None
    if market is None and operators:
        problems.append(f"{region} delay {delay} {universe} is not downloaded. Sync it in the Data Explorer.")
    rate = float(body.get("mutation_rate") or MUTATION_RATE)
    if not 0.01 <= rate <= 0.2:
        problems.append("Mutation rate must be from 0.01 to 0.2.")
    ids = list(dict.fromkeys(body.get("alpha_ids") or []))[:MAX_SEEDS]
    rows = await stored_seeds(ids)
    seeds, skipped = [], []
    for aid in ids:
        reason = seed_problem(rows.get(aid), region, delay, universe, market)
        if reason:
            skipped.append({"alpha_id": aid, "reason": reason})
        else:
            seeds.append(rows[aid])
    seeds.sort(key=seed_score, reverse=True)
    if len(seeds) < 2:
        problems.append("Choose at least 2 seeds this market can breed from.")
    sims = int(body.get("simulations") or 0)
    population = population_for(sims, body.get("population") or None)
    run = {"region": region, "delay": delay, "universe": universe}
    sample: list[dict[str, Any]] = []
    if market is not None and len(seeds) >= 2 and not problems:
        parents = [p for r in seeds if (p := Parent.of(r["alpha_id"], r["expression"], r))]
        shapes = {p.tree: Shape.of(p.tree, market) for p in parents}

        def draw(rng: random.Random) -> dict[str, Any] | None:
            a, b = rng.choice(parents), rng.choice(parents)
            made = child(rng, a, b, market, rate, shapes, keep_decay=bool(body.get("keep_decay")))
            return {"expression": render(made[0]), "settings": settings_for(made[1], run)} if made else None

        sample = launch.preview_samples(draw, tries=50)
    return {"seeds": [seed_view(r) for r in seeds], "seed_rows": seeds, "skipped": skipped, "population": population,
            "generations": sims // population if population else 0, "sample": sample, "problems": problems,
            "neutralizations": list(neutralizations), "run": run, "mutation_rate": rate}


# --- the lab -------------------------------------------------------------------

class EvolutionLab(Lab):
    name = "evolution"
    title = "Evolution Lab"
    objective = "train_fitness"
    max_cores = MAX_CORES

    def __init__(self) -> None:
        self._stalled: dict[int, bool] = {}
        self._fetched: dict[int, set[str]] = {}
        self._nothing: dict[int, str] = {}

    async def plan(self, task: LabTask) -> dict[str, Any]:
        params = dict(task.params or {})
        planned = await plan({**params, "simulations": task.max_trials})
        if planned["problems"]:
            raise ValueError(planned["problems"][0])
        params.update(run=planned["run"], population=planned["population"], mutation_rate=planned["mutation_rate"],
                      neutralizations=planned["neutralizations"],
                      seeds=[r["alpha_id"] for r in planned["seed_rows"]], seed_rows=planned["seed_rows"])
        task.params = params
        task.universe = planned["run"]["universe"]
        return {"neutralizations": planned["neutralizations"]}

    async def initial(self, task: LabTask) -> list[TrialSpec]:
        """Generation 0: the seeds, scored on their stored Train Fitness (whole-period Fitness without a split, which
        would let the held-out years pick the parents); nothing is simulated."""
        run = task.params["run"]
        out = []
        for seed in task.params.get("seed_rows") or []:
            settings = {"instrumentType": "EQUITY", "region": run["region"], "delay": run["delay"],
                        "universe": run["universe"], "neutralization": seed.get("neutralization") or "NONE",
                        "decay": int(seed.get("decay") or 0), "truncation": float(seed.get("truncation") or TRUNCATION)}
            value = seed["fitness"] if seed.get("train_fitness") is None else seed["train_fitness"]
            out.append(TrialSpec(seed["expression"], settings, params={"parents": [], "generation": 0},
                                 generation=0, state="COMPLETE", alpha_id=seed["alpha_id"], value=float(value),
                                 free=True, message=FREE_MESSAGE,
                                 figures={k: seed.get(k) for k in ("sharpe", "fitness", "turnover", "train_sharpe",
                                                                   "train_fitness", "test_sharpe")}))
        return out

    async def _spent(self, task: LabTask) -> list[float | None]:
        async with db.session() as s:
            rows = (await s.execute(select(LabTrial.number, LabTrial.state, LabTrial.value, LabTrial.implausible).where(
                LabTrial.task_id == task.id, LabTrial.generation > 0, LabTrial.free.is_(False),
                LabTrial.state.in_(["COMPLETE", "FAIL"])))).all()
        # an implausible child failed too: counted as found, a straight line would keep a stalled search breeding
        return [float(v) if st == "COMPLETE" and v is not None and v > FAILURE and not imp else None
                for _, st, v, imp in sorted(rows)]

    async def on_scored(self, task: LabTask, trials: list[LabTrial]) -> None:
        self._stalled[task.id] = stalled(await self._spent(task))

    def done_early(self, task: LabTask, ctx: RefillContext) -> str | None:
        return STALLED if self._stalled.get(task.id) else None

    def nothing_left(self, task: LabTask) -> str:
        return self._nothing.pop(task.id, NO_CHILD)

    def forget(self, task_id: int) -> None:
        for memo in (self._stalled, self._fetched, self._nothing):
            memo.pop(task_id, None)

    async def _parents(self, task: LabTask, ranked: list[LabTrial], limit: int) -> list[LabTrial]:
        """The ranked trials kept apart by train-year daily PnL. Missing PnL downloads in the background."""
        from ..engine import series
        ids = [t.alpha_id for t in ranked if t.alpha_id]
        profiles = await train_profiles(ids)
        missing = [a for a in await series.missing(ids, turnover=False) if a not in self._fetched.setdefault(task.id,
                                                                                                           set())]
        if missing:
            self._fetched[task.id].update(missing)
            try:
                from ..jobs.queue import jobs
                await jobs.submit("fetch_series", f"Daily PnL for Evolution task {task.name or task.id}",
                                  {"alpha_ids": missing, "turnover": False}, priority=7)
            except Exception:
                log.warning("could not queue the PnL download for task %s", task.id, exc_info=True)
        kept = set(await asyncio.to_thread(independent, ids, dict(profiles), limit))
        return [t for t in ranked if t.alpha_id in kept]

    async def refill(self, task: LabTask, want: int, ctx: RefillContext) -> list[TrialSpec]:
        params = task.params or {}
        run = params["run"]
        if stalled(await self._spent(task)):
            self._stalled[task.id] = True
            return []
        market = await market_for(run["region"], run["delay"], run["universe"],
                                  list(params.get("neutralizations") or []))
        if market is None:
            self._nothing[task.id] = f"{run['region']} delay {run['delay']} {run['universe']} is no longer downloaded."
            return []
        population = int(params.get("population") or 100)
        trials = [t for t in ctx.scored if t.value is not None]
        superseded = {int((t.params or {}).get("fullOf")) for t in trials if (t.params or {}).get("fullOf") is not None}
        ranked = [t for t in trials if t.value > FAILURE and not t.implausible and t.number not in superseded]
        ranked.sort(key=lambda t: (-(t.value or FAILURE), t.number))
        ranked = ranked[:population]
        chosen = await self._parents(task, ranked, population // 2)
        parents = [p for t in chosen if (p := Parent.of(t.alpha_id, t.expression, t.settings or {}))]
        if not parents:
            return []
        async with db.session() as s:
            stored = (await s.execute(select(LabTrial.expression, LabTrial.settings, LabTrial.generation).where(
                LabTrial.task_id == task.id))).all()
        bred = sum(1 for _, _, g in stored if (g or 0) > 0)
        generation = 1 + bred // population
        seen = {ga_key(e, st) for e, st, _ in stored}
        rate = float(params.get("mutation_rate") or MUTATION_RATE)
        shapes = {p.tree: Shape.of(p.tree, market) for p in parents}
        rng = random.Random()
        out: list[TrialSpec] = []
        for _ in range(want * 10):
            if len(out) >= want:
                break
            a, b = rng.choice(parents), rng.choice(parents)
            made = child(rng, a, b, market, rate, shapes, keep_decay=bool(params.get("keep_decay")))
            if made is None:
                continue
            expression, settings = render(made[0]), settings_for(made[1], run)
            key = ga_key(expression, settings)
            if key in seen:
                continue
            seen.add(key)
            out.append(TrialSpec(expression, settings, params={"parents": [a.alpha_id, b.alpha_id],
                                                               "generation": generation}, generation=generation))
        return out


__all__ = ["EvolutionLab", "IMPLAUSIBLE_SHARPE", "Market", "Parent", "Shape", "auto_seeds", "build_market", "child",
           "mutate", "plan", "population_for", "seed_score", "stalled"]
