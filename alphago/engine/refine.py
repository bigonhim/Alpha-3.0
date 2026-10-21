"""Repair loop and certification: turn near misses into submittable alphas.

Shared by research tasks (engine/research.py) and the Autopilot's AI batches (engine/autopilot.py).

refine(seeds, ctx)
    Beam search over targeted fixes. For each near miss it diagnoses which BRAIN checks fail and by how much (value
    against BRAIN's own limit), picks fixes aimed at those checks — ordered by how often each fix has actually worked
    on that check before (Thompson sampling over the recorded outcomes, with today's order as the prior) — simulates
    them together, and keeps expanding the children that closed the gap they aimed at, stacking fixes, for up to
    `research.repair_depth` rounds.
certify(alpha_ids, ctx)
    BRAIN's submission check on every strong alpha (most different first), then the robustness test for those that
    pass; alphas that fail only on correlation get one decorrelation round.

Submission is never automatic.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import math
import random
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import func, select, update

from ..brain import brain
from ..brain.simulations import normalize_settings, settings_hash
from ..config import settings as app_settings
from ..db import db
from ..db.models import Alpha, Candidate, DataField, OptimizationAttempt
from ..events import bus
from ..fastexpr import (Call, Num, negate, num, parse, replace_identifier, scale_windows, trade_when_wrap, transform,
                        wrap)
from .evaluator import CORRELATION_CHECKS, blocking_failures, check_map, evaluate, limits_for
from .simulator import CandidateSpec, simulator

log = logging.getLogger(__name__)

SHARPE_NOISE = 0.15            # Sharpe gains below this are within noise (paired SE over ~5 IS years)
PRIOR_STRENGTH = 4.0           # pseudo-observations behind each fix's prior success rate
HELPED_BONUS = 1.3             # the fix family that improved this lineage is likely to help again
SHARPE_DROP_OK = 0.10          # a fix may cost this much Sharpe while it fixes something else
CLOSE_ENOUGH = 0.5             # a fix "worked" when it passes the check or closes this share of the gap
UPPER_BOUND = {"HIGH_TURNOVER", "CONCENTRATED_WEIGHT", "SELF_CORRELATION", "PROD_CORRELATION"}
SHARPE_LIKE = {"LOW_SHARPE", "LOW_SUB_UNIVERSE_SHARPE", "IS_LADDER_SHARPE", "LOW_2Y_SHARPE"}
PENDING_RECHECK = timedelta(minutes=30)
SENT = ("queued", "simulating", "done", "error")

S, F, HT, LT = "LOW_SHARPE", "LOW_FITNESS", "HIGH_TURNOVER", "LOW_TURNOVER"
CW, SU, LAD, Y2 = "CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE", "IS_LADDER_SHARPE", "LOW_2Y_SHARPE"
SC, PC = "SELF_CORRELATION", "PROD_CORRELATION"
GROUP_NEUTRALIZATIONS = ["SUBINDUSTRY", "INDUSTRY", "SECTOR", "MARKET"]
RISK_NEUTRALIZATIONS = ["STATISTICAL", "CROWDING", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM", "FAST", "SLOW"]
GATE_EVENTS = ["ts_mean(volume, 10) > ts_mean(volume, 60)", "ts_std_dev(returns, 5) > ts_std_dev(returns, 20)",
               "ts_arg_max(volume, 5) == 0"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _f(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- diagnosis

def check_gap(check: dict[str, Any]) -> float | None:
    """How far a check is from its limit, as a share of the limit (> 0 = failing side). None without numbers."""
    value, limit = _f(check.get("value")), _f(check.get("limit"))
    if value is None or limit is None:
        return None
    name = str(check.get("name"))
    upper = name in UPPER_BOUND or name.startswith("HIGH_")
    raw = (value - limit) if upper else (limit - value)
    return raw / max(abs(limit), 1e-6)


@dataclass
class Diagnosis:
    failing: list[str]                       # blocking failing checks
    gaps: dict[str, float]                   # normalised gap per failing check
    targets: list[str]                       # failing checks, biggest gap first
    values: dict[str, float] = field(default_factory=dict)
    flip: bool = False
    fitness_driver: str | None = None        # "turnover" (cutting turnover can pass Fitness) or "sharpe"
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    returns: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {"failing": self.failing, "gaps": self.gaps, "values": self.values, "flip": self.flip,
                "sharpe": self.sharpe, "fitness": self.fitness, "turnover": self.turnover, "returns": self.returns}


def alpha_checks(a: Alpha) -> list[dict[str, Any]]:
    """BRAIN's latest full check when one ran, else the simulation's IS checks; synthesised from the headline
    numbers when BRAIN returned none (old or synced alphas)."""
    checks = list(a.checks or a.is_checks or [])
    if checks:
        return checks
    sharpe_lim, fitness_lim = limits_for(None, a.delay)
    out = []
    if a.sharpe is not None:
        out.append({"name": S, "result": "PASS" if a.sharpe >= sharpe_lim else "FAIL", "value": a.sharpe,
                    "limit": sharpe_lim})
    if a.fitness is not None:
        out.append({"name": F, "result": "PASS" if a.fitness >= fitness_lim else "FAIL", "value": a.fitness,
                    "limit": fitness_lim})
    if a.turnover is not None:
        out.append({"name": HT, "result": "PASS" if a.turnover <= 0.7 else "FAIL", "value": a.turnover,
                    "limit": 0.7})
    return out


def diagnose(a: Alpha) -> Diagnosis:
    checks = alpha_checks(a)
    ev = evaluate(sharpe=a.sharpe, fitness=a.fitness, turnover=a.turnover, checks=checks, delay=a.delay,
                  region=a.region)
    d = Diagnosis(failing=[], gaps={}, targets=[], sharpe=a.sharpe or 0.0, fitness=a.fitness or 0.0,
                  turnover=a.turnover or 0.0, returns=a.returns or 0.0)
    for c in checks:
        v = _f(c.get("value"))
        if v is not None and c.get("name"):
            d.values[str(c["name"])] = v
    if ev.flip_sign:
        d.flip = True
        d.failing = blocking_failures(checks)
        d.targets = [S]
        d.gaps = {S: 1.0}
        return d
    d.failing = blocking_failures(checks)
    cm = check_map(checks)
    for name in d.failing:
        gap = check_gap(cm.get(name, {}))
        d.gaps[name] = gap if gap is not None and gap > 0 else 0.25
    d.targets = sorted(d.failing, key=lambda n: d.gaps[n], reverse=True)
    if F in d.failing:
        limit = _f(cm.get(F, {}).get("limit")) or ev.fitness_limit
        potential = abs(d.sharpe) * math.sqrt(abs(d.returns) / 0.125) if d.returns else 0.0
        d.fitness_driver = "turnover" if d.turnover > 0.15 and potential >= limit * 1.1 else "sharpe"
    return d


# ---------------------------------------------------------------- fix library

@dataclass(frozen=True)
class Fix:
    key: str
    family: str
    targets: frozenset[str]
    prior: float          # starting guess of the success rate (today's order) until outcomes are recorded


ALL = frozenset({"*"})
FIXES: list[Fix] = [
    Fix("sign_flip", "sign", frozenset({S, F}), 0.9),
    Fix("decay_up", "turnover", frozenset({HT, F}), 0.5),
    Fix("decay_linear", "turnover", frozenset({HT, F}), 0.35),
    Fix("hump", "turnover", frozenset({HT, F}), 0.3),
    Fix("decay_down", "turnover", frozenset({LT}), 0.6),
    Fix("event_gate", "event", frozenset({HT, F, S, SC, PC}), 0.2),
    Fix("truncation", "weight", frozenset({CW}), 0.5),
    Fix("rank_wrap", "weight", frozenset({CW, SU}), 0.35),
    Fix("winsorize", "weight", frozenset({CW}), 0.3),
    Fix("size_neutral", "group", frozenset({SU}), 0.35),
    Fix("liquidity_neutral", "group", frozenset({SU}), 0.25),
    Fix("neutralize_group", "neutral", frozenset({S, F, LAD, Y2, SU}), 0.3),
    Fix("neutralize_risk", "neutral", frozenset({S, F, LAD, Y2, SC, PC}), 0.25),
    Fix("group_rank", "group", frozenset({S, F, SU}), 0.25),
    Fix("windows_short", "windows", frozenset({S, F, LAD, Y2, SC, PC}), 0.2),
    Fix("windows_long", "windows", frozenset({S, F, HT, SC, PC}), 0.2),
    Fix("backfill_longer", "data", frozenset({S, F}), 0.15),
    Fix("sibling_field", "data", frozenset({SC, PC}), 0.25),
    Fix("llm_refine", "llm", ALL, 0.3),
]
FIX_BY_KEY = {f.key: f for f in FIXES}
GENERIC = {"neutralize_group", "neutralize_risk", "windows_short", "windows_long", "group_rank", "llm_refine"}


def fixes_for(target: str) -> list[Fix]:
    known = [f for f in FIXES if target in f.targets or f.targets is ALL]
    if any(target in f.targets for f in FIXES):
        return known
    return [f for f in FIXES if f.key in GENERIC]         # a check this library does not know: general fixes


@dataclass
class FixEnv:
    region: str
    neutralizations: list[str]
    rng: random.Random
    siblings: dict[str, str] = field(default_factory=dict)    # crowded field -> a less-used field of its dataset


@dataclass
class Variant:
    expr: str
    settings: dict[str, Any]
    fix: str
    target: str
    detail: str

    @property
    def family(self) -> str:
        return FIX_BY_KEY[self.fix].family if self.fix in FIX_BY_KEY else "llm"


def _src(p: Any) -> str:
    return p.to_source() if hasattr(p, "to_source") else str(p)


def _top_call(expr: str) -> str | None:
    try:
        node = parse(expr).expression
    except Exception:
        return None
    return node.name if isinstance(node, Call) else None


def _scale_backfill(expr: str, factor: float, cap: int = 504) -> str | None:
    """Lengthen every ts_backfill lookback (sparse data held longer)."""
    program = parse(expr)
    changed = False

    def rewrite(node):  # noqa: ANN001, ANN202
        nonlocal changed
        if isinstance(node, Call) and node.name == "ts_backfill" and len(node.args) >= 2 and isinstance(
                node.args[1], Num) and node.args[1].value.is_integer():
            new = min(cap, int(node.args[1].value * factor))
            if new != int(node.args[1].value):
                changed = True
                return Call(node.name, (node.args[0], num(new)) + node.args[2:], node.kwargs)
        return node

    out = transform(program, rewrite)
    return _src(out) if changed else None


def generate(fix: Fix, expr: str, st: dict[str, Any], d: Diagnosis, env: FixEnv) -> list[tuple[str, dict, str]]:
    """Concrete variants (expression, settings, detail) of one fix; detail strings keep today's wording."""
    out: list[tuple[str, dict, str]] = []
    decay = int(st.get("decay") or 0)
    t = d.turnover
    try:
        if fix.key == "sign_flip":
            out.append((_src(negate(expr)), st, "sign flip"))
        elif fix.key == "decay_up":
            from .optimizer import retune_decay
            nd = retune_decay(decay, t) if t > 0.3 else decay + max(2, decay)
            nd = min(nd, 60)
            if nd != decay:
                out.append((expr, {**st, "decay": nd}, f"decay {decay}→{nd}"))
        elif fix.key == "decay_down" and decay > 0:
            out.append((expr, {**st, "decay": decay // 2}, f"decay {decay}→{decay // 2}"))
        elif fix.key == "decay_linear":
            n = 10 if t > 0.5 else 5
            out.append((_src(wrap(expr, "ts_decay_linear", n)), st, f"ts_decay_linear {n}"))
        elif fix.key == "hump":
            out.append((_src(wrap(expr, "hump", hump=0.01)), st, "hump 0.01"))
        elif fix.key == "event_gate":
            event = env.rng.choice(GATE_EVENTS)
            out.append((_src(trade_when_wrap(expr, event, "-1")), st, f"trade_when {event}"))
        elif fix.key == "truncation":
            cur = float(st.get("truncation") or 0.08)
            new = 0.05 if cur > 0.05 else 0.02 if cur > 0.02 else None
            if new:
                out.append((expr, {**st, "truncation": new}, f"truncation {new}"))
        elif fix.key == "rank_wrap" and _top_call(expr) != "rank":
            out.append((_src(wrap(expr, "rank")), st, "rank wrap"))
        elif fix.key == "winsorize" and _top_call(expr) != "winsorize":
            out.append((_src(wrap(expr, "winsorize", std=4)), st, "winsorize std 4"))
        elif fix.key == "size_neutral":
            out.append((_src(wrap(expr, "group_neutralize", 'bucket(rank(cap), range="0.1,1,0.1")')), st,
                        "size-bucket neutral"))
        elif fix.key == "liquidity_neutral":
            out.append((_src(wrap(expr, "group_neutralize", 'bucket(rank(adv20), range="0.1,1,0.1")')), st,
                        "liquidity-bucket neutral"))
        elif fix.key in ("neutralize_group", "neutralize_risk"):
            pool = GROUP_NEUTRALIZATIONS if fix.key == "neutralize_group" else RISK_NEUTRALIZATIONS
            current = st.get("neutralization")
            for n in [x for x in pool if x in env.neutralizations and x != current][:2]:
                out.append((expr, {**st, "neutralization": n}, f"neutralization {current}→{n}"))
        elif fix.key == "group_rank":
            group = "industry" if _top_call(expr) == "group_rank" else "subindustry"
            out.append((_src(wrap(expr, "group_rank", group)), st, f"group_rank {group}"))
        elif fix.key == "windows_short":
            out.append((_src(scale_windows(expr, 0.5)), st, "windows ×0.5"))
        elif fix.key == "windows_long":
            out.append((_src(scale_windows(expr, 2.0)), st, "windows ×2"))
        elif fix.key == "backfill_longer":
            new = _scale_backfill(expr, 2.0)
            if new:
                out.append((new, st, "ts_backfill ×2"))
        elif fix.key == "sibling_field":
            for old, new in list(env.siblings.items())[:1]:
                out.append((_src(replace_identifier(expr, old, new)), st, f"field {old}→{new}"))
    except Exception as exc:          # a transform that does not apply to this expression
        log.debug("fix %s failed on %s: %s", fix.key, expr, exc)
    norm = normalize_settings(st)
    return [(e, normalize_settings(s), detail) for e, s, detail in out
            if e and not (e == expr and normalize_settings(s) == norm)]


# ---------------------------------------------------------------- learned fix order

@dataclass
class FixStats:
    by_region: dict[tuple[str, str, str], tuple[int, int]] = field(default_factory=dict)
    overall: dict[tuple[str, str], tuple[int, int]] = field(default_factory=dict)

    def get(self, fix: str, target: str, region: str | None) -> tuple[int, int]:
        regional = self.by_region.get((fix, target, region or ""))
        if regional and sum(regional) >= 8:
            return regional
        return self.overall.get((fix, target), (0, 0))


async def load_fix_stats() -> FixStats:
    """Successes and failures per (fix, target check), overall and per region, from recorded attempts."""
    async with db.session() as s:
        rows = (await s.execute(select(OptimizationAttempt.fix, OptimizationAttempt.target_check,
                                       OptimizationAttempt.region, OptimizationAttempt.outcome, func.count())
                                .where(OptimizationAttempt.outcome.in_(["success", "failure"]),
                                       OptimizationAttempt.fix.is_not(None))
                                .group_by(OptimizationAttempt.fix, OptimizationAttempt.target_check,
                                          OptimizationAttempt.region, OptimizationAttempt.outcome))).all()
    st = FixStats()
    for fix, target, region, outcome, n in rows:
        for table, key in ((st.by_region, (fix, target, region or "")), (st.overall, (fix, target))):
            succ, fail = table.get(key, (0, 0))
            table[key] = (succ + n, fail) if outcome == "success" else (succ, fail + n)
    return st


def rank_fixes(d: Diagnosis, stats: FixStats, region: str | None, rng: random.Random | None, *,
               used_families: frozenset[str] = frozenset(), helped_family: str | None = None,
               allow_llm: bool = True) -> list[tuple[Fix, str]]:
    """(fix, target) pairs, best first. rng=None ranks by the prior/observed mean (deterministic); otherwise each
    pair is a Thompson draw from Beta(prior + successes, prior + failures)."""
    scored: list[tuple[float, Fix, str]] = []
    for target in d.targets:
        weight = 1.0 + min(d.gaps.get(target, 0.25), 2.0)
        for fx in fixes_for(target):
            if fx.key == "sign_flip" and not d.flip:
                continue
            if d.flip and fx.key != "sign_flip":
                continue
            if fx.key == "llm_refine" and not allow_llm:
                continue
            if fx.family in used_families and fx.family != helped_family:
                continue
            succ, fail = stats.get(fx.key, target, region)
            a0, b0 = PRIOR_STRENGTH * fx.prior + succ, PRIOR_STRENGTH * (1 - fx.prior) + fail
            draw = rng.betavariate(a0, b0) if rng else a0 / (a0 + b0)
            if target == F and fx.family == "turnover" and d.fitness_driver:
                draw *= 1.5 if d.fitness_driver == "turnover" else 0.6
            if helped_family and fx.family == helped_family:
                draw *= HELPED_BONUS
            scored.append((draw * weight, fx, target))
    scored.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    out = []
    for _, fx, target in scored:
        if fx.key not in seen:
            seen.add(fx.key)
            out.append((fx, target))
    return out


def plan_static(a: Alpha, max_variants: int = 10) -> list[Variant]:
    """Deterministic plan without the database or the AI (prior order) — used by `optimizer.plan_variants`."""
    d = diagnose(a)
    st = normalize_settings(a.settings or {})
    env = FixEnv(st["region"], brain.options.neutralizations(st["region"]), random.Random(0))
    out: list[Variant] = []
    for fx, target in rank_fixes(d, FixStats(), st["region"], None, allow_llm=False):
        for e, s, detail in generate(fx, a.code or "", st, d, env):
            out.append(Variant(e, s, fx.key, target, detail))
    return _dedupe(out)[:max_variants]


def _dedupe(variants: list[Variant]) -> list[Variant]:
    seen, out = set(), []
    for v in variants:
        key = (v.expr, settings_hash(v.settings))
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


# ---------------------------------------------------------------- outcomes

def judge(snapshot: dict[str, Any], target: str, child: Alpha | None, cand: Candidate) -> str | None:
    """success | failure | neutral for one repair attempt, or None while its result is not known yet."""
    if cand.status in ("cancelled", "invalid", "rejected_by_user"):
        return "neutral"
    if child is None:
        if cand.status == "error":
            return "failure" if cand.sim_id else "neutral"     # no sim id: the whole pack failed to start
        return None
    if child.sharpe is None:
        return "failure"
    checks = child.is_checks or child.checks or []
    child_fail = set(blocking_failures(checks))
    parent_fail = set(snapshot.get("failing") or [])
    if target in CORRELATION_CHECKS:
        if child_fail - CORRELATION_CHECKS:
            return "failure"                  # decorrelated, but no longer passes the IS tests
        c = check_map(child.checks).get(target) if child.checked_at else None
        result = str((c or {}).get("result", "")).upper()
        return "success" if result == "PASS" else "failure" if result in ("FAIL", "ERROR") else None
    if (child_fail - parent_fail) - CORRELATION_CHECKS:
        return "failure"                      # fixed one thing, broke another
    p_sharpe = _f(snapshot.get("sharpe")) or 0.0
    if S in child_fail and child.sharpe < p_sharpe - SHARPE_DROP_OK and not snapshot.get("flip"):
        return "failure"
    c = check_map(checks).get(target)
    if c is None:
        return "success" if target not in child_fail else "failure"
    if str(c.get("result", "")).upper() not in ("FAIL", "ERROR"):
        return "success"
    parent_gap = (snapshot.get("gaps") or {}).get(target)
    child_gap = check_gap(c)
    if parent_gap and parent_gap > 0 and child_gap is not None and (parent_gap - child_gap) / parent_gap >= CLOSE_ENOUGH:
        if target in SHARPE_LIKE:
            before, after = (snapshot.get("values") or {}).get(target), _f(c.get("value"))
            if before is not None and after is not None and after - before < SHARPE_NOISE:
                return "failure"              # inside the noise band: not evidence the fix worked
        return "success"
    return "failure"


async def resolve_attempts(*, attempt_ids: list[int] | None = None, candidate_ids: list[int] | None = None,
                           child_alpha_id: str | None = None) -> int:
    """Settle unresolved attempts whose child result is now known. Returns how many were settled."""
    async with db.session() as s:
        stmt = select(OptimizationAttempt).where(OptimizationAttempt.outcome.is_(None),
                                                 OptimizationAttempt.fix.is_not(None))
        if attempt_ids is not None:
            stmt = stmt.where(OptimizationAttempt.id.in_(attempt_ids or [-1]))
        if candidate_ids is not None:
            stmt = stmt.where(OptimizationAttempt.candidate_id.in_(candidate_ids or [-1]))
        if child_alpha_id is not None:
            stmt = stmt.where(OptimizationAttempt.child_alpha_id == child_alpha_id)
        attempts = list(await s.scalars(stmt))
        if not attempts:
            return 0
        cands = {c.id: c for c in await s.scalars(select(Candidate).where(
            Candidate.id.in_([x.candidate_id for x in attempts if x.candidate_id])))}
        alpha_ids = {c.alpha_id for c in cands.values() if c.alpha_id}
        alphas = {a.id: a for a in await s.scalars(select(Alpha).where(Alpha.id.in_(alpha_ids)))} if alpha_ids else {}
    settled = 0
    async with db.write() as s:
        for att in attempts:
            cand = cands.get(att.candidate_id)
            if cand is None:
                continue
            child = alphas.get(cand.alpha_id) if cand.alpha_id else None
            outcome = judge(att.parent_snapshot or {}, att.target_check or "", child, cand)
            values: dict[str, Any] = {"child_alpha_id": cand.alpha_id}
            if child is not None:
                values["child_score"] = child.score
            if outcome is not None:
                values.update(outcome=outcome, resolved_at=utcnow())
                settled += 1
            await s.execute(update(OptimizationAttempt).where(OptimizationAttempt.id == att.id).values(**values))
    return settled


async def _on_simulated(cand: Candidate, alpha: Alpha | None) -> None:
    if cand.origin == "optimize":
        await resolve_attempts(candidate_ids=[cand.id])


simulator.add_listener(_on_simulated)


async def tried_lines(alpha_id: str, limit: int = 12) -> list[str]:
    """Fixes already tried on an alpha (and on its lineage root), with what happened — for the AI refine prompt."""
    async with db.session() as s:
        root = await s.scalar(select(OptimizationAttempt.root_alpha_id).where(
            OptimizationAttempt.parent_alpha_id == alpha_id, OptimizationAttempt.root_alpha_id.is_not(None)).limit(1))
        stmt = select(OptimizationAttempt).where(OptimizationAttempt.outcome.is_not(None))
        stmt = stmt.where((OptimizationAttempt.parent_alpha_id == alpha_id)
                          | (OptimizationAttempt.root_alpha_id == (root or alpha_id)))
        rows = list(await s.scalars(stmt.order_by(OptimizationAttempt.id.desc()).limit(limit)))
        kids = {a.id: a for a in await s.scalars(select(Alpha).where(
            Alpha.id.in_([r.child_alpha_id for r in rows if r.child_alpha_id])))} if rows else {}
    out = []
    for r in rows:
        before = r.parent_snapshot or {}
        kid = kids.get(r.child_alpha_id or "")
        after = (f"Sharpe {kid.sharpe:.2f}, Fitness {kid.fitness or 0:.2f}, turnover {(kid.turnover or 0):.0%}"
                 if kid and kid.sharpe is not None else "no result")
        out.append(f"- {r.kind} (aimed at {r.target_check}): Sharpe {before.get('sharpe', 0):.2f}, "
                   f"Fitness {before.get('fitness', 0):.2f}, turnover {before.get('turnover', 0):.0%} → {after} "
                   f"[{r.outcome}]")
    return out


# ---------------------------------------------------------------- the beam

@dataclass
class RefineContext:
    task_id: int | None = None
    batch_id: int | None = None
    use_llm: bool = True
    stop: asyncio.Event | None = None
    allowance: Callable[[], Awaitable[int]] | None = None   # simulations the caller may still spend
    max_depth: int = 3
    root_fanout: int = 8
    node_fanout: int = 4
    beam: int = 2
    wait_timeout: float | None = 6 * 3600
    collide: str | None = None          # decorrelation: the submitted alpha's expression it collides with
    robustness: bool | None = None      # run the robustness test after a pass (None = settings.research)
    seed: int | None = None

    def stopped(self) -> bool:
        return bool(self.stop is not None and self.stop.is_set())

    async def budget(self) -> int:
        n = 10 ** 9
        if self.allowance is not None:
            n = min(n, int(await self.allowance()))
        n = min(n, brain.client.quota.effective_remaining - app_settings.brain.quota_reserve)
        return max(0, n)


@dataclass
class Node:
    alpha: Alpha
    depth: int
    root: str
    families: frozenset[str] = frozenset()
    helped: str | None = None


@dataclass
class RefineReport:
    simulated: int = 0
    rounds: int = 0
    attempts: int = 0
    strong: list[str] = field(default_factory=list)
    expanded: list[str] = field(default_factory=list)
    stopped: str | None = None


async def _siblings(a: Alpha, st: dict[str, Any]) -> dict[str, str]:
    """For decorrelation: the expression's most crowded data field -> the closest-named, less-used field of the same
    dataset (MATRIX, coverage ≥ 50%)."""
    ds = str((a.dataset_ids or [""])[0] or "").split(",")[0]
    used = set(a.fields or [])
    if not ds or not used:
        return {}
    async with db.session() as s:
        rows = list(await s.scalars(select(DataField).where(
            DataField.dataset_id == ds, DataField.region == st["region"], DataField.delay == st["delay"],
            DataField.universe == st["universe"], DataField.type == "MATRIX")))
    by_id = {r.id: r for r in rows}
    mine = [by_id[f] for f in used if f in by_id]
    if not mine:
        return {}
    crowded = max(mine, key=lambda r: r.alpha_count or 0)
    pool = [r for r in rows if r.id not in used and (r.coverage or 0) >= 0.5
            and (r.alpha_count or 0) < (crowded.alpha_count or 0)]
    if not pool:
        return {}
    best = max(pool, key=lambda r: difflib.SequenceMatcher(None, r.id, crowded.id).ratio())
    return {crowded.id: best.id}


async def plan_node(node: Node, d: Diagnosis, fanout: int, stats: FixStats, ctx: RefineContext,
                    rng: random.Random) -> list[Variant]:
    a = node.alpha
    st = normalize_settings(a.settings or {})
    env = FixEnv(st["region"], brain.options.neutralizations(st["region"]), rng)
    if set(d.targets) & CORRELATION_CHECKS:
        env.siblings = await _siblings(a, st)
    from ..llm import llm
    allow_llm = ctx.use_llm and llm.configure()
    variants: list[Variant] = []
    llm_slots, llm_target = 0, None
    for fx, target in rank_fixes(d, stats, st["region"], rng, used_families=node.families,
                                 helped_family=node.helped, allow_llm=allow_llm):
        if len(variants) + llm_slots >= fanout:
            break
        if fx.key == "llm_refine":
            llm_slots, llm_target = min(3, fanout - len(variants)), target
            continue
        for e, s, detail in generate(fx, a.code or "", st, d, env):
            if len(variants) + llm_slots < fanout:
                variants.append(Variant(e, s, fx.key, target, detail))
    if llm_slots:
        from .optimizer import llm_refine
        try:
            specs = await llm_refine(a, k=llm_slots, tried=await tried_lines(a.id), collide=ctx.collide)
            variants += [Variant(sp.expression, normalize_settings(sp.settings), "llm_refine", llm_target or S,
                                 "llm refine") for sp in specs]
        except Exception as exc:
            log.warning("AI refine failed for %s: %s", a.id, exc)
    return _dedupe(variants)


def _round_robin(planned: list[tuple[Node, Variant]], n: int) -> list[tuple[Node, Variant]]:
    """Keep at most n, taking one variant per node in turn so every node gets some of a tight budget."""
    queues: dict[str, list[tuple[Node, Variant]]] = {}
    for item in planned:
        queues.setdefault(item[0].alpha.id, []).append(item)
    out: list[tuple[Node, Variant]] = []
    while len(out) < n and any(queues.values()):
        for key in list(queues):
            if queues[key] and len(out) < n:
                out.append(queues[key].pop(0))
    return out


async def _load_alphas(ids: list[str]) -> dict[str, Alpha]:
    if not ids:
        return {}
    async with db.session() as s:
        return {a.id: a for a in await s.scalars(select(Alpha).where(Alpha.id.in_(ids)))}


async def _mark_expanded(ids: list[str]) -> None:
    if ids:
        async with db.write() as s:
            await s.execute(update(Alpha).where(Alpha.id.in_(ids)).values(expanded_at=utcnow()))


async def _submit_round(planned: list[tuple[Node, Variant]], ctx: RefineContext,
                        depth: int) -> tuple[list[int], list[int], list[int]]:
    """Store, record and queue one round. Returns (created candidate ids, reused candidate ids, attempt ids)."""
    specs, keyed = [], {}
    for node, v in planned:
        spec = CandidateSpec(
            expression=v.expr, settings=v.settings, origin="optimize", parent_alpha_id=node.alpha.id,
            dataset_id=(node.alpha.dataset_ids or [None])[0], task_id=ctx.task_id,
            hypothesis=f"Repair of {node.alpha.id}: {v.detail} (aimed at {v.target})",
            rationale={"kind": v.detail, "fix": v.fix, "target": v.target, "depth": depth, "root": node.root})
        specs.append(spec)
        keyed[(v.expr.strip(), settings_hash(v.settings))] = (node, v)
    res = await simulator.prepare(specs, batch_id=ctx.batch_id, task_id=ctx.task_id, initial_status="approved")
    attempt_ids: list[int] = []
    async with db.write() as s:
        for cand in res.created + res.reused:
            node, v = keyed.get((cand.expression.strip(), cand.settings_hash), (None, None))
            if node is None:
                continue
            if cand.status == "reused":
                dup = await s.scalar(select(OptimizationAttempt.id).where(
                    OptimizationAttempt.parent_alpha_id == node.alpha.id,
                    OptimizationAttempt.child_alpha_id == cand.alpha_id))
                if dup:
                    continue
            d = diagnose(node.alpha)
            att = OptimizationAttempt(
                parent_alpha_id=node.alpha.id, candidate_id=cand.id, kind=v.detail[:48], detail=v.expr[:500],
                fix=v.fix, target_check=v.target[:48], depth=depth, root_alpha_id=node.root,
                region=node.alpha.region, parent_snapshot=d.snapshot(),
                child_alpha_id=cand.alpha_id if cand.status == "reused" else None)
            s.add(att)
            await s.flush()
            attempt_ids.append(att.id)
    await simulator.enqueue([c.id for c in res.created])
    return [c.id for c in res.created], [c.id for c in res.reused], attempt_ids


async def _wait(ids: list[int], ctx: RefineContext) -> None:
    if not ids:
        return
    try:
        await simulator.wait_for(ids, timeout=ctx.wait_timeout)
    except asyncio.TimeoutError:
        log.warning("repair round: %d simulations still unfinished after %.0f s", len(ids), ctx.wait_timeout or 0)


async def expand_once(a: Alpha, ctx: RefineContext, *, depth: int = 1) -> list[int]:
    """One round of fixes for one alpha without waiting (the alpha page's 'Optimise'). Returns candidate ids."""
    node = Node(alpha=a, depth=0, root=a.id)
    rng = random.Random(ctx.seed)
    planned = [(node, v) for v in await plan_node(node, diagnose(a), ctx.root_fanout, await load_fix_stats(), ctx,
                                                  rng)]
    await _mark_expanded([a.id])
    created, reused, attempts = await _submit_round(planned, ctx, depth)
    await resolve_attempts(attempt_ids=attempts)
    return created + reused


async def refine(seeds: list[Alpha], ctx: RefineContext) -> RefineReport:
    """Repair near misses: diagnose → targeted fixes → simulate → expand the children that closed their gap."""
    report = RefineReport()
    rng = random.Random(ctx.seed)
    frontier = [Node(alpha=a, depth=0, root=a.id) for a in seeds if a.code and a.expanded_at is None]
    for depth in range(1, max(0, ctx.max_depth) + 1):
        if not frontier:
            break
        if ctx.stopped():
            report.stopped = "stopped"
            break
        allowance = await ctx.budget()
        if allowance <= 0:
            report.stopped = "budget"
            break
        stats = await load_fix_stats()          # reloaded each round: this run's outcomes count at once
        planned: list[tuple[Node, Variant]] = []
        for node in frontier:
            d = diagnose(node.alpha)
            if not d.targets:
                continue
            fanout = ctx.root_fanout if node.depth == 0 else ctx.node_fanout
            planned += [(node, v) for v in await plan_node(node, d, fanout, stats, ctx, rng)]
        await _mark_expanded([n.alpha.id for n in frontier])
        report.expanded += [n.alpha.id for n in frontier]
        planned = _round_robin(planned, allowance)
        if not planned:
            break
        created, reused, attempt_ids = await _submit_round(planned, ctx, depth)
        report.rounds += 1
        report.simulated += len(created)
        report.attempts += len(attempt_ids)
        bus.publish("optimize", f"Repair round {depth}: {len(created)} variants of {len(frontier)} near miss(es) "
                                f"queued ({len(reused)} already simulated).")
        await _wait(created, ctx)
        await resolve_attempts(attempt_ids=attempt_ids)
        async with db.session() as s:
            atts = list(await s.scalars(select(OptimizationAttempt).where(
                OptimizationAttempt.id.in_(attempt_ids or [-1]))))
        kids = await _load_alphas([x.child_alpha_id for x in atts if x.child_alpha_id])
        by_parent: dict[str, list[OptimizationAttempt]] = {}
        for att in atts:
            by_parent.setdefault(att.parent_alpha_id, []).append(att)
        next_frontier: list[Node] = []
        for node in frontier:
            winners = []
            for att in by_parent.get(node.alpha.id, []):
                kid = kids.get(att.child_alpha_id or "")
                if kid is None:
                    continue
                if kid.verdict in ("strong", "ready") and kid.id not in report.strong:
                    report.strong.append(kid.id)
                elif att.outcome == "success" and kid.expanded_at is None and kid.verdict in ("promising", "weak"):
                    winners.append((kid.score or 0.0, kid, att))
            winners.sort(key=lambda w: w[0], reverse=True)
            for _, kid, att in winners[:ctx.beam]:
                fam = FIX_BY_KEY[att.fix].family if att.fix in FIX_BY_KEY else "llm"
                next_frontier.append(Node(alpha=kid, depth=depth, root=node.root, families=node.families | {fam},
                                          helped=fam))
        frontier = next_frontier
    if report.strong:
        bus.publish("optimize", f"Repair loop: {len(report.strong)} near miss(es) turned strong "
                                f"({report.simulated} simulations, {report.rounds} rounds).", "success")
    return report


# ---------------------------------------------------------------- certification

@dataclass
class CertifyReport:
    checked: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)
    fragile: list[str] = field(default_factory=list)
    corr_failed: list[str] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)


def _needs_check(a: Alpha, now: datetime) -> bool:
    if a.verdict != "strong" or (a.type or "REGULAR") != "REGULAR":
        return False
    if a.check_state in (None, "", "none", "stale"):
        return True
    return a.check_state == "pending" and (a.checked_at is None or now - a.checked_at > PENDING_RECHECK)


async def order_for_check(alphas: list[Alpha]) -> list[Alpha]:
    """Most promising and most different first: lowest estimated self-correlation, best score, and alphas whose daily
    PnL is > 0.9 correlated with one already queued go to the back (they would pass or fail together)."""
    from . import pnl
    ranked = sorted(alphas, key=lambda a: (a.est_self_corr if a.est_self_corr is not None else 0.5,
                                           -(a.score or 0.0) * (a.os_retention_pred if a.os_retention_pred
                                                                is not None else 1.0)))
    rows = await pnl.stored([a.id for a in ranked])
    df = pnl.frame(rows) if len(rows) >= 2 else None
    first, later = [], []
    for a in ranked:
        twin = df is not None and a.id in df.columns and any(
            b.id in df.columns and _corr(df, a.id, b.id) > 0.9 for b in first)
        (later if twin else first).append(a)
    return first + later


async def _predict_retention(alpha_ids: list[str]) -> None:
    """Predicted OS/IS Sharpe from your submitted alphas' history (only once enough of them have OS numbers)."""
    from . import pnl
    from .honesty import gap_model, predict_retention
    try:
        model = await gap_model()
        if not model.get("ready"):
            return
        rows = await pnl.stored(alpha_ids)
        async with db.write() as s:
            for a in await s.scalars(select(Alpha).where(Alpha.id.in_(alpha_ids))):
                a.os_retention_pred = predict_retention(model, a, rows.get(a.id))
    except Exception:
        log.exception("OS/IS prediction failed")


def _corr(df, a: str, b: str) -> float:  # noqa: ANN001
    pair = df[[a, b]].dropna()
    if len(pair) < 60:
        return 0.0
    c = pair[a].corr(pair[b])
    return float(c) if c == c else 0.0


async def _colliding_expression(a: Alpha) -> str | None:
    """Expression of the submitted alpha this one is most correlated with (BRAIN's correlation list, else our
    estimate)."""
    from . import actions
    other = None
    try:
        res = await actions.correlations(a.id, "self")
        best = None
        for rec in res.get("records") or []:
            corr = next((v for k, v in rec.items() if "corr" in k.lower() and isinstance(v, (int, float))), None)
            aid = rec.get("id") or rec.get("alphaId") or rec.get("alpha_id")
            if aid and corr is not None and (best is None or corr > best[0]):
                best = (corr, str(aid))
        other = best[1] if best else None
    except Exception as exc:
        log.info("correlation list for %s unavailable: %s", a.id, getattr(exc, "message", exc))
    other = other or a.est_corr_with
    if not other:
        return None
    async with db.session() as s:
        row = await s.get(Alpha, other)
    return row.code if row and row.code else None


async def certify(alpha_ids: list[str], ctx: RefineContext, *, cap: int | None = None,
                  corr_repair: bool = True) -> CertifyReport:
    """BRAIN /check on the task's strong alphas (no simulation quota), robustness for those that pass, one
    decorrelation round for those that fail only on correlation."""
    from . import actions
    from .similarity import estimate_self_corr
    report = CertifyReport()
    cap = app_settings.research.check_cap if cap is None else cap
    now = utcnow()
    alphas = [a for a in (await _load_alphas(list(dict.fromkeys(alpha_ids)))).values() if _needs_check(a, now)]
    if not alphas or cap <= 0:
        return report
    sem = asyncio.Semaphore(3)

    async def estimate(a: Alpha) -> None:
        async with sem:
            await estimate_self_corr(a.id)

    await asyncio.gather(*(estimate(a) for a in alphas))
    await _predict_retention([a.id for a in alphas])
    alphas = await order_for_check(list((await _load_alphas([a.id for a in alphas])).values()))
    check_sem = asyncio.Semaphore(2)

    async def check(a: Alpha) -> tuple[str, str]:
        async with check_sem:
            if ctx.stopped():
                return a.id, "skipped"
            try:
                state, _ = await actions.run_check(a.id)
                return a.id, state
            except Exception as exc:
                bus.publish("check", f"Check failed for {a.id}: {getattr(exc, 'message', exc)}", "warning")
                return a.id, "error"

    results = await asyncio.gather(*(check(a) for a in alphas[:cap]))
    report.checked = [aid for aid, st in results if st not in ("skipped", "error")]
    after = await _load_alphas(report.checked)
    for aid in report.checked:
        a = after.get(aid)
        if a is None:
            continue
        if a.check_state == "pass":
            report.passed.append(aid)
        elif set(blocking_failures(a.checks)) and set(blocking_failures(a.checks)) <= CORRELATION_CHECKS:
            report.corr_failed.append(aid)
    for aid in report.passed:
        if ctx.stopped():
            break
        try:
            from .honesty import assess
            await assess(aid)               # walk-forward, search-adjusted Sharpe, predicted OS/IS (no quota)
        except Exception:
            log.exception("honesty figures failed for %s", aid)
        robustness = app_settings.research.robustness_gate if ctx.robustness is None else ctx.robustness
        if robustness and await ctx.budget() >= 3:
            res = await check_robustness(aid, ctx.batch_id, task_id=ctx.task_id)
            if res.get("state") == "fragile":
                report.fragile.append(aid)
                best = res.get("best_neighbour")
                if best:                       # the neighbour that held up best is a free second candidate
                    await certify([best], replace(ctx), cap=1, corr_repair=False)
    if corr_repair:
        for aid in report.corr_failed[:3]:
            if ctx.stopped() or await ctx.budget() <= 0:
                break
            a = after[aid]
            sub = replace(ctx, max_depth=1, collide=await _colliding_expression(a))
            rep = await refine([a], sub)
            if rep.strong:
                more = await certify(rep.strong, sub, cap=min(3, cap), corr_repair=False)
                report.repaired += more.passed
    return report


async def check_robustness(alpha_id: str, bid: int | None = None, task_id: int | None = None) -> dict[str, Any]:
    """Sensitivity test: neighbouring settings/windows should keep ≥ 80% of the Sharpe.

    It labels the alpha (robust / fragile / unknown) and never changes its verdict: an alpha that passed BRAIN's check
    stays "ready" and carries the badge, so you decide. Fewer than 2 finished neighbours = "unknown"."""
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
    if a is None or not a.code or a.sharpe is None:
        return {}
    st = normalize_settings(a.settings or {})
    specs = [
        CandidateSpec(expression=a.code, settings={**st, "decay": int(st["decay"]) + 2}, origin="robustness",
                      parent_alpha_id=a.id, hypothesis="robustness: decay +2", task_id=task_id),
        CandidateSpec(expression=scale_windows(a.code, 0.8).to_source(), settings=st, origin="robustness",
                      parent_alpha_id=a.id, hypothesis="robustness: windows ×0.8", task_id=task_id),
        CandidateSpec(expression=scale_windows(a.code, 1.25).to_source(), settings=st, origin="robustness",
                      parent_alpha_id=a.id, hypothesis="robustness: windows ×1.25", task_id=task_id),
    ]
    result = await simulator.submit(specs, batch_id=bid)
    ids = [c.id for c in result.created + result.reused]
    try:
        await simulator.wait_for(ids, timeout=6 * 3600)
    except asyncio.TimeoutError:
        log.warning("robustness neighbours of %s did not finish", alpha_id)
    async with db.session() as s:
        alpha_ids = [x for x in await s.scalars(select(Candidate.alpha_id).where(Candidate.id.in_(ids))) if x]
        neighbours = [n for n in await s.scalars(select(Alpha).where(Alpha.id.in_(alpha_ids)))
                      if n.sharpe is not None and n.id != alpha_id]
    sharpes = [n.sharpe for n in neighbours]
    ratio = (min(sharpes) / a.sharpe) if sharpes and a.sharpe else None
    state = "unknown" if len(sharpes) < 2 or ratio is None else ("robust" if ratio >= 0.8 else "fragile")
    best = max((n for n in neighbours if n.verdict == "strong"), key=lambda n: n.sharpe, default=None)
    info = {"ratio": round(ratio, 3) if ratio is not None else None, "robust": state == "robust", "state": state,
            "n": len(sharpes), "neighbours": [n.id for n in neighbours],
            "best_neighbour": best.id if best and state == "fragile" else None}
    await db.kv_set(f"robust:{alpha_id}", info)
    async with db.write() as s:
        row = await s.get(Alpha, alpha_id)
        if row is not None:
            row.robust_ratio, row.robust_state = info["ratio"], state
    if state == "robust":
        bus.publish("check", f"{alpha_id} is ready and robust (neighbours keep {ratio:.0%} of Sharpe) — review it "
                             "on the Alphas page.", "success", alpha_id=alpha_id, kind="ready")
    elif state == "fragile":
        bus.publish("check", f"{alpha_id} passed BRAIN's check but is fragile (neighbours keep only {ratio:.0%} of "
                             "Sharpe) — it is ready with a 'fragile' badge; you decide.", "warning")
    else:
        bus.publish("check", f"{alpha_id}: robustness could not be measured ({len(sharpes)} neighbour(s) finished).",
                    "info")
    return info


async def robustness_info(alpha_id: str) -> dict[str, Any] | None:
    """Robustness of an alpha from its columns, else from the older kv record."""
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
    if a is not None and a.robust_state:
        return {"ratio": a.robust_ratio, "state": a.robust_state}
    old = await db.kv_get(f"robust:{alpha_id}")
    if old:
        return {"ratio": old.get("ratio"), "state": old.get("state") or ("robust" if old.get("robust") else "fragile")}
    return None


async def repair_spent(task_id: int) -> int:
    """Repair and robustness simulations a research task has sent."""
    async with db.session() as s:
        return await s.scalar(select(func.count(Candidate.id)).where(
            Candidate.task_id == task_id, Candidate.origin.in_(["optimize", "robustness"]),
            Candidate.status.in_(SENT))) or 0
