"""Targeted repair: measure a near miss, name why each check fails, and work out the moves that can fix it.

Pure functions (no database, no network), so every rule can be tested on numbers. engine/refine.py supplies the data
and runs the moves.

The arithmetic comes from BRAIN's own definition, Fitness = Sharpe * sqrt(|Returns| / max(Turnover, 0.125)):
- turnover below 12.5% earns nothing, so a Fitness failure down there is never a turnover problem;
- above it, what a turnover cut buys depends on how much Sharpe and returns it costs, so decay is chosen along that
  path rather than read from a table.
The decay setting's pull on turnover is measured, not assumed: T(n) ~ T(0) * (n+1)^-0.47 held for simulated rank signals
at every persistence from 0.0 to 0.98. A cut also costs Sharpe, so the decay to try is the one forecast to land closest
to passing every metric, not the one that hits a turnover number. Once an alpha has run at two decays, its own curve
replaces these defaults.

Effects below are starting points measured on the synthetic market in tests/marketsim.py. They only rank moves before
your account has results; your recorded outcomes then take over.
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..fastexpr import Call, compute_metrics, parse, walk

T_FLOOR = 0.125                # BRAIN's Fitness treats any turnover below this as this
DECAY_BETA = 0.47              # T(n) = T(0) * (n+1)^-beta, measured
RETURNS_PER_T = 0.25           # d log(returns) / d log(turnover) along a decay path, measured on tests/marketsim.py
SHARPE_PER_T = 0.27            # d log(Sharpe) / d log(turnover) along a decay path, measured the same way
MAX_DECAY = 60
DECAY_GRID = (1, 2, 3, 4, 6, 8, 10, 14, 20, 30, 45, 60)
HIGH_TURNOVER_GOAL = 0.6       # share of the turnover limit to aim at
TURNOVER_REACH = 0.85          # turnover is the root cause when cutting it alone can reach this share of the limit
GAIN_FLOOR = 0.05              # a move with no modelled effect keeps a small score, so your results can still lift it
SPARSE_COVERAGE = 0.5          # (long + short) / universe below this: the signal is missing for many stocks
GAP_CAP = 2.0
UNIVERSE_SIZE = re.compile(r"(\d+)")

S, F, HT, LT = "LOW_SHARPE", "LOW_FITNESS", "HIGH_TURNOVER", "LOW_TURNOVER"
CW, SU, LAD, Y2 = "CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE", "IS_LADDER_SHARPE", "LOW_2Y_SHARPE"
SC, PC = "SELF_CORRELATION", "PROD_CORRELATION"
CORRELATION = {SC, PC}
TURNOVER_CUTS = {"decay_up", "decay_linear", "hump", "event_gate", "windows_long"}
BOUNDED_OPS = {"rank", "group_rank", "ts_rank", "quantile", "ts_scale", "group_scale", "scale", "sign"}


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def fitness_of(sharpe: float, returns: float, turnover: float) -> float:
    return sharpe * math.sqrt(abs(returns) / max(turnover, T_FLOOR)) if sharpe else 0.0


@dataclass
class Elasticity:
    """How this alpha trades turnover for returns and Sharpe along a decay path."""
    beta: float = DECAY_BETA
    gamma: float = RETURNS_PER_T
    sigma: float = SHARPE_PER_T
    fitted: bool = False

    @property
    def slope(self) -> float:
        """What a turnover cut does to Fitness²: d log(F²) / d log(1/turnover)."""
        return 1.0 - 2.0 * self.sigma - self.gamma


def fit_elasticity(points: list[tuple[int, float, float, float]]) -> Elasticity:
    """From (decay, turnover, Sharpe, returns) of the same expression at different decays, lowest and highest decay."""
    usable = sorted({p[0]: p for p in points if p[1] > 0}.values())
    if len(usable) < 2:
        return Elasticity()
    (n1, t1, s1, r1), (n2, t2, s2, r2) = usable[0], usable[-1]
    if n1 == n2 or t1 == t2:
        return Elasticity()
    dlog_t = math.log(t2 / t1)
    beta = -dlog_t / math.log((n2 + 1) / (n1 + 1))
    gamma = math.log(abs(r2 / r1)) / dlog_t if r1 and r2 else RETURNS_PER_T
    sigma = math.log(abs(s2 / s1)) / dlog_t if s1 and s2 else SHARPE_PER_T
    return Elasticity(min(1.2, max(0.15, beta)), min(1.5, max(-0.2, gamma)), min(1.5, max(-0.5, sigma)), True)


def solve_decay(turnover: float, decay: int, goal: float, beta: float = DECAY_BETA) -> int:
    """Smallest decay expected to bring `turnover` (measured at `decay`) down to `goal`, or up to it."""
    if turnover <= 0 or goal <= 0:
        return decay
    n = (decay + 1) * (turnover / goal) ** (1.0 / beta) - 1
    return min(MAX_DECAY, max(0, math.ceil(n - 1e-9)))


def fitness_cause(sharpe: float, returns: float, turnover: float, fitness_limit: float, sharpe_limit: float,
                  e: Elasticity) -> str:
    """Why Fitness fails: "turnover" (cutting it alone gets most of the way), "returns" (Sharpe is fine but each traded
    dollar earns too little) or "sharpe" (nothing else will do)."""
    now = fitness_of(sharpe, returns, turnover)
    if turnover > T_FLOOR * 1.1 and e.slope > 0.05:
        reach = now * math.sqrt((turnover / T_FLOOR) ** e.slope)
        if reach >= fitness_limit * TURNOVER_REACH:
            return "turnover"
    return "returns" if sharpe >= sharpe_limit else "sharpe"


@dataclass
class YearlyShape:
    shape: str = "unknown"             # steady | recent_decay | one_bad_year | unknown
    last2: float | None = None
    earlier: float | None = None
    worst_year: str | None = None
    worst: float | None = None


def _sharpe(values: list[float]) -> float | None:
    if len(values) < 40:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean / math.sqrt(var) * math.sqrt(252) if var > 0 else None


def yearly_shape(dates: list[str] | None, daily: list[float] | None) -> YearlyShape:
    """When the Sharpe was earned, from daily PnL: fading recently, one bad year, or steady."""
    if not dates or not daily or len(daily) < 300 or len(dates) != len(daily):
        return YearlyShape()
    out = YearlyShape(last2=_sharpe(daily[-504:]), earlier=_sharpe(daily[:-504]))
    years: dict[str, list[float]] = {}
    for d, v in zip(dates, daily):
        years.setdefault(str(d)[:4], []).append(v)
    scored = {y: s for y, s in ((y, _sharpe(v)) for y, v in years.items()) if s is not None}
    if scored:
        out.worst_year, out.worst = min(scored.items(), key=lambda kv: kv[1])
    others = [s for y, s in scored.items() if y != out.worst_year]
    if out.last2 is not None and out.earlier is not None and out.earlier > 0.5 and out.last2 < 0.5 * out.earlier:
        out.shape = "recent_decay"
    elif out.worst is not None and others and out.worst < -0.5 and sum(others) / len(others) > 1.0:
        out.shape = "one_bad_year"
    else:
        out.shape = "steady"
    return out


@dataclass
class ExprFacts:
    top: str | None = None
    bounded: bool = False              # rank-like output: outliers cannot dominate the book
    backfilled: bool = False
    grouped: bool = False
    gated: bool = False
    windows: list[int] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    n_ops: int = 0


@functools.lru_cache(maxsize=8192)
def expr_facts(code: str | None) -> ExprFacts:
    """What an expression is made of. Cached: repairs re-diagnose the same alphas often, and parsing is the slow part."""
    if not code:
        return ExprFacts()
    try:
        program = parse(code)
        metrics = compute_metrics(program)
        names = {n.name for n in walk(program) if isinstance(n, Call)}
    except Exception:
        return ExprFacts()
    node = program.expression
    top = node.name if isinstance(node, Call) else None
    return ExprFacts(top=top, bounded=top in BOUNDED_OPS, backfilled="ts_backfill" in names or "group_backfill" in names,
                     grouped=any(n.startswith("group_") for n in names), gated="trade_when" in names,
                     windows=sorted(metrics.windows), fields=sorted(metrics.non_group_fields),
                     n_ops=metrics.operator_count)


@dataclass
class Profile:
    """Everything measured about one alpha, in the units the moves need."""
    sharpe: float = 0.0
    fitness: float = 0.0
    returns: float = 0.0
    turnover: float = 0.0
    sharpe_limit: float = 1.25
    fitness_limit: float = 1.0
    turnover_high: float = 0.7
    turnover_low: float = 0.01
    decay: int = 0
    neutralization: str = ""
    truncation: float = 0.08
    universe: str = ""
    coverage: float | None = None
    sub_ratio: float | None = None     # sub-universe Sharpe / Sharpe
    fitness_needs: dict[str, float] = field(default_factory=dict)     # turnover / returns / sharpe that would pass
    causes: dict[str, str] = field(default_factory=dict)              # failing check -> root cause
    expr: ExprFacts = field(default_factory=ExprFacts)
    yearly: YearlyShape = field(default_factory=YearlyShape)
    elasticity: Elasticity = field(default_factory=Elasticity)


def coverage_of(long_count: Any, short_count: Any, universe: str | None) -> float | None:
    m = UNIVERSE_SIZE.search(universe or "")
    held = _num(long_count) + _num(short_count)
    return min(1.0, held / int(m.group(1))) if m and held and int(m.group(1)) else None


def check_family(name: str) -> str:
    """What kind of test a check is, from its name, so checks this library has never seen still get a real plan."""
    n = name.upper()
    if n in CORRELATION or "CORRELATION" in n:
        return "correlation"
    if any(k in n for k in ("SUB_UNIVERSE", "ROBUST_UNIVERSE", "INVESTABILITY", "LIQUID")):
        return "universe"
    if "LADDER" in n or "2Y" in n or "RECENT" in n:
        return "recency"
    if "TURNOVER" in n:
        return "turnover"
    if "CONCENTRATED" in n or "WEIGHT" in n:
        return "weight"
    if "FITNESS" in n:
        return "fitness"
    if "SHARPE" in n or "RETURNS" in n:
        return "signal"
    return "unknown"


def cause_of(name: str, p: Profile) -> str:
    family = check_family(name)
    if name == F:
        return fitness_cause(p.sharpe, p.returns, p.turnover, p.fitness_limit, p.sharpe_limit, p.elasticity)
    if family == "signal":
        if p.coverage is not None and p.coverage < SPARSE_COVERAGE:
            return "sparse"
        if not p.expr.bounded:
            return "outliers"
        if p.yearly.shape == "recent_decay":
            return "fading"
        if p.yearly.shape == "one_bad_year":
            return "regime"
        return "weak_signal"
    if family == "weight":
        return "sparse" if p.coverage is not None and p.coverage < SPARSE_COVERAGE else "outliers"
    if family == "universe":
        return "illiquid_edge"
    if family == "recency":
        return "fading"
    if family == "correlation":
        return "crowded"
    return family


def build_profile(a: Any, limits: tuple[float, float], values: dict[str, float], limits_by_check: dict[str, float],
                  failing: list[str], *, yearly: YearlyShape | None = None, elasticity: Elasticity | None = None,
                  settings: dict[str, Any] | None = None) -> Profile:
    st = settings if settings is not None else (a.settings or {})
    p = Profile(sharpe=_num(a.sharpe), fitness=_num(a.fitness), returns=_num(a.returns), turnover=_num(a.turnover),
                sharpe_limit=limits[0], fitness_limit=limits[1],
                turnover_high=limits_by_check.get(HT, 0.7), turnover_low=limits_by_check.get(LT, 0.01),
                decay=int(_num(st.get("decay"))), neutralization=str(st.get("neutralization") or ""),
                truncation=_num(st.get("truncation")) or 0.08, universe=str(st.get("universe") or ""),
                coverage=coverage_of(a.long_count, a.short_count, st.get("universe")), expr=expr_facts(a.code),
                yearly=yearly or YearlyShape(), elasticity=elasticity or Elasticity())
    if p.sharpe and SU in values:
        p.sub_ratio = values[SU] / p.sharpe
    if p.returns:
        p.fitness_needs = {"turnover": p.sharpe ** 2 * abs(p.returns) / p.fitness_limit ** 2,
                           "returns": (p.fitness_limit / (p.sharpe or 1)) ** 2 * max(p.turnover, T_FLOOR),
                           "sharpe": p.fitness_limit / math.sqrt(abs(p.returns) / max(p.turnover, T_FLOOR))}
    p.causes = {name: cause_of(name, p) for name in failing}
    return p


# ---------------------------------------------------------------- moves with computed parameters

def beyond_reach(gaps: dict[str, float]) -> str | None:
    """Why no move is likely to rescue this alpha, or None. Repairs reshape an edge; they do not create one where the
    alpha has none, so these spend no simulations unless you pick the alpha yourself."""
    for name, gap in gaps.items():
        family = check_family(name)
        if family == "universe" and gap > 0.6:
            return f"{name}: under 40% of its limit, so the edge sits almost entirely in illiquid stocks."
        if family == "recency" and gap >= 1.0:
            return f"{name}: the last two years earned nothing, so the edge looks gone."
    return None


def blocked(key: str, target: str, p: Profile) -> bool:
    """A move that cannot help this check on this alpha, by arithmetic rather than by opinion."""
    if target == F and key in TURNOVER_CUTS and p.turnover <= T_FLOOR * 1.05:
        return True                # Fitness floors turnover at 12.5%: cutting further only loses signal
    if target == F and key == "decay_down" and (p.decay == 0 or p.turnover >= T_FLOOR):
        return True                # nothing to spend: only worth it while turnover is free
    if target == LT and key in TURNOVER_CUTS:
        return True
    if key in ("add_backfill", "backfill_longer") and (p.coverage is None or p.coverage >= 0.8):
        return True                # the signal is already there for nearly every stock
    if check_family(target) == "recency" and key in TURNOVER_CUTS:
        return True
    if target == S and key in TURNOVER_CUTS and p.turnover <= T_FLOOR and p.expr.bounded:
        return True                # already slow: smoothing a slow, clean signal has nothing to remove
    return False


def decay_steps(p: Profile, targets: list[str]) -> list[int]:
    """Decay values worth simulating, best first. With no curve of its own yet, an alpha gets two measured steps rather
    than a jump: what a cut costs in Sharpe varies hugely (a fast signal can lose most of it to a large decay), and those
    points become its curve. With a curve, the values forecast to land closest to passing every metric."""
    if p.turnover <= 0:
        return []
    e = p.elasticity
    if not e.fitted:
        if HT in targets:
            goals = [p.turnover_high * 0.85, p.turnover_high * HIGH_TURNOVER_GOAL]
        else:
            goals = [max(T_FLOOR, p.turnover * 0.8), max(T_FLOOR, p.turnover * 0.6)]
        return list(dict.fromkeys(n for n in (solve_decay(p.turnover, p.decay, g, e.beta) for g in goals) if n > p.decay))
    now = sum(metric_gaps(p, forecast(p, (0.0, 0.0, 0.0))).values())
    scored = []
    for n in DECAY_GRID:
        if n <= p.decay:
            continue
        turnover = p.turnover * ((n + 1) / (p.decay + 1)) ** -e.beta
        if turnover < T_FLOOR * 0.8:
            break                  # past the floor a bigger decay only costs signal
        gap = sum(metric_gaps(p, forecast(p, _turnover_effect(p, turnover))).values())
        if gap < now:
            scored.append((round(gap, 4), n))
    return [n for _, n in sorted(scored)[:2]]


def worth_measuring(p: Profile) -> bool:
    """The forecast may be wrong about this alpha's decay curve: one step measures it, and the curve then guides the
    rest. The default slopes are an average; the Sharpe a turnover cut costs varies a lot from signal to signal."""
    return not p.elasticity.fitted and p.turnover > T_FLOOR * 1.2 and bool(decay_steps(p, [F]))


def free_turnover_decay(p: Profile) -> int | None:
    """Decay that spends the turnover Fitness does not charge for: back up to just under the 12.5% floor."""
    if p.decay <= 0 or p.turnover <= 0 or p.turnover >= T_FLOOR:
        return None
    goal = T_FLOOR * 0.95
    n = (p.decay + 1) * (p.turnover / goal) ** (1.0 / p.elasticity.beta) - 1
    n = max(0, math.floor(n))
    return n if n < p.decay else max(0, p.decay // 2)


def smoothing_window(p: Profile) -> int:
    """ts_decay_linear length for the cut still needed (the operator follows the same curve as the decay setting)."""
    steps = decay_steps(p, [F] if p.turnover <= p.turnover_high else [HT])
    return min(30, max(3, (steps[0] if steps else p.decay + 4) - p.decay + 1))


def hump_size(p: Profile) -> float:
    return 0.05 if p.turnover > p.turnover_high else 0.01


# (Δlog Sharpe, Δlog returns, Δlog turnover) per move before any of your own results exist; turnover cuts are computed
# from the solved step instead. `CLOSES` is the share of a check's gap a move closed on the synthetic market.
EFFECTS: dict[str, tuple[float, float, float]] = {
    "neutralize_group": (0.03, 0.02, -0.01),
    "neutralize_risk": (-0.01, -0.05, -0.01),
    "group_rank": (-0.08, -0.14, 0.03),
    "rank_wrap": (0.0, -0.02, -0.03),
    "winsorize": (0.0, 0.0, 0.0),
    "windows_short": (-0.01, -0.02, 0.07),
    "windows_long": (0.0, 0.0, -0.02),
    "add_backfill": (0.08, 0.05, -0.1),
    "backfill_longer": (0.05, 0.03, -0.05),
    "sharpen": (0.01, 0.02, 0.04),
    "decay_down": (0.03, 0.02, 0.18),
    "speed_up": (-0.16, -0.14, 0.12),
    "liquid_tilt": (-0.25, -0.27, -0.03),
    "liquid_filter": (-0.28, -0.16, 0.02),
    "size_neutral": (-0.04, -0.03, 0.01),
    "liquidity_neutral": (-0.02, -0.01, 0.01),
    "universe_down": (-0.66, -0.25, 0.0),
    "regime_gate": (0.0, 0.0, -0.05),
    "hump": (-0.13, -0.13, -0.11),
    "event_gate": (-0.31, -0.29, -0.30),
    "horizon_far": (-0.08, -0.07, 0.15),
}
# Moves that make a nearly static signal trade: no such alpha came up on the synthetic market, so these are estimates.
LOW_TURNOVER_EFFECTS: dict[str, tuple[float, float, float]] = {"speed_up": (-0.1, 0.1, 1.5),
                                                              "windows_short": (0.0, 0.05, 0.7),
                                                              "decay_down": (0.0, 0.02, 0.7)}
CLOSES: dict[str, dict[str, float]] = {
    "universe": {"liquid_tilt": 0.4, "liquid_filter": 0.15, "universe_down": 0.1, "size_neutral": 0.2,
                 "liquidity_neutral": 0.4, "group_rank": 0.2, "neutralize_group": 0.35, "neutralize_risk": 0.35},
    "recency": {"windows_short": 0.35, "neutralize_group": 0.3, "neutralize_risk": 0.3, "sibling_field": 0.3,
                "group_rank": 0.2},
    "weight": {"add_backfill": 0.5, "truncation": 0.5, "rank_wrap": 0.5, "winsorize": 0.35, "backfill_longer": 0.3},
    "correlation": {"residualize": 0.7, "sibling_field": 0.5, "horizon_far": 0.3, "event_gate": 0.25,
                    "neutralize_risk": 0.2, "outperform": 0.4},
    "unknown": {},
}
CAUSE_BONUS: dict[str, dict[str, float]] = {
    "sparse": {"add_backfill": 2.0, "backfill_longer": 1.5},
    "outliers": {"rank_wrap": 2.0, "winsorize": 1.5, "group_rank": 1.3},
    "fading": {"windows_short": 2.0, "sibling_field": 1.3},
    "regime": {"regime_gate": 2.0, "event_gate": 1.2},
    "weak_signal": {"neutralize_group": 1.3, "neutralize_risk": 1.2, "group_rank": 1.2},
    "returns": {"sharpen": 1.5, "decay_down": 2.0, "windows_short": 1.3},
    "turnover": {"decay_up": 1.3},
}


def _turnover_effect(p: Profile, target_t: float) -> tuple[float, float, float]:
    e = p.elasticity
    dt = math.log(target_t / p.turnover)
    return e.sigma * dt, e.gamma * dt, dt


def effect_of(key: str, p: Profile, targets: list[str]) -> tuple[float, float, float]:
    if key in ("decay_up", "decay_linear") and p.turnover > 0:
        steps = decay_steps(p, targets)
        n = steps[0] if steps else p.decay
        return _turnover_effect(p, p.turnover * ((n + 1) / (p.decay + 1)) ** -p.elasticity.beta)
    if LT in targets and key in LOW_TURNOVER_EFFECTS:
        return LOW_TURNOVER_EFFECTS[key]
    return EFFECTS.get(key, (0.0, 0.0, 0.0))


@dataclass
class Forecast:
    sharpe: float
    fitness: float
    turnover: float


def forecast(p: Profile, eff: tuple[float, float, float]) -> Forecast:
    s = p.sharpe * math.exp(eff[0])
    r = p.returns * math.exp(eff[1])
    t = p.turnover * math.exp(eff[2])
    identity = fitness_of(p.sharpe, p.returns, p.turnover)
    scale = p.fitness / identity if identity and p.fitness else 1.0   # keep BRAIN's reported Fitness as the base
    return Forecast(s, fitness_of(s, r, t) * scale, t)


def metric_gaps(p: Profile, f: Forecast) -> dict[str, float]:
    return {S: max(0.0, (p.sharpe_limit - f.sharpe) / p.sharpe_limit),
            F: max(0.0, (p.fitness_limit - f.fitness) / p.fitness_limit),
            HT: max(0.0, (f.turnover - p.turnover_high) / p.turnover_high),
            LT: max(0.0, (p.turnover_low - f.turnover) / p.turnover_low)}


def predicted_gain(key: str, target: str, gaps: dict[str, float], p: Profile, targets: list[str]) -> float:
    """Share of the alpha's total distance to passing this move is expected to remove (0 = no use, can be negative
    when it would break a check that passes today). Metric checks use the Fitness identity; the rest use CLOSES."""
    def capped(values: Any) -> float:
        return sum(min(g, GAP_CAP) for g in values)

    before = metric_gaps(p, forecast(p, (0.0, 0.0, 0.0)))      # the same yardstick before and after the move
    after = metric_gaps(p, forecast(p, effect_of(key, p, targets)))
    other = {name: gap for name, gap in gaps.items() if name not in after}
    total = capped(before.values()) + capped(other.values()) or 1e-6
    left = capped(after.values()) + sum(min(gap, GAP_CAP) * (1 - CLOSES.get(check_family(name), {}).get(key, 0.0))
                                        for name, gap in other.items())
    return (total - left) / total * CAUSE_BONUS.get(p.causes.get(target, ""), {}).get(key, 1.0)


# ---------------------------------------------------------------- explanation

CAUSE_TEXT = {
    "turnover": "turnover is the problem: cutting it can pass Fitness",
    "returns": "Sharpe is fine but each traded dollar earns too little",
    "sharpe": "the signal itself is too weak",
    "sparse": "the signal is missing for many stocks",
    "outliers": "a few extreme values dominate the book",
    "fading": "the edge has weakened in recent years",
    "regime": "one bad year is dragging the result down",
    "weak_signal": "the signal is steady but weak",
    "illiquid_edge": "the edge sits in small, illiquid stocks",
    "crowded": "it overlaps with existing alphas",
    "turnover_check": "turnover is outside the allowed range",
}


def prescription(failing: list[str], gaps: dict[str, float], values: dict[str, float], p: Profile,
                 limits_by_check: dict[str, float]) -> list[str]:
    """The diagnosis in plain sentences, with the numbers each check needs. Shown on the alpha and given to the AI."""
    lines = []
    for name in sorted(failing, key=lambda n: gaps.get(n, 0), reverse=True):
        now = values.get(name)
        lim = limits_by_check.get(name)
        nums = f" ({now:.3g} vs limit {lim:.3g})" if now is not None and lim is not None else ""
        cause = p.causes.get(name, "")
        lines.append(f"{name}{nums}: {CAUSE_TEXT.get(cause, cause or 'cause unknown')}.")
        if name == F and p.fitness_needs:
            need = p.fitness_needs
            parts = [f"Sharpe {p.sharpe:.2f}, returns {p.returns:.1%}, turnover {p.turnover:.1%}"]
            if need["turnover"] >= T_FLOOR:
                parts.append(f"at this Sharpe and returns it passes at turnover ≤ {need['turnover']:.1%}")
            else:
                parts.append("turnover alone cannot pass it (Fitness ignores turnover below 12.5%)")
            parts.append(f"or returns ≥ {need['returns']:.1%}, or Sharpe ≥ {need['sharpe']:.2f}")
            lines.append("  " + "; ".join(parts) + ".")
    if p.coverage is not None:
        lines.append(f"Coverage: {p.coverage:.0%} of the universe held on average.")
    if p.yearly.shape not in ("unknown", "steady"):
        y = p.yearly
        lines.append(f"Yearly shape: {y.shape.replace('_', ' ')} (last 2 years Sharpe {y.last2 or 0:.2f}, earlier "
                     f"{y.earlier or 0:.2f}, worst year {y.worst_year} at {y.worst or 0:.2f}).")
    return lines
