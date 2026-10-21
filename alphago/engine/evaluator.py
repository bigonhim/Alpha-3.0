"""Turn BRAIN in-sample results into a verdict (rejected / weak / promising / strong / ready) and a score.

BRAIN's own check `limit` values are authoritative; the fallbacks below are only used when a check is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# informational checks that never block submission readiness
NON_BLOCKING = {"MATCHES_PYRAMID", "MATCHES_THEMES", "MATCHES_COMPETITION", "POWER_POOL_CORRELATION"}
CORRELATION_CHECKS = {"SELF_CORRELATION", "PROD_CORRELATION"}

FALLBACK_LIMITS = {
    # delay -> (sharpe, fitness)
    1: (1.25, 1.0),
    0: (2.0, 1.3),
}

VERDICT_ORDER = ["rejected", "weak", "promising", "strong", "ready", "submitted"]


@dataclass
class Evaluation:
    verdict: str
    score: float
    failing: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sharpe_limit: float = 1.25
    fitness_limit: float = 1.0
    flip_sign: bool = False
    notes: list[str] = field(default_factory=list)


def _f(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def check_map(checks: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {str(c.get("name")): c for c in (checks or []) if c.get("name")}


def blocking_failures(checks: list[dict[str, Any]] | None) -> list[str]:
    """Names of FAIL/ERROR checks that block submission (the one rule shared by the evaluator and /check)."""
    return [str(c.get("name")) for c in checks or []
            if str(c.get("result", "")).upper() in ("FAIL", "ERROR") and str(c.get("name")) not in NON_BLOCKING]


def pending_checks(checks: list[dict[str, Any]] | None) -> list[str]:
    return [str(c.get("name")) for c in checks or []
            if str(c.get("result", "")).upper() == "PENDING" and str(c.get("name")) not in NON_BLOCKING]


def check_state_of(checks: list[dict[str, Any]] | None) -> str:
    """State of a full BRAIN /check result: fail | pending | pass."""
    return "fail" if blocking_failures(checks) else ("pending" if pending_checks(checks) else "pass")


def _delay(delay: Any) -> int:
    """Delay as an int; None means the usual Delay 1 (0 stays 0 — `int(delay or 1)` turned D0 into D1)."""
    try:
        return 1 if delay is None else int(delay)
    except (TypeError, ValueError):
        return 1


def limits_for(checks: list[dict[str, Any]] | None, delay: int | None) -> tuple[float, float]:
    cm = check_map(checks)
    fb_sharpe, fb_fitness = FALLBACK_LIMITS.get(_delay(delay), FALLBACK_LIMITS[1])
    sharpe_lim = _f((cm.get("LOW_SHARPE") or {}).get("limit")) or fb_sharpe
    fitness_lim = _f((cm.get("LOW_FITNESS") or {}).get("limit")) or fb_fitness
    return sharpe_lim, fitness_lim


def evaluate(
    *,
    sharpe: Any,
    fitness: Any,
    turnover: Any,
    checks: list[dict[str, Any]] | None,
    delay: int | None = 1,
    region: str | None = None,
    status: str | None = None,
    full_check_passed: bool = False,
) -> Evaluation:
    sharpe_v, fitness_v, turnover_v = _f(sharpe), _f(fitness), _f(turnover)
    sharpe_lim, fitness_lim = limits_for(checks, delay)
    if str(status or "").upper() in ("ACTIVE", "DECOMMISSIONED"):
        return Evaluation("submitted", 100.0, sharpe_limit=sharpe_lim, fitness_limit=fitness_lim)
    if sharpe_v is None:
        return Evaluation("rejected", 0.0, ["NO_RESULT"], sharpe_limit=sharpe_lim, fitness_limit=fitness_lim)

    failing, pending, warnings = [], [], []
    for c in checks or []:
        name = str(c.get("name"))
        result = str(c.get("result", "")).upper()
        if name in NON_BLOCKING:
            continue
        if result in ("FAIL", "ERROR"):
            failing.append(name)
        elif result == "PENDING":
            pending.append(name)
        elif result == "WARNING":
            warnings.append(name)

    ratio = sharpe_v / sharpe_lim if sharpe_lim else 0.0
    fit_ratio = (fitness_v or 0.0) / fitness_lim if fitness_lim else 0.0
    ev = Evaluation("rejected", 0.0, failing, pending, warnings, sharpe_lim, fitness_lim)

    # strongly negative alphas are just flipped signals (not in CHN, where shorting is constrained)
    if ratio <= -0.7 and (region or "").upper() != "CHN":
        ev.flip_sign = True
        ev.notes.append("Negative Sharpe — the sign-flipped alpha is a candidate")
        ratio, fit_ratio = -ratio, abs(fit_ratio)

    blocking = [f for f in failing if f not in CORRELATION_CHECKS]
    if not blocking and not ev.flip_sign:
        corr_failed = [f for f in failing if f in CORRELATION_CHECKS]
        if corr_failed:
            ev.verdict = "promising"
            ev.notes.append("Passes IS tests but is too correlated with existing alphas")
        elif full_check_passed:
            ev.verdict = "ready"
        else:
            ev.verdict = "strong"
    elif ratio >= 0.7 and fit_ratio >= 0.5:
        ev.verdict = "promising"
    elif ratio >= 0.5:
        ev.verdict = "weak"
    else:
        ev.verdict = "rejected"

    # score: weighted margin to the limits, small turnover preference for the 5–30% band
    score = 50 * min(max(ratio, 0), 2.0) + 30 * min(max(fit_ratio, 0), 2.0)
    if turnover_v is not None:
        if turnover_v > 0.7 or turnover_v < 0.01:
            score -= 15
        elif turnover_v > 0.4:
            score -= 5
    score -= 6 * len(blocking)
    ev.score = round(max(score, 0.0), 2)
    return ev


def reverdict(row: Any) -> Evaluation:
    """Recompute an Alpha row's verdict and score from its stored state.

    Shared by simulation ingest, account sync and /check, so the label cannot flip between code paths. `checks` holds
    BRAIN's latest full check result when one ran (else the simulation's IS checks); `check_state` "pass" means BRAIN's
    /check passed. A "stale" pass (another alpha was submitted since) no longer counts until it is re-checked. The
    app's robustness test never changes the verdict: a fragile alpha stays "ready" and carries a badge."""
    ev = evaluate(sharpe=row.sharpe, fitness=row.fitness, turnover=row.turnover, checks=row.checks, delay=row.delay,
                  region=row.region, status=row.status, full_check_passed=row.check_state == "pass")
    row.verdict, row.score = ev.verdict, ev.score
    return ev


def is_better(a: str | None, b: str | None) -> bool:
    """True when verdict a ranks above verdict b."""
    ia = VERDICT_ORDER.index(a) if a in VERDICT_ORDER else -1
    ib = VERDICT_ORDER.index(b) if b in VERDICT_ORDER else -1
    return ia > ib


def promising_or_better(verdict: str | None) -> bool:
    return verdict in ("promising", "strong", "ready", "submitted")
