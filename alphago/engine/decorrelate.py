"""Strong to Ready: an alpha that passes every metric but collides with an existing alpha needs its own route.

Generic rewrites rarely lower a PnL correlation, so this works from what the collision actually is:

- Outperform (SELF only): community reports say a correlated alpha may still pass with Sharpe at least 10% above the
  alpha it collides with. Unverified, so it only steers the choice of moves; BRAIN's /check decides.
- Residualise (SELF only): regress the alpha on the submitted alpha it collides with and keep what is left. Terminal,
  because the child embeds that alpha and further rewrites would change it too. It also adds the other alpha's data
  and operators, so the child may lose single-dataset status and Power Pool shape.
- New information: the closest less-crowded fields of the same dataset.
- Horizon and condition: much shorter or longer windows, and event gating.
- PROD_CORRELATION names no alpha, so residualising is impossible there; it leans on data, gating and CROWDING.

Each round is simulated and checked by BRAIN; the next round continues from the child that got closest.
Submission is never automatic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from ..db import db
from ..db.models import Alpha
from .evaluator import CORRELATION_CHECKS, blocking_failures, check_map

log = logging.getLogger(__name__)

OUTPERFORM = 1.1            # community-reported: 10% better Sharpe may pass a correlation failure


@dataclass
class Collision:
    check: str                          # SELF_CORRELATION or PROD_CORRELATION
    value: float | None = None
    other_id: str | None = None
    other_sharpe: float | None = None
    code: str | None = None             # the submitted alpha's expression, when it is yours and known
    sharpe_gap: float | None = None     # Sharpe still needed to clear the 10% rule


def _corr_of(record: dict) -> float | None:
    return next((v for k, v in record.items() if "corr" in k.lower() and isinstance(v, (int, float))), None)


def _closest(records: list[dict]) -> tuple[float, str, float | None] | None:
    best = None
    for rec in records:
        corr = _corr_of(rec)
        aid = rec.get("id") or rec.get("alphaId") or rec.get("alpha_id")
        if aid and corr is not None and (best is None or corr > best[0]):
            sharpe = rec.get("sharpe")
            best = (corr, str(aid), sharpe if isinstance(sharpe, (int, float)) else None)
    return best


def failing_correlation(a: Alpha) -> str | None:
    failing = set(blocking_failures(a.checks)) & CORRELATION_CHECKS
    return "SELF_CORRELATION" if "SELF_CORRELATION" in failing else next(iter(failing), None)


def correlation_value(a: Alpha, check: str) -> float | None:
    v = check_map(a.checks).get(check, {}).get("value")
    return float(v) if isinstance(v, (int, float)) else None


async def find_collision(a: Alpha) -> Collision:
    """What this alpha collides with: BRAIN's correlation list when it answers, else the local PnL estimate."""
    from . import actions
    check = failing_correlation(a) or "SELF_CORRELATION"
    hit = Collision(check, correlation_value(a, check))
    if check != "SELF_CORRELATION":
        return hit
    other = None
    try:
        res = await actions.correlations(a.id, "self")
        best = _closest(res.get("records") or [])
        if best:
            other, hit.other_sharpe = best[1], best[2]
            hit.value = hit.value if hit.value is not None else best[0]
    except Exception as exc:
        log.info("correlation list for %s unavailable: %s", a.id, getattr(exc, "message", exc))
    other = other or a.est_corr_with
    hit.other_id = other
    if other:
        async with db.session() as s:
            row = await s.get(Alpha, other)
        if row is not None:
            hit.code = row.code
            hit.other_sharpe = hit.other_sharpe if hit.other_sharpe is not None else row.sharpe
    if hit.other_sharpe is not None and a.sharpe is not None:
        hit.sharpe_gap = OUTPERFORM * hit.other_sharpe - a.sharpe
    return hit


async def decorrelate(a: Alpha, ctx, *, cap: int = 3) -> list[str]:  # noqa: ANN001
    """Repair a correlation failure, round by round, within the caller's budget. Returns the ids that passed."""
    from .refine import _load_alphas, certify, refine
    current = a
    for _ in range(max(1, ctx.max_depth)):
        if ctx.stopped() or await ctx.budget() <= 0:
            break
        hit = await find_collision(current)
        sub = replace(ctx, max_depth=1, collide=hit.code, collision=hit, revisit=True, resume=False)
        report = await refine([current], sub)
        if not report.strong:
            break
        checked = await certify(report.strong, sub, cap=min(3, cap), corr_repair=False)
        if checked.passed:
            return checked.passed
        kids = await _load_alphas(checked.corr_failed)
        scored = [(correlation_value(k, hit.check), k) for k in kids.values()]
        scored = [(v, k) for v, k in scored if v is not None]
        if not scored:
            break
        value, best = min(scored, key=lambda x: x[0])
        if hit.value is not None and value >= hit.value - 0.01:
            break                      # no child moved the correlation: more of the same would not either
        current = best
    return []
