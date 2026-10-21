"""How close a new alpha is to the alphas you already submitted — before BRAIN's own correlation check.

Two estimates, both free of simulation quota:

- structural (before simulating): same expression skeleton, shared data fields and shared operators. Only a
  near-certain duplicate — same skeleton and the same fields as a submitted alpha — is left out by default.
- PnL (after simulating): the correlation of the alpha's daily in-sample PnL (one GET of its `pnl` record set) with
  the stored daily PnL of your submitted alphas in the same region and delay.

BRAIN does not publish exactly how SELF_CORRELATION is computed, so the PnL figure is an estimate. It is also a lower
bound: only submitted alphas whose daily PnL has been fetched ("Prepare data") are compared, and PROD correlation
(against other consultants' alphas) cannot be estimated locally. It orders BRAIN checks; it never replaces one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import select

from ..db import db
from ..db.models import Alpha
from . import pnl

log = logging.getLogger(__name__)

MIN_OVERLAP_DAYS = 120


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def structural_score(skeleton_a: str | None, fields_a: set[str], ops_a: set[str],
                     skeleton_b: str | None, fields_b: set[str], ops_b: set[str]) -> float:
    """0..1: half from the shared fields, a quarter each from the skeleton and the shared operators."""
    same_skeleton = 1.0 if skeleton_a and skeleton_a == skeleton_b else 0.0
    return round(0.5 * jaccard(fields_a, fields_b) + 0.25 * same_skeleton + 0.25 * jaccard(ops_a, ops_b), 4)


@dataclass
class SubmittedShape:
    alpha_id: str
    skeleton: str | None
    fields: frozenset[str]
    operators: frozenset[str]


async def submitted_shapes(region: str, delay: int) -> list[SubmittedShape]:
    """Skeleton, fields and operators of your submitted (ACTIVE) alphas in one region/delay."""
    from .ingest import expression_hashes
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE", Alpha.region == region,
                                                        Alpha.delay == delay, Alpha.type == "REGULAR")))
    out = []
    for a in rows:
        skeleton, fields = a.skeleton_hash, a.fields
        if a.code and (skeleton is None or fields is None):
            _, skeleton, _, fields = expression_hashes(a.code)
        out.append(SubmittedShape(a.id, skeleton, frozenset(fields or []), frozenset(_operators(a.code))))
    return out


def _operators(code: str | None) -> set[str]:
    if not code:
        return set()
    try:
        from ..fastexpr import compute_metrics, parse
        return set(compute_metrics(parse(code)).unique_operators)
    except Exception:
        return set()


def likely_duplicate(skeleton: str | None, fields: set[str] | frozenset[str],
                     submitted: list[SubmittedShape]) -> str | None:
    """Id of a submitted alpha with the same skeleton and exactly the same data fields (a near-certain duplicate)."""
    if not skeleton:
        return None
    for s in submitted:
        if s.skeleton == skeleton and s.fields == frozenset(fields):
            return s.alpha_id
    return None


def best_structural_match(skeleton: str | None, fields: set[str], operators: set[str],
                          submitted: list[SubmittedShape]) -> tuple[float, str | None]:
    best: tuple[float, str | None] = (0.0, None)
    for s in submitted:
        score = structural_score(skeleton, fields, operators, s.skeleton, set(s.fields), set(s.operators))
        if score > best[0]:
            best = (score, s.alpha_id)
    return best


# ---------------------------------------------------------------- PnL-based estimate

def max_correlation(df: pd.DataFrame, column: str, others: list[str]) -> tuple[float | None, str | None]:
    """Highest correlation of `column` with any of `others` over their overlapping days."""
    best: tuple[float | None, str | None] = (None, None)
    for o in others:
        if o == column or o not in df.columns:
            continue
        pair = df[[column, o]].dropna()
        if len(pair) < MIN_OVERLAP_DAYS:
            continue
        c = pair[column].corr(pair[o])
        if pd.notna(c) and (best[0] is None or c > best[0]):
            best = (float(c), o)
    return best


async def estimate_self_corr(alpha_id: str, *, fetch: bool = True) -> tuple[float | None, str | None]:
    """Estimate (max correlation, closest submitted alpha) from daily PnL and store it on the alpha."""
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
    if a is None or a.region is None:
        return None, None
    try:
        row = await pnl.fetch(alpha_id, keep_raw=False) if fetch else (await pnl.stored([alpha_id])).get(alpha_id)
    except Exception as exc:
        log.warning("daily PnL for %s unavailable: %s", alpha_id, getattr(exc, "message", exc))
        return None, None
    if row is None:
        return None, None
    submitted = [x for x in await pnl.active_alphas(a.region, a.delay) if x.id != alpha_id]
    rows: dict[str, Any] = await pnl.stored([x.id for x in submitted])
    if not rows:
        return None, None
    rows[alpha_id] = row
    corr, closest = max_correlation(pnl.frame(rows), alpha_id, [k for k in rows if k != alpha_id])
    if corr is not None:
        async with db.write() as s:
            target = await s.get(Alpha, alpha_id)
            if target is not None:
                target.est_self_corr, target.est_corr_with = round(corr, 4), closest
    return corr, closest


async def calibration() -> dict[str, Any]:
    """How far the local estimate has been from BRAIN's SELF_CORRELATION so far."""
    from .actions import CALIBRATION_KEY
    rows = await db.kv_get(CALIBRATION_KEY) or []
    if not rows:
        return {"n": 0, "mae": None, "bias": None, "rows": []}
    errors = [r["estimate"] - r["real"] for r in rows]
    return {"n": len(rows), "mae": round(sum(abs(e) for e in errors) / len(errors), 3),
            "bias": round(sum(errors) / len(errors), 3), "rows": rows[-50:]}
