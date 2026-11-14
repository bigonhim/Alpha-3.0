"""The quarter BRAIN judges a consultant on (alpha-harness api/quarter.py).

Levels are evaluated quarterly; two of the criteria are countable here — signals submitted in the quarter and pyramids
formulated in it. Ties are broken partly on the distinct operators the quarter's alphas use, which is countable too.
The quarter runs on BRAIN's clock (New York): a UTC date starts the next quarter hours early.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable

from sqlalchemy import select

from ..brain.catalog_api import catalog_api, next_quarter_start, ny_today, quarter_label, quarter_start
from ..brain.platform import PYRAMID_MIN_ALPHAS, UNCOUNTED_OPERATORS
from ..db import db
from ..db.models import Operator

log = logging.getLogger(__name__)

TTL = 600
_cache: dict[str, tuple[float, Any]] = {}


def bounds(today: date | None = None) -> tuple[date, date]:
    """The quarter in progress: its first day, and the next quarter's."""
    today = today or ny_today()
    return quarter_start(today), next_quarter_start(today)


@dataclass
class Standing:
    label: str
    start: str
    end: str
    #: Alphas submitted inside the quarter, summed from BRAIN's own daily counts.
    submitted: int
    #: Pyramids with at least PYRAMID_MIN_ALPHAS submitted alphas.
    pyramids_formulated: int
    #: Pyramids with at least one submitted alpha but not yet enough to count.
    pyramids_started: int
    alphas_per_pyramid: int = PYRAMID_MIN_ALPHAS
    errors: list[str] = field(default_factory=list)


def tally(days: list[tuple[str, int]], pyramids: list[dict[str, Any]], start: date, after: date) -> Standing:
    end = after - timedelta(days=1)
    first, last = start.isoformat(), end.isoformat()
    submitted = sum(n for day, n in days if first <= day[:10] <= last)
    counts = [int(p.get("alphaCount") or 0) for p in pyramids]
    return Standing(label=quarter_label(start), start=first, end=last, submitted=submitted,
                    pyramids_formulated=sum(1 for n in counts if n >= PYRAMID_MIN_ALPHAS),
                    pyramids_started=sum(1 for n in counts if 0 < n < PYRAMID_MIN_ALPHAS))


async def standing(today: date | None = None, *, fresh: bool = False) -> Standing:
    """Submitted alphas and formulated pyramids for the quarter in progress. Neither is derivable from the other: an
    alpha belongs to as many pyramids as it has data categories, so pyramid counts sum to more than the alphas."""
    start, after = bounds(today)
    key = f"standing:{start}"
    hit = _cache.get(key)
    if hit and not fresh and time.monotonic() - hit[0] < TTL:
        return hit[1]
    api = catalog_api()
    days, pyramids = await asyncio.gather(
        api.submission_activity(),
        # BRAIN's own site asks up to the day after the quarter, so its last day counts
        api.pyramid_alphas(start.isoformat(), after.isoformat()), return_exceptions=True)
    errors = []
    if isinstance(days, BaseException):
        errors.append(f"submissions: {getattr(days, 'message', days)}")
        days = []
    if isinstance(pyramids, BaseException):
        errors.append(f"pyramids: {getattr(pyramids, 'message', pyramids)}")
        pyramids = []
    result = tally(days, pyramids, start, after)
    result.errors = errors
    if not errors:
        _cache[key] = (time.monotonic(), result)
    return result


@dataclass
class OperatorUse:
    name: str
    category: str
    alphas: int


@dataclass
class OperatorUsage:
    label: str
    alphas: int
    unreadable: int
    unlisted: int
    operators: list[OperatorUse]


def usage(expressions: Iterable[str], operators: list[dict[str, Any]]) -> tuple[list[OperatorUse], int]:
    """How many expressions use each operator a regular alpha can (unused first, then the least used), and how many
    expressions could not be read. ts_backfill and group_backfill are left out: BRAIN does not count them."""
    from ..labs.fastexpr import operators_in
    used: Counter[str] = Counter()
    unreadable = 0
    for expression in expressions:
        try:
            used.update(set(operators_in(expression)))       # an alpha counts once per operator
        except Exception:
            unreadable += 1
    rows = [OperatorUse(name=name, category=str(o.get("category") or ""), alphas=used[name])
            for o in operators
            if isinstance(name := o.get("name"), str) and name not in UNCOUNTED_OPERATORS
            and "REGULAR" in (o.get("scope") or ["REGULAR"])]
    return sorted(rows, key=lambda r: (r.alphas, r.name)), unreadable


async def account_operators() -> list[dict[str, Any]]:
    """The account's own operators from the local catalogue; fetched from BRAIN when none is stored yet."""
    async with db.session() as s:
        rows = list(await s.scalars(select(Operator)))
    if rows:
        return [{"name": o.name, "category": o.category, "scope": o.scope} for o in rows]
    from ..engine import sync as sync_engine
    await sync_engine.sync_operators()
    async with db.session() as s:
        return [{"name": o.name, "category": o.category, "scope": o.scope} for o in await s.scalars(select(Operator))]


def _expression(alpha: dict[str, Any]) -> str | None:
    if str(alpha.get("type") or "").upper() in ("SUPER", "RA_PARENT"):
        return None                           # a SuperAlpha has no regular code; a parent repeats its children's
    regular = alpha.get("regular")
    code = regular.get("code") if isinstance(regular, dict) else regular
    return code if isinstance(code, str) and code.strip() else None


async def operator_usage(today: date | None = None, *, fresh: bool = False) -> OperatorUsage:
    """How many of the quarter's submitted alphas use each of the account's operators; the unused lead, since
    BRAIN ranks the distinct operators a consultant used in the quarter."""
    start, after = bounds(today)
    key = f"operators:{start}"
    hit = _cache.get(key)
    if hit and not fresh and time.monotonic() - hit[0] < TTL:
        return hit[1]
    listed = await account_operators()
    if not listed:
        raise RuntimeError("BRAIN's operator list could not be read. Sign in again and retry.")
    alphas, total = await catalog_api().submitted_alphas(start, after)
    expressions = [e for a in alphas if (e := _expression(a))]
    rows, unreadable = usage(expressions, listed)
    result = OperatorUsage(label=quarter_label(start), alphas=len(expressions) - unreadable, unreadable=unreadable,
                           unlisted=max(0, total - len(alphas)), operators=rows)
    _cache[key] = (time.monotonic(), result)
    return result


def clear_cache() -> None:
    _cache.clear()
