"""Pyramids: a dataset category in one region and delay (alpha-harness catalog/pyramids.py).

BRAIN pays a multiplier per pyramid and counts one as formulated once three alphas are submitted in it in a quarter.
Both numbers come from the account's activity endpoints and the quarters from its consultant summary; whether a
pyramid can be researched here comes from the downloaded fields.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import date, timedelta
from typing import Any, Awaitable, Callable

from ..brain.catalog_api import catalog_api, next_quarter_start, ny_today, quarter_label, quarter_start
from ..brain.errors import BrainError
from ..brain.platform import PYRAMID_MIN_ALPHAS
from . import queries

log = logging.getLogger(__name__)

MULTIPLIERS_TTL = 6 * 3600
COUNTS_TTL = 600
#: The period that counts every alpha the account has submitted.
ALL_TIME = "all"

# in-process rather than persisted: a restart costs two cheap GETs
_cache: dict[str, tuple[float, Any]] = {}


class UnknownPeriodError(ValueError):
    pass


async def _cached(key: str, ttl: float, fetch: Callable[[], Awaitable[Any]]) -> Any:
    hit = _cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < ttl:
        return hit[1]
    value = await fetch()
    _cache[key] = (time.monotonic(), value)
    return value


def clear_cache() -> None:
    _cache.clear()


def _mean(values: list[float] | None) -> float:
    """Average of what BRAIN offers; nothing offered ranks last rather than as zero."""
    return sum(values) / len(values) if values else -math.inf


def assemble(multipliers: list[dict[str, Any]], counts: list[dict[str, Any]],
             synced: set[tuple[str, str, int]]) -> dict[str, Any]:
    """Merge the two activity lists and the download state into one grid.

    The multipliers decide which pyramids exist: they are the ones BRAIN pays for, and they match the markets the
    account can simulate. Submitted-alpha counts only fill in pyramids already on the grid — that list also carries
    regions the account has no access to, which would otherwise show as columns nothing can be researched in."""
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    categories: dict[str, str] = {}

    def cell(item: dict[str, Any], *, create: bool) -> dict[str, Any] | None:
        category = item.get("category") or {}
        if not isinstance(category, dict):
            category = {"id": category, "name": category}
        category_id, region, delay = category.get("id"), item.get("region"), item.get("delay")
        if not category_id or not region or delay is None:
            return None
        key = (str(category_id), str(region), int(delay))
        if not create:
            return cells.get(key)
        categories.setdefault(key[0], category.get("name") or key[0])
        return cells.setdefault(key, {"category_id": key[0], "region": key[1], "delay": key[2], "multiplier": None,
                                      "alpha_count": 0})

    for item in multipliers:
        if (found := cell(item, create=True)) is not None:
            found["multiplier"] = item.get("multiplier")
    for item in counts:
        if (found := cell(item, create=False)) is not None:
            found["alpha_count"] = int(item.get("alphaCount") or 0)
    for key, found in cells.items():
        found["synced"] = key in synced

    # both axes rank on the multiplier they average, richest first, ties alphabetical; a pyramid BRAIN does not offer
    # is absent from the average rather than counted as zero
    by_category: dict[str, list[float]] = {}
    by_column: dict[tuple[str, int], list[float]] = {}
    for found in cells.values():
        if found["multiplier"] is None:
            continue
        by_category.setdefault(found["category_id"], []).append(float(found["multiplier"]))
        by_column.setdefault((found["region"], found["delay"]), []).append(float(found["multiplier"]))
    columns = sorted({(c["region"], c["delay"]) for c in cells.values()},
                     key=lambda col: (-_mean(by_column.get(col)), col[0], col[1]))
    ordered = sorted(categories.items(), key=lambda item: (-_mean(by_category.get(item[0])), item[1]))
    return {"columns": [{"region": r, "delay": d} for r, d in columns],
            "categories": [{"id": cid, "name": name} for cid, name in ordered],
            "cells": list(cells.values()), "alphas_per_pyramid": PYRAMID_MIN_ALPHAS}


def fallback_quarter(today: date | None = None) -> dict[str, str]:
    """The calendar quarter in progress on BRAIN's clock (New York), which BRAIN's own site falls back to."""
    today = today or ny_today()
    start = quarter_start(today)
    return {"name": quarter_label(start), "startDate": start.isoformat(),
            "endDate": (next_quarter_start(today) - timedelta(days=1)).isoformat()}


async def quarters() -> list[dict[str, str]]:
    """BRAIN's quarters, newest first; the current calendar quarter when BRAIN gives no consultant summary."""
    try:
        listed = await _cached("quarters", COUNTS_TTL, catalog_api().quarters)
    except BrainError as exc:
        log.warning("pyramids: quarters unavailable: %s", exc)
        listed = []
    return listed or [fallback_quarter()]


def count_window(quarter: dict[str, str]) -> tuple[str, str]:
    """BRAIN's own site asks up to the day after the quarter, so its last day counts."""
    return quarter["startDate"], (date.fromisoformat(quarter["endDate"]) + timedelta(days=1)).isoformat()


async def pyramid_grid(period: str | None = None) -> dict[str, Any]:
    """The grid for `period`: a quarter's name, ALL_TIME, or (None) the quarter in progress."""
    api = catalog_api()
    listed = await quarters()
    if period == ALL_TIME:
        start = end = None
    else:
        chosen = listed[0] if not period else next((q for q in listed if q["name"] == period), None)
        if chosen is None:
            raise UnknownPeriodError(f"BRAIN offers no quarter named {period}.")
        period = chosen["name"]
        start, end = count_window(chosen)
    multipliers = await _cached("multipliers", MULTIPLIERS_TTL, api.pyramid_multipliers)
    counts = await _cached(f"alphas:{period}", COUNTS_TTL, lambda: api.pyramid_alphas(start, end))
    synced = await queries.synced_categories()
    return assemble(multipliers, counts, synced) | {
        "period": period,
        "quarters": [{"name": q["name"], "start": q["startDate"], "end": q["endDate"]} for q in listed]}


def standing(count: int, needed: int, all_time: bool) -> tuple[str, str]:
    """A pyramid's standing as a label and a note: formulated, n/needed started, or none submitted. Formulating is
    per quarter, so across all time a count is only submitted or not."""
    if all_time and count > 0:
        return f"{count:,}", "formulating counts per quarter"
    if count >= needed:
        return f"{count:,}", "formulated"
    if count > 0:
        return f"{count}/{needed}", f"{needed - count} more to formulate"
    return "—", "none submitted"


def peaks(groups: dict[Any, list[float]]) -> dict[Any, float]:
    """The strongest multiplier in a row or column, only where something is actually lower."""
    best = {}
    for key, values in groups.items():
        top = max(values)
        if min(values) != top:
            best[key] = top
    return best


TOP_MULTIPLIER = 2.0
RAMP = ["#14241c", "#163323", "#17402a", "#184d31", "#195a38", "#1a673f", "#1b7446", "#1c814d", "#1d8e54",
        "#1e9b5b", "#1fa862"]


def ramp_colour(multiplier: float | None, ceiling: float) -> str:
    """Colour means payout, not rank: anchored at ×1.0 (standard payout), topping out at ×2.0 unless BRAIN offers
    more, so ×1.4 looks like ×1.4 whatever else the quarter offers."""
    if multiplier is None:
        return RAMP[0]
    top = max(TOP_MULTIPLIER, ceiling)
    ratio = min(1.0, max(0.0, (float(multiplier) - 1) / (top - 1)))
    return RAMP[round(ratio * (len(RAMP) - 1))]
