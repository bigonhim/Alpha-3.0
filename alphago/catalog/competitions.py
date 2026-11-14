"""Competitions: what BRAIN is running, and where you stand in each (alpha-harness api/competitions.py).

BRAIN lists every competition with your status in it, but your rank only comes with one competition at a time, so only
the ongoing ones are read in full. Joining and submitting both happen on BRAIN.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..brain.catalog_api import catalog_api

log = logging.getLogger(__name__)

PLATFORM_COMPETITIONS_URL = "https://platform.worldquantbrain.com/competition/"


@dataclass
class Competition:
    id: str
    name: str
    #: BRAIN's word for your place in it, e.g. ACCEPTED or EXCLUDED.
    status: str
    enrolled: bool
    ongoing: bool
    scoring: str | None
    start_date: str | None
    end_date: str | None
    sign_up_end_date: str | None
    faq: str | None
    #: Only for an ongoing competition: {rank, alphas}.
    rank: int | None = None
    alphas: int | None = None
    standing: bool = False

    @property
    def url(self) -> str:
        from urllib.parse import quote
        return f"{PLATFORM_COMPETITIONS_URL}{quote(self.id, safe='')}"

    def days_left(self, now: datetime | None = None) -> int | None:
        return days_until(self.end_date, now)

    def sign_up_open(self, now: datetime | None = None) -> bool:
        return not self.enrolled and (days_until(self.sign_up_end_date, now) or 0) > 0


def when(value: Any) -> datetime | None:
    """A BRAIN timestamp as an aware datetime; one without an offset is read as UTC rather than left naive."""
    if not value:
        return None
    try:
        at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return at if at.tzinfo is not None else at.replace(tzinfo=timezone.utc)


def days_until(value: Any, now: datetime | None = None) -> int | None:
    at = when(value)
    if at is None:
        return None
    now = now or datetime.now(timezone.utc)
    return max(0, math.ceil((at - now).total_seconds() / 86400))


def is_ongoing(c: dict[str, Any], now: datetime) -> bool:
    start, end = when(c.get("startDate")), when(c.get("endDate"))
    return start is not None and end is not None and start <= now <= end


async def competitions(now: datetime | None = None) -> list[Competition]:
    """Ongoing first, soonest to end; then the rest, most recent first (by the moment, not the text: BRAIN's
    offsets change with daylight saving)."""
    api = catalog_api()
    listed = await api.competitions()
    now = now or datetime.now(timezone.utc)
    ongoing = [c for c in listed if is_ongoing(c, now)]
    # one competition BRAIN will not show (restricted, or a passing error) costs only its rank, not the page
    details = await asyncio.gather(*(api.competition(str(c["id"])) for c in ongoing), return_exceptions=True)
    boards: dict[str, Any] = {}
    for c, d in zip(ongoing, details):
        if isinstance(d, BaseException):
            log.warning("competition %s: detail unavailable: %s", c.get("id"), d)
        else:
            boards[str(d.get("id") or c.get("id"))] = d.get("leaderboard")

    rows = []
    for c in listed:
        cid = str(c.get("id"))
        board = boards.get(cid)
        rows.append(Competition(
            id=cid, name=str(c.get("name") or cid), status=str(c.get("status") or ""),
            enrolled=c.get("signUpDate") is not None, ongoing=is_ongoing(c, now), scoring=c.get("scoring"),
            start_date=c.get("startDate"), end_date=c.get("endDate"), sign_up_end_date=c.get("signUpEndDate"),
            faq=c.get("faq"), rank=board.get("rank") if isinstance(board, dict) else None,
            alphas=board.get("alphas") if isinstance(board, dict) else None, standing=isinstance(board, dict)))
    floor = datetime.min.replace(tzinfo=timezone.utc)
    live = sorted((r for r in rows if r.ongoing), key=lambda r: when(r.end_date) or floor)
    past = sorted((r for r in rows if not r.ongoing), key=lambda r: when(r.end_date) or floor, reverse=True)
    return live + past
