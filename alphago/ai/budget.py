"""What each key has spent (ported from alpha-harness `llm/budget.py`).

No provider publishes a remaining-quota endpoint, so the only way to rotate keys sensibly is to count locally,
against the limits the user set up. Two windows, kept differently because they behave differently:

* **Requests per minute** is a sliding sixty-second window, kept in memory: losing it on a restart costs at most one
  minute of over-caution.
* **Requests per day** is a calendar day and survives a restart (`AIKeyUsage`), or a spent key looks fresh and every
  rotation walks into a 429.

**The day boundary is the model's own.** Google starts a day at midnight Pacific, OpenRouter at midnight UTC, so each
model carries the time zone its user named. Accounting is conservative — a request is refused here when it *would*
exceed a limit, and one still in flight already counts — because a local refusal can name another key or model, and
a 429 cannot.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from datetime import time as clock_time
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from ..db import db
from ..db.models import utcnow
from ..db.models_ai import AIKeyUsage
from .registry import ModelInfo

WINDOW_SECONDS = 60.0
#: The usage row that counts a key's requests to every model, for its own daily cap.
ALL_MODELS = "*"
#: Above this many requests a minute a window is not filled on a 429; the pair is cooled for a minute instead.
_FILLABLE_RPM = 10_000


def quota_day(tz: tzinfo, moment: datetime) -> str:
    """The day a moment falls in on `tz`'s clock, as YYYY-MM-DD."""
    return moment.astimezone(tz).strftime("%Y-%m-%d")


def seconds_until_reset(tz: tzinfo, moment: datetime) -> float:
    """Until midnight on `tz`'s clock, built from the next local date so 23- and 25-hour days come out right."""
    now = moment.astimezone(tz)
    midnight = datetime.combine(now.date() + timedelta(days=1), clock_time.min, tzinfo=tz)
    return max(0.0, (midnight.astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds())


@dataclass(slots=True)
class Window:
    """A sliding sixty-second record of requests."""

    events: deque[float] = field(default_factory=deque)

    def trim(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        while self.events and self.events[0] < cutoff:
            self.events.popleft()

    def add(self, now: float) -> None:
        self.events.append(now)
        self.trim(now)

    def requests(self, now: float) -> int:
        self.trim(now)
        return len(self.events)

    def next_free(self, now: float) -> float:
        """Seconds until the oldest request ages out of the window."""
        self.trim(now)
        if not self.events:
            return 0.0
        return max(0.0, WINDOW_SECONDS - (now - self.events[0]))


@dataclass(slots=True)
class Headroom:
    """What is left on one (key, model) pair, and what is blocking it."""

    key_id: int
    model: str
    requests_today: int
    requests_per_day: int
    requests_this_minute: int
    requests_per_minute: int
    blocked_by: str | None = None          # requests_per_day | requests_per_minute | None
    retry_after: float = 0.0               # large for a daily block, small for a per-minute one

    @property
    def available(self) -> bool:
        return self.blocked_by is None

    @property
    def daily_remaining(self) -> int:
        return max(0, self.requests_per_day - self.requests_today)


class Ledger:
    """Per-key, per-model usage accounting. `clock` (monotonic seconds) and `now` (aware UTC datetime) are injectable
    so tests can move time."""

    def __init__(self, clock: Callable[[], float] | None = None, now: Callable[[], datetime] | None = None) -> None:
        self.clock = clock or time.monotonic
        self.now = now or utcnow
        self._windows: dict[tuple[int, str], Window] = {}
        self._daily: dict[tuple[int, str, str], int] = {}          # read through from the database, kept in step
        self._in_flight: Counter[tuple[int, str]] = Counter()       # sent, not yet settled
        self._spent: set[tuple[int, str, str]] = set()               # the provider said: spent for the day
        self._cool_until: dict[tuple[int, str], float] = {}          # minute penalty of an unlimited model

    def reset(self) -> None:
        """Forget everything held in memory (a new database was opened)."""
        self.__init__(self.clock, self.now)  # type: ignore[misc]

    def forget(self, key_id: int) -> None:
        """Drop a removed key's counts."""
        self._windows = {k: v for k, v in self._windows.items() if k[0] != key_id}
        self._daily = {k: v for k, v in self._daily.items() if k[0] != key_id}
        self._spent = {k for k in self._spent if k[0] != key_id}
        self._cool_until = {k: v for k, v in self._cool_until.items() if k[0] != key_id}
        for k in [k for k in self._in_flight if k[0] == key_id]:
            del self._in_flight[k]

    # -- clocks ----------------------------------------------------------

    def model_day(self, model: ModelInfo) -> str:
        return quota_day(ZoneInfo(model.reset_timezone), self.now())

    def model_reset(self, model: ModelInfo) -> float:
        """Seconds until the model's provider starts a new day."""
        return seconds_until_reset(ZoneInfo(model.reset_timezone), self.now())

    def _window(self, key_id: int, model: str) -> Window:
        return self._windows.setdefault((key_id, model), Window())

    async def _count(self, key_id: int, model_id: str, day: str) -> int:
        cached = self._daily.get((key_id, model_id, day))
        if cached is not None:
            return cached
        async with db.session() as s:
            row = await s.scalar(select(AIKeyUsage).where(AIKeyUsage.key_id == key_id, AIKeyUsage.model == model_id,
                                                          AIKeyUsage.day == day))
        count = int(row.requests) if row else 0
        self._daily[(key_id, model_id, day)] = count
        return count

    # -- reading ---------------------------------------------------------

    async def headroom(self, key_id: int, model: ModelInfo, *, cap: int | None = None) -> Headroom:
        """What is left, and the first limit that would stop the next request. `cap` is the key's own daily limit
        over every model; where both apply, the one closer to running out stops."""
        now = self.clock()
        window = self._window(key_id, model.id)
        day = self.model_day(model)
        today = await self._count(key_id, model.id, day) + self._in_flight[(key_id, model.id)]
        daily = model.rpd
        if cap is not None:
            spent = await self._count(key_id, ALL_MODELS, day) + sum(
                n for (key, _), n in self._in_flight.items() if key == key_id)
            if cap - spent < daily - today:
                today, daily = spent, cap
        if (key_id, model.id, day) in self._spent:
            today = max(today, daily)
        state = Headroom(key_id=key_id, model=model.id, requests_today=today, requests_per_day=daily,
                         requests_this_minute=window.requests(now), requests_per_minute=model.rpm)
        cooling = self._cool_until.get((key_id, model.id), 0.0) - now
        if today >= daily:                       # daily first: it cannot be waited out in any useful sense
            state.blocked_by = "requests_per_day"
            state.retry_after = self.model_reset(model)
        elif state.requests_this_minute >= model.rpm:
            state.blocked_by = "requests_per_minute"
            state.retry_after = window.next_free(now)
        elif cooling > 0:
            state.blocked_by = "requests_per_minute"
            state.retry_after = cooling
        return state

    def in_flight(self, key_id: int, model_id: str) -> int:
        return self._in_flight[(key_id, model_id)]

    # -- writing ---------------------------------------------------------

    def sent(self, key_id: int, model: ModelInfo) -> None:
        """Count a request against the minute as it leaves (one in flight already counts at the provider), and
        against the day until `settled`."""
        self._window(key_id, model.id).add(self.clock())
        self._in_flight[(key_id, model.id)] += 1

    def settled(self, key_id: int, model: ModelInfo) -> None:
        """The request `sent` counted has been answered and recorded, or has failed."""
        self._in_flight[(key_id, model.id)] = max(0, self._in_flight[(key_id, model.id)] - 1)

    async def record(self, key_id: int, model: ModelInfo, tokens: int) -> None:
        """Count an answered (or possibly billed) request against the day, with its tokens; atomically, so calls
        finishing together cannot lose increments."""
        day, spent, at = self.model_day(model), max(0, int(tokens or 0)), self.now()
        counted: dict[str, int] = {}
        async with db.write() as s:
            for model_id in (model.id, ALL_MODELS):
                stmt = (insert(AIKeyUsage)
                        .values(key_id=key_id, model=model_id, day=day, requests=1, tokens=spent, last_request_at=at)
                        .on_conflict_do_update(index_elements=["key_id", "model", "day"],
                                               set_={"requests": AIKeyUsage.requests + 1,
                                                     "tokens": AIKeyUsage.tokens + spent, "last_request_at": at})
                        .returning(AIKeyUsage.requests))
                counted[model_id] = int((await s.execute(stmt)).scalar_one())
        for model_id, requests in counted.items():
            self._daily[(key_id, model_id, day)] = requests
        for stale in [k for k in self._daily if k[0] == key_id and k[1] in (model.id, ALL_MODELS) and k[2] != day]:
            del self._daily[stale]

    def penalise(self, key_id: int, model: ModelInfo, *, daily: bool) -> None:
        """Believe the provider over our own arithmetic: a 429 means the local count was wrong, so this pair is
        treated as spent at once instead of being retried into. The request counts stay the real ones."""
        if daily:
            day = self.model_day(model)
            self._spent = {k for k in self._spent if k[:2] != (key_id, model.id)}
            self._spent.add((key_id, model.id, day))
            return
        now = self.clock()
        if model.rpm > _FILLABLE_RPM:
            self._cool_until[(key_id, model.id)] = now + WINDOW_SECONDS
            return
        window = self._window(key_id, model.id)
        while window.requests(now) < model.rpm:
            window.add(now)

    def minute(self, key_id: int, model_id: str) -> int:
        return self._window(key_id, model_id).requests(self.clock())

    async def usage(self, key_families: Mapping[int, str], today_of: Callable[[str, str], str]
                    ) -> list[dict[str, Any]]:
        """Today's spend per key and model, each "today" on that model's clock: `today_of(family, model_id)` names
        the day (usage rows name only their key and model id, and ids repeat across providers)."""
        if not key_families:
            return []
        async with db.session() as s:
            rows = list((await s.scalars(select(AIKeyUsage).where(AIKeyUsage.key_id.in_(list(key_families)))
                                         )).all())
        return [{"key_id": r.key_id, "model": r.model, "day": r.day, "requests": r.requests, "tokens": r.tokens,
                 "last_request_at": r.last_request_at}
                for r in rows if r.key_id in key_families and today_of(key_families[r.key_id], r.model) == r.day]
