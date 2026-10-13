"""Daily simulation quota, learned from the X-Ratelimit-* headers BRAIN returns on POST /simulations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
ASSUMED_DAILY_LIMIT = 5000  # placeholder until the first header arrives


def eastern_day(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), EASTERN).strftime("%Y-%m-%d")


def next_eastern_midnight(ts: float | None = None) -> float:
    now = datetime.fromtimestamp(ts or time.time(), EASTERN)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


@dataclass
class QuotaTracker:
    limit: int | None = None
    remaining: int | None = None
    reset_at: float | None = None       # epoch seconds
    updated_at: float | None = None
    day: str = field(default_factory=eastern_day)
    sent_today: int = 0                  # simulations this app submitted today (counted locally)

    def _roll_day(self) -> None:
        today = eastern_day()
        if today != self.day:
            self.day = today
            self.sent_today = 0
            if self.limit is not None:
                self.remaining = self.limit

    def update_from_headers(self, headers) -> bool:  # noqa: ANN001  (httpx.Headers or dict)
        """Returns True when the headers contained quota information."""
        lim = headers.get("X-Ratelimit-Limit") or headers.get("x-ratelimit-limit")
        rem = headers.get("X-Ratelimit-Remaining") or headers.get("x-ratelimit-remaining")
        rst = headers.get("X-Ratelimit-Reset") or headers.get("x-ratelimit-reset")
        if lim is None and rem is None:
            return False
        self._roll_day()
        now = time.time()
        try:
            if lim is not None:
                self.limit = int(float(lim))
            if rem is not None:
                self.remaining = int(float(rem))
            if rst is not None:
                value = float(rst)
                # either seconds-until-reset or an absolute epoch timestamp
                self.reset_at = value if value > 1e9 else now + value
        except ValueError:
            return False
        self.updated_at = now
        return True

    def record_sent(self, n: int) -> None:
        self._roll_day()
        self.sent_today += n
        if self.remaining is not None:
            self.remaining = max(0, self.remaining - n)

    @property
    def effective_limit(self) -> int:
        return self.limit or ASSUMED_DAILY_LIMIT

    @property
    def effective_remaining(self) -> int:
        self._roll_day()
        if self.remaining is not None:
            return self.remaining
        return max(0, self.effective_limit - self.sent_today)

    @property
    def seconds_to_reset(self) -> float:
        target = self.reset_at if self.reset_at and self.reset_at > time.time() else next_eastern_midnight()
        return max(0.0, target - time.time())

    def can_spend(self, n: int, reserve: int) -> bool:
        return self.effective_remaining - n >= reserve
