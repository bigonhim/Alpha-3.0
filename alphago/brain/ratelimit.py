"""Client-side pacing for the BRAIN API.

BRAIN answers `429 API rate limit exceeded` when an account sends too many requests in a short time. It publishes no
numbers, so the gate learns them (AIMD): every bucket starts at a configured rate, halves it on each 429 and creeps back
up after a run of successes. A 429 also opens a cooldown for its bucket: every request in that bucket waits it out
instead of hammering BRAIN with retries (four workers retrying in parallel kept the limit tripped during a field sync).

Buckets: "catalog" (data-sets, data-fields, operators, categories: the bulk listing calls) is paced; "default" (every
other call, including simulation polling) is not paced but still honours a cooldown after a rate-limit answer.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

CATALOG_PREFIXES = ("/data-fields", "/data-sets", "/operators", "/data-categories")


def bucket_for(url: str) -> str:
    path = str(url)
    if "://" in path:
        path = "/" + path.split("://", 1)[1].split("/", 1)[-1]
    return "catalog" if path.startswith(CATALOG_PREFIXES) else "default"


@dataclass
class _Bucket:
    rate: float                       # requests per second; 0 = unpaced
    max_rate: float
    min_rate: float = 0.2
    next_slot: float = 0.0
    cooldown_until: float = 0.0
    strikes: int = 0                  # 429s in a row (sets the cooldown length when BRAIN gives no Retry-After)
    ok_streak: int = 0
    throttled_total: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RateGate:
    RECOVER_AFTER = 25                # successes in a row before the rate goes up again
    RECOVER_FACTOR = 1.15

    def __init__(self, catalog_rps: float = 2.0) -> None:
        self.buckets: dict[str, _Bucket] = {
            "catalog": _Bucket(rate=max(0.0, catalog_rps), max_rate=max(0.0, catalog_rps)),
            "default": _Bucket(rate=0.0, max_rate=0.0),
        }

    def bucket(self, name: str) -> _Bucket:
        return self.buckets.get(name) or self.buckets["default"]

    def cooldown_left(self, name: str = "catalog") -> float:
        return max(0.0, self.bucket(name).cooldown_until - time.monotonic())

    async def acquire(self, name: str) -> None:
        b = self.bucket(name)
        async with b.lock:                # one waiter at a time: the slots are handed out in order
            now = time.monotonic()
            wait = max(b.cooldown_until - now, (b.next_slot - now) if b.rate > 0 else 0.0)
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            if b.rate > 0:
                b.next_slot = max(now, b.next_slot) + 1.0 / b.rate

    def throttled(self, name: str, retry_after: float | None = None) -> float:
        """Record a 429; returns the cooldown (seconds) every request in the bucket now waits."""
        b = self.bucket(name)
        b.strikes += 1
        b.ok_streak = 0
        b.throttled_total += 1
        if b.rate > 0:
            b.rate = max(b.min_rate, b.rate * 0.5)
        wait = retry_after if retry_after and retry_after > 0 else min(120.0, 5.0 * 2 ** (b.strikes - 1))
        b.cooldown_until = max(b.cooldown_until, time.monotonic() + wait)
        return wait

    def succeeded(self, name: str) -> None:
        b = self.bucket(name)
        b.strikes = 0
        b.ok_streak += 1
        if b.rate > 0 and b.ok_streak >= self.RECOVER_AFTER and b.rate < b.max_rate:
            b.rate = min(b.max_rate, b.rate * self.RECOVER_FACTOR)
            b.ok_streak = 0

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {n: {"rate": round(b.rate, 3), "cooldown_s": round(max(0.0, b.cooldown_until - time.monotonic()), 1),
                    "throttled": b.throttled_total} for n, b in self.buckets.items()}
