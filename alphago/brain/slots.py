"""Adaptive concurrency control for in-flight (multi-)simulations.

BRAIN limits concurrent simulations per account (consultants: 8 multi-simulations). We start at the configured
capacity, halve on a concurrency rejection, and creep back up one slot per minute without rejections.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator


class SlotManager:
    RECOVER_EVERY_S = 60.0

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self.limit = self.capacity
        self.in_use = 0
        self.waiting = 0
        self._cond = asyncio.Condition()
        self._recover_at = 0.0
        self.busy_streak = 0

    def resize(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self.limit = min(self.limit, self.capacity) if self.limit else self.capacity

    def on_throttle(self) -> None:
        # the rejected request already holds a slot, so the account's real limit is below in_use
        self.limit = max(1, min(self.limit - 1, self.in_use - 1))
        self._recover_at = time.time() + self.RECOVER_EVERY_S

    def _maybe_recover(self) -> None:
        if self.limit < self.capacity and time.time() >= self._recover_at:
            self.limit += 1
            self._recover_at = time.time() + self.RECOVER_EVERY_S

    def on_accepted(self) -> None:
        """A start went through: the concurrency backoff (if any) is over."""
        self.busy_streak = 0

    @asynccontextmanager
    async def slot(self, cost: int = 1) -> AsyncIterator[None]:
        """Hold `cost` slots (a GLB run takes 2) while a simulation is in flight. A run larger than the current
        limit still goes when nothing else is in flight, so it can never wait forever."""
        cost = max(1, min(int(cost), self.capacity))
        async with self._cond:
            self.waiting += 1
            try:
                while True:
                    self._maybe_recover()
                    if self.in_use + cost <= self.limit or self.in_use == 0:
                        break
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=5.0)
                    except TimeoutError:
                        pass
            finally:
                self.waiting -= 1
            self.in_use += cost
        try:
            yield
        finally:
            async with self._cond:
                self.in_use -= cost
                self._cond.notify_all()
