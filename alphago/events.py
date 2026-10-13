"""In-process event bus. Feeds the dashboard's "Recent AI activity" and persists events to the database."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

Persister = Callable[["Event"], Awaitable[None]]


@dataclass
class Event:
    source: str           # autopilot | generate | simulate | analyse | optimize | check | submit | sync | llm | system
    message: str
    level: str = "info"   # info | success | warning | error
    payload: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EventBus:
    def __init__(self, maxlen: int = 500) -> None:
        self._recent: deque[Event] = deque(maxlen=maxlen)
        self._listeners: list[Callable[[Event], None]] = []
        self._persister: Persister | None = None
        self.version = 0  # bumps on every publish; UI timers poll this cheaply

    def set_persister(self, persister: Persister) -> None:
        self._persister = persister

    def subscribe(self, listener: Callable[[Event], None]) -> None:
        self._listeners.append(listener)

    def publish(self, source: str, message: str, level: str = "info", **payload: Any) -> Event:
        event = Event(source=source, message=message, level=level, payload=payload)
        self._recent.appendleft(event)
        self.version += 1
        log.info("[%s] %s", source, message)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                log.exception("event listener failed")
        if self._persister is not None:
            try:
                asyncio.get_running_loop().create_task(self._persist(event))
            except RuntimeError:
                pass  # no loop (tests / shutdown)
        return event

    async def _persist(self, event: Event) -> None:
        try:
            await self._persister(event)  # type: ignore[misc]
        except Exception:
            log.exception("failed to persist event")

    def preload(self, events: list[Event]) -> None:
        """Seed the in-memory feed from the database at startup (newest first)."""
        for event in reversed(events):
            self._recent.appendleft(event)

    def recent(self, limit: int = 50) -> list[Event]:
        return list(self._recent)[:limit]


bus = EventBus()
