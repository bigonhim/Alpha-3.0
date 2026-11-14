"""Registers this package's labs, job kinds and listeners (imported once at startup by alphago.extensions).

The AI package has no lab or job of its own: what it needs at startup is the key pool and the model registry in
memory, which `LLMRouter.restore_models` loads before this module is imported. Loading them again here is cheap and
covers a start-up path that skips the router (the key mirror must exist before any provider is asked if it is set
up)."""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)


async def _load() -> None:
    from .keys import keystore
    from .registry import registry
    try:
        await registry.load()
        await keystore.load()
    except Exception:
        log.exception("AI key pool / model registry could not be loaded")


def _schedule() -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_load(), name="ai-load")


_schedule()
