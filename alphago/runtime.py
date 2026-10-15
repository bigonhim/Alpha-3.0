"""Application lifecycle: database, BRAIN session keep-alive, background services."""

from __future__ import annotations

import asyncio
import logging

from .brain import brain
from .brain.errors import AuthError, BrainError, PersonaRequired
from .brain.options import SimOptions
from .credentials import brain_credentials
from .db import db
from .events import Event, bus

log = logging.getLogger(__name__)

_tasks: list[asyncio.Task] = []
started = asyncio.Event()


async def _keepalive() -> None:
    """Keep the BRAIN session alive: refresh expiry every 5 min, re-login before it lapses."""
    from .config import settings
    while True:
        try:
            client = brain.client
            if client.state == "logged_in":
                left = await client.refresh_session()
                if 0 < left < settings.brain.relogin_threshold_s and brain_credentials():
                    await client.login()
                elif left == 0 and brain_credentials():
                    await client.login()
        except PersonaRequired:
            bus.publish("sync", "BRAIN asks for biometric verification — open 'Sync with BRAIN' to continue.", "warning")
        except (AuthError, BrainError) as exc:
            log.warning("keepalive: %s", exc)
        except Exception:
            log.exception("keepalive failed")
        await asyncio.sleep(300)


async def try_auto_login() -> None:
    if not brain_credentials():
        return
    try:
        await brain.client.login()
        bus.publish("sync", "Signed in to WorldQuant BRAIN.", "success")
        await load_options_quietly()
    except PersonaRequired:
        bus.publish("sync", "BRAIN requires biometric verification — open 'Sync with BRAIN' and click Verify.", "warning")
    except (AuthError, BrainError) as exc:
        bus.publish("sync", f"Automatic BRAIN login failed: {exc.message}", "error")
    except Exception as exc:  # network down etc.
        bus.publish("sync", f"Automatic BRAIN login failed: {exc}", "error")


async def load_options_quietly() -> None:
    try:
        opts = await brain.load_options()
        await db.kv_set("sim_options", _options_to_json(opts))
    except BrainError as exc:
        log.warning("OPTIONS /simulations failed: %s", exc)


def _options_to_json(opts: SimOptions) -> dict:
    return {"regions": opts.regions, "per_region": opts.per_region, "global_choices": opts.global_choices,
            "ranges": {k: list(v) for k, v in opts.ranges.items()}, "types": opts.types}


async def _restore_options() -> None:
    data = await db.kv_get("sim_options")
    if data:
        brain.options = SimOptions(regions=data["regions"], per_region=data["per_region"],
                                   global_choices=data["global_choices"],
                                   ranges={k: tuple(v) for k, v in data.get("ranges", {}).items()},
                                   types=data.get("types", ["REGULAR", "SUPER"]), live=True)


def _quiet_connection_resets(loop: asyncio.AbstractEventLoop) -> None:
    """Windows' proactor loop logs a traceback whenever a browser/webview socket drops; it is harmless."""
    previous = loop.get_exception_handler()

    def handler(lp: asyncio.AbstractEventLoop, context: dict) -> None:
        if isinstance(context.get("exception"), ConnectionResetError):
            return
        if previous is not None:
            previous(lp, context)
        else:
            lp.default_exception_handler(context)

    loop.set_exception_handler(handler)


async def _close_stale_runs() -> None:
    """Autopilot runs/batches left 'running' by a previous session are marked halted (their loop is gone)."""
    from datetime import datetime, timezone

    from sqlalchemy import update

    from .db.models import AutopilotRun, Batch
    now = datetime.now(timezone.utc)
    async with db.write() as s:
        await s.execute(update(AutopilotRun).where(AutopilotRun.status == "running").values(
            status="stopped", reason="app closed", stopped_at=now))
        await s.execute(update(Batch).where(Batch.source == "autopilot",
                                            Batch.status.in_(["pending", "awaiting_approval", "running"]))
                        .values(status="halted", finished_at=now))


async def startup() -> None:
    from .config import log_dir
    from .engine.simulator import simulator
    from .jobs.queue import jobs

    logging.getLogger().setLevel(logging.INFO)
    handler = logging.FileHandler(log_dir() / "alphago.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    _quiet_connection_resets(asyncio.get_running_loop())

    await db.init()
    bus.set_persister(db.persist_event)
    rows = await db.recent_events(100)
    bus.preload([Event(source=r.source, message=r.message, level=r.level, payload=r.payload or {}, ts=r.ts)
                 for r in rows])
    await brain.client.start()
    await _restore_options()
    from .llm import llm
    llm.configure()
    await llm.restore_models()
    from .engine import sync  # noqa: F401  (registers job kinds)
    from . import services  # noqa: F401  (registers remaining job kinds / listeners)
    await simulator.start()
    from .engine.research import daily_scheduler, research
    await research.restore()                # tasks interrupted by a restart wait for the user
    await _close_stale_runs()
    await jobs.start()
    _tasks.append(asyncio.create_task(try_auto_login(), name="auto-login"))
    _tasks.append(asyncio.create_task(_keepalive(), name="keepalive"))
    _tasks.append(asyncio.create_task(daily_scheduler(), name="daily-scheduler"))
    started.set()
    log.info("AlphaGo started")


async def shutdown() -> None:
    from .engine.simulator import simulator
    from .jobs.queue import jobs
    for task in _tasks:
        task.cancel()
    await jobs.stop()
    await simulator.stop()
    await brain.client.close()
    await db.close()
