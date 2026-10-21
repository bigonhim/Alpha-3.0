"""Email alerts outside the app: something is ready, stuck or failing.

Sent by SMTP (any server; e.g. Gmail with an app password). The password lives in Windows Credential Manager. Messages
carry alpha IDs and counts only — never expressions or credentials. The same trigger is sent at most once per
`quiet_minutes`; anything suppressed in between is counted in the next message.

Triggers: ready (an alpha passed BRAIN's check) · task_finished · stuck (simulations queued but nothing finished for
`stuck_minutes`) · quota (the queue paused at the daily quota or reserve) · persona (BRAIN asks for biometric
verification) · login_failed · llm_failing (failover used, or the last calls all failed) · daily_summary.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
import time
from datetime import datetime
from email.message import EmailMessage
from typing import Any

from sqlalchemy import select

from ..config import settings
from ..credentials import _get, _set
from ..db import db
from ..db.models import LLMCall
from ..events import Event, bus

log = logging.getLogger(__name__)

PASSWORD_KEY = "smtp_password"
TRIGGERS = {"ready": "An alpha passed BRAIN's submission check", "task_finished": "A research task finished",
            "stuck": "The simulation queue is stuck", "quota": "The queue paused at the daily quota",
            "persona": "BRAIN asks for biometric verification", "login_failed": "Signing in to BRAIN failed",
            "llm_failing": "The AI provider keeps failing", "daily_summary": "Daily summary"}

_last_sent: dict[str, float] = {}
_suppressed: dict[str, int] = {}
_last_progress = time.monotonic()
_tasks: set[asyncio.Task] = set()


def smtp_password() -> str | None:
    return _get(PASSWORD_KEY)


def set_smtp_password(value: str | None) -> None:
    _set(PASSWORD_KEY, value)


def configured() -> bool:
    a = settings.alerts
    return bool(a.enabled and a.smtp_host and a.recipient)


def _send_sync(subject: str, body: str) -> None:
    a = settings.alerts
    msg = EmailMessage()
    msg["Subject"] = f"[AlphaGo] {subject}"
    msg["From"] = a.sender or a.smtp_user or a.recipient
    msg["To"] = a.recipient
    msg.set_content(body)
    context = ssl.create_default_context()
    if a.security == "ssl":
        with smtplib.SMTP_SSL(a.smtp_host, a.smtp_port, timeout=30, context=context) as server:
            if a.smtp_user:
                server.login(a.smtp_user, smtp_password() or "")
            server.send_message(msg)
        return
    with smtplib.SMTP(a.smtp_host, a.smtp_port, timeout=30) as server:
        if a.security == "starttls":
            server.starttls(context=context)
        if a.smtp_user:
            server.login(a.smtp_user, smtp_password() or "")
        server.send_message(msg)


async def send(subject: str, body: str) -> None:
    """Send one email now (the 'Send test' button uses this directly)."""
    await asyncio.to_thread(_send_sync, subject, body)


async def notify(trigger: str, subject: str, body: str, *, force: bool = False) -> bool:
    """Send an alert for a trigger unless it is switched off or was sent within the quiet period."""
    a = settings.alerts
    if not configured() or not a.triggers.get(trigger, True):
        return False
    now = time.monotonic()
    if not force and now - _last_sent.get(trigger, -1e9) < a.quiet_minutes * 60:
        _suppressed[trigger] = _suppressed.get(trigger, 0) + 1
        return False
    extra = _suppressed.pop(trigger, 0)
    if extra:
        body += f"\n\n(+{extra} similar alert(s) held back in the last {a.quiet_minutes} minutes)"
    _last_sent[trigger] = now
    try:
        await send(subject, body + f"\n\n— AlphaGo, {datetime.now():%Y-%m-%d %H:%M}")
        return True
    except Exception as exc:
        log.warning("alert email failed: %s", exc)
        return False


def trigger_for(ev: Event) -> str | None:
    """Which alert an activity event stands for (None: not alert-worthy)."""
    kind = (ev.payload or {}).get("kind")
    msg = ev.message.lower()
    if kind == "ready" or (ev.source == "check" and "passed every submission check" in msg):
        return "ready"
    if kind == "task_finished":
        return "task_finished"
    if kind == "llm_failover":
        return "llm_failing"
    if "biometric verification" in msg and ev.level == "warning":
        return "persona"
    if "login failed" in msg and ev.level == "error":
        return "login_failed"
    if ev.source == "simulate" and ev.level == "warning" and ("paused" in msg and ("quota" in msg
                                                                                  or "simulation limit" in msg)):
        return "quota"
    return None


def _on_event(ev: Event) -> None:
    trigger = trigger_for(ev)
    if trigger is None or not configured():
        return
    try:
        task = asyncio.get_running_loop().create_task(notify(trigger, TRIGGERS[trigger], ev.message))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    except RuntimeError:
        pass


def _on_simulated(cand: Any, alpha: Any) -> None:
    global _last_progress
    _last_progress = time.monotonic()


async def _llm_failing(n: int = 5) -> bool:
    async with db.session() as s:
        rows = list(await s.scalars(select(LLMCall.status).order_by(LLMCall.id.desc()).limit(n)))
    return len(rows) == n and all(r != "ok" for r in rows)


async def daily_summary_text() -> str:
    from .stats import dashboard_numbers, pipeline_today
    n = await dashboard_numbers()
    p = await pipeline_today()
    return (f"Submittable now: {n['submittable']} ({n['ready_robust']} robust, {n['ready_fragile']} fragile)\n"
            f"Strong candidates: {n['strong']} · need a re-check: {n['recheck']}\n"
            f"Simulated today: {n['simulated_today']} · simulations left: {n['sims_left']}\n"
            f"Today's pipeline: " + ", ".join(f"{k} {v}" for k, v in p.items() if v))


async def watchdog() -> None:
    """Every minute: queue stuck? AI calls all failing? time for the daily summary?"""
    from ..brain import brain
    from .simulator import simulator
    last_summary_day = None
    while True:
        await asyncio.sleep(60)
        try:
            if not configured():
                continue
            a = settings.alerts
            idle = time.monotonic() - _last_progress
            if (simulator.queued_count or simulator.running_count) and not simulator.paused \
                    and brain.client.logged_in and idle > a.stuck_minutes * 60:
                await notify("stuck", TRIGGERS["stuck"],
                             f"{simulator.queued_count} simulation(s) queued, {simulator.running_count} running, "
                             f"but none finished for {idle / 60:.0f} minutes.")
            if await _llm_failing():
                await notify("llm_failing", TRIGGERS["llm_failing"],
                             "The last 5 AI calls all failed — see the Health page.")
            now = datetime.now()
            if now.hour == a.summary_hour and last_summary_day != now.date():
                last_summary_day = now.date()
                await notify("daily_summary", TRIGGERS["daily_summary"], await daily_summary_text(), force=True)
        except Exception:
            log.exception("alert watchdog failed")


def install() -> None:
    from .simulator import simulator
    if _on_event not in bus._listeners:
        bus.subscribe(_on_event)
    if _on_simulated not in simulator._listeners:
        simulator.add_listener(_on_simulated)
