"""Audit log: every action the app took on your BRAIN account, who triggered it and what it cost.

Recorded at the BRAIN client: each final response to a request that changes something on BRAIN (POST, PATCH, PUT,
DELETE) plus each submission check. Reads (GET) are not recorded. The actor comes from a context variable:
"you" for a click in the app, "you (confirmed)" for a confirmed job (submit, Osmosis push, tags), "task:<id>" for a
research task, "batch:<id>" / "autopilot" / "assisted" for Autopilot runs, "optimizer" for repairs started from the
alpha page, "system" for sign-in and session keep-alive, "job:<kind>" for other background jobs.
The log is append-only: the app never edits or deletes entries.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import re
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from ..db import db
from ..db.models import AuditEntry

log = logging.getLogger(__name__)

ACTOR: ContextVar[str] = ContextVar("brain_actor", default="you")
_ALPHA_IN_PATH = re.compile(r"/alphas/([A-Za-z0-9]+)")
_pending: set[asyncio.Task] = set()


def classify(method: str, path: str) -> str | None:
    """Audit action for a request, or None when it is not recorded."""
    m, p = method.upper(), path.split("?")[0].rstrip("/")
    if m == "GET":
        return "check" if p.endswith("/check") else None
    if m == "OPTIONS":
        return None
    if p.endswith("/authentication"):
        return "sign_in" if m == "POST" else "sign_out" if m == "DELETE" else "session"
    if p.endswith("/persona") or "inquiries" in p:
        return "verification"
    if p == "/simulations" or p.endswith("/simulations"):
        return "simulate"
    if p.endswith("/submit"):
        return "submit"
    if p.startswith("/tags"):
        return "tag_list"
    if m == "PATCH" and _ALPHA_IN_PATH.search(p):
        return "properties"
    return m.lower()


def _cost_and_ids(action: str, path: str, body: Any) -> tuple[int, list[str], str | None]:
    ids = [m.group(1) for m in _ALPHA_IN_PATH.finditer(path)]
    detail = None
    cost = 0
    if action == "simulate":
        items = body if isinstance(body, list) else [body] if body else []
        cost = len(items)
        kinds = {str(i.get("type", "REGULAR")) for i in items if isinstance(i, dict)}
        detail = f"{cost} simulation(s) · {', '.join(sorted(kinds)) or 'REGULAR'}"
    elif action == "properties" and isinstance(body, dict):
        detail = "changed: " + ", ".join(sorted(body))
        if "osmosisPoints" in body:
            detail += f" (osmosis points {body['osmosisPoints']})"
    elif action == "tag_list" and isinstance(body, dict):
        ids = list(body.get("alphas") or [])
        detail = f"tag list {body.get('name')!r}"
    return cost, ids, detail


async def _write(entry: AuditEntry) -> None:
    try:
        async with db.write() as s:
            s.add(entry)
    except Exception:
        log.exception("audit write failed")


def observe(method: str, url: str, status: int, body: Any, final: bool, response: Any = None) -> None:
    """BrainClient observer: queue one audit entry per final, recordable response."""
    if not final:
        return
    path = url if url.startswith("/") else "/" + url.split("://", 1)[-1].split("/", 1)[-1]
    action = classify(method, path)
    if action is None:
        return
    if action == "check" and (status != 200 or (response is not None and "Retry-After" in response.headers)):
        return                                     # still computing: only the finished check is recorded
    cost, ids, detail = _cost_and_ids(action, path, body)
    if action == "simulate" and not (200 <= status < 300):
        cost = 0                                   # BRAIN did not accept it, nothing was spent
    entry = AuditEntry(actor=ACTOR.get()[:48], action=action, method=method.upper(), path=path[:256],
                       status=status or None, quota_cost=cost, alpha_ids=ids or None, detail=detail)
    try:
        task = asyncio.get_running_loop().create_task(_write(entry))
        _pending.add(task)
        task.add_done_callback(_pending.discard)
    except RuntimeError:
        pass


def install() -> None:
    from ..brain import brain
    if observe not in brain.client.observers:
        brain.client.observers.append(observe)


async def flush() -> None:
    if _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)


async def entries(*, days: int = 30, action: str | None = None, actor: str | None = None,
                  limit: int = 5000) -> list[AuditEntry]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = select(AuditEntry).where(AuditEntry.ts >= since)
    if action:
        stmt = stmt.where(AuditEntry.action == action)
    if actor:
        stmt = stmt.where(AuditEntry.actor.like(f"{actor}%"))
    async with db.session() as s:
        return list(await s.scalars(stmt.order_by(AuditEntry.id.desc()).limit(limit)))


async def summary(days: int = 30) -> dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    async with db.session() as s:
        by_actor = (await s.execute(select(AuditEntry.actor, func.sum(AuditEntry.quota_cost), func.count())
                                    .where(AuditEntry.ts >= since).group_by(AuditEntry.actor))).all()
        submits = await s.scalar(select(func.count(AuditEntry.id)).where(
            AuditEntry.ts >= since, AuditEntry.action == "submit", AuditEntry.method == "POST")) or 0
    return {"by_actor": [{"actor": a, "simulations": int(q or 0), "actions": n} for a, q, n in by_actor],
            "submissions": submits}


def to_csv(rows: list[AuditEntry]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time (UTC)", "actor", "action", "method", "path", "status", "simulations", "alphas", "detail"])
    for r in rows:
        w.writerow([r.ts.strftime("%Y-%m-%d %H:%M:%S") if r.ts else "", r.actor, r.action, r.method, r.path,
                    r.status, r.quota_cost, " ".join(r.alpha_ids or []), r.detail or ""])
    return buf.getvalue()
