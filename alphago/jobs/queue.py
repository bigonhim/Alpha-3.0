"""Persistent background jobs (sync, dataset analysis, checks, submissions, Osmosis pushes, ...).

A job is an async function registered under a `kind`. Jobs are rows in the `jobs` table so the Tasks page can show
history, progress and errors; running jobs are cancellable. Jobs interrupted by an app restart are marked failed.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from sqlalchemy import select, update

from ..db import db
from ..db.models import Job
from ..events import bus

log = logging.getLogger(__name__)


class JobContext:
    def __init__(self, job_id: int, queue: "JobQueue") -> None:
        self.job_id = job_id
        self._queue = queue
        self.progress = 0.0
        self.message = ""

    async def update(self, progress: float | None = None, message: str | None = None) -> None:
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self.message = message
        self._queue.version += 1
        async with db.write() as s:
            await s.execute(update(Job).where(Job.id == self.job_id).values(progress=self.progress,
                                                                             message=self.message[:2000]))


JobFn = Callable[[JobContext, dict[str, Any]], Awaitable[Any]]


class JobQueue:
    def __init__(self, concurrency: int = 3) -> None:
        self._handlers: dict[str, JobFn] = {}
        self._pending: asyncio.Queue[int] = asyncio.Queue()
        self._running: dict[int, asyncio.Task] = {}
        self._workers: list[asyncio.Task] = []
        self.concurrency = concurrency
        self.version = 0

    def register(self, kind: str) -> Callable[[JobFn], JobFn]:
        def deco(fn: JobFn) -> JobFn:
            self._handlers[kind] = fn
            return fn
        return deco

    async def start(self) -> None:
        async with db.write() as s:
            await s.execute(update(Job).where(Job.status == "running").values(
                status="failed", error="interrupted by app restart", finished_at=datetime.now(timezone.utc)))
            queued = list(await s.scalars(select(Job.id).where(Job.status == "queued").order_by(Job.priority, Job.id)))
        for jid in queued:
            self._pending.put_nowait(jid)
        self._workers = [asyncio.create_task(self._worker(i), name=f"job-worker-{i}") for i in range(self.concurrency)]

    async def stop(self) -> None:
        for task in self._workers + list(self._running.values()):
            task.cancel()

    async def submit(self, kind: str, title: str, params: dict[str, Any] | None = None, priority: int = 5) -> int:
        if kind not in self._handlers:
            raise KeyError(f"unknown job kind {kind}")
        async with db.write() as s:
            job = Job(kind=kind, title=title, params=params or {}, priority=priority)
            s.add(job)
            await s.flush()
            jid = job.id
        self._pending.put_nowait(jid)
        self.version += 1
        return jid

    async def cancel(self, job_id: int) -> None:
        task = self._running.get(job_id)
        if task:
            task.cancel()
        async with db.write() as s:
            await s.execute(update(Job).where(Job.id == job_id, Job.status.in_(["queued", "running"])).values(
                status="cancelled", finished_at=datetime.now(timezone.utc)))
        self.version += 1

    async def retry(self, job_id: int) -> int:
        async with db.session() as s:
            job = await s.get(Job, job_id)
        if job is None:
            raise KeyError(job_id)
        return await self.submit(job.kind, job.title, job.params, job.priority)

    def is_running(self, kind: str) -> bool:
        return any(not t.done() and t.get_name() == kind for t in self._running.values())

    async def _worker(self, idx: int) -> None:
        while True:
            jid = await self._pending.get()
            async with db.write() as s:
                job = await s.get(Job, jid)
                if job is None or job.status != "queued":
                    continue
                job.status = "running"
                job.started_at = datetime.now(timezone.utc)
                kind, params, title = job.kind, dict(job.params or {}), job.title
            self.version += 1
            handler = self._handlers.get(kind)
            ctx = JobContext(jid, self)
            task = asyncio.create_task(handler(ctx, params), name=kind)  # type: ignore[misc]
            self._running[jid] = task
            status, error, message = "done", None, None
            try:
                result = await task
                message = str(result)[:2000] if result is not None else ctx.message
            except asyncio.CancelledError:
                status, error = "cancelled", "cancelled"
            except Exception as exc:
                status, error = "failed", f"{getattr(exc, 'message', None) or exc}"
                log.error("job %s (%s) failed\n%s", jid, kind, traceback.format_exc())
                bus.publish("system", f"Task '{title}' failed: {error}", "error")
            finally:
                self._running.pop(jid, None)
            async with db.write() as s:
                await s.execute(update(Job).where(Job.id == jid).values(
                    status=status, error=error, message=message, progress=1.0 if status == "done" else ctx.progress,
                    finished_at=datetime.now(timezone.utc)))
            self.version += 1


jobs = JobQueue()
