"""Catalogue sync jobs against a mocked BRAIN (respx) and a throwaway database."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import respx
from sqlalchemy import func, select

from alphago.brain import brain
from alphago.config import settings
from alphago.db import db
from alphago.db.models import DataField, Dataset, Job
from alphago.engine import sync
from alphago.jobs.queue import JobContext, jobs

API = "https://api.worldquantbrain.com"
SCOPE = {"region": "USA", "delay": 1, "universe": "TOP3000"}


@pytest.fixture
async def env(tmp_path, monkeypatch):
    # the app-wide write lock may be bound to an earlier test's event loop; these tests contend for it
    monkeypatch.setattr(db, "_write_lock", asyncio.Lock())
    await db.init(tmp_path / "sync.db")
    monkeypatch.setattr(brain.client, "state", "logged_in")
    monkeypatch.setattr(brain.client, "expiry_at", time.time() + 14400)
    yield
    await brain.client.close()
    await db.close()


async def _job_ctx() -> JobContext:
    async with db.write() as s:
        job = Job(kind="sync_scope", title="test", params=SCOPE)
        s.add(job)
        await s.flush()
        return JobContext(job.id, jobs)


async def _count(model) -> int:
    async with db.session() as s:
        return await s.scalar(select(func.count()).select_from(model)) or 0


@respx.mock
async def test_dataset_sync_reports_progress_without_deadlock(env) -> None:
    # Regression: progress updates used to wait on the write lock the sync itself held (hung at 25 datasets).
    datasets = [{"id": f"ds{i}", "name": f"Dataset {i}", "category": {"id": "pv", "name": "Price Volume"}}
                for i in range(60)]

    def handler(request: httpx.Request) -> httpx.Response:
        offset, limit = int(request.url.params["offset"]), int(request.url.params["limit"])
        return httpx.Response(200, json={"count": len(datasets), "results": datasets[offset:offset + limit]})

    respx.get(f"{API}/data-sets").mock(side_effect=handler)
    ctx = await _job_ctx()
    assert await asyncio.wait_for(sync.sync_datasets(ctx, SCOPE), 10) == "60 datasets"
    assert ctx.message == "50 datasets"
    assert await _count(Dataset) == 60


@respx.mock
async def test_field_sync_pages_concurrently_and_stores_every_field(env, monkeypatch) -> None:
    monkeypatch.setattr(settings.brain, "catalog_concurrency", 3)
    sizes = {"small": 7, "mid": 120, "big": 430, "empty": 0}
    async with db.write() as s:
        for ds in sizes:
            s.add(Dataset(id=ds, name=ds, **SCOPE))
    in_flight = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        p = request.url.params
        ds, offset, limit = p["dataset.id"], int(p["offset"]), int(p["limit"])
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        rows = [{"id": f"{ds}_f{i}", "dataset": {"id": ds}, "type": "MATRIX", "description": f"field {i}",
                 "category": {"id": "pv", "name": "Price Volume"}, "themes": ["t"], "userCount": i}
                for i in range(offset, min(offset + limit, sizes[ds]))]
        return httpx.Response(200, json={"count": sizes[ds], "results": rows})

    respx.get(f"{API}/data-fields").mock(side_effect=handler)
    ctx = await _job_ctx()
    assert await asyncio.wait_for(sync.sync_fields(ctx, SCOPE), 20) == "557 fields in 4 datasets"
    assert await _count(DataField) == 557 and ctx.progress == 1.0
    assert 1 < peak <= 3                                   # concurrent, but never above the setting
    async with db.session() as s:
        assert all(d.fields_synced_at is not None for d in await s.scalars(select(Dataset)))
        f = await s.scalar(select(DataField).where(DataField.id == "big_f429"))
        assert f.dataset_id == "big" and f.category == "Price Volume" and f.themes == ["t"] and f.user_count == 429
    # a second run updates in place
    assert await asyncio.wait_for(sync.sync_fields(ctx, SCOPE), 20) == "557 fields in 4 datasets"
    assert await _count(DataField) == 557
