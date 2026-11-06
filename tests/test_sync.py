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
    # a second run resumes: every dataset was just synced, so nothing is fetched again
    assert (await asyncio.wait_for(sync.sync_fields(ctx, SCOPE), 20)
            == "0 fields in 0 datasets; 4 recently synced dataset(s) skipped")
    # a full run updates in place
    assert await asyncio.wait_for(sync.sync_fields(ctx, {**SCOPE, "full": True}), 20) == "557 fields in 4 datasets"
    assert await _count(DataField) == 557


@respx.mock
async def test_field_sync_survives_rate_limits(env, monkeypatch) -> None:
    """Regression: GET /data-fields -> 429 "API rate limit exceeded" failed the whole sync. Now the client waits out a
    shared cooldown, slows the catalogue pace and retries; the sync finishes with every field."""
    from alphago.brain.ratelimit import RateGate
    monkeypatch.setattr(brain.client, "gate", RateGate(50.0))
    monkeypatch.setattr(settings.brain, "catalog_concurrency", 2)
    async with db.write() as s:
        for ds in ("a", "b"):
            s.add(Dataset(id=ds, name=ds, **SCOPE))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] in (2, 3):              # BRAIN throttles two requests in a row
            return httpx.Response(429, headers={"Retry-After": "0.05"}, json={"detail": "API rate limit exceeded"})
        p = request.url.params
        ds, offset = p["dataset.id"], int(p["offset"])
        rows = [{"id": f"{ds}_f{i}", "dataset": {"id": ds}, "type": "MATRIX"} for i in range(offset, min(offset + 50, 120))]
        return httpx.Response(200, json={"count": 120, "results": rows})

    respx.get(f"{API}/data-fields").mock(side_effect=handler)
    ctx = await _job_ctx()
    assert await asyncio.wait_for(sync.sync_fields(ctx, SCOPE), 20) == "240 fields in 2 datasets"
    assert await _count(DataField) == 240
    snap = brain.client.gate.snapshot()["catalog"]
    assert snap["throttled"] == 2 and snap["rate"] < 50.0          # the pace was halved


@respx.mock
async def test_field_sync_keeps_going_when_one_dataset_fails(env) -> None:
    async with db.write() as s:
        for ds in ("good", "bad"):
            s.add(Dataset(id=ds, name=ds, **SCOPE))

    def handler(request: httpx.Request) -> httpx.Response:
        ds = request.url.params["dataset.id"]
        if ds == "bad":
            return httpx.Response(403, json={"detail": "no access"})
        return httpx.Response(200, json={"count": 3, "results": [{"id": f"g{i}", "dataset": {"id": ds}} for i in range(3)]})

    respx.get(f"{API}/data-fields").mock(side_effect=handler)
    ctx = await _job_ctx()
    out = await asyncio.wait_for(sync.sync_fields(ctx, SCOPE), 20)
    assert out == "3 fields in 1 datasets, 1 failed" and await _count(DataField) == 3


def test_rate_gate_halves_on_429_and_recovers() -> None:
    from alphago.brain.ratelimit import RateGate, bucket_for
    g = RateGate(4.0)
    assert bucket_for("/data-fields?x=1") == "catalog" and bucket_for("/simulations") == "default"
    assert bucket_for("https://api.worldquantbrain.com/data-sets") == "catalog"
    g.throttled("catalog", 1.0)
    assert g.buckets["catalog"].rate == 2.0 and g.cooldown_left("catalog") > 0.5
    for _ in range(RateGate.RECOVER_AFTER):
        g.succeeded("catalog")
    assert 2.0 < g.buckets["catalog"].rate <= 4.0
