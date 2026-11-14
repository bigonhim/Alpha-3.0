"""Portfolio: members' facets and Selection Expression, the pyramid grid, the computed book, Production Correlation
with BRAIN's hourly limit, and smoke tests of the Alpha Pool and Portfolio pages."""

from __future__ import annotations

import asyncio
import time
from datetime import date, timedelta

import httpx
import numpy as np
import pytest
import respx

from alphago.brain import brain
from alphago.db import db
from alphago.db.models import Alpha, AlphaPnL, Job
from alphago.pool import brainops
from alphago.pool import portfolio as pf

API = "https://api.worldquantbrain.com"


def weekdays(end: date, n: int) -> list[str]:
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= timedelta(days=1)
    return out[::-1]


DATES = weekdays(date(2025, 12, 31), 600)


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_write_lock", asyncio.Lock())
    await db.init(tmp_path / "pf.db")
    monkeypatch.setattr(brain.client, "state", "logged_in")
    monkeypatch.setattr(brain.client, "expiry_at", time.time() + 14400)
    yield
    await brain.client.close()
    await db.close()


async def seed() -> None:
    rng = np.random.default_rng(3)
    async with db.write() as s:
        s.add(Alpha(id="U1", status="ACTIVE", region="USA", delay=1, sharpe=1.6, returns=0.12, drawdown=0.05,
                    turnover=0.1, test_start="2025-01-02", max_trade="ON",
                    classifications=[{"id": "X", "name": "Power Pool Alpha"}],
                    pyramids=[{"name": "USA/D1/PV"}, {"name": "USA/D1/MODEL"}]))
        s.add(Alpha(id="U2", status="ACTIVE", region="USA", delay=1, sharpe=1.3, returns=0.05, drawdown=0.08,
                    turnover=0.2, test_start="2025-01-02", pyramids=[{"name": "USA/D1/PV"}]))
        s.add(Alpha(id="E1", status="ACTIVE", region="EUR", delay=0, sharpe=None, sim_type="RA_CHILD",
                    pyramids=[{"name": "EUR/D0/PV"}]))
        for aid in ("U1", "U2"):
            s.add(AlphaPnL(alpha_id=aid, dates=[], daily=[], s_dates=DATES,
                           s_pnl=list(rng.normal(2e3, 1e4, len(DATES))), s_turnover=[0.1] * len(DATES),
                           has_turnover=True))


async def test_facets_selection_and_pyramid_grid(env) -> None:
    await seed()
    from alphago.pool import store
    pool = await store.submitted_members()
    counts = pf.facet_counts(pool)
    assert ("USA", 2) in counts["region"] and ("Region Agnostic Alpha", 1) in counts["classifications"]
    assert ("Max Trade", 1) in counts["investability"]
    ids = lambda xs: sorted(a.id for a in xs)  # noqa: E731
    assert ids(pf.select_members(pool, "returns / drawdown >= 2", {})) == ["U1"]
    assert ids(pf.select_members(pool, "", {"region": {"USA", "EUR"}, "delay": {"D1"}})) == ["U1", "U2"]
    assert ids(pf.select_members(pool, "", {"pyramids": {"USA/D1/MODEL", "EUR/D0/PV"}})) == ["E1", "U1"]
    assert ids(pf.select_members(pool, "sharpe > 0", {})) == ["U1", "U2"]        # E1 lacks a Sharpe
    categories, markets, cells = pf.pyramid_grid(pool)
    assert categories[0] == "PV" and markets[0] == "USA D1" and cells[("PV", "USA D1")] == ("USA/D1/PV", 2)


async def test_compute_uses_engine_metrics_and_lists_missing(env) -> None:
    await seed()
    result = await pf.compute(["U1", "U2", "E1"], bps=5.0)
    assert result["missing"] == ["E1"] and sorted(result["alphas"]) == ["U1", "U2"]
    assert result["split"] == "2025-01-02" and result["is"]["sharpe"] is not None
    assert result["after_cost"]["is"]["sharpe"] < result["is"]["sharpe"]
    assert len(result["correlation"]) == 2 and {m["alpha_id"] for m in result["members"]} == {"U1", "U2"}
    assert all(m["net_sharpe"] is not None and m["k_ratio"] is not None for m in result["members"])


@respx.mock
async def test_prod_correlation_waits_out_the_hourly_limit_and_keeps_answers(env, monkeypatch) -> None:
    from alphago.jobs.queue import JobContext, jobs
    async with db.write() as s:
        s.add(Alpha(id="Q1", status="UNSUBMITTED"))
        s.add(Alpha(id="Q2", status="UNSUBMITTED", checks=[{"name": "PROD_CORRELATION", "value": 0.4}]))
        s.add(Alpha(id="Q3", status="ACTIVE"))
        job = Job(kind="pool_prod_correlation", title="t", params={})
        s.add(job)
        await s.flush()
        ctx = JobContext(job.id, jobs)
    waits: list[float] = []

    async def no_sleep(seconds: float) -> None:
        waits.append(seconds)
    monkeypatch.setattr(brainops.asyncio, "sleep", no_sleep)
    route = respx.get(f"{API}/alphas/Q1/correlations/prod").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "60"}, json={"detail": "THROTTLED"}),
        httpx.Response(200, json={"max": 0.62, "min": -0.1}),
    ])
    monkeypatch.setattr(brain.client, "RATE_LIMIT_RETRIES", 0)
    msg = await brainops._job_prod(ctx, {"alpha_ids": ["Q1", "Q2", "Q3"]})
    assert route.call_count == 2 and waits[0] == brainops.PROD_WAIT       # never less than five minutes
    assert msg.startswith("Production Correlation for 1 Alphas")
    assert await db.kv_get("correlation:prod:Q1") == {"max": 0.62, "min": -0.1}
    async with db.session() as s:
        assert (await s.get(Alpha, "Q1")).prod_corr == 0.62


# ---------------------------------------------------------------- pages

@pytest.fixture
async def ui_env(tmp_path, monkeypatch):
    from alphago import background
    monkeypatch.setattr(background, "autostart_enabled", lambda: False)
    await db.init(tmp_path / "ui.db")
    await seed()
    async with db.write() as s:
        s.add(Alpha(id="UN1", code="rank(close)", status="UNSUBMITTED", region="USA", delay=1, universe="TOP3000",
                    sharpe=1.7, fitness=1.2, train_sharpe=1.8, test_sharpe=1.1, tags=["ace"],
                    checks=[{"name": "LOW_SHARPE", "result": "PASS", "limit": 1.25, "value": 1.7},
                            {"name": "POWER_POOL_CORRELATION", "result": "PENDING"}]))
    yield
    await db.close()


@pytest.mark.parametrize("path,texts", [
    ("/pool", ["Alpha Pool", "Stored Alphas", "Submittable", "Tag Groups", "Power Pool", "Add Tag",
               "Import from BRAIN", "Show correlation as angles", "After-cost mode"]),
    ("/portfolio", ["Portfolio", "Selection Expression", "Compute portfolio", "Sync submitted alphas",
                    "Pyramids (a cell picks one, a heading every pyramid in its row or column)"]),
])
async def test_pages_show_their_sections(ui_env, path: str, texts: list[str]) -> None:
    from nicegui.testing.user_simulation import user_simulation

    from alphago.ui.layout import root
    async with user_simulation(root=root) as user:
        await user.open(path)
        for text in texts:
            await user.should_see(text)


async def test_portfolio_page_computes(ui_env) -> None:
    from nicegui.testing.user_simulation import user_simulation

    from alphago.ui.layout import root
    async with user_simulation(root=root) as user:
        await user.open("/portfolio")
        await user.should_see("3 alphas loaded")
        user.find("Compute portfolio").click()
        await user.should_see("What each alpha adds")
        await user.should_see("Cumulative PnL")


async def test_pool_tabs_render(ui_env) -> None:
    from nicegui.testing.user_simulation import user_simulation

    from alphago.ui.layout import root
    async with user_simulation(root=root) as user:
        await user.open("/pool")
        await user.should_see("Import from BRAIN")
        user.find("Submittable").click()
        await user.should_see("Shortlist")
        await user.should_see("Region-agnostic families")
        user.find("Tag Groups").click()
        await user.should_see("Save as a Group")
        user.find("Power Pool").click()
        user.find("Measure").click()
        await user.should_see("Power Pool alphas")
