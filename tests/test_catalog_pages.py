"""Page smoke tests for the Data Explorer, Pyramids, Competitions, the Dashboard and the Sync matrix (NiceGUI's
browserless user simulation, BRAIN mocked with respx)."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx
import pytest
import respx
from nicegui.testing.user_simulation import user_simulation

from alphago.brain import brain
from alphago.catalog import pyramids as pyr_mod
from alphago.catalog import quarter as quarter_mod
from alphago.catalog import sync as sync_mod
from alphago.db import db
from alphago.db.models import Alpha, DataField, Dataset

API = "https://api.worldquantbrain.com"


@pytest.fixture
async def dbenv(tmp_path, monkeypatch):
    from alphago import background
    monkeypatch.setattr(background, "autostart_enabled", lambda: False)
    await db.init(tmp_path / "pages.db")
    stamp = datetime.now(timezone.utc)
    base = dict(region="USA", delay=1, universe="TOP3000", type="MATRIX", category="Price Volume", category_id="pv",
                subcategory_id="price", subcategory="Price", coverage=0.9, date_coverage=0.8, user_count=4,
                alpha_count=7, pyramid_multiplier=1.2, date_created="2024-01-01", synced_at=stamp)
    async with db.write() as s:
        await s.execute(DataField.__table__.insert(), [
            {**base, "id": "close", "dataset_id": "pv1", "description": "Daily close price"},
            {**base, "id": "anl4_eps", "dataset_id": "analyst4", "description": "Earnings per share estimate",
             "category_id": "analyst", "category": "Analyst"}])
        s.add(Dataset(id="pv1", region="USA", delay=1, universe="TOP3000", name="Price Volume"))
        s.add(Alpha(id="UI1", code="rank(close)", verdict="ready", status="UNSUBMITTED", region="USA", delay=1))
    pyr_mod.clear_cache()
    quarter_mod.clear_cache()
    sync_mod.live = None
    yield
    brain.client.state, brain.client.expiry_at = "logged_out", None
    await brain.client.close()
    await db.close()


def _signed_in() -> None:
    brain.client.state, brain.client.expiry_at = "logged_in", time.time() + 3600


async def _see_all(path: str, texts: list[str]) -> None:
    from alphago.ui.layout import root
    async with user_simulation(root=root) as user:
        await user.open(path)
        for text in texts:
            await user.should_see(text, retries=30)


async def test_data_explorer_shows_the_market_filters_and_fields(dbenv):
    await _see_all("/data", ["Data Explorer", "Filters", "Copy Data Fields", "Download as Markdown",
                             "Category → Subcategory → Dataset", "Search all markets", "2 matching fields"])


async def test_data_explorer_opens_the_market_another_page_asked_for(dbenv):
    await db.kv_set("pref:data_explorer", {"scope": {"region": "USA", "delay": 1, "universe": "TOP3000"},
                                           "filter": {"category_ids": ["analyst"]}})
    await _see_all("/data", ["1 matching fields"])


async def test_pyramids_and_competitions_ask_to_sign_in(dbenv):
    await _see_all("/pyramids", ["Pyramids", "Sign in to BRAIN to see your pyramids"])
    await _see_all("/competitions", ["Competitions", "Sign in to BRAIN to see its competitions"])


@respx.mock
async def test_pyramids_grid_signed_in(dbenv):
    _signed_in()
    respx.get(f"{API}/users/self/consultant/summary").mock(return_value=httpx.Response(200, json={
        "performance": {"currentQuarter": {"name": "2026-Q4", "startDate": "2026-10-01",
                                           "endDate": "2026-12-31"}}}))
    respx.get(f"{API}/users/self/activities/pyramid-multipliers").mock(return_value=httpx.Response(200, json={
        "pyramids": [{"category": {"id": "pv", "name": "Price Volume"}, "region": "USA", "delay": 1,
                      "multiplier": 1.4}]}))
    respx.get(f"{API}/users/self/activities/pyramid-alphas").mock(return_value=httpx.Response(200, json={
        "pyramids": [{"category": {"id": "pv"}, "region": "USA", "delay": 1, "alphaCount": 2}]}))
    await _see_all("/pyramids", ["Pyramid Multiplier Map", "USA D1", "Price Volume", "×1.4"])


@respx.mock
async def test_competitions_signed_in(dbenv):
    _signed_in()
    respx.get(f"{API}/competitions").mock(return_value=httpx.Response(200, json={"results": [
        {"id": "LIVE", "name": "Live Cup", "startDate": "2020-01-01T00:00:00-05:00",
         "endDate": "2099-01-01T00:00:00-05:00", "signUpDate": "2020-01-02"},
        {"id": "OLD", "name": "Old Cup", "startDate": "2019-01-01T00:00:00-05:00",
         "endDate": "2019-03-01T00:00:00-05:00"}]}))
    respx.get(f"{API}/competitions/LIVE").mock(return_value=httpx.Response(200, json={
        "id": "LIVE", "leaderboard": {"rank": 7, "alphas": 3}}))
    await _see_all("/competitions", ["Live Cup", "Enrolled", "#7", "Open in BRAIN", "Past Competitions (1)"])


async def test_dashboard_keeps_its_sections_and_shows_the_new_ones(dbenv):
    await _see_all("/", ["Submittable Alphas", "Simulations Left Today", "Getting Started", "Add a Free API Key",
                         "Submitted This Quarter", "Pyramids Completed", "Operators Used", "Work in Flight",
                         "Nothing is queued right now."])


async def test_sync_page_keeps_its_sections_and_shows_the_matrix(dbenv):
    await _see_all("/sync", ["Email alerts", "Run in the background", "Start with Windows", "BRAIN Datasets",
                             "markets synced", "Fetching Fields", "Not Offered", "TOP3000"])
