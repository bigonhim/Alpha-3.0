"""All-markets catalogue sync, Data Explorer queries (FTS search, facets, availability), outlines, pyramids, the
quarter standing, competitions and the Today headline — against a respx-mocked BRAIN."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone

import httpx
import pytest
import respx

from alphago.brain import brain
from alphago.brain import catalog_api as api_mod
from alphago.catalog import competitions as comp_mod
from alphago.catalog import outline as outline_mod
from alphago.catalog import pyramids as pyr_mod
from alphago.catalog import quarter as quarter_mod
from alphago.catalog import queries, search
from alphago.catalog import sync as sync_mod
from alphago.catalog import today as today_mod
from alphago.catalog.queries import FieldFilter, Scope
from alphago.db import db
from alphago.db.models import DataField, Dataset, Operator

API = "https://api.worldquantbrain.com"
AUTH = {"user": {"id": "U1"}, "token": {"expiry": 14400.0}, "permissions": []}
USA = Scope("USA", 1, "TOP3000")


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setattr(sync_mod, "RETRY_WAIT", 0.0)
    monkeypatch.setattr(sync_mod, "BULK_DATASETS_WAIT", 2.0)
    await db.init(tmp_path / "cat.db")
    brain.client._credentials = lambda: ("me@x.com", "pw")
    pyr_mod.clear_cache()
    quarter_mod.clear_cache()
    sync_mod.live = None
    yield
    sync_mod.live = None
    await brain.client.close()
    await db.close()


def _auth(router) -> None:  # noqa: ANN001
    router.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH))
    router.get(f"{API}/authentication").mock(return_value=httpx.Response(200, json=AUTH))


def bulk_field(fid: str, dataset: str, cat: str, sub: str, **extra) -> dict:
    return {"id": fid, "description": extra.pop("description", f"{fid} description"), "type": extra.pop("type", "MATRIX"),
            "dataset": {"id": dataset, "name": f"{dataset} name"}, "category": {"id": cat, "name": cat.title()},
            "subcategory": {"id": sub, "name": sub.title()}, "coverage": extra.pop("coverage", 0.9),
            "dateCoverage": 0.8, "userCount": 5, "alphaCount": extra.pop("alphaCount", 10),
            "pyramidMultiplier": 1.3, "dateCreated": "2024-03-01T00:00:00", "themes": [], **extra}


# ---------------------------------------------------------------- all-markets sync

@respx.mock
async def test_sync_all_markets_starter_first_details_and_failures(env):
    _auth(respx)
    order: list[str] = []
    attempts = {"EUR": 0}

    def fields(request: httpx.Request) -> httpx.Response:
        assert request.headers["Accept"] == "application/json;version=3.0"
        region = request.url.params["region"]
        order.append(region)
        if region == "EUR":
            attempts["EUR"] += 1
            return httpx.Response(400, json={"detail": "Invalid query"})
        if region == "ASI":            # the enveloped shape is read too
            return httpx.Response(200, json={"results": [bulk_field("asi_close", "pv1", "pv", "price")]})
        return httpx.Response(200, json=[
            bulk_field("close", "pv1", "pv", "price", regionCoverage=3),
            bulk_field("close", "pv1", "pv", "price", type="UNIVERSE"),         # repeated id: MATRIX wins
            bulk_field("anl44_eps_prevalue", "analyst4", "analyst", "estimates")])

    def datasets(request: httpx.Request) -> httpx.Response:
        region = request.url.params["region"]
        assert "universe" not in request.url.params      # one read per region, shared by its markets
        return httpx.Response(200, json={"count": 1, "results": [
            {"id": "pv1", "name": "Price Volume", "description": "Prices", "region": region, "delay": 1,
             "universe": "TOP3000" if region == "USA" else "MINVOL1M", "valueScore": 4.5, "fieldCount": 99,
             "category": {"id": "pv", "name": "Price Volume"}, "subcategory": {"id": "price", "name": "Price"}}]})

    respx.get(f"{API}/data-fields").mock(side_effect=fields)
    respx.get(f"{API}/data-sets").mock(side_effect=datasets)
    markets = [{"region": "EUR", "delay": 1, "universe": "TOP2500"}, {"region": "USA", "delay": 1,
                                                                        "universe": "TOP3000"},
               {"region": "ASI", "delay": 1, "universe": "MINVOL1M"}]
    summary = await sync_mod.run_sync(None, {"markets": markets})
    assert order[0] == "USA"                              # the starter has the connection to itself
    assert attempts["EUR"] == 3                           # three tries, then noted and skipped
    states = {m["region"]: m["state"] for m in sync_mod.live.markets.values()}
    assert states == {"USA": "done", "EUR": "failed", "ASI": "done"}
    assert "Synced 2 of 3 markets" in summary and "EUR/D1/TOP2500" in summary
    async with db.session() as s:
        close = (await s.execute(DataField.__table__.select().where(DataField.id == "close"))).mappings().one()
        ds = (await s.execute(Dataset.__table__.select().where(Dataset.region == "USA"))).mappings().all()
    assert close["type"] == "MATRIX" and close["category_id"] == "pv" and close["subcategory_id"] == "price"
    assert close["subcategory"] == "Price" and close["date_coverage"] == 0.8 and close["pyramid_multiplier"] == 1.3
    assert close["date_created"] == "2024-03-01" and close["region_coverage"] == 3
    pv1 = next(d for d in ds if d["id"] == "pv1")
    assert pv1["value_score"] == 4.5 and pv1["description"] == "Prices"       # full record from the details step
    assert {d["id"] for d in ds} == {"pv1", "analyst4"}                      # derived rows for the rest
    run = await sync_mod.latest_run()
    assert run["status"] == "COMPLETE" and "EUR/D1/TOP2500" in run["error"]
    assert await search.ready()


@respx.mock
async def test_region_all_is_read_dataset_by_dataset(env):
    _auth(respx)
    pages: list[tuple[str, int]] = []

    def fields(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        assert p["region"] == "ALL" and p["limit"] == "50"
        pages.append((p["dataset.id"], int(p["offset"])))
        n = 50 if p["dataset.id"] == "big" and p["offset"] == "0" else 3
        return httpx.Response(200, json={"results": [bulk_field(f"{p['dataset.id']}_{p['offset']}_{i}",
                                                                p["dataset.id"], "pv", "price") for i in range(n)]})

    respx.get(f"{API}/data-fields").mock(side_effect=fields)
    respx.get(f"{API}/data-sets").mock(return_value=httpx.Response(200, json={"count": 2, "results": [
        {"id": "big", "region": "ALL", "delay": 1, "universe": "TOP3000"},
        {"id": "small", "region": "ALL", "delay": 1, "universe": "TOP3000"}]}))
    await sync_mod.run_sync(None, {"region_agnostic": True, "markets": [{"region": "ALL", "delay": 1,
                                                                          "universe": "TOP3000"}]})
    assert sorted(pages) == [("big", 0), ("big", 50), ("small", 0)]
    assert (await queries.counts(Scope("ALL", 1, "TOP3000")))["fields"] == 56


@respx.mock
async def test_sync_can_be_cancelled(env):
    _auth(respx)
    gate = asyncio.Event()

    async def slow(_request: httpx.Request) -> httpx.Response:
        await gate.wait()
        return httpx.Response(200, json=[])

    respx.get(f"{API}/data-fields").mock(side_effect=slow)
    respx.get(f"{API}/data-sets").mock(return_value=httpx.Response(200, json=[]))
    task = asyncio.create_task(sync_mod.run_sync(None, {"markets": [{"region": "USA", "delay": 1,
                                                                     "universe": "TOP3000"}]}))
    for _ in range(50):
        await asyncio.sleep(0.01)
        if sync_mod.live and sync_mod.live.markets["USA/D1/TOP3000"]["state"] == "fetching":
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    run = await sync_mod.latest_run()
    assert run["status"] == "CANCELLED"


def test_offered_markets_cover_every_region_delay_universe():
    from alphago.brain.options import SimOptions
    opts = SimOptions()
    markets = sync_mod.offered_markets(opts)
    assert sync_mod.Market("USA", 0, "TOP3000") in markets and sync_mod.Market("GLB", 1, "MINVOL1M") in markets
    assert all(m.delay in opts.delays(m.region) for m in markets)


# ---------------------------------------------------------------- explorer queries

async def _seed(rows: list[dict]) -> None:
    stamp = datetime.now(timezone.utc)
    full = []
    for r in rows:
        base = dict(dataset_id="pv1", region="USA", delay=1, universe="TOP3000", type="MATRIX", description=None,
                    category="Price Volume", category_id="pv", subcategory_id="price", subcategory="Price",
                    coverage=0.9, date_coverage=0.9, user_count=1, alpha_count=1, pyramid_multiplier=1.2, themes=None,
                    date_created="2024-01-01", region_coverage=None, synced_at=stamp)
        full.append({**base, **r})
    async with db.write() as s:
        await s.execute(DataField.__table__.insert(), full)


async def test_smart_search_ranks_words_then_falls_back_to_any_word(env):
    await _seed([
        dict(id="anl44_eps_revision", description="Analyst earnings revisions momentum", dataset_id="analyst4",
             category_id="analyst", category="Analyst", alpha_count=3),
        dict(id="anl44_eps_prevalue", description="Previous earnings per share estimate", dataset_id="analyst4",
             category_id="analyst", category="Analyst", alpha_count=50),
        dict(id="close", description="Daily close price", alpha_count=900),
        dict(id="steps_count", description="Unrelated steps", alpha_count=1),
    ])
    assert await search.stale() and not await search.ready()
    # before the index exists a smart search still answers, as a substring match
    assert (await queries.fields(USA, FieldFilter(search="close")))["total"] == 1
    await search.rebuild()
    assert not await search.stale()
    page = await queries.fields(USA, FieldFilter(search="earnings revision", sort_by="relevance"))
    assert [r["field_id"] for r in page["results"]] == ["anl44_eps_revision"]     # every word first
    page = await queries.fields(USA, FieldFilter(search="earnings price", sort_by="relevance"))
    assert not page["conjunctive"] and {r["field_id"] for r in page["results"]} >= {"close", "anl44_eps_prevalue"}
    page = await queries.fields(USA, FieldFilter(search="eps", sort_by="relevance"))
    assert {r["field_id"] for r in page["results"]} == {"anl44_eps_revision", "anl44_eps_prevalue"}
    # text mode is a literal, escaped substring: `_eps_` must not match "steps"
    page = await queries.fields(USA, FieldFilter(search="_eps_", search_mode="text"))
    assert {r["field_id"] for r in page["results"]} == {"anl44_eps_revision", "anl44_eps_prevalue"}
    page = await queries.fields(USA, FieldFilter(search="eps", search_mode="text", sort_by="alpha_count"))
    assert page["results"][0]["field_id"] == "anl44_eps_prevalue"


async def test_filters_facets_availability_and_stats(env):
    await _seed([
        dict(id="close", alpha_count=10, coverage=1.0),
        dict(id="volume", type="VECTOR", alpha_count=0, coverage=0.5),
        dict(id="eps_fast_d1", dataset_id="analyst4", category_id="analyst", category="Analyst",
             subcategory_id="est", subcategory="Estimates", date_created="2025-06-01"),
        dict(id="close", region="ALL"),                     # region-agnostic copy
        dict(id="volume", region="EUR", universe="TOP2500"),
    ])
    flt = FieldFilter(category_ids=["pv"], field_types=["MATRIX"])
    page = await queries.fields(USA, flt)
    assert [r["field_id"] for r in page["results"]] == ["close"]
    f = await queries.facets(USA, flt)
    # a facet ignores its own selection: both categories and both types still show their counts
    assert {c["id"]: c["n"] for c in f["categories"]} == {"pv": 1, "analyst": 1}
    assert {t["id"]: t["n"] for t in f["types"]} == {"MATRIX": 1, "VECTOR": 1}
    assert {d["id"] for d in f["datasets"]} == {"pv1"}
    assert f["availability"] == {"region_agnostic": 1, "region_exclusive": 0, "fast_d1": 0}
    all_f = await queries.facets(USA, FieldFilter())
    assert all_f["availability"] == {"region_agnostic": 1, "region_exclusive": 1, "fast_d1": 1}
    assert [m["month"] for m in all_f["date_added"]] == ["2024-01-01", "2025-06-01"]
    assert (await queries.fields(USA, FieldFilter(region_exclusive=True)))["total"] == 1
    assert (await queries.fields(USA, FieldFilter(fast_d1=True)))["results"][0]["field_id"] == "eps_fast_d1"
    assert (await queries.fields(USA, FieldFilter(coverage_max=0.6)))["results"][0]["field_id"] == "volume"
    assert (await queries.fields(USA, FieldFilter(date_created_from=date(2025, 1, 1))))["total"] == 1
    stats = await queries.stats(USA)
    assert stats["coverage_min"] == 0.5 and stats["coverage_max"] == 1.0 and stats["coverage_median"] == 0.9
    avail = await queries.field_availability("volume")
    assert [(a["region"], a["universe"]) for a in avail] == [("EUR", "TOP2500"), ("USA", "TOP3000")]
    anywhere = await queries.fields_anywhere("clo")
    assert [r["field_id"] for r in anywhere] == ["close"]
    assert await queries.synced_categories() >= {("pv", "USA", 1), ("analyst", "USA", 1)}


async def test_outline_text_matches_alpha_harness_format(env):
    await _seed([dict(id="close", description="Daily\nclose", coverage=0.97, date_coverage=None),
                 dict(id="anl4_x", dataset_id="analyst4", category_id="analyst", category="Analyst",
                      subcategory=None, subcategory_id=None, type="VECTOR", coverage=0.5, date_coverage=0.25)])
    async with db.write() as s:
        s.add(Dataset(id="pv1", region="USA", delay=1, universe="TOP3000", name="Price Volume",
                      description="Prices and volumes"))
    out = await outline_mod.field_outline(USA, field_ids=["close", "anl4_x", "nope"])
    assert out.fields == 2 and out.missing == ["nope"]
    assert out.text == "\n".join([
        "# Data Fields", "Region: USA", "Delay: 1", "",
        "## Category: Analyst", "### Subcategory: Uncategorised", "#### Dataset: analyst4", "",
        "- Field: anl4_x", "  Description: none", "  Type: VECTOR",
        "  Instrument Coverage: 50%", "  Date Coverage: 25%",
        "", "## Category: Price Volume", "### Subcategory: Price", "#### Dataset: pv1", "Name: Price Volume",
        "Description: Prices and volumes", "",
        "- Field: close", "  Description: Daily close", "  Type: MATRIX", "  Instrument Coverage: 97%",
        "  Date Coverage: unknown"])
    by_set = await outline_mod.field_outline(USA, dataset_ids=["pv1"])
    assert by_set.fields == 1 and "#### Dataset: pv1" in by_set.text
    with pytest.raises(ValueError):
        await outline_mod.field_outline(USA)


# ---------------------------------------------------------------- pyramids and the quarter

SUMMARY = {"performance": {
    "currentQuarter": {"name": "2026-Q4", "startDate": "2026-10-01", "endDate": "2026-12-31"},
    "history": [{"quarter": {"name": "2026-Q3", "startDate": "2026-07-01", "endDate": "2026-09-30"}},
                {"quarter": {"name": "2026-Q4", "startDate": "2026-10-01", "endDate": "2026-12-31"}}]}}
MULTIPLIERS = {"pyramids": [
    {"category": {"id": "pv", "name": "Price Volume"}, "region": "USA", "delay": 1, "multiplier": 1.1},
    {"category": {"id": "analyst", "name": "Analyst"}, "region": "USA", "delay": 1, "multiplier": 1.8},
    {"category": {"id": "analyst", "name": "Analyst"}, "region": "EUR", "delay": 0, "multiplier": 1.4}]}


@respx.mock
async def test_pyramid_grid_quarter_end_plus_one_and_all_time(env):
    _auth(respx)
    respx.get(f"{API}/users/self/consultant/summary").mock(return_value=httpx.Response(200, json=SUMMARY))
    respx.get(f"{API}/users/self/activities/pyramid-multipliers").mock(
        return_value=httpx.Response(200, json=MULTIPLIERS))
    seen: list[dict] = []

    def counts(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"pyramids": [
            {"category": {"id": "analyst"}, "region": "USA", "delay": 1, "alphaCount": 3},
            {"category": {"id": "pv"}, "region": "USA", "delay": 1, "alphaCount": 1},
            {"category": {"id": "pv"}, "region": "CHN", "delay": 1, "alphaCount": 7}]})   # no multiplier: not shown

    respx.get(f"{API}/users/self/activities/pyramid-alphas").mock(side_effect=counts)
    await _seed([dict(id="close")])
    grid = await pyr_mod.pyramid_grid()
    assert grid["period"] == "2026-Q4" and [q["name"] for q in grid["quarters"]] == ["2026-Q4", "2026-Q3"]
    assert seen[-1] == {"startDate": "2026-10-01", "endDate": "2027-01-01"}      # the last day counts
    assert [c["id"] for c in grid["categories"]] == ["analyst", "pv"]             # richest first
    assert grid["columns"] == [{"region": "USA", "delay": 1}, {"region": "EUR", "delay": 0}]
    cells = {(c["category_id"], c["region"], c["delay"]): c for c in grid["cells"]}
    assert ("pv", "CHN", 1) not in cells
    assert cells[("analyst", "USA", 1)]["alpha_count"] == 3 and cells[("pv", "USA", 1)]["synced"]
    assert not cells[("analyst", "USA", 1)]["synced"]
    await pyr_mod.pyramid_grid("2026-Q3")
    assert seen[-1] == {"startDate": "2026-07-01", "endDate": "2026-10-01"}
    await pyr_mod.pyramid_grid(pyr_mod.ALL_TIME)
    assert seen[-1] == {}
    with pytest.raises(pyr_mod.UnknownPeriodError):
        await pyr_mod.pyramid_grid("1999-Q1")
    assert pyr_mod.standing(3, 3, False) == ("3", "formulated")
    assert pyr_mod.standing(1, 3, False) == ("1/3", "2 more to formulate")
    assert pyr_mod.standing(2, 3, True)[1] == "formulating counts per quarter"


@respx.mock
async def test_quarters_fall_back_to_the_new_york_calendar_quarter(env):
    _auth(respx)
    respx.get(f"{API}/users/self/consultant/summary").mock(return_value=httpx.Response(403, json={}))
    listed = await pyr_mod.quarters()
    assert len(listed) == 1 and listed[0]["name"].count("-Q") == 1
    assert pyr_mod.fallback_quarter(date(2026, 12, 31)) == {"name": "2026-Q4", "startDate": "2026-10-01",
                                                            "endDate": "2026-12-31"}


def test_quarter_boundaries():
    assert quarter_mod.bounds(date(2026, 11, 14)) == (date(2026, 10, 1), date(2027, 1, 1))
    assert quarter_mod.bounds(date(2026, 3, 31)) == (date(2026, 1, 1), date(2026, 4, 1))
    days = [("2026-09-30", 4), ("2026-10-01", 2), ("2026-12-31", 1), ("2027-01-01", 9)]
    pyramids = [{"alphaCount": 3}, {"alphaCount": 5}, {"alphaCount": 1}, {"alphaCount": 0}]
    st = quarter_mod.tally(days, pyramids, date(2026, 10, 1), date(2027, 1, 1))
    assert (st.label, st.start, st.end, st.submitted) == ("2026-Q4", "2026-10-01", "2026-12-31", 3)
    assert (st.pyramids_formulated, st.pyramids_started) == (2, 1)


@respx.mock
async def test_quarter_standing_reads_submissions_and_pyramids(env):
    _auth(respx)
    respx.get(f"{API}/users/self/activities/submissions").mock(return_value=httpx.Response(200, json={
        "records": {"records": [["2026-10-02", 2], ["2026-09-29", 5], ["2026-11-01", "1"]]}}))
    route = respx.get(f"{API}/users/self/activities/pyramid-alphas").mock(return_value=httpx.Response(
        200, json={"pyramids": [{"alphaCount": 4}, {"alphaCount": 2}]}))
    st = await quarter_mod.standing(date(2026, 11, 14))
    assert st.submitted == 3 and st.pyramids_formulated == 1 and st.pyramids_started == 1
    assert dict(route.calls.last.request.url.params) == {"startDate": "2026-10-01", "endDate": "2027-01-01"}


@respx.mock
async def test_operator_usage_counts_each_alpha_once_and_skips_backfills(env):
    _auth(respx)
    async with db.write() as s:
        for name, scope in (("rank", ["REGULAR"]), ("ts_delta", ["REGULAR"]), ("ts_zscore", ["REGULAR"]),
                            ("ts_backfill", ["REGULAR"]), ("subtract", ["REGULAR"]), ("reverse", ["REGULAR"]),
                            ("combo_a", ["COMBO"])):
            s.add(Operator(name=name, category="Test", scope=scope))
    alphas = [{"id": "A", "type": "REGULAR", "regular": {"code": "rank(-ts_delta(close, 5))"}},
              {"id": "B", "type": "REGULAR", "regular": {"code": "rank(close) - rank(open)"}},
              {"id": "C", "type": "REGULAR", "regular": {"code": "rank("}},
              {"id": "D", "type": "SUPER", "regular": None}]
    route = respx.get(url__startswith=f"{API}/users/self/alphas").mock(return_value=httpx.Response(
        200, json={"count": 4, "results": alphas}))
    use = await quarter_mod.operator_usage(date(2026, 11, 14))
    counts = {o.name: o.alphas for o in use.operators}
    assert counts["rank"] == 2 and counts["subtract"] == 1 and counts["ts_zscore"] == 0
    assert "ts_backfill" not in counts and "combo_a" not in counts
    assert use.operators[0].alphas == 0 and use.unreadable == 1 and use.alphas == 2 and use.label == "2026-Q4"
    url = str(route.calls.last.request.url)
    assert "dateSubmitted%3E=2026-10-01T00%3A00%3A00-04%3A00" in url and "dateSubmitted%3C2027-01-01T00" in url


# ---------------------------------------------------------------- competitions

@respx.mock
async def test_competitions_ongoing_with_leaderboard_then_past(env):
    _auth(respx)
    respx.get(f"{API}/competitions").mock(return_value=httpx.Response(200, json={"results": [
        {"id": "OLD", "name": "Old Cup", "startDate": "2025-01-01T00:00:00-05:00",
         "endDate": "2025-03-01T00:00:00-05:00", "signUpDate": "2025-01-02"},
        {"id": "LIVE", "name": "Live Cup", "startDate": "2026-10-01T00:00:00-04:00",
         "endDate": "2026-12-01T00:00:00-05:00", "signUpDate": None, "signUpEndDate": "2026-11-20T00:00:00-05:00",
         "faq": "https://x/faq", "status": "ACCEPTED"},
    ]}))
    respx.get(f"{API}/competitions/LIVE").mock(return_value=httpx.Response(200, json={
        "id": "LIVE", "leaderboard": {"rank": 12, "alphas": 4}}))
    now = datetime(2026, 11, 14, tzinfo=timezone.utc)
    rows = await comp_mod.competitions(now)
    assert [r.id for r in rows] == ["LIVE", "OLD"]
    live = rows[0]
    assert live.ongoing and live.rank == 12 and live.alphas == 4 and not live.enrolled
    assert live.days_left(now) == 18 and live.sign_up_open(now) and live.url.endswith("/competition/LIVE")
    assert rows[1].enrolled and not rows[1].ongoing and rows[1].rank is None


# ---------------------------------------------------------------- today

def test_today_headline_rules():
    h = today_mod.headline
    assert h(0, 0) == "No simulations available today."
    assert h(0, 5000) == "You have used every simulation today. Nothing was wasted."
    assert h(100, 5000, 100) == "Every simulation left today is queued. Nothing is going to waste."
    assert h(4800, 5000) == "4,800 simulations are unused today. They cannot be saved for tomorrow."
    assert h(3000, 5000, 200) == "2,800 simulations left today. 200 more are already queued."
    assert h(1000, 5000) == "1,000 left of 5,000. Good day so far."


def test_today_exact_versus_estimated():
    import time

    from alphago.brain.quota import QuotaTracker
    q = QuotaTracker()
    q.sent_today = 40
    est = today_mod.simulations_today(q, queued=10)
    assert not est.exact and est.remaining == 5000 - 40 and est.remaining_text == "~4,960" and est.unspoken == 4950
    q.update_from_headers({"X-Ratelimit-Limit": "8000", "X-Ratelimit-Remaining": "7000"})
    exact = today_mod.simulations_today(q, queued=0)
    assert exact.exact and exact.limit == 8000 and exact.used == 1000 and exact.remaining_text == "7,000"
    q.updated_at = time.time() - 3 * 86400            # yesterday's figure is not today's
    assert not today_mod.simulations_today(q, queued=0).exact
    assert 0.0 <= exact.day_gone <= 1.0


def test_platform_timezone_is_new_york():
    assert str(api_mod.PLATFORM_TZ) == "America/New_York"
