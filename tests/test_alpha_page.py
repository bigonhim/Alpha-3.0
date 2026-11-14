"""The Alpha page as BRAIN's own page shows it, against a respx-mocked BRAIN: a REGULAR alpha (header, Testing
Status in BRAIN's sentences, Summary with the yearly table, chart, correlations from the cache), one with a hidden
test period, a region-agnostic parent with its children, the kept-copy fallback and the pure helpers."""

from __future__ import annotations

import httpx
import pytest
import respx
from nicegui import ui
from nicegui.testing.user_simulation import user_simulation

from alphago.brain import brain
from alphago.db import db
from alphago.db.models import Alpha
from alphago.ui import alpha_detail as ad
from alphago.ui import layout

API = "https://api.worldquantbrain.com"
AUTH = {"user": {"id": "U1"}, "token": {"expiry": 14400.0}, "permissions": []}

CHECKS = [
    {"name": "LOW_SHARPE", "result": "PASS", "limit": 1.25, "value": 1.61},
    {"name": "LOW_FITNESS", "result": "FAIL", "limit": 1.0, "value": 0.8},
    {"name": "HIGH_TURNOVER", "result": "PASS", "limit": 0.7, "value": 0.25},
    {"name": "SELF_CORRELATION", "result": "PENDING"},
    {"name": "MATCHES_COMPETITION", "result": "PASS", "competitions": [{"id": "IQC26", "name": "IQC 2026"}]},
]
FIGURES = {"sharpe": 1.61, "fitness": 0.8, "turnover": 0.25, "returns": 0.081, "drawdown": 0.06, "margin": 0.00065,
           "longCount": 1400, "shortCount": 1380}
SETTINGS = {"instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000", "delay": 1, "decay": 4,
            "neutralization": "SUBINDUSTRY", "truncation": 0.08, "language": "FASTEXPR"}


def regular(alpha_id: str, **extra) -> dict:
    return {"id": alpha_id, "type": "REGULAR", "status": "UNSUBMITTED", "stage": "IS", "name": None,
            "dateCreated": "2026-10-20T10:00:00-04:00", "settings": SETTINGS,
            "regular": {"code": "rank(-ts_delta(close, 5))", "description": None}, "tags": [],
            "classifications": [{"id": "DATA_USAGE:SINGLE_DATA_SET", "name": "Single Data Set Alpha"}],
            "is": {**FIGURES, "checks": CHECKS}, **extra}


YEARLY = {"schema": {"properties": [{"name": n} for n in
                                    ("year", "sharpe", "turnover", "fitness", "returns", "drawdown", "margin",
                                     "longCount", "shortCount", "stage")]},
          "records": [["2019", 1.2, 0.2, 0.9, 0.07, 0.05, 0.0006, 1400, 1380, "TRAIN"],
                      ["2020", 1.9, 0.3, 1.1, 0.09, 0.04, 0.0007, 1410, 1370, "TRAIN"]]}
PNL = {"schema": {"name": "pnl", "properties": [{"name": "date", "type": "date"}, {"name": "pnl", "type": "amount"},
                                               {"name": "risk-neutralized-pnl", "type": "amount",
                                                "title": "Risk neutralized"}]},
       "records": [["2019-01-02", 100.0, 80.0], ["2019-01-03", 150.0, 90.0], ["2022-01-03", 300.0, 200.0]]}


@pytest.fixture
async def env(tmp_path, monkeypatch):
    await db.init(tmp_path / "page.db")
    brain.client._credentials = lambda: ("me@x.com", "pw")

    async def stub() -> None:
        ui.label("stub")
    monkeypatch.setattr(layout, "_nav", lambda: [layout.NavItem("/", "Dashboard", "home", "#fff", stub),
                                                 layout.NavItem("/alphas", "Alphas", "layers", "#fff", stub)])
    yield
    await brain.client.close()
    await db.close()


def mock_brain(router, bodies: dict[str, dict]) -> None:
    router.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH))
    router.get(f"{API}/authentication").mock(return_value=httpx.Response(200, json=AUTH))
    for alpha_id, body in bodies.items():
        router.get(f"{API}/alphas/{alpha_id}").mock(return_value=httpx.Response(200, json=body))
        router.get(f"{API}/alphas/{alpha_id}/recordsets/yearly-stats").mock(
            return_value=httpx.Response(200, json=YEARLY))
        router.get(f"{API}/alphas/{alpha_id}/recordsets/pnl").mock(return_value=httpx.Response(200, json=PNL))


async def _open(user, alpha_id: str, ids=None):
    with user.client:
        page = await ad.open_alpha(alpha_id, ids)
    await page.ready
    return page


@respx.mock
async def test_regular_alpha_page_shows_brains_sections(env) -> None:
    mock_brain(respx, {"REG0001": regular("REG0001")})
    await db.kv_set("correlation:self:REG0001", {"body": {"max": 0.4321, "min": -0.1, "schema": {"properties": []},
                                                          "records": []}, "fetchedAt": "2026-11-01T12:00:00+00:00"})
    async with user_simulation(root=layout.root) as user:
        await user.open("/")
        page = await _open(user, "REG0001", ["REG0000", "REG0001", "REG0002"])
        assert page.info is not None and page.info["type"] == "REGULAR"
        for text in ("UNSUB", "Created 10/20/2026 EDT", "anonymous", "Check Submission", "IS Testing Status",
                     "1 FAIL", "3 PASS", "1 PENDING", "IS Summary", "Aggregate Data", "Sharpe", "1.61", "25.00%",
                     "6.50 bps", "2019", "2020", "Chart", "Correlation", "Self Correlation", "Prod Correlation",
                     "0.4321", "Performance Comparison", "Simulation Settings", "SUBINDUSTRY",
                     "Single Data Set Alpha", "2 of 3", "Copy as Markdown", "Repair…", "Open on BRAIN ↗"):
            await user.should_see(text)
    assert (await db.kv_get("alpha:REG0001"))["body"]["id"] == "REG0001"
    assert (await db.kv_get("recordset:yearly-stats:REG0001"))["body"] == YEARLY


@respx.mock
async def test_test_period_hides_until_shown(env) -> None:
    body = regular("REG0003", test={**FIGURES, "startDate": "2021-01-01"}, train={**FIGURES})
    mock_brain(respx, {"REG0003": body})
    async with user_simulation(root=layout.root) as user:
        await user.open("/")
        page = await _open(user, "REG0003")
        await user.should_see("Test period and overall stats are hidden by default when test period is specified.")
        await user.should_see("Show test period")
        await user.should_see("IS Summary")                      # TRAIN's summary still shows, titled IS
        await user.should_not_see("IS Testing Status")
        assert not page.test_shown
        opts = ad.chart_options(PNL, "pnl", "2021-01-01", None, test_shown=False)
        assert all(d < "2021-01-01" for s in opts["series"] for d, _ in s["data"])


@respx.mock
async def test_region_agnostic_parent_lists_its_children(env) -> None:
    parent = {"id": "RAP0001", "type": "RA_PARENT", "status": "UNSUBMITTED", "stage": "IS",
              "dateCreated": "2026-10-21", "settings": {**SETTINGS, "region": "GLB"},
              "regular": {"code": "rank(close)"}, "children": ["RAC0001", "RAC0002"],
              "is": {"checks": [{"name": "LOW_SHARPE", "result": "PASS", "value": 1.3, "limit": 1.25}],
                     "subregions": {"USA": {**FIGURES}, "EUR": {**FIGURES, "sharpe": 1.1}}}}

    def child(alpha_id: str, region: str, fail: bool) -> dict:
        checks = [{"name": "LOW_SHARPE", "result": "FAIL" if fail else "PASS", "value": 1.0, "limit": 1.25},
                  {"name": "SELF_CORRELATION", "result": "PENDING"}]
        return {"id": alpha_id, "type": "RA_CHILD", "status": "UNSUBMITTED", "stage": "IS",
                "settings": {**SETTINGS, "region": region}, "regular": {"code": "rank(close)"},
                "is": {**FIGURES, "checks": checks}}

    mock_brain(respx, {"RAP0001": parent, "RAC0001": child("RAC0001", "USA", False),
                       "RAC0002": child("RAC0002", "EUR", True)})
    async with user_simulation(root=layout.root) as user:
        await user.open("/")
        await _open(user, "RAP0001")
        for text in ("RA_PARENT", "Regional Child Alphas", "Tests: 1 failed", "Tests: 1 pass, 1 pending submission check",
                     "View alpha details", "Max Drawdown", "Created 10/21/2026 EDT"):
            await user.should_see(text)
        await user.should_not_see("IS Summary")                  # a parent has no figures of its own
        await _open(user, "RAC0001")
        await user.should_see("Only the parent region-agnostic alpha can be submitted.")


@respx.mock
async def test_kept_copy_shows_when_brain_does_not_answer(env) -> None:
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH))
    respx.get(f"{API}/alphas/OLD0001").mock(return_value=httpx.Response(404, json={"detail": "Not found."}))
    await db.kv_set("alpha:OLD0001", {"body": regular("OLD0001", name="kept one"),
                                      "fetchedAt": "2026-11-02T08:15:00+00:00"})
    await db.kv_set("recordset:yearly-stats:OLD0001", {"body": YEARLY, "fetchedAt": "2026-11-02T08:15:00+00:00"})
    await db.kv_set("recordset:pnl:OLD0001", {"body": PNL, "fetchedAt": "2026-11-02T08:15:00+00:00"})
    async with user_simulation(root=layout.root) as user:
        await user.open("/")
        await _open(user, "OLD0001")
        await user.should_see("showing the copy from Nov 02, 08:15 UTC.")
        await user.should_see("kept one")


async def test_local_alpha_opens_signed_out_with_the_repair_plan(env) -> None:
    brain.client._credentials = lambda: None
    async with db.write() as s:
        s.add(Alpha(id="LOC0001", code="rank(close)", sharpe=-1.4, fitness=-0.8, turnover=0.25, verdict="weak",
                    status="UNSUBMITTED", region="USA", delay=1, universe="TOP3000",
                    checks=[{"name": "LOW_SHARPE", "result": "FAIL", "limit": 1.25, "value": -1.4}],
                    settings={"region": "USA", "delay": 1, "universe": "TOP3000"}))
    async with user_simulation(root=layout.root) as user:
        await user.open("/")
        await _open(user, "LOC0001")
        await user.should_see("Could not load LOC0001 from BRAIN")
        await user.should_see("Flip Sign")                       # Sharpe -1 or below
        await user.should_see("Submit…")


# ── pure helpers ─────────────────────────────────────────────────────────────────────────────


def test_period_tabs_and_first_period() -> None:
    assert [t[0] for t in ad.period_tabs({"is": {}}, True, False)] == ["train", "test", "is", "os"]
    assert [t[2] for t in ad.period_tabs({"is": {}}, True, False)] == [True, False, False, False]
    assert ad.period_tabs({"is": {}, "os": {}}, False, True) == [("is", "IS", True), ("os", "OS", True)]
    assert ad.first_period("OS", False, True) == "os" and ad.first_period("PROD", False, True) == "os"
    assert ad.first_period("IS", True, False) == "train" and ad.first_period("IS", True, True) == "test"
    assert ad.stage_of("test") == "is"


def test_platform_day_is_new_york() -> None:
    assert ad.platform_day("2026-09-30") == "09/30/2026 EDT"
    assert ad.platform_day("2026-12-01T03:00:00Z") == "11/30/2026 EST"
    assert ad.platform_day(None) == ""


def test_regions_of_global_and_region_agnostic() -> None:
    assert [r for r, _ in ad.regions_of({"glbAmer": {"sharpe": 1}, "glbEmea": {"sharpe": 2}})] == ["AMER", "EMEA"]
    assert [r for r, _ in ad.regions_of({"subregions": {"USA": {}, "EUR": {}}})] == ["USA", "EUR"]


def test_power_pool_description_problems() -> None:
    assert len(ad.description_problems("")) == 2
    good = ("Idea: prices that fell hard over five days tend to recover as liquidity providers step in.\n"
            "Rationale for data used: close is the cleanest daily price.\n"
            "Rationale for operators used: ts_delta measures the move and rank makes it cross-sectional.")
    assert ad.description_problems(good) == []
    missing = ad.description_problems("Idea: something long enough " * 5)
    assert any('"Rationale for data used:" is missing' in p for p in missing)
    assert ad.POWER_POOL_TEMPLATE == "Idea: \nRationale for data used: \nRationale for operators used: "


def test_chart_colours_the_main_line_by_period_and_dashes_side_lines() -> None:
    opts = ad.chart_options(PNL, "pnl", "2021-01-01", None, test_shown=True)
    names = [s["name"] for s in opts["series"]]
    assert names == ["Train", "Test", "Risk neutralized"]
    assert opts["series"][-1]["lineStyle"]["type"] == "dashed"


def test_info_of_prefers_the_kept_check() -> None:
    info = ad.info_of("X1", regular("X1"), [{"name": "LOW_SHARPE", "result": "PASS"}])
    assert info["periods"]["is"]["checks"] == [{"name": "LOW_SHARPE", "result": "PASS"}]
    assert info["code"] == "rank(-ts_delta(close, 5))" and info["brainUrl"].endswith("/alpha/X1")
