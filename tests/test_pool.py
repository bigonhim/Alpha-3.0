"""Alpha Pool: Selection Expression, Power Pool verdicts and submission order, the submittable shortlist, filters,
tagging against a mocked BRAIN and the local self-correlation."""

from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import date, datetime, timedelta, timezone

import httpx
import numpy as np
import pytest
import respx

from alphago.brain import brain
from alphago.db import db
from alphago.db.models import Alpha, AlphaPnL
from alphago.pool import brainops, correlation, store, submittable
from alphago.pool.selection import SelectionError, compile_selection

API = "https://api.worldquantbrain.com"
NAMES = ["sharpe", "fitness", "returns", "turnover", "drawdown", "margin", "long_count", "short_count"]


# ---------------------------------------------------------------- Selection Expression

def sel(text: str, **metrics: float | None) -> bool:
    return compile_selection(text, NAMES).test(metrics)


def test_selection_conditions_are_anded_across_statements_and_lines() -> None:
    text = "sharpe >= 1.58;\nreturns / turnover >= 1\nreturns / drawdown >= 1"
    assert sel(text, sharpe=1.6, returns=0.2, turnover=0.1, drawdown=0.1)
    assert not sel(text, sharpe=1.5, returns=0.2, turnover=0.1, drawdown=0.1)
    assert compile_selection(text, NAMES).conditions == 3


def test_selection_literals_percent_and_bps() -> None:
    assert sel("returns >= 25%", returns=0.25)
    assert not sel("returns >= 25%", returns=0.2499)
    assert sel("margin >= 5bps", margin=0.0005)


def test_selection_missing_metric_and_three_valued_logic() -> None:
    assert not sel("sharpe > 1", sharpe=None)
    assert not sel("sharpe != 1", sharpe=None)            # NaN on either side is no answer
    assert not sel("not(or(sharpe > 1, fitness > 1))")     # neither measured: still missing, not selected
    assert sel("or(sharpe > 1, fitness > 1)", sharpe=2.0)  # a true side settles `or`
    assert sel("!(sharpe > 1 && fitness > 1)", sharpe=0.5)  # a false side settles `and`
    assert sel("is_nan(fitness)", fitness=None)


def test_selection_division_by_zero_is_nan_and_power_binds_below_sign() -> None:
    assert not sel("returns / drawdown >= 0", returns=0.1, drawdown=0.0)
    assert sel("-2 ^ 2 == 4")
    assert sel("2 ^ 3 ^ 2 == 512")                          # right-associative


def test_selection_definitions_functions_ternary_and_comments() -> None:
    text = "# calmar\nc = returns / drawdown  // named\nc >= 2\nmax(sharpe, fitness, 0) > 1.5 ? 1 : 0\nabs(-1) == 1"
    assert sel(text, returns=0.4, drawdown=0.1, sharpe=1.6, fitness=1.0)
    assert not sel(text, returns=0.1, drawdown=0.1, sharpe=1.6, fitness=1.0)
    assert sel("if_else(sharpe > 1, 1, 0) && sqrt(4) == 2 && sign(-3) == -1 && power(2, 3) == 8", sharpe=2)
    assert sel("min(\n  sharpe,\n  fitness\n) > 1", sharpe=2, fitness=1.5)   # a line break inside brackets wraps


def test_selection_empty_selects_everything() -> None:
    assert compile_selection("  # only a comment\n", NAMES).test({})


@pytest.mark.parametrize("text,message", [
    ("sharpe > 1\nShar_pe > 1", 'Line 2: no metric named "Shar_pe"; did you mean "sharpe"?'),
    ("sharpe > 1;\n\nAbs(sharpe) > 1", 'Line 3: no function named "Abs"; did you mean "abs"?'),
    ("max(sharpe)", "Line 1: max takes at least 2 arguments"),
    ("abs = 3", 'Line 1: "abs" is a function'),
    ("sharpe >", "Line 1: the statement ends too early"),
    ("sharpe > 1\nfitness > (", "Line 2:"),
])
def test_selection_errors_name_the_line(text: str, message: str) -> None:
    with pytest.raises(SelectionError) as err:
        compile_selection(text, NAMES)
    assert str(err.value).startswith(message)


# ---------------------------------------------------------------- synthetic series

def weekdays(end: date, n: int) -> list[str]:
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= timedelta(days=1)
    return out[::-1]


DATES = weekdays(date(2025, 12, 31), 700)
RNG = np.random.default_rng(7)
BASE = RNG.normal(0, 1e4, len(DATES))


def mix(weight: float, seed: int) -> list[float]:
    noise = np.random.default_rng(seed).normal(0, 1e4, len(DATES))
    return list(weight * BASE + math.sqrt(1 - weight ** 2) * noise)


def loaded(**cols: list[float]) -> dict[str, dict]:
    return {k: {"dates": DATES, "pnl": v, "closing": [False] * len(DATES)} for k, v in cols.items()}


def test_power_pool_verdicts_clear_beats_blocked_unmeasured() -> None:
    grid = correlation.build_grid(loaded(P1=list(BASE), C1=mix(0.95, 1), C2=mix(0.95, 2), C3=mix(0.0, 3)))
    meta = {"P1": {"sharpe": 1.5, "region": "USA"}, "C1": {"sharpe": 1.65, "region": "USA"},
            "C2": {"sharpe": 1.6, "region": "USA"}, "C3": {"sharpe": 1.0, "region": "USA"}}
    rows = {r["alpha_id"]: r for r in correlation.correlate(grid, meta, ["C1", "C2", "C3"], ["P1"])}
    assert rows["C1"]["verdict"] == "beats" and rows["C1"]["needed"] == 1.65       # 1.1 × 1.5 rounded
    assert rows["C2"]["verdict"] == "blocked" and rows["C2"]["closest"] == "P1"
    assert rows["C3"]["verdict"] == "clear" and rows["C3"]["needed"] is None
    # a pool member without a series could hide a collision: unmeasured, unless already blocked
    rows = {r["alpha_id"]: r for r in correlation.correlate(grid, meta, ["C2", "C3"], ["P1", "MISSING"])}
    assert rows["C3"]["verdict"] == "unmeasured" and rows["C3"]["against"] == 2
    assert rows["C2"]["verdict"] == "blocked"
    # nothing submitted in the region: the first alpha is clear
    assert correlation.correlate(grid, meta, ["C2"], [])[0]["verdict"] == "clear"


def test_power_pool_bar_is_the_tallest_collision_not_the_closest() -> None:
    grid = correlation.build_grid(loaded(P1=list(BASE), P2=mix(0.9, 5), C=mix(0.97, 6)))
    meta = {"P1": {"sharpe": 1.7, "region": "USA"}, "P2": {"sharpe": 2.62, "region": "USA"},
            "C": {"sharpe": 2.43, "region": "USA"}}
    row = correlation.correlate(grid, meta, ["C"], ["P1", "P2"])[0]
    assert row["closest"] == "P1" and row["needed"] == round(2.62 * 1.1, 4) and row["verdict"] == "blocked"


def test_too_few_shared_days_is_unmeasured() -> None:
    short = {"dates": DATES[-200:], "pnl": list(BASE[-200:]), "closing": [False] * 200}
    grid = correlation.build_grid({**loaded(P1=list(BASE)), "C": short})
    meta = {"P1": {"sharpe": 1.0, "region": "USA"}, "C": {"sharpe": 2.0, "region": "USA"}}
    assert correlation.correlate(grid, meta, ["C"], ["P1"])[0]["verdict"] == "unmeasured"


def test_closing_days_are_never_correlated() -> None:
    data = loaded(P1=list(BASE), C=list(BASE))
    data["C"]["closing"] = [True] * len(DATES)          # every day rebuilt, none exported
    grid = correlation.build_grid(data)
    meta = {"P1": {"sharpe": 1.0, "region": "USA"}, "C": {"sharpe": 1.0, "region": "USA"}}
    assert correlation.correlate(grid, meta, ["C"], ["P1"])[0]["correlation"] is None


def test_submission_order_takes_best_after_cost_first_and_rejudges_rivals() -> None:
    grid = correlation.build_grid(loaded(A=mix(0.0, 11), B=mix(0.0, 12), B2=mix(0.0, 13)))
    # B2 is B's twin: once B is picked, B2 collides with it
    grid.pnl[:, grid.at["B2"]] = grid.pnl[:, grid.at["B"]] * 0.99 + 1.0
    meta = {"A": {"sharpe": 1.4, "region": "USA", "after_cost_5bps": 1.2},
            "B": {"sharpe": 1.5, "region": "USA", "after_cost_5bps": 1.3},
            "B2": {"sharpe": 1.5, "region": "USA", "after_cost_5bps": 1.1}}
    picks, blocked, unmeasured = correlation.submission_order(grid, meta, ["A", "B", "B2"], {})
    assert [p["alpha_id"] for p in picks] == ["B", "A"]
    assert blocked == {"B2": "B"} and unmeasured == []


def test_submission_order_starts_only_from_what_passes_today() -> None:
    grid = correlation.build_grid(loaded(P=list(BASE), X=mix(0.98, 21), Y=mix(0.0, 22)))
    meta = {"P": {"sharpe": 3.0, "region": "EUR"}, "X": {"sharpe": 1.0, "region": "EUR", "after_cost_5bps": 2.0},
            "Y": {"sharpe": 1.0, "region": "EUR", "after_cost_5bps": 1.0}}
    picks, blocked, _ = correlation.submission_order(grid, meta, ["X", "Y"], {"EUR": ["P"]})
    assert [p["alpha_id"] for p in picks] == ["Y"] and blocked == {}


# ---------------------------------------------------------------- shortlist rules

def test_held_out_rule() -> None:
    assert submittable.held_out_ok(1.0, 1.8)
    assert not submittable.held_out_ok(0.8, 1.8)           # below half the train Sharpe
    assert not submittable.held_out_ok(0.0, None)
    assert not submittable.held_out_ok(None, 1.0)          # never tested has not held
    assert submittable.held_out_ok(0.3, None)


def test_stability_order_ranks_test_sharpe_and_sub_universe_margin() -> None:
    sub = lambda v: [{"name": "LOW_SUB_UNIVERSE_SHARPE", "value": v, "limit": 1.0}]  # noqa: E731
    rows = [{"alpha_id": "c", "test_sharpe": 1.5, "checks": []},          # no margin: ranks last on it
            {"alpha_id": "b", "test_sharpe": 1.0, "checks": sub(3.0)},
            {"alpha_id": "a", "test_sharpe": 2.0, "checks": sub(2.0)}]
    assert submittable.stability_order(rows) == ["a", "b", "c"]


def test_independent_keeps_at_most_five_uncorrelated_per_region() -> None:
    prof = {f"x{i}": {"USA": (DATES, mix(0.0, 100 + i))} for i in range(8)}
    prof["twin"] = {"USA": (DATES, list(np.asarray(prof["x0"]["USA"][1]) * 2))}
    prof["other_region_twin"] = {"EUR": (DATES, list(prof["x0"]["USA"][1]))}
    picked = submittable.independent(["x0", "twin", "other_region_twin", "x1", "x2", "x3", "x4"], prof)
    assert picked == ["x0", "other_region_twin", "x1", "x2", "x3"]


def test_sparkline_thins_and_keeps_the_last_point() -> None:
    line = submittable.sparkline([1.0] * 1000)
    assert len(line) == 120 and line[-1] == 1000.0


# ---------------------------------------------------------------- database-backed

@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_write_lock", asyncio.Lock())
    await db.init(tmp_path / "pool.db")
    monkeypatch.setattr(brain.client, "state", "logged_in")
    monkeypatch.setattr(brain.client, "expiry_at", time.time() + 14400)
    yield
    await brain.client.close()
    await db.close()


PASSING = [{"name": "LOW_SHARPE", "result": "PASS", "limit": 1.25, "value": 1.6},
           {"name": "LOW_FITNESS", "result": "PASS", "limit": 1.0, "value": 1.2}]


async def add_series(alpha_id: str, pnl: list[float]) -> None:
    async with db.write() as s:
        s.add(AlphaPnL(alpha_id=alpha_id, dates=[], daily=[], s_dates=DATES, s_pnl=pnl,
                       s_turnover=[0.1] * len(DATES), has_turnover=True))


async def test_stored_filters_tags_days_and_calmar(env) -> None:
    eastern_late = datetime(2026, 10, 8, 3, 30, tzinfo=timezone.utc)      # still Oct 7 in New York
    async with db.write() as s:
        s.add(Alpha(id="late", status="UNSUBMITTED", tags=["momentum", "x"], date_created=eastern_late,
                    returns=0.2, drawdown=0.1, turnover=0.3))
        s.add(Alpha(id="next", status="UNSUBMITTED", tags=["value"], date_created=eastern_late + timedelta(hours=1),
                    returns=0.1, drawdown=0.1, turnover=0.1))
        s.add(Alpha(id="sub", status="ACTIVE", tags=["value"], date_created=eastern_late))

    async def ids(**kw) -> list[str]:
        return [a.id for a in (await store.page(store.StoredFilter(**kw)))[1]]
    assert await ids(created_before=date(2026, 10, 7)) == ["late"]
    assert await ids(created_after=date(2026, 10, 8)) == ["next"]
    assert set(await ids(tags=["momentum", "value"])) == {"late", "next"}
    assert await ids(tags=["momentum", "x"], tags_all=True) == ["late"]
    assert await ids(tags=["momentum", "value"], tags_all=True) == []
    assert await ids(sort_by="calmar") == ["late", "next"]
    assert await ids(minimum={"turnover": 20}) == ["late"]               # 20 % typed, 0.20 compared
    assert await ids(submitted=True) == ["sub"]
    o = await store.overview()
    assert o["tags"] == ["momentum", "value", "x"] and o["submitted"] == 1


@respx.mock
async def test_add_tags_patches_only_tags_and_saves_them_locally(env) -> None:
    async with db.write() as s:
        s.add(Alpha(id="A1", status="UNSUBMITTED", tags=["old"]))
        s.add(Alpha(id="A2", status="UNSUBMITTED"))
    respx.get(f"{API}/alphas/A1").mock(return_value=httpx.Response(200, json={"id": "A1", "tags": ["mine"]}))
    patch = respx.patch(f"{API}/alphas/A1").mock(
        return_value=httpx.Response(200, json={"id": "A1", "tags": ["mine", "ace", "v2"]}))
    respx.get(f"{API}/alphas/A2").mock(return_value=httpx.Response(200, json={"id": "A2", "tags": []}))
    respx.patch(f"{API}/alphas/A2").mock(return_value=httpx.Response(200, json={"id": "A2", "tags": ["ace"]}))
    tagged, failed = await brainops.add_tags(["A1", "A2"], "ace, v2")
    assert tagged == ["A1"] and failed == {"A2": "BRAIN answered, but did not keep the tag."}
    assert json.loads(patch.calls[0].request.content) == {"tags": ["mine", "ace", "v2"]}
    rows = await store.by_ids(["A1", "A2"])
    assert rows["A1"].tags == ["mine", "ace", "v2"] and rows["A2"].tags is None


async def test_add_tags_refuses_the_power_pool_tag_and_bad_tags(env) -> None:
    with pytest.raises(ValueError, match="keeps an Alpha in the Power Pool"):
        await brainops.add_tags(["A1"], "powerpoolselected")
    with pytest.raises(ValueError, match="letters, digits"):
        await brainops.add_tags(["A1"], "bad tag!")


@respx.mock
async def test_import_by_tag_lists_hidden_too_and_ingests(env) -> None:
    from alphago.db.models import Job
    from alphago.jobs.queue import JobContext, jobs
    route = respx.get(url__startswith=f"{API}/users/self/alphas").mock(return_value=httpx.Response(
        200, json={"count": 1, "next": None, "results": [{"id": "T1", "tags": ["ace"], "status": "UNSUBMITTED",
                                                          "settings": {"region": "USA"}}]}))
    async with db.write() as s:
        job = Job(kind="pool_import_tagged", title="t", params={})
        s.add(job)
        await s.flush()
        ctx = JobContext(job.id, jobs)
    msg = await brainops._job_import_tagged(ctx, {"tags": ["ace", "v2"]})
    assert msg == "Imported 1 Alpha tagged ace, v2"
    query = str(route.calls[0].request.url)
    assert "tags=ace%1Fv2" in query and "hidden=true%1Ffalse" in query
    assert (await store.by_ids(["T1"]))["T1"].tags == ["ace"]


async def test_local_self_correlation_skips_power_pool_only_alphas(env) -> None:
    pp_only = [{"id": "X", "name": "Power Pool Alpha"}]
    async with db.write() as s:
        s.add(Alpha(id="S1", status="ACTIVE", region="USA", sharpe=1.5))
        s.add(Alpha(id="S2", status="ACTIVE", region="USA", sharpe=1.5, classifications=pp_only))
        s.add(Alpha(id="S3", status="ACTIVE", region="EUR", sharpe=1.5))
        s.add(Alpha(id="C", status="UNSUBMITTED", region="USA", sharpe=1.5))
    await add_series("S1", mix(0.6, 31))
    await add_series("S2", list(BASE))             # would collide at ~1.0, but BRAIN leaves it out
    await add_series("S3", list(BASE))             # another region
    await add_series("C", list(BASE))
    found = await correlation.local_self_correlation(["C", "S1"])
    value, against = found["C"]
    assert against == 1 and 0.5 < value < 0.7
    assert found["S1"][1] == 0                      # nothing else in its pool: nothing to compare against


async def test_submittable_lists_counts_and_shortlist(env) -> None:
    async with db.write() as s:
        s.add(Alpha(id="good", status="UNSUBMITTED", region="USA", delay=1, universe="TOP3000", sharpe=1.8,
                    train_sharpe=1.9, test_sharpe=1.2, checks=PASSING))
        s.add(Alpha(id="twin", status="UNSUBMITTED", region="USA", delay=1, universe="TOP3000", sharpe=1.7,
                    train_sharpe=1.8, test_sharpe=1.1, checks=PASSING))
        s.add(Alpha(id="third", status="UNSUBMITTED", region="USA", delay=1, universe="TOP3000", sharpe=1.6,
                    train_sharpe=1.6, test_sharpe=1.0, checks=PASSING))
        s.add(Alpha(id="decayed", status="UNSUBMITTED", region="USA", sharpe=2.0, train_sharpe=2.2,
                    test_sharpe=0.5, checks=PASSING))
        s.add(Alpha(id="crazy", status="UNSUBMITTED", region="USA", sharpe=14.0, test_sharpe=12.0,
                    checks=PASSING))
        s.add(Alpha(id="wait", status="UNSUBMITTED", checks=[{"name": "SELF_CORRELATION", "result": "PENDING"},
                                                              *PASSING]))
        s.add(Alpha(id="near", status="UNSUBMITTED", checks=[{"name": "LOW_FITNESS", "result": "FAIL"}]))
    await add_series("good", list(BASE))
    await add_series("twin", list(BASE * 1.01))
    await add_series("third", mix(0.0, 51))
    found = await submittable.submittable(schedule=False)
    assert found["total"] == 4 and found["implausible_hidden"] == 1
    assert found["pending"] == 1 and found["near_misses"] == 1
    assert found["held_out_failed"] == 1 and found["unvalidated"] == 0
    assert [a.id for a in found["shortlist"]] == ["good", "third"] and found["correlated_pruned"] == 1
    assert found["markets"][0]["count"] == 3 and found["pnl_missing"] == []


async def test_power_pool_analysis_writes_verdicts(env) -> None:
    pp_check = [{"name": "POWER_POOL_CORRELATION", "result": "PENDING"}, *PASSING]
    async with db.write() as s:
        s.add(Alpha(id="P", status="ACTIVE", region="USA", sharpe=1.5, checks=pp_check))
        s.add(Alpha(id="C", status="UNSUBMITTED", region="USA", sharpe=1.0, checks=pp_check))
        s.add(Alpha(id="E", status="UNSUBMITTED", region="EUR", sharpe=1.0, checks=pp_check))
        s.add(Alpha(id="R", status="UNSUBMITTED", region="USA", sharpe=1.0, checks=PASSING))
    await add_series("P", list(BASE))
    await add_series("C", mix(0.9, 41))
    found = await correlation.power_pool_analysis(["C", "E", "R"])
    verdicts = {r["alpha_id"]: r["verdict"] for r in found["rows"]}
    assert verdicts == {"C": "blocked", "E": "clear"} and found["other_alphas"] == 1
    rows = await store.by_ids(["C", "E"])
    assert rows["C"].pp_verdict == "blocked" and rows["C"].pp_corr > 0.5 and rows["E"].pp_verdict == "clear"


async def test_ra_family_needs_two_passing_children(env) -> None:
    when = datetime(2026, 1, 5, 12, tzinfo=timezone.utc)
    common = {"code": "rank(close)", "date_created": when, "delay": 1, "decay": 0, "neutralization": "MARKET",
              "truncation": 0.08, "status": "UNSUBMITTED"}
    async with db.write() as s:
        s.add(Alpha(id="RP", sim_type="RA_PARENT", region="ALL", **common))
        s.add(Alpha(id="c1", sim_type="RA_CHILD", region="USA", fitness=1.5, checks=PASSING, **common))
        s.add(Alpha(id="c2", sim_type="RA_CHILD", region="EUR", fitness=1.2,
                    checks=[{"name": "LOW_SHARPE", "result": "PENDING"}], **common))
        s.add(Alpha(id="c3", sim_type="RA_CHILD", region="ASI", fitness=0.5,
                    checks=[{"name": "LOW_SHARPE", "result": "FAIL"}], **common))
    fams = await submittable.families()
    assert len(fams) == 1 and fams[0]["check"]["result"] == "PENDING" and fams[0]["check"]["value"] == 1
    assert fams[0]["standing"].id == "c2" and fams[0]["regions"] == ["USA", "EUR", "ASI"]


async def test_tag_groups_save_replace_and_forget(env) -> None:
    from alphago.pool import groups
    await groups.save_group("ACE", ["ace", "v2"], every=False)
    await groups.save_group("ACE", ["ace"], every=True)            # the same name replaces the group
    found = await groups.all_groups()
    assert [(g.name, g.tags, g.every) for g in found] == [("ACE", ["ace"], True)]
    with pytest.raises(ValueError):
        await groups.save_group(" ", ["ace"], every=False)
    await groups.forget_group(found[0].id)
    assert await groups.all_groups() == []
