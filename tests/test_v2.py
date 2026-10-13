"""v2 features: daily-PnL statistics, outlook, Osmosis plan, Power Pool pick, SPC checks, research tasks, migration."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from alphago.config import SPCSettings, settings
from alphago.db import db
from alphago.db.models import Alpha, AlphaPnL, Candidate, Dataset, DataField, ResearchTask
from alphago.engine import outlook, pnl, portfolio, spc
from alphago.engine.simulator import SimulationService
from alphago.fastexpr import Catalog, FieldInfo, validate
from alphago.knowledge.templates import (PV_PARTNERS, enumerate_group, enumerate_pairs, enumerate_single,
                                         second_order)


def bdays(n: int, end: str = "2025-12-31") -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(end=end, periods=n)]


def series(n: int, mean: float, sd: float, seed: int, *, last2_mean: float | None = None) -> tuple[list[str], list[float]]:
    rng = np.random.default_rng(seed)
    dates = bdays(n)
    vals = rng.normal(mean, sd, n)
    if last2_mean is not None:
        cut = (date.fromisoformat(dates[-1]) - timedelta(days=730)).isoformat()
        for i, d in enumerate(dates):
            if d > cut:
                vals[i] = rng.normal(last2_mean, sd)
    return dates, list(vals)


# ---------------------------------------------------------------- PnL statistics

def test_parse_records_turns_cumulative_into_daily() -> None:
    rows = [{"date": "2025-01-02", "pnl": 0.0}, {"date": "2025-01-03", "pnl": 100.0}, {"date": "2025-01-06", "pnl": 50.0}]
    dates, daily = pnl.parse_records(rows)
    assert dates == ["2025-01-03", "2025-01-06"] and daily == [100.0, -50.0]


def test_compute_stats_early_vs_recent_and_drawdown() -> None:
    dates, daily = series(1500, 2500, 10000, 1, last2_mean=-1500)
    st = pnl.compute_stats(dates, daily)
    assert st["n_days"] == 1500
    assert st["sharpe_early"] > 1.0 and st["sharpe_2y"] < 0
    assert st["max_dd"] > 0 and len(st["spark"]) == pnl.SPARK_POINTS
    assert st["calmar"] == pytest.approx(st["ann_pnl"] / st["max_dd"])


def test_greedy_select_prefers_diversifying_alphas() -> None:
    dates = bdays(800)
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, 800)
    df = pd.DataFrame({
        "A": 0.3 + base, "A_copy": 0.3 + base + rng.normal(0, 0.05, 800),   # nearly identical to A
        "B": 0.12 + rng.normal(0, 1, 800), "C": 0.12 + rng.normal(0, 1, 800), "bad": -0.1 + rng.normal(0, 1, 800),
    }, index=dates)
    picked = pnl.greedy_select(df, list(df.columns), max_n=3)
    assert picked[0] in ("A", "A_copy") and len(picked) == 3
    assert not {"A", "A_copy"} <= set(picked)      # the copy adds nothing, so it is not picked alongside A
    assert "bad" not in picked
    assert pnl.combo_sharpe(df, pnl.risk_weights(df, picked)) > pnl.sharpe(df[picked[0]].to_numpy())


# ---------------------------------------------------------------- outlook

def _pnl_row(alpha_id: str, dates, daily) -> AlphaPnL:  # noqa: ANN001
    row = AlphaPnL(alpha_id=alpha_id, dates=dates, daily=daily)
    for k, v in pnl.compute_stats(dates, daily).items():
        setattr(row, k, v)
    return row


def test_outlook_classes() -> None:
    good = Alpha(id="G", sharpe=1.8, turnover=0.1, op_count=4, field_count=1)
    bad = Alpha(id="B", sharpe=1.6, turnover=0.45, op_count=6, field_count=2)
    o_good = outlook.assess(good, _pnl_row("G", *series(1500, 2500, 10000, 2)))
    o_bad = outlook.assess(bad, _pnl_row("B", *series(1500, 2500, 10000, 3, last2_mean=-1500)))
    assert o_good.outlook in ("robust", "fine") and o_good.risk < 35
    assert o_bad.outlook == "overfit" and o_bad.risk >= 50 and o_bad.holds < 0
    assert "lost money" in o_bad.reason
    assert o_bad.est_os < o_good.est_os
    assert outlook.assess(Alpha(id="X", sharpe=1.5), None).reason.startswith("No daily PnL")


# ---------------------------------------------------------------- Osmosis plan + Power Pool pick

@pytest.fixture
async def dbenv(tmp_path):
    await db.init(tmp_path / "v2.db")
    yield
    await db.close()


async def _seed_scope(n: int = 14) -> list[str]:
    ids = []
    async with db.write() as s:
        for i in range(n):
            aid = f"OS{i:02d}"
            ids.append(aid)
            turnover = 0.6 if i == 0 else 0.1                     # alpha 0 breaks the turnover rule
            last2 = 50 if i == 1 else None                        # alpha 1 has a weak recent period
            dates, daily = series(1200, 900 + 30 * i, 10000, 100 + i, last2_mean=last2)
            s.add(Alpha(id=aid, type="REGULAR", status="ACTIVE", region="USA", delay=1, sharpe=1.5 + i * 0.05,
                        turnover=turnover, code=f"rank(ts_delta(close, {i + 2}))", osmosis_points=0,
                        date_submitted=datetime(2026, 9, 3 + i, tzinfo=timezone.utc), tags=[]))
            s.add(_pnl_row(aid, dates, daily))
    return ids


async def test_osmosis_plan_rules(dbenv, monkeypatch) -> None:
    await _seed_scope()
    monkeypatch.setattr(settings.programs, "osmosis_min_alphas", 5)
    plan = await portfolio.make_plan(max_n=8)
    sp = plan["scopes"][0]
    left = {x["id"] for x in sp["left_out"]}
    assert {"OS00", "OS01"} <= left                               # turnover > 50% and last-2y Sharpe < 0.5
    assert 1 <= len(sp["picked"]) <= 8 and not left & set(sp["picked"])
    assert sum(sp["points"].values()) == settings.programs.osmosis_points_total
    assert all(settings.programs.osmosis_min_points <= v <= settings.programs.osmosis_max_points
               for v in sp["points"].values())
    assert (await portfolio.saved_plan())["max_n"] == 8


async def test_osmosis_review_flags_weak_holdings(dbenv) -> None:
    await _seed_scope()
    async with db.write() as s:
        for aid, pts in (("OS00", 50_000), ("OS05", 30_000), ("OS06", 20_000)):
            (await s.get(Alpha, aid)).osmosis_points = pts
    [rev] = await portfolio.review()
    assert rev.alphas == 3 and rev.points == 100_000
    assert any("fail the rules" in m for m in rev.big)            # OS00 holds points despite turnover > 50%
    assert any("Only 3 alphas" in m for m in rev.big)
    assert "OS00" not in portfolio.review_numbers([rev])          # numbers only, no alpha ids


async def test_power_pool_pick_respects_correlation_and_max(dbenv) -> None:
    dates = bdays(900)
    rng = np.random.default_rng(11)
    base = rng.normal(0, 10000, 900)
    async with db.write() as s:
        for i in range(6):
            aid = f"PP{i}"
            daily = list(2000 + (base if i < 3 else rng.normal(0, 10000, 900)) + rng.normal(0, 500, 900))
            s.add(Alpha(id=aid, type="REGULAR", status="ACTIVE", region="USA", delay=1, sharpe=2.0, turnover=0.1,
                        code="rank(-ts_delta(close, 5))", date_submitted=datetime(2026, 9, 10, tzinfo=timezone.utc),
                        tags=[]))
            s.add(_pnl_row(aid, dates, daily))
    res = await portfolio.pp_pick("2026-09", max_per_scope=4, min_rating="average", max_corr=0.5)
    picked = res["picked"]
    assert len(picked) <= 4
    assert len({"PP0", "PP1", "PP2"} & set(picked)) == 1          # the three clones correlate ~1: only one survives


# ---------------------------------------------------------------- SPC

def test_spc_check_output() -> None:
    rules = SPCSettings()
    ok = spc.check_output('```json\n[{"company_name": "Apple Inc.", "ticker": "aapl"}]\n```', rules)
    assert ok.ok and ok.tickers == ["AAPL"] and ok.warnings
    bad = spc.check_output('[{"company_name": "A", "ticker": "X"}, {"ticker": "X"}]', rules)
    assert not bad.ok and any("duplicate" in e for e in bad.errors) and any("missing" in e for e in bad.errors)
    assert not spc.check_output('{"ticker": "AAPL"}', rules).ok
    side = SPCSettings(side_field="side")
    assert not spc.check_output('[{"company_name": "A", "ticker": "A", "side": "up"}]', side).ok


def test_spc_portfolio_returns_and_stability() -> None:
    idx = pd.bdate_range("2026-09-14", periods=5)
    closes = pd.DataFrame({"AAA": [100, 110, 121, 121, 121], "BBB": [50, 50, 45, 45, 45]}, index=idx, dtype=float)
    rows = [{"company_name": "A", "ticker": "AAA", "side": "long"}, {"company_name": "B", "ticker": "BBB", "side": "short"},
            {"company_name": "Z", "ticker": "ZZZ", "side": "long"}]
    port, per_name, missing = spc.portfolio_returns(closes, rows, SPCSettings(side_field="side"), date(2026, 9, 14))
    assert missing == ["ZZZ"]
    assert port.iloc[0] == pytest.approx(0.05) and port.iloc[1] == pytest.approx(0.5 * 0.1 + 0.5 * 0.1)
    assert per_name["AAA"] == pytest.approx(0.21)
    cmp_ = spc.compare_runs([rows, rows[:2], [rows[0]]])
    assert cmp_["common"] == ["AAA"] and cmp_["pairs"][0]["overlap"] == pytest.approx(2 / 3)


async def test_spc_test_pnl_uses_prices(monkeypatch) -> None:
    idx = pd.bdate_range("2026-09-14", periods=4)

    async def fake_closes(tickers, start, end):  # noqa: ANN001
        return pd.DataFrame({"AAA": [10, 11, 12, 12.0], "SPY": [100, 101, 102, 103.0]}, index=idx)

    monkeypatch.setattr(spc, "fetch_closes", fake_closes)
    res = await spc.test_pnl([{"company_name": "A", "ticker": "AAA"}], SPCSettings(), date(2026, 9, 14))
    assert res["days"] == 3 and res["total_return"] == pytest.approx(0.2)
    assert res["benchmark_return"] == pytest.approx(0.03) and res["curve"]["dates"]


# ---------------------------------------------------------------- templates

def test_enumerated_templates_are_valid_fastexpr() -> None:
    fields = [("fnd_a", "MATRIX"), ("fnd_b", "MATRIX"), ("anl_v", "VECTOR")]
    pv = [(f, "MATRIX") for f in ("close", "open", "high", "low", "volume", "adv20")]
    cat = Catalog(fields=[FieldInfo(f, t) for f, t in fields + PV_PARTNERS + pv])
    exprs = [e for e, _ in list(enumerate_single(fields))[:40]] + [e for e, _ in list(enumerate_group(fields))[:40]]
    exprs += [e for e, _ in list(enumerate_pairs(fields[:2], fields[1:]))[:60]]
    exprs += [e for e, _ in second_order("rank(fnd_a)")]
    bad = [(e, validate(e, cat).errors) for e in exprs if not validate(e, cat).ok]
    assert not bad, bad[:3]
    assert len(set(exprs)) == len(exprs)


# ---------------------------------------------------------------- research tasks / simulator cores

async def test_simulator_shares_cores_between_tasks(dbenv) -> None:
    sim = SimulationService()
    settings_ = {"region": "USA", "delay": 1, "universe": "TOP3000", "instrumentType": "EQUITY", "language": "FASTEXPR"}
    async with db.write() as s:
        rows = [Candidate(expression=f"rank(close+{i})", settings=settings_, settings_hash=f"h{i}", region="USA",
                          delay=1, status="approved", task_id=(1 if i < 40 else 2 if i < 60 else None))
                for i in range(70)]
        s.add_all(rows)
        await s.flush()
        ids = [r.id for r in rows]
    await sim.enqueue(ids)
    sim.set_task_cores(1, 2)
    sim.set_task_cores(2, 1)
    taken: list[int | None] = []
    for _ in range(6):
        pack = await sim._next_pack(10)
        if not pack:
            break
        assert len({c.task_id for c in pack}) == 1                 # never mixes tasks
        key = pack[0].task_id
        sim._inflight_task[key] = sim._inflight_task.get(key, 0) + 1  # pretend it is running
        await sim._set_status([c.id for c in pack], status="simulating")
        taken.append(key)
    assert taken.count(1) == 2 and taken.count(2) == 1             # capped at their cores
    assert taken.count(None) >= 1                                   # work without a task still flows
    sim._inflight_task = {}
    sim.pause_task(1)
    assert all(c.task_id != 1 for c in await sim._next_pack(10))
    n = await sim.cancel_queued(task_id=2)
    assert n == 10 and sim.task_queued(2) == 0


async def test_prepare_skips_answered_shapes_and_tags_task(dbenv) -> None:
    from alphago.engine.simulator import CandidateSpec
    sim = SimulationService()
    st = {"region": "USA", "delay": 1, "universe": "TOP3000"}
    first = await sim.prepare([CandidateSpec(f"ts_rank(close, {d})", st) for d in (5, 10, 20, 40)],
                              initial_status="approved", task_id=7)
    assert all(c.task_id == 7 for c in first.created)
    async with db.write() as s:
        for c in first.created:
            (await s.get(Candidate, c.id)).status = "done"
    again = await sim.prepare([CandidateSpec("ts_rank(close, 80)", st)], history_cap=3, task_id=7)
    assert not again.created and again.left_out == {"shape this market has already answered": 1}


async def test_research_plan_and_controls(dbenv, monkeypatch) -> None:
    from alphago.brain import brain
    from alphago.engine.research import research
    from alphago.llm import gemini
    monkeypatch.setattr(gemini, "configure", lambda *a, **k: False)
    monkeypatch.setattr(settings.research, "research_focus", "")
    async with db.write() as s:
        for i, cat in enumerate(("Fundamental", "Analyst", "Sentiment")):
            s.add(Dataset(id=f"ds{i}", region="USA", delay=1, universe="TOP3000", name=f"Dataset {i}",
                          category_name=cat, field_count=5, alpha_count=100 * i, user_count=10, coverage=0.8,
                          value_score=1.0, pyramid_multiplier=1.2))
            s.add(DataField(id=f"f{i}", dataset_id=f"ds{i}", region="USA", delay=1, universe="TOP3000", type="MATRIX",
                            coverage=0.9))
    plan = await research.plan(region="USA", delay=1, universe="TOP3000")
    assert plan["dataset_ids"] and plan["kind"] in ("single", "pair", "interaction") and plan["focus_reason"]
    monkeypatch.setattr(brain.client.quota, "remaining", 5000)
    tid = await research.plan_and_create(source="assisted", start=False)
    t = await research.get(tid)
    assert t.status == "planned" and t.reason
    await research.stop(tid, "skipped by user")
    assert (await research.get(tid)).status == "stopped"
    assert await research.delete(tid) and await research.get(tid) is None
    with pytest.raises(ValueError):
        await research.create(kind="pair", region="USA", delay=1, universe="TOP3000", dataset_ids=["ds0"])


# ---------------------------------------------------------------- migration of a v1 database

async def test_v1_database_migrates(tmp_path) -> None:
    path = tmp_path / "v1.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE candidates (id INTEGER PRIMARY KEY, expression TEXT, settings JSON, settings_hash "
                "VARCHAR(32), region VARCHAR(16), delay INTEGER, origin VARCHAR(32), status VARCHAR(16))")
    con.execute("CREATE TABLE spc_prompts (id INTEGER PRIMARY KEY, name VARCHAR(128), version INTEGER, prompt TEXT)")
    con.execute("INSERT INTO candidates (expression, settings, settings_hash, region, delay, origin, status) VALUES "
                "('rank(close)', '{}', 'h', 'USA', 1, 'manual', 'done')")
    con.commit()
    con.close()
    await db.init(path)
    try:
        async with db.session() as s:
            c = await s.scalar(select(Candidate))
            assert c.expression == "rank(close)" and c.task_id is None
            assert await s.scalar(select(ResearchTask)) is None   # new table exists
    finally:
        await db.close()
    con = sqlite3.connect(path)
    cols = {r[1] for r in con.execute('PRAGMA table_info("spc_prompts")')}
    idx = {r[1] for r in con.execute('PRAGMA index_list("candidates")')}
    con.close()
    assert "form_settings" in cols and "ix_cand_task" in idx


async def test_gemini_without_key_fails_fast(dbenv, monkeypatch) -> None:
    import sys
    import time

    from alphago.llm import LLMError
    gm = sys.modules["alphago.llm.gemini"]          # the package attribute `gemini` is the client, not the module
    from alphago.llm.schemas import OsmosisPrediction
    monkeypatch.setattr(gm, "gemini_key", lambda: None)
    monkeypatch.setattr(gm.gemini, "_client", None)
    started = time.monotonic()
    with pytest.raises(LLMError, match="No Gemini API key"):
        await gm.gemini.generate_json("hello", OsmosisPrediction)
    assert time.monotonic() - started < 2          # not retried with back-off like a network error
