"""Phase 1 of the submittable-alpha work: one verdict rule, robustness as a badge, every strong alpha checked, the
targeted multi-round repair loop with learned fix order, duplicate filter, idea clusters, survival-weighted dataset
yield and BRAIN's observed limits in the prompt. Each test fails on the code before this change."""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from alphago.config import settings
from alphago.db import db
from alphago.db.models import Alpha, AlphaPnL, Candidate, Dataset, OptimizationAttempt
from alphago.engine import actions, refine
from alphago.engine.ingest import apply_alpha_json, upsert_alpha
from alphago.engine.simulator import CandidateSpec, SimulationService

BASE = {"region": "USA", "delay": 1, "universe": "TOP3000", "neutralization": "SUBINDUSTRY", "decay": 0}


@pytest.fixture
async def dbenv(tmp_path):
    await db.init(tmp_path / "refine.db")
    yield
    await db.close()


def checks_for(sharpe: float, fitness: float, turnover: float, extra: list[dict] | None = None) -> list[dict]:
    return [
        {"name": "LOW_SHARPE", "result": "PASS" if sharpe >= 1.25 else "FAIL", "limit": 1.25, "value": sharpe},
        {"name": "LOW_FITNESS", "result": "PASS" if fitness >= 1.0 else "FAIL", "limit": 1.0, "value": fitness},
        {"name": "LOW_TURNOVER", "result": "PASS", "limit": 0.01, "value": turnover},
        {"name": "HIGH_TURNOVER", "result": "PASS" if turnover <= 0.7 else "FAIL", "limit": 0.7, "value": turnover},
        {"name": "SELF_CORRELATION", "result": "PENDING"},
    ] + (extra or [])


def alpha_json(alpha_id: str, code: str, st: dict, sharpe: float, fitness: float, turnover: float,
               returns: float = 0.08, extra: list[dict] | None = None) -> dict:
    return {"id": alpha_id, "type": "REGULAR", "settings": st, "regular": {"code": code}, "status": "UNSUBMITTED",
            "dateCreated": "2026-10-20T10:00:00-04:00",
            "is": {"sharpe": sharpe, "fitness": fitness, "turnover": turnover, "returns": returns, "drawdown": 0.05,
                   "margin": 0.0008, "longCount": 1500, "shortCount": 1480,
                   "checks": checks_for(sharpe, fitness, turnover, extra)}}


class FakeSim(SimulationService):
    """The real prepare/submit (validation, de-duplication, caps) with an instant, deterministic 'BRAIN'."""

    def __init__(self, score) -> None:  # noqa: ANN001
        super().__init__()
        self.score = score
        self.n = 0
        self.sent: list[tuple[str, dict]] = []

    async def enqueue(self, candidate_ids: list[int]) -> int:
        async with db.session() as s:
            cands = list(await s.scalars(select(Candidate).where(Candidate.id.in_(candidate_ids),
                                                                 Candidate.status.in_(["new", "approved"]))))
        for c in cands:
            self.n += 1
            aid = f"F{self.n:03d}"
            self.sent.append((c.expression, c.settings))
            sharpe, fitness, turnover = self.score(c.expression, c.settings)
            await upsert_alpha(alpha_json(aid, c.expression, c.settings, sharpe, fitness, turnover),
                               candidate_id=c.id)
            await self._set_status([c.id], status="done", alpha_id=aid, sim_id=f"S{self.n}",
                                   finished_at=datetime.now(timezone.utc))
        return len(cands)

    async def wait_for(self, candidate_ids: list[int], timeout: float | None = None) -> None:
        return None


async def add_alpha(alpha_id: str, code: str, sharpe: float, fitness: float, turnover: float, **kw) -> Alpha:
    data = alpha_json(alpha_id, code, {**BASE, **kw.pop("st", {})}, sharpe, fitness, turnover,
                      extra=kw.pop("extra", None))
    row = await upsert_alpha(data, origin="app")
    if kw:
        async with db.write() as s:
            a = await s.get(Alpha, alpha_id)
            for k, v in kw.items():
                setattr(a, k, v)
            from alphago.engine.evaluator import reverdict
            reverdict(a)
        async with db.session() as s:
            row = await s.get(Alpha, alpha_id)
    return row


# ---------------------------------------------------------------- 1.0 one verdict rule

async def test_power_pool_correlation_fail_no_longer_blocks_ready(dbenv, monkeypatch) -> None:
    await add_alpha("PP1", "rank(-ts_delta(close, 5))", 1.6, 1.2, 0.2)
    full = checks_for(1.6, 1.2, 0.2)[:-1] + [
        {"name": "SELF_CORRELATION", "result": "PASS", "limit": 0.7, "value": 0.31},
        {"name": "POWER_POOL_CORRELATION", "result": "FAIL", "limit": 0.5, "value": 0.62}]

    async def fake_check(alpha_id, max_wait_s=900):  # noqa: ANN001, ANN202
        return full

    monkeypatch.setattr(actions.brain.alphas, "check", fake_check)
    state, _ = await actions.run_check("PP1")
    async with db.session() as s:
        a = await s.get(Alpha, "PP1")
    assert state == "pass" and a.check_state == "pass" and a.verdict == "ready"


async def test_old_power_pool_fails_are_reread_at_startup(dbenv) -> None:
    await add_alpha("PP2", "rank(-ts_delta(close, 6))", 1.6, 1.2, 0.2, checked_at=datetime.now(timezone.utc),
                    check_state="fail",
                    checks=checks_for(1.6, 1.2, 0.2) + [{"name": "POWER_POOL_CORRELATION", "result": "FAIL"}])
    assert await actions.rederive_check_states() == 1
    async with db.session() as s:
        a = await s.get(Alpha, "PP2")
    assert a.check_state == "pending"            # SELF_CORRELATION was PENDING in that stored result
    assert a.verdict == "strong"


async def test_submission_makes_other_passes_stale(dbenv) -> None:
    now = datetime.now(timezone.utc)
    await add_alpha("S1", "rank(-ts_delta(close, 5))", 1.6, 1.2, 0.2, check_state="pass", checked_at=now)
    await add_alpha("S2", "rank(ts_delta(volume, 5))", 1.7, 1.3, 0.2, check_state="pass", checked_at=now)
    await add_alpha("S3", "rank(ts_delta(volume, 7))", 1.7, 1.3, 0.2, check_state="pass", checked_at=now,
                    st={"region": "EUR", "universe": "TOP2500"})
    assert await actions.mark_stale_after_submission("S1") == 1
    async with db.session() as s:
        s2, s3 = await s.get(Alpha, "S2"), await s.get(Alpha, "S3")
    assert s2.check_state == "stale" and s2.verdict == "strong"     # needs a fresh BRAIN check
    assert s3.check_state == "pass" and s3.verdict == "ready"       # other region: unaffected


async def test_refresh_keeps_the_full_check_result(dbenv) -> None:
    a = await add_alpha("R1", "rank(-ts_delta(close, 5))", 1.6, 1.2, 0.2, check_state="fail",
                        checked_at=datetime.now(timezone.utc),
                        checks=checks_for(1.6, 1.2, 0.2)[:-1] + [{"name": "SELF_CORRELATION", "result": "FAIL",
                                                                  "limit": 0.7, "value": 0.85}])
    assert a.verdict == "promising"
    apply_alpha_json(a, alpha_json("R1", a.code, BASE, 1.6, 1.2, 0.2))     # a sync re-reads the IS checks
    assert a.verdict == "promising" and any(c["name"] == "SELF_CORRELATION" and c["result"] == "FAIL"
                                            for c in a.checks)
    assert any(c["name"] == "SELF_CORRELATION" and c["result"] == "PENDING" for c in a.is_checks)


# ---------------------------------------------------------------- 1.1 robustness is a badge, not a demotion

async def test_fragile_alpha_stays_ready_and_survives_refresh(dbenv, monkeypatch) -> None:
    await add_alpha("RB1", "rank(-ts_delta(close, 10))", 2.0, 1.4, 0.2, check_state="pass",
                    checked_at=datetime.now(timezone.utc))
    fake = FakeSim(lambda expr, st: (1.2, 0.9, 0.2))          # neighbours keep only 60% of the Sharpe
    monkeypatch.setattr(refine, "simulator", fake)
    info = await refine.check_robustness("RB1")
    assert info["state"] == "fragile" and info["ratio"] == pytest.approx(0.6)
    async with db.write() as s:
        a = await s.get(Alpha, "RB1")
        assert a.verdict == "ready" and a.robust_state == "fragile"
        apply_alpha_json(a, alpha_json("RB1", a.code, BASE, 2.0, 1.4, 0.2))
        assert a.verdict == "ready" and a.robust_state == "fragile" and a.robust_ratio == pytest.approx(0.6)


async def test_robustness_unknown_when_neighbours_fail(dbenv, monkeypatch) -> None:
    await add_alpha("RB2", "rank(-ts_delta(close, 12))", 2.0, 1.4, 0.2, check_state="pass",
                    checked_at=datetime.now(timezone.utc))

    class Broken(FakeSim):
        async def enqueue(self, candidate_ids):  # noqa: ANN001, ANN202
            await self._set_status(candidate_ids, status="error", error="boom")
            return 0

    monkeypatch.setattr(refine, "simulator", Broken(lambda e, s: (0, 0, 0)))
    info = await refine.check_robustness("RB2")
    async with db.session() as s:
        a = await s.get(Alpha, "RB2")
    assert info["state"] == "unknown" and a.verdict == "ready"     # old code: ratio 0 → demoted to strong


# ---------------------------------------------------------------- 1.2 every strong alpha gets BRAIN's check

async def test_certify_checks_every_strong_alpha(dbenv, monkeypatch) -> None:
    for i in range(5):
        await add_alpha(f"C{i}", f"rank(-ts_delta(close, {i + 3}))", 1.5 + i / 10, 1.1, 0.2)
    checked: list[str] = []

    async def fake_run_check(alpha_id):  # noqa: ANN001, ANN202
        checked.append(alpha_id)
        return "fail", []

    async def no_estimate(alpha_id, fetch=True):  # noqa: ANN001, ANN202
        return None, None

    monkeypatch.setattr(actions, "run_check", fake_run_check)
    monkeypatch.setattr("alphago.engine.similarity.estimate_self_corr", no_estimate)
    report = await refine.certify([f"C{i}" for i in range(5)], refine.RefineContext(use_llm=False), cap=15)
    assert sorted(checked) == [f"C{i}" for i in range(5)] and len(report.checked) == 5


async def test_check_order_puts_pnl_twins_last(dbenv) -> None:
    import numpy as np
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 400)
    dates = [f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(400)]
    alphas = []
    for aid, series, score in (("T1", base, 90), ("T2", base + rng.normal(0, 0.05, 400), 85),
                               ("T3", rng.normal(0, 1, 400), 60)):
        alphas.append(Alpha(id=aid, score=score))
        async with db.write() as s:
            s.add(AlphaPnL(alpha_id=aid, dates=dates, daily=list(series)))
    ordered = await refine.order_for_check(alphas)
    assert [a.id for a in ordered] == ["T1", "T3", "T2"]     # T2 is T1's twin: checked after the different one


# ---------------------------------------------------------------- 1.5 the repair loop

def test_turnover_driven_fitness_gets_a_decay_fix() -> None:
    from alphago.engine.optimizer import plan_variants
    a = Alpha(id="X", code="rank(-ts_delta(close, 5))", settings=BASE, sharpe=1.4, fitness=0.8, turnover=0.25,
              returns=0.08, region="USA", delay=1, checks=checks_for(1.4, 0.8, 0.25))
    kinds = [k for _, _, k in plan_variants(a)]
    assert kinds[0].startswith("decay 0→")          # old code proposed no decay change below 30% turnover


def test_diagnose_measures_gaps_against_brain_limits() -> None:
    a = Alpha(id="X", code="rank(close)", settings=BASE, sharpe=1.0, fitness=0.5, turnover=0.2, returns=0.02,
              region="USA", delay=1, checks=checks_for(1.0, 0.5, 0.2))
    d = refine.diagnose(a)
    assert d.targets == ["LOW_FITNESS", "LOW_SHARPE"]
    assert d.gaps["LOW_FITNESS"] == pytest.approx(0.5) and d.gaps["LOW_SHARPE"] == pytest.approx(0.2)
    assert d.fitness_driver == "sharpe"             # cutting turnover cannot reach Fitness 1.0 here


def banded(expr: str, st: dict) -> tuple[float, float, float]:
    """Decay 0: Fitness 0.6 · decay 1–5: 0.85 (closer, still failing) · decay ≥ 6: 1.1 (passes). Expression
    rewrites make it worse."""
    if expr != "rank(-ts_delta(close, 5))":
        return 1.3, 0.5, 0.6
    d = int(st.get("decay") or 0)
    if d >= 6:
        return 1.5, 1.1, 0.25
    if d >= 1:
        return 1.5, 0.85, 0.45
    return 1.5, 0.6, 0.6


async def test_repair_loop_stacks_fixes_until_strong(dbenv, monkeypatch) -> None:
    fake = FakeSim(banded)
    monkeypatch.setattr(refine, "simulator", fake)
    seed = await add_alpha("P0", "rank(-ts_delta(close, 5))", 1.5, 0.6, 0.6)
    assert seed.verdict == "promising"
    report = await refine.refine([seed], refine.RefineContext(use_llm=False, seed=1, max_depth=3))
    assert report.rounds == 2 and report.strong
    async with db.session() as s:
        strong = await s.get(Alpha, report.strong[0])
        atts = list(await s.scalars(select(OptimizationAttempt).where(OptimizationAttempt.fix == "decay_up")))
        parent = await s.get(Alpha, "P0")
    assert strong.verdict == "strong" and int(strong.decay) >= 6
    assert {a.depth for a in atts} == {1, 2} and all(a.outcome == "success" for a in atts)
    assert parent.expanded_at is not None
    # the loop never expands the same alpha twice (the v1 rounds re-picked the same parents)
    again = await refine.refine([parent], refine.RefineContext(use_llm=False, seed=1))
    assert again.rounds == 0 and again.simulated == 0


async def test_repair_budget_is_respected(dbenv, monkeypatch) -> None:
    fake = FakeSim(banded)
    monkeypatch.setattr(refine, "simulator", fake)
    seed = await add_alpha("P1", "rank(-ts_delta(close, 5))", 1.5, 0.6, 0.6)

    async def three() -> int:
        return 3 - len(fake.sent)

    report = await refine.refine([seed], refine.RefineContext(use_llm=False, seed=2, allowance=three))
    assert len(fake.sent) <= 3 and report.simulated <= 3


async def test_fix_order_follows_recorded_outcomes(dbenv) -> None:
    async with db.write() as s:
        for i in range(20):
            s.add(OptimizationAttempt(parent_alpha_id=f"p{i}", kind="hump", fix="hump", target_check="LOW_FITNESS",
                                      region="USA", outcome="success"))
            s.add(OptimizationAttempt(parent_alpha_id=f"q{i}", kind="decay", fix="decay_up",
                                      target_check="LOW_FITNESS", region="USA", outcome="failure"))
    stats = await refine.load_fix_stats()
    a = Alpha(id="X", code="rank(-ts_delta(close, 5))", settings=BASE, sharpe=1.5, fitness=0.6, turnover=0.4,
              returns=0.1, region="USA", delay=1, checks=checks_for(1.5, 0.6, 0.4))
    d = refine.diagnose(a)
    prior = [f.key for f, _ in refine.rank_fixes(d, refine.FixStats(), "USA", None, allow_llm=False)]
    learned = [f.key for f, _ in refine.rank_fixes(d, stats, "USA", None, allow_llm=False)]
    assert prior.index("decay_up") < prior.index("hump")
    assert learned[0] == "hump" and learned.index("decay_up") > learned.index("hump")
    # Thompson draws still try other fixes sometimes
    firsts = {refine.rank_fixes(d, stats, "USA", random.Random(k), allow_llm=False)[0][0].key for k in range(200)}
    assert len(firsts) > 1


def test_judge_ignores_sharpe_gains_inside_the_noise() -> None:
    snap = {"failing": ["LOW_SHARPE"], "gaps": {"LOW_SHARPE": 0.16}, "values": {"LOW_SHARPE": 1.05},
            "sharpe": 1.05}
    cand = Candidate(status="done", sim_id="S")
    small = Alpha(id="k1", sharpe=1.17, is_checks=checks_for(1.17, 1.1, 0.2))      # closes 60% but +0.12 only
    big = Alpha(id="k2", sharpe=1.22, is_checks=checks_for(1.22, 1.1, 0.2))
    passed = Alpha(id="k3", sharpe=1.3, is_checks=checks_for(1.3, 1.1, 0.2))
    assert refine.judge(snap, "LOW_SHARPE", small, cand) == "failure"
    assert refine.judge(snap, "LOW_SHARPE", big, cand) == "success"
    assert refine.judge(snap, "LOW_SHARPE", passed, cand) == "success"
    assert refine.judge(snap, "LOW_SHARPE", None, Candidate(status="error")) == "neutral"   # pack never started


# ---------------------------------------------------------------- 1.4 duplicate filter, 1.8 idea clusters

async def test_generated_copy_of_a_submitted_alpha_is_left_out(dbenv) -> None:
    await add_alpha("SUB1", "rank(-ts_delta(close, 5))", 1.6, 1.2, 0.2, status="ACTIVE")
    sim = SimulationService()
    gen = await sim.prepare([CandidateSpec("rank(-ts_delta(close, 20))", BASE, origin="llm")])
    manual = await sim.prepare([CandidateSpec("rank(-ts_delta(close, 20))", BASE, origin="manual")])
    other = await sim.prepare([CandidateSpec("rank(-ts_delta(volume, 20))", BASE, origin="llm")])
    assert not gen.created and gen.left_out == {"likely duplicate of submitted alpha SUB1": 1}
    assert len(manual.created) == 1 and len(other.created) == 1


def test_clusters() -> None:
    from alphago.engine.clusters import classify
    assert classify("rank(-ts_delta(close, 5))", fields={"close"}, operators={"rank", "ts_delta"},
                    windows=[5]) == "reversal"
    assert classify("ts_rank(close, 252)", fields={"close"}, operators={"ts_rank"}, windows=[252]) == "momentum"
    assert classify("rank(operating_income / cap)", fields={"operating_income", "cap"},
                    operators={"rank", "divide"}) == "quality"
    assert classify("x", fields={"est_eps"}, category="Analyst") == "analyst"
    assert classify("x", fields={"anything"}, mechanism="post-earnings drift after a surprise") == "event"


async def test_daily_cap_per_idea(dbenv, monkeypatch) -> None:
    monkeypatch.setattr(settings.research, "cluster_min", 2)
    monkeypatch.setattr(settings.research, "cluster_share", 0.01)
    monkeypatch.setattr(settings.research, "autopilot_daily_cap", 100)
    sim = SimulationService()
    specs = [CandidateSpec(f"rank(-ts_delta(close, {w}))", BASE, origin="template") for w in (3, 4, 6, 7)]
    res = await sim.prepare(specs, cluster_cap=True, initial_status="approved")
    assert len(res.created) == 2 and res.left_out == {"idea 'reversal' reached today's share of simulations": 2}
    uncapped = await sim.prepare([CandidateSpec("rank(-ts_delta(close, 8))", BASE, origin="template")])
    assert len(uncapped.created) == 1          # manual work and the Matrix are never capped


# ---------------------------------------------------------------- 1.7 limits, 1.9 dataset survival

async def test_prompt_carries_the_accounts_observed_limits(dbenv) -> None:
    from alphago.knowledge import pack
    await add_alpha("L1", "rank(close)", 1.0, 0.5, 0.2)
    text = await pack.static_prefix("USA", 1)
    assert "Your account's limits as last reported by BRAIN for USA D1" in text and "Sharpe ≥ 1.25" in text
    assert "limits as last reported" not in await pack.static_prefix()


async def test_dataset_yield_rewards_alphas_that_pass_brains_check(dbenv) -> None:
    from alphago.engine.datasets import scored_datasets
    async with db.write() as s:
        for ds in ("dsA", "dsB"):
            s.add(Dataset(id=ds, region="USA", delay=1, universe="TOP3000", name=ds, field_count=10))
    for i in range(10):
        verdict_a = "promising" if i < 5 else "rejected"
        verdict_b = "ready" if i == 0 else "rejected"
        for ds, verdict in (("dsA", verdict_a), ("dsB", verdict_b)):
            aid = f"{ds}{i}"
            async with db.write() as s:
                s.add(Alpha(id=aid, region="USA", delay=1, verdict=verdict))
                s.add(Candidate(expression="x", settings=BASE, settings_hash="h", region="USA", delay=1,
                                origin="search", status="done", alpha_id=aid, dataset_id=ds))
    async with db.write() as s:            # repair children are not new tries
        s.add(Candidate(expression="y", settings=BASE, settings_hash="h", region="USA", delay=1, origin="optimize",
                        status="done", alpha_id="dsA0", dataset_id="dsA"))
    scores = {d.id: d for d in await scored_datasets("USA", 1, "TOP3000")}
    assert scores["dsA"].tried == 10 and scores["dsB"].tried == 10
    assert scores["dsB"].good > scores["dsA"].good          # 1 submittable beats 5 merely promising
    assert scores["dsB"].passed == 1 and scores["dsA"].passed == 0


# ---------------------------------------------------------------- a research task end to end

async def test_research_task_repairs_and_certifies_within_budget(dbenv, monkeypatch) -> None:
    from alphago.db.models import DataField
    from alphago.engine import catalogs, research as research_mod
    from alphago.engine.research import research
    from alphago.llm import gemini

    catalogs.invalidate()
    monkeypatch.setattr(gemini, "configure", lambda *a, **k: False)
    async with db.write() as s:
        s.add(Dataset(id="fund1", region="USA", delay=1, universe="TOP3000", name="Fundamentals",
                      category_name="Fundamental", field_count=2))
        for f in ("operating_income", "sales"):
            s.add(DataField(id=f, dataset_id="fund1", region="USA", delay=1, universe="TOP3000", type="MATRIX",
                            coverage=0.9, alpha_count=10))

    def score(expr: str, st: dict) -> tuple[float, float, float]:
        if "operating_income" in expr and expr.startswith("ts_rank(") and ", 66)" in expr:
            return banded("rank(-ts_delta(close, 5))", st)       # one near miss that decay can repair
        return 0.3, 0.1, 0.3

    fake = FakeSim(score)
    monkeypatch.setattr(refine, "simulator", fake)
    monkeypatch.setattr(research_mod, "simulator", fake)
    checked: list[str] = []

    async def fake_run_check(alpha_id):  # noqa: ANN001, ANN202
        checked.append(alpha_id)
        async with db.write() as s:
            a = await s.get(Alpha, alpha_id)
            a.check_state, a.checked_at = "pass", datetime.now(timezone.utc)
            from alphago.engine.evaluator import reverdict
            reverdict(a)
        return "pass", []

    async def no_estimate(alpha_id, fetch=True):  # noqa: ANN001, ANN202
        return None, None

    monkeypatch.setattr(actions, "run_check", fake_run_check)
    monkeypatch.setattr("alphago.engine.similarity.estimate_self_corr", no_estimate)
    tid = await research.create(kind="single", region="USA", delay=1, universe="TOP3000", dataset_ids=["fund1"],
                                planned_sims=60, cores=2)
    await research._run_inner(tid)
    t = await research.get(tid)
    async with db.session() as s:
        cands = list(await s.scalars(select(Candidate).where(Candidate.task_id == tid)))
    first = [c for c in cands if c.origin not in ("optimize", "robustness") and c.status == "done"]
    repair = [c for c in cands if c.origin in ("optimize", "robustness") and c.status == "done"]
    assert t.status == "complete" and t.n_ready >= 1 and checked
    assert len(first) <= 48 and len(repair) <= 12             # 20% of the 60 planned simulations for repairs
    assert len(fake.sent) <= 60
