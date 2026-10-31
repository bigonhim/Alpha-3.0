"""Targeted repair: the road to Ready cannot jam or lie, near misses get moves computed for their actual cause, and repair
tasks spend only their own budget. Each test fails on the code before this change."""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from alphago.config import settings
from alphago.db import db
from alphago.db.models import Alpha, Candidate, OptimizationAttempt
from alphago.engine import actions, playbook, refine
from alphago.engine.evaluator import check_state_of, evaluate
from alphago.engine.optimizer import plan_variants
from alphago.engine.simulator import CandidateSpec, SimulationService

from .test_refine import BASE, FakeSim, add_alpha, alpha_json, checks_for, dbenv  # noqa: F401

NOW = datetime.now(timezone.utc)


def stub_check(monkeypatch, *replies) -> list[str]:  # noqa: ANN001, ANN002
    """brain.alphas.check answers from a list, repeating the last reply. Returns the alpha ids it was asked about."""
    asked: list[str] = []
    queue = list(replies)

    async def fake(alpha_id, max_wait_s=900):  # noqa: ANN001, ANN202
        asked.append(alpha_id)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(actions.brain.alphas, "check", fake)
    monkeypatch.setattr(actions, "PENDING_WAIT_S", 0.0)
    return asked


def passing(extra: list[dict] | None = None) -> list[dict]:
    return checks_for(1.6, 1.2, 0.2)[:-1] + [{"name": "SELF_CORRELATION", "result": "PASS", "limit": 0.7,
                                              "value": 0.3}] + (extra or [])


def near_miss(sharpe: float, fitness: float, turnover: float, returns: float, *, st: dict | None = None,
              extra: list[dict] | None = None, code: str = "rank(-ts_delta(close, 5))", **kw) -> Alpha:
    return Alpha(id="X", code=code, settings={**BASE, **(st or {})}, sharpe=sharpe, fitness=fitness, turnover=turnover,
                 returns=returns, region="USA", delay=1, universe="TOP3000",
                 checks=checks_for(sharpe, fitness, turnover, extra), **kw)


# ---------------------------------------------------------------- the road to Ready

async def test_empty_check_reply_is_never_a_pass(dbenv, monkeypatch) -> None:  # noqa: F811
    await add_alpha("E1", "rank(-ts_delta(close, 5))", 0.2, 0.1, 0.2)
    stub_check(monkeypatch, [])
    state, _ = await actions.run_check("E1")
    async with db.session() as s:
        a = await s.get(Alpha, "E1")
    assert state == "pending" and a.verdict != "ready" and a.check_tries == 1


async def test_empty_reply_keeps_the_earlier_result(dbenv, monkeypatch) -> None:  # noqa: F811
    await add_alpha("E2", "rank(-ts_delta(close, 7))", 1.6, 1.2, 0.2, check_state="stale", checked_at=NOW,
                    checks=passing())
    stub_check(monkeypatch, [])
    await actions.run_check("E2")
    assert await actions.rederive_check_states() == 0
    async with db.session() as s:
        a = await s.get(Alpha, "E2")
    assert a.check_state == "stale" and a.verdict == "strong" and a.checks     # old: an empty body read as a pass


async def test_startup_does_not_promote_pending_to_pass(dbenv) -> None:  # noqa: F811
    await add_alpha("E3", "rank(-ts_delta(close, 8))", 1.6, 1.2, 0.2, check_state="pending", checked_at=NOW,
                    checks=passing())
    assert await actions.rederive_check_states() == 0
    async with db.session() as s:
        assert (await s.get(Alpha, "E3")).verdict == "strong"


async def test_old_empty_pass_is_undone_at_startup(dbenv) -> None:  # noqa: F811
    await add_alpha("E4", "rank(-ts_delta(close, 9))", 0.2, 0.1, 0.2, check_state="pass", checked_at=NOW, checks=[])
    assert await actions.rederive_check_states() == 1
    async with db.session() as s:
        a = await s.get(Alpha, "E4")
    assert a.check_state is None and a.checks == a.is_checks and a.verdict != "ready"


def test_numbers_decide_when_brain_gave_no_answer() -> None:
    assert check_state_of([]) == "pending"
    assert evaluate(sharpe=0.2, fitness=0.1, turnover=0.2, checks=[], full_check_passed=True).verdict != "ready"
    broken = [{"name": "LOW_SHARPE", "result": "ERROR"}, {"name": "LOW_FITNESS", "result": "PASS", "limit": 1.0}]
    assert evaluate(sharpe=0.2, fitness=1.1, turnover=0.2, checks=broken).verdict not in ("strong", "ready")
    assert evaluate(sharpe=1.6, fitness=1.2, turnover=0.2, checks=broken).verdict == "strong"


async def test_brain_error_on_a_check_is_unanswered_not_failed(dbenv, monkeypatch) -> None:  # noqa: F811
    await add_alpha("B1", "rank(-ts_delta(close, 10))", 1.6, 1.2, 0.2)
    errored = passing()[:-1] + [{"name": "SELF_CORRELATION", "result": "ERROR"}]
    stub_check(monkeypatch, errored)
    state, _ = await actions.run_check("B1")
    async with db.session() as s:
        a = await s.get(Alpha, "B1")
    assert state == "pending" and a.verdict == "strong" and a.check_tries == 1     # old: "fail", demoted for good


async def test_old_error_fails_come_free_at_startup(dbenv) -> None:  # noqa: F811
    await add_alpha("B2", "rank(-ts_delta(close, 11))", 1.6, 1.2, 0.2, check_state="fail", checked_at=NOW,
                    checks=passing()[:-1] + [{"name": "SELF_CORRELATION", "result": "ERROR"}])
    assert await actions.rederive_check_states() == 1
    async with db.session() as s:
        a = await s.get(Alpha, "B2")
    assert a.check_state == "pending" and a.verdict == "strong"


async def test_daily_limit_check_does_not_block_ready(dbenv, monkeypatch) -> None:  # noqa: F811
    await add_alpha("L1", "rank(-ts_delta(close, 12))", 1.6, 1.2, 0.2)
    stub_check(monkeypatch, passing([{"name": "REGULAR_SUBMISSION", "result": "FAIL", "limit": 4, "value": 4}]))
    state, _ = await actions.run_check("L1")
    async with db.session() as s:
        assert state == "pass" and (await s.get(Alpha, "L1")).verdict == "ready"


async def test_pending_answers_are_polled_before_giving_up(dbenv, monkeypatch) -> None:  # noqa: F811
    await add_alpha("P1", "rank(-ts_delta(close, 13))", 1.6, 1.2, 0.2)
    waiting = passing()[:-1] + [{"name": "SELF_CORRELATION", "result": "PENDING"}]
    asked = stub_check(monkeypatch, waiting, waiting, passing())
    state, _ = await actions.run_check("P1")
    assert state == "pass" and len(asked) == 3


async def test_recheck_finishes_unanswered_checks_then_stops(dbenv) -> None:  # noqa: F811
    old = NOW - timedelta(hours=2)
    for aid, tries, when in (("R1", 1, old), ("R2", actions.MAX_CHECK_TRIES, old), ("R3", 1, NOW)):
        await add_alpha(aid, f"rank(-ts_delta(close, {3 + tries}))", 1.6, 1.2, 0.2, check_state="pending",
                        checked_at=when, check_tries=tries, st={"decay": len(aid) + tries + int(aid[1])})
    assert await actions.recheck_due() == ["R1"]       # R2 used its tries, R3 was asked just now


async def test_recheck_counts_a_failed_call_as_a_try(dbenv, monkeypatch) -> None:  # noqa: F811
    from alphago.brain.errors import BrainError
    from alphago.engine.simulator import simulator
    await add_alpha("R5", "rank(-ts_delta(close, 15))", 1.6, 1.2, 0.2, check_state="pending",
                    checked_at=NOW - timedelta(hours=2), check_tries=2)

    async def down(alpha_id):  # noqa: ANN001, ANN202
        raise BrainError("BRAIN is down")

    async def stop(_s):  # noqa: ANN001, ANN202
        if stop.calls:
            raise asyncio.CancelledError
        stop.calls += 1

    stop.calls = 0
    monkeypatch.setattr(actions, "run_check", down)
    monkeypatch.setattr(actions.asyncio, "sleep", stop)
    monkeypatch.setattr(type(actions.brain.client), "logged_in", property(lambda self: True))
    monkeypatch.setattr(simulator, "paused", False)
    with pytest.raises(asyncio.CancelledError):
        await actions.recheck_loop()
    assert await actions.recheck_due() == []             # third try used, and it was asked just now


async def test_certify_stops_asking_after_the_tries_run_out(dbenv) -> None:  # noqa: F811
    a = await add_alpha("R4", "rank(-ts_delta(close, 14))", 1.6, 1.2, 0.2, check_state="pending",
                        checked_at=NOW - timedelta(hours=2), check_tries=actions.MAX_CHECK_TRIES)
    assert not refine._needs_check(a, NOW)


# ---------------------------------------------------------------- measuring and naming the cause

def test_decay_search_rises_with_turnover() -> None:
    picks = []
    for t in (0.75, 0.9, 1.1, 1.4):
        a = near_miss(1.6, 1.0, t, 0.14, extra=[{"name": "HIGH_TURNOVER", "result": "FAIL", "limit": 0.7, "value": t}])
        picks.append(playbook.decay_steps(refine.diagnose(a).profile, [playbook.HT])[0])
    assert picks == sorted(picks) and picks[-1] > picks[0]        # old table: 37% → decay 4, 45% → decay 2
    assert playbook.solve_decay(0.6, 0, 0.3) == 4 and playbook.solve_decay(0.6, 0, 0.6) == 0


def test_decay_stops_where_sharpe_would_fall_through_its_limit() -> None:
    p = refine.diagnose(near_miss(1.3, 0.8, 0.5, 0.1)).profile
    n = playbook.decay_steps(p, [playbook.F])
    forecast = playbook.forecast(p, playbook.effect_of("decay_up", p, [playbook.F]))
    assert n and forecast.sharpe >= p.sharpe_limit * 0.95


def test_own_decay_curve_replaces_the_default() -> None:
    e = playbook.fit_elasticity([(0, 0.6, 1.8, 0.14), (8, 0.2, 1.6, 0.1)])
    assert e.fitted and e.beta == pytest.approx(0.5, abs=0.01) and 0 < e.sigma < 0.2
    assert not playbook.fit_elasticity([(4, 0.3, 1.5, 0.1)]).fitted


def test_fitness_under_the_turnover_floor_is_never_a_turnover_problem() -> None:
    a = near_miss(1.5, 0.8, 0.06, 0.018, st={"decay": 8})
    d = refine.diagnose(a)
    kinds = [k for _, _, k in plan_variants(a)]
    assert d.profile.causes["LOW_FITNESS"] == "returns"
    assert kinds and not any(k.startswith(("hump", "ts_decay", "trade_when")) for k in kinds)
    decays = [int(k.split("→")[1]) for k in kinds if k.startswith("decay 8→")]
    assert decays and max(decays) < 8                           # spends the turnover Fitness does not charge for
    spent = playbook.free_turnover_decay(d.profile)
    assert spent is not None and spent < 8


def test_fitness_cause_follows_the_numbers() -> None:
    e = playbook.Elasticity()
    assert playbook.fitness_cause(1.9, 0.14, 0.6, 1.0, 1.25, e) == "turnover"
    assert playbook.fitness_cause(1.5, 0.018, 0.06, 1.0, 1.25, e) == "returns"
    assert playbook.fitness_cause(1.0, 0.02, 0.2, 1.0, 1.25, e) == "sharpe"


def test_low_turnover_at_decay_zero_gets_a_plan() -> None:
    a = Alpha(id="Y", code="rank(sales / assets)", settings=BASE, sharpe=1.5, fitness=1.1, turnover=0.005,
              returns=0.05, region="USA", delay=1,
              checks=checks_for(1.5, 1.1, 0.005)[:2] + [{"name": "LOW_TURNOVER", "result": "FAIL", "limit": 0.01,
                                                         "value": 0.005}])
    assert plan_variants(a)                              # old: no plan at all


def test_sub_universe_failure_gets_liquidity_moves() -> None:
    fail = [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL", "limit": 0.69, "value": 0.35}]
    a = near_miss(1.6, 1.2, 0.3, 0.1, extra=fail)
    d = refine.diagnose(a)
    kinds = [k for _, _, k in plan_variants(a)]
    assert d.profile.causes["LOW_SUB_UNIVERSE_SHARPE"] == "illiquid_edge"
    assert d.profile.sub_ratio == pytest.approx(0.35 / 1.6)
    assert any("liquidity" in k or "universe" in k or "size" in k for k in kinds)


def test_unknown_checks_are_read_from_their_name() -> None:
    assert playbook.check_family("LOW_ROBUST_UNIVERSE_SHARPE") == "universe"
    assert playbook.check_family("LOW_INVESTABILITY_CONSTRAINED_SHARPE") == "universe"
    assert playbook.check_family("LOW_GLB_EMEA_SHARPE") == "signal"
    assert playbook.check_family("SOMETHING_NEW") == "unknown"
    keys = {f.key for f in refine.fixes_for("LOW_ROBUST_UNIVERSE_SHARPE")}
    assert "liquid_tilt" in keys                         # old: generic fixes only


def test_sparse_signal_is_recognised_and_backfilled() -> None:
    a = near_miss(1.1, 0.7, 0.2, 0.08, long_count=300, short_count=300, code="rank(sales / assets)")
    d = refine.diagnose(a)
    assert d.profile.coverage == pytest.approx(0.2) and d.profile.causes["LOW_SHARPE"] == "sparse"
    assert any("backfill" in k for _, _, k in plan_variants(a))
    full = near_miss(1.1, 0.7, 0.2, 0.08, long_count=1500, short_count=1450, code="rank(sales / assets)")
    assert not any("backfill" in k for _, _, k in plan_variants(full))


def test_yearly_shape_tells_fading_from_one_bad_year() -> None:
    rng = random.Random(1)
    dates = [f"{y}-{m:02d}-{d:02d}" for y in range(2019, 2025) for m in range(1, 13) for d in range(1, 22)][:1260]
    fading = [rng.gauss(0.002 if i < 756 else -0.0002, 0.006) for i in range(1260)]
    assert playbook.yearly_shape(dates, fading).shape == "recent_decay"
    bad = [rng.gauss(-0.003 if d.startswith("2021") else 0.002, 0.006) for d in dates]
    assert playbook.yearly_shape(dates, bad).shape == "one_bad_year"
    assert playbook.yearly_shape([], []).shape == "unknown"


def test_prescription_gives_the_numbers_each_check_needs() -> None:
    d = refine.diagnose(near_miss(1.4, 0.8, 0.25, 0.08))
    text = "\n".join(playbook.prescription(d.failing, d.gaps, d.values, d.profile, d.limits))
    assert "LOW_FITNESS" in text and "15.7%" in text and "turnover" in text


def test_correlation_waits_until_the_metrics_pass() -> None:
    fail = [{"name": "SELF_CORRELATION", "result": "FAIL", "limit": 0.7, "value": 0.85}]
    a = near_miss(1.0, 1.2, 0.2, 0.1, extra=fail)
    d = refine.diagnose(a)
    assert "SELF_CORRELATION" in d.failing and d.targets == ["LOW_SHARPE"]


# ---------------------------------------------------------------- choosing and judging moves

def test_judge_accepts_a_fix_that_costs_a_hair_of_sharpe() -> None:
    snap = {"failing": ["LOW_FITNESS"], "gaps": {"LOW_FITNESS": 0.3}, "values": {"LOW_FITNESS": 0.7}, "sharpe": 1.28,
            "distance": 0.3}
    child = Alpha(id="k", sharpe=1.22, fitness=1.02, turnover=0.2, returns=0.1, is_checks=checks_for(1.22, 1.02, 0.2))
    assert refine.judge(snap, "LOW_FITNESS", child, Candidate(status="done", sim_id="S")) == "success"


def test_judge_rejects_a_fix_that_trades_one_failure_for_a_bigger_one() -> None:
    snap = {"failing": ["LOW_FITNESS"], "gaps": {"LOW_FITNESS": 0.1}, "values": {"LOW_FITNESS": 0.9}, "sharpe": 1.3,
            "distance": 0.1}
    child = Alpha(id="k", sharpe=0.8, fitness=1.02, turnover=0.2, returns=0.1, is_checks=checks_for(0.8, 1.02, 0.2))
    assert refine.judge(snap, "LOW_FITNESS", child, Candidate(status="done", sim_id="S")) == "failure"


def test_judge_waits_when_the_simulation_does_not_report_the_check() -> None:
    snap = {"failing": ["LOW_2Y_SHARPE"], "gaps": {"LOW_2Y_SHARPE": 0.3}, "values": {}, "sharpe": 1.5}
    child = Alpha(id="k", sharpe=1.5, fitness=1.2, turnover=0.2, returns=0.1, is_checks=checks_for(1.5, 1.2, 0.2))
    assert refine.judge(snap, "LOW_2Y_SHARPE", child, Candidate(status="done", sim_id="S")) is None  # old: "success"


def test_judge_waits_on_a_brain_error() -> None:
    cand = Candidate(status="done", sim_id="S")
    snap = {"failing": ["SELF_CORRELATION"], "gaps": {"SELF_CORRELATION": 0.2}, "sharpe": 1.5}
    child = Alpha(id="k", sharpe=1.5, is_checks=checks_for(1.5, 1.2, 0.2), checked_at=NOW,
                  checks=[{"name": "SELF_CORRELATION", "result": "ERROR"}])
    assert refine.judge(snap, "SELF_CORRELATION", child, cand) is None             # old: "failure"


def test_beam_needs_real_progress_not_noise() -> None:
    base = refine.diagnose(near_miss(1.05, 1.2, 0.2, 0.1))
    noise = refine.diagnose(near_miss(1.14, 1.2, 0.2, 0.1))
    real = refine.diagnose(near_miss(1.22, 1.2, 0.2, 0.1))
    assert not refine.progressed(base, noise) and refine.progressed(base, real)


def test_weak_alpha_with_a_passing_sharpe_is_a_seed() -> None:
    fast = near_miss(1.6, 0.45, 0.6, 0.05, verdict="weak")
    dull = near_miss(0.9, 0.45, 0.6, 0.05, verdict="weak")
    corr = near_miss(1.6, 1.2, 0.2, 0.1, verdict="promising",
                     extra=[{"name": "SELF_CORRELATION", "result": "FAIL", "limit": 0.7, "value": 0.8}])
    done = near_miss(1.6, 0.9, 0.2, 0.1, verdict="promising", repair_state="exhausted")
    assert refine.repairable(fast) and not refine.repairable(dull)     # old: "weak" was never repaired
    assert not refine.repairable(corr) and not refine.repairable(done)


def test_alpha_with_no_edge_in_liquid_names_is_left_alone() -> None:
    hopeless = [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL", "limit": 0.6, "value": 0.1}]
    reachable = [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL", "limit": 0.6, "value": 0.45}]
    assert not refine.repairable(near_miss(1.3, 0.9, 0.2, 0.1, verdict="promising", extra=hopeless))
    assert refine.repairable(near_miss(1.3, 0.9, 0.2, 0.1, verdict="promising", extra=reachable))
    assert playbook.beyond_reach({"LOW_2Y_SHARPE": 1.2}) and not playbook.beyond_reach({"LOW_2Y_SHARPE": 0.4})


def test_closest_first_orders_by_distance_to_passing() -> None:
    near = near_miss(1.3, 0.95, 0.2, 0.1, verdict="promising")
    far = near_miss(1.0, 0.6, 0.2, 0.1, verdict="promising")
    assert refine.closest_first([far, near]) == [near, far]


def stacked(expr: str, st: dict) -> tuple[float, float, float]:
    """Needs two moves of different kinds: each alone leaves Fitness short."""
    if expr != "rank(-ts_delta(close, 5))":
        return 0.9, 0.4, 0.3
    fitness = 0.6 + (0.28 if int(st.get("decay") or 0) >= 1 else 0.0) + (
        0.28 if st.get("neutralization") != "SUBINDUSTRY" else 0.0)
    return 1.5, fitness, 0.3


async def test_two_moves_stack_across_rounds(dbenv, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(refine, "simulator", FakeSim(stacked))
    seed = await add_alpha("S0", "rank(-ts_delta(close, 5))", 1.5, 0.6, 0.3)
    report = await refine.refine([seed], refine.RefineContext(use_llm=False, seed=1, max_depth=3))
    assert report.rounds == 2 and report.strong
    async with db.session() as s:
        best = await s.get(Alpha, report.strong[0])
    assert int(best.decay) >= 1 and best.neutralization != "SUBINDUSTRY"


def test_two_failing_checks_share_one_variant() -> None:
    fail = [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "FAIL", "limit": 0.69, "value": 0.35}]
    a = near_miss(1.0, 0.7, 0.3, 0.07, extra=fail)
    d = refine.diagnose(a)
    env = refine.FixEnv("USA", ["SUBINDUSTRY", "INDUSTRY", "SECTOR"], random.Random(0), universes=["TOP3000", "TOP1000"])
    first = refine.Variant(a.code, {**BASE, "neutralization": "INDUSTRY"}, "neutralize_group", "LOW_SHARPE", "n",
                           "weak_signal")
    second = refine.Variant("x", BASE, "liquid_tilt", "LOW_SUB_UNIVERSE_SHARPE", "t", "illiquid_edge")
    both = refine._compound([first, second], d, env)
    assert both is not None and both.fix == "compound" and both.settings["neutralization"] == "INDUSTRY"
    assert "adv20" in both.expr


# ---------------------------------------------------------------- spending simulations

async def test_a_queued_repair_is_not_paid_for_twice(dbenv) -> None:  # noqa: F811
    sim = SimulationService()
    spec = CandidateSpec("rank(-ts_delta(close, 5))", BASE, origin="optimize")
    first = await sim.prepare([spec], initial_status="approved")
    second = await sim.prepare([spec], initial_status="approved")
    assert len(first.created) == 1 and not second.created
    assert second.left_out == {"already queued for simulation": 1}


async def test_revisit_never_repeats_a_move_and_stops_when_none_are_left(dbenv, monkeypatch) -> None:  # noqa: F811
    fake = FakeSim(lambda e, st: (0.9, 0.4, 0.3))
    monkeypatch.setattr(refine, "simulator", fake)
    seed = await add_alpha("V0", "rank(-ts_delta(close, 5))", 1.3, 0.8, 0.3)
    await refine.refine([seed], refine.RefineContext(use_llm=False, seed=1, max_depth=1))
    spent = len(fake.sent)
    async with db.session() as s:
        a = await s.get(Alpha, "V0")
    sent_before = {(e, str(st.get("decay")), st.get("neutralization")) for e, st in fake.sent}
    for _ in range(12):
        async with db.session() as s:
            a = await s.get(Alpha, "V0")
        report = await refine.refine([a], refine.RefineContext(use_llm=False, seed=2, max_depth=1, revisit=True))
        if not report.attempts:
            break
    else:
        pytest.fail("revisiting never ran out of moves")
    keys = [(e, str(st.get("decay")), st.get("neutralization")) for e, st in fake.sent[spent:]]
    assert len(keys) == len(set(keys)) and not set(keys) & sent_before


async def test_exhausted_alpha_says_why_and_leaves_the_queue(dbenv, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(refine, "simulator", FakeSim(lambda e, st: (0.8, 0.3, 0.3)))
    seed = await add_alpha("X0", "rank(-ts_delta(close, 5))", 1.3, 0.8, 0.3)
    await refine.refine([seed], refine.RefineContext(use_llm=False, seed=1, max_depth=2))
    async with db.session() as s:
        a = await s.get(Alpha, "X0")
    assert a.repair_state == "exhausted" and a.repair_note and not refine.repairable(a)


async def test_budget_cut_does_not_mark_exhausted(dbenv, monkeypatch) -> None:  # noqa: F811
    fake = FakeSim(lambda e, st: (0.8, 0.3, 0.3))
    monkeypatch.setattr(refine, "simulator", fake)
    seed = await add_alpha("X1", "rank(-ts_delta(close, 5))", 1.3, 0.8, 0.3)

    async def two() -> int:
        return 2 - len(fake.sent)

    await refine.refine([seed], refine.RefineContext(use_llm=False, seed=1, allowance=two))
    async with db.session() as s:
        assert (await s.get(Alpha, "X1")).repair_state is None


async def test_repair_resumes_from_the_closest_child(dbenv) -> None:  # noqa: F811
    seed = await add_alpha("Z0", "rank(-ts_delta(close, 5))", 1.3, 0.7, 0.3)
    await add_alpha("Z1", "rank(-ts_delta(close, 6))", 1.3, 0.95, 0.3)
    async with db.write() as s:
        s.add(OptimizationAttempt(parent_alpha_id="Z0", kind="decay", fix="decay_up", target_check="LOW_FITNESS",
                                  root_alpha_id="Z0", child_alpha_id="Z1", outcome="success"))
    node = await refine._resume_node(seed)
    assert node.alpha.id == "Z1" and node.root == "Z0"


# ---------------------------------------------------------------- one budget across many near misses

def near_ones_pass(expr: str, st: dict) -> tuple[float, float, float]:
    """Only the expression seeded nearest to passing can be repaired; decay does it."""
    if "ts_delta(close, 5)" in expr and int(st.get("decay") or 0) >= 1:
        return 1.6, 1.2, 0.2
    return 0.9, 0.4, 0.3


async def test_best_first_spends_on_the_alpha_closest_to_passing(dbenv, monkeypatch) -> None:  # noqa: F811
    fake = FakeSim(near_ones_pass)
    monkeypatch.setattr(refine, "simulator", fake)
    far = await add_alpha("B1", "rank(-ts_delta(close, 9))", 1.0, 0.55, 0.3)
    near = await add_alpha("B2", "rank(-ts_delta(close, 5))", 1.3, 0.9, 0.3)

    async def eight() -> int:
        return 8 - len(fake.sent)

    report = await refine.best_first([far, near], refine.RefineContext(use_llm=False, seed=1, allowance=eight),
                                     width=1, fanout=4)
    assert report.strong and len(fake.sent) <= 8
    assert "ts_delta(close, 5)" in fake.sent[0][0]        # near first, whatever order the seeds came in


async def test_best_first_stops_on_a_seed_once_brain_passes_it(dbenv, monkeypatch) -> None:  # noqa: F811
    fake = FakeSim(near_ones_pass)
    monkeypatch.setattr(refine, "simulator", fake)
    near = await add_alpha("B3", "rank(-ts_delta(close, 5))", 1.3, 0.9, 0.3)

    async def passes(ids):  # noqa: ANN001, ANN202
        return refine.CertifyReport(checked=ids, passed=ids)

    report = await refine.best_first([near], refine.RefineContext(use_llm=False, seed=1), check=passes, fanout=4)
    assert report.rounds == 1 and len(fake.sent) <= 4


async def test_best_first_keeps_trying_when_brain_fails_the_child(dbenv, monkeypatch) -> None:  # noqa: F811
    fake = FakeSim(near_ones_pass)
    monkeypatch.setattr(refine, "simulator", fake)
    near = await add_alpha("B4", "rank(-ts_delta(close, 5))", 1.3, 0.9, 0.3)

    async def fails(ids):  # noqa: ANN001, ANN202
        return refine.CertifyReport(checked=ids)

    report = await refine.best_first([near], refine.RefineContext(use_llm=False, seed=1), check=fails, fanout=2)
    assert report.rounds > 1                              # the seed is not finished until BRAIN says so


async def test_best_first_respects_the_budget_and_stops_when_moves_run_out(dbenv, monkeypatch) -> None:  # noqa: F811
    fake = FakeSim(lambda e, st: (0.8, 0.3, 0.3))
    monkeypatch.setattr(refine, "simulator", fake)
    a = await add_alpha("B5", "rank(-ts_delta(close, 9))", 1.3, 0.8, 0.3)

    async def three() -> int:
        return 3 - len(fake.sent)

    report = await refine.best_first([a], refine.RefineContext(use_llm=False, seed=1, allowance=three))
    assert len(fake.sent) <= 3 and report.stopped == "budget"
    async with db.session() as s:
        assert (await s.get(Alpha, "B5")).repair_state is None      # cut short by budget, not out of ideas
        fresh = await s.get(Alpha, "B5")
    await refine.best_first([fresh], refine.RefineContext(use_llm=False, seed=1))
    async with db.session() as s:
        assert (await s.get(Alpha, "B5")).repair_state == "exhausted"


async def test_best_first_never_expands_a_residual_child(dbenv, monkeypatch) -> None:  # noqa: F811
    assert "residualize" in refine.TERMINAL


# ---------------------------------------------------------------- the repair task

async def scoped(tid: int) -> list[Candidate]:
    async with db.session() as s:
        return list(await s.scalars(select(Candidate).where(Candidate.task_id == tid)))


async def test_repair_task_turns_existing_alphas_ready_within_budget(dbenv, monkeypatch) -> None:  # noqa: F811
    from alphago.engine import research as research_mod
    from alphago.engine.research import KINDS, SEARCH_KINDS, research, task_label
    from alphago.llm import gemini

    assert "repair" in KINDS and "repair" not in SEARCH_KINDS          # never offered to the planner or the Autopilot
    monkeypatch.setattr(gemini, "configure", lambda *a, **k: False)

    def score(expr: str, st: dict) -> tuple[float, float, float]:
        return (1.6, 1.2, 0.2) if int(st.get("decay") or 0) >= 1 else (1.0, 0.5, 0.3)

    fake = FakeSim(score)
    monkeypatch.setattr(refine, "simulator", fake)
    monkeypatch.setattr(research_mod, "simulator", fake)
    await add_alpha("T1", "rank(-ts_delta(close, 5))", 1.6, 1.2, 0.2, st={"decay": 9})   # strong from an old task
    await add_alpha("T2", "rank(-ts_delta(close, 6))", 1.3, 0.8, 0.3)                    # near miss
    checked: list[str] = []

    async def passes(alpha_id):  # noqa: ANN001, ANN202
        checked.append(alpha_id)
        return passing()

    async def no_estimate(alpha_id, fetch=True):  # noqa: ANN001, ANN202
        return None, None

    monkeypatch.setattr(actions.brain.alphas, "check", lambda aid, max_wait_s=900: passes(aid))
    monkeypatch.setattr("alphago.engine.similarity.estimate_self_corr", no_estimate)
    tid = await research.create(kind="repair", region="USA", delay=1, universe="TOP3000", dataset_ids=[],
                                planned_sims=30, cores=2)
    await research._run_inner(tid)
    t = await research.get(tid)
    cands = await scoped(tid)
    spent = [c for c in cands if c.status == "done"]
    assert t.status == "complete" and t.n_ready >= 2 and "T1" in checked and len(fake.sent) <= 30
    assert {c.alpha_id for c in cands if c.origin == "seed"} >= {"T1", "T2"} and len(spent) == len(fake.sent)
    assert task_label(t).startswith("Repair near misses")


async def test_repair_task_defaults_to_its_own_budget(dbenv) -> None:  # noqa: F811
    from alphago.engine.research import research
    tid = await research.create(kind="repair", region="USA", delay=1, universe="TOP3000", dataset_ids=[],
                                alpha_ids=["A1"])
    t = await research.get(tid)
    assert t.planned_sims == settings.research.repair_task_sims and t.alpha_ids == ["A1"]


async def test_repair_child_counts_against_the_search_that_found_it(dbenv) -> None:  # noqa: F811
    from alphago.engine.honesty import search_size
    from alphago.engine.research import research
    tid = await research.create(kind="single", region="USA", delay=1, universe="TOP3000", dataset_ids=["d"], status="done")
    fix = await research.create(kind="repair", region="USA", delay=1, universe="TOP3000", dataset_ids=[])
    await add_alpha("H1", "rank(-ts_delta(close, 5))", 1.3, 0.8, 0.3)
    await add_alpha("H2", "rank(-ts_delta(close, 6))", 1.6, 1.2, 0.3)
    async with db.write() as s:
        c1 = Candidate(expression="a", settings=BASE, settings_hash="h", region="USA", delay=1, origin="search",
                       status="done", alpha_id="H1", task_id=tid)
        extra = [Candidate(expression=f"e{i}", settings=BASE, settings_hash=f"h{i}", region="USA", delay=1,
                           origin="search", status="done", alpha_id="H1", task_id=tid) for i in range(9)]
        c2 = Candidate(expression="b", settings=BASE, settings_hash="h2", region="USA", delay=1, origin="optimize",
                       status="done", alpha_id="H2", task_id=fix, rationale={"root": "H1"})
        s.add_all([c1, c2, *extra])
        await s.flush()
        (await s.get(Alpha, "H1")).candidate_id = c1.id
        (await s.get(Alpha, "H2")).candidate_id = c2.id
    assert (await search_size("H2")).n_eff == 10       # old: a search of one, so a lucky repair looked certain


# ---------------------------------------------------------------- the pipeline report

async def test_pipeline_names_what_is_holding_alphas_back(dbenv) -> None:  # noqa: F811
    from alphago.engine.funnel import pipeline, shareable
    await add_alpha("F1", "rank(-ts_delta(close, 5))", 1.6, 1.2, 0.2)
    await add_alpha("F2", "rank(-ts_delta(close, 6))", 1.3, 0.8, 0.3)
    p = await pipeline("USA", 1)
    assert p["never_checked"] == 1 and "never had BRAIN's check" in p["blocker"]
    assert p["failing"][0]["check"] == "LOW_FITNESS" and p["causes"]
    text = str(shareable(p))
    assert "F1" not in text and "ts_delta" not in text


async def test_pipeline_points_at_ready_alphas_first(dbenv) -> None:  # noqa: F811
    from alphago.engine.funnel import pipeline
    await add_alpha("F3", "rank(-ts_delta(close, 7))", 1.6, 1.2, 0.2, check_state="pass", checked_at=NOW,
                    checks=passing())
    assert "ready for your review" in (await pipeline("USA", 1))["blocker"]


async def test_alpha_json_helper_has_the_fields_the_plan_reads() -> None:
    j = alpha_json("Q", "rank(close)", BASE, 1.0, 0.5, 0.2)
    assert j["is"]["longCount"] and j["is"]["returns"]
