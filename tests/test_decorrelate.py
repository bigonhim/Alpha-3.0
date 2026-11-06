"""Strong to Ready: an alpha that passes every metric but collides with an existing alpha. Each test fails on the code
before this change."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from alphago.db import db
from alphago.db.models import Alpha, Candidate
from alphago.engine import actions, refine
from alphago.engine import decorrelate as deco
from alphago.engine.decorrelate import Collision, find_collision

from .test_refine import BASE, FakeSim, add_alpha, checks_for, dbenv  # noqa: F401

SUBMITTED = "x = rank(volume); ts_mean(x, 5)"


def collided(value: float = 0.85) -> list[dict]:
    return checks_for(1.6, 1.2, 0.2)[:-1] + [{"name": "SELF_CORRELATION", "result": "FAIL", "limit": 0.7, "value": value}]


async def seed(aid: str = "C0", code: str = "a = rank(close); ts_delta(a, 5)") -> Alpha:
    return await add_alpha(aid, code, 1.6, 1.2, 0.2, check_state="fail", checks=collided(), self_corr=0.85,
                           checked_at=refine.utcnow())


async def submitted() -> None:
    await add_alpha("OLD", SUBMITTED, 1.5, 1.2, 0.2, status="ACTIVE")


def stub_lists(monkeypatch, records: list[dict]) -> None:  # noqa: ANN001
    async def fake(alpha_id, kind="self"):  # noqa: ANN001, ANN202
        return {"max": 0.85, "records": records}

    monkeypatch.setattr(actions, "correlations", fake)


async def test_finds_what_it_collides_with_and_the_sharpe_to_beat(dbenv, monkeypatch) -> None:  # noqa: F811
    a = await seed()
    await submitted()
    stub_lists(monkeypatch, [{"id": "NOPE", "correlation": 0.4}, {"id": "OLD", "correlation": 0.85, "sharpe": 1.5}])
    hit = await find_collision(a)
    assert hit.check == "SELF_CORRELATION" and hit.other_id == "OLD" and hit.code == SUBMITTED
    assert hit.value == pytest.approx(0.85) and hit.sharpe_gap == pytest.approx(1.65 - 1.6)


async def test_falls_back_to_the_local_estimate(dbenv, monkeypatch) -> None:  # noqa: F811
    a = await seed()
    await submitted()
    a.est_corr_with = "OLD"

    async def down(alpha_id, kind="self"):  # noqa: ANN001, ANN202
        raise RuntimeError("BRAIN is busy")

    monkeypatch.setattr(actions, "correlations", down)
    assert (await find_collision(a)).code == SUBMITTED


async def test_prod_collision_names_no_alpha(dbenv) -> None:  # noqa: F811
    checks = checks_for(1.6, 1.2, 0.2)[:-1] + [{"name": "PROD_CORRELATION", "result": "FAIL", "limit": 0.7, "value": 0.9}]
    a = await add_alpha("P0", "rank(close)", 1.6, 1.2, 0.2, check_state="fail", checks=checks,
                        checked_at=refine.utcnow())
    hit = await find_collision(a)
    assert hit.check == "PROD_CORRELATION" and hit.code is None and hit.value == pytest.approx(0.9)


def env_with(hit: Collision) -> refine.FixEnv:
    import random
    return refine.FixEnv("USA", ["SUBINDUSTRY", "INDUSTRY", "SECTOR", "CROWDING"], random.Random(0), collision=hit)


def corr_alpha(code: str = "a = rank(close); ts_delta(a, 5)") -> Alpha:
    return Alpha(id="C", code=code, settings=BASE, sharpe=1.6, fitness=1.2, turnover=0.2, returns=0.1, region="USA",
                 delay=1, checks=collided())


def test_residual_inlines_both_programs() -> None:
    a = corr_alpha()
    made = refine.generate(refine.FIX_BY_KEY["residualize"], a.code, BASE, refine.diagnose(a),
                           env_with(Collision("SELF_CORRELATION", 0.85, "OLD", code=SUBMITTED)))
    expr = made[0][0]
    assert expr == "regression_neut(ts_delta(rank(close), 5), ts_mean(rank(volume), 5))"
    assert "=" not in expr                               # no variable of either program can shadow a field


def test_residual_is_terminal_and_drops_the_parents_dataset() -> None:
    assert "residualize" in refine.TERMINAL


async def test_residual_child_is_not_filed_under_the_parents_dataset(dbenv, monkeypatch) -> None:  # noqa: F811
    fake = FakeSim(lambda e, st: (1.6, 1.2, 0.2))
    monkeypatch.setattr(refine, "simulator", fake)
    a = await seed()
    node = refine.Node(alpha=a, depth=0, root=a.id)
    v = refine.Variant("regression_neut(rank(close), rank(volume))", BASE, "residualize", "SELF_CORRELATION",
                       "residual against OLD", "crowded", other_data=True)
    await refine._submit_round([(node, v)], refine.RefineContext(use_llm=False), 1)
    async with db.session() as s:
        cand = await s.scalar(select(Candidate).where(Candidate.origin == "optimize"))
    assert cand is not None and cand.dataset_id != (a.dataset_ids or ["x"])[0]


def test_outperform_only_when_the_sharpe_is_within_reach() -> None:
    a = corr_alpha()
    d = refine.diagnose(a)
    fix = refine.FIX_BY_KEY["outperform"]
    assert refine.generate(fix, a.code, BASE, d, env_with(Collision("SELF_CORRELATION", sharpe_gap=0.1)))
    assert not refine.generate(fix, a.code, BASE, d, env_with(Collision("SELF_CORRELATION", sharpe_gap=0.9)))
    assert not refine.generate(fix, a.code, BASE, d, env_with(Collision("SELF_CORRELATION", sharpe_gap=-0.2)))


def test_horizon_moves_are_far_enough_to_matter() -> None:
    a = corr_alpha("ts_rank(rank(close), 66)")
    made = refine.generate(refine.FIX_BY_KEY["horizon_far"], a.code, BASE, refine.diagnose(a), env_with(Collision("x")))
    assert any("22" in e for e, _, _ in made) and any("198" in e for e, _, _ in made)     # old: 33 and 132


def test_prod_correlation_tries_crowding_first() -> None:
    a = corr_alpha()
    a.checks = checks_for(1.6, 1.2, 0.2)[:-1] + [{"name": "PROD_CORRELATION", "result": "FAIL", "limit": 0.7,
                                                  "value": 0.9}]
    made = refine.generate(refine.FIX_BY_KEY["neutralize_risk"], a.code, BASE, refine.diagnose(a),
                           env_with(Collision("PROD_CORRELATION")))
    assert made[0][1]["neutralization"] == "CROWDING"


async def test_siblings_offer_several_fields(dbenv) -> None:  # noqa: F811
    from alphago.db.models import DataField
    a = await seed("C5", "rank(sales_a)")
    async with db.write() as s:
        row = await s.get(Alpha, "C5")
        row.dataset_ids, row.fields = ["d1"], ["sales_a"]
        for name, used in (("sales_a", 900), ("sales_b", 5), ("sales_c", 7), ("sales_d", 9), ("other", 1)):
            s.add(DataField(id=name, dataset_id="d1", region="USA", delay=1, universe="TOP3000", type="MATRIX",
                            coverage=0.9, alpha_count=used))
    async with db.session() as s:
        a = await s.get(Alpha, "C5")
    found = await refine._siblings(a, {"region": "USA", "delay": 1, "universe": "TOP3000"})
    assert len(found["sales_a"]) == 3                    # old: one field, one try


class CorrSim(FakeSim):
    """Every child passes the metrics; BRAIN's check then reads each child's correlation from `corr`."""

    def __init__(self, corr) -> None:  # noqa: ANN001
        super().__init__(lambda e, st: (1.6, 1.2, 0.2))
        self.corr = corr


def stub_full_check(monkeypatch, sim: CorrSim) -> list[str]:  # noqa: ANN001
    asked: list[str] = []

    async def check(alpha_id, max_wait_s=900):  # noqa: ANN001, ANN202
        asked.append(alpha_id)
        async with db.session() as s:
            a = await s.get(Alpha, alpha_id)
        value = sim.corr(a.code, a.settings)
        return checks_for(1.6, 1.2, 0.2)[:-1] + [
            {"name": "SELF_CORRELATION", "result": "PASS" if value < 0.7 else "FAIL", "limit": 0.7, "value": value}]

    async def no_estimate(alpha_id, fetch=True):  # noqa: ANN001, ANN202
        return None, None

    monkeypatch.setattr(actions.brain.alphas, "check", check)
    monkeypatch.setattr("alphago.engine.similarity.estimate_self_corr", no_estimate)
    return asked


async def test_decorrelation_walks_down_over_rounds(dbenv, monkeypatch) -> None:  # noqa: F811
    """A round lowers the correlation without clearing it; the next starts from that child (old: one round, stop)."""
    sim = CorrSim(lambda code, st: 0.6 if code.count("regression_neut")
                  else 0.75 if st.get("neutralization") != "SUBINDUSTRY" else 0.85)

    def corr(code, st):  # noqa: ANN001, ANN202
        return sim.corr(code, st)

    monkeypatch.setattr(refine, "simulator", sim)
    stub_full_check(monkeypatch, sim)
    await submitted()
    stub_lists(monkeypatch, [{"id": "OLD", "correlation": 0.85, "sharpe": 1.45}])
    a = await seed()
    passed = await deco.decorrelate(a, refine.RefineContext(use_llm=False, seed=1, max_depth=3), cap=5)
    assert passed
    async with db.session() as s:
        ready = await s.get(Alpha, passed[0])
    assert ready.verdict == "ready"


async def test_decorrelation_stops_when_nothing_moves(dbenv, monkeypatch) -> None:  # noqa: F811
    sim = CorrSim(lambda code, st: 0.85)
    monkeypatch.setattr(refine, "simulator", sim)
    stub_full_check(monkeypatch, sim)
    stub_lists(monkeypatch, [])
    a = await seed()
    passed = await deco.decorrelate(a, refine.RefineContext(use_llm=False, seed=1, max_depth=3), cap=5)
    assert passed == [] and len(sim.sent) <= 8           # one round of tries, not three


async def test_certify_sends_each_failure_down_its_own_route(dbenv, monkeypatch) -> None:  # noqa: F811
    await add_alpha("M1", "rank(-ts_delta(close, 5))", 1.6, 1.2, 0.2)
    await add_alpha("M2", "rank(-ts_delta(close, 6))", 1.6, 1.2, 0.2)
    calls: list[str] = []

    async def fake_run_check(alpha_id):  # noqa: ANN001, ANN202
        checks = collided() if alpha_id == "M1" else checks_for(1.6, 1.2, 0.2)[:-1] + [
            {"name": "LOW_2Y_SHARPE", "result": "FAIL", "limit": 1.0, "value": 0.4}]
        async with db.write() as s:
            a = await s.get(Alpha, alpha_id)
            a.checks, a.check_state, a.checked_at = checks, "fail", refine.utcnow()
        return "fail", checks

    async def fake_decorrelate(a, ctx, cap=3):  # noqa: ANN001, ANN202
        calls.append(f"decorrelate {a.id}")
        return []

    async def fake_refine(seeds, ctx):  # noqa: ANN001, ANN202
        calls.append(f"refine {seeds[0].id}")
        return refine.RefineReport()

    async def no_estimate(alpha_id, fetch=True):  # noqa: ANN001, ANN202
        return None, None

    monkeypatch.setattr(actions, "run_check", fake_run_check)
    monkeypatch.setattr(deco, "decorrelate", fake_decorrelate)
    monkeypatch.setattr(refine, "refine", fake_refine)
    monkeypatch.setattr("alphago.engine.similarity.estimate_self_corr", no_estimate)
    report = await refine.certify(["M1", "M2"], refine.RefineContext(use_llm=False), cap=5)
    assert report.corr_failed == ["M1"] and report.metric_failed == ["M2"]
    assert calls == ["decorrelate M1", "refine M2"]      # old: a metric failure from /check was never repaired
