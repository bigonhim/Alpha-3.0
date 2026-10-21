"""Against a mock BRAIN: contract tests on recorded response shapes, and the full research path to "ready".

Contract tests replay every file in tests/fixtures/brain (the format `alphago.brain.recorder` writes from the real
API when ALPHAGO_RECORD_BRAIN is set) through the app's parsers. Re-record after BRAIN changes its API and run pytest
to see what broke."""

from __future__ import annotations

import asyncio
import functools
import json
from pathlib import Path

import pytest
import respx
from sqlalchemy import select

from alphago.brain import brain
from alphago.brain import client as client_mod
from alphago.brain import simulations as sims_mod
from alphago.brain.alphas import _records
from alphago.db import db
from alphago.db.models import Alpha, Candidate, Dataset, DataField
from alphago.engine import pnl
from alphago.engine.evaluator import check_state_of
from alphago.engine.ingest import apply_alpha_json

from .fakebrain import FakeBrain

FIXTURES = Path(__file__).parent / "fixtures" / "brain"
REPAIR_SEED = 1


def fixtures() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(FIXTURES.glob("*.json"))]


@pytest.mark.parametrize("fx", fixtures(), ids=lambda f: f["shape"])
def test_recorded_shapes_still_parse(fx: dict) -> None:
    body = fx["json"]
    shape = fx["shape"]
    if shape == "/alphas/{id}":
        row = apply_alpha_json(Alpha(id=body["id"]), body)
        assert row.code and row.sharpe is not None and row.is_checks and row.verdict
    elif shape == "/alphas/{id}/check":
        assert check_state_of(body["is"]["checks"]) in ("pass", "fail", "pending")
    elif shape.endswith("/recordsets/pnl"):
        dates, daily = pnl.parse_records(_records(body))
        assert dates and len(daily) == len(dates)
    elif shape == "/users/self/alphas":
        assert all(apply_alpha_json(Alpha(id=a["id"]), a).verdict for a in body["results"])
    else:
        pytest.skip(f"no contract for {shape} yet")


def test_check_fixture_with_power_pool_fail_reads_as_pass() -> None:
    fx = next(f for f in fixtures() if f["shape"] == "/alphas/{id}/check")
    assert check_state_of(fx["json"]["is"]["checks"]) == "pass"


@pytest.fixture
async def fake_env(tmp_path, monkeypatch):
    real_sleep = asyncio.sleep

    async def fast(_s):  # noqa: ANN001, ANN202
        await real_sleep(0)

    monkeypatch.setattr(client_mod.asyncio, "sleep", fast)
    monkeypatch.setattr(sims_mod.asyncio, "sleep", fast)
    await db.init(tmp_path / "fake.db")
    brain.client._credentials = lambda: ("me@example.com", "pw")
    yield
    brain.client.observers.clear()
    await brain.client.close()
    await db.close()


def banded(expr: str, st: dict) -> tuple[float, float, float]:
    """Only one template on operating_income is a near miss; more decay repairs it (decay ≥ 6 passes)."""
    if not (expr.startswith("ts_rank(") and "operating_income" in expr and ", 66)" in expr):
        return 0.3, 0.1, 0.3
    d = int(st.get("decay") or 0)
    return (1.5, 1.1, 0.25) if d >= 6 else (1.5, 0.85, 0.45) if d >= 1 else (1.5, 0.6, 0.6)


async def test_research_task_reaches_ready_against_mock_brain(fake_env, monkeypatch) -> None:
    from alphago.engine import audit, catalogs, health, refine
    from alphago.engine import research as research_mod
    from alphago.engine.research import research
    from alphago.engine.simulator import SimulationService
    from alphago.llm import gemini

    catalogs.invalidate()
    monkeypatch.setattr(gemini, "configure", lambda *a, **k: False)
    async with db.write() as s:
        s.add(Dataset(id="fund1", region="USA", delay=1, universe="TOP3000", name="Fundamentals",
                      category_name="Fundamental", field_count=2))
        for f in ("operating_income", "sales"):
            s.add(DataField(id=f, dataset_id="fund1", region="USA", delay=1, universe="TOP3000", type="MATRIX",
                            coverage=0.9, alpha_count=10))
    sim = SimulationService()
    monkeypatch.setattr(research_mod, "simulator", sim)
    monkeypatch.setattr(refine, "simulator", sim)
    # the repair loop's Thompson draws are unseeded in the app; with this tight budget an unlucky draw spends the last
    # repair slot on another fix, so pin them (as tests/test_refine.py does) to keep the run reproducible
    monkeypatch.setattr(research_mod, "RefineContext", functools.partial(refine.RefineContext, seed=REPAIR_SEED))
    audit.install()
    health.install()
    with respx.mock(assert_all_called=False) as router:
        fake = FakeBrain(banded, throttle_posts=1).install(router)
        await brain.client.login()
        await sim.start()
        try:
            tid = await research.create(kind="single", region="USA", delay=1, universe="TOP3000",
                                        dataset_ids=["fund1"], planned_sims=60, cores=2)
            await asyncio.wait_for(research._run_inner(tid), timeout=120)
        finally:
            await sim.stop()
    t = await research.get(tid)
    async with db.session() as s:
        ready = list(await s.scalars(select(Alpha).where(Alpha.verdict == "ready")))
        cands = list(await s.scalars(select(Candidate).where(Candidate.task_id == tid)))
    assert t.status == "complete" and t.n_ready >= 1 and ready
    assert all(int(a.decay) >= 6 for a in ready)              # found by the repair loop, not the first pass
    assert all(a.check_state == "pass" for a in ready)
    assert any(a.robust_state in ("robust", "fragile", "unknown") for a in ready)   # room was kept for the test
    assert fake.posted <= 60 and fake.checked                    # within the planned budget; BRAIN checked it
    assert not fake.submitted                                    # never submitted without a click
    await audit.flush()
    entries = await audit.entries()
    sims_logged = sum(e.quota_cost for e in entries if e.action == "simulate")
    assert sims_logged == fake.posted                            # the audit log accounts for every simulation
    assert {e.actor for e in entries if e.action == "simulate"} == {f"task:{tid}"}
    assert any(e.action == "check" for e in entries)
    await health.flush()
    stats = next(iter((await db.kv_get(health.BRAIN_STATS_KEY)).values()))
    assert stats["throttled"] >= 1 and stats["simulations_started"] == fake.posted
    assert len(cands) >= fake.posted
