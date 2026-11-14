"""The lab task runner against the fake BRAIN: cores, trials, scoring, free answers, refusals and lifecycle."""

from __future__ import annotations

import asyncio

import pytest
import respx
from sqlalchemy import select

from alphago.brain import brain
from alphago.brain import client as client_mod
from alphago.brain import simulations as sims_mod
from alphago.db import db
from alphago.db.models_labs import LabTrial
from alphago.labs import runner as runner_mod
from alphago.labs.runner import Lab, LabRunner, TrialSpec
from tests.fakebrain import FakeBrain

SETTINGS = {"region": "USA", "delay": 1, "universe": "TOP3000", "decay": 0, "neutralization": "INDUSTRY"}


class Counting(Lab):
    """A lab that proposes rank(close / N) for N = 1.. until the budget is spent."""

    name = "counting"
    title = "Counting"

    async def plan(self, task):  # noqa: ANN001, ANN201
        if not task.params.get("ok", True):
            raise ValueError("refused by plan")
        return {"start": 1}

    async def refill(self, task, want, ctx):  # noqa: ANN001, ANN201
        n0 = len(ctx.seen) + 1
        return [TrialSpec(f"rank(close / {n})", SETTINGS, params={"n": n}) for n in range(n0, n0 + want)]


@pytest.fixture
async def env(tmp_path, monkeypatch):
    real_sleep = asyncio.sleep

    async def fast(_s):  # noqa: ANN001, ANN202
        await real_sleep(0)

    monkeypatch.setattr(client_mod.asyncio, "sleep", fast)
    monkeypatch.setattr(sims_mod.asyncio, "sleep", fast)
    await db.init(tmp_path / "labs.db")
    brain.client._credentials = lambda: ("me@example.com", "pw")
    from alphago.engine.simulator import SimulationService
    sim = SimulationService()
    monkeypatch.setattr("alphago.engine.simulator.simulator", sim)
    r = LabRunner()
    r.register(Counting())
    monkeypatch.setattr(runner_mod, "runner", r)
    yield r, sim
    await sim.stop()
    brain.client.observers.clear()
    await brain.client.close()
    await db.close()


async def _drive(r: LabRunner, tid: int, rounds: int = 80) -> None:
    for _ in range(rounds):
        await r.tick(tid)
        t = await r.get(tid)
        if t.status in ("COMPLETE", "FAILED"):
            return
        await asyncio.sleep(0.02)


async def test_task_spends_its_budget_and_ranks_results(env):
    r, sim = env
    fake = FakeBrain(lambda e, st: (1.0 + len(e) / 100, 1.1, 0.2))
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await sim.start()
        tid = await r.create("counting", {"region": "USA"}, cores=1, max_trials=12)
        await r.run(tid)
        await _drive(r, tid)
    t = await r.get(tid)
    assert t.status == "COMPLETE", t.message
    summary = await r.summary(tid)
    assert summary["simulated"] == 12 and fake.posted == 12
    top = await r.top(tid)
    assert top and top[0]["value"] >= top[-1]["value"] and t.best_alpha_id == top[0]["alpha_id"]


async def test_identical_request_is_answered_free(env):
    r, sim = env
    fake = FakeBrain(lambda e, st: (1.5, 1.1, 0.2))
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await sim.start()
        first = await r.create("counting", {}, cores=1, max_trials=10)
        await r.run(first)
        await _drive(r, first)
        posted = fake.posted
        second = await r.create("counting", {}, cores=1, max_trials=10)
        await r.run(second)
        await _drive(r, second)
    summary = await r.summary(second)
    # the 10 repeats were answered free and do not count against the budget, which went on 10 new ideas
    assert summary["cached"] == 10 and summary["simulated"] == 10
    assert fake.posted == posted + 10                  # nothing paid for twice


async def test_plan_refusal_creates_nothing(env):
    r, _ = env
    with pytest.raises(ValueError):
        await r.create("counting", {"ok": False})
    assert await r.list() == []


async def test_refused_setting_closes_repeats_unsent(env):
    r, _ = env
    tid = await r.create("counting", {}, cores=1, max_trials=5)
    await r.add_trials(tid, [TrialSpec("rank(open)", {**SETTINGS, "truncation": 0.5})])
    async with db.write() as s:
        row = (await s.scalars(select(LabTrial).where(LabTrial.task_id == tid))).one()
        row.state, row.message = "FAIL", "BRAIN rejected: settings.truncation: must be at most 0.3"
    await r.add_trials(tid, [TrialSpec("rank(high)", {**SETTINGS, "truncation": 0.5})])
    t = await r.get(tid)
    ctx = await r._context(t)
    parked = await r.trials(tid, ("PARKED",))
    await r._send(t, parked, ctx)
    after = await r.trials(tid, ("FAIL",))
    assert any("Not sent: BRAIN refused truncation" in (x.message or "") for x in after)


async def test_cores_queue_until_they_fit(env):
    r, _ = env
    a = await r.create("counting", {}, cores=6)
    b = await r.create("counting", {}, cores=4)
    c = await r.create("counting", {}, cores=2)
    for tid in (a, b, c):
        await r.run(tid)
    statuses = {tid: (await r.get(tid)).status for tid in (a, b, c)}
    assert statuses == {a: "RUNNING", b: "QUEUED", c: "RUNNING"}   # the big one waits; the small one fits
    await r.pause(a)
    assert (await r.get(b)).status == "RUNNING"
    with pytest.raises(ValueError):
        await r.delete(b)
    await r.rename(c, "  my task  ")
    assert (await r.get(c)).name == "my task"


class QuickLab(Counting):
    """Quick-mode simulations of rank(close / N)."""

    name = "quick"
    title = "Quick"

    async def refill(self, task, want, ctx):  # noqa: ANN001, ANN201
        n0 = len([k for k in ctx.seen]) + 1
        return [TrialSpec(f"rank(close / {n})", {**SETTINGS, "simulationMode": "QUICK"}, params={"n": n})
                for n in range(n0, n0 + want)]


async def test_passing_quick_alpha_is_owed_a_full_run(env):
    r, sim = env
    r.register(QuickLab())
    fake = FakeBrain(lambda e, st: (2.0, 1.5, 0.2))           # every alpha passes
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await sim.start()
        tid = await r.create("quick", {}, cores=1, max_trials=3, run=True)
        await _drive(r, tid)
    trials = await r.trials(tid)
    owed = [t for t in trials if (t.params or {}).get("fullOf") is not None]
    assert len(owed) == 3 and all(t.settings["simulationMode"] == "FULL" for t in owed)
    assert all(t.state != "PARKED" for t in owed)                 # sent, even after the budget was spent
    assert (await r.summary(tid))["simulated"] >= 3
    ctx = await r._context(await r.get(tid))
    assert ctx.spent == 3                                          # Full runs never count against the task
    top = await r.top(tid)
    assert {t["trial"] for t in top}.isdisjoint({t.params["fullOf"] for t in owed if t.state == "COMPLETE"})


async def test_withdrawn_setting_is_pruned_unsent(env, monkeypatch):
    r, _ = env
    monkeypatch.setattr(brain.options, "live", True)
    monkeypatch.setattr(brain.options, "problems", lambda st: ["universe 'NOPE' is not available."]
                        if st.get("universe") == "NOPE" else [])
    tid = await r.create("counting", {}, cores=1, max_trials=5)
    await r.add_trials(tid, [TrialSpec("rank(open)", {**SETTINGS, "universe": "NOPE"})])
    t = await r.get(tid)
    await r._send(t, await r.trials(tid, ("PARKED",)), await r._context(t))
    pruned = await r.trials(tid, ("PRUNED",))
    assert pruned and pruned[0].message.startswith("Withdrawn: universe 'NOPE'")
