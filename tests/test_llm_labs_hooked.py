"""Both labs end to end through the runner with its `Lab.busy` / `Lab.send` hooks: the Power Pool call runs in the
background (the task neither blocks the runner nor ends while the model writes) and Super Alpha tasks send their own
SUPER simulations, three at a time, until every pairing is scored.

Until the hooks are in alphago/labs/runner.py this file applies them to a copy of the runner class (the exact change
requested from the coordinator); once they are there it tests the real runner unchanged."""

from __future__ import annotations

import asyncio
import inspect
import textwrap

import respx

from alphago.brain import brain
from alphago.labs import power_pool_lab as pp
from alphago.labs import runner as runner_mod
from alphago.labs import super_lab as sl
from alphago.labs.runner import Lab, LabRunner
from tests.fakebrain import FakeBrain
from tests.test_llm_labs import FakeProvider, brain_env, catalog_db, good_batch, params, providers  # noqa: F401

FINISH = """        if not parked:
            if not running:
                await self._finish(task_id, "COMPLETE", lab.nothing_left(t))
            return
        await self._send(t, parked[:want], ctx)"""
HOOKED = """        if not parked:
            if not running and not await lab.busy(t):
                await self._finish(task_id, "COMPLETE", lab.nothing_left(t))
            return
        if not await lab.send(t, parked[:want], ctx):
            await self._send(t, parked[:want], ctx)"""


def hooked_runner(monkeypatch) -> LabRunner:  # noqa: ANN001
    if "busy" in Lab.__dict__ and "send" in Lab.__dict__:
        r = LabRunner()
    else:
        async def busy(self, task):  # noqa: ANN001, ANN202
            return False

        async def send(self, task, trials, ctx):  # noqa: ANN001, ANN202
            return False

        monkeypatch.setattr(Lab, "busy", busy, raising=False)
        monkeypatch.setattr(Lab, "send", send, raising=False)
        source = textwrap.dedent(inspect.getsource(LabRunner._tick))
        block, hooked = (textwrap.indent(textwrap.dedent(x), "    ") for x in (FINISH, HOOKED))
        source = source.replace(block, hooked)
        assert "lab.busy(t)" in source, "the runner's _tick changed: update this test's patch"
        namespace: dict = {}
        exec(compile(source, runner_mod.__file__, "exec"), vars(runner_mod), namespace)  # noqa: S102
        monkeypatch.setattr(LabRunner, "_tick", namespace["_tick"])
        r = LabRunner()
    r.register(pp.PowerPoolLab())
    r.register(sl.SuperLab())
    monkeypatch.setattr(runner_mod, "runner", r)
    pp._calls.clear()
    pp._retry.clear()
    return r


async def _drive(r: LabRunner, tid: int, rounds: int = 300) -> None:
    for _ in range(rounds):
        await r.tick(tid)
        t = await r.get(tid)
        if t.status in ("COMPLETE", "FAILED", "PAUSED"):
            return
        await asyncio.sleep(0.01)


async def test_power_pool_writes_in_the_background(catalog_db, providers, brain_env, monkeypatch):  # noqa: F811
    r = hooked_runner(monkeypatch)
    assert pp.runner_hooks()
    providers(FakeProvider("fake", [good_batch(0), good_batch(10)]))
    fake = FakeBrain(lambda e, st: (1.3, 1.1, 0.2))
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await brain_env.start()
        tid = await r.create("power_pool", params(simulations=30), cores=2, max_trials=30)
        await r.run(tid)
        await _drive(r, tid)
    t = await r.get(tid)
    assert t.status == "COMPLETE", t.message
    assert (await r.summary(tid))["simulated"] == 30 and t.state["llm"]["calls"] == 2


async def test_super_task_runs_every_pairing(catalog_db, brain_env, monkeypatch):  # noqa: F811
    r = hooked_runner(monkeypatch)
    monkeypatch.setattr(sl, "_recovered", True)
    assert sl.runner_hooks()
    p = {"region": "USA", "delay": 1, "universe": "TOP3000", "neutralization": "SUBINDUSTRY", "selection_limit": 20,
         "activation": "BOTH", "selections": [{"name": "Least correlated", "code": "(1 - prod_correlation)"}],
         "combos": [{"name": "Equal weight", "code": "1"}, {"name": "Inverse volatility", "code": "2"}]}
    fake = FakeBrain(lambda e, st: (1.5, 1.2, 0.1))
    peak = {"now": 0, "max": 0}
    real_run = brain.sims.run

    async def counted(payloads, **kw):  # noqa: ANN001, ANN003, ANN202
        peak["now"] += 1
        peak["max"] = max(peak["max"], peak["now"])
        try:
            await asyncio.sleep(0.01)
            return await real_run(payloads, **kw)
        finally:
            peak["now"] -= 1

    monkeypatch.setattr(brain.sims, "run", counted)
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        tid = await r.create("super", p, cores=3, run=True)
        for _ in range(400):
            await r.tick(tid)
            if (await r.get(tid)).status in ("COMPLETE", "FAILED"):
                break
            await asyncio.sleep(0.01)
    t = await r.get(tid)
    assert t.status == "COMPLETE", t.message
    summary = await r.summary(tid)
    assert summary["simulated"] == 8 == fake.posted and summary["failed"] == 0
    assert peak["max"] <= 3 and len(await r.top(tid)) == 8
