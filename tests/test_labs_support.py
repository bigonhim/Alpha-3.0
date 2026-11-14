"""Shared fixtures for the Search / Template / Evolution lab tests: a fake account catalogue in a fresh database,
the fake BRAIN, and a runner with the three labs registered (no tests of its own)."""

from __future__ import annotations

import asyncio
import zlib

import pytest

from alphago.brain import brain
from alphago.brain import client as client_mod
from alphago.brain import simulations as sims_mod
from alphago.db import db
from alphago.db.models import DataField, Dataset, Operator

OPERATORS = [
    ("rank", "Cross Sectional", "rank(x, rate=2)"),
    ("zscore", "Cross Sectional", "zscore(x)"),
    ("ts_rank", "Time Series", "ts_rank(x, d, constant = 0)"),
    ("ts_mean", "Time Series", "ts_mean(x, d)"),
    ("ts_zscore", "Time Series", "ts_zscore(x, d)"),
    ("ts_delta", "Time Series", "ts_delta(x, d)"),
    ("ts_av_diff", "Time Series", "ts_av_diff(x, d)"),
    ("ts_arg_min", "Time Series", "ts_arg_min(x, d)"),
    ("ts_arg_max", "Time Series", "ts_arg_max(x, d)"),
    ("ts_backfill", "Time Series", 'ts_backfill(x,lookback = d, k=1, ignore="NAN")'),
    ("group_rank", "Group", "group_rank(x, group)"),
    ("group_zscore", "Group", "group_zscore(x, group)"),
    ("group_neutralize", "Group", "group_neutralize(x, group)"),
    ("vec_avg", "Vector", "vec_avg(x)"),
    ("vec_max", "Vector", "vec_max(x)"),
    ("abs", "Arithmetic", "abs(x)"),
    ("add", "Arithmetic", "add(x, y, filter = false), x + y"),
    ("subtract", "Arithmetic", "subtract(x, y, filter=false), x - y"),
    ("multiply", "Arithmetic", "multiply(x ,y, ... , filter=false), x * y"),
    ("divide", "Arithmetic", "divide(x, y), x / y"),
    ("reverse", "Arithmetic", "reverse(x), -x"),
    ("greater", "Logical", "input1 > input2"),
    ("less", "Logical", "input1 < input2"),
    ("if_else", "Logical", "if_else(input1, input2, input3)"),
]
# (field, dataset, type, universes, category)
FIELDS = [
    ("fnd1_a", "fnd1", "MATRIX", ("TOP3000", "TOP1000"), "fundamental"),
    ("fnd1_b", "fnd1", "MATRIX", ("TOP3000", "TOP1000"), "fundamental"),
    ("fnd1_c", "fnd1", "MATRIX", ("TOP3000",), "fundamental"),
    ("anl1_v", "anl1", "VECTOR", ("TOP3000",), "analyst"),
    ("pv1_x", "pv1", "MATRIX", ("TOP3000", "TOP1000"), "pv"),
]
SETTINGS = {"region": "USA", "delay": 1, "universe": "TOP3000", "decay": 0, "neutralization": "INDUSTRY"}


async def seed_catalogue() -> None:
    from alphago.engine import catalogs
    catalogs.invalidate()
    async with db.write() as s:
        for name, cat, definition in OPERATORS:
            s.add(Operator(name=name, category=cat, definition=definition, scope=["REGULAR"], level="ALL"))
        datasets = {f[1]: f[4] for f in FIELDS}
        for universe in ("TOP3000", "TOP1000"):
            for ds, cat in datasets.items():
                s.add(Dataset(id=ds, region="USA", delay=1, universe=universe, name=ds.upper(), category_id=cat,
                              category_name=cat.title()))
        for fid, ds, ftype, universes, cat in FIELDS:
            for i, universe in enumerate(universes):
                s.add(DataField(id=fid, dataset_id=ds, region="USA", delay=1, universe=universe, type=ftype,
                                coverage=0.9 - 0.1 * i, category_id=cat))


def score(expression: str, _settings: dict) -> tuple[float, float, float]:
    """Deterministic: some expressions pass, most do not."""
    h = zlib.crc32(expression.encode()) % 1000
    return 0.5 + h / 400, 0.4 + h / 600, 0.05 + (h % 7) / 20


@pytest.fixture
async def labenv(tmp_path, monkeypatch):
    real_sleep = asyncio.sleep

    async def fast(_s):  # noqa: ANN001, ANN202
        await real_sleep(0)

    monkeypatch.setattr(client_mod.asyncio, "sleep", fast)
    monkeypatch.setattr(sims_mod.asyncio, "sleep", fast)
    await db.init(tmp_path / "labs.db")
    await seed_catalogue()
    brain.client._credentials = lambda: ("me@example.com", "pw")
    from alphago.engine.simulator import SimulationService
    from alphago.labs import runner as runner_mod
    from alphago.labs.ga import EvolutionLab
    from alphago.labs.runner import LabRunner
    from alphago.labs.search import SearchLab
    from alphago.labs.template import TemplateLab
    sim = SimulationService()
    monkeypatch.setattr("alphago.engine.simulator.simulator", sim)
    r = LabRunner()
    for lab in (SearchLab(), TemplateLab(), EvolutionLab()):
        r.register(lab)
    monkeypatch.setattr(runner_mod, "runner", r)
    yield r, sim
    await sim.stop()
    brain.client.observers.clear()
    await brain.client.close()
    await db.close()


async def drive(r, tid: int, rounds: int = 200) -> None:  # noqa: ANN001
    for _ in range(rounds):
        await r.tick(tid)
        t = await r.get(tid)
        if t.status in ("COMPLETE", "FAILED"):
            return
        await asyncio.sleep(0.02)
