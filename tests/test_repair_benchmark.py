"""The repair engine against a real backtest: near misses on tests/marketsim.py, one shared budget, no BRAIN involved.

A floor, not a forecast. The market is synthetic, so this shows the logic works on levers that behave by mechanics; it
says nothing about BRAIN's pass rate."""

from __future__ import annotations

import random
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import select

from alphago.db import db
from alphago.db.models import Alpha, Candidate, DataField, Dataset
from alphago.engine import catalogs, refine
from alphago.engine.ingest import upsert_alpha
from alphago.engine.simulator import CandidateSpec, SimulationService
from alphago.knowledge.templates import enumerate_single

from .marketsim import Market, alpha_json

BASE = {"region": "USA", "delay": 1, "universe": "TOP3000", "neutralization": "SUBINDUSTRY", "decay": 2,
        "truncation": 0.08}


class MarketSim(SimulationService):
    def __init__(self, market: Market) -> None:
        super().__init__()
        self.market, self.n = market, 0

    async def enqueue(self, candidate_ids: list[int]) -> int:
        async with db.session() as s:
            cands = list(await s.scalars(select(Candidate).where(Candidate.id.in_(candidate_ids),
                                                                 Candidate.status.in_(["new", "approved"]))))
        for c in cands:
            self.n += 1
            data = alpha_json(self.market, c.expression, c.settings)
            await upsert_alpha(data, candidate_id=c.id)
            await self._set_status([c.id], status="done", alpha_id=data["id"], sim_id=f"S{self.n}",
                                   finished_at=datetime.now(timezone.utc))
        return len(cands)

    async def wait_for(self, candidate_ids: list[int], timeout: float | None = None) -> None:
        return None


def test_turnover_follows_the_measured_decay_curve() -> None:
    m = Market(n=200, days=500, warm=100)
    t = {n: m.simulate("rank(mid1)", {**BASE, "decay": n}).turnover for n in (0, 3, 15)}
    slope = np.log(t[0] / t[15]) / np.log(16)
    assert 0.3 < slope < 0.7                  # playbook.DECAY_BETA is 0.47


async def test_best_first_turns_near_misses_strong_on_a_real_backtest(tmp_path, monkeypatch) -> None:
    await db.init(tmp_path / "bench.db")
    catalogs.invalidate()
    market = Market()
    sim = MarketSim(market)
    monkeypatch.setattr(refine, "simulator", sim)

    async def no_pnl(alpha_id, *, force=False, keep_raw=True):  # noqa: ANN001, ANN202
        return None

    monkeypatch.setattr("alphago.engine.pnl.fetch", no_pnl)
    async with db.write() as s:
        s.add(Dataset(id="mkt", region="USA", delay=1, universe="TOP3000", name="Market", field_count=30))
        for f in market.data_fields():
            s.add(DataField(id=f, dataset_id="mkt", region="USA", delay=1, universe="TOP3000", type="MATRIX",
                            coverage=0.9, alpha_count=10))
    exprs = [e for e, _ in enumerate_single([(f, "MATRIX") for f in market.data_fields()], windows=(66,))]
    random.Random(1).shuffle(exprs)
    made = await sim.prepare([CandidateSpec(e, BASE, origin="search") for e in exprs[:40]], initial_status="approved")
    await sim.enqueue([c.id for c in made.created])
    async with db.session() as s:
        alphas = list(await s.scalars(select(Alpha)))
    seeds = refine.closest_first(alphas)
    assert seeds
    start = sim.n

    async def forty() -> int:
        return 40 - (sim.n - start)

    report = await refine.best_first(seeds, refine.RefineContext(use_llm=False, seed=1, allowance=forty))
    await db.close()
    assert sim.n - start <= 40 and report.strong


@pytest.mark.parametrize("code", ["hump(rank(mid1), hump=0.01)", "trade_when(rank(volume) > 0.5, rank(mid1), -1)",
                                  "if_else(rank(adv20) > 0.3, rank(mid1), nan)",
                                  "signed_power(rank(mid1) - 0.5, 2)",
                                  "regression_neut(rank(mid1), rank(crowd1))"])
def test_every_move_the_engine_writes_can_be_backtested(code: str) -> None:
    m = Market(n=120, days=300, warm=60)
    assert m.simulate(code, BASE).turnover >= 0
