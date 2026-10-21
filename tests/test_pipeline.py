"""End-to-end pipeline against a mocked BRAIN: prepare → pack → multi-simulate → ingest → verdict → reuse."""

from __future__ import annotations

import asyncio
import itertools
import json

import httpx
import pytest
import respx

from alphago.brain import brain
from alphago.brain import client as client_mod
from alphago.brain import simulations as sims_mod
from alphago.db import db
from alphago.db.models import Alpha, Candidate
from alphago.engine.evaluator import evaluate
from alphago.engine.optimizer import plan_variants
from alphago.engine.simulator import CandidateSpec, SimulationService
from sqlalchemy import select

API = "https://api.worldquantbrain.com"
AUTH = {"user": {"id": "U1"}, "token": {"expiry": 14400.0}, "permissions": []}


def alpha_json(alpha_id: str, code: str, settings: dict, sharpe: float) -> dict:
    checks = [
        {"name": "LOW_SHARPE", "result": "PASS" if sharpe >= 1.58 else "FAIL", "limit": 1.58, "value": sharpe},
        {"name": "LOW_FITNESS", "result": "PASS" if sharpe >= 1.58 else "FAIL", "limit": 1.0, "value": sharpe * 0.8},
        {"name": "LOW_TURNOVER", "result": "PASS", "limit": 0.01, "value": 0.12},
        {"name": "HIGH_TURNOVER", "result": "PASS", "limit": 0.7, "value": 0.12},
        {"name": "SELF_CORRELATION", "result": "PENDING"},
    ]
    return {"id": alpha_id, "type": "REGULAR", "settings": settings, "regular": {"code": code},
            "status": "UNSUBMITTED", "dateCreated": "2026-10-13T10:00:00-04:00",
            "is": {"sharpe": sharpe, "fitness": sharpe * 0.8, "turnover": 0.12, "returns": 0.1, "drawdown": 0.05,
                   "margin": 0.0008, "longCount": 1500, "shortCount": 1480, "checks": checks}}


@pytest.fixture
async def env(tmp_path, monkeypatch):
    real_sleep = asyncio.sleep

    async def fast(_s):
        await real_sleep(0)

    monkeypatch.setattr(client_mod.asyncio, "sleep", fast)
    monkeypatch.setattr(sims_mod.asyncio, "sleep", fast)
    await db.init(tmp_path / "t.db")
    brain.client._credentials = lambda: ("me@x.com", "pw")
    yield
    await brain.client.close()
    await db.close()


@respx.mock
async def test_pipeline_simulate_ingest_reuse(env):
    counter = itertools.count(1)
    sims: dict[str, list[dict]] = {}
    children: dict[str, dict] = {}

    def post_sim(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        items = body if isinstance(body, list) else [body]
        pid = f"P{next(counter)}"
        sims[pid] = items
        return httpx.Response(201, headers={"Location": f"{API}/simulations/{pid}",
                                            "X-Ratelimit-Limit": "5000", "X-Ratelimit-Remaining": "4000"})

    def get_sim(request: httpx.Request) -> httpx.Response:
        sid = request.url.path.rsplit("/", 1)[-1]
        if sid in sims:
            items = sims[sid]
            if len(items) == 1:
                aid = f"A_{sid}"
                children[aid] = items[0]
                return httpx.Response(200, json={"id": sid, "status": "COMPLETE", "alpha": aid})
            kids = []
            for i, item in enumerate(items):
                cid = f"{sid}C{i}"
                children[f"A_{cid}"] = item
                kids.append(cid)
            return httpx.Response(200, json={"id": sid, "status": "COMPLETE", "children": kids})
        return httpx.Response(200, json={"id": sid, "status": "COMPLETE", "alpha": f"A_{sid}"})

    def get_alpha(request: httpx.Request) -> httpx.Response:
        aid = request.url.path.rsplit("/", 1)[-1]
        item = children[aid]
        sharpe = 2.1 if "operating_income" in item["regular"] else 0.4
        return httpx.Response(200, json=alpha_json(aid, item["regular"], item["settings"], sharpe))

    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH))
    respx.post(f"{API}/simulations").mock(side_effect=post_sim)
    respx.get(url__regex=rf"{API}/simulations/\w+").mock(side_effect=get_sim)
    respx.get(url__regex=rf"{API}/alphas/A_\w+").mock(side_effect=get_alpha)

    await brain.client.login()
    sim = SimulationService()
    await sim.start()
    base = {"region": "USA", "delay": 1, "universe": "TOP3000", "neutralization": "SUBINDUSTRY"}
    specs = [
        CandidateSpec("group_rank(ts_rank(operating_income/cap, 126), subindustry)", base, origin="llm"),
        CandidateSpec("rank(-ts_delta(close, 5))", base, origin="template"),
        CandidateSpec("rank(-ts_delta(close, 5))", base, origin="template"),          # duplicate → skipped
        CandidateSpec("rank(close ^ 2)", base, origin="llm"),                           # invalid → no quota spent
        CandidateSpec("ts_mean(close, 10)", {**base, "decay": 4}, origin="template"),
    ]
    result = await sim.submit(specs)
    assert len(result.created) == 3 and len(result.invalid) == 1 and result.skipped == 1
    await sim.wait_for([c.id for c in result.created], timeout=30)

    async with db.session() as s:
        cands = {c.expression: c for c in await s.scalars(select(Candidate))}
        alphas = {a.id: a for a in await s.scalars(select(Alpha))}
    assert all(cands[e].status == "done" for e in (specs[0].expression, specs[1].expression, specs[4].expression))
    assert cands["rank(close ^ 2)"].status == "invalid"
    good = alphas[cands[specs[0].expression].alpha_id]
    assert good.verdict == "strong" and good.sharpe == pytest.approx(2.1)
    assert good.canonical_hash and good.op_count == 3   # group_rank, ts_rank, divide
    weak = alphas[cands[specs[1].expression].alpha_id]
    assert weak.verdict == "rejected"
    assert len(sims) == 1, "three compatible candidates should share one multi-simulation"

    # re-submitting the same idea + settings must reuse the stored alpha
    again = await sim.submit([CandidateSpec("group_rank(ts_rank(operating_income / cap, 126), subindustry)", base)])
    assert len(again.reused) == 1 and again.reused[0].alpha_id == good.id
    assert len(sims) == 1

    # optimiser proposes valid variants for a near miss
    weak.checks = [{"name": "LOW_SHARPE", "result": "FAIL", "limit": 1.58, "value": 1.3},
                   {"name": "HIGH_TURNOVER", "result": "FAIL", "limit": 0.7, "value": 0.8}]
    weak.sharpe, weak.fitness, weak.turnover = 1.3, 0.9, 0.8
    variants = plan_variants(weak)
    kinds = {k for _, _, k in variants}
    assert any(k.startswith("decay") for k in kinds) and any(k.startswith("neutralization") for k in kinds)
    await sim.stop()


def test_evaluate_verdicts():
    checks = [{"name": "LOW_SHARPE", "result": "FAIL", "limit": 1.58, "value": 1.2},
              {"name": "LOW_FITNESS", "result": "FAIL", "limit": 1.0, "value": 0.7}]
    assert evaluate(sharpe=1.2, fitness=0.7, turnover=0.2, checks=checks).verdict == "promising"
    assert evaluate(sharpe=0.85, fitness=0.3, turnover=0.2, checks=checks).verdict == "weak"
    assert evaluate(sharpe=0.2, fitness=0.1, turnover=0.2, checks=checks).verdict == "rejected"
    neg = evaluate(sharpe=-2.0, fitness=-1.5, turnover=0.2, checks=checks)
    assert neg.flip_sign and neg.verdict == "promising"
    passing = [{"name": "LOW_SHARPE", "result": "PASS", "limit": 1.58, "value": 2.0},
               {"name": "SELF_CORRELATION", "result": "PENDING"}]
    assert evaluate(sharpe=2.0, fitness=1.4, turnover=0.1, checks=passing).verdict == "strong"
    assert evaluate(sharpe=2.0, fitness=1.4, turnover=0.1, checks=passing, full_check_passed=True).verdict == "ready"
    assert evaluate(sharpe=2.0, fitness=1.4, turnover=0.1, checks=passing, status="ACTIVE").verdict == "submitted"


async def test_datetimes_round_trip_as_aware_utc(tmp_path):
    from datetime import datetime, timedelta, timezone

    from alphago.brain.quota import EASTERN

    await db.init(tmp_path / "dt.db")
    try:
        submitted = datetime(2026, 3, 9, 20, 30, tzinfo=EASTERN)   # stored in UTC regardless of input zone
        async with db.write() as s:
            s.add(Alpha(id="A1", date_submitted=submitted))
        async with db.session() as s:
            row = await s.get(Alpha, "A1", populate_existing=True)
            found = await s.scalar(select(Alpha.id).where(Alpha.date_submitted > submitted - timedelta(minutes=1)))
        assert row.date_submitted.tzinfo is not None
        assert row.date_submitted == submitted
        assert row.date_submitted.utcoffset() == timedelta(0) and row.date_submitted.tzinfo == timezone.utc
        assert found == "A1"
        assert (datetime.now(timezone.utc) - row.date_submitted).days > 0   # aware arithmetic no longer raises
    finally:
        await db.close()


async def test_llm_ideas_chunks_by_ideas_per_call(tmp_path, monkeypatch):
    from alphago.config import settings
    from alphago.engine import generators
    from alphago.llm.schemas import AlphaIdea, IdeaBatch

    await db.init(tmp_path / "llm.db")
    calls: list[int] = []
    counter = itertools.count(2)

    async def fake_generate_json(prompt, schema, **kw):
        if kw.get("purpose") == "self_check":      # the batch critique keeps every idea here
            return schema(reviews=[])
        k = int(prompt.split("Propose ")[1].split()[0]) if "Propose " in prompt else 0
        calls.append(k)
        ideas = [AlphaIdea(expression=f"rank(ts_mean(close, {next(counter)}))", idea="i", mechanism="momentum",
                           data_rationale="d", operator_rationale="o") for _ in range(k)]
        ideas.append(ideas[0])   # a duplicate inside the reply must be dropped
        return IdeaBatch(ideas=ideas)

    monkeypatch.setattr(generators.llm, "generate_json", fake_generate_json)
    monkeypatch.setattr(settings.llm, "ideas_per_call", 4)
    try:
        specs = await generators.llm_ideas({"region": "USA", "delay": 1, "universe": "TOP3000"}, dataset_id=None,
                                           n=10, goal="pyramids")
        assert calls == [4, 4, 2]
        assert len(specs) == 10 and len({s.expression for s in specs}) == 10
    finally:
        await db.close()
