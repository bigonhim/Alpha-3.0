"""Quota safety of the simulation engine: a start is never paid for twice, and refusals are told apart.

Ported from alpha-harness' lifecycle and reconcile rules: the daily cap stops the queue instead of being retried as
a concurrency limit; a start that may have reached BRAIN (500/502/504, read timeout) is never resent but matched
against the alpha listing; a multi-simulation 400 that names some alphas rejects only those.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx
from sqlalchemy import select

from alphago.brain import brain
from alphago.brain import client as client_mod
from alphago.brain import simulations as sims_mod
from alphago.brain.errors import MaybeDelivered, QuotaExhausted, SimulationRejected, StartThrottled
from alphago.db import db
from alphago.db.models import Candidate
from alphago.engine import reconcile
from alphago.engine.simulator import CandidateSpec, SimulationService

API = "https://api.worldquantbrain.com"
AUTH = {"user": {"id": "U1"}, "token": {"expiry": 14400.0}, "permissions": []}
SETTINGS = {"region": "USA", "universe": "TOP3000", "delay": 1, "decay": 0, "neutralization": "INDUSTRY"}


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
    from alphago.brain.quota import QuotaTracker
    from alphago.brain.slots import SlotManager
    brain.client.quota = QuotaTracker()              # the shared client: leave no spent quota for the next test
    brain.client.slots = SlotManager(brain.client.cfg.max_slots)
    await brain.client.close()
    await db.close()


def _auth(router) -> None:  # noqa: ANN001
    router.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH))
    router.get(f"{API}/authentication").mock(return_value=httpx.Response(200, json=AUTH))


@respx.mock
async def test_daily_limit_429_is_quota_not_concurrency(env):
    _auth(respx)
    route = respx.post(f"{API}/simulations").mock(return_value=httpx.Response(
        429, json={"detail": "DAILY_SIMULATION_LIMIT_EXCEEDED"}))
    payload = sims_mod.regular_payload("rank(close)", SETTINGS)
    with pytest.raises(QuotaExhausted):
        await brain.sims.start([payload])
    assert route.call_count == 1
    assert brain.client.quota.effective_remaining == 0


@respx.mock
async def test_plain_rate_limit_429_waits_without_shrinking_slots(env):
    _auth(respx)
    respx.post(f"{API}/simulations").mock(return_value=httpx.Response(429, json={"detail": "API rate limit"}))
    limit = brain.client.slots.limit
    with pytest.raises(StartThrottled):
        await brain.sims.start([sims_mod.regular_payload("rank(close)", SETTINGS)])
    assert brain.client.slots.limit == limit


@respx.mock
async def test_server_error_on_start_is_never_resent(env):
    _auth(respx)
    route = respx.post(f"{API}/simulations").mock(return_value=httpx.Response(502, text="Bad gateway"))
    with pytest.raises(MaybeDelivered):
        await brain.sims.start([sims_mod.regular_payload("rank(close)", SETTINGS)])
    assert route.call_count == 1


@respx.mock
async def test_read_timeout_on_start_is_never_resent(env):
    _auth(respx)
    route = respx.post(f"{API}/simulations").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(MaybeDelivered):
        await brain.sims.start([sims_mod.regular_payload("rank(close)", SETTINGS)])
    assert route.call_count == 1


@respx.mock
async def test_503_on_start_is_retried_because_nothing_was_taken(env):
    _auth(respx)
    route = respx.post(f"{API}/simulations").mock(side_effect=[
        httpx.Response(503), httpx.Response(201, headers={"Location": f"{API}/simulations/S1"})])
    location = await brain.sims.start([sims_mod.regular_payload("rank(close)", SETTINGS)])
    assert location.endswith("/S1") and route.call_count == 2


@respx.mock
async def test_batch_400_names_the_faulty_alphas(env):
    _auth(respx)
    respx.post(f"{API}/simulations").mock(return_value=httpx.Response(
        400, json=[{}, {"regular": ["Unknown variable foo"]}]))
    payloads = [sims_mod.regular_payload("rank(close)", SETTINGS), sims_mod.regular_payload("rank(foo)", SETTINGS)]
    with pytest.raises(SimulationRejected) as info:
        await brain.sims.start(payloads)
    assert info.value.per_child is not None
    assert info.value.per_child[0] is None and "foo" in info.value.per_child[1]


@respx.mock
async def test_simulator_orphans_then_reconciles_by_listing(env, monkeypatch):
    _auth(respx)
    posts = respx.post(f"{API}/simulations").mock(return_value=httpx.Response(500, text="oops"))
    svc = SimulationService()
    monkeypatch.setattr("alphago.engine.simulator.simulator", svc)
    await brain.client.login()
    res = await svc.prepare([CandidateSpec("rank(-returns)", SETTINGS, origin="manual")])
    cand = res.created[0]
    async with db.session() as s:
        cand = await s.get(Candidate, cand.id)
    await svc._run_pack_inner([cand])
    assert posts.call_count == 1
    async with db.session() as s:
        row = await s.get(Candidate, cand.id)
    assert row.status == "orphaned" and row.sent_at is not None

    created = (row.sent_at + timedelta(seconds=20)).isoformat()
    alpha = {"id": "ORPH1", "type": "REGULAR", "status": "UNSUBMITTED", "dateCreated": created,
             "regular": {"code": "rank( -returns )"}, "settings": {**sims_mod.normalize_settings(SETTINGS)},
             "is": {"sharpe": 1.3, "fitness": 1.0, "turnover": 0.2, "checks": []}}
    respx.get(url__regex=rf"{API}/users/self/alphas\?.*").mock(
        return_value=httpx.Response(200, json={"count": 1, "results": [alpha], "next": None}))
    respx.get(f"{API}/alphas/ORPH1").mock(return_value=httpx.Response(200, json=alpha))
    counts = await reconcile.reconcile_once()
    assert counts["matched"] == 1
    async with db.session() as s:
        row = (await s.scalars(select(Candidate).where(Candidate.id == cand.id))).one()
    assert row.status == "done" and row.alpha_id == "ORPH1"
    assert posts.call_count == 1          # matched, never paid for twice


@respx.mock
async def test_orphan_not_on_brain_is_requeued_after_30_minutes(env, monkeypatch):
    _auth(respx)
    svc = SimulationService()
    monkeypatch.setattr("alphago.engine.simulator.simulator", svc)
    await brain.client.login()
    res = await svc.prepare([CandidateSpec("rank(-returns)", SETTINGS, origin="manual")])
    sent = datetime.now(timezone.utc) - timedelta(minutes=45)
    async with db.write() as s:
        row = await s.get(Candidate, res.created[0].id)
        row.status, row.sent_at = "orphaned", sent
    respx.get(url__regex=rf"{API}/users/self/alphas\?.*").mock(
        return_value=httpx.Response(200, json={"count": 0, "results": [], "next": None}))
    counts = await reconcile.reconcile_once()
    assert counts["requeued"] == 1
    async with db.session() as s:
        row = await s.get(Candidate, res.created[0].id)
    assert row.status == "queued" and row.orphan_requeues == 1
    assert svc.queued_count == 1


def test_squash_and_match_tolerate_brain_echo():
    assert reconcile.squash("{ rank(close) ;}") == "rank(close)"
    alpha = {"regular": {"code": "rank(close)"}, "settings": {**sims_mod.normalize_settings(SETTINGS),
                                                             "truncation": 0.08000000001}}
    assert reconcile.matches("rank( close )", SETTINGS, alpha)
    assert not reconcile.matches("rank(close)", {**SETTINGS, "decay": 4}, alpha)


async def test_refused_login_turns_silent_sign_in_off(env):
    with respx.mock:
        respx.post(f"{API}/authentication").mock(return_value=httpx.Response(
            401, json={"detail": "INVALID_CREDENTIALS"}))
        from alphago.brain.errors import AuthError
        with pytest.raises(AuthError):
            await brain.client.login()
    assert brain.client.silent_off
    assert "will not retry" in (brain.client.last_error or "")
    assert not brain.client.may_sign_in_silently()
    brain.client.silent_off = False
