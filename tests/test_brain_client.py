"""BRAIN client protocol tests against a mocked API (respx)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from alphago.brain import client as client_mod
from alphago.brain.alphas import AlphasAPI
from alphago.brain.client import BrainClient
from alphago.brain.errors import AuthError, PersonaRequired
from alphago.brain.options import SimOptions
from alphago.brain.simulations import SimulationAPI, regular_payload, settings_hash
from alphago.config import BrainSettings

API = "https://api.worldquantbrain.com"
AUTH_OK = {"user": {"id": "AB12345"}, "token": {"expiry": 14400.0}, "permissions": ["CONSULTANT"]}


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    real_sleep = asyncio.sleep

    async def _sleep(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(client_mod.asyncio, "sleep", _sleep)
    from alphago.brain import alphas as alphas_mod, simulations as sims_mod
    monkeypatch.setattr(alphas_mod.asyncio, "sleep", _sleep)
    monkeypatch.setattr(sims_mod.asyncio, "sleep", _sleep)


def make_client(creds=("me@example.com", "pw")) -> BrainClient:
    return BrainClient(BrainSettings(max_slots=2), lambda: creds)


@respx.mock
async def test_login_ok_sets_session():
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    c = make_client()
    await c.login()
    assert c.logged_in and c.user_id == "AB12345"
    assert 14000 < c.seconds_left <= 14400
    await c.close()


@respx.mock
async def test_login_bad_credentials():
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(401, json={"detail": "Invalid"}))
    c = make_client()
    with pytest.raises(AuthError):
        await c.login()
    await c.close()


async def test_login_without_credentials():
    c = make_client(creds=None)
    with pytest.raises(AuthError):
        await c.login()
    await c.close()


@respx.mock
async def test_persona_flow():
    route = respx.post(f"{API}/authentication")
    route.mock(return_value=httpx.Response(
        401, headers={"WWW-Authenticate": "persona", "Location": "/authentication/persona?inquiry=inq_1"}))
    persona = respx.post(f"{API}/authentication/persona").mock(side_effect=[
        httpx.Response(401), httpx.Response(201, json=AUTH_OK)])
    respx.get(f"{API}/authentication").mock(return_value=httpx.Response(200, json=AUTH_OK))
    c = make_client()
    with pytest.raises(PersonaRequired) as info:
        await c.login()
    assert "inquiry=inq_1" in info.value.url
    assert c.state == "persona_required"
    assert await c.complete_persona(timeout_s=60)
    assert c.logged_in and persona.call_count == 2
    await c.close()


@respx.mock
async def test_single_simulation_polls_retry_after_and_tracks_quota():
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    respx.post(f"{API}/simulations").mock(return_value=httpx.Response(
        201, headers={"Location": f"{API}/simulations/S1", "X-Ratelimit-Limit": "5000",
                      "X-Ratelimit-Remaining": "4321", "X-Ratelimit-Reset": "3600"}))
    respx.get(f"{API}/simulations/S1").mock(side_effect=[
        httpx.Response(200, headers={"Retry-After": "2.5"}, json={"progress": 0.4}),
        httpx.Response(200, headers={"Retry-After": "1"}, json={"progress": 0.9}),
        httpx.Response(200, json={"id": "S1", "status": "COMPLETE", "alpha": "A1"}),
    ])
    c = make_client()
    sims = SimulationAPI(c)
    progress = []
    outcomes = await sims.run([regular_payload("rank(close)")], on_progress=progress.append)
    assert outcomes[0].alpha_id == "A1" and outcomes[0].ok
    assert progress == [0.4, 0.9]
    assert c.quota.limit == 5000 and c.quota.remaining == 4320  # header then -1 locally
    assert 3500 < c.quota.seconds_to_reset <= 3600
    await c.close()


@respx.mock
async def test_multi_simulation_children():
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    respx.post(f"{API}/simulations").mock(return_value=httpx.Response(201, headers={"Location": f"{API}/simulations/P"}))
    respx.get(f"{API}/simulations/P").mock(side_effect=[
        httpx.Response(200, headers={"Retry-After": "1"}, json={"progress": 0.5}),
        httpx.Response(200, json={"id": "P", "status": "COMPLETE", "children": ["C1", "C2", "C3"]}),
    ])
    respx.get(f"{API}/simulations/C1").mock(return_value=httpx.Response(200, json={"id": "C1", "status": "COMPLETE", "alpha": "A1"}))
    respx.get(f"{API}/simulations/C2").mock(return_value=httpx.Response(
        200, json={"id": "C2", "status": "ERROR", "message": "Attempted to use unknown variable \"foo\""}))
    respx.get(f"{API}/simulations/C3").mock(return_value=httpx.Response(200, json={"id": "C3", "status": "WARNING", "alpha": "A3"}))
    c = make_client()
    outcomes = await SimulationAPI(c).run([regular_payload(e) for e in ("rank(close)", "rank(foo)", "rank(open)")])
    assert [o.alpha_id for o in outcomes] == ["A1", None, "A3"]
    assert "unknown variable" in outcomes[1].message
    await c.close()


@respx.mock
async def test_concurrency_limit_reduces_slots_then_retries():
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    respx.post(f"{API}/simulations").mock(side_effect=[
        httpx.Response(429, json={"detail": "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"}),
        httpx.Response(201, headers={"Location": f"{API}/simulations/S9"}),
    ])
    respx.get(f"{API}/simulations/S9").mock(return_value=httpx.Response(200, json={"status": "COMPLETE", "alpha": "A9"}))
    c = make_client()
    outcomes = await SimulationAPI(c).run([regular_payload("rank(close)")])
    assert outcomes[0].alpha_id == "A9"
    assert c.slots.limit == 1  # dropped from 2
    await c.close()


@respx.mock
async def test_relogin_on_401():
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    respx.get(f"{API}/alphas/A1").mock(side_effect=[httpx.Response(401), httpx.Response(200, json={"id": "A1"})])
    c = make_client()
    assert (await AlphasAPI(c).get("A1"))["id"] == "A1"
    await c.close()


@respx.mock
async def test_submit_paths():
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    c = make_client()
    api = AlphasAPI(c)
    # queued (503) -> in flight -> accepted -> ACTIVE
    respx.post(f"{API}/alphas/A1/submit").mock(return_value=httpx.Response(503, headers={"Retry-After": "1"}))
    respx.get(f"{API}/alphas/A1/submit").mock(side_effect=[
        httpx.Response(303, headers={"Location": f"{API}/alphas/A1/submit"}),
        httpx.Response(200, json={}),
    ])
    respx.get(f"{API}/alphas/A1").mock(side_effect=[
        httpx.Response(200, json={"status": "UNSUBMITTED"}), httpx.Response(200, json={"status": "ACTIVE"})])
    assert (await api.submit("A1")).status == "active"
    # rejected (403 with checks)
    respx.post(f"{API}/alphas/A2/submit").mock(return_value=httpx.Response(403, json={"is": {"checks": [
        {"name": "SELF_CORRELATION", "result": "FAIL", "limit": 0.7, "value": 0.82}]}}))
    res = await api.submit("A2")
    assert res.status == "rejected" and "SELF_CORRELATION" in res.message
    # throttled
    respx.post(f"{API}/alphas/A3/submit").mock(return_value=httpx.Response(429, json={"detail": "THROTTLED"}))
    assert (await api.submit("A3")).status == "throttled"
    await c.close()


@respx.mock
async def test_check_long_poll():
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    respx.get(f"{API}/alphas/A1/check").mock(side_effect=[
        httpx.Response(200, headers={"Retry-After": "2"}),
        httpx.Response(200, json={"is": {"checks": [{"name": "LOW_SHARPE", "result": "PASS", "limit": 1.58, "value": 2.1},
                                                     {"name": "SELF_CORRELATION", "result": "PASS", "limit": 0.7, "value": 0.3}]}}),
    ])
    c = make_client()
    checks = await AlphasAPI(c).check("A1")
    assert {ch["name"] for ch in checks} == {"LOW_SHARPE", "SELF_CORRELATION"}
    await c.close()


def test_options_parser():
    payload = {"actions": {"POST": {
        "type": {"choices": [{"value": "REGULAR", "label": "Regular"}, {"value": "SUPER", "label": "Super"}]},
        "settings": {"children": {
            "region": {"choices": {"instrumentType": {"EQUITY": [{"value": "USA", "label": "USA"}, {"value": "GLB", "label": "GLB"}]}}},
            "universe": {"choices": {"instrumentType": {"EQUITY": {"region": {
                "USA": [{"value": "TOP3000", "label": "TOP3000"}], "GLB": [{"value": "MINVOL1M", "label": "MINVOL1M"}]}}}}},
            "delay": {"choices": {"instrumentType": {"EQUITY": {"region": {"USA": [{"value": 1}, {"value": 0}], "GLB": [{"value": 1}]}}}}},
            "pasteurization": {"choices": [{"value": "ON"}, {"value": "OFF"}]},
            "truncation": {"type": "float", "minValue": 0, "maxValue": 1},
        }}}}}
    opts = SimOptions.from_payload(payload)
    assert opts.regions == ["USA", "GLB"]
    assert opts.universes("GLB") == ["MINVOL1M"]
    assert opts.delays("USA") == [1, 0]
    assert opts.choices("pasteurization") == ["ON", "OFF"]
    assert opts.ranges["truncation"] == (0, 1)


def test_settings_hash_ignores_visualization():
    a = settings_hash({"region": "USA", "decay": 4, "visualization": False})
    b = settings_hash({"region": "USA", "decay": 4, "visualization": True})
    c = settings_hash({"region": "USA", "decay": 5})
    assert a == b != c
