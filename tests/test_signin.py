"""Sign-in as BRAIN gates it: ALTCHA proof-of-work, a session kept across restarts, identity-check expiry."""

from __future__ import annotations

import base64
import hashlib
import json

import httpx
import pytest
import respx

from alphago.brain import session_store
from alphago.brain.altcha import AltchaError, Challenge, solve
from alphago.brain.client import BrainClient
from alphago.brain.errors import BrainError
from alphago.config import BrainSettings

API = "https://api.worldquantbrain.com"
AUTH_OK = {"user": {"id": "AB12345"}, "token": {"expiry": 14400.0}, "permissions": ["CONSULTANT"]}


def challenge(salt: str, number: int, maxnumber: int = 1_000_000) -> dict:
    return {"algorithm": "SHA-256", "challenge": hashlib.sha256(f"{salt}{number}".encode()).hexdigest(),
            "salt": salt, "signature": "sig", "maxnumber": maxnumber}


@pytest.mark.parametrize("number", [0, 7, 99, 100, 12_345, 999_999])
def test_solver_finds_the_number(number):
    sol = solve(Challenge.from_payload(challenge("abc?expires=1", number)))
    assert sol.number == number
    payload = json.loads(base64.b64decode(sol.encode()))
    assert payload["number"] == number and payload["signature"] == "sig" and list(payload) == sorted(payload)


def test_unsolvable_or_unknown_challenges_are_refused():
    with pytest.raises(AltchaError):
        solve(Challenge.from_payload(challenge("s", 500, maxnumber=100)))
    with pytest.raises(AltchaError):
        Challenge.from_payload({"algorithm": "SHA-1", "challenge": "00", "salt": "s"})


@respx.mock
async def test_login_sends_the_solved_captcha():
    respx.get(f"{API}/captcha").mock(return_value=httpx.Response(200, json=challenge("salty", 4242)))
    post = respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    c = BrainClient(BrainSettings(), lambda: ("me@example.com", "pw"))
    await c.login()
    sent = json.loads(post.calls.last.request.content)
    assert json.loads(base64.b64decode(sent["captcha"]))["number"] == 4242
    assert c.logged_in
    await c.close()
    session_store.clear()


@respx.mock
async def test_rejected_captcha_is_not_a_credential_refusal():
    respx.get(f"{API}/captcha").mock(return_value=httpx.Response(200, json=challenge("s", 3)))
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(400, json={"captcha": ["invalid"]}))
    c = BrainClient(BrainSettings(), lambda: ("me@example.com", "pw"))
    with pytest.raises(BrainError):
        await c.login()
    assert "captcha" in (c.last_error or "") and not c.silent_off
    await c.close()


@respx.mock
async def test_session_is_resumed_without_signing_in():
    respx.get(f"{API}/captcha").mock(return_value=httpx.Response(404))
    signin = respx.post(f"{API}/authentication").mock(return_value=httpx.Response(
        201, json=AUTH_OK, headers={"Set-Cookie": "t=abc; Path=/; Domain=api.worldquantbrain.com"}))
    respx.delete(f"{API}/authentication").mock(return_value=httpx.Response(204))
    first = BrainClient(BrainSettings(), lambda: ("me@example.com", "pw"))
    await first.login()
    await first.close()

    auth_get = respx.get(f"{API}/authentication").mock(return_value=httpx.Response(200, json=AUTH_OK))
    post_count = signin.call_count
    second = BrainClient(BrainSettings(), lambda: ("me@example.com", "pw"))
    assert await second.resume_session()
    assert second.logged_in and signin.call_count == post_count     # no new sign-in
    assert "t=abc" in auth_get.calls.last.request.headers.get("cookie", "")
    await second.logout()
    third = BrainClient(BrainSettings(), lambda: ("me@example.com", "pw"))
    assert not await third.resume_session()                                  # signed out: nothing kept
    await second.close()
    await third.close()


@respx.mock
async def test_expired_identity_check_stops_waiting():
    respx.post(f"{API}/authentication/persona").mock(return_value=httpx.Response(410, json={"detail": "gone"}))
    c = BrainClient(BrainSettings(), lambda: ("me@example.com", "pw"))
    await c.start()
    c.persona_url = f"{API}/authentication/persona?inquiry=inq_1"
    assert not await c.complete_persona(timeout_s=30, interval_s=0)
    assert c.persona_expired and "expired" in (c.last_error or "")
    await c.close()
