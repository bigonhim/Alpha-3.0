"""Key pool rules (duplicates, hints, paid caps), keys kept in the (in-memory) Credential Manager, the free key check
over models.list, adopting the single saved key, and how pooled keys reach each provider's client."""

from __future__ import annotations

import asyncio
import json

import anthropic
import httpx2
import openai
import pytest
from sqlalchemy import select

from alphago import credentials
from alphago.ai import service
from alphago.ai.keys import fingerprint, hint, keystore
from alphago.ai.registry import registry
from alphago.db import db
from alphago.db.models_ai import AIKey
from alphago.llm import base
from alphago.llm.base import LLMError
from alphago.llm.router import PROVIDERS


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_write_lock", asyncio.Lock())
    await db.init(tmp_path / "keys.db")
    keystore.clear()
    keystore.ledger.reset()
    registry.clear()
    real_sleep = asyncio.sleep

    async def fast(_s):
        await real_sleep(0)

    monkeypatch.setattr(base.asyncio, "sleep", fast)
    yield
    for row in keystore.rows():
        credentials.set_pool_key(row.id, None)
    keystore.clear()
    keystore.ledger.reset()
    registry.clear()
    await db.close()


def test_hint_and_fingerprint() -> None:
    assert hint("AIzaSyABCDEFGHIJKLmnop") == "AIzaSy…mnop"
    assert hint("short") == "…ort"
    assert fingerprint(" key ") == fingerprint("key") and len(fingerprint("key")) == 64


async def test_add_stores_secret_in_keyring_not_database(env) -> None:
    row = await keystore.add("  gsk_abcdefghijklmnop1234 ", None, "groq")
    assert credentials.pool_key(row.id) == "gsk_abcdefghijklmnop1234"
    assert row.hint == "gsk_ab…1234" and row.label == "Groq key 1" and row.enabled
    async with db.session() as s:
        stored = (await s.scalars(select(AIKey))).one()
    assert "gsk_abcdefghijklmnop1234" not in json.dumps({c: str(getattr(stored, c)) for c in
                                                         ("label", "hint", "fingerprint", "last_error")})
    await keystore.remove(row.id)
    assert credentials.pool_key(row.id) is None and keystore.rows() == []


async def test_duplicates_are_refused_across_labels(env) -> None:
    await keystore.add("AIzaSyDUPLICATEKEY0001", "personal", "gemini")
    with pytest.raises(LLMError, match="already stored as 'personal'"):
        await keystore.add("AIzaSyDUPLICATEKEY0001", "work", "google")       # google uses the same AI Studio keys


async def test_paid_providers_need_a_cap_that_cannot_be_cleared(env) -> None:
    with pytest.raises(LLMError, match="need a daily request cap"):
        await keystore.add("sk-ant-api03-abcdefgh", None, "anthropic")        # alpha-harness name of claude
    with pytest.raises(LLMError, match="at least one request"):
        await keystore.add("gsk_abcdefghijk00", None, "groq", daily_limit=0)
    row = await keystore.add("sk-ant-api03-abcdefgh", None, "claude", daily_limit=50)
    assert row.provider == "claude" and row.daily_limit == 50
    with pytest.raises(LLMError, match="cannot be removed"):
        await keystore.set_enabled(row.id, True, clear=True)
    moved = await keystore.set_enabled(row.id, False, cap=80)
    assert moved.daily_limit == 80 and not moved.enabled
    free = await keystore.add("gsk_freefreefree0001", None, "groq", daily_limit=10)
    assert (await keystore.set_enabled(free.id, True, clear=True)).daily_limit is None
    assert keystore.usable("claude") == []                                   # disabled keys do not rotate


async def test_mirror_reloads_from_database(env) -> None:
    row = await keystore.add("csk-abcdefghijklmn", "mine", "cerebras")
    keystore.clear()
    assert not keystore.has_keys("cerebras")
    await keystore.load()
    assert keystore.has_keys("cerebras") and keystore.get(row.id).label == "mine"


async def test_saved_single_key_is_adopted_once(env) -> None:
    credentials.set_llm_key("openrouter", "sk-or-v1-savedkey0001")
    credentials.set_llm_key("claude", "sk-ant-savedkey000001")
    try:
        assert sorted(await keystore.adopt_saved_keys(300)) == ["claude", "openrouter"]
        assert await keystore.adopt_saved_keys(300) == []
        claude_row = keystore.usable("claude")[0]
        assert claude_row.label == "Saved key" and claude_row.daily_limit == 300
    finally:
        credentials.set_llm_key("openrouter", None)
        credentials.set_llm_key("claude", None)


def _mock_openai(captured: list[httpx2.Request], body: dict, status: int = 200):  # noqa: ANN202
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        return httpx2.Response(status, json=body)
    return lambda secret: openai.AsyncOpenAI(api_key=secret, base_url="https://api.groq.com/openai/v1", max_retries=0,
                                             http_client=openai.DefaultAsyncHttpxClient(
                                                 transport=httpx2.MockTransport(handler)))


async def test_free_key_check_lists_models_and_marks_the_key(env, monkeypatch) -> None:
    row = await keystore.add("gsk_checkme000000001", None, "groq")
    seen: list[httpx2.Request] = []
    groq = PROVIDERS["groq"]
    monkeypatch.setattr(groq, "client_for", _mock_openai(seen, {"object": "list", "data": [
        {"id": "llama-test", "object": "model", "created": 0, "owned_by": "x"}]}))
    result = await service.check_key(row.id)
    assert result["ok"] and result["models"] == 1
    assert seen[0].url.path.endswith("/models") and seen[0].method == "GET"
    assert seen[0].headers["authorization"] == "Bearer gsk_checkme000000001"
    assert keystore.get(row.id).last_ok_at is not None
    seen.clear()
    monkeypatch.setattr(groq, "client_for", _mock_openai(seen, {"error": {"message": "Invalid API Key"}}, 401))
    result = await service.check_key(row.id)
    assert not result["ok"] and "rejected" in result["error"] and keystore.get(row.id).last_error


async def test_anthropic_key_check_asks_for_a_thousand_models(env, monkeypatch) -> None:
    row = await keystore.add("sk-ant-checkme00001", None, "claude", daily_limit=5)
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json={"data": [{"id": "claude-opus-5-5", "type": "model", "display_name": "Opus",
                                                    "created_at": "2026-01-01T00:00:00Z"}],
                                          "has_more": False, "first_id": "a", "last_id": "a"})

    claude = PROVIDERS["claude"]
    monkeypatch.setattr(claude, "client_for", lambda secret: anthropic.AsyncAnthropic(
        api_key=secret, max_retries=0, http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(handler))))
    result = await service.check_key(row.id)
    assert result["ok"] and seen[0].url.params["limit"] == "1000"
    assert "sk-ant-checkme00001" in (seen[0].headers.get("x-api-key") or seen[0].headers.get("authorization"))


async def test_pooled_keys_configure_providers_and_rotate_real_requests(env, monkeypatch) -> None:
    groq = PROVIDERS["groq"]
    assert not groq.configure()
    a = await keystore.add("gsk_rotateAAAA00001", "a", "groq")
    b = await keystore.add("gsk_rotateBBBB00002", "b", "groq")
    assert groq.configure()
    await registry.set("groq", "llama-test", 30, 1000, "UTC")
    assert groq.pick("ideation") == "llama-test"                    # the set-up model is the auto-pick
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        auth = request.headers["authorization"]
        seen.append(auth)
        if auth.endswith("AAAA00001"):
            return httpx2.Response(429, json={"error": {"message": "Rate limit reached: requests per day (RPD)"}})
        return httpx2.Response(200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                                          "choices": [{"index": 0, "finish_reason": "stop",
                                                       "message": {"role": "assistant", "content": "OK"}}],
                                          "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})

    monkeypatch.setattr(groq, "client_for", lambda secret: openai.AsyncOpenAI(
        api_key=secret, base_url=groq.base_url, max_retries=0,
        http_client=openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))))
    assert await groq.generate_text("hi", thinking="off") == "OK"
    assert seen == ["Bearer gsk_rotateAAAA00001", "Bearer gsk_rotateBBBB00002"]
    status = await keystore.status()
    row = next(b_ for b_ in status["budget"] if b_["ref"] == "groq:llama-test")
    assert row["allowed_today"] == 2000 and row["remaining_today"] == 999     # a spent for the day, b used once
    key_b = next(k for k in status["keys"] if k["id"] == b.id)
    assert key_b["requests_today"] == 1 and key_b["tokens_today"] == 7
    assert "daily allowance" in keystore.get(a.id).last_error
