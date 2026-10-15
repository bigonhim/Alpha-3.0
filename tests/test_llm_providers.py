"""LLM providers behind the router: Ollama (respx), OpenAI-compatible and Claude (httpx2 mock transport), helpers."""

from __future__ import annotations

import asyncio
import json

import anthropic
import httpx
import httpx2
import openai
import pytest
import respx
from sqlalchemy import select

from alphago.config import AppSettings, settings
from alphago.db import db
from alphago.db.models import LLMCall
from alphago.events import bus
from alphago.llm import base, openai_compat
from alphago.llm import claude as claude_mod
from alphago.llm.base import LLMError, inline_schema, parse_json, strict_schema
from alphago.llm.claude import ClaudeProvider
from alphago.llm.ollama import OllamaProvider
from alphago.llm.openai_compat import OpenAIProvider
from alphago.llm.router import llm
from alphago.llm.schemas import AlphaDescription, GeniusPlan, IdeaBatch

OLLAMA = "http://localhost:11434"
IDEAS = {"ideas": [{"expression": "rank(-ts_delta(close, 5))", "idea": "short-term reversal", "mechanism": "reversal",
                    "data_rationale": "price", "operator_rationale": "delta", "neutralization": "INDUSTRY",
                    "decay": 4, "truncation": 0.08}]}


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_write_lock", asyncio.Lock())
    await db.init(tmp_path / "llm.db")
    real_sleep = asyncio.sleep

    async def fast(_s):
        await real_sleep(0)

    monkeypatch.setattr(base.asyncio, "sleep", fast)        # pacing and back-off waits
    monkeypatch.setattr(settings.llm, "thinking_level", "low")
    yield
    await db.close()


async def _logged() -> list[LLMCall]:
    async with db.session() as s:
        return list(await s.scalars(select(LLMCall).order_by(LLMCall.id)))


# ---------------------------------------------------------------- helpers

def test_inline_and_strict_schema_keep_field_names_and_close_objects() -> None:
    flat = inline_schema(GeniusPlan)
    text = json.dumps(flat)
    assert "$ref" not in text and "$defs" not in text
    signal = flat["properties"]["signals"]["items"]
    assert "title" in signal["properties"]            # a field called "title" survives title-stripping
    strict = strict_schema(GeniusPlan)
    item = strict["properties"]["signals"]["items"]
    assert item["additionalProperties"] is False and set(item["required"]) == set(item["properties"])


def test_parse_json_tolerates_fences_think_blocks_and_chatter() -> None:
    body = json.dumps({"idea": "a", "rationale_data": "b", "rationale_operators": "c"})
    for text in (body, f"```json\n{body}\n```", f"<think>hmm</think>{body}", f"Sure! Here it is: {body} Hope it helps."):
        assert parse_json(text, AlphaDescription).idea == "a"


def test_old_settings_file_keeps_gemini_active() -> None:
    old = {"llm": {"ideation_model": "gemini-x", "thinking_level": "low", "daily_call_budget": 300}}
    s = AppSettings.model_validate(old)
    assert s.llm.provider == "gemini" and s.llm.ideation_model == "gemini-x"
    assert s.llm.ollama.base_url == OLLAMA and s.llm.openai.base_url == "https://api.openai.com/v1"


async def test_router_follows_the_active_provider(monkeypatch) -> None:
    monkeypatch.setattr(settings.llm, "provider", "ollama")
    assert llm.name == "ollama" and llm.local and llm.rpm is None
    monkeypatch.setattr(settings.llm, "provider", "claude")
    assert llm.label == "Claude" and llm.pick("ideation") == "claude-opus-5-5"
    monkeypatch.setattr(settings.llm, "provider", "nonsense")
    assert llm.name == "gemini"


async def test_budget_counts_cloud_calls_only(env) -> None:
    async with db.write() as s:
        s.add_all([LLMCall(provider=None, model="g", purpose="x"), LLMCall(provider="claude", model="c", purpose="x"),
                   LLMCall(provider="ollama", model="o", purpose="x"), LLMCall(provider="ollama", model="o", purpose="x")])
    assert await base.calls_today() == 2


# ---------------------------------------------------------------- Ollama

def _ollama_routes(chat):  # noqa: ANN001
    respx.get(f"{OLLAMA}/api/tags").mock(return_value=httpx.Response(200, json={"models": [
        {"name": "qwen-test:8b", "model": "qwen-test:8b", "size": 1, "details": {"family": "qwen"}}]}))
    respx.post(f"{OLLAMA}/api/show").mock(return_value=httpx.Response(200, json={
        "model_info": {"qwen.context_length": 8192}, "capabilities": ["completion", "thinking"]}))
    return respx.post(f"{OLLAMA}/api/chat").mock(side_effect=chat)


@respx.mock
async def test_ollama_sends_schema_context_and_think_and_is_not_budgeted(env, monkeypatch) -> None:
    monkeypatch.setattr(settings.llm, "daily_call_budget", 0)           # cloud budget exhausted: local still runs
    route = _ollama_routes(lambda req: httpx.Response(200, json={
        "message": {"role": "assistant", "content": json.dumps(IDEAS)}, "prompt_eval_count": 900, "eval_count": 80}))
    p = OllamaProvider()
    await p.list_models()
    assert p.pick("ideation") == "qwen-test:8b"
    batch = await p.generate_json("propose", IdeaBatch, system="pack", purpose="ideation")
    assert batch.ideas[0].expression == "rank(-ts_delta(close, 5))"
    sent = json.loads(route.calls[0].request.content)
    assert sent["options"]["num_ctx"] == 8192                          # capped at the model's context length
    assert sent["options"]["num_predict"] == 8192                      # a looping model cannot run forever
    assert sent["think"] is True and sent["stream"] is False
    assert sent["messages"][0] == {"role": "system", "content": "pack"}
    assert "$defs" not in json.dumps(sent["format"]) and sent["format"]["properties"]["ideas"]["type"] == "array"
    await p.generate_text("hi", thinking="minimal")
    assert json.loads(route.calls[1].request.content)["think"] is False
    calls = await _logged()
    assert [c.provider for c in calls] == ["ollama", "ollama"] and calls[0].prompt_tokens == 900
    await p.http.aclose()


@respx.mock
async def test_ollama_drops_think_when_rejected_and_warns_on_full_context(env) -> None:
    replies = iter([httpx.Response(400, json={"error": "model does not support thinking"}),
                    httpx.Response(200, json={"message": {"content": "OK"}, "prompt_eval_count": 8192})])
    route = _ollama_routes(lambda req: next(replies))
    p = OllamaProvider()
    await p.list_models()
    assert await p.generate_text("hi") == "OK"
    assert "think" in json.loads(route.calls[0].request.content)
    assert "think" not in json.loads(route.calls[1].request.content)
    assert any("was cut" in e.message for e in bus.recent(5))
    await p.http.aclose()


@respx.mock
async def test_ollama_reply_cut_at_the_limit_fails_without_a_repair_round(env) -> None:
    route = _ollama_routes(lambda req: httpx.Response(200, json={
        "message": {"content": '{"ideas": [{"expression": "rank('}, "done_reason": "length", "eval_count": 8192}))
    p = OllamaProvider()
    await p.list_models()
    with pytest.raises(LLMError, match="reply limit"):
        await p.generate_json("propose", IdeaBatch)
    assert route.call_count == 1
    await p.http.aclose()


@respx.mock
async def test_ollama_not_running_fails_fast_with_a_clear_message(env) -> None:
    respx.get(f"{OLLAMA}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    p = OllamaProvider()
    with pytest.raises(LLMError, match="not reachable"):
        await p.list_models()
    await p.http.aclose()


# ---------------------------------------------------------------- OpenAI / compatible

def _openai(monkeypatch, handler) -> OpenAIProvider:  # noqa: ANN001
    monkeypatch.setattr(openai_compat, "llm_key", lambda _p: "sk-test")
    p = OpenAIProvider()
    assert p.configure()
    p._client = openai.AsyncOpenAI(api_key="sk-test", base_url=p.base_url, max_retries=0,
                                   http_client=openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))
    return p


def _completion(content: str) -> dict:
    return {"id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}}


async def test_openai_degrades_json_mode_and_drops_reasoning_effort(env, monkeypatch) -> None:
    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "reasoning_effort" in body:
            return httpx2.Response(400, json={"error": {"message": "Unrecognized request argument supplied: "
                                                                   "reasoning_effort"}})
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx2.Response(400, json={"error": {"message": "response_format json_schema is not supported"}})
        return httpx2.Response(200, json=_completion(json.dumps(IDEAS)))

    p = _openai(monkeypatch, handler)
    monkeypatch.setattr(settings.llm.openai, "ideation_model", "my-model")
    batch = await p.generate_json("propose", IdeaBatch, purpose="ideation")
    assert batch.ideas[0].mechanism == "reversal"
    assert seen[0]["response_format"]["json_schema"]["strict"] is True and "reasoning_effort" in seen[0]
    assert "reasoning_effort" not in seen[1] and seen[2]["response_format"] == {"type": "json_object"}
    assert "JSON schema" in seen[2]["messages"][-1]["content"]
    # both lessons are remembered: the next call goes straight to json_object without reasoning_effort
    await p.generate_json("again", IdeaBatch)
    assert seen[3]["response_format"] == {"type": "json_object"} and "reasoning_effort" not in seen[3]


async def test_openai_429_slows_the_pacer_then_succeeds(env, monkeypatch) -> None:
    replies = iter([httpx2.Response(429, headers={"retry-after": "1"}, json={"error": {"message": "slow down"}}),
                    httpx2.Response(200, json=_completion("OK"))])
    p = _openai(monkeypatch, lambda request: next(replies))
    monkeypatch.setattr(settings.llm.openai, "describe_model", "my-model")
    start = p.rpm
    assert await p.generate_text("hi", thinking="off") == "OK"
    assert p.rpm == start * 0.5
    assert [c.status for c in await _logged()] == ["error", "ok"]


async def test_openai_model_list_hides_non_chat_models(env, monkeypatch) -> None:
    ids = ["gpt-test", "text-embedding-test", "whisper-1", "tts-1", "o-test"]
    p = _openai(monkeypatch, lambda request: httpx2.Response(200, json={"object": "list", "data": [
        {"id": i, "object": "model", "created": 0, "owned_by": "x"} for i in ids]}))
    assert [m["name"] for m in await p.list_models()] == ["gpt-test", "o-test"]
    assert p.pick("ideation") == ""                    # several models and none chosen: the user must pick


# ---------------------------------------------------------------- Claude

def _claude(monkeypatch, handler) -> ClaudeProvider:  # noqa: ANN001
    monkeypatch.setattr(claude_mod, "llm_key", lambda _p: "sk-ant-test")
    p = ClaudeProvider()
    assert p.configure()
    p._client = anthropic.AsyncAnthropic(api_key="sk-ant-test", max_retries=0, http_client=anthropic.DefaultAsyncHttpxClient(
        transport=httpx2.MockTransport(handler)))
    return p


def _message(text: str, stop: str = "end_turn", **extra) -> dict:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 40, "output_tokens": 90, "cache_read_input_tokens": 3000,
                      "cache_creation_input_tokens": 0}, **extra}


async def test_claude_structured_output_effort_cache_and_fallback(env, monkeypatch) -> None:
    seen: list[tuple[dict, dict]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append((dict(request.headers), json.loads(request.content)))
        return httpx2.Response(200, json=_message(json.dumps(IDEAS)))

    p = _claude(monkeypatch, handler)
    batch = await p.generate_json("propose", IdeaBatch, system="knowledge pack", purpose="ideation")
    assert batch.ideas[0].decay == 4
    headers, body = seen[0]
    assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in headers.get("anthropic-beta", "")
    assert body["output_config"]["effort"] == "low" and body["output_config"]["format"]["type"] == "json_schema"
    assert body["system"] == [{"type": "text", "text": "knowledge pack", "cache_control": {"type": "ephemeral"}}]
    assert "temperature" not in body and "thinking" not in body
    call = (await _logged())[0]
    assert call.provider == "claude" and call.cached_tokens == 3000


async def test_claude_haiku_gets_no_fallback_and_refusals_raise(env, monkeypatch) -> None:
    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=_message("", stop="refusal", stop_details={
            "type": "refusal", "category": "cyber", "explanation": "declined"}))

    p = _claude(monkeypatch, handler)
    monkeypatch.setattr(settings.llm.claude, "describe_model", "claude-haiku-5-5")
    with pytest.raises(LLMError, match="declined"):
        await p.generate_text("hi")
    assert "fallbacks" not in seen[0] and len(seen) == 1           # a refusal is not retried
