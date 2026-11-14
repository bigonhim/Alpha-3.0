"""Request shaping of the OpenAI-compatible providers ported from alpha-harness, the provider catalogue, model
registry rules, temperature-refusal learning and the router's failover when a provider's pool is spent."""

from __future__ import annotations

import asyncio
import hashlib
import json

import httpx2
import openai
import pytest
from pydantic import BaseModel, Field

from alphago.ai import providers
from alphago.ai.keys import BudgetExhaustedError, keystore
from alphago.ai.registry import DEFAULT_PROMPT_TOKENS, ModelInfo, registry
from alphago.config import settings
from alphago.db import db
from alphago.llm import base
from alphago.llm import router as router_mod
from alphago.llm.base import TEMPERATURE, LLMError
from alphago.llm.openai_compat import CompatProvider, OpenAIProvider


class Ranked(BaseModel):
    items: list[str] = Field(min_length=3, max_length=5)


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_write_lock", asyncio.Lock())
    await db.init(tmp_path / "prov.db")
    registry.clear()
    keystore.clear()
    real_sleep = asyncio.sleep

    async def fast(_s):
        await real_sleep(0)

    monkeypatch.setattr(base.asyncio, "sleep", fast)
    yield
    registry.clear()
    keystore.clear()
    await db.close()


def test_catalogue_has_every_provider_with_alpha_harness_urls() -> None:
    assert providers.get("google").base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert providers.get("groq").base_url == "https://api.groq.com/openai/v1"
    assert providers.get("cerebras").base_url == "https://api.cerebras.ai/v1"
    assert providers.get("nvidia").base_url == "https://integrate.api.nvidia.com/v1"
    assert providers.get("mistral").base_url == "https://api.mistral.ai/v1"
    assert providers.get("huggingface").base_url == "https://router.huggingface.co/v1"
    assert providers.family("google") == "gemini" and providers.get("anthropic").id == "claude"
    assert [p.id for p in providers.key_providers()][-2:] in (["openai", "claude"], ["claude", "openai"])
    for pid in providers.COMPAT:
        assert pid in router_mod.PROVIDERS and pid in router_mod.PROVIDER_LABELS
    assert providers.get("nope").kind == "unknown"


def _shape(pid: str, model: str, thinking: str = "low", system: str | None = "SYS") -> dict:
    p = CompatProvider(pid)
    return p.build_request(model, "hello", system, Ranked, thinking, "schema")


def test_cache_key_only_for_mistral_and_official_openai() -> None:
    sent = _shape("mistral", "mistral-large")
    assert sent["extra_body"]["prompt_cache_key"] == hashlib.sha256(b"SYS").hexdigest()[:16]
    assert "extra_body" not in _shape("groq", "llama")
    official = OpenAIProvider().build_request("gpt-x", "hi", "SYS", None, "off", None)
    assert "prompt_cache_key" in official["extra_body"]


def test_array_lengths_kept_only_for_google_and_openai() -> None:
    google = json.dumps(_shape("google", "gemini-3-flash")["response_format"])
    cerebras = json.dumps(_shape("cerebras", "llama")["response_format"])
    assert "minItems" in google and "maxItems" in google
    assert "minItems" not in cerebras and "maxItems" not in cerebras
    assert json.loads(cerebras)["json_schema"]["strict"] is True


def test_reasoning_effort_only_for_gemini_3_on_google() -> None:
    assert _shape("google", "gemini-3-flash", "medium")["reasoning_effort"] == "medium"
    assert _shape("google", "gemini-3.8-flash", "minimal")["reasoning_effort"] == "low"
    assert "reasoning_effort" not in _shape("google", "gemini-2.5-flash", "high")
    assert "reasoning_effort" not in _shape("groq", "openai/gpt-oss-120b", "high")


def test_temperature_is_sent_when_asked() -> None:
    token = TEMPERATURE.set(0.6)
    try:
        assert _shape("groq", "llama")["temperature"] == 0.6
    finally:
        TEMPERATURE.reset(token)
    assert "temperature" not in _shape("groq", "llama")


async def test_temperature_refusal_is_learned_on_the_same_key(env, monkeypatch) -> None:
    await keystore.add("gsk_temperature0001", None, "groq")
    p = CompatProvider("groq")
    p._settings.ideation_model = "fixed-temp-model"
    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "temperature" in body:
            return httpx2.Response(400, json={"error": {"message": "temperature is not supported by this model"}})
        return httpx2.Response(200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                                          "choices": [{"index": 0, "finish_reason": "stop",
                                                       "message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr(p, "client_for", lambda secret: openai.AsyncOpenAI(
        api_key=secret, base_url=p.base_url, max_retries=0,
        http_client=openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))))
    assert await p.generate_text("hi", role="ideation", thinking="off", temperature=0.6) == "OK"
    assert await p.generate_text("again", role="ideation", thinking="off", temperature=0.6) == "OK"
    assert ["temperature" in b for b in seen] == [True, False, False]


async def test_registry_rules_and_order(env) -> None:
    with pytest.raises(LLMError, match="time zone"):
        await registry.set("groq", "m", 1, 1, "Mars/Olympus")
    with pytest.raises(LLMError, match="at least one request"):
        await registry.set("groq", "m", 0, 10, "UTC")
    with pytest.raises(LLMError, match="Name the model"):
        await registry.set("groq", " ", 1, 1, "UTC")
    await registry.set("groq", "small", 30, 1000, "UTC")
    await registry.set("google", "gemini-3-flash", 10, 250, "America/Los_Angeles", 20_000)
    await registry.set("openrouter", "x/y:free", 20, 1000, "UTC")
    assert [m.ref for m in registry.all()] == ["groq:small", "openrouter:x/y:free", "google:gemini-3-flash"]
    assert registry.get("openrouter:x/y:free").id == "x/y:free"           # model ids may hold a colon
    assert registry.get("gemini-3-flash").prompt_tokens == 20_000
    assert registry.info("groq", "unknown").prompt_tokens == DEFAULT_PROMPT_TOKENS
    assert not registry.info("groq", "unknown").registered
    registry.clear()
    await registry.load()                                                   # written through to the database
    assert len(registry.all()) == 3
    await registry.remove("groq", "small")
    with pytest.raises(LLMError, match="not set up"):
        await registry.remove("groq", "small")


async def test_spent_pool_fails_over_to_the_next_provider(monkeypatch) -> None:
    model = ModelInfo(id="m", provider="groq", rpm=1, rpd=1, reset_timezone="UTC")

    class Spent:
        name, label, local = "groq", "Groq", False

        def configure(self, api_key=None) -> bool:  # noqa: ANN001
            return True

        async def generate_json(self, prompt, schema, **kw):  # noqa: ANN001, ANN003, ANN201
            raise BudgetExhaustedError(model, [], 3600)

    class Backup(Spent):
        name, label = "cerebras", "Cerebras"

        async def generate_json(self, prompt, schema, **kw):  # noqa: ANN001, ANN003, ANN201
            return "backup answered"

    monkeypatch.setitem(router_mod.PROVIDERS, "groq", Spent())
    monkeypatch.setitem(router_mod.PROVIDERS, "cerebras", Backup())
    monkeypatch.setattr(settings.llm, "provider", "groq")
    monkeypatch.setattr(settings.llm, "fallbacks", ["cerebras"])
    monkeypatch.setattr(router_mod.bus, "publish", lambda *a, **k: None)
    assert await router_mod.LLMRouter().generate_json("p", dict) == "backup answered"
