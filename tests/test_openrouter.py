"""OpenRouter provider: live free-model catalogue, ranking, fallback routing and model rotation."""

from __future__ import annotations

import asyncio
import json

import httpx
import httpx2
import openai
import pytest
import respx

from alphago.config import settings
from alphago.db import db
from alphago.llm import base, openai_compat
from alphago.llm.base import LLMError
from alphago.llm.openrouter import OPENROUTER_URL, OpenRouterProvider, base_score, is_free
from alphago.llm.router import PROVIDERS
from alphago.llm.schemas import IdeaBatch

IDEAS = {"ideas": [{"expression": "rank(-ts_delta(close, 5))", "idea": "reversal", "mechanism": "reversal",
                    "data_rationale": "price", "operator_rationale": "delta"}]}
FREE = {"prompt": "0", "completion": "0"}
CATALOGUE = {"data": [
    {"id": "acme/giant-405b:free", "name": "Giant", "context_length": 131072, "pricing": FREE,
     "supported_parameters": ["response_format", "reasoning"]},
    {"id": "acme/mid-70b:free", "name": "Mid", "context_length": 65536, "pricing": FREE,
     "supported_parameters": ["response_format"]},
    {"id": "acme/tiny-3b:free", "name": "Tiny", "context_length": 8192, "pricing": FREE},
    {"id": "acme/retired-120b:free", "name": "Old", "context_length": 131072, "pricing": FREE,
     "expiration_date": "2020-01-01"},
    {"id": "acme/paid-200b", "name": "Paid", "context_length": 131072, "pricing": {"prompt": "0.000002",
                                                                                 "completion": "0.000008"}},
    {"id": "openrouter/auto", "name": "Auto", "pricing": {"prompt": "-1", "completion": "-1"}},
    {"id": "acme/painter:free", "name": "Image", "pricing": FREE,
     "architecture": {"output_modalities": ["image"]}},
]}


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_write_lock", asyncio.Lock())
    await db.init(tmp_path / "or.db")
    real_sleep = asyncio.sleep

    async def fast(_s):
        await real_sleep(0)

    monkeypatch.setattr(base.asyncio, "sleep", fast)
    monkeypatch.setattr(settings.llm, "thinking_level", "off")
    monkeypatch.setattr(settings.llm.openrouter, "free_only", True)
    for role in ("ideation_model", "refine_model", "describe_model"):
        monkeypatch.setattr(settings.llm.openrouter, role, "")
    yield
    await db.close()


def _completion(content: str) -> dict:
    return {"id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _provider(monkeypatch, handler) -> OpenRouterProvider:  # noqa: ANN001
    monkeypatch.setattr(openai_compat, "llm_key", lambda _p: "or-test")
    p = OpenRouterProvider()
    assert p.configure()
    p._client = openai.AsyncOpenAI(api_key="or-test", base_url=p.base_url, max_retries=0,
                                   http_client=openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))
    return p


def test_free_detection_and_scores() -> None:
    assert is_free(CATALOGUE["data"][0]) and not is_free(CATALOGUE["data"][4]) and not is_free(CATALOGUE["data"][5])
    giant, tiny, old = CATALOGUE["data"][0], CATALOGUE["data"][2], CATALOGUE["data"][3]
    assert base_score(giant) > base_score(tiny)
    assert base_score(old) < -10                                   # past its retirement date


def test_registered_in_the_router() -> None:
    assert "openrouter" in PROVIDERS and PROVIDERS["openrouter"].label == "OpenRouter"


@respx.mock
async def test_live_catalogue_lists_ranked_free_text_models_only(env, monkeypatch) -> None:
    respx.get(f"{OPENROUTER_URL}/models").mock(return_value=httpx.Response(200, json=CATALOGUE))
    p = _provider(monkeypatch, lambda r: httpx2.Response(500))
    names = [m["name"] for m in await p.list_models()]
    assert names == ["acme/giant-405b:free", "acme/mid-70b:free", "acme/tiny-3b:free"]
    assert p.pick("ideation") == "acme/giant-405b:free" and p.pick("describe") == "acme/mid-70b:free"
    monkeypatch.setattr(settings.llm.openrouter, "free_only", False)
    assert "acme/paid-200b" in [m["name"] for m in await p.list_models()]


@respx.mock
async def test_sends_backup_models_and_rotates_past_a_retired_model(env, monkeypatch) -> None:
    respx.get(f"{OPENROUTER_URL}/models").mock(return_value=httpx.Response(200, json=CATALOGUE))
    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body["model"] == "acme/giant-405b:free":
            return httpx2.Response(404, json={"error": {"message": "No endpoints found for acme/giant-405b:free"}})
        return httpx2.Response(200, json=_completion(json.dumps(IDEAS)))

    p = _provider(monkeypatch, handler)
    await p.list_models()
    batch = await p.generate_json("propose", IdeaBatch)
    assert batch.ideas[0].mechanism == "reversal"
    assert seen[0]["models"] == ["acme/giant-405b:free", "acme/mid-70b:free", "acme/tiny-3b:free"]
    assert seen[1]["model"] == "acme/mid-70b:free"                 # rotated at once, no waiting
    assert p.pick("ideation") == "acme/mid-70b:free"               # the retired model rests


@respx.mock
async def test_daily_free_cap_is_reported_not_retried(env, monkeypatch) -> None:
    respx.get(f"{OPENROUTER_URL}/models").mock(return_value=httpx.Response(200, json=CATALOGUE))
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        return httpx2.Response(429, json={"error": {"message": "Rate limit exceeded: free-models-per-day"}})

    p = _provider(monkeypatch, handler)
    await p.list_models()
    with pytest.raises(LLMError) as err:
        await p.generate_text("hi")
    assert "daily free-model limit" in err.value.message and err.value.code == 429
    assert calls["n"] == 1
