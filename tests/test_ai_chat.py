"""The assistant's turn (alpha-harness llm/chat.py `say`) with a fake model over a small DataField catalogue: the field
menu (coverage floor, dataset filter, prompt budget), transcript replay and clipping, dropped picks, persistence only
after an answer, plain-text degrade, the Search Lab hand-off, and the assistant / LLM page smoke tests."""

from __future__ import annotations

import asyncio
import json

import pytest
from nicegui.testing.user_simulation import user_simulation

from alphago.ai import chat
from alphago.ai.keys import keystore
from alphago.ai.prompts import ASSISTANT
from alphago.ai.registry import registry
from alphago.db import db
from alphago.db.models import DataField
from alphago.llm.base import LLMError

SCOPE = chat.Scope("USA", 1, "TOP3000")


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_write_lock", asyncio.Lock())
    await db.init(tmp_path / "chat.db")
    registry.clear()
    keystore.clear()
    async with db.write() as s:
        s.add_all([
            DataField(id="close", dataset_id="pv1", region="USA", delay=1, universe="TOP3000", coverage=1.0,
                      description="Daily close price"),
            DataField(id="est_eps", dataset_id="analyst4", region="USA", delay=1, universe="TOP3000", coverage=0.8,
                      description="Analyst EPS estimate,\n consensus mean"),
            DataField(id="news_buzz", dataset_id="news12", region="USA", delay=1, universe="TOP3000", coverage=0.1,
                      description="News volume"),
            DataField(id="eu_only", dataset_id="pv1", region="EUR", delay=1, universe="TOP2500", coverage=1.0,
                      description="Elsewhere"),
        ])
    yield
    registry.clear()
    keystore.clear()
    await db.close()


class FakeModel:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def __call__(self, provider, model, prompt, schema, **kw):  # noqa: ANN001, ANN003, ANN204
        self.calls.append({"provider": provider, "model": model, "prompt": prompt, "schema": schema, **kw})
        if not self.replies:
            raise LLMError("Gemini error 503: overloaded", 503)
        return self.replies.pop(0), provider, model or "fake-model"


def _reply(picks: list[tuple[str, str]], reply: str = "Here is what I picked.", datasets=("analyst4",)) -> str:
    return json.dumps({"reply": reply, "picks": [{"field": f, "why": w} for f, w in picks],
                       "datasets": list(datasets)})


async def test_say_drops_invented_fields_and_persists_the_exchange(env, monkeypatch) -> None:
    fake = FakeModel([_reply([("est_eps", "estimate revisions"), ("made_up", "nope")])])
    monkeypatch.setattr(chat.llm, "generate_raw_with", fake)
    out = await chat.say(None, "analysts raising estimates", scope=SCOPE, model="groq:llama", reasoning="careful")
    assert [p["field"] for p in out["picks"]] == ["est_eps"] and out["dropped"] == ["made_up"]
    call = fake.calls[0]
    assert call["provider"] == "groq" and call["model"] == "llama"
    assert call["system"] == ASSISTANT and call["temperature"] == 0.6 and call["thinking"] == "medium"
    assert call["schema"] is chat.FieldPickReply
    prompt = call["prompt"]
    assert "DATA FIELDS AVAILABLE IN USA/D1/TOP3000:" in prompt and "(this is the first message)" in prompt
    assert "  close (pv1): Daily close price" in prompt
    assert "est_eps (analyst4): Analyst EPS estimate, consensus mean" in prompt   # one line per field
    assert "news_buzz" not in prompt and "eu_only" not in prompt                   # coverage floor and scope
    found = await chat.thread(out["thread_id"])
    assert found is not None
    row, messages = found
    assert row.title == "analysts raising estimates" and (row.region, row.delay) == ("USA", 1)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].meta["dropped"] == ["made_up"] and messages[1].meta["reasoning"] == "careful"


async def test_naming_a_dataset_lifts_the_coverage_floor(env, monkeypatch) -> None:
    fake = FakeModel([_reply([("news_buzz", "buzz")], datasets=("news12",))])
    monkeypatch.setattr(chat.llm, "generate_raw_with", fake)
    out = await chat.say(None, "news", scope=SCOPE, dataset_ids=["news12"])
    assert [p["field"] for p in out["picks"]] == ["news_buzz"]
    assert "close (pv1)" not in fake.calls[0]["prompt"]


async def test_transcript_replays_picks_and_clips_long_turns(env, monkeypatch) -> None:
    long = "x" * 900
    fake = FakeModel([_reply([("close", "price")], reply="First " + "y" * 900), _reply([])])
    monkeypatch.setattr(chat.llm, "generate_raw_with", fake)
    first = await chat.say(None, long, scope=SCOPE)
    await chat.say(first["thread_id"], "why that one?", scope=chat.Scope("EUR", 1, "TOP2500"))
    prompt = fake.calls[1]["prompt"]
    assert f"  They: {'x' * 600}\n" in prompt and "x" * 601 not in prompt
    assert f"  You: First {'y' * 594} [picked: close]" in prompt
    assert "DATA FIELDS AVAILABLE IN USA/D1/TOP3000" in prompt           # the thread keeps its market
    assert prompt.rstrip().endswith("THEIR MESSAGE:\nwhy that one?")


async def test_history_is_capped_at_twelve_messages(env, monkeypatch) -> None:
    fake = FakeModel([_reply([], reply=f"answer {i}") for i in range(8)])
    monkeypatch.setattr(chat.llm, "generate_raw_with", fake)
    tid = None
    for i in range(8):
        tid = (await chat.say(tid, f"question {i}", scope=SCOPE))["thread_id"]
    last = fake.calls[-1]["prompt"]
    assert "question 0" not in last and "  They: question 1" in last            # 14 stored, last 12 replayed


async def test_nothing_is_stored_when_the_model_fails(env, monkeypatch) -> None:
    monkeypatch.setattr(chat.llm, "generate_raw_with", FakeModel([]))
    with pytest.raises(LLMError):
        await chat.say(None, "anything", scope=SCOPE)
    assert await chat.threads() == []


async def test_unparseable_reply_degrades_to_text(env, monkeypatch) -> None:
    monkeypatch.setattr(chat.llm, "generate_raw_with", FakeModel(["I could not find anything that fits."]))
    out = await chat.say(None, "weather", scope=SCOPE)
    assert out["reply"] == "I could not find anything that fits." and out["picks"] == []
    assert chat.parse('```json\n{"reply": "r", "picks": [{"field": 3}], "datasets": ["a"]}\n```') == {
        "reply": "r", "picks": [], "datasets": ["a"]}


async def test_empty_market_and_tiny_prompt_budget_are_explained(env, monkeypatch) -> None:
    monkeypatch.setattr(chat.llm, "generate_raw_with", FakeModel([_reply([])]))
    with pytest.raises(chat.ChatError, match="not in the local catalogue|No data fields"):
        await chat.say(None, "hi", scope=chat.Scope("JPN", 1, "TOP1600"))
    await registry.set("groq", "tiny", 10, 10, "UTC", max_prompt_tokens=250)
    with pytest.raises(chat.ChatError, match="no room for a single field"):
        await chat.say(None, "hi", scope=SCOPE, model="groq:tiny")
    fixed = len(ASSISTANT) + len(chat._transcript([])) + len("THEIR MESSAGE:\nhi") + 200
    await registry.set("groq", "small", 10, 10, "UTC", max_prompt_tokens=(fixed + 45) // 4 + 1)
    fake = FakeModel([_reply([])])
    monkeypatch.setattr(chat.llm, "generate_raw_with", fake)
    out = await chat.say(None, "hi", scope=SCOPE, model="groq:small")
    assert out["catalog_note"] and "widest-coverage" in out["catalog_note"]


async def test_hand_off_to_search_lab(env) -> None:
    payload = await chat.hand_off(SCOPE, ["analyst4", "analyst4"], ["est_eps"])
    assert await db.kv_get("handoff:search") == payload == {
        "region": "USA", "delay": 1, "universe": "TOP3000", "dataset_ids": ["analyst4"], "field_ids": ["est_eps"]}


# ---------------------------------------------------------------- pages

async def test_assistant_page_without_a_provider_explains_what_to_do(env, monkeypatch) -> None:
    from alphago.ui.layout import root
    monkeypatch.setattr(chat.llm, "configure", lambda *a, **k: False)
    async with user_simulation(root=root) as user:
        await user.open("/assistant")
        await user.should_see("The assistant needs a key")


async def test_assistant_page_shows_a_conversation_with_its_picks(env, monkeypatch) -> None:
    from alphago.ui.layout import root
    monkeypatch.setattr(chat.llm, "generate_raw_with", FakeModel([_reply([("est_eps", "estimate revisions"),
                                                                           ("ghost", "invented")])]))
    await chat.say(None, "analysts raising estimates", scope=SCOPE)
    monkeypatch.setattr(chat.llm, "configure", lambda *a, **k: True)
    async with user_simulation(root=root) as user:
        await user.open("/assistant")
        await user.should_see("Conversations")
        await user.should_see("analysts raising estimates")
        await user.should_see("Start with an idea")
        user.find("analysts raising estimates").click()
        await user.should_see("est_eps")
        await user.should_see("Use in Search Lab")
        await user.should_see("Named by the model but not in the catalogue, so dropped: ghost")


async def test_llm_page_shows_the_ai_screens(env) -> None:
    from alphago.ui.layout import root
    await keystore.add("gsk_pagesmoke000001", "smoke", "groq")
    await registry.set("groq", "llama-page", 30, 1000, "UTC")
    keystore.clear()
    async with user_simulation(root=root) as user:
        await user.open("/llm")
        for text in ("AI keys, models & budget", "Keys", "Models", "Budget", "Providers", "Prompts",
                     "Fallback providers", "AI self-check before simulating", "Model tournament (ideation)",
                     "Versions — every save is kept", "Check all keys", "Requests left today"):
            await user.should_see(text)
    from alphago import credentials
    for row in keystore.rows():
        credentials.set_pool_key(row.id, None)
