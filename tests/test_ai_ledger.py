"""Budget ledger and key rotation (alpha-harness llm/budget.py + KeyStore.choose + LLMService.generate's loop), with a
fake clock: sliding minute window, model-day counting in the model's time zone, in-flight requests, caps, penalties,
and rotation over real providers' error shapes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from alphago.ai import rotation
from alphago.ai.budget import Ledger, seconds_until_reset
from alphago.ai.keys import BudgetExhaustedError, KeyStore
from alphago.ai.registry import ModelInfo, ModelRegistry
from alphago.db import db
from alphago.llm.base import CURRENT_KEY, LLMError


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.now = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)

    def mono(self) -> float:
        return self.t

    def utc(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.t += seconds
        self.now += timedelta(seconds=seconds)


MODEL = ModelInfo(id="gemini-x", provider="gemini", rpm=2, rpd=3, reset_timezone="America/Los_Angeles")


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_write_lock", asyncio.Lock())
    await db.init(tmp_path / "ai.db")
    clock = Clock()
    models = ModelRegistry()
    store = KeyStore(Ledger(clock.mono, clock.utc), models)
    monkeypatch.setattr(rotation, "keystore", store)
    monkeypatch.setattr(rotation, "registry", models)
    yield store, clock, models
    await db.close()


async def test_minute_window_slides_and_says_when_it_frees(env) -> None:
    store, clock, _ = env
    ledger = store.ledger
    ledger.sent(1, MODEL)
    clock.advance(10)
    ledger.sent(1, MODEL)
    state = await ledger.headroom(1, MODEL)
    assert state.blocked_by == "requests_per_minute" and state.retry_after == pytest.approx(50)
    clock.advance(51)
    assert (await ledger.headroom(1, MODEL)).available            # the first request aged out


async def test_in_flight_counts_until_settled_and_day_persists(env) -> None:
    store, clock, _ = env
    ledger = store.ledger
    (key,) = await _keys(store, 1)
    for _ in range(3):
        ledger.sent(key, MODEL)
        clock.advance(61)
    state = await ledger.headroom(key, MODEL)
    assert state.blocked_by == "requests_per_day" and state.requests_today == 3   # in flight already counts
    for _ in range(3):
        await ledger.record(key, MODEL, 100)
        ledger.settled(key, MODEL)
    fresh = Ledger(clock.mono, clock.utc)                          # a restart: the day comes back from the database
    state = await fresh.headroom(key, MODEL)
    assert state.requests_today == 3 and state.blocked_by == "requests_per_day"
    # midnight Pacific is 08:00 UTC: until then it waits for the reset, after it the budget is back
    assert state.retry_after == pytest.approx(20 * 3600 - 183)
    clock.advance(20 * 3600 + 1)
    assert (await Ledger(clock.mono, clock.utc).headroom(key, MODEL)).requests_today == 0


def test_reset_is_built_from_the_next_local_date() -> None:
    from zoneinfo import ZoneInfo
    # the day the clocks go back in Los Angeles (1 Nov 2026) is 25 hours long
    before = datetime(2026, 11, 1, 7, 0, tzinfo=timezone.utc)       # its midnight, PDT
    assert seconds_until_reset(ZoneInfo("America/Los_Angeles"), before) == pytest.approx(25 * 3600)


async def test_paid_cap_counts_every_model(env) -> None:
    store, clock, _ = env
    ledger = store.ledger
    (key,) = await _keys(store, 1)
    other = ModelInfo(id="other", provider="gemini", rpm=100, rpd=1000, reset_timezone="America/Los_Angeles")
    big = ModelInfo(id="big", provider="gemini", rpm=100, rpd=1000, reset_timezone="America/Los_Angeles")
    for _ in range(2):
        ledger.sent(key, other)
        await ledger.record(key, other, 10)
        ledger.settled(key, other)
    state = await ledger.headroom(key, big, cap=2)
    assert state.blocked_by == "requests_per_day" and state.requests_per_day == 2


async def test_penalties_daily_and_minute(env) -> None:
    store, clock, _ = env
    ledger = store.ledger
    ledger.penalise(1, MODEL, daily=False)
    state = await ledger.headroom(1, MODEL)
    assert state.blocked_by == "requests_per_minute" and state.requests_this_minute == MODEL.rpm
    clock.advance(61)
    assert (await ledger.headroom(1, MODEL)).available
    ledger.penalise(1, MODEL, daily=True)
    state = await ledger.headroom(1, MODEL)
    assert state.blocked_by == "requests_per_day" and state.requests_today == MODEL.rpd
    unlimited = ModelInfo(id="auto", provider="gemini", rpm=1_000_000, rpd=10**9, reset_timezone="UTC",
                          registered=False)
    ledger.penalise(2, unlimited, daily=False)                     # cooled for a minute, not filled with 1e6 entries
    state = await ledger.headroom(2, unlimited)
    assert state.blocked_by == "requests_per_minute" and 59 <= state.retry_after <= 60


async def _keys(store: KeyStore, n: int, provider: str = "gemini") -> list[int]:
    return [(await store.add(f"AIzaSyTESTKEY{provider}{i:04d}xyz", f"k{i}", provider)).id for i in range(n)]


async def test_choose_prefers_most_headroom_then_ok_keys(env) -> None:
    store, clock, _ = env
    a, b = await _keys(store, 2)
    ledger = store.ledger
    model = ModelInfo(id="m", provider="gemini", rpm=100, rpd=10, reset_timezone="UTC")
    for _ in range(3):
        ledger.sent(a, model)
        await ledger.record(a, model, 1)
        ledger.settled(a, model)
    assert await store.choose(model, skip=set()) == b             # most daily budget left
    ledger.settled(b, model)
    await store.mark(b, error="boom")
    assert await store.choose(model, skip=set()) == a             # a key whose last call failed goes last
    ledger.settled(a, model)
    with pytest.raises(BudgetExhaustedError):
        await store.choose(model, skip={a, b})


async def test_rotation_penalises_a_429_and_moves_on(env) -> None:
    store, clock, models = env
    a, b = await _keys(store, 2)
    await models.set("gemini", "m", 5, 100, "America/Los_Angeles")
    used: list[int] = []

    async def fn():
        key = CURRENT_KEY.get()
        used.append(key.key_id)
        if key.key_id == a:
            raise LLMError("Gemini error 429: Resource has been exhausted", 429,
                           detail="quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        return "ok"

    assert await rotation.complete("gemini", "m", fn) == "ok"
    assert used == [a, b]
    info = models.info("gemini", "m")
    assert (await store.ledger.headroom(a, info)).blocked_by == "requests_per_day"
    assert "daily allowance" in store.get(a).last_error and store.get(b).last_ok_at is not None
    assert (await store.ledger.headroom(b, info)).requests_today == 1          # recorded, in-flight settled
    assert store.ledger.in_flight(b, "m") == 0
    # the next call goes straight to b: a is spent for its Pacific day
    used.clear()
    assert await rotation.complete("gemini", "m", fn) == "ok" and used == [b]


async def test_minute_429_fills_the_window_and_all_spent_raises_budget_message(env) -> None:
    store, clock, models = env
    a, b = await _keys(store, 2)
    await models.set("gemini", "m", 5, 100, "America/Los_Angeles")

    async def fn():
        raise LLMError("Gemini error 429: rate limit", 429, detail="GenerateRequestsPerMinutePerProjectPerModel")

    with pytest.raises(BudgetExhaustedError) as err:
        await rotation.complete("gemini", "m", fn)
    assert "per-minute limit for m. Try again in about" in err.value.message
    assert "Already tried: 2 key(s)." in err.value.message and err.value.code == 429 and err.value.transient
    info = models.info("gemini", "m")
    assert (await store.ledger.headroom(a, info)).requests_this_minute == 5


async def test_daily_exhaustion_message_names_the_reset(env) -> None:
    store, clock, models = env
    (a,) = await _keys(store, 1)
    await models.set("gemini", "m", 5, 100, "America/Los_Angeles")
    store.ledger.penalise(a, models.info("gemini", "m"), daily=True)

    async def fn():
        return "never"

    with pytest.raises(BudgetExhaustedError) as err:
        await rotation.complete("gemini", "m", fn)
    assert "resets in about 20 hours (midnight, America/Los_Angeles)" in err.value.message
    assert err.value.daily and not err.value.transient


async def test_refused_key_rotates_and_is_reported_when_none_answer(env) -> None:
    store, clock, models = env
    a, b = await _keys(store, 2, "groq")

    async def fn():
        key = CURRENT_KEY.get()
        if key.key_id == a:
            raise LLMError("Groq error 401: the server rejected the API key", 401)
        raise LLMError("Groq: quota exhausted — You exceeded your current quota", 429,
                       detail='{"code": "insufficient_quota"}')

    with pytest.raises(LLMError) as err:
        await rotation.complete("groq", "llama", fn)
    assert not isinstance(err.value, BudgetExhaustedError)         # the refusal says more than a budget
    assert store.get(a).last_error and store.get(b).last_error
    # out of credit is a refusal, not a rate limit: nothing was penalised
    assert (await store.ledger.headroom(b, models.info("groq", "llama"))).available


async def test_no_pool_runs_the_single_key_path(env) -> None:
    async def fn():
        assert CURRENT_KEY.get() is None
        return 42

    assert await rotation.complete("claude", "claude-opus-5-5", fn) == 42


def test_rate_limit_classification() -> None:
    assert rotation.is_rate_limit(LLMError("x 429", 429, detail="limit: 0")) == (True, True)
    assert rotation.is_rate_limit(LLMError("RESOURCE_EXHAUSTED per day")) == (True, True)
    assert rotation.is_rate_limit(LLMError("slow down", 429)) == (True, False)
    assert rotation.is_rate_limit(LLMError("429", 429, detail="insufficient_quota")) == (False, False)
    assert rotation.refused(LLMError("forbidden", 403))
