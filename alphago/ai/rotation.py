"""One provider request, sent on the key with the most budget left (ported from alpha-harness `LLMService.generate`).

Every AlphaGo LLM call (Gemini, Claude, OpenAI-compatible, OpenRouter) runs its single request through `complete`:

1. pick the key of the provider's pool with the most headroom for that model (`KeyStore.choose`), counting the
   request against the minute and the day before it leaves;
2. send it with that key (`CURRENT_KEY`, which each provider's client reads);
3. on success record what it really cost, and mark the key as working;
4. on a 429 believe the provider over the local count: "per day" / "daily" / "limit: 0" marks the pair spent for the
   model's day, anything else fills its minute window, and the next key is tried at once — no retrying into the
   same wall;
5. on 401 / 403 / insufficient_quota remember the refusal against the key and try the next one;
6. when no key is left, raise `BudgetExhaustedError` (a 429, so the router's cross-provider failover still applies),
   or the refusal if one key was turned away.

A provider with no key in the pool takes the single key saved on the LLM page exactly as before (`fn` runs once).
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from ..llm.base import CURRENT_KEY, KeyChoice, LLMError
from .keys import BudgetExhaustedError, keystore
from .registry import registry

R = TypeVar("R")
OnError = Callable[[LLMError, float], Awaitable[None]]


def _text(exc: LLMError) -> str:
    return f"{exc.message} {getattr(exc, 'detail', '') or ''}".lower()


def out_of_credit(text: str) -> bool:
    """An account with no credit left: a 429 naming a quota, but no wait ends it."""
    return "insufficient_quota" in text


def no_quota(text: str) -> bool:
    """Google says `limit: 0` when a model has no free quota at all on that key."""
    return re.search(r"limit:\s*0(?![\d.])|quotavalue\W+0(?![\d.])", text) is not None


def is_rate_limit(exc: LLMError) -> tuple[bool, bool]:
    """(rate limited, daily). Only the provider's details say which limit was hit."""
    if isinstance(exc, BudgetExhaustedError):
        return False, False
    text = _text(exc)
    if out_of_credit(text):
        return False, False
    limited = exc.code == 429 or "resource_exhausted" in text or "quota" in text
    daily = ("per day" in text or "perday" in text or "per-day" in text or "daily" in text or no_quota(text))
    return limited, daily


def refused(exc: LLMError) -> bool:
    """The key was turned away rather than the request: another key may get in."""
    text = _text(exc)
    return exc.code in (401, 403) or "api_key_invalid" in text or out_of_credit(text)


def rate_limit_reason(exc: LLMError, model_id: str, *, daily: bool) -> str:
    """What to show against a key the provider turned away, instead of its JSON."""
    if no_quota(_text(exc)):
        return f"{model_id} has no free quota on this key. Choose another model."
    if daily:
        return f"{model_id} has used its daily allowance on this key."
    return f"{model_id} is over its per-minute limit on this key."


def pooled(provider: str) -> bool:
    """Whether `provider`'s calls rotate over pooled keys."""
    return keystore.has_keys(provider)


async def complete(provider: str, model_id: str, fn: Callable[[], Awaitable[R]], *,
                   tokens: Callable[[R], int] = lambda _r: 0, on_error: OnError | None = None) -> R:
    """Run one request `fn` on the best key of `provider`'s pool (see the module docstring). `on_error` is awaited
    for every failed attempt, the budget refusal included, with the attempt's start time (for the call log)."""
    if not keystore.has_keys(provider):
        started = time.monotonic()
        try:
            return await fn()
        except LLMError as exc:
            if on_error is not None:
                await on_error(exc, started)
            raise

    model = registry.info(provider, model_id)
    ledger = keystore.ledger
    tried: set[int] = set()
    refusal: LLMError | None = None
    while True:
        started = time.monotonic()
        try:
            key_id = await keystore.choose(model, skip=tried)
        except BudgetExhaustedError as exhausted:
            err: LLMError = exhausted
            if refusal is not None:               # a key that was turned away says more than a budget
                err = refusal
            elif tried:
                exhausted.message = f"{exhausted.message} Already tried: {len(tried)} key(s)."
                exhausted.args = (exhausted.message,)
            if on_error is not None:
                await on_error(err, started)
            raise err from None
        tried.add(key_id)
        token = None
        try:
            try:
                secret = keystore.secret(key_id)
            except LLMError as exc:
                await keystore.mark(key_id, error=exc.message)
                refusal = exc
                continue
            token = CURRENT_KEY.set(KeyChoice(key_id, secret))
            try:
                result = await fn()
            except LLMError as exc:
                if on_error is not None:
                    await on_error(exc, started)
                limited, daily = is_rate_limit(exc)
                if limited:
                    await keystore.mark(key_id, error=rate_limit_reason(exc, model.id, daily=daily))
                    ledger.penalise(key_id, model, daily=daily)
                    continue
                if refused(exc):
                    await keystore.mark(key_id, error=exc.message[:300])
                    refusal = exc
                    continue
                if "timed out" in exc.message.lower() or "timeout" in exc.message.lower():
                    await ledger.record(key_id, model, 0)    # the provider had it and may bill it
                await keystore.mark(key_id, error=exc.message[:300])
                raise
            await ledger.record(key_id, model, int(tokens(result) or 0))
            await keystore.mark(key_id)
            return result
        finally:
            if token is not None:
                CURRENT_KEY.reset(token)
            ledger.settled(key_id, model)


def reply_tokens(reply: Any) -> int:
    """Tokens of a provider `Reply` (prompt + output)."""
    return int(getattr(reply, "prompt_tokens", 0) or 0) + int(getattr(reply, "output_tokens", 0) or 0)
