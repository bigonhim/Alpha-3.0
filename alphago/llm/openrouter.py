"""OpenRouter: one key, hundreds of models, many of them free (`:free`).

The free line-up changes every few weeks (models are added, rate-limited upstream or retired with a 404), so nothing
here is a fixed list: the catalogue is read live from OpenRouter's public `/models` endpoint and ranked. The ranking
uses each model's size, context window, supported parameters and retirement date, plus how it has actually behaved
in this app over the last two days (errors from the call log). Per call:

- the chosen model goes first, and the next healthy free models ride along in OpenRouter's `models` fallback list, so
  OpenRouter itself re-routes when the first one is down;
- if the request still fails with a model-specific error (404 retired, 429 upstream busy, 502/503 no endpoint), that
  model is put on a cooldown and the next ranked model is tried at once, without waiting;
- OpenRouter's account-wide daily cap for free models ("free-models-per-day") is reported as such, so the router can
  fail over to another provider instead of retrying.
"""

from __future__ import annotations

import logging
import math
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from pydantic import BaseModel
from sqlalchemy import case, func, select

from ..db import db
from ..db.models import LLMCall
from .base import CURRENT_KEY, LLMError, Reply
from .openai_compat import OpenAIProvider

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1"
ROUTE_FALLBACKS = 2               # extra models sent in the request's `models` list
MAX_ROTATIONS = 4                 # models tried in one call before giving up
COOL_BUSY_S = 90.0                # a model that answered 429 / 503 rests this long
COOL_GONE_S = 24 * 3600.0         # a model that answered 404 (retired / no endpoints) rests a day
# families with a track record of following structured-output instructions well; a small nudge only — size, context
# and observed health matter more, and an unknown new model still ranks on its own numbers
STRONG_FAMILIES = ("deepseek", "qwen3", "qwen-3", "kimi", "glm", "gpt-oss", "nemotron", "llama-4", "gemini", "gemma-3",
                   "gemma-4", "mistral", "minimax", "hermes", "grok", "north", "command", "dots")
WEAK_HINTS = ("nano", "tiny", "mini", "lite", "small", "lightning", "1b", "3b", "edge")
SKIP_IDS = ("openrouter/auto",)   # the paid auto-router


def _price(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 1.0


def is_free(m: dict[str, Any]) -> bool:
    mid = str(m.get("id") or "")
    if mid in SKIP_IDS:
        return False
    p = m.get("pricing") or {}
    return mid.endswith(":free") or (_price(p.get("prompt")) == 0 and _price(p.get("completion")) == 0)


def _size_b(mid: str) -> float:
    """Parameter count in billions read from the id ("…-120b", "235b-a22b" → 235); 0 when not stated."""
    sizes = [float(x) for x in re.findall(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z])", mid.lower())]
    return max(sizes) if sizes else 0.0


def text_model(m: dict[str, Any]) -> bool:
    arch = m.get("architecture") or {}
    out = set(arch.get("output_modalities") or ["text"])
    return out == {"text"} and not any(x in str(m.get("id", "")).lower() for x in ("embed", "tts", "whisper", "image-"))


def base_score(m: dict[str, Any], now: datetime | None = None) -> float:
    """Static quality guess for ranking: bigger, longer-context, JSON-capable, reasoning-capable models first."""
    mid = str(m.get("id") or "").lower()
    size = _size_b(mid)
    score = 2.0 * math.log2(1 + size) if size else 6.0          # unknown size: assume mid-sized
    ctx = int(m.get("context_length") or 0)
    if ctx:
        score += max(-2.0, min(3.0, math.log2(ctx / 32_768) + 1))
    params = set(m.get("supported_parameters") or [])
    if params & {"response_format", "structured_outputs"}:
        score += 2.0
    if params & {"reasoning", "include_reasoning"}:
        score += 1.0
    if params and "tools" in params:
        score += 0.3
    if any(f in mid for f in STRONG_FAMILIES):
        score += 1.5
    if any(re.search(rf"(?<![a-z0-9]){w}(?![a-z0-9])", mid) for w in WEAK_HINTS):
        score -= 3.0
    if any(x in mid for x in ("ultra", "super", "large", "max", "pro")):
        score += 1.0
    exp = m.get("expiration_date")
    if exp:
        try:
            when = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            days = (when - (now or datetime.now(timezone.utc))).total_seconds() / 86400
            if days <= 0:
                return -100.0                                   # retired: never ranked
            score -= 6.0 if days < 7 else 0.0
        except ValueError:
            pass
    return round(score, 3)


class OpenRouterProvider(OpenAIProvider):
    name = "openrouter"
    needs_key = True
    cache_key = False                 # alpha-harness sends no prompt_cache_key or array lengths to OpenRouter
    array_lengths = False
    extra_headers = {"HTTP-Referer": "https://github.com/alphago-brain", "X-Title": "AlphaGo BRAIN workstation"}

    def __init__(self) -> None:
        super().__init__()
        self._cool_until: dict[str, float] = {}
        self._health: dict[str, float] = {}          # model -> error share over the last 2 days
        self._health_at = 0.0
        # the free daily cap is per account, so per key: pooled key id (or None for the single saved key) -> until
        self._free_cap: dict[int | None, float] = {}

    @property
    def free_cap_until(self) -> float:
        """When the free daily cap of the key in use (an account) comes back; 0 when it is not reached."""
        choice = CURRENT_KEY.get()
        return self._free_cap.get(choice.key_id if choice else None, 0.0)

    @free_cap_until.setter
    def free_cap_until(self, value: float) -> None:
        choice = CURRENT_KEY.get()
        self._free_cap[choice.key_id if choice else None] = value

    async def list_model_ids(self, secret: str) -> list[str]:
        """The free key check. OpenRouter's model list is public, so it says nothing about a key: its /key endpoint
        does (and costs no request from the daily allowance)."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                r = await http.get(f"{self.base_url}/key", headers={"Authorization": f"Bearer {secret}"})
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenRouter could not be reached: {exc}", transient=True) from exc
        if r.status_code >= 400:
            raise LLMError(f"OpenRouter error {r.status_code}: {r.text[:300]}", r.status_code, detail=r.text[:2000])
        data = (r.json() or {}).get("data") or {}
        return [f"key {data.get('label') or 'ok'}" + (" · free tier" if data.get("is_free_tier") else "")]

    @property
    def base_url(self) -> str:
        return (self.cfg.base_url or OPENROUTER_URL).rstrip("/")

    @property
    def label(self) -> str:  # type: ignore[override]
        return "OpenRouter"

    @property
    def free_only(self) -> bool:
        return bool(getattr(self.cfg, "free_only", True))

    # ------------------------------------------------------------ catalogue

    async def fetch_catalogue(self) -> list[dict[str, Any]]:
        """OpenRouter's public model list (no key needed)."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                r = await http.get(f"{self.base_url}/models")
                r.raise_for_status()
                return list((r.json() or {}).get("data") or [])
        except Exception as exc:
            raise LLMError(f"OpenRouter model list unavailable: {exc}", transient=True) from exc

    def rank(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        out = []
        for m in raw:
            if not text_model(m):
                continue
            free = is_free(m)
            if self.free_only and not free:
                continue
            score = base_score(m, now)
            if score < -10:                          # already retired
                continue
            ctx = int(m.get("context_length") or 0)
            out.append({"name": str(m["id"]), "display": f"{m.get('name') or m['id']}"
                        + (f" · {ctx // 1000}k ctx" if ctx else "") + (" · free" if free else ""),
                        "preview": False, "free": free, "score": score, "context": ctx,
                        "json": bool(set(m.get("supported_parameters") or []) & {"response_format",
                                                                                 "structured_outputs"})})
        out.sort(key=lambda x: (-x["score"], x["name"]))
        return out

    async def list_models(self) -> list[dict[str, Any]]:
        models = self.rank(await self.fetch_catalogue())
        if not models:
            raise LLMError("OpenRouter lists no " + ("free " if self.free_only else "") + "text models right now.")
        return await self._remember_models(models)

    async def _refresh_health(self) -> None:
        if time.monotonic() - self._health_at < 300:
            return
        self._health_at = time.monotonic()
        since = datetime.now(timezone.utc) - timedelta(days=2)
        try:
            async with db.session() as s:
                rows = (await s.execute(select(LLMCall.model, func.count(LLMCall.id),
                                               func.sum(case((LLMCall.status != "ok", 1), else_=0)))
                                        .where(LLMCall.provider == self.name, LLMCall.ts >= since)
                                        .group_by(LLMCall.model))).all()
            self._health = {m: (bad or 0) / n for m, n, bad in rows if n >= 3}
        except Exception:
            log.debug("OpenRouter health refresh failed", exc_info=True)

    def healthy(self, exclude: set[str] | None = None) -> list[str]:
        """Ranked model ids that are not cooling down, best first (recent error share lowers a model's rank)."""
        now = time.monotonic()
        exclude = exclude or set()
        scored = [(m["score"] - 8.0 * self._health.get(m["name"], 0.0), m["name"]) for m in self.models
                  if m["name"] not in exclude and self._cool_until.get(m["name"], 0.0) <= now]
        return [name for _, name in sorted(scored, key=lambda x: (-x[0], x[1]))]

    def pick(self, role: str) -> str:
        explicit = self.explicit_model(role)
        if explicit:
            return explicit
        ranked = self.healthy() or [m["name"] for m in self.models]
        if not ranked:
            return ""
        # descriptions go to the runner-up, so the best model's per-model limits are kept for ideation/refinement
        return ranked[1] if role == "describe" and len(ranked) > 1 else ranked[0]

    def request_extras(self, model: str) -> dict[str, Any]:
        backups = [m for m in self.healthy({model})][:ROUTE_FALLBACKS]
        return {"models": [model, *backups]} if backups else {}

    def cool(self, model: str, seconds: float) -> None:
        self._cool_until[model] = time.monotonic() + seconds

    # ------------------------------------------------------------ calls

    async def _complete(self, model: str, prompt: str, system: str | None, schema: type[BaseModel] | None,
                        thinking: str) -> Reply:
        if self.free_cap_until > time.time():
            raise LLMError("OpenRouter: the daily free-model limit is reached (resets at midnight UTC; adding $10 of "
                           "credit raises it to 1,000 requests a day).", 429)
        await self._refresh_health()
        tried: set[str] = set()
        current = model
        last: LLMError | None = None
        for _ in range(MAX_ROTATIONS):
            tried.add(current)
            try:
                return await super()._complete(current, prompt, system, schema, thinking)
            except LLMError as exc:
                last = exc
                text = (exc.message or "").lower()
                if exc.code == 429 and ("per-day" in text or "per day" in text or "free-models-per" in text):
                    tomorrow = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0) + timedelta(days=1)
                    self.free_cap_until = tomorrow.timestamp()
                    raise LLMError(f"OpenRouter: daily free-model limit reached — {exc.message}", 429) from exc
                if exc.code == 404 or "no endpoints" in text or "not a valid model" in text:
                    self.cool(current, COOL_GONE_S)
                elif exc.code in (429, 502, 503, 529) or "overloaded" in text or "provider returned error" in text:
                    self.cool(current, COOL_BUSY_S)
                else:
                    raise
                nxt = next(iter(self.healthy(tried)), None)
                if not nxt:
                    raise
                log.info("OpenRouter: %s unavailable (%s), trying %s", current, exc.code, nxt)
                current = nxt
        raise last or LLMError("OpenRouter: no model answered")


openrouter = OpenRouterProvider()
