"""Provider-neutral LLM plumbing: the error type, adaptive pacing, call logging / daily budget, JSON helpers and
the retry loop shared by the Claude, OpenAI and Ollama providers (Gemini keeps its own loop in gemini.py)."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, TypeVar
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ValidationError
from sqlalchemy import func, or_, select

from ..config import settings
from ..db import db
from ..db.models import LLMCall

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
PACIFIC = ZoneInfo("America/Los_Angeles")
ROLES = ("ideation", "refine", "describe")

# per-call context: the model tournament forces a model; ideation records which prompt version it rendered
MODEL_OVERRIDE: ContextVar[str | None] = ContextVar("llm_model_override", default=None)
PROMPT_VERSION: ContextVar[int | None] = ContextVar("llm_prompt_version", default=None)

ERROR_KINDS = ("rate_limited", "timeout", "model_unavailable", "server_error", "auth", "budget", "bad_json", "other")


def classify_error(error: str | None) -> str | None:
    """Kind of a logged LLM error ("429: ...", "Gemini error 404: ...", "timed out", ...) for the Health page."""
    if not error:
        return None
    e = error.lower()
    code = None
    m = re.search(r"\b([1-5]\d\d)\b", e)
    if m:
        code = int(m.group(1))
    if "budget" in e:
        return "budget"
    if code == 429 or "rate limit" in e or "rate_limit" in e or "resource_exhausted" in e or "quota" in e \
            or "too many requests" in e:
        return "rate_limited"
    if code in (408, 504) or "timeout" in e or "timed out" in e:
        return "timeout"
    if code == 404 or ("model" in e and ("not found" in e or "does not exist" in e or "unavailable" in e
                                         or "not supported" in e)):
        return "model_unavailable"
    if code in (401, 403) or "api key" in e or "permission" in e or "unauthorized" in e:
        return "auth"
    if (code is not None and code >= 500) or "overloaded" in e or "unavailable" in e:
        return "server_error"
    if "json" in e or "schema" in e:
        return "bad_json"
    return "other"


def chosen_model(provider: Any, role: str) -> str:
    """The tournament's forced model for this call, else the provider's pick for the role."""
    return MODEL_OVERRIDE.get() or provider.pick(role)


class LLMError(Exception):
    def __init__(self, message: str, code: int | None = None, *, transient: bool = False,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.transient = transient          # worth retrying (throttled, overloaded, network blip)
        self.retry_after = retry_after


class Pacer:
    """Adaptive request spacing: starts at `rpm`, halves on every 429, speeds up 20 % after 8 successes (max 60)."""

    def __init__(self, rpm: float) -> None:
        self.rpm = rpm
        self._next_slot = 0.0
        self._lock = asyncio.Lock()
        self._ok_streak = 0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_slot = max(now, self._next_slot) + 60.0 / max(0.2, self.rpm)

    def throttled(self) -> None:
        self.rpm = max(0.5, self.rpm * 0.5)
        self._ok_streak = 0

    def succeeded(self) -> None:
        self._ok_streak += 1
        if self._ok_streak >= 8:
            self.rpm = min(60.0, self.rpm * 1.2)
            self._ok_streak = 0


# ---------------------------------------------------------------- usage log & budget

def _day_start() -> datetime:
    return datetime.now(PACIFIC).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


async def calls_today() -> int:
    """Cloud calls since midnight Pacific; local (Ollama) calls are logged but never count toward the budget."""
    async with db.session() as s:
        return await s.scalar(select(func.count(LLMCall.id)).where(
            LLMCall.ts >= _day_start(), or_(LLMCall.provider.is_(None), LLMCall.provider != "ollama"))) or 0


async def check_budget(label: str) -> None:
    if await calls_today() >= settings.llm.daily_call_budget:
        raise LLMError(f"Daily {label} call budget ({settings.llm.daily_call_budget}) reached — raise it on the "
                       "LLM Integration page or wait for the reset at midnight Pacific.")


async def log_call(provider: str, model: str, purpose: str, started: float, *, prompt_tokens: int | None = None,
                   output_tokens: int | None = None, cached_tokens: int | None = None, status: str = "ok",
                   error: str | None = None) -> None:
    try:
        async with db.write() as s:
            s.add(LLMCall(provider=provider, model=model[:64], purpose=purpose, prompt_tokens=prompt_tokens,
                          output_tokens=output_tokens, cached_tokens=cached_tokens,
                          latency_ms=int((time.monotonic() - started) * 1000), status=status, error=error,
                          error_kind=classify_error(error) if status != "ok" else None,
                          prompt_version=PROMPT_VERSION.get()))
    except Exception:
        log.exception("could not log LLM call")


# ---------------------------------------------------------------- JSON helpers

_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_json(text: str, schema: type[T]) -> T:
    """Validate a reply against `schema`, tolerating code fences, <think> blocks and chatter around the JSON."""
    cleaned = _FENCE.sub("", _THINK.sub("", text or "")).strip()
    try:
        return schema.model_validate_json(cleaned)
    except ValidationError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if 0 <= start < end and (start > 0 or end < len(cleaned) - 1):
            return schema.model_validate_json(cleaned[start:end + 1])
        raise


def inline_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """JSON schema with every $ref expanded and titles dropped (many servers ignore or reject $defs).
    The app's schemas are not recursive, so expansion terminates."""
    raw = schema.model_json_schema()
    defs = raw.pop("$defs", {})

    def walk(node: Any, properties: bool = False) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        if properties:                      # keys here are field names (a field may be called "title")
            return {k: walk(v) for k, v in node.items()}
        if "$ref" in node:
            return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
        return {k: walk(v, k == "properties") for k, v in node.items() if k != "title"}

    return walk(raw)


def strict_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Inline schema in OpenAI 'strict' form: every object closed and every property required."""
    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        out = {k: (({p: walk(v) for p, v in val.items()}) if k == "properties" else walk(val))
               for k, val in node.items()}
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    return walk(inline_schema(schema))


def schema_hint(schema: type[BaseModel]) -> str:
    return "Reply with JSON only (no prose, no code fences), matching this JSON schema:\n" + json.dumps(
        inline_schema(schema))


# ---------------------------------------------------------------- shared provider loop

@dataclass
class Reply:
    text: str = ""
    parsed: Any = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None


class ChatProvider:
    """Budget, pacing, retries, logging and the JSON repair round; subclasses implement `_complete`."""

    name = ""
    label = ""
    local = False                    # local providers: no budget, no pacing, one request at a time

    def __init__(self) -> None:
        self.models: list[dict[str, Any]] = []
        self.status = "unconfigured"
        self.last_error: str | None = None
        self.pacer = None if self.local else Pacer(settings.llm.start_rpm)
        self._one_at_a_time = asyncio.Lock() if self.local else None

    @property
    def cfg(self):  # noqa: ANN201  (config.ProviderSettings)
        return getattr(settings.llm, self.name)

    @property
    def rpm(self) -> float | None:
        return self.pacer.rpm if self.pacer else None

    def explicit_model(self, role: str) -> str:
        return getattr(self.cfg, f"{role}_model", "") or ""

    async def restore_models(self) -> None:
        cached = await db.kv_get(f"{self.name}_models")
        if cached:
            self.models = cached

    async def _remember_models(self, models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.models = models
        self.status = "ok"
        await db.kv_set(f"{self.name}_models", models)
        return models

    @asynccontextmanager
    async def _slot(self) -> AsyncIterator[None]:
        if self._one_at_a_time is None:
            yield
        else:
            async with self._one_at_a_time:
                yield

    # subclasses -------------------------------------------------------------
    def configure(self, api_key: str | None = None) -> bool:
        raise NotImplementedError

    def pick(self, role: str) -> str:
        raise NotImplementedError

    async def list_models(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def _complete(self, model: str, prompt: str, system: str | None, schema: type[BaseModel] | None,
                        thinking: str) -> Reply:
        raise NotImplementedError

    # shared -----------------------------------------------------------------
    async def _run(self, *, prompt: str, system: str | None, role: str, purpose: str,
                   schema: type[BaseModel] | None, thinking: str | None, max_attempts: int = 5) -> Reply:
        if not self.local:
            await check_budget(self.label)
        if not self.configure():
            raise LLMError(f"{self.label} is not set up — add it on the 'LLM Integration' page.")
        model = chosen_model(self, role)
        if not model:
            raise LLMError(f"No {self.label} model chosen — pick one on the 'LLM Integration' page.")
        level = (thinking or settings.llm.thinking_level or "").lower()
        attempt = 0
        while True:
            attempt += 1
            if self.pacer:
                await self.pacer.wait()
            started = time.monotonic()
            try:
                async with self._slot():
                    reply = await self._complete(model, prompt, system, schema, level)
            except LLMError as exc:
                await log_call(self.name, model, purpose, started, status="error",
                               error=f"{exc.code}: {exc.message}"[:1000])
                if exc.transient and attempt < max_attempts:
                    if exc.code == 429 and self.pacer:
                        self.pacer.throttled()
                        self.status = "rate_limited"
                    await asyncio.sleep(min(60.0, exc.retry_after or (2 ** attempt + random.random() * 2)))
                    continue
                self.status, self.last_error = "error", exc.message
                raise
            await log_call(self.name, model, purpose, started, prompt_tokens=reply.prompt_tokens,
                           output_tokens=reply.output_tokens, cached_tokens=reply.cached_tokens)
            if self.pacer:
                self.pacer.succeeded()
            self.status, self.last_error = "ok", None
            return reply

    async def generate_json(self, prompt: str, schema: type[T], *, system: str | None = None, role: str = "ideation",
                            purpose: str = "ideation", thinking: str | None = None) -> T:
        reply = await self._run(prompt=prompt, system=system, role=role, purpose=purpose, schema=schema,
                                thinking=thinking)
        if isinstance(reply.parsed, schema):
            return reply.parsed
        try:
            return parse_json(reply.text, schema)
        except ValidationError as exc:
            problems = [{"loc": e.get("loc"), "msg": e.get("msg")} for e in exc.errors()[:3]]
        retry = (f"{prompt}\n\nYour previous reply did not match the required JSON schema ({problems}). "
                 + schema_hint(schema))
        reply = await self._run(prompt=retry, system=system, role=role, purpose=purpose, schema=schema,
                                thinking=thinking)
        if isinstance(reply.parsed, schema):
            return reply.parsed
        try:
            return parse_json(reply.text, schema)
        except ValidationError as exc:
            raise LLMError(f"{self.label} returned malformed JSON for {schema.__name__}: {exc.errors()[:2]}") from exc

    async def generate_text(self, prompt: str, *, system: str | None = None, role: str = "describe",
                            purpose: str = "text", thinking: str | None = None) -> str:
        reply = await self._run(prompt=prompt, system=system, role=role, purpose=purpose, schema=None,
                                thinking=thinking)
        return _THINK.sub("", reply.text or "").strip()

    async def test(self) -> str:
        models = await self.list_models()
        if not self.pick("ideation"):
            return f"{len(models)} models available — choose the models below, then test again."
        reply = await self.generate_text("Reply with exactly: OK", role="describe", purpose="test", thinking="minimal")
        return f"{len(models)} models available · test reply: {reply[:40]!r} · using {self.pick('ideation')}"
