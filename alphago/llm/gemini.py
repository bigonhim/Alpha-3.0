"""Gemini client: model discovery, structured JSON output, adaptive request pacing and usage accounting.

Notes (Oct 2026): `temperature` is deprecated for current Gemini models, so variety comes from prompts; reasoning
depth is set with `thinking_level`. Rate limits are per Google Cloud project and are learned from 429 responses.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import Any, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from ..config import settings
from ..credentials import gemini_key
from ..db import db
from .base import LLMError, Pacer, chosen_model, log_call
from .base import calls_today as _cloud_calls_today

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_EXCLUDE = ("embedding", "image", "tts", "audio", "live", "aqa", "imagen", "veo", "native", "robotics", "computer-use",
            "learnlm", "gemma")


def _version(name: str) -> tuple[float, ...]:
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    if not m:
        return (0.0,)
    return (float(m.group(1)), float(m.group(2) or 0))


def _tier(name: str) -> int:
    n = name.lower()
    if "pro" in n or "argon" in n or "ultra" in n:
        return 3
    if "lite" in n:
        return 1
    if "flash" in n:
        return 2
    return 0


class GeminiClient:
    name = "gemini"
    label = "Gemini"
    local = False

    def __init__(self) -> None:
        self._client: genai.Client | None = None
        self._key: str | None = None
        self.models: list[dict[str, Any]] = []
        self.pacer = Pacer(settings.llm.start_rpm)
        self.status = "unconfigured"
        self.last_error: str | None = None

    @property
    def rpm(self) -> float:
        return self.pacer.rpm

    # ------------------------------------------------------------ setup

    def configure(self, api_key: str | None = None) -> bool:
        key = api_key or gemini_key()
        if not key:
            self._client, self.status = None, "unconfigured"
            return False
        if key != self._key or self._client is None:
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)))
            self._key = key
            self.status = "configured"
        return True

    @property
    def client(self) -> genai.Client:
        if self._client is None and not self.configure():
            raise LLMError("No Gemini API key configured — add one on the 'LLM Integration' page.")
        assert self._client is not None
        return self._client

    async def list_models(self) -> list[dict[str, Any]]:
        out = []
        pager = await self.client.aio.models.list()
        async for m in pager:
            name = (m.name or "").removeprefix("models/")
            actions = [a.lower() for a in (m.supported_actions or [])]
            if "generatecontent" not in [a.replace("_", "") for a in actions]:
                continue
            if not name.startswith("gemini") or any(x in name.lower() for x in _EXCLUDE):
                continue
            out.append({"name": name, "display": m.display_name or name, "input_limit": m.input_token_limit,
                        "output_limit": m.output_token_limit, "thinking": bool(getattr(m, "thinking", False)),
                        "preview": any(x in name for x in ("preview", "exp"))})
        out.sort(key=lambda d: (_version(d["name"]), _tier(d["name"]), not d["preview"]), reverse=True)
        self.models = out
        self.status = "ok"
        await db.kv_set("gemini_models", out)
        return out

    async def restore_models(self) -> None:
        cached = await db.kv_get("gemini_models")
        if cached:
            self.models = cached

    def pick(self, role: str) -> str:
        cfg = settings.llm
        explicit = {"ideation": cfg.ideation_model, "refine": cfg.refine_model, "describe": cfg.describe_model}.get(role)
        if explicit:
            return explicit
        names = [m["name"] for m in self.models if not m["preview"]] or [m["name"] for m in self.models]
        if not names:
            return "gemini-flash-latest"

        def best(tier: int) -> str | None:
            cands = [n for n in names if _tier(n) == tier]
            return max(cands, key=_version) if cands else None

        if role == "refine":
            return best(3) or best(2) or names[0]
        if role == "describe":
            return best(1) or best(2) or names[0]
        return best(2) or best(1) or names[0]

    # ------------------------------------------------------------ budget & pacing

    async def calls_today(self) -> int:
        return await _cloud_calls_today()

    async def _pace(self) -> None:
        await self.pacer.wait()

    def _on_throttle(self) -> None:
        self.pacer.throttled()
        self.status = "rate_limited"

    def _on_success(self) -> None:
        self.pacer.succeeded()
        self.status = "ok"

    async def _log(self, model: str, purpose: str, started: float, response: Any = None, error: str | None = None,
                   status: str = "ok") -> None:
        usage = getattr(response, "usage_metadata", None)
        await log_call(self.name, model, purpose, started,
                       prompt_tokens=getattr(usage, "prompt_token_count", None),
                       output_tokens=getattr(usage, "candidates_token_count", None),
                       cached_tokens=getattr(usage, "cached_content_token_count", None), status=status, error=error)

    # ------------------------------------------------------------ generation

    async def _generate(self, *, prompt: str, system: str | None, role: str, purpose: str,
                        schema: type[BaseModel] | None, thinking: str | None, max_attempts: int = 5) -> Any:
        if await self.calls_today() >= settings.llm.daily_call_budget:
            raise LLMError(f"Daily Gemini call budget ({settings.llm.daily_call_budget}) reached — raise it on the "
                           "LLM Integration page or wait for the reset at midnight Pacific.")
        client = self.client          # raises LLMError right away when no key is configured (never retried)
        model = chosen_model(self, role)
        level = (thinking or settings.llm.thinking_level or "").upper()
        use_thinking = level in ("MINIMAL", "LOW", "MEDIUM", "HIGH")
        attempt = 0
        while True:
            attempt += 1
            config: dict[str, Any] = {}
            if system:
                config["system_instruction"] = system
            if schema is not None:
                config["response_mime_type"] = "application/json"
                config["response_schema"] = schema
            if use_thinking:
                config["thinking_config"] = types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, level))
            await self._pace()
            started = time.monotonic()
            try:
                response = await client.aio.models.generate_content(
                    model=model, contents=prompt, config=types.GenerateContentConfig(**config))
            except genai_errors.APIError as exc:
                code = getattr(exc, "code", None)
                msg = getattr(exc, "message", None) or str(exc)
                await self._log(model, purpose, started, error=f"{code}: {msg}"[:1000], status="error")
                if code == 400 and use_thinking and "think" in msg.lower():
                    use_thinking = False  # older model: no thinking_level support
                    continue
                if code in (429, 408, 500, 502, 503, 504) and attempt < max_attempts:
                    if code == 429:
                        self._on_throttle()
                        if "quota" in msg.lower() and "day" in msg.lower():
                            raise LLMError("Gemini daily quota exhausted for this project — it resets at midnight "
                                           "Pacific.", 429) from exc
                    await asyncio.sleep(min(60.0, 2 ** attempt + random.random() * 2))
                    continue
                self.status, self.last_error = "error", msg
                raise LLMError(f"Gemini error {code}: {msg}", code) from exc
            except LLMError:
                raise
            except Exception as exc:  # network
                await self._log(model, purpose, started, error=str(exc)[:1000], status="error")
                if attempt < max_attempts:
                    await asyncio.sleep(min(30.0, 2 ** attempt))
                    continue
                raise LLMError(f"Gemini request failed: {exc}") from exc
            await self._log(model, purpose, started, response=response)
            self._on_success()
            return response

    async def generate_json(self, prompt: str, schema: type[T], *, system: str | None = None, role: str = "ideation",
                            purpose: str = "ideation", thinking: str | None = None) -> T:
        response = await self._generate(prompt=prompt, system=system, role=role, purpose=purpose, schema=schema,
                                        thinking=thinking)
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        text = (getattr(response, "text", None) or "").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(f"Gemini returned malformed JSON for {schema.__name__}: {exc.errors()[:2]}") from exc

    async def generate_text(self, prompt: str, *, system: str | None = None, role: str = "describe",
                            purpose: str = "text", thinking: str | None = None) -> str:
        response = await self._generate(prompt=prompt, system=system, role=role, purpose=purpose, schema=None,
                                        thinking=thinking)
        return (getattr(response, "text", None) or "").strip()

    async def test(self) -> str:
        models = await self.list_models()
        reply = await self.generate_text("Reply with exactly: OK", role="describe", purpose="test", thinking="minimal")
        return f"{len(models)} models available · test reply: {reply[:40]!r} · using {self.pick('ideation')}"


gemini = GeminiClient()
