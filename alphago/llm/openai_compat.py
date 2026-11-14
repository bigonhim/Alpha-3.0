"""OpenAI, or any server speaking the OpenAI chat API (OpenRouter, Groq, DeepSeek, LM Studio, vLLM, ...).

Structured output degrades gracefully: strict `json_schema` → `json_object` → schema in the prompt; the first mode
a server accepts is remembered. `reasoning_effort` is sent until a model rejects it; a temperature a model refuses is
dropped for that model from then on (nothing publishes which models take one).

The OpenAI-compatible providers from alpha-harness (Google's compatible endpoint, Groq, Cerebras, NVIDIA NIM,
Mistral, Hugging Face) are `CompatProvider`s at the bottom, shaped as alpha-harness sends them: `prompt_cache_key`
for Mistral and OpenAI, `minItems`/`maxItems` stripped except for Google and OpenAI, `reasoning_effort` only for
Gemini 3 models on Google (low instead of minimal on 3.7 / 3.8).

When the provider has keys in the pool, every request runs on the key `alphago.ai.rotation` chose (`CURRENT_KEY`);
with none it uses the single key saved on the LLM page, as before."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from ..config import ProviderSettings
from ..credentials import llm_key
from .base import TEMPERATURE, ChatProvider, LLMError, Reply, has_pool, pool_secret, schema_hint, strict_schema

log = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1"
_NOT_CHAT = ("embedding", "tts", "whisper", "dall-e", "image", "audio", "realtime", "transcribe", "moderation",
             "davinci", "babbage", "sora", "computer-use")
_EFFORT = {"minimal": "minimal", "low": "low", "medium": "medium", "high": "high"}
_MODES = ("schema", "object", "prompt")
#: Gemini models that reject `minimal` ("not supported for Gemini 3.8 Flash" per Google's docs).
_NO_MINIMAL = ("gemini-3.7", "gemini-3.8")
DETAIL_CHARS = 2_000


def without_lengths(node: Any) -> Any:
    """A schema with every minItems / maxItems removed (Anthropic, Cerebras and others refuse them in strict mode)."""
    if isinstance(node, dict):
        return {k: without_lengths(v) for k, v in node.items() if k not in {"minItems", "maxItems"}}
    if isinstance(node, list):
        return [without_lengths(v) for v in node]
    return node


def error_detail(exc: Any) -> str:
    """The provider's own words about what went wrong, details included: whether a Google 429 is the per-minute or
    the per-day limit is said only in its details, near the end."""
    body = getattr(exc, "body", None)
    if body is None:
        response = getattr(exc, "response", None)
        try:
            return (response.text or "")[:DETAIL_CHARS] if response is not None else ""
        except Exception:
            return ""
    if isinstance(body, list) and body and isinstance(body[0], dict):
        body = body[0]
    try:
        return json.dumps(body, default=str)[:DETAIL_CHARS]
    except (TypeError, ValueError):
        return str(body)[:DETAIL_CHARS]


def _registered_pick(provider: str) -> str:
    """The set-up model of `provider` with the largest daily budget (LLM Integration › Models), or ""."""
    from ..ai.registry import registry
    models = [m for m in registry.all() if m.provider == provider]
    return models[0].id if models else ""


class OpenAIProvider(ChatProvider):
    name = "openai"
    needs_key = False                 # True for hosted gateways that always need a key (OpenRouter)
    extra_headers: dict[str, str] = {}
    cache_key: bool | None = None     # sends prompt_cache_key; None = only to api.openai.com (Mistral sets True)
    array_lengths = True              # keeps minItems / maxItems in strict schemas
    effort_policy = "learn"           # learn: send until refused | gemini3: Gemini 3 models only | never

    def __init__(self) -> None:
        super().__init__()
        self._client: Any = None
        self._client_key: tuple[str, str, float] | None = None
        self._mode: dict[str, str] = {}            # base_url -> structured-output mode that works there
        self._no_effort: set[str] = set()          # models that rejected reasoning_effort
        self._no_temperature: set[str] = set()     # models that take only their own default temperature
        self._keyed: dict[tuple[str, str, float], Any] = {}   # one client per pooled key

    @property
    def base_url(self) -> str:
        return (self.cfg.base_url or OPENAI_URL).rstrip("/")

    @property
    def official(self) -> bool:
        return urlparse(self.base_url).hostname == "api.openai.com"

    @property
    def label(self) -> str:  # type: ignore[override]
        return "OpenAI" if self.official else f"OpenAI-compatible ({urlparse(self.base_url).netloc})"

    def configure(self, api_key: str | None = None) -> bool:
        key = api_key or llm_key(self.name) or ""
        if (self.official or self.needs_key) and not key:
            self._client = None
            if api_key is None and has_pool(self.name):
                if self.status == "unconfigured":
                    self.status = "configured"
                return True
            self.status = "unconfigured"
            return False
        wanted = (self.base_url, key, float(self.cfg.timeout_s or 120))
        if self._client is None or self._client_key != wanted:
            try:
                import openai
            except ImportError:
                self.status, self.last_error = "error", "The 'openai' package is missing — run: pip install openai"
                return False
            # local servers usually need no key, but the SDK insists on one
            self._client = openai.AsyncOpenAI(api_key=key or "not-needed", base_url=wanted[0], timeout=wanted[2],
                                              max_retries=0, default_headers=self.extra_headers or None)
            self._client_key = wanted
            self.status = "configured"
        return True

    @property
    def client(self) -> Any:
        if not self.configure():
            raise LLMError(f"No {self.label} API key configured — add one on the 'LLM Integration' page.")
        secret = pool_secret(self.name) if has_pool(self.name) else None
        if secret:
            return self.client_for(secret)
        if self._client is None:
            raise LLMError(f"No {self.label} API key configured — add one on the 'LLM Integration' page.")
        return self._client

    def client_for(self, secret: str) -> Any:
        """A client sending `secret` (one per pooled key)."""
        import openai
        wanted = (self.base_url, secret, float(self.cfg.timeout_s or 120))
        client = self._keyed.get(wanted)
        if client is None:
            client = openai.AsyncOpenAI(api_key=secret, base_url=wanted[0], timeout=wanted[2], max_retries=0,
                                        default_headers=self.extra_headers or None)
            self._keyed[wanted] = client
        return client

    async def list_model_ids(self, secret: str) -> list[str]:
        """Model ids `secret` can reach — the free key check: models.list is not billed against generation quota."""
        try:
            page = await self.client_for(secret).models.list()
            return sorted({m.id.removeprefix("models/") async for m in page})
        except Exception as exc:
            raise self._error(exc) from exc

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            page = await self.client.models.list()
            ids = [m.id async for m in page]
        except LLMError:
            raise
        except Exception as exc:
            raise self._error(exc) from exc
        names = sorted({i.removeprefix("models/") for i in ids if not any(x in i.lower() for x in _NOT_CHAT)})
        return await self._remember_models([{"name": n, "display": n, "preview": False} for n in names])

    def pick(self, role: str) -> str:
        return (self.explicit_model(role) or _registered_pick(self.name)
                or (self.models[0]["name"] if len(self.models) == 1 else ""))

    def request_extras(self, model: str) -> dict[str, Any]:
        """Extra JSON fields for one chat request (OpenRouter's fallback `models` list)."""
        return {}

    def effort_for(self, model: str, thinking: str) -> str | None:
        """The `reasoning_effort` this model is sent for a thinking level, or None."""
        if self.effort_policy == "never":
            return None
        effort = _EFFORT.get(thinking)
        if self.effort_policy == "gemini3":
            bare = model.removeprefix("models/")
            if not effort or not bare.startswith("gemini-3"):
                return None
            return "low" if effort == "minimal" and bare.startswith(_NO_MINIMAL) else effort
        return effort

    def _error(self, exc: Exception) -> LLMError:
        import openai
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, openai.APIStatusError):
            code = exc.status_code
            retry = exc.response.headers.get("retry-after") if exc.response is not None else None
            try:
                retry_after = float(retry) if retry else None
            except ValueError:
                retry_after = None
            body = getattr(exc, "body", None)
            msg = (body.get("message") if isinstance(body, dict) else None) or getattr(exc, "message", "") or str(exc)
            detail = error_detail(exc)
            if code == 401:
                msg = f"the server rejected the API key ({msg})"
            elif code == 429 and "quota" in msg.lower():
                return LLMError(f"{self.label}: quota exhausted — {msg}", 429, detail=detail)
            return LLMError(f"{self.label} error {code}: {msg}", code,
                            transient=code in (408, 409, 429, 500, 502, 503, 504), retry_after=retry_after,
                            detail=detail)
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return LLMError(f"{self.label} request failed: {exc}", transient=True)
        return LLMError(f"{self.label} request failed: {exc}")

    def build_request(self, model: str, prompt: str, system: str | None, schema: type[BaseModel] | None,
                      thinking: str, mode: str | None) -> dict[str, Any]:
        """The chat-completions arguments of one request (separate so its shaping can be tested without a server)."""
        user = prompt if mode in (None, "schema") else f"{prompt}\n\n{schema_hint(schema)}"
        kwargs: dict[str, Any] = {"model": model, "messages": (
            [{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]}
        if mode == "schema":
            shape = strict_schema(schema)
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": shape if self.array_lengths else without_lengths(shape),
                "strict": True}}
        elif mode == "object":
            kwargs["response_format"] = {"type": "json_object"}
        effort = self.effort_for(model, thinking)
        if effort and model not in self._no_effort:
            kwargs["reasoning_effort"] = effort
        temperature = TEMPERATURE.get()
        if temperature is not None and model not in self._no_temperature:
            kwargs["temperature"] = temperature
        extra = dict(self.request_extras(model))
        if (self.official if self.cache_key is None else self.cache_key) and system:
            # one per system prompt: every call of a feature starts the same way, so they share a prompt cache
            extra["prompt_cache_key"] = hashlib.sha256(system.encode()).hexdigest()[:16]
        if extra:
            kwargs["extra_body"] = extra
        return kwargs

    async def _complete(self, model: str, prompt: str, system: str | None, schema: type[BaseModel] | None,
                        thinking: str) -> Reply:
        mode = self._mode.get(self.base_url, _MODES[0]) if schema is not None else None
        while True:
            kwargs = self.build_request(model, prompt, system, schema, thinking, mode)
            try:
                completion = await self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                err = self._error(exc)
                text = f"{err.message} {err.detail}".lower()
                if err.code == 400 and "temperature" in kwargs and "temperature" in text:
                    self._no_temperature.add(model)         # it only takes its own default: same key, again
                    continue
                if err.code == 400 and "reasoning_effort" in kwargs and ("reasoning" in text or "effort" in text):
                    self._no_effort.add(model)              # not a reasoning model: retry without it
                    continue
                if err.code in (400, 422) and mode in ("schema", "object") and (
                        "response_format" in text or "json" in text or "schema" in text):
                    mode = _MODES[_MODES.index(mode) + 1]   # this server lacks that JSON mode: degrade
                    self._mode[self.base_url] = mode
                    continue
                raise err from exc
            choice = completion.choices[0] if completion.choices else None
            message = getattr(choice, "message", None)
            if getattr(message, "refusal", None):
                raise LLMError(f"{self.label} declined the request: {message.refusal}")
            usage = getattr(completion, "usage", None)
            details = getattr(usage, "prompt_tokens_details", None)
            return Reply(text=getattr(message, "content", None) or "",
                         prompt_tokens=getattr(usage, "prompt_tokens", None),
                         output_tokens=getattr(usage, "completion_tokens", None),
                         cached_tokens=getattr(details, "cached_tokens", None))


openai_provider = OpenAIProvider()


class CompatProvider(OpenAIProvider):
    """One of alpha-harness' OpenAI-compatible providers at its fixed endpoint. It has keys only in the pool; its
    per-role model choices and timeout live in `db.kv("pref:ai_provider:<id>")`, not in settings.json."""

    needs_key = True

    def __init__(self, provider_id: str) -> None:
        from ..ai import providers as catalogue
        self.spec = catalogue.get(provider_id)
        self.name = provider_id
        self._settings = ProviderSettings(base_url=self.spec.base_url)
        self.cache_key = provider_id in catalogue.CACHE_KEYS
        self.array_lengths = provider_id in catalogue.ARRAY_LENGTHS
        self.effort_policy = "gemini3" if provider_id == "google" else "never"
        super().__init__()

    @property
    def cfg(self) -> ProviderSettings:  # type: ignore[override]
        return self._settings

    @property
    def base_url(self) -> str:
        return self.spec.base_url.rstrip("/")

    @property
    def official(self) -> bool:
        return False

    @property
    def label(self) -> str:  # type: ignore[override]
        return self.spec.label

    def configure(self, api_key: str | None = None) -> bool:
        if api_key:
            return super().configure(api_key)
        self._client = None
        if has_pool(self.name):
            if self.status == "unconfigured":
                self.status = "configured"
            return True
        self.status = "unconfigured"
        return False

    async def restore_models(self) -> None:
        await super().restore_models()
        from ..db import db
        saved = await db.kv_get(f"pref:ai_provider:{self.name}")
        if isinstance(saved, dict):
            self._settings = ProviderSettings.model_validate({**saved, "base_url": self.spec.base_url})

    async def save_settings(self) -> None:
        from ..db import db
        await db.kv_set(f"pref:ai_provider:{self.name}", self._settings.model_dump(
            include={"ideation_model", "refine_model", "describe_model", "timeout_s"}))


def compat_providers() -> dict[str, CompatProvider]:
    from ..ai.providers import COMPAT
    return {pid: CompatProvider(pid) for pid in COMPAT}
