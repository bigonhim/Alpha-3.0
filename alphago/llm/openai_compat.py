"""OpenAI, or any server speaking the OpenAI chat API (OpenRouter, Groq, DeepSeek, LM Studio, vLLM, ...).

Structured output degrades gracefully: strict `json_schema` → `json_object` → schema in the prompt; the first mode
a server accepts is remembered. `reasoning_effort` is sent until a model rejects it."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from ..credentials import llm_key
from .base import ChatProvider, LLMError, Reply, schema_hint, strict_schema

log = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1"
_NOT_CHAT = ("embedding", "tts", "whisper", "dall-e", "image", "audio", "realtime", "transcribe", "moderation",
             "davinci", "babbage", "sora", "computer-use")
_EFFORT = {"minimal": "minimal", "low": "low", "medium": "medium", "high": "high"}
_MODES = ("schema", "object", "prompt")


class OpenAIProvider(ChatProvider):
    name = "openai"

    def __init__(self) -> None:
        super().__init__()
        self._client: Any = None
        self._client_key: tuple[str, str, float] | None = None
        self._mode: dict[str, str] = {}            # base_url -> structured-output mode that works there
        self._no_effort: set[str] = set()          # models that rejected reasoning_effort

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
        key = api_key or llm_key("openai") or ""
        if self.official and not key:
            self._client, self.status = None, "unconfigured"
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
                                              max_retries=0)
            self._client_key = wanted
            self.status = "configured"
        return True

    @property
    def client(self) -> Any:
        if not self.configure():
            raise LLMError("No OpenAI API key configured — add one on the 'LLM Integration' page.")
        return self._client

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            page = await self.client.models.list()
            ids = [m.id async for m in page]
        except LLMError:
            raise
        except Exception as exc:
            raise self._error(exc) from exc
        names = sorted({i for i in ids if not any(x in i.lower() for x in _NOT_CHAT)})
        return await self._remember_models([{"name": n, "display": n, "preview": False} for n in names])

    def pick(self, role: str) -> str:
        return self.explicit_model(role) or (self.models[0]["name"] if len(self.models) == 1 else "")

    def _error(self, exc: Exception) -> LLMError:
        import openai
        if isinstance(exc, openai.APIStatusError):
            code = exc.status_code
            retry = exc.response.headers.get("retry-after") if exc.response is not None else None
            try:
                retry_after = float(retry) if retry else None
            except ValueError:
                retry_after = None
            body = getattr(exc, "body", None)
            msg = (body.get("message") if isinstance(body, dict) else None) or getattr(exc, "message", "") or str(exc)
            if code == 401:
                msg = f"the server rejected the API key ({msg})"
            elif code == 429 and "quota" in msg.lower():
                return LLMError(f"{self.label}: quota exhausted — {msg}", 429)
            return LLMError(f"{self.label} error {code}: {msg}", code,
                            transient=code in (408, 409, 429, 500, 502, 503, 504), retry_after=retry_after)
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return LLMError(f"{self.label} request failed: {exc}", transient=True)
        return LLMError(f"{self.label} request failed: {exc}")

    async def _complete(self, model: str, prompt: str, system: str | None, schema: type[BaseModel] | None,
                        thinking: str) -> Reply:
        mode = self._mode.get(self.base_url, _MODES[0]) if schema is not None else None
        while True:
            user = prompt if mode in (None, "schema") else f"{prompt}\n\n{schema_hint(schema)}"
            kwargs: dict[str, Any] = {"model": model, "messages": (
                [{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]}
            if mode == "schema":
                kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": schema.__name__, "schema": strict_schema(schema), "strict": True}}
            elif mode == "object":
                kwargs["response_format"] = {"type": "json_object"}
            effort = _EFFORT.get(thinking)
            if effort and model not in self._no_effort:
                kwargs["reasoning_effort"] = effort
            try:
                completion = await self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                err = self._error(exc)
                text = err.message.lower()
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
