"""Anthropic Claude via the official `anthropic` SDK (Messages API).

- Structured output: `beta.messages.parse(output_format=Schema)` (forced tool use is rejected by current models).
- Depth: `output_config.effort`; the `thinking` parameter is omitted (current models think adaptively by default).
- The knowledge pack is sent as a cached system block, so repeated calls read it from the prompt cache.
- Refusals: `stop_reason == "refusal"` is checked before reading output. Server-side `fallbacks: "default"` re-runs a
  declined request on Anthropic's recommended fallback model (not available for Haiku 5.5).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from ..credentials import llm_key
from .base import ChatProvider, LLMError, Reply

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_EFFORT = {"minimal": "low", "off": "low", "low": "low", "medium": "medium", "high": "high"}
MAX_TOKENS = 16_000


class ClaudeProvider(ChatProvider):
    name = "claude"
    label = "Claude"

    def __init__(self) -> None:
        super().__init__()
        self._client: Any = None
        self._key: str | None = None
        self._no_effort: set[str] = set()       # models that rejected output_config.effort
        self._no_fallback: set[str] = set()     # models that rejected the fallbacks parameter

    def configure(self, api_key: str | None = None) -> bool:
        key = api_key or llm_key("claude")
        if not key:
            self._client, self.status = None, "unconfigured"
            return False
        if key != self._key or self._client is None:
            try:
                import anthropic
            except ImportError:
                self.status, self.last_error = "error", "The 'anthropic' package is missing — run: pip install anthropic"
                return False
            kwargs: dict[str, Any] = {"api_key": key, "max_retries": 0}     # the app's own loop retries and paces
            if self.cfg.timeout_s:
                kwargs["timeout"] = float(self.cfg.timeout_s)
            self._client = anthropic.AsyncAnthropic(**kwargs)
            self._key = key
            self.status = "configured"
        return True

    @property
    def client(self) -> Any:
        if not self.configure():
            raise LLMError("No Anthropic API key configured — add one on the 'LLM Integration' page.")
        return self._client

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            out = [{"name": m.id, "display": getattr(m, "display_name", None) or m.id,
                    "input_limit": getattr(m, "max_input_tokens", None), "preview": False}
                   async for m in self.client.models.list()]
        except LLMError:
            raise
        except Exception as exc:
            raise self._error(exc) from exc
        return await self._remember_models(out)

    def pick(self, role: str) -> str:
        return self.explicit_model(role) or DEFAULT_MODEL

    def _error(self, exc: Exception) -> LLMError:
        import anthropic
        if isinstance(exc, anthropic.APIStatusError):
            code = exc.status_code
            retry = exc.response.headers.get("retry-after") if exc.response is not None else None
            try:
                retry_after = float(retry) if retry else None
            except ValueError:
                retry_after = None
            body = getattr(exc, "body", None)
            err = body.get("error") if isinstance(body, dict) else None
            msg = (err.get("message") if isinstance(err, dict) else None) or getattr(exc, "message", "") or str(exc)
            if isinstance(exc, anthropic.AuthenticationError):
                msg = f"the API key was rejected ({msg})"
            return LLMError(f"Claude error {code}: {msg}", code,
                            transient=code in (408, 409, 429, 500, 502, 503, 504, 529), retry_after=retry_after)
        if isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
            return LLMError(f"Claude request failed: {exc}", transient=True)
        return LLMError(f"Claude request failed: {exc}")

    async def _complete(self, model: str, prompt: str, system: str | None, schema: type[BaseModel] | None,
                        thinking: str) -> Reply:
        while True:
            kwargs: dict[str, Any] = {"model": model, "max_tokens": MAX_TOKENS,
                                      "messages": [{"role": "user", "content": prompt}]}
            if system:      # stable knowledge pack first, cached across calls
                kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if model not in self._no_effort:
                kwargs["output_config"] = {"effort": _EFFORT.get(thinking, "medium")}
            if model in FALLBACK_MODELS and model not in self._no_fallback:
                kwargs["betas"] = [FALLBACK_BETA]
                kwargs["fallbacks"] = "default"
            if schema is not None:
                kwargs["output_format"] = schema
            messages = self.client.beta.messages
            try:
                msg = await (messages.parse(**kwargs) if schema is not None else messages.create(**kwargs))
            except ValidationError as exc:      # structured outputs should prevent this; let the repair round try
                log.warning("Claude reply did not validate: %s", exc.errors()[:2])
                return Reply(text="")
            except Exception as exc:
                err = self._error(exc)
                text = err.message.lower()
                if err.code == 400 and "output_config" in kwargs and "effort" in text:
                    self._no_effort.add(model)
                    continue
                if err.code == 400 and "fallbacks" in kwargs and "fallback" in text:
                    self._no_fallback.add(model)
                    continue
                raise err from exc
            if msg.stop_reason == "refusal":
                details = getattr(msg, "stop_details", None)
                category = getattr(details, "category", None)
                raise LLMError("Claude declined this request" + (f" ({category})" if category else "") + ".")
            if msg.stop_reason == "max_tokens":
                raise LLMError(f"Claude's reply hit the {MAX_TOKENS:,}-token limit before finishing.")
            usage = getattr(msg, "usage", None)
            text = "".join(getattr(b, "text", "") for b in (msg.content or []) if getattr(b, "type", "") == "text")
            return Reply(text=text, parsed=getattr(msg, "parsed_output", None) if schema is not None else None,
                         prompt_tokens=getattr(usage, "input_tokens", None),
                         output_tokens=getattr(usage, "output_tokens", None),
                         cached_tokens=getattr(usage, "cache_read_input_tokens", None))


claude = ClaudeProvider()
