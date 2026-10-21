"""The app's single LLM entry point. Every call goes to the primary provider chosen on the LLM Integration page
(`settings.llm.provider`); the public surface is the one the Gemini client always had.

Failover: when the primary fails after its own retries (rate-limited, overloaded, timed out, model unavailable,
network), the call moves to the next provider in `settings.llm.fallbacks` that is set up, instead of falling back to
templates. A provider that failed is tried last for the next FAILOVER_COOLDOWN seconds, then first again. With no
fallbacks configured (the default) behaviour is exactly as before. The daily budget stays shared.
"""

from __future__ import annotations

import time
from typing import Any, TypeVar

from pydantic import BaseModel

from ..config import settings
from ..events import bus
from .base import MODEL_OVERRIDE, LLMError, calls_today
from .claude import claude
from .gemini import gemini
from .ollama import ollama
from .openai_compat import openai_provider

T = TypeVar("T", bound=BaseModel)

PROVIDERS: dict[str, Any] = {"gemini": gemini, "claude": claude, "openai": openai_provider, "ollama": ollama}
PROVIDER_LABELS = {"gemini": "Gemini (Google AI Studio)", "claude": "Claude (Anthropic)",
                   "openai": "OpenAI / compatible", "ollama": "Ollama (local)"}
FAILOVER_COOLDOWN = 600.0
FAILOVER_CODES = (None, 404, 408, 429, 500, 502, 503, 504, 529)


def failover_worthy(exc: LLMError) -> bool:
    """Provider trouble (worth another provider), not a problem another provider would share."""
    msg = (exc.message or "").lower()
    if "budget" in msg or "malformed json" in msg:
        return False
    return bool(exc.transient) or exc.code in FAILOVER_CODES or "timed out" in msg or "timeout" in msg


class LLMRouter:
    def __init__(self) -> None:
        self._down_until: dict[str, float] = {}
        self.last_failover: dict[str, Any] | None = None

    @property
    def active(self) -> Any:
        return PROVIDERS.get(settings.llm.provider) or gemini

    @property
    def name(self) -> str:
        return self.active.name

    @property
    def label(self) -> str:
        return self.active.label

    @property
    def local(self) -> bool:
        return self.active.local

    @property
    def models(self) -> list[dict[str, Any]]:
        return self.active.models

    @property
    def status(self) -> str:
        return self.active.status

    @property
    def last_error(self) -> str | None:
        return self.active.last_error

    @property
    def rpm(self) -> float | None:
        return self.active.rpm

    def chain(self) -> list[Any]:
        """Primary first, then the fallbacks in order; a provider in its cooldown after a failure goes last."""
        names = [settings.llm.provider] + [n for n in settings.llm.fallbacks if n != settings.llm.provider]
        providers = [PROVIDERS[n] for n in dict.fromkeys(names) if n in PROVIDERS]
        now = time.monotonic()
        return sorted(providers, key=lambda p: self._down_until.get(p.name, 0.0) > now)

    def configure(self, api_key: str | None = None) -> bool:
        """True when the primary (or, without a key argument, any fallback) is set up — "is an LLM available?"."""
        if self.active.configure(api_key):
            return True
        return api_key is None and any(p.configure() for p in self.chain() if p is not self.active)

    def pick(self, role: str) -> str:
        return self.active.pick(role)

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        chain = [p for p in self.chain() if p.configure()] or [self.active]
        for i, provider in enumerate(chain):
            try:
                result = await getattr(provider, method)(*args, **kwargs)
                self._down_until.pop(provider.name, None)
                return result
            except LLMError as exc:
                if i == len(chain) - 1 or not failover_worthy(exc):
                    raise
                self._down_until[provider.name] = time.monotonic() + FAILOVER_COOLDOWN
                nxt = chain[i + 1]
                self.last_failover = {"from": provider.name, "to": nxt.name, "error": exc.message[:300],
                                      "at": time.time()}
                bus.publish("llm", f"{provider.label} failed ({exc.message[:120]}) — switching to {nxt.label}; "
                                   f"{provider.label} is retried after {FAILOVER_COOLDOWN / 60:.0f} minutes.",
                            "warning", kind="llm_failover", provider=provider.name, to=nxt.name)
        raise LLMError("No LLM provider available")      # unreachable: the loop returns or raises

    async def generate_json(self, prompt: str, schema: type[T], **kwargs: Any) -> T:
        return await self._call("generate_json", prompt, schema, **kwargs)

    async def generate_text(self, prompt: str, **kwargs: Any) -> str:
        return await self._call("generate_text", prompt, **kwargs)

    async def generate_json_with(self, provider_name: str, model: str, prompt: str, schema: type[T],
                                 **kwargs: Any) -> T:
        """One call on a given provider and model (the model tournament). If that provider is in trouble, the
        normal chain answers instead, with each provider's own model."""
        provider = PROVIDERS.get(provider_name)
        if provider is None or not provider.configure():
            return await self.generate_json(prompt, schema, **kwargs)
        token = MODEL_OVERRIDE.set(model or None)
        try:
            return await provider.generate_json(prompt, schema, **kwargs)
        except LLMError as exc:
            if not failover_worthy(exc):
                raise
        finally:
            MODEL_OVERRIDE.reset(token)
        return await self.generate_json(prompt, schema, **kwargs)

    async def calls_today(self) -> int:
        return await calls_today()

    async def restore_models(self) -> None:
        for provider in PROVIDERS.values():
            await provider.restore_models()

    async def test(self) -> str:
        return await self.active.test()


llm = LLMRouter()
