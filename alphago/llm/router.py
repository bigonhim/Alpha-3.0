"""The app's single LLM entry point. Every call goes to the one provider chosen on the LLM Integration page
(`settings.llm.provider`); the public surface is the one the Gemini client always had."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from ..config import settings
from .base import calls_today
from .claude import claude
from .gemini import gemini
from .ollama import ollama
from .openai_compat import openai_provider

T = TypeVar("T", bound=BaseModel)

PROVIDERS: dict[str, Any] = {"gemini": gemini, "claude": claude, "openai": openai_provider, "ollama": ollama}
PROVIDER_LABELS = {"gemini": "Gemini (Google AI Studio)", "claude": "Claude (Anthropic)",
                   "openai": "OpenAI / compatible", "ollama": "Ollama (local)"}


class LLMRouter:
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

    def configure(self, api_key: str | None = None) -> bool:
        """True when the active provider is set up (key / server address); call sites use it as 'LLM available?'."""
        return self.active.configure(api_key)

    def pick(self, role: str) -> str:
        return self.active.pick(role)

    async def generate_json(self, prompt: str, schema: type[T], **kwargs: Any) -> T:
        return await self.active.generate_json(prompt, schema, **kwargs)

    async def generate_text(self, prompt: str, **kwargs: Any) -> str:
        return await self.active.generate_text(prompt, **kwargs)

    async def calls_today(self) -> int:
        return await calls_today()

    async def restore_models(self) -> None:
        for provider in PROVIDERS.values():
            await provider.restore_models()

    async def test(self) -> str:
        return await self.active.test()


llm = LLMRouter()
