"""LLM integration: Gemini, Claude, OpenAI (or compatible) and local Ollama behind one router (`llm`)."""

from .base import LLMError
from .gemini import gemini
from .router import PROVIDER_LABELS, PROVIDERS, llm

__all__ = ["PROVIDERS", "PROVIDER_LABELS", "LLMError", "gemini", "llm"]
