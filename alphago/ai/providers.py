"""Every AI provider AlphaGo can call, and how to get a key for each (ported from alpha-harness `llm/providers.py`).

AlphaGo's own providers keep their native SDKs (Gemini, Claude) or their special handling (OpenRouter's ranked
free models, local Ollama). The alpha-harness providers that only speak the OpenAI chat-completions protocol are
added beside them: Google's OpenAI-compatible endpoint, Groq, Cerebras, NVIDIA NIM, Mistral and Hugging Face.

**A key only ever answers for its own provider**, so rotation filters on the key's family before it looks at budget.
The one shared family is Google AI Studio: the native Gemini provider and Google's OpenAI-compatible endpoint take
the same key, so both draw on the same pool (and on the same per-model counters, which is what Google counts).

**Paid providers** (OpenAI, Anthropic) bill the user's own account. Their keys are refused without a daily request
cap, and that cap can be moved but never removed: a free tier stops by itself, a paid account only when told to.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Provider:
    id: str
    label: str
    base_url: str               # chat-completions endpoint ("" for a native SDK provider)
    onboarding_url: str         # where to get a key
    limits_url: str             # where the provider shows its rate limits
    key_hint: str               # what a key looks like
    paid: bool = False
    tier_note: str = ""
    kind: str = "compat"        # native | compat | local
    key_family: str = ""        # whose keys answer for it ("" = its own id)
    reset_timezone: str = "UTC"  # whose midnight starts a new day when a model names none
    needs_key: bool = True

    @property
    def family(self) -> str:
        return self.key_family or self.id


PROVIDERS: dict[str, Provider] = {
    "gemini": Provider("gemini", "Gemini (Google AI Studio, native)", "", "https://aistudio.google.com/apikey",
                       "https://aistudio.google.com/rate-limit", "AIza…", kind="native",
                       reset_timezone="America/Los_Angeles"),
    "google": Provider("google", "Google AI Studio (OpenAI-compatible)",
                       "https://generativelanguage.googleapis.com/v1beta/openai", "https://aistudio.google.com/apikey",
                       "https://aistudio.google.com/rate-limit", "AIza…", key_family="gemini",
                       reset_timezone="America/Los_Angeles"),
    "groq": Provider("groq", "Groq", "https://api.groq.com/openai/v1", "https://console.groq.com/keys",
                     "https://console.groq.com/settings/limits", "gsk_…"),
    "cerebras": Provider("cerebras", "Cerebras", "https://api.cerebras.ai/v1", "https://cloud.cerebras.ai/",
                         "https://inference-docs.cerebras.ai/support/rate-limits", "csk-…"),
    "openrouter": Provider("openrouter", "OpenRouter (free models)", "https://openrouter.ai/api/v1",
                           "https://openrouter.ai/keys", "https://openrouter.ai/docs/api-reference/limits", "sk-or-…"),
    "nvidia": Provider("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", "https://build.nvidia.com/",
                       "https://build.nvidia.com/faq", "nvapi-…"),
    "mistral": Provider("mistral", "Mistral", "https://api.mistral.ai/v1", "https://console.mistral.ai/api-keys/",
                        "https://admin.mistral.ai/plateforme/limits", "…"),
    "huggingface": Provider("huggingface", "Hugging Face", "https://router.huggingface.co/v1",
                            "https://huggingface.co/settings/tokens",
                            "https://huggingface.co/docs/inference-providers/pricing", "hf_…"),
    # --- bring your own, and your own bill
    "openai": Provider("openai", "OpenAI / compatible", "https://api.openai.com/v1",
                       "https://platform.openai.com/api-keys", "https://platform.openai.com/settings/organization/limits",
                       "sk-…", paid=True, tier_note="Your own OpenAI account — billed to you, not free."),
    "claude": Provider("claude", "Claude (Anthropic)", "", "https://platform.claude.com/settings/keys",
                       "https://platform.claude.com/settings/limits", "sk-ant-…", paid=True, kind="native",
                       tier_note="Your own Anthropic account — billed to you, not free."),
    # --- this PC
    "ollama": Provider("ollama", "Ollama (local)", "", "https://ollama.com", "", "", kind="local", needs_key=False),
}

#: Alpha-harness names for the same providers, so a ref typed from its docs still resolves.
ALIASES = {"anthropic": "claude"}

#: The OpenAI-compatible providers added from alpha-harness (each one is a provider object in the router).
COMPAT = ("google", "groq", "cerebras", "nvidia", "mistral", "huggingface")

#: Providers that take a `prompt_cache_key` (Mistral: raises the chance of a cache hit; OpenAI: steers routing).
CACHE_KEYS = frozenset({"mistral", "openai"})

#: Providers whose strict schemas take an array's length; Anthropic and Cerebras refuse minItems/maxItems.
ARRAY_LENGTHS = frozenset({"google", "openai"})

PAID_NOTE = ("These bill your own account. Nothing here is needed — every provider above is free — but a key you "
             "already pay for is not held to a free tier's daily limit. You set a daily request cap when you add "
             "one, and AlphaGo stops at it.")


def get(provider_id: str | None) -> Provider:
    """One provider; an unknown id gets a stub that names it, never another provider (which would be sent the
    wrong key)."""
    pid = ALIASES.get(provider_id or "", provider_id or "gemini")
    return PROVIDERS.get(pid) or Provider(pid, f"{pid} (unknown)", "", "", "", "", kind="unknown")


def family(provider_id: str) -> str:
    """Whose keys answer for `provider_id`."""
    return get(provider_id).family


def key_providers() -> list[Provider]:
    """Key families a key can be added for, free first and paid last (the order is the argument)."""
    return sorted((p for p in PROVIDERS.values() if p.needs_key and not p.key_family), key=lambda p: p.paid)


def users_of(family_id: str) -> list[str]:
    """Provider ids that draw on keys of `family_id`."""
    return [p.id for p in PROVIDERS.values() if p.family == family_id and p.needs_key]
