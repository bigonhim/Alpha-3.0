"""The models the user has set up, and the limits they gave each one (ported from alpha-harness `llm/registry.py`).

Nothing about limits is built in: which models a provider offers and what its free tier allows change every few
weeks. The user names the model and copies its requests per minute and per day from the provider's limits page.
Requests per day is the limit that ends a session: it comes back only at midnight in the model's own time zone.

A model AlphaGo calls without it being set up here (the auto-picked Gemini flash, say) still rotates keys and still
learns from 429s; it just has no local ceiling, see `implicit`.
"""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert

from ..db import db
from ..db.models_ai import AIModel
from ..llm.base import LLMError
from . import providers

#: Tokens one prompt may spend (on fields, in the assistant) when its model names no ceiling.
DEFAULT_PROMPT_TOKENS = 40_000
#: Limits of a model that is not set up: none locally (the provider's 429s still teach the ledger).
UNLIMITED_RPM = 1_000_000
UNLIMITED_RPD = 1_000_000_000

NOTE = ("Requests per day is the limit that ends a session: it does not come back until the provider's day resets. "
        "Limits apply to each key, so a second account's key doubles them.")


def model_ref(provider: str, model_id: str) -> str:
    """How a request names one model: ids repeat across providers, so both. Provider ids hold no colon; model ids can
    (`…:free`), so a ref splits on the first one."""
    return f"{provider}:{model_id}"


def split_ref(ref: str) -> tuple[str, str]:
    provider, _, model = (ref or "").partition(":")
    return providers.ALIASES.get(provider, provider), model


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    provider: str
    rpm: int
    rpd: int
    reset_timezone: str
    max_prompt_tokens: int | None = None
    registered: bool = True

    @property
    def ref(self) -> str:
        return model_ref(self.provider, self.id)

    @property
    def prompt_tokens(self) -> int:
        return self.max_prompt_tokens or DEFAULT_PROMPT_TOKENS

    @property
    def unlimited(self) -> bool:
        return not self.registered


def implicit(provider: str, model_id: str) -> ModelInfo:
    """A model that is not set up: no local limits, counted on its provider's usual clock."""
    return ModelInfo(id=model_id, provider=provider, rpm=UNLIMITED_RPM, rpd=UNLIMITED_RPD,
                     reset_timezone=providers.get(provider).reset_timezone, registered=False)


def _info(row: AIModel) -> ModelInfo:
    return ModelInfo(id=row.model, provider=row.provider, rpm=row.rpm, rpd=row.rpd,
                     reset_timezone=row.reset_timezone, max_prompt_tokens=row.max_prompt_tokens)


class ModelRegistry:
    """The set-up models, held in memory (calls read it synchronously) and written through to the database."""

    def __init__(self) -> None:
        self._models: dict[str, ModelInfo] = {}

    async def load(self) -> None:
        async with db.session() as s:
            rows = (await s.scalars(select(AIModel))).all()
        self._models = {info.ref: info for info in map(_info, rows)}

    def clear(self) -> None:
        self._models = {}

    def get(self, ref: str) -> ModelInfo | None:
        """A model by its ref; a bare id is found when only one provider has it."""
        provider, model = split_ref(ref)
        if found := self._models.get(model_ref(provider, model)):
            return found
        named = [m for m in self._models.values() if m.id == ref]
        return named[0] if len(named) == 1 else None

    def info(self, provider: str, model_id: str) -> ModelInfo:
        """The set-up model, or its implicit stand-in."""
        return self._models.get(model_ref(provider, model_id)) or implicit(provider, model_id)

    def all(self) -> list[ModelInfo]:
        """Richest daily budget first, because that is what runs out."""
        return sorted(self._models.values(), key=lambda m: (-m.rpd, -m.rpm, m.ref))

    async def set(self, provider: str, model_id: str, rpm: int, rpd: int, reset_timezone: str,
                  max_prompt_tokens: int | None = None) -> ModelInfo:
        """Set a model up, or change the limits of one already set up."""
        model_id = (model_id or "").strip()
        provider = providers.ALIASES.get(provider, provider)
        try:
            ZoneInfo(reset_timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise LLMError(f"{reset_timezone!r} is not a time zone this machine knows.") from None
        spec = providers.get(provider)
        if spec.kind == "unknown":
            raise LLMError(f"{spec.label} cannot take a model.")
        if spec.kind == "local":
            raise LLMError("Ollama runs on this PC with no request limits, so its models need no setting up.")
        if not model_id:
            raise LLMError("Name the model: pick one from the list or type its ID.")
        if rpm < 1 or rpd < 1:
            raise LLMError("Both limits have to be at least one request.")
        if max_prompt_tokens is not None and max_prompt_tokens < 1:
            raise LLMError("Max prompt tokens has to be at least one, or left empty.")
        limits = {"rpm": int(rpm), "rpd": int(rpd), "reset_timezone": reset_timezone,
                  "max_prompt_tokens": max_prompt_tokens}
        statement = (insert(AIModel).values(provider=provider, model=model_id, **limits)
                     .on_conflict_do_update(index_elements=["provider", "model"], set_=limits))
        async with db.write() as s:
            await s.execute(statement)
        info = ModelInfo(id=model_id, provider=provider, rpm=int(rpm), rpd=int(rpd), reset_timezone=reset_timezone,
                         max_prompt_tokens=max_prompt_tokens)
        self._models[info.ref] = info
        return info

    async def remove(self, provider: str, model_id: str) -> None:
        ref = model_ref(provider, model_id)
        if ref not in self._models:
            raise LLMError(f"{model_id} is not set up for {providers.get(provider).label}.")
        async with db.write() as s:
            await s.execute(delete(AIModel).where(AIModel.provider == provider, AIModel.model == model_id))
        del self._models[ref]


registry = ModelRegistry()
