"""Several API keys per provider, rotated by remaining budget (ported from alpha-harness `llm/keys.py`).

Free-tier quota is per account, so keys from several accounts add their budgets together. Rotation picks by
*headroom* rather than round-robin: round-robin spreads load evenly, which is exactly wrong when one key is nearly
spent and another is untouched.

The secret of every key stays in Windows Credential Manager (`credentials.pool_key`); the `ai_keys` table and the UI
only ever see a hint ("first6…last4") and a sha256 fingerprint. Key rows are mirrored in memory, because whether a
provider is set up is asked synchronously all over the app (`provider.configure()`).
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select

from .. import credentials
from ..db import db
from ..db.models import utcnow
from ..db.models_ai import AIKey
from ..llm.base import LLMError
from . import providers
from .budget import Headroom, Ledger
from .registry import ModelInfo, ModelRegistry, registry


class NoKeysError(LLMError):
    def __init__(self, provider: str) -> None:
        spec = providers.get(provider)
        tail = ("That one bills your own account; the free providers above it do not." if spec.paid
                else "Most of the providers listed there are free and need no card.")
        super().__init__(f"No {spec.label} key has been added yet. Open LLM Integration › Keys. {tail}")


class BudgetExhaustedError(LLMError):
    """Every key is out of budget for this model. A 429 to the router (so it can fail over to another provider);
    worth waiting for only when the block is per-minute."""

    def __init__(self, model: ModelInfo, states: list[Headroom], reset_in: float) -> None:
        daily = [s for s in states if s.blocked_by == "requests_per_day"]
        self.daily = bool(daily) and len(daily) == len(states)
        wait = min((s.retry_after for s in states), default=60.0)
        if self.daily:
            message = (f"Every key has used its daily allowance of {model.id}. It resets in about "
                       f"{reset_in / 3600:.0f} hours (midnight, {model.reset_timezone}). Switch to a model with a "
                       "larger daily budget, or add another key.")
        else:
            message = f"Every key is at its per-minute limit for {model.id}. Try again in about {wait:.0f} seconds."
        super().__init__(message, 429, transient=not self.daily and wait <= 120, retry_after=wait)
        self.model = model.id
        self.states = states


def fingerprint(key: str) -> str:
    """Stable identity of a key, so the same one is not added twice: a hash, not a second copy of the secret."""
    return hashlib.sha256(key.strip().encode()).hexdigest()


def hint(key: str) -> str:
    """Enough of a key to recognise it, not enough to use it."""
    clean = key.strip()
    if len(clean) <= 10:
        return "…" + clean[-3:]
    return f"{clean[:6]}…{clean[-4:]}"


@dataclass(frozen=True, slots=True)
class KeyRow:
    id: int
    provider: str
    label: str
    hint: str
    fingerprint: str
    enabled: bool
    daily_limit: int | None
    last_ok_at: datetime | None
    last_error: str | None
    created_at: datetime | None

    @classmethod
    def of(cls, r: AIKey) -> KeyRow:
        return cls(r.id, r.provider, r.label, r.hint, r.fingerprint, bool(r.enabled), r.daily_limit, r.last_ok_at,
                   r.last_error, r.created_at)


class KeyStore:
    """Holds the keys and decides which one to use next."""

    def __init__(self, ledger: Ledger | None = None, models: ModelRegistry | None = None) -> None:
        self.ledger = ledger or Ledger()
        self.registry = models or registry
        self._rows: dict[int, KeyRow] = {}
        self._lock: asyncio.Lock | None = None
        self._lock_loop: Any = None

    # -- the in-memory mirror ---------------------------------------------

    async def load(self) -> None:
        async with db.session() as s:
            rows = (await s.scalars(select(AIKey).order_by(AIKey.id))).all()
        self._rows = {r.id: KeyRow.of(r) for r in rows}

    def clear(self) -> None:
        self._rows = {}

    def rows(self) -> list[KeyRow]:
        return sorted(self._rows.values(), key=lambda r: r.id)

    def usable(self, provider: str) -> list[KeyRow]:
        """Enabled keys whose secret is present, that can answer for `provider` (a key family or a provider id)."""
        fam = providers.family(provider)
        return [r for r in self.rows() if r.enabled and r.provider == fam]

    def has_keys(self, provider: str) -> bool:
        return bool(self.usable(provider))

    def first_secret(self, provider: str) -> str | None:
        """A working secret of `provider`'s pool, for calls made outside a rotation (listing models)."""
        rows = self.usable(provider)
        rows.sort(key=lambda r: (r.last_error is not None, r.id))
        for r in rows:
            secret = credentials.pool_key(r.id)
            if secret:
                return secret
        return None

    def get(self, key_id: int) -> KeyRow | None:
        return self._rows.get(key_id)

    def secret(self, key_id: int) -> str:
        value = credentials.pool_key(key_id)
        if not value:
            raise LLMError(f"Key {key_id} is missing from Windows Credential Manager — remove it and add it again.")
        return value

    # -- storage ----------------------------------------------------------

    async def add(self, key: str, label: str | None = None, provider: str = "gemini",
                  daily_limit: int | None = None) -> KeyRow:
        clean = (key or "").strip()
        provider = providers.family(providers.ALIASES.get(provider, provider))
        spec = providers.get(provider)
        if not clean:
            raise LLMError("That key is empty.")
        if spec.kind in ("unknown", "local") or not spec.needs_key:
            raise LLMError(f"{spec.label} cannot take a key.")
        # A free tier stops on its own; a paid account stops when told to. Refusing the key outright is the only point
        # at which the app can be sure it was told — afterwards the key is stored, and anything that reads it can
        # spend money.
        if spec.paid and not daily_limit:
            raise LLMError(f"Keys for {spec.label} bill your own account, so they need a daily request cap. Set one "
                           "and add the key again.")
        if daily_limit is not None and daily_limit < 1:
            raise LLMError("A daily cap has to be at least one request.")
        print_ = fingerprint(clean)
        async with db.write() as s:
            existing = await s.scalar(select(AIKey).where(AIKey.fingerprint == print_))
            if existing is not None:
                raise LLMError(f"That key is already stored as {existing.label!r}. Adding it twice would not increase "
                               "your quota — quota is per account.")
            count = await s.scalar(select(func.count(AIKey.id)).where(AIKey.provider == provider)) or 0
            row = AIKey(label=(label or "").strip()[:64] or f"{spec.label.split(' (')[0]} key {count + 1}",
                        provider=provider, hint=hint(clean), fingerprint=print_, daily_limit=daily_limit,
                        enabled=True, created_at=utcnow())
            s.add(row)
            await s.flush()
            credentials.set_pool_key(row.id, clean)
            if credentials.pool_key(row.id) != clean:
                raise LLMError("Windows Credential Manager did not keep the key, so it was not added.")
            snapshot = KeyRow.of(row)
        self._rows[snapshot.id] = snapshot
        return snapshot

    async def set_enabled(self, key_id: int, enabled: bool, cap: int | None = None, *, clear: bool = False,
                          label: str | None = None) -> KeyRow:
        """Turn a key on or off, and optionally move its daily cap. `clear` removes the cap — never allowed for a
        paid provider, where the cap is the only thing between the key and an open-ended bill."""
        if cap is not None and cap < 1:
            raise LLMError("A daily cap has to be at least one request.")
        async with db.write() as s:
            row = await s.get(AIKey, key_id)
            if row is None:
                raise LLMError(f"No key {key_id}.")
            spec = providers.get(row.provider)
            if clear and spec.paid:
                raise LLMError(f"Keys for {spec.label} bill your own account, so the daily cap cannot be removed. "
                               "Raise it instead.")
            row.enabled = bool(enabled)
            if clear:
                row.daily_limit = None
            elif cap is not None:
                row.daily_limit = int(cap)
            if label is not None and label.strip():
                row.label = label.strip()[:64]
            snapshot = KeyRow.of(row)
        self._rows[key_id] = snapshot
        return snapshot

    async def remove(self, key_id: int) -> None:
        async with db.write() as s:
            row = await s.get(AIKey, key_id)
            if row is None:
                raise LLMError(f"No key {key_id}.")
            await s.execute(delete(AIKey).where(AIKey.id == key_id))
        credentials.set_pool_key(key_id, None)
        self._rows.pop(key_id, None)
        self.ledger.forget(key_id)

    async def mark(self, key_id: int, *, error: str | None = None) -> None:
        """Record the outcome of a call, so a dead key is visible rather than mysterious."""
        row = self._rows.get(key_id)
        if row is None:
            return
        now = utcnow()
        async with db.write() as s:
            stored = await s.get(AIKey, key_id)
            if stored is None:
                return
            if error is None:
                stored.last_ok_at, stored.last_error = now, None
            else:
                stored.last_error = error[:500]
        self._rows[key_id] = (replace(row, last_ok_at=now, last_error=None) if error is None
                              else replace(row, last_error=error[:500]))

    # -- rotation ---------------------------------------------------------

    def _choosing(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    async def choose(self, model: ModelInfo, *, skip: set[int]) -> int:
        """Take a request slot on the key with the most daily budget left for this model, other than those this
        request already tried. A key whose last call failed goes after the rest; ties go to the lower id, so the
        choice is reproducible. Reading the budget and taking the slot happen under one lock, or two callers could
        both take the last slot."""
        async with self._choosing():
            rows = self.usable(model.provider)
            if not rows:
                raise NoKeysError(model.provider)
            states = [await self.ledger.headroom(r.id, model, cap=r.daily_limit) for r in rows]
            usable = [s for s in states if s.available and s.key_id not in skip]
            if not usable:
                raise BudgetExhaustedError(model, states, self.ledger.model_reset(model))
            failing = {r.id for r in rows if r.last_error}
            best = max(usable, key=lambda s: (s.key_id not in failing, s.daily_remaining, -s.key_id))
            self.ledger.sent(best.key_id, model)
            return best.key_id

    async def status(self) -> dict[str, Any]:
        """Keys, their health, and what budget remains — the whole picture for the Keys and Budget screens."""
        rows = self.rows()
        models = self.registry.all()
        fams = {r.id: r.provider for r in rows}

        def today_of(fam: str, model_id: str) -> str:
            # a set-up model counts on its own clock; any other (and the all-models row) on its provider's usual one
            known = [m for m in models if providers.family(m.provider) == fam and m.id == model_id]
            return self.ledger.model_day(known[0] if known else self.registry.info(fam, model_id))

        usage = await self.ledger.usage(fams, today_of)
        by_key: dict[int, list[dict[str, Any]]] = {}
        for entry in usage:
            by_key.setdefault(entry["key_id"], []).append(entry)
        budget: list[dict[str, Any]] = []
        left: dict[int, dict[str, int]] = {}
        for model in models:
            usable = self.usable(model.provider)
            if not usable:
                continue
            remaining = allowed = minute = 0
            blocked: list[Headroom] = []
            for row in usable:
                state = await self.ledger.headroom(row.id, model, cap=row.daily_limit)
                remaining += state.daily_remaining
                allowed += state.requests_per_day
                minute += state.requests_this_minute
                left.setdefault(row.id, {})[model.id] = state.daily_remaining
                if not state.available:
                    blocked.append(state)
            budget.append({"model": model.id, "ref": model.ref, "provider": model.provider,
                           "keys": len(usable), "allowed_today": allowed, "remaining_today": remaining,
                           "used_today": allowed - remaining, "minute_used": minute,
                           "minute_limit": model.rpm * len(usable), "reset_timezone": model.reset_timezone,
                           "reset_in": round(self.ledger.model_reset(model)),
                           "blocked": len(blocked) == len(usable),
                           "retry_after": min((s.retry_after for s in blocked), default=0.0)})
        keys = [{"id": r.id, "label": r.label, "provider": r.provider, "hint": r.hint, "enabled": r.enabled,
                 "daily_limit": r.daily_limit, "last_ok_at": r.last_ok_at, "last_error": r.last_error,
                 "created_at": r.created_at, "usage": by_key.get(r.id, []), "remaining_today": left.get(r.id, {}),
                 "requests_today": sum(u["requests"] for u in by_key.get(r.id, []) if u["model"] == "*"),
                 "tokens_today": sum(u["tokens"] for u in by_key.get(r.id, []) if u["model"] == "*")}
                for r in rows]
        return {"keys": keys, "enabled": sum(1 for r in rows if r.enabled), "budget": budget,
                "reset_in": min((b["reset_in"] for b in budget), default=None)}

    # -- the single key saved before the pool existed ---------------------

    async def adopt_saved_keys(self, paid_cap: int) -> list[str]:
        """Bring each provider's single saved key (the LLM page's key field) into the pool, once, so it rotates with
        the others. Paid providers get `paid_cap` as their daily cap. Returns the providers adopted."""
        adopted = []
        known = {r.fingerprint for r in self.rows()}
        for provider in credentials.saved_llm_providers():
            secret = credentials.llm_key(provider) or ""
            if not secret.strip() or fingerprint(secret) in known:
                continue
            spec = providers.get(provider)
            try:
                await self.add(secret, "Saved key", provider, daily_limit=max(1, paid_cap) if spec.paid else None)
                adopted.append(provider)
            except LLMError:
                continue
        return adopted


keystore = KeyStore()
