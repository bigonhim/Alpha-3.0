"""Tables of the AI features ported from alpha-harness: the key pool, the per-key budget ledger, the model registry
and the assistant's conversations.

Secrets never live here: a key's value stays in Windows Credential Manager (`alphago.credentials.pool_key`); this
table holds only what is safe to show (a hint) and a fingerprint, a hash rather than a recoverable copy, so the same
key cannot be added twice.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base, UTCDateTime, utcnow


class AIKey(Base):
    """One API key of one provider. Several per provider add their free budgets together (quota is per account)."""

    __tablename__ = "ai_keys"
    # ids are never handed out twice: the ledger and the keyring entry are keyed by id
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)       # key family: gemini, claude, openai, groq …
    label: Mapped[str] = mapped_column(String(64))
    hint: Mapped[str] = mapped_column(String(32))                      # "AIzaSy…x1Q9"
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)   # sha256 of the stripped key
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # the user's own daily request ceiling over every model; required (≥ 1) for a provider that bills the user
    daily_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_ok_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class AIKeyUsage(Base):
    """Requests and tokens per (key, model, quota day). `day` is the date on the model's own reset clock; model "*"
    counts every model the key served, for its own daily cap. Survives restarts, so a spent key never looks fresh."""

    __tablename__ = "ai_key_usage"
    __table_args__ = (UniqueConstraint("key_id", "model", "day", name="uq_ai_key_model_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_id: Mapped[int] = mapped_column(Integer, ForeignKey("ai_keys.id", ondelete="CASCADE"), index=True)
    model: Mapped[str] = mapped_column(String(200))
    day: Mapped[str] = mapped_column(String(10))                         # YYYY-MM-DD in the model's reset time zone
    requests: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    last_request_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class AIModel(Base):
    """A model the user set up with the limits their provider shows them. Limits apply per key."""

    __tablename__ = "ai_models"
    __table_args__ = (Index("uq_ai_model_provider_model", "provider", "model", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(200))
    rpm: Mapped[int] = mapped_column(Integer)
    rpd: Mapped[int] = mapped_column(Integer)
    reset_timezone: Mapped[str] = mapped_column(String(64), default="America/Los_Angeles")
    max_prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class ChatThread(Base):
    """One conversation with the assistant, pinned to the market it is about."""

    __tablename__ = "chat_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(160))
    instrument_type: Mapped[str] = mapped_column(String(16), default="EQUITY")
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow, index=True)


class ChatMessage(Base):
    """One turn. `meta` carries an assistant turn's picks, dropped fields, datasets, model, reasoning and tokens."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[int] = mapped_column(Integer, ForeignKey("chat_threads.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))                       # user | assistant
    text: Mapped[str] = mapped_column(Text)
    meta: Mapped[Any] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
