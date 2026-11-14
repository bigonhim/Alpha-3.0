"""Tables of the pool features ported from alpha-harness (owned by that package; add new tables here)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base, UTCDateTime, utcnow


class TagGroup(Base):
    """A set of tags kept under a name, as alpha-harness' Tag Groups: click it to filter the pool by them.

    `every` says how the tags combine: all of them (True) or any of them (False)."""

    __tablename__ = "pool_tag_groups"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(60), unique=True)
    tags: Mapped[Any] = mapped_column(JSON)
    every: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
