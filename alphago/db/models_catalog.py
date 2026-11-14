"""Tables of the catalog features ported from alpha-harness (owned by that package; add new tables here).

The field search index (`field_search`, an SQLite FTS5 table) is not an ORM table: SQLAlchemy cannot declare a
virtual table, so `alphago.catalog.search` creates and rebuilds it with raw SQL.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base, UTCDateTime, utcnow  # noqa: F401


class CatalogSyncRun(Base):
    """One all-markets catalogue download: kept so the Sync matrix still shows what failed after a restart."""

    __tablename__ = "catalog_sync_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16), default="all")          # all | agnostic
    status: Mapped[str] = mapped_column(String(12), default="RUNNING")    # RUNNING | COMPLETE | FAILED | CANCELLED
    job_id: Mapped[int | None] = mapped_column(Integer)
    fields_synced: Mapped[int] = mapped_column(Integer, default=0)
    datasets_synced: Mapped[int] = mapped_column(Integer, default=0)
    markets: Mapped[Any] = mapped_column(JSON, nullable=True)            # [{region, delay, universe, state, fields}]
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
