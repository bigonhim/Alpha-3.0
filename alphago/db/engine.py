"""Async SQLite engine. Reads use `session()`, writes use `write()` (serialised by a lock to avoid SQLITE_BUSY)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from ..config import db_path
from .models import KV, Base, EventRow

log = logging.getLogger(__name__)


def _add_missing_columns(sync_conn) -> None:  # noqa: ANN001
    """Tiny forward-only migration: add columns that newer app versions introduced (SQLite ALTER TABLE ADD)."""
    for table in Base.metadata.sorted_tables:
        existing = {row[1] for row in sync_conn.exec_driver_sql(f'PRAGMA table_info("{table.name}")')}
        for column in table.columns:
            if column.name not in existing:
                coltype = column.type.compile(dialect=sync_conn.dialect)
                sync_conn.exec_driver_sql(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {coltype}')
                log.info("migrated: added %s.%s", table.name, column.name)


# Indexes on columns that later versions added to existing tables (create_all only indexes brand-new tables).
_EXTRA_INDEXES = {
    "ix_cand_task": ("candidates", "task_id, status"),
    "ix_batch_task": ("batches", "task_id"),
}


def _ensure_indexes(sync_conn) -> None:  # noqa: ANN001
    for name, (table, cols) in _EXTRA_INDEXES.items():
        sync_conn.exec_driver_sql(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}" ({cols})')


class Database:
    def __init__(self) -> None:
        self.engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None
        self._write_lock = asyncio.Lock()

    async def init(self, path: Path | str | None = None) -> None:
        url = "sqlite+aiosqlite:///:memory:" if path == ":memory:" else f"sqlite+aiosqlite:///{path or db_path()}"
        self.engine = create_async_engine(url, connect_args={"timeout": 30})

        @event.listens_for(self.engine.sync_engine, "connect")
        def _pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(_add_missing_columns)
            await conn.run_sync(_ensure_indexes)
        self._sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)
        log.info("database ready at %s", path or db_path())

    async def close(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        assert self._sessionmaker is not None, "Database.init() not called"
        async with self._sessionmaker() as s:
            yield s

    @asynccontextmanager
    async def write(self) -> AsyncIterator[AsyncSession]:
        """A session that commits on success. Writes are serialised across the app."""
        assert self._sessionmaker is not None, "Database.init() not called"
        async with self._write_lock:
            async with self._sessionmaker() as s:
                try:
                    yield s
                    await s.commit()
                except Exception:
                    await s.rollback()
                    raise

    # ------------------------------------------------------------ helpers

    async def kv_get(self, key: str, default: Any = None) -> Any:
        async with self.session() as s:
            row = await s.get(KV, key)
            return row.value if row is not None else default

    async def kv_set(self, key: str, value: Any) -> None:
        async with self.write() as s:
            row = await s.get(KV, key)
            if row is None:
                s.add(KV(key=key, value=value))
            else:
                row.value = value

    async def persist_event(self, ev) -> None:  # noqa: ANN001  (events.Event)
        async with self.write() as s:
            s.add(EventRow(ts=ev.ts, source=ev.source, level=ev.level, message=ev.message, payload=ev.payload or None))

    async def recent_events(self, limit: int = 100) -> list[EventRow]:
        async with self.session() as s:
            rows = await s.scalars(select(EventRow).order_by(EventRow.id.desc()).limit(limit))
            return list(rows)


db = Database()
