"""Finding a data field by meaning rather than by substring (alpha-harness catalog/search.py, on SQLite FTS5).

BRAIN's field ids are compressed jargon (`anl44_eps_prevalue`) but their descriptions are plain English, so a BM25
index over the id's own words *and* the description finds a field from the words a person would actually use. The id
is split where BRAIN put a separator and nowhere else; the description is taken as written; the porter stemmer makes
"revision" find "revisions".

One row per field id rather than per market: the text is identical across markets, and BM25's term statistics are
better drawn from the whole catalogue than from one slice. Narrowing to a market stays the caller's job.
"""

from __future__ import annotations

import logging
import re

from sqlalchemy import text

from ..db import db

log = logging.getLogger(__name__)

TABLE = "field_search"
_CREATE = f"CREATE VIRTUAL TABLE {TABLE} USING fts5(field_id UNINDEXED, text, tokenize='porter unicode61')"
_WORDS = re.compile(r"[^0-9a-z]+")

#: The rows a query matches, each with its BM25 score (FTS5's bm25() is lower-is-better, so it is negated).
SCORED = (f"(SELECT field_id, -bm25({TABLE}) AS score FROM {TABLE} WHERE {TABLE} MATCH :fts_query) AS s")


def query_terms(search: str) -> list[str]:
    """A typed phrase as the words BM25 can match; empty when nothing is left of it."""
    return [w for w in _WORDS.split((search or "").lower()) if w]


def match_expression(terms: list[str], *, conjunctive: bool) -> str:
    """An FTS5 MATCH string: every word (AND) or any word (OR). Each word is quoted, so nothing in it is read as
    FTS syntax."""
    quoted = [f'"{t}"' for t in terms]
    return " ".join(quoted) if conjunctive else " OR ".join(quoted)


def id_words(field_id: str) -> str:
    """`anl44_eps_prevalue` → `anl44 eps prevalue`: split on the separators, and nowhere else."""
    return " ".join(w for w in _WORDS.split(field_id.lower()) if w)


async def ready() -> bool:
    """Whether a search can use the index (it exists and holds something)."""
    async with db.session() as s:
        exists = await s.scalar(text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n"), {"n": TABLE})
        if not exists:
            return False
        return bool(await s.scalar(text(f"SELECT 1 FROM {TABLE} LIMIT 1")))


async def stale() -> bool:
    """Whether the index misses fields the catalogue holds (a per-market sync from the Sync page's older controls
    stores fields without rebuilding it)."""
    async with db.session() as s:
        held = await s.scalar(text("SELECT count(DISTINCT id) FROM datafields")) or 0
        exists = await s.scalar(text("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n"), {"n": TABLE})
        indexed = (await s.scalar(text(f"SELECT count(*) FROM {TABLE}")) or 0) if exists else 0
    return held != indexed


async def rebuild() -> int:
    """Rebuild the corpus and its index from the stored fields; returns how many fields were indexed.

    The corpus is read first and written in one transaction, so a search never meets a half-built index."""
    async with db.session() as s:
        rows = (await s.execute(text(
            "SELECT id, max(description) FROM datafields GROUP BY id"))).all()
    corpus = [{"f": str(fid), "t": f"{id_words(str(fid))} {desc or ''}".strip()} for fid, desc in rows]
    async with db.write() as s:
        await s.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
        await s.execute(text(_CREATE))
        if corpus:
            await s.execute(text(f"INSERT INTO {TABLE} (field_id, text) VALUES (:f, :t)"), corpus)
    log.info("field search index rebuilt: %d fields", len(corpus))
    return len(corpus)
