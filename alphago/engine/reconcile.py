"""Settle simulations whose start got no answer from BRAIN, without ever paying for them twice.

A start that timed out or hit a 500/502/504 may still have run. Its candidates are `orphaned`, not resent. Every
minute this loop lists the alphas BRAIN created around the time each was sent and matches them by expression and
settings:

- found: the candidate is completed with that alpha, exactly as if the answer had arrived;
- not found 30 minutes after sending: BRAIN did not take it, so it is queued again (at most twice);
- still nothing after two requeues: it ends as an error, so it never loops.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

from sqlalchemy import select, update

from ..brain import brain
from ..brain.errors import BrainError
from ..brain.simulations import RESULT_KEYS, SimOutcome, normalize_settings
from ..db import db
from ..db.models import Candidate
from ..events import bus

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
SLACK = timedelta(minutes=2)          # clock skew allowed between our send and BRAIN's dateCreated
GIVE_UP = timedelta(minutes=30)       # not in the listing this long after sending = BRAIN never took it
MAX_REQUEUES = 2


def squash(code: str | None) -> str:
    """An expression as BRAIN may echo it: whitespace collapsed, a `{...}` wrapper dropped."""
    text = re.sub(r"\s+", "", str(code or ""))
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    return text.rstrip(";")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-9)
    return str(a).upper() == str(b).upper()


def matches(cand_expression: str, cand_settings: dict[str, Any], alpha: dict[str, Any]) -> bool:
    """True when `alpha` (a BRAIN alpha body) is what this candidate's request would have produced."""
    regular = alpha.get("regular")
    code = regular.get("code") if isinstance(regular, dict) else regular
    if squash(code) != squash(cand_expression):
        return False
    theirs = alpha.get("settings") or {}
    mine = normalize_settings(cand_settings)
    for key in RESULT_KEYS:
        if key in mine and mine[key] is not None and key in theirs and not _same(mine[key], theirs[key]):
            return False
    return True


def _bound(ts: datetime) -> str:
    return quote(ts.astimezone(NY).isoformat(timespec="seconds"), safe=":")


async def listing(start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Every alpha BRAIN created in [start, end), hidden ones included."""
    filters = f"dateCreated%3E={_bound(start)}&dateCreated%3C{_bound(end)}&hidden=true%1Ffalse"
    out: list[dict[str, Any]] = []
    async for item in brain.alphas.iterate(filters, order="dateCreated", page_size=100):
        out.append(item)
    return out


def _clusters(cands: list[Candidate], now: datetime) -> list[tuple[datetime, datetime, list[Candidate]]]:
    """Merge the overlapping listing windows of the orphans, so each stretch of time is read once."""
    spans = []
    for c in cands:
        anchor = c.sent_at or c.created_at
        spans.append((anchor - SLACK, min(now, anchor + GIVE_UP + SLACK), c))
    spans.sort(key=lambda x: x[0])
    out: list[tuple[datetime, datetime, list[Candidate]]] = []
    for lo, hi, c in spans:
        if out and lo <= out[-1][1]:
            plo, phi, members = out[-1]
            out[-1] = (plo, max(phi, hi), members + [c])
        else:
            out.append((lo, hi, [c]))
    return out


async def reconcile_once(now: datetime | None = None) -> dict[str, int]:
    """One pass over the orphaned candidates. Returns counts: matched, requeued, abandoned, waiting."""
    from .simulator import simulator
    now = now or datetime.now(timezone.utc)
    async with db.session() as s:
        orphans = list(await s.scalars(select(Candidate).where(Candidate.status == "orphaned")))
    counts = {"matched": 0, "requeued": 0, "abandoned": 0, "waiting": 0}
    if not orphans or not brain.client.logged_in:
        counts["waiting"] = len(orphans)
        return counts
    for lo, hi, members in _clusters(orphans, now):
        try:
            alphas = await listing(lo, hi + timedelta(seconds=1))
        except BrainError as exc:
            log.warning("orphan reconcile: listing failed: %s", exc.message)
            counts["waiting"] += len(members)
            continue
        taken: set[str] = set()
        for c in members:
            anchor = c.sent_at or c.created_at
            hit = None
            for a in alphas:
                aid = str(a.get("id"))
                created = a.get("dateCreated")
                try:
                    created_at = datetime.fromisoformat(str(created)) if created else None
                except ValueError:
                    created_at = None
                if aid in taken or (created_at and created_at < anchor - SLACK):
                    continue
                if matches(c.expression, c.settings or {}, a):
                    hit = aid
                    break
            if hit:
                taken.add(hit)
                await simulator._ingest([c], [SimOutcome(0, c.sim_id, "COMPLETE", hit)], None)
                counts["matched"] += 1
            elif now - anchor > GIVE_UP:
                tries = (c.orphan_requeues or 0) + 1
                async with db.write() as s:
                    if tries > MAX_REQUEUES:
                        await s.execute(update(Candidate).where(Candidate.id == c.id).values(
                            status="error", finished_at=now,
                            error="BRAIN never answered this simulation, and it never appeared in your alpha list "
                                  f"after {MAX_REQUEUES} tries; left out so it cannot loop."))
                    else:
                        await s.execute(update(Candidate).where(Candidate.id == c.id).values(
                            status="queued", sent_at=None, sim_id=None, orphan_requeues=tries))
                if tries > MAX_REQUEUES:
                    simulator._resolve(c.id)
                    counts["abandoned"] += 1
                else:
                    c.status = "queued"
                    simulator.requeue([c])
                    counts["requeued"] += 1
            else:
                counts["waiting"] += 1
    if counts["matched"] or counts["requeued"] or counts["abandoned"]:
        bus.publish("simulate", f"Unanswered simulations settled: {counts['matched']} found on BRAIN, "
                                f"{counts['requeued']} queued again, {counts['abandoned']} given up.", "info")
    return counts


async def reconcile_loop(interval_s: float = 60.0) -> None:
    while True:
        try:
            await reconcile_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("orphan reconcile failed")
        await asyncio.sleep(interval_s)
