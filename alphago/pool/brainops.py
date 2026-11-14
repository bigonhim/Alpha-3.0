"""What the pool asks of BRAIN: tags, the account's alpha counts, alphas imported by tag, Production Correlation and
the Power Pool Workflow. None of it spends simulation quota.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Sequence
from typing import Any

import httpx
from sqlalchemy import not_, select

from ..brain import brain
from ..brain.errors import BrainError
from ..brain.platform import POWER_POOL_TAG, PROD_CORRELATION_CHECK, is_power_pool
from ..db import db
from ..db.models import Alpha
from ..jobs.queue import JobContext, jobs
from . import correlation, store

log = logging.getLogger(__name__)

TAG = re.compile(r"^[A-Za-z0-9_-]{1,40}$")      # a tag as BRAIN keeps them
MAX_TAGGED = 500
MAX_TAGS = 20
TAG_CONCURRENCY = 4
PROD_WAIT = 300.0                                # BRAIN limits correlation checks per hour: never re-ask sooner
PROD_WAIT_MAX = 3600.0
POWER_POOL_REFUSAL = (f"{POWER_POOL_TAG} keeps an Alpha in the Power Pool's correlation checks for good, even once "
                      "removed, so add it on each Alpha's own page.")


def parse_tags(text: str | Sequence[str]) -> list[str]:
    """Tags typed comma- or space-separated, deduplicated; raises ValueError with what to fix."""
    raw = re.split(r"[\s,]+", text) if isinstance(text, str) else list(text)
    wanted = list(dict.fromkeys(t.strip() for t in raw if t and t.strip()))
    if not wanted:
        raise ValueError("Write a tag to add.")
    if bad := [t for t in wanted if not TAG.match(t)]:
        raise ValueError(f"BRAIN keeps tags of letters, digits, '_' and '-', up to 40 characters: {', '.join(bad)}")
    if len(wanted) > MAX_TAGS:
        raise ValueError(f"Add at most {MAX_TAGS} tags at a time.")
    return wanted


async def add_tags(alpha_ids: Sequence[str], tags: str | Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """Add tags beside each alpha's own on BRAIN and change nothing else; returns (tagged, failed by alpha).

    Each alpha is read from BRAIN first: saving replaces the whole list, and the local copy can be older than an edit
    made on BRAIN since. Only `tags` is patched, and what BRAIN kept is verified."""
    wanted = parse_tags(tags)
    if any(t.casefold() == POWER_POOL_TAG.casefold() for t in wanted):
        raise ValueError(POWER_POOL_REFUSAL)
    ids = list(dict.fromkeys(alpha_ids))
    if not ids:
        raise ValueError("Tick the alphas to tag first.")
    if len(ids) > MAX_TAGGED:
        raise ValueError(f"Tag at most {MAX_TAGGED} alphas at a time.")
    gate = asyncio.Semaphore(TAG_CONCURRENCY)
    tagged: dict[str, list[str]] = {}
    failed: dict[str, str] = {}

    async def one(alpha_id: str) -> None:
        async with gate:
            try:
                current = await brain.alphas.get(alpha_id) or {}
                had = [t for t in current.get("tags") or [] if isinstance(t, str)]
                if missing := [t for t in wanted if t not in had]:
                    current = await brain.alphas.patch(alpha_id, tags=had + missing) or {}
                    if not current.get("id"):
                        current = await brain.alphas.get(alpha_id) or {}
            except BrainError as exc:
                failed[alpha_id] = exc.message
                return
        now = [t for t in current.get("tags") or [] if isinstance(t, str)]
        if any(t not in now for t in wanted):
            failed[alpha_id] = "BRAIN answered, but did not keep the tag."
        else:
            tagged[alpha_id] = now

    await asyncio.gather(*(one(a) for a in ids))
    await store.save_tags(tagged)
    return list(tagged), failed


async def brain_summary() -> dict[str, int]:
    """BRAIN's own counts (GET /users/self/alphas/summary, version 4.0): Unsubmitted, and Submitted as BRAIN's
    Submitted tab counts it (active + decommissioned)."""
    client = brain.client
    response = await client.request("GET", "/users/self/alphas/summary",
                                    headers={"Accept": "application/json;version=4.0"})
    body = client.raise_for(response) or {}

    def count(key: str) -> int:
        value = body.get(key) if isinstance(body, dict) else None
        if isinstance(value, dict):
            value = value.get("count")
        return int(value or 0)

    unsubmitted, active, decommissioned = count("unsubmitted"), count("active"), count("decommissioned")
    return {"unsubmitted": unsubmitted, "active": active, "decommissioned": decommissioned,
            "submitted": active + decommissioned}


# ---------------------------------------------------------------- import by tag

async def _job_import_tagged(ctx: JobContext, params: dict) -> str:
    """Every alpha BRAIN lists under any of these tags, hidden or not, made here or elsewhere (e.g. by ACE)."""
    from ..engine.sync import HIDDEN_TOO, _ingest_alpha_page
    tags = parse_tags(params.get("tags") or [])
    filters = f"tags={'%1F'.join(tags)}&{HIDDEN_TOO}"
    total = 0
    page: list[dict[str, Any]] = []
    async for item in brain.alphas.iterate(filters, order="-dateCreated"):
        page.append(item)
        if len(page) >= 100:
            total += await _ingest_alpha_page(page)
            page = []
            await ctx.update(None, f"{total} alphas imported")
    if page:
        total += await _ingest_alpha_page(page)
    return f"Imported {total} Alpha{'' if total == 1 else 's'} tagged {', '.join(tags)}"


# ---------------------------------------------------------------- production correlation

class _RateLimited(Exception):
    def __init__(self, retry_after: float | None) -> None:
        super().__init__("rate limited")
        self.retry_after = retry_after


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


async def _prod_once(alpha_id: str, max_wait_s: float = 600.0) -> dict[str, Any]:
    """GET /alphas/{id}/correlations/prod, following BRAIN's Retry-After while it computes; a 429 is raised to the
    caller, which waits out BRAIN's hourly limit rather than spend more refusals."""
    url = f"/alphas/{alpha_id}/correlations/prod"
    waited = 0.0
    while True:
        response = await brain.client.request("GET", url, retry_statuses=(500, 502, 504))
        if response.status_code == 429:
            raise _RateLimited(_retry_after(response))
        wait = _retry_after(response)
        if wait is not None and response.status_code < 400 and waited < max_wait_s:
            await asyncio.sleep(max(0.5, min(wait, 30.0)))
            waited += max(0.5, min(wait, 30.0))
            continue
        return brain.client.raise_for(response) or {}


def prod_key(alpha_id: str) -> str:
    return f"correlation:prod:{alpha_id}"


async def known_prod(alphas: dict[str, Alpha]) -> dict[str, float]:
    """Production Correlation where BRAIN gave one: a kept answer of its correlation check, else PROD_CORRELATION."""
    known: dict[str, float] = {}
    for aid, a in alphas.items():
        for c in a.checks or []:
            if c.get("name") == PROD_CORRELATION_CHECK and isinstance(c.get("value"), (int, float)):
                known[aid] = float(c["value"])
        kept = await db.kv_get(prod_key(aid))
        if isinstance(kept, dict) and isinstance(kept.get("max"), (int, float)):
            known[aid] = float(kept["max"])
    return known


async def _job_prod(ctx: JobContext, params: dict) -> str:
    ids = list(dict.fromkeys(params.get("alpha_ids") or []))
    alphas = await store.by_ids(ids)
    known = await known_prod(alphas)
    wanted = [a for a in ids if a in alphas and not store.is_submitted(alphas[a]) and a not in known]
    failed = 0
    for done, alpha_id in enumerate(wanted):
        progress = f"{done} of {len(wanted)}"
        await ctx.update(done / max(1, len(wanted)), progress)
        while True:
            try:
                answer = await _prod_once(alpha_id)
            except _RateLimited as exc:
                wait = min(max(exc.retry_after or 0.0, PROD_WAIT), PROD_WAIT_MAX)
                await ctx.update(None, f"{progress} · BRAIN's hourly limit, next in {math.ceil(wait / 60)} min")
                await asyncio.sleep(wait)
                continue
            except BrainError as exc:          # one alpha BRAIN cannot check must not abandon the rest
                log.warning("prod correlation of %s failed: %s", alpha_id, exc.message)
                failed += 1
                break
            await db.kv_set(prod_key(alpha_id), {"max": answer.get("max"), "min": answer.get("min")})
            if isinstance(answer.get("max"), (int, float)):
                async with db.write() as s:
                    row = await s.get(Alpha, alpha_id)
                    if row is not None:
                        row.prod_corr = float(answer["max"])
            break
    await ctx.update(1.0, f"{len(wanted)} of {len(wanted)}")
    detail = f"Production Correlation for {len(wanted) - failed} Alphas"
    if skipped := len(ids) - len(wanted):
        detail += f" ({skipped} already known or submitted)"
    return detail + (f". BRAIN could not check {failed}" if failed else "")


# ---------------------------------------------------------------- Power Pool Workflow

async def unsubmitted_ids() -> list[str]:
    async with db.session() as s:
        return list(await s.scalars(select(Alpha.id).where(not_(store.submitted_clause()))))


async def _job_workflow(ctx: JobContext, params: dict) -> str:
    """Narrow in three steps, each on what the one before kept: unsubmitted, then no check FAIL or ERROR, then
    Power Pool Correlation passed. PnL alone decides the correlation, so it is downloaded first, and only in a
    region where a submitted Power Pool alpha exists to collide with; turnover follows only for those that pass."""
    from ..engine import series
    ids = list(dict.fromkeys(params.get("alpha_ids") or [])) or await unsubmitted_ids()
    stored = await store.by_ids(ids)
    unsubmitted = [a for a in ids if a in stored and not store.is_submitted(stored[a])]
    survivors = [a for a in unsubmitted if store.verdict_of(stored[a]) in ("submittable", "pending")]
    judged = [a for a in survivors if is_power_pool(stored[a].checks)]
    await ctx.update(0.05, f"{len(unsubmitted)} Unsubmitted → {len(survivors)} with no Checks FAIL or ERROR")
    members = await store.submitted_members()
    pools = correlation.power_pools(members)
    contested = {stored[a].region for a in judged} & set(pools)
    need = [a for a in judged if stored[a].region in contested] + [p for r in contested for p in pools[r]]
    lacking = await series.missing(need, turnover=False)
    _, errors = await series.fetch_many(lacking, turnover=False, ctx=ctx)
    found = await correlation.power_pool_analysis(judged)
    passed = [r["alpha_id"] for r in found["rows"] if r["verdict"] in correlation.PASSING]
    lacking = await series.missing(passed, turnover=True)
    _, more = await series.fetch_many(lacking, turnover=True, ctx=ctx)
    errors += more
    detail = (f"{len(unsubmitted)} Unsubmitted Alphas → {len(survivors)} with no Checks FAIL or ERROR → "
              f"{len(passed)} Pass Power Pool Correlation")
    return detail + (f". {len(errors)} could not be downloaded" if errors else "")


jobs.register("pool_import_tagged")(_job_import_tagged)
jobs.register("pool_prod_correlation")(_job_prod)
jobs.register("pool_power_pool_workflow")(_job_workflow)
