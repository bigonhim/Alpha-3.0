"""Sync BRAIN → local database. Each function is also registered as a background job kind."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..brain import brain
from ..db import db
from ..db.models import (Alpha, ConsultantSnapshot, DataCategory, DataField, Dataset, Operator, Pyramid)
from ..events import bus
from ..jobs.queue import JobContext, jobs
from . import catalogs
from .ingest import apply_alpha_json

log = logging.getLogger(__name__)


def now() -> datetime:
    return datetime.now(timezone.utc)


def current_quarter(d: date | None = None) -> str:
    d = d or date.today()
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def quarter_bounds(d: date | None = None) -> tuple[date, date]:
    d = d or date.today()
    q = (d.month - 1) // 3
    start = date(d.year, q * 3 + 1, 1)
    end = date(d.year + (q == 3), (q * 3 + 3) % 12 + 1, 1) - timedelta(days=1)
    return start, end


async def _mark(key: str) -> None:
    await db.kv_set(f"last_sync:{key}", now().isoformat())


async def last_synced(key: str) -> datetime | None:
    value = await db.kv_get(f"last_sync:{key}")
    return datetime.fromisoformat(value) if value else None


# ---------------------------------------------------------------- catalogue

async def sync_operators(ctx: JobContext | None = None, _params: dict | None = None) -> str:
    ops = await brain.data.operators()
    async with db.write() as s:
        for op in ops:
            stmt = sqlite_insert(Operator).values(
                name=op.get("name"), category=op.get("category"), scope=op.get("scope"),
                definition=op.get("definition"), description=op.get("description"),
                documentation=op.get("documentation"), level=op.get("level"), synced_at=now())
            await s.execute(stmt.on_conflict_do_update(index_elements=["name"], set_={
                "category": stmt.excluded.category, "scope": stmt.excluded.scope,
                "definition": stmt.excluded.definition, "description": stmt.excluded.description,
                "documentation": stmt.excluded.documentation, "level": stmt.excluded.level,
                "synced_at": stmt.excluded.synced_at}))
    catalogs.invalidate()
    await _mark("operators")
    bus.publish("sync", f"Synced {len(ops)} operators available to your account.", "success")
    return f"{len(ops)} operators"


async def sync_categories(ctx: JobContext | None = None, _params: dict | None = None) -> str:
    cats = await brain.data.categories()

    def flatten(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for c in items:
            out.append(c)
            out.extend(flatten(c.get("children") or []))
        return out

    flat = flatten(cats)
    async with db.write() as s:
        for c in flat:
            stmt = sqlite_insert(DataCategory).values(id=str(c.get("id")), name=c.get("name") or str(c.get("id")),
                                                      payload={k: v for k, v in c.items() if k != "children"})
            await s.execute(stmt.on_conflict_do_update(index_elements=["id"], set_={
                "name": stmt.excluded.name, "payload": stmt.excluded.payload}))
    await _mark("categories")
    return f"{len(flat)} categories"


def _cat(value: Any) -> tuple[str | None, str | None]:
    if isinstance(value, dict):
        return (str(value.get("id")) if value.get("id") is not None else None), value.get("name")
    if value:
        return str(value), str(value)
    return None, None


async def sync_datasets(ctx: JobContext | None, params: dict) -> str:
    region, delay, universe = params["region"], int(params["delay"]), params["universe"]
    count = 0
    async with db.write() as s:
        async for d in brain.data.datasets(region, delay, universe):
            cat_id, cat_name = _cat(d.get("category"))
            sub_id, sub_name = _cat(d.get("subcategory"))
            values = dict(id=str(d.get("id")), region=region, delay=delay, universe=universe, name=d.get("name"),
                          description=d.get("description"), category_id=cat_id, category_name=cat_name,
                          subcategory=sub_name, coverage=d.get("coverage"), value_score=d.get("valueScore"),
                          user_count=d.get("userCount"), alpha_count=d.get("alphaCount"),
                          field_count=d.get("fieldCount"), pyramid_multiplier=d.get("pyramidMultiplier"),
                          themes=d.get("themes"), synced_at=now())
            stmt = sqlite_insert(Dataset).values(**values)
            await s.execute(stmt.on_conflict_do_update(
                index_elements=["id", "region", "delay", "universe"],
                set_={k: getattr(stmt.excluded, k) for k in values if k not in ("id", "region", "delay", "universe")}))
            count += 1
            if ctx and count % 25 == 0:
                await ctx.update(message=f"{count} datasets")
    await _mark(f"datasets:{region}:{delay}:{universe}")
    bus.publish("sync", f"Synced {count} datasets for {region} D{delay} {universe}.", "success")
    return f"{count} datasets"


async def _upsert_fields(rows: list[dict[str, Any]], region: str, delay: int, universe: str) -> None:
    async with db.write() as s:
        for f in rows:
            ds = f.get("dataset") or {}
            _cid, cname = _cat(f.get("category"))
            values = dict(id=str(f.get("id")), dataset_id=str(ds.get("id") if isinstance(ds, dict) else ds),
                          region=region, delay=delay, universe=universe, type=str(f.get("type") or "MATRIX"),
                          description=f.get("description"), category=cname, coverage=f.get("coverage"),
                          user_count=f.get("userCount"), alpha_count=f.get("alphaCount"), themes=f.get("themes"),
                          synced_at=now())
            stmt = sqlite_insert(DataField).values(**values)
            await s.execute(stmt.on_conflict_do_update(
                index_elements=["id", "region", "delay", "universe"],
                set_={k: getattr(stmt.excluded, k) for k in values if k not in ("id", "region", "delay", "universe")}))


async def sync_fields(ctx: JobContext | None, params: dict) -> str:
    """Fields for every dataset in scope (per-dataset paging avoids the 10,000-result cap)."""
    region, delay, universe = params["region"], int(params["delay"]), params["universe"]
    only = params.get("dataset_ids")
    async with db.session() as s:
        q = select(Dataset).where(Dataset.region == region, Dataset.delay == delay, Dataset.universe == universe)
        datasets = list(await s.scalars(q))
    if not datasets:
        await sync_datasets(ctx, params)
        async with db.session() as s:
            datasets = list(await s.scalars(select(Dataset).where(
                Dataset.region == region, Dataset.delay == delay, Dataset.universe == universe)))
    if only:
        datasets = [d for d in datasets if d.id in set(only)]
    total = 0
    for i, ds in enumerate(datasets):
        batch = [f async for f in brain.data.datafields(region, delay, universe, dataset_id=ds.id)]
        await _upsert_fields(batch, region, delay, universe)
        async with db.write() as s:
            row = await s.get(Dataset, ds.pk)
            if row:
                row.fields_synced_at = now()
        total += len(batch)
        if ctx:
            await ctx.update((i + 1) / len(datasets), f"{ds.id}: {len(batch)} fields ({total} total)")
    catalogs.invalidate((region, delay, universe))
    await _mark(f"fields:{region}:{delay}:{universe}")
    bus.publish("analyse", f"Catalogued {total:,} fields across {len(datasets)} datasets ({region} D{delay} {universe}).",
                "success")
    return f"{total} fields in {len(datasets)} datasets"


async def sync_scope(ctx: JobContext | None, params: dict) -> str:
    """Operators + datasets + fields for one scope."""
    await sync_operators(ctx, params)
    if ctx:
        await ctx.update(0.05, "datasets")
    await sync_datasets(ctx, params)
    return await sync_fields(ctx, params)


# ---------------------------------------------------------------- account

async def _ingest_alpha_page(items: list[dict[str, Any]]) -> int:
    async with db.write() as s:
        for data in items:
            if not data.get("id"):
                continue
            row = await s.get(Alpha, str(data["id"]))
            if row is None:
                row = Alpha(id=str(data["id"]), origin="synced")
                s.add(row)
            apply_alpha_json(row, data)
    return len(items)


async def sync_alphas(ctx: JobContext | None, params: dict) -> str:
    """ACTIVE alphas in full + recent UNSUBMITTED ones (incremental unless params['full'])."""
    full = bool(params.get("full"))
    max_unsubmitted = int(params.get("max_unsubmitted", 3000))
    total = 0
    page: list[dict[str, Any]] = []
    async for item in brain.alphas.iterate({"status": "ACTIVE"}, order="-dateSubmitted"):
        page.append(item)
        if len(page) >= 100:
            total += await _ingest_alpha_page(page)
            page = []
            if ctx:
                await ctx.update(0.3, f"{total} submitted alphas")
    if page:
        total += await _ingest_alpha_page(page)
    active = total
    await _mark("active")                  # points/tags of submitted alphas are now fresh from BRAIN
    if params.get("active_only"):
        bus.publish("sync", f"Synced {active} submitted alphas from BRAIN.", "success")
        return f"{active} submitted"
    since = None if full else await last_synced("alphas")
    page = []
    count_unsub = 0
    async for item in brain.alphas.iterate({"status": "UNSUBMITTED"}, max_items=None if full else max_unsubmitted):
        created = item.get("dateCreated")
        if since and created and datetime.fromisoformat(created.replace("Z", "+00:00")) < since - timedelta(days=1):
            break
        page.append(item)
        count_unsub += 1
        if len(page) >= 100:
            await _ingest_alpha_page(page)
            page = []
            if ctx:
                await ctx.update(0.6, f"{active} submitted, {count_unsub} unsubmitted")
    if page:
        await _ingest_alpha_page(page)
    await _mark("alphas")
    bus.publish("sync", f"Synced {active} submitted and {count_unsub} unsubmitted alphas from BRAIN.", "success")
    return f"{active} submitted, {count_unsub} unsubmitted"


async def sync_programs(ctx: JobContext | None, params: dict | None = None) -> str:
    """Consultant summary (Genius), pyramid multipliers and this quarter's pyramid alpha counts."""
    notes = []
    try:
        summary = await brain.users.consultant_summary()
        async with db.write() as s:
            s.add(ConsultantSnapshot(payload=summary))
        notes.append("consultant summary")
    except Exception as exc:
        notes.append(f"summary unavailable ({getattr(exc, 'message', exc)})")
    quarter = current_quarter()
    start, end = quarter_bounds()
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    try:
        mult = await brain.users.pyramid_multipliers()
        for p in (mult or {}).get("pyramids", []):
            cat = p.get("category") or {}
            key = (str(cat.get("id")), p.get("region"), int(p.get("delay", 1)))
            cells.setdefault(key, {"name": cat.get("name")})["multiplier"] = p.get("multiplier")
        notes.append("pyramid multipliers")
    except Exception as exc:
        notes.append(f"multipliers unavailable ({getattr(exc, 'message', exc)})")
    try:
        counts = await brain.users.pyramid_alphas(start.isoformat(), end.isoformat())
        for p in (counts or {}).get("pyramids", []):
            cat = p.get("category") or {}
            key = (str(cat.get("id")), p.get("region"), int(p.get("delay", 1)))
            cells.setdefault(key, {"name": cat.get("name")})["alpha_count"] = p.get("alphaCount", 0)
        notes.append("pyramid counts")
    except Exception as exc:
        notes.append(f"pyramid counts unavailable ({getattr(exc, 'message', exc)})")
    if cells:
        async with db.write() as s:
            for (cat_id, region, delay), v in cells.items():
                stmt = sqlite_insert(Pyramid).values(category_id=cat_id, category_name=v.get("name"), region=region,
                                                     delay=delay, quarter=quarter, multiplier=v.get("multiplier"),
                                                     alpha_count=v.get("alpha_count") or 0, updated_at=now())
                await s.execute(stmt.on_conflict_do_update(
                    index_elements=["category_id", "region", "delay", "quarter"],
                    set_={"category_name": stmt.excluded.category_name, "multiplier": stmt.excluded.multiplier,
                          "alpha_count": stmt.excluded.alpha_count, "updated_at": stmt.excluded.updated_at}))
    await _mark("programs")
    bus.publish("sync", "Synced program data: " + ", ".join(notes) + ".")
    return "; ".join(notes)


async def sync_options(ctx: JobContext | None = None, _params: dict | None = None) -> str:
    from ..runtime import load_options_quietly
    await load_options_quietly()
    await _mark("options")
    return f"{len(brain.options.regions)} regions"


async def catalog_counts() -> dict[str, int]:
    async with db.session() as s:
        return {
            "operators": await s.scalar(select(func.count()).select_from(Operator)) or 0,
            "datasets": await s.scalar(select(func.count()).select_from(Dataset)) or 0,
            "fields": await s.scalar(select(func.count()).select_from(DataField)) or 0,
            "alphas": await s.scalar(select(func.count()).select_from(Alpha)) or 0,
            "active": await s.scalar(select(func.count()).select_from(Alpha).where(Alpha.status == "ACTIVE")) or 0,
        }


# register job kinds
jobs.register("sync_operators")(sync_operators)
jobs.register("sync_categories")(sync_categories)
jobs.register("sync_datasets")(sync_datasets)
jobs.register("sync_fields")(sync_fields)
jobs.register("sync_scope")(sync_scope)
jobs.register("sync_alphas")(sync_alphas)
jobs.register("sync_programs")(sync_programs)
jobs.register("sync_options")(sync_options)
