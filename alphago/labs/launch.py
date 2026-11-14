"""From a lab's form to a task: what the account and the market allow (alpha-harness labs/launch.py).

Every lab asks the same questions — which operators the account has, which universes and neutralizations BRAIN
allows, whether the market is downloaded — so the answers live here and the labs cannot drift apart on them.
Nothing here raises: a problem is returned as a sentence and the lab decides whether it blocks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from sqlalchemy import func, select

from ..brain import brain
from ..db import db
from ..db.models import DataField, Dataset, Operator

OPERATORS_UNREAD = "Your BRAIN operators are not synced yet. Run a catalogue sync (BRAIN › Sync), then try again."
NO_SIMULATIONS = "Assign the simulations for this task."
NO_NEUTRALIZATION = "Choose at least one Neutralization."
SAMPLE_SIZE = 5
POOL_LIMIT = 20_000
#: Data cleaning around every field: days a missing value is carried. Fundamentals are reported a quarter apart,
#: option fields move with the live surface (BRAIN advises "a short ts_backfill of 5 days").
BACKFILL = 21
BACKFILL_BY_CATEGORY = {"fundamental": 63, "analyst": 63, "earnings": 63, "institutions": 63, "insiders": 63,
                        "option": 5}
RISK_NEUTRALIZATIONS = ("REVERSION_AND_MOMENTUM", "STATISTICAL", "CROWDING", "SLOW", "FAST", "SLOW_AND_FAST")
#: The lab's own default when nothing is chosen (only Auto Select relies on it).
DEFAULT_NEUTRALIZATIONS = ("MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY")


async def account_operators() -> list[dict[str, Any]]:
    """The account's operators as synced from BRAIN (`operators` table)."""
    async with db.session() as s:
        rows = list(await s.scalars(select(Operator)))
    return [{"name": o.name, "category": o.category, "scope": o.scope, "definition": o.definition,
             "description": o.description, "level": o.level} for o in rows]


def neutralization_family(value: str) -> str:
    """BRAIN's two families: Risk (a risk model) or Group (a group of instruments, or nothing)."""
    return "risk" if value in RISK_NEUTRALIZATIONS else "group"


def neutralizations_for(region: str, chosen: Iterable[str] = ()) -> list[str]:
    """The neutralizations a lab searches, in BRAIN's own order: the chosen ones BRAIN accepts in this region.

    Never intersected with the lab's default four: that would drop STATISTICAL or CROWDING from a sweep that asked
    for them. Nothing chosen falls back to the default four."""
    offered = list(brain.options.neutralizations(region))
    chosen = list(chosen)
    if chosen:
        wanted = set(chosen)
        return [n for n in offered if n in wanted]
    return [n for n in DEFAULT_NEUTRALIZATIONS if n in offered] or offered[:1]


async def synced_universes(region: str, delay: int, first: str | None = None) -> list[str]:
    """The market's universes whose fields are downloaded and BRAIN offers, the chosen one first."""
    async with db.session() as s:
        synced = set(await s.scalars(select(DataField.universe).where(
            DataField.region == region, DataField.delay == int(delay)).distinct()))
    legal = list(brain.options.universes(region))
    if brain.options.live:
        universes = [u for u in legal if u in synced]
    else:   # before OPTIONS is read the fallback list may not know this market: trust what is downloaded
        universes = [u for u in legal if u in synced] + sorted(u for u in synced if u not in legal)
    if first in universes:
        universes.remove(first)
        universes.insert(0, first)
    return universes


@dataclass(frozen=True)
class Pool:
    #: Field id to MATRIX or VECTOR, across every searched universe, most complete first.
    fields: dict[str, str]
    #: Universes holding at least one field, the chosen market's first.
    universes: tuple[str, ...]
    #: Per universe, the fields it does not have; universes missing none are left out.
    absent: dict[str, list[str]]
    #: Vector fields left out because no vector operator is allowed.
    vector_skipped: int
    #: Fields carried other than BACKFILL days, for their category.
    backfill: dict[str, int] = field(default_factory=dict)


async def field_pool(region: str, delay: int, universes: list[str], dataset_ids: list[str], *, allow_vector: bool,
                     field_types: list[str] | None = None, min_coverage: float | None = None,
                     only_fields: list[str] | None = None) -> Pool:
    """Fields of the chosen datasets, and which searched universes have each.

    BRAIN scopes fields by universe, so a field missing from one universe is not dropped: the search only pairs it
    with the universes that have it. Ordered by coverage in the first universe, so a capped first pass tries the
    most complete first."""
    types = [t for t in ("MATRIX", "VECTOR") if t in (field_types or ("MATRIX", "VECTOR"))]
    if not types or not universes or not dataset_ids:
        return Pool({}, (), {}, 0)
    found: dict[str, dict[str, str]] = {}
    carry: dict[str, int] = {}
    async with db.session() as s:
        for universe in universes:
            stmt = (select(DataField.id, DataField.type, DataField.category_id, DataField.category)
                    .where(DataField.region == region, DataField.delay == int(delay), DataField.universe == universe,
                           DataField.dataset_id.in_(list(dataset_ids)), DataField.type.in_(types))
                    .order_by(func.coalesce(DataField.coverage, 0).desc(), DataField.id).limit(POOL_LIMIT))
            if min_coverage is not None:
                stmt = stmt.where(DataField.coverage >= float(min_coverage))
            if only_fields:          # a hand-off (the assistant, the Data Explorer) narrowed it to these fields
                stmt = stmt.where(DataField.id.in_(list(only_fields)))
            rows = (await s.execute(stmt)).all()
            if rows:
                found[universe] = {str(fid): str(ftype or "MATRIX") for fid, ftype, _, _ in rows}
            for fid, _, cat_id, cat in rows:
                days = BACKFILL_BY_CATEGORY.get(str(cat_id or cat or "").lower())
                if days is not None:
                    carry[str(fid)] = days
    fields: dict[str, str] = {}
    for held in found.values():
        for fid, ftype in held.items():
            fields.setdefault(fid, ftype)
    skipped = 0
    if not allow_vector:
        skipped = sum(1 for t in fields.values() if t == "VECTOR")
        fields = {f: t for f, t in fields.items() if t != "VECTOR"}
    kept = tuple(u for u, held in found.items() if any(f in held for f in fields))
    absent = {u: [f for f in fields if f not in found[u]] for u in kept}
    return Pool(fields, kept, {u: ids for u, ids in absent.items() if ids}, skipped,
                {f: d for f, d in carry.items() if f in fields})


async def universes_holding(region: str, delay: int, names: Iterable[str]) -> dict[str, set[str]]:
    """Per data field, the universes of this market that have it."""
    names = sorted(set(names))
    held: dict[str, set[str]] = {n: set() for n in names}
    if not names:
        return held
    async with db.session() as s:
        for i in range(0, len(names), 400):
            for fid, universe in (await s.execute(select(DataField.id, DataField.universe).where(
                    DataField.region == region, DataField.delay == int(delay),
                    DataField.id.in_(names[i:i + 400])))).all():
                held.setdefault(fid, set()).add(universe)
    return held


async def datasets_of_fields(region: str, delay: int, names: Iterable[str]) -> dict[str, str]:
    """Each field's dataset in this market; a field the market lacks keeps one if every market agrees on it."""
    names = sorted(set(names))
    out: dict[str, str] = {}
    if not names:
        return out
    async with db.session() as s:
        rows = (await s.execute(select(DataField.id, DataField.dataset_id, DataField.region, DataField.delay).where(
            DataField.id.in_(names)))).all()
    everywhere: dict[str, set[str]] = {}
    for fid, ds, reg, dl in rows:
        everywhere.setdefault(fid, set()).add(ds)
        if reg == region and int(dl) == int(delay):
            out.setdefault(fid, ds)
    for fid, sets in everywhere.items():
        if fid not in out and len(sets) == 1:
            out[fid] = next(iter(sets))
    return out


async def dataset_categories(ids: Iterable[str]) -> dict[str, str]:
    ids = sorted(set(ids))
    if not ids:
        return {}
    async with db.session() as s:
        rows = (await s.execute(select(Dataset.id, Dataset.category_id, Dataset.category_name).where(
            Dataset.id.in_(ids)))).all()
    return {d: str(c or n) for d, c, n in rows if c or n}


async def datasets_for(region: str, delay: int) -> list[dict[str, Any]]:
    """The market's datasets for the pickers: id, name, category, subcategory, fields."""
    async with db.session() as s:
        rows = (await s.execute(select(Dataset.id, Dataset.name, Dataset.category_id, Dataset.category_name,
                                       Dataset.subcategory, Dataset.field_count).where(
            Dataset.region == region, Dataset.delay == int(delay)))).all()
        counted = dict((await s.execute(select(DataField.dataset_id, func.count(func.distinct(DataField.id))).where(
            DataField.region == region, DataField.delay == int(delay)).group_by(DataField.dataset_id))).all())
    out: dict[str, dict[str, Any]] = {}
    for did, name, cat_id, cat, sub, n in rows:
        out.setdefault(did, {"id": did, "name": name or did, "category_id": cat_id, "category": cat or cat_id,
                             "subcategory_id": sub, "fields": counted.get(did) or n or 0})
    for did, n in counted.items():
        out.setdefault(did, {"id": did, "name": did, "category_id": None, "category": None, "subcategory_id": None,
                             "fields": n})
    return sorted(out.values(), key=lambda d: d["id"])


def settings_problems(settings: dict[str, Any]) -> list[str]:
    """Why BRAIN would refuse these settings now, checked against the account's OPTIONS /simulations.

    Only once OPTIONS has been read: the fallback lists do not know every market and would refuse good ones."""
    if not brain.options.live:
        return []
    return brain.options.problems(settings)


def preview_samples(draw: Callable[[random.Random], dict[str, Any] | None], *, tries: int) -> list[dict[str, Any]]:
    """Up to SAMPLE_SIZE Alphas a task could send, from at most `tries` draws: {expression, settings}."""
    rng = random.Random()
    sample: list[dict[str, Any]] = []
    for _ in range(tries):
        if len(sample) >= SAMPLE_SIZE:
            break
        made = draw(rng)
        if made is not None:
            sample.append(made)
    return sample


def startup_trials(fields: int, size: int, per_round: int) -> int:
    """Random draws before the search steers: one per field, at most half the task, in whole rounds."""
    cover = min(fields, size // 2)
    return max(per_round, -(-cover // per_round) * per_round)


def field_counts(fields: dict[str, str]) -> dict[str, int]:
    matrix = sum(1 for t in fields.values() if t == "MATRIX")
    return {"total": len(fields), "matrix": matrix, "vector": len(fields) - matrix}
