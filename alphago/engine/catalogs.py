"""Per-scope FASTEXPR catalogs (operators available to the account + data fields of a region/delay/universe)."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from ..db import db
from ..db.models import DataField, Dataset, Operator
from ..fastexpr import Catalog
from ..knowledge.operators import build_catalog

log = logging.getLogger(__name__)

_cache: dict[tuple[str, int, str], Catalog] = {}
_field_dataset: dict[tuple[str, int, str], dict[str, str]] = {}


def invalidate(scope: tuple[str, int, str] | None = None) -> None:
    if scope is None:
        _cache.clear()
        _field_dataset.clear()
    else:
        _cache.pop(scope, None)
        _field_dataset.pop(scope, None)


async def api_operators() -> list[dict[str, Any]]:
    async with db.session() as s:
        rows = list(await s.scalars(select(Operator)))
    return [{"name": r.name, "category": r.category, "scope": r.scope, "definition": r.definition,
             "description": r.description, "documentation": r.documentation, "level": r.level} for r in rows]


async def scope_fields(region: str, delay: int, universe: str) -> list[DataField]:
    async with db.session() as s:
        return list(await s.scalars(select(DataField).where(
            DataField.region == region, DataField.delay == delay, DataField.universe == universe)))


async def get_catalog(region: str, delay: int, universe: str) -> Catalog:
    key = (region, int(delay), universe)
    if key in _cache:
        return _cache[key]
    ops = await api_operators()
    fields = await scope_fields(*key)
    field_dicts = [{"id": f.id, "type": f.type, "dataset": {"id": f.dataset_id}, "category": f.category,
                    "coverage": f.coverage, "alphaCount": f.alpha_count, "description": f.description} for f in fields]
    catalog = build_catalog(ops or None, field_dicts or None)
    _cache[key] = catalog
    _field_dataset[key] = {f.id: f.dataset_id for f in fields}
    log.info("catalog %s: %d operators, %d fields", key, len(catalog.operators), len(field_dicts))
    return catalog


async def field_dataset_map(region: str, delay: int, universe: str) -> dict[str, str]:
    key = (region, int(delay), universe)
    if key not in _field_dataset:
        await get_catalog(*key)
    return _field_dataset.get(key, {})


async def dataset_category_map(region: str, delay: int, universe: str) -> dict[str, tuple[str | None, str | None]]:
    """dataset id -> (category id, category name) for a scope."""
    async with db.session() as s:
        rows = list(await s.scalars(select(Dataset).where(
            Dataset.region == region, Dataset.delay == delay, Dataset.universe == universe)))
    return {d.id: (d.category_id, d.category_name) for d in rows}
