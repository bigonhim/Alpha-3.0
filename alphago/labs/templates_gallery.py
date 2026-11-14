"""Template Lab's built-in gallery and the consultant's saved templates (alpha-harness template/gallery.tsx and the
template routes of api/template_lab.py).

A built-in family is an idea, written with its lookback as ``d`` and its signals as ``...``, and complete templates
built on it. Saved templates live in `lab_templates`; names are unique whatever their case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select

from ..db import db
from ..db.models_labs import LabTemplate
from . import template

MAX_NAME = 128
MAX_DESCRIPTION = 500


@dataclass(frozen=True)
class Family:
    """A built-in template: an idea, and complete templates built on it."""

    shown: str                      # the idea as the gallery shows it
    text: str                       # what opening the idea writes into the editor; each ... is a signal to fill
    variations: list[str] = field(default_factory=list)
    title: str | None = None


GALLERY: list[Family] = [
    Family(
        shown="if_else(abs(ts_arg_min(low, d) - ts_arg_max(high, d)) > d / 2, ..., ...)",
        text="if_else(abs(ts_arg_min(low, $lookback) - ts_arg_max(high, $lookback)) > $lookback / 2, ..., ...)",
        variations=[
            "r = rank($field);\nif_else(abs(ts_arg_min(low, $lookback) - ts_arg_max(high, $lookback)) > "
            "$lookback / 2, r, 1 - r)",
            "rank(if_else(abs(ts_arg_min(low, $lookback) - ts_arg_max(high, $lookback)) > $lookback / 2, "
            "$field, -$field))",
        ],
    ),
    # Named after where their fields come from, so each variable starts with its own datasets.
    Family(title="Two fields, two datasets", shown="rank($fundamental / $pv1)", text="rank($fundamental / $pv1)",
           variations=["group_rank($fundamental / $pv1, $group)"]),
    Family(title="Try operators", shown="$ts_op($field, $lookback)", text="$ts_op($field, $lookback)",
           variations=["$group_op($ts_op($field, $lookback), $group)"]),
]


def _clean(variables: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Variables as stored: every default filled in, so a fields variable always has its dataset_ids."""
    out: dict[str, dict[str, Any]] = {}
    for name, v in (variables or {}).items():
        if not isinstance(v, dict):
            continue
        if v.get("kind") == "values":
            out[name] = {"kind": "values", "values": str(v.get("values") or "")[:4000]}
        elif v.get("kind") == "fields":
            out[name] = {"kind": "fields", "dataset_ids": list(v.get("dataset_ids") or [])[:500],
                         "field_types": [t for t in v.get("field_types") or [] if t in ("MATRIX", "VECTOR")],
                         "vector_operators": list(v.get("vector_operators") or [])[:20]}
            if v.get("min_coverage") is not None:
                out[name]["min_coverage"] = float(v["min_coverage"])
    return out


def _row(t: LabTemplate) -> dict[str, Any]:
    text, variables = t.text or "", dict(t.variables or {})
    if not variables and ("{" in text or "FIELD" in text or "LOOKBACK" in text) and "$" not in text:
        text, variables = template.from_blocks(text)         # saved while templates were built from blocks
    return {"id": t.id, "name": t.name, "description": t.description, "text": text, "variables": _clean(variables),
            "updated_at": t.updated_at.isoformat() if t.updated_at else None}


def _check(name: str, text: str, description: str | None) -> tuple[str, str | None]:
    name = (name or "").strip()
    if not name:
        raise ValueError("Give the template a name.")
    if len(name) > MAX_NAME:
        raise ValueError(f"A name holds at most {MAX_NAME} characters.")
    if len(text or "") > template.MAX_TEXT:
        raise ValueError(f"A template holds at most {template.MAX_TEXT:,} characters.")
    description = (description or "").strip() or None
    if description and len(description) > MAX_DESCRIPTION:
        raise ValueError(f"A description holds at most {MAX_DESCRIPTION} characters.")
    return name, description


async def saved() -> list[dict[str, Any]]:
    """The saved templates, newest first."""
    async with db.session() as s:
        rows = list(await s.scalars(select(LabTemplate).order_by(LabTemplate.updated_at.desc(), LabTemplate.id.desc())))
    return [_row(r) for r in rows]


async def _name_free(s: Any, name: str, template_id: int | None = None) -> None:
    clash = await s.scalar(select(LabTemplate.id).where(LabTemplate.name_key == name.lower()))
    if clash is not None and clash != template_id:
        raise ValueError(f"A template named {name} already exists.")


async def create(name: str, text: str, variables: dict[str, Any] | None = None,
                 description: str | None = None) -> dict[str, Any]:
    name, description = _check(name, text, description)
    async with db.write() as s:
        await _name_free(s, name)
        row = LabTemplate(name=name, name_key=name.lower(), description=description, text=text or "",
                          variables=_clean(variables))
        s.add(row)
        await s.flush()
        out = _row(row)
    return out


async def update(template_id: int, name: str, text: str, variables: dict[str, Any] | None = None,
                 description: str | None = None) -> dict[str, Any]:
    name, description = _check(name, text, description)
    async with db.write() as s:
        row = await s.get(LabTemplate, template_id)
        if row is None:
            raise ValueError("That template no longer exists.")
        await _name_free(s, name, template_id)
        row.name, row.name_key, row.description = name, name.lower(), description
        row.text, row.variables = text or "", _clean(variables)
        await s.flush()
        out = _row(row)
    return out


async def delete(template_id: int) -> None:
    async with db.write() as s:
        row = await s.get(LabTemplate, template_id)
        if row is None:
            raise ValueError("That template no longer exists.")
        await s.delete(row)


def gallery_items() -> list[dict[str, Any]]:
    """Every built-in card: the idea and each variation, with the key its stats are batched under."""
    out = []
    for i, f in enumerate(GALLERY):
        out.append({"key": f"gallery:{i}:idea", "family": i, "title": f.title, "shown": f.shown, "text": f.text,
                    "variations": list(f.variations)})
    return out
