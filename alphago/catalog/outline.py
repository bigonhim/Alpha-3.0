"""The chosen fields as compact Markdown for an LLM (alpha-harness api/catalog.py `/fields/outline`).

Category, Subcategory and Dataset headings are each written once, then one short key-value block per field. The
universe is left out: a field is the same field in every universe of its region. Categories and subcategories carry
only a name, because BRAIN sends no description for either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import queries
from .queries import Scope

#: Enough for a focused idea, and short enough to paste whole into any chat LLM.
MAX_OUTLINE_FIELDS = 100
#: Every dataset in the largest category, with room to spare.
MAX_OUTLINE_DATASETS = 1000


@dataclass
class FieldOutline:
    text: str
    #: How many fields `text` holds.
    fields: int
    #: Chosen ids this market does not carry; they are left out of `text`.
    missing: list[str] = field(default_factory=list)


def _flat(value: Any) -> str:
    """One line: a description's line breaks would read as new keys."""
    return " ".join(str(value or "").split())


def _pct(value: Any) -> str:
    return "unknown" if value is None else f"{float(value) * 100:.0f}%"


def outline(scope: Scope, rows: list[dict[str, Any]]) -> str:
    tree: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for r in rows:
        category = _flat(r.get("category_name")) or "Uncategorised"
        subcategory = _flat(r.get("subcategory_name")) or "Uncategorised"
        tree.setdefault(category, {}).setdefault(subcategory, {}).setdefault(str(r["dataset_id"]), []).append(r)
    lines = ["# Data Fields", f"Region: {scope.region}", f"Delay: {scope.delay}"]

    def heading(value: str) -> None:
        # a blank line only after content: stacked headings read as one path
        if not lines[-1].startswith("#"):
            lines.append("")
        lines.append(value)

    for category, subcategories in sorted(tree.items()):
        heading(f"## Category: {category}")
        for subcategory, sets in sorted(subcategories.items()):
            heading(f"### Subcategory: {subcategory}")
            for dataset_id, items in sorted(sets.items()):
                heading(f"#### Dataset: {dataset_id}")
                name = _flat(items[0].get("dataset_name"))
                if name and name != dataset_id:
                    lines.append(f"Name: {name}")
                if about := _flat(items[0].get("dataset_description")):
                    lines.append(f"Description: {about}")
                lines.append("")
                for f in sorted(items, key=lambda f: str(f["field_id"])):
                    lines += [
                        f"- Field: {f['field_id']}",
                        f"  Description: {_flat(f.get('description')) or 'none'}",
                        f"  Type: {f.get('field_type') or 'unknown'}",
                        f"  Instrument Coverage: {_pct(f.get('coverage'))}",
                        f"  Date Coverage: {_pct(f.get('date_coverage'))}",
                    ]
    return "\n".join(lines)


async def field_outline(scope: Scope, *, field_ids: list[str] | None = None,
                        dataset_ids: list[str] | None = None) -> FieldOutline:
    """Fields picked one by one, or whole datasets (a category or subcategory is its datasets) — one kind at a time."""
    field_ids, dataset_ids = list(field_ids or []), list(dataset_ids or [])
    if bool(field_ids) == bool(dataset_ids):
        raise ValueError("Choose some fields or some datasets (not both) to copy.")
    if len(field_ids) > MAX_OUTLINE_FIELDS:
        raise ValueError(f"Select at most {MAX_OUTLINE_FIELDS} fields at once.")
    if len(dataset_ids) > MAX_OUTLINE_DATASETS:
        raise ValueError(f"Select at most {MAX_OUTLINE_DATASETS} datasets at once.")
    by_dataset = bool(dataset_ids)
    ids = dataset_ids if by_dataset else field_ids
    rows = await queries.fields_by_id(scope, ids, whole_datasets=by_dataset)
    found = {r["dataset_id" if by_dataset else "field_id"] for r in rows}
    return FieldOutline(text=outline(scope, rows), fields=len(rows), missing=[i for i in ids if i not in found])
