"""The Data Explorer's dataset filter as a tree: Category → Subcategory → Dataset (alpha-harness dataset-tree.ts).

Built from the facets, so every dataset sits exactly once under its own category, largest first. The choice itself is
always a list of dataset ids: ticking a category or subcategory ticks its datasets (the tree draws the tri-state).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Branch:
    key: str
    id: str
    name: str
    ids: list[str] = field(default_factory=list)
    n: int = 0


@dataclass
class Trunk(Branch):
    subcategories: list[Branch] = field(default_factory=list)
    #: Datasets without a subcategory.
    loose: list[str] = field(default_factory=list)


def build_tree(facets: dict[str, Any]) -> list[Trunk]:
    trunks: dict[str, Trunk] = {}

    def trunk_of(cid: str | None, name: str | None = None) -> Trunk:
        key = cid or ""
        if key not in trunks:
            trunks[key] = Trunk(key=key, id=key, name=name or cid or "Uncategorised")
        return trunks[key]

    for c in facets.get("categories") or []:
        trunk_of(c.get("id"), c.get("name"))
    sub_names = {s["id"]: s.get("name") for s in facets.get("subcategories") or []}
    sub_rank = {s["id"]: i for i, s in enumerate(facets.get("subcategories") or [])}
    branches: dict[str, Branch] = {}
    for d in facets.get("datasets") or []:
        trunk = trunk_of(d.get("category_id"))
        trunk.ids.append(d["id"])
        trunk.n += int(d.get("n") or 0)
        sub = d.get("subcategory_id")
        if not sub:
            trunk.loose.append(d["id"])
            continue
        key = f"{trunk.key}|{sub}"
        if key not in branches:
            branches[key] = Branch(key=key, id=sub, name=sub_names.get(sub) or sub)
            trunk.subcategories.append(branches[key])
        branches[key].ids.append(d["id"])
        branches[key].n += int(d.get("n") or 0)
    for trunk in trunks.values():
        trunk.subcategories.sort(key=lambda b: sub_rank.get(b.id, 1 << 30))
    return [t for t in trunks.values() if t.ids]


def dataset_names(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Dataset ids to the names people read, each with its id: BRAIN gives some datasets the very same name."""
    return {r["dataset_id"]: (f"{r['name']} ({r['dataset_id']})" if r.get("name") and r["name"] != r["dataset_id"]
                              else r["dataset_id"]) for r in rows}


def tree_nodes(tree: list[Trunk], names: dict[str, str], counts: dict[str, int]) -> list[dict[str, Any]]:
    """Quasar tree nodes; leaves are `d:<dataset id>`, so the ticked leaves are the chosen datasets."""
    def leaf(ds: str) -> dict[str, Any]:
        return {"id": f"d:{ds}", "label": f"{names.get(ds, ds)} · {counts.get(ds, 0):,}"}

    nodes = []
    for t in tree:
        children = [{"id": f"s:{b.key}", "label": f"{b.name} · {b.n:,}", "children": [leaf(d) for d in b.ids]}
                    for b in t.subcategories]
        children += [leaf(d) for d in t.loose]
        nodes.append({"id": f"c:{t.key}", "label": f"{t.name} · {t.n:,}", "children": children})
    return nodes


def ticked_datasets(ticked: list[str]) -> list[str]:
    return [t[2:] for t in ticked if t.startswith("d:")]
