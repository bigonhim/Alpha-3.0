"""Reading the catalogue: filters, facets, stats and field availability (alpha-harness catalog/queries.py).

Every field is stored once per (region, delay, universe), so browsing a market is a filter on one table and
availability is a group-by across markets. All values are bound parameters; the only interpolated identifiers come
from whitelists, because a user-chosen sort key must never reach SQL directly.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from ..brain.platform import FAST_D1_SUFFIX, REGION_AGNOSTIC
from ..db import db
from . import search

#: Sort keys the explorer offers, and the column each one orders by.
SORTABLE = {
    "field_id": "f.id", "dataset_id": "f.dataset_id", "category_id": "f.category_id", "coverage": "f.coverage",
    "date_coverage": "f.date_coverage", "date_created": "f.date_created", "user_count": "f.user_count",
    "alpha_count": "f.alpha_count", "pyramid_multiplier": "f.pyramid_multiplier", "field_type": "f.type",
}
#: Ordering by how well a row answers the search, which is not a column.
RELEVANCE = "relevance"
#: `smart` ranks whole words against the id and the description; `text` is the literal substring match, still the
#: only way to find a fragment like `_eps_` in the middle of an id.
SMART, TEXT = "smart", "text"

COLUMNS = ("f.id AS field_id, f.dataset_id, f.category_id, f.category AS category_name, f.subcategory_id, "
           "f.subcategory AS subcategory_name, f.description, f.type AS field_type, f.coverage, f.date_coverage, "
           "f.user_count, f.alpha_count, f.pyramid_multiplier, f.themes, f.date_created")
NAMED_OR_DESCRIBED = r"(lower(f.id) LIKE :needle ESCAPE '\' OR lower(f.description) LIKE :needle ESCAPE '\')"
FAST_D1_CLAUSE = "f.id LIKE '%" + FAST_D1_SUFFIX.replace("_", "\\_") + "' ESCAPE '\\'"
AGNOSTIC_CLAUSE = "f.id IN (SELECT id FROM datafields WHERE region = :agnostic_region)"
EXCLUSIVE_CLAUSE = "f.id NOT IN (SELECT id FROM datafields WHERE region <> :scope_region)"


@dataclass(frozen=True)
class Scope:
    """A catalogue market."""

    region: str
    delay: int
    universe: str

    WHERE = "f.region = :region AND f.delay = :delay AND f.universe = :universe"

    @property
    def params(self) -> dict[str, Any]:
        return {"region": self.region, "delay": int(self.delay), "universe": self.universe}

    @property
    def label(self) -> str:
        return f"{self.region} D{self.delay} {self.universe}"


def containing(value: str) -> str:
    """A LIKE pattern for `value` anywhere, case-insensitive. `_` and `%` are LIKE wildcards and field ids are full of
    underscores: unescaped, searching `_eps_` would match "steps" too."""
    escaped = value.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class FieldFilter(BaseModel):
    """Everything the Data Explorer can narrow on; every numeric attribute BRAIN returns is filterable."""

    search: str | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    field_ids: list[str] = Field(default_factory=list)
    category_ids: list[str] = Field(default_factory=list)
    field_types: list[str] = Field(default_factory=list)
    coverage_min: float | None = None
    coverage_max: float | None = None
    alpha_count_min: int | None = None
    alpha_count_max: int | None = None
    user_count_min: int | None = None
    user_count_max: int | None = None
    pyramid_multiplier_min: float | None = None
    pyramid_multiplier_max: float | None = None
    date_coverage_min: float | None = None
    date_coverage_max: float | None = None
    date_created_from: date | None = None
    date_created_to: date | None = None
    #: Only fields that also exist in region ALL: the ones an idea could run region-agnostically on.
    region_agnostic: bool = False
    #: Only fields found in no other synced region, region ALL included.
    region_exclusive: bool = False
    #: Only BRAIN's Fast D1 fields (`…_fast_d1`).
    fast_d1: bool = False
    search_mode: str = SMART
    sort_by: str = "alpha_count"
    sort_desc: bool = True
    limit: int = 100
    offset: int = 0

    def terms(self) -> list[str] | None:
        """The search as BM25 words, or None when it is not a ranked search at all."""
        if not self.search or self.search_mode != SMART:
            return None
        return search.query_terms(self.search) or None

    def order_clause(self, *, scored: bool = False) -> str:
        if self.sort_by == RELEVANCE:
            # without a score to order by, fall back to the default column so a stale relevance sort cannot empty
            # the ordering; most fields tie on a count of zero, so the id keeps pages from overlapping
            return "ORDER BY s.score DESC, f.id ASC" if scored else "ORDER BY f.alpha_count DESC NULLS LAST, f.id ASC"
        column = SORTABLE.get(self.sort_by, "f.alpha_count")
        return f"ORDER BY {column} {'DESC' if self.sort_desc else 'ASC'} NULLS LAST, f.id ASC"

    def where(self, scope: Scope, *, search_text: bool = True) -> tuple[str, dict[str, Any]]:
        """Every filter as one clause. `search_text=False` is for a caller that joined the ranked index, which
        already drops the rows that do not match."""
        clauses = [Scope.WHERE]
        params: dict[str, Any] = dict(scope.params)
        if search_text and self.search:
            clauses.append(NAMED_OR_DESCRIBED)
            params["needle"] = containing(self.search)
        for column, values, key in (("f.dataset_id", self.dataset_ids, "ds"), ("f.id", self.field_ids, "fid"),
                                    ("f.category_id", self.category_ids, "cat"), ("f.type", self.field_types, "ty")):
            if values:
                names = [f"{key}{i}" for i in range(len(values))]
                clauses.append(f"{column} IN ({', '.join(':' + n for n in names)})")
                params.update(zip(names, values))
        bounds = (("coverage", self.coverage_min, ">="), ("coverage", self.coverage_max, "<="),
                  ("alpha_count", self.alpha_count_min, ">="), ("alpha_count", self.alpha_count_max, "<="),
                  ("user_count", self.user_count_min, ">="), ("user_count", self.user_count_max, "<="),
                  ("pyramid_multiplier", self.pyramid_multiplier_min, ">="),
                  ("pyramid_multiplier", self.pyramid_multiplier_max, "<="),
                  ("date_coverage", self.date_coverage_min, ">="), ("date_coverage", self.date_coverage_max, "<="),
                  ("date_created", self.date_created_from, ">="), ("date_created", self.date_created_to, "<="))
        for i, (column, value, op) in enumerate(bounds):
            if value is not None:
                clauses.append(f"f.{column} {op} :b{i}")
                params[f"b{i}"] = value.isoformat() if isinstance(value, date) else value
        # by field id, across delays: whether a field exists elsewhere is a question about the field
        if self.region_agnostic and scope.region != REGION_AGNOSTIC:
            clauses.append(AGNOSTIC_CLAUSE)
            params["agnostic_region"] = REGION_AGNOSTIC
        if self.region_exclusive:
            clauses.append(EXCLUSIVE_CLAUSE)
            params["scope_region"] = scope.region
        if self.fast_d1:
            clauses.append(FAST_D1_CLAUSE)
        return " AND ".join(clauses), params


def _row(mapping: Any) -> dict[str, Any]:
    row = dict(mapping)
    themes = row.get("themes")
    if isinstance(themes, str):
        try:
            row["themes"] = json.loads(themes)
        except ValueError:
            pass
    return row


async def _query(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    async with db.session() as s:
        return [_row(m) for m in (await s.execute(text(sql), params or {})).mappings().all()]


async def _scalar(sql: str, params: dict[str, Any] | None = None) -> Any:
    async with db.session() as s:
        return await s.scalar(text(sql), params or {})


def _when(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if isinstance(value, datetime) and value.tzinfo is None else value
    try:
        return datetime.fromisoformat(str(value)).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ---------------------------------------------------------------- overview

async def counts(scope: Scope) -> dict[str, int]:
    """Headline numbers for the Data Explorer's counts strip."""
    rows = await _query(f"""SELECT count(*) AS fields, count(DISTINCT f.dataset_id) AS datasets,
                                   count(DISTINCT f.category_id) AS categories,
                                   count(DISTINCT f.subcategory_id) AS subcategories
                            FROM datafields f WHERE {Scope.WHERE}""", scope.params)
    row = rows[0] if rows else {}
    return {k: int(row.get(k) or 0) for k in ("fields", "datasets", "categories", "subcategories")}


async def synced_tuples() -> list[dict[str, Any]]:
    """Which markets hold fields, how many, and when they last arrived."""
    rows = await _query("""SELECT region, delay, universe, count(*) AS fields, max(synced_at) AS synced_at
                           FROM datafields GROUP BY region, delay, universe ORDER BY region, delay, universe""")
    for r in rows:
        r["synced_at"] = _when(r["synced_at"])
        r["delay"] = int(r["delay"])
    return rows


# ---------------------------------------------------------------- fields

async def _page(scope: Scope, flt: FieldFilter, terms: list[str] | None, *, conjunctive: bool,
                counted: bool = True) -> dict[str, Any]:
    ranked = terms is not None
    where, params = flt.where(scope, search_text=not ranked)
    source = f"datafields f JOIN {search.SCORED} ON s.field_id = f.id" if ranked else "datafields f"
    if ranked:
        params["fts_query"] = search.match_expression(terms, conjunctive=conjunctive)
    total = await _scalar(f"SELECT count(*) FROM {source} WHERE {where}", params) if counted else None
    select = COLUMNS + (", s.score AS score" if ranked else "")
    rows = await _query(f"SELECT {select} FROM {source} WHERE {where} "
                        f"{flt.order_clause(scored=ranked)} LIMIT :lim OFFSET :off",
                        {**params, "lim": int(flt.limit), "off": int(flt.offset)})
    return {"total": int(total or 0) if counted else None, "limit": flt.limit, "offset": flt.offset,
            "results": rows, "conjunctive": conjunctive, "ranked": ranked}


async def fields(scope: Scope, flt: FieldFilter, *, counted: bool = True) -> dict[str, Any]:
    """A filtered, sorted page of fields plus the total. A smart search first requires every word (what makes a
    multi-word search mean something); only when that finds nothing does it take any of the words."""
    terms = flt.terms()
    if not (terms and await search.ready()):
        return await _page(scope, flt, None, conjunctive=True, counted=counted)
    page = await _page(scope, flt, terms, conjunctive=True, counted=counted)
    found = page["total"] if counted else len(page["results"])
    if found == 0 and len(terms) > 1:
        return await _page(scope, flt, terms, conjunctive=False, counted=counted)
    return page


async def _all_words_match(scope: Scope, flt: FieldFilter, terms: list[str] | None) -> bool:
    """Whether requiring every word still finds something: the facets must make the table's decision."""
    if not terms or len(terms) < 2:
        return True
    where, params = flt.where(scope, search_text=False)
    params["fts_query"] = search.match_expression(terms, conjunctive=True)
    return bool(await _scalar(f"SELECT 1 FROM datafields f JOIN {search.SCORED} ON s.field_id = f.id "
                              f"WHERE {where} LIMIT 1", params))


async def field(scope: Scope, field_id: str) -> dict[str, Any] | None:
    rows = await _query(f"SELECT {COLUMNS}, f.region, f.delay, f.universe, f.synced_at, f.region_coverage "
                        f"FROM datafields f WHERE {Scope.WHERE} AND f.id = :fid", {**scope.params, "fid": field_id})
    return rows[0] if rows else None


async def field_anywhere(field_id: str) -> dict[str, Any] | None:
    """One field from whichever market covers it best: what a field is does not change between markets."""
    rows = await _query(f"SELECT {COLUMNS}, f.region, f.delay, f.universe FROM datafields f WHERE f.id = :fid "
                        "ORDER BY f.coverage DESC NULLS LAST, f.region, f.delay, f.universe LIMIT 1",
                        {"fid": field_id})
    return rows[0] if rows else None


async def fields_anywhere(text_: str, limit: int = 40) -> list[dict[str, Any]]:
    """Fields named or described by `text_` in any market, each once from the market that covers it best; the most
    used first."""
    return await _query(f"""SELECT * FROM (
                                SELECT {COLUMNS}, f.region, f.delay, f.universe, row_number() OVER (
                                    PARTITION BY f.id ORDER BY f.coverage DESC NULLS LAST, f.region, f.delay,
                                    f.universe) AS rn
                                FROM datafields f WHERE {NAMED_OR_DESCRIBED})
                            WHERE rn = 1 ORDER BY alpha_count DESC NULLS LAST, field_id LIMIT :lim""",
                        {"needle": containing(text_), "lim": int(limit)})


async def field_availability(field_id: str) -> list[dict[str, Any]]:
    """Every market this field exists in: a field in USA D1 may simply not exist in EUR D0, and simulating it there
    fails."""
    return await _query("""SELECT region, delay, universe, coverage, date_coverage, alpha_count, user_count,
                                  type AS field_type
                           FROM datafields WHERE id = :fid ORDER BY region, delay, universe""", {"fid": field_id})


async def fields_by_id(scope: Scope, ids: list[str], *, whole_datasets: bool = False) -> list[dict[str, Any]]:
    """The named fields (or every field of the named datasets) in `scope`, with their dataset's name and
    description. Unknown ids are left out."""
    if not ids:
        return []
    names = {f"i{n}": v for n, v in enumerate(ids)}
    column = "f.dataset_id" if whole_datasets else "f.id"
    return await _query(f"""SELECT f.id AS field_id, f.description, f.type AS field_type, f.coverage,
                                   f.date_coverage, f.category AS category_name, f.subcategory AS subcategory_name,
                                   f.dataset_id, d.name AS dataset_name, d.description AS dataset_description
                            FROM datafields f
                            LEFT JOIN datasets d ON d.id = f.dataset_id AND d.region = f.region
                                 AND d.delay = f.delay AND d.universe = f.universe
                            WHERE {Scope.WHERE} AND {column} IN ({', '.join(':' + k for k in names)})""",
                        {**scope.params, **names})


async def datasets(scope: Scope) -> list[dict[str, Any]]:
    return await _query("""SELECT id AS dataset_id, name, description, category_id, category_name,
                                  subcategory AS subcategory_name, coverage, value_score, user_count, alpha_count,
                                  field_count, pyramid_multiplier
                           FROM datasets WHERE region = :region AND delay = :delay AND universe = :universe
                           ORDER BY value_score DESC NULLS LAST, id""", scope.params)


# ---------------------------------------------------------------- facets and stats

async def facets(scope: Scope, flt: FieldFilter | None = None) -> dict[str, Any]:
    """Distinct values with counts under the other active filters. A facet ignores its own selection, and a level
    of the Category → Subcategory → Dataset hierarchy also ignores the levels below it, so choosing MATRIX still
    shows how many VECTOR fields the rest of the filter would match."""
    active = flt or FieldFilter()
    terms = active.terms()
    ranked = bool(terms) and await search.ready()
    conjunctive = await _all_words_match(scope, active, terms) if ranked else True
    source = f"datafields f JOIN {search.SCORED} ON s.field_id = f.id" if ranked else "datafields f"
    head = {"fts_query": search.match_expression(terms, conjunctive=conjunctive)} if ranked and terms else {}

    async def group(column: str, ignore: tuple[str, ...], *extra: tuple[str, str]) -> list[dict[str, Any]]:
        narrowed = active.model_copy(update={name: [] for name in ignore})
        where, params = narrowed.where(scope, search_text=not ranked)
        more = "".join(f", max({src}) AS {alias}" for src, alias in extra)
        return await _query(f"SELECT {column} AS id{more}, count(*) AS n FROM {source} "
                            f"WHERE {where} AND {column} IS NOT NULL GROUP BY {column} ORDER BY n DESC, id",
                            {**head, **params})

    categories, subcategories, sets, types, availability, months = await asyncio.gather(
        group("f.category_id", ("category_ids", "dataset_ids"), ("f.category", "name")),
        group("f.subcategory_id", ("dataset_ids",), ("f.subcategory", "name"), ("f.category_id", "category_id")),
        group("f.dataset_id", ("dataset_ids",), ("f.category_id", "category_id"),
              ("f.subcategory_id", "subcategory_id")),
        group("f.type", ("field_types",)),
        _availability(scope, active, source, head, ranked),
        _months(scope, active, source, head, ranked),
    )
    return {"categories": categories, "subcategories": subcategories, "datasets": sets, "types": types,
            "availability": availability, "date_added": months}


async def _months(scope: Scope, active: FieldFilter, source: str, head: dict[str, Any],
                  ranked: bool) -> list[dict[str, Any]]:
    """Fields per month added under every other filter (the month range is left out, like a chip ignores its own
    selection)."""
    where, params = active.model_copy(update={"date_created_from": None, "date_created_to": None}).where(
        scope, search_text=not ranked)
    return await _query(f"SELECT f.date_created AS month, count(*) AS fields FROM {source} "
                        f"WHERE {where} AND f.date_created IS NOT NULL GROUP BY 1 ORDER BY 1", {**head, **params})


async def _availability(scope: Scope, active: FieldFilter, source: str, head: dict[str, Any],
                        ranked: bool) -> dict[str, int]:
    """How many fields each availability toggle would show, pressed on its own, everything else as filtered.
    Region Agnostic and Region Exclusive cannot both hold (a field in region ALL is in another region), so pressing
    one lets go of the other; Fast D1 combines with either."""
    base = active.model_copy(update={"region_agnostic": False, "region_exclusive": False, "fast_d1": False})
    where, params = base.where(scope, search_text=not ranked)
    agnostic = "1" if scope.region == REGION_AGNOSTIC else AGNOSTIC_CLAUSE
    fast = FAST_D1_CLAUSE if active.fast_d1 else "1"
    regional = (agnostic if active.region_agnostic else EXCLUSIVE_CLAUSE if active.region_exclusive else "1")
    rows = await _query(f"""SELECT sum(CASE WHEN {agnostic} AND {fast} THEN 1 ELSE 0 END) AS region_agnostic,
                                   sum(CASE WHEN {EXCLUSIVE_CLAUSE} AND {fast} THEN 1 ELSE 0 END) AS region_exclusive,
                                   sum(CASE WHEN {FAST_D1_CLAUSE} AND {regional} THEN 1 ELSE 0 END) AS fast_d1
                            FROM {source} WHERE {where}""",
                        {**head, **params, "agnostic_region": REGION_AGNOSTIC, "scope_region": scope.region})
    row = rows[0] if rows else {}
    return {k: int(row.get(k) or 0) for k in ("region_agnostic", "region_exclusive", "fast_d1")}


async def stats(scope: Scope) -> dict[str, Any]:
    """Value ranges, so the filter controls bound themselves to real data."""
    rows = await _query(f"""SELECT min(f.coverage) AS coverage_min, max(f.coverage) AS coverage_max,
                                   min(f.alpha_count) AS alpha_count_min, max(f.alpha_count) AS alpha_count_max,
                                   min(f.user_count) AS user_count_min, max(f.user_count) AS user_count_max,
                                   min(f.pyramid_multiplier) AS pyramid_multiplier_min,
                                   max(f.pyramid_multiplier) AS pyramid_multiplier_max,
                                   min(f.date_coverage) AS date_coverage_min, max(f.date_coverage) AS date_coverage_max,
                                   count(f.coverage) AS n
                            FROM datafields f WHERE {Scope.WHERE}""", scope.params)
    out = rows[0] if rows else {}
    n = int(out.pop("n", 0) or 0)
    out["coverage_median"] = None
    if n:
        upper = await _scalar(f"SELECT f.coverage FROM datafields f WHERE {Scope.WHERE} AND f.coverage IS NOT NULL "
                              "ORDER BY f.coverage LIMIT 1 OFFSET :o", {**scope.params, "o": n // 2})
        lower = upper if n % 2 else await _scalar(
            f"SELECT f.coverage FROM datafields f WHERE {Scope.WHERE} AND f.coverage IS NOT NULL "
            "ORDER BY f.coverage LIMIT 1 OFFSET :o", {**scope.params, "o": n // 2 - 1})
        out["coverage_median"] = (float(upper) + float(lower)) / 2
    # BRAIN dates a field by month (always the 1st), so the months are the choices
    out["date_added"] = await _query(f"SELECT f.date_created AS month, count(*) AS fields FROM datafields f "
                                     f"WHERE {Scope.WHERE} AND f.date_created IS NOT NULL GROUP BY 1 ORDER BY 1",
                                     scope.params)
    return out


async def synced_categories() -> set[tuple[str, str, int]]:
    """(category, region, delay) triples the catalogue holds fields for: the pyramids that can be researched here."""
    rows = await _query("SELECT DISTINCT category_id, region, delay FROM datafields WHERE category_id IS NOT NULL")
    return {(str(r["category_id"]), str(r["region"]), int(r["delay"])) for r in rows}


async def best_universes() -> dict[tuple[str, int], str]:
    """For each (region, delay), the synced universe with the most fields: where a pyramid opens in the explorer."""
    best: dict[tuple[str, int], tuple[str, int]] = {}
    for r in await synced_tuples():
        key = (r["region"], r["delay"])
        if key not in best or r["fields"] > best[key][1]:
            best[key] = (r["universe"], r["fields"])
    return {k: v[0] for k, v in best.items()}
