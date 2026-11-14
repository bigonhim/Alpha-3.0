"""Data Explorer (ported from alpha-harness screens/data): every data field of every synced market.

Search (smart BM25 ranking over the id's words and the description, or a literal substring), filter by dataset tree,
category, type, every numeric attribute and availability across regions, compare a field across markets, and copy the
chosen fields or datasets as a Markdown outline for an LLM. The filter survives leaving the page (kept in
`pref:data_explorer`), and other pages open a market here with `open_market`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

from nicegui import ui

from ...brain import brain
from ...brain.platform import FAST_D1_MARKET, REGION_AGNOSTIC
from ...catalog import outline as outline_mod
from ...catalog import queries, search
from ...catalog.queries import RELEVANCE, SMART, TEXT, FieldFilter, Scope
from ...catalog.tree import build_tree, dataset_names, ticked_datasets, tree_nodes
from ...db import db
from ..components import (
    ago,
    card,
    empty_state,
    fmt_int,
    fmt_num,
    fmt_pct,
    grid_options,
    notify_error,
    num_col,
    page_header,
    segmented,
)

log = logging.getLogger(__name__)
PREF = "pref:data_explorer"
PAGE_SIZE = 100
SORTS = {RELEVANCE: "Best match", "alpha_count": "Alphas", "user_count": "Users", "coverage": "Instrument Coverage",
         "date_coverage": "Date Coverage", "pyramid_multiplier": "Pyramid Multiplier", "date_created": "Date Added",
         "field_id": "Field", "dataset_id": "Dataset", "field_type": "Type"}
COLUMNS = [
    {"field": "dataset_id", "headerName": "Dataset", "width": 120, "sortable": False},
    {"field": "field_id", "headerName": "Field", "width": 230, "sortable": False, "cellClass": "ag-mono ag-cyan"},
    {"field": "description", "headerName": "Description", "flex": 1, "minWidth": 260, "sortable": False,
     "tooltipField": "description"},
    {"field": "category_name", "headerName": "Category", "width": 140, "sortable": False},
    {"field": "field_type", "headerName": "Type", "width": 90, "sortable": False},
    {**num_col("pyramid_multiplier", "Pyramid Multiplier", 1, 120), "sortable": False},
    {**num_col("coverage", "Instrument Coverage", 0, 120, pct=True), "sortable": False},
    {**num_col("date_coverage", "Date Coverage", 0, 110, pct=True), "sortable": False},
    {**num_col("user_count", "Users", 0, 80), "sortable": False},
    {**num_col("alpha_count", "Alphas", 0, 80), "sortable": False},
    {"field": "date_created", "headerName": "Date Added", "width": 110, "sortable": False},
]


async def open_market(region: str, delay: int, universe: str, *, category_ids: list[str] | None = None) -> None:
    """Open a market in the Data Explorer; with `category_ids` the filter is replaced by just that category (a
    category carried in beside a stale dataset or coverage filter lands on an empty table and no clue why)."""
    pref = dict(await db.kv_get(PREF, {}) or {})
    pref["scope"] = {"region": region, "delay": int(delay), "universe": universe}
    if category_ids is not None:
        pref["filter"] = {"category_ids": list(category_ids)}
        pref["sort"] = {"sort_by": "alpha_count", "sort_desc": True}
    await db.kv_set(PREF, pref)
    ui.navigate.to("/data")


def _default_scope(synced: list[dict[str, Any]]) -> dict[str, Any]:
    if any(r["region"] == "USA" and r["delay"] == 1 and r["universe"] == "TOP3000" for r in synced) or not synced:
        return {"region": "USA", "delay": 1, "universe": "TOP3000"}
    best = max(synced, key=lambda r: r["fields"])
    return {"region": best["region"], "delay": best["delay"], "universe": best["universe"]}


async def field_dialog(scope: Scope, field_id: str) -> None:
    """One field: what it is, and every market it exists in (simulating it where it does not exist fails)."""
    detail = await queries.field(scope, field_id) or await queries.field_anywhere(field_id)
    markets = await queries.field_availability(field_id)
    with ui.dialog() as dialog, ui.card().classes("ag-card p-6 w-[860px] max-w-[95vw]"):
        with ui.row().classes("items-center w-full"):
            ui.label(field_id).classes("text-xl ag-mono ag-cyan")
            ui.space()
            ui.button(icon="content_copy", on_click=lambda: (ui.clipboard.write(field_id),
                                                             ui.notify(f"Copied {field_id}"))).props("flat dense")
        if detail is None:
            ui.label("This field is not in the catalogue any more. Sync again from Sync with BRAIN.").classes(
                "ag-muted")
        else:
            ui.label(detail.get("description") or "No description from BRAIN.").classes("text-sm")
            with ui.row().classes("gap-6 flex-wrap text-sm mt-2"):
                for label, value in (("Dataset", detail.get("dataset_id")), ("Category", detail.get("category_name")),
                                     ("Subcategory", detail.get("subcategory_name")),
                                     ("Type", detail.get("field_type")),
                                     ("Instrument Coverage", fmt_pct(detail.get("coverage"), 0)),
                                     ("Date Coverage", fmt_pct(detail.get("date_coverage"), 0)),
                                     ("Users", fmt_int(detail.get("user_count"))),
                                     ("Alphas", fmt_int(detail.get("alpha_count"))),
                                     ("Pyramid Multiplier", fmt_num(detail.get("pyramid_multiplier"), 1)),
                                     ("Date Added", detail.get("date_created") or "—")):
                    with ui.column().classes("gap-0"):
                        ui.label(label).classes("ag-muted text-xs")
                        ui.label(str(value or "—")).classes("ag-mono")
        ui.label(f"Availability: {len(markets)} market(s)").classes("ag-section-title mt-4")
        ui.aggrid(grid_options([
            {"field": "region", "headerName": "Region", "width": 90},
            {"field": "delay", "headerName": "Delay", "width": 80},
            {"field": "universe", "headerName": "Universe", "width": 150},
            num_col("coverage", "Coverage", 0, 110, pct=True),
            num_col("date_coverage", "Date Coverage", 0, 120, pct=True),
            num_col("alpha_count", "Alpha Count", 0, 110),
            {"field": "field_type", "headerName": "Type", "width": 100},
        ], markets, pagination=False), theme="quartz").classes("w-full h-[320px]")
        ui.button("Close", on_click=dialog.close).props("flat").classes("self-end")
    dialog.open()


async def page(q: str | None = None) -> None:
    """`/data?q=<words>` opens with that search (the command palette links here)."""
    page_header("Data Explorer", "Every data field of every market you synced: search, filter, compare availability "
                                 "and copy outlines.")
    synced = await queries.synced_tuples()
    pref = dict(await db.kv_get(PREF, {}) or {})
    scope_d = pref.get("scope") or _default_scope(synced)
    try:
        flt = FieldFilter(**(pref.get("filter") or {}))
    except Exception:
        flt = FieldFilter()
    sort = pref.get("sort") or {"sort_by": "alpha_count", "sort_desc": True}
    flt.sort_by, flt.sort_desc = sort.get("sort_by", "alpha_count"), bool(sort.get("sort_desc", True))
    if q and q.strip():
        flt.search = q.strip()[:200]
    state: dict[str, Any] = {"scope": Scope(scope_d["region"], int(scope_d["delay"]), scope_d["universe"]),
                             "filter": flt, "offset": 0, "token": 0, "rows": [], "chosen_sort": "sort" in pref}

    # ------------------------------------------------------------ market
    regions = sorted({*brain.options.regions, *(r["region"] for r in synced)})
    with card("w-full"):
        with ui.row().classes("items-end gap-3 w-full flex-wrap"):
            region = ui.select(regions, value=state["scope"].region, label="Region").classes("w-32")
            delay = ui.select([0, 1], value=state["scope"].delay, label="Delay").classes("w-24")
            universe = ui.select([state["scope"].universe], value=state["scope"].universe,
                                 label="Universe").classes("w-48")
            synced_note = ui.label().classes("ag-muted text-sm")
            ui.space()
            anywhere = ui.input("Search all markets", placeholder="Field id or description").props(
                "dense clearable").classes("w-72")
            ui.button("Find", icon="travel_explore", on_click=lambda: search_everywhere()).props("flat")
        counts_row = ui.row().classes("gap-3 w-full mt-2")

    # ------------------------------------------------------------ filters + results
    with ui.row().classes("w-full gap-5 items-start no-wrap"):
        with card("w-[360px] min-w-[320px] gap-3"):
            ui.label("Filters").classes("ag-section-title")
            box = ui.input("Search Field ID or Description", value=flt.search or "").props("dense clearable") \
                .classes("w-full")
            mode = segmented({SMART: "Smart", TEXT: "Exact text"}, flt.search_mode)
            mode.tooltip("Smart ranks whole words against the id and the description (stemmed: 'revision' finds "
                         "'revisions'). Exact text finds a fragment like _eps_ anywhere in an id.")
            types = ui.select({}, multiple=True, label="Field type", value=list(flt.field_types)).props(
                "dense use-chips").classes("w-full")
            cats = ui.select({}, multiple=True, label="Category", value=list(flt.category_ids)).props(
                "dense use-chips").classes("w-full")
            with ui.row().classes("gap-2 flex-wrap"):
                t_agn = ui.button().props("dense no-caps")
                t_exc = ui.button().props("dense no-caps")
                t_fast = ui.button().props("dense no-caps")
            ui.label("Category → Subcategory → Dataset").classes("text-sm mt-1")
            tree_box = ui.column().classes("w-full max-h-[340px] overflow-auto gap-0")
            ranges: dict[str, tuple[ui.number, ui.number]] = {}
            for key, label in (("pyramid_multiplier", "Pyramid Multiplier"), ("coverage", "Instrument Coverage (%)"),
                               ("date_coverage", "Date Coverage (%)"), ("user_count", "Users"),
                               ("alpha_count", "Alphas")):
                ui.label(label).classes("text-xs ag-muted mt-1")
                with ui.row().classes("gap-2 no-wrap w-full"):
                    pct = key in ("coverage", "date_coverage")
                    lo_v, hi_v = getattr(flt, f"{key}_min"), getattr(flt, f"{key}_max")
                    lo = ui.number(placeholder="min", value=lo_v * 100 if pct and lo_v is not None else lo_v) \
                        .props("dense").classes("w-[48%]")
                    hi = ui.number(placeholder="max", value=hi_v * 100 if pct and hi_v is not None else hi_v) \
                        .props("dense").classes("w-[48%]")
                    ranges[key] = (lo, hi)
            ui.label("Date Added").classes("text-xs ag-muted mt-1")
            with ui.row().classes("gap-2 no-wrap w-full"):
                month_from = ui.select({}, label="from", clearable=True,
                                       value=flt.date_created_from.isoformat() if flt.date_created_from else None) \
                    .props("dense").classes("w-[48%]")
                month_to = ui.select({}, label="to", clearable=True,
                                     value=flt.date_created_to.isoformat() if flt.date_created_to else None) \
                    .props("dense").classes("w-[48%]")
            ui.button("Clear filters", icon="filter_alt_off", on_click=lambda: clear()).props("flat")

        with card("grow min-w-0 gap-2"):
            with ui.row().classes("items-center w-full gap-3 flex-wrap"):
                ui.label("Fields").classes("ag-section-title")
                total_label = ui.label().classes("ag-muted text-sm")
                ui.space()
                sort_sel = ui.select(SORTS, value=flt.sort_by, label="Sort").props("dense").classes("w-48")
                desc = ui.switch("Descending", value=flt.sort_desc)
            with ui.row().classes("items-center w-full gap-2 flex-wrap"):
                ui.button("Copy Data Fields", icon="content_copy", on_click=lambda: copy_outline(False)).props(
                    "unelevated no-caps").tooltip("The selected fields (or, with none selected, the datasets ticked in "
                                                  "the tree) as a Markdown outline to paste into any chat LLM.")
                ui.button("Download as Markdown", icon="download", on_click=lambda: copy_outline(True)).props(
                    "flat no-caps")
                pick_note = ui.label().classes("ag-muted text-xs")
            notice = ui.column().classes("w-full")
            grid = ui.aggrid(grid_options(COLUMNS, [], multi=True, pagination=False, tooltipShowDelay=400),
                             theme="quartz").classes("w-full h-[620px]")
            with ui.row().classes("items-center gap-2"):
                prev_btn = ui.button(icon="chevron_left", on_click=lambda: page_to(-1)).props("flat dense")
                page_label = ui.label().classes("text-sm ag-mono")
                next_btn = ui.button(icon="chevron_right", on_click=lambda: page_to(1)).props("flat dense")

    tree_state: dict[str, Any] = {"tree": None, "ticked": [f"d:{d}" for d in flt.dataset_ids]}

    # ------------------------------------------------------------ behaviour
    def scope() -> Scope:
        return state["scope"]

    def offers_agnostic() -> bool:
        return scope().region != REGION_AGNOSTIC and any(r["region"] == REGION_AGNOSTIC for r in synced)

    def offers_fast() -> bool:
        return (scope().region, scope().delay) == FAST_D1_MARKET

    def read_filter() -> FieldFilter:
        f = state["filter"]
        values: dict[str, Any] = {
            "search": (box.value or "").strip() or None, "search_mode": mode.value or SMART,
            "field_types": list(types.value or []), "category_ids": list(cats.value or []),
            "dataset_ids": ticked_datasets(tree_state["ticked"]), "field_ids": [],
            "region_agnostic": f.region_agnostic and offers_agnostic(),
            "region_exclusive": f.region_exclusive and scope().region != REGION_AGNOSTIC,
            "fast_d1": f.fast_d1 and offers_fast(),
            "date_created_from": date.fromisoformat(month_from.value) if month_from.value else None,
            "date_created_to": date.fromisoformat(month_to.value) if month_to.value else None,
            "sort_by": sort_sel.value or "alpha_count", "sort_desc": bool(desc.value),
            "limit": PAGE_SIZE, "offset": state["offset"]}
        for key, (lo, hi) in ranges.items():
            scale = 100 if key in ("coverage", "date_coverage") else 1
            for end, el in (("min", lo), ("max", hi)):
                v = el.value
                values[f"{key}_{end}"] = None if v is None else (int(v) if key in ("user_count", "alpha_count")
                                                                 else float(v) / scale)
        return FieldFilter(**values)

    async def save_pref(f: FieldFilter) -> None:
        keep = f.model_dump(mode="json", exclude={"limit", "offset", "sort_by", "sort_desc", "field_ids"},
                            exclude_defaults=True)
        await db.kv_set(PREF, {"scope": {"region": scope().region, "delay": scope().delay,
                                         "universe": scope().universe},
                               "filter": keep, "sort": {"sort_by": f.sort_by, "sort_desc": f.sort_desc}})

    def render_toggle(btn: ui.button, on: bool, label: str, n: int, offered: bool, tip: str) -> None:
        btn.set_text(f"{label} · {n:,}")
        btn.props(f"color={'primary' if on else 'grey-8'} {'unelevated' if on else 'outline'}")
        btn.set_visibility(offered)
        btn.tooltip(tip)

    async def reload(*, reset_page: bool = True) -> None:
        if reset_page:
            state["offset"] = 0
        f = read_filter()
        state["filter"] = f
        sc = scope()
        try:
            page_data, facet, stats, sets, counts = await asyncio.gather(
                queries.fields(sc, f), queries.facets(sc, f), queries.stats(sc), queries.datasets(sc),
                queries.counts(sc))
        except Exception as exc:
            notify_error(exc, "Could not read the catalogue: ")
            return
        rows = page_data["results"]
        state["rows"] = rows
        grid.options["rowData"] = rows
        grid.update()
        total = page_data["total"] or 0
        ranked = page_data.get("ranked")
        how = "" if not ranked else (" · every word matched" if page_data.get("conjunctive")
                                     else " · no field has every word, so any word counts")
        total_label.set_text(f"{total:,} matching fields{how}")
        last = min(total, state["offset"] + PAGE_SIZE)
        page_label.set_text(f"{state['offset'] + 1 if total else 0:,}–{last:,} of {total:,}")
        prev_btn.set_enabled(state["offset"] > 0)
        next_btn.set_enabled(last < total)
        # counts strip and the not-synced notice
        counts_row.clear()
        notice.clear()
        with counts_row:
            for label, value in (("Fields", counts["fields"]), ("Datasets", counts["datasets"]),
                                 ("Categories", counts["categories"]), ("Subcategories", counts["subcategories"])):
                with ui.card().classes("ag-tile p-3 min-w-[140px]"):
                    ui.label(label).classes("ag-muted text-xs")
                    ui.label(fmt_int(value)).classes("ag-kpi-sm")
        if not counts["fields"]:
            with notice:
                empty_state("cloud_download", f"{sc.label} is not synced yet",
                            "Open Sync with BRAIN and press Sync to download every market.")
                ui.button("Open Sync with BRAIN", on_click=lambda: ui.navigate.to("/sync")).props("flat")
        local = next((r for r in synced if (r["region"], r["delay"], r["universe"]) ==
                      (sc.region, sc.delay, sc.universe)), None)
        synced_note.set_text(f"Synced {ago(local['synced_at'])}" if local else "")
        # facet choices (a facet ignores its own selection, so its counts say what choosing would show)
        types.options = {t["id"]: f"{t['id']} ({t['n']:,})" for t in facet["types"]}
        for v in types.value or []:
            types.options.setdefault(v, v)
        types.update()
        cats.options = {c["id"]: f"{c.get('name') or c['id']} ({c['n']:,})" for c in facet["categories"]}
        for v in cats.value or []:
            cats.options.setdefault(v, v)
        cats.update()
        months = {m["month"]: f"{m['month'][:7]} ({m['fields']:,})" for m in stats.get("date_added") or []}
        for sel in (month_from, month_to):
            sel.options = dict(months)
            if sel.value and sel.value not in sel.options:
                sel.options[sel.value] = sel.value
            sel.update()
        a = facet["availability"]
        render_toggle(t_agn, f.region_agnostic, "Region Agnostic", a["region_agnostic"], offers_agnostic(),
                      "Only fields that also exist in region ALL: the ones an idea could run region-agnostically on.")
        render_toggle(t_exc, f.region_exclusive, "Region Exclusive", a["region_exclusive"],
                      sc.region != REGION_AGNOSTIC, "Only fields found in no other synced region, region ALL included.")
        render_toggle(t_fast, f.fast_d1, "Fast D1", a["fast_d1"], offers_fast(),
                      "Only BRAIN's Fast D1 fields (…_fast_d1), offered in USA Delay 1 alone.")
        for key, (lo, hi) in ranges.items():
            scale = 100 if key in ("coverage", "date_coverage") else 1
            mn, mx = stats.get(f"{key}_min"), stats.get(f"{key}_max")
            lo.props(f"placeholder='min {fmt_num(mn * scale if mn is not None else None, 0 if scale > 1 else 1)}'")
            hi.props(f"placeholder='max {fmt_num(mx * scale if mx is not None else None, 0 if scale > 1 else 1)}'")
        # the dataset tree
        names = dataset_names(sets)
        counts_by_ds = {d["id"]: int(d["n"]) for d in facet["datasets"]}
        nodes = tree_nodes(build_tree(facet), names, counts_by_ds)
        tree_box.clear()
        with tree_box:
            if not nodes:
                ui.label("No datasets under the other filters.").classes("ag-muted text-xs")
            else:
                tree = ui.tree(nodes, label_key="label", tick_strategy="leaf",
                               on_tick=lambda e: on_tick(e.value)).props("dense")
                tree._props["ticked"] = list(tree_state["ticked"])
                tree.update()
                tree_state["tree"] = tree
        n_sets = len(ticked_datasets(tree_state["ticked"]))
        pick_note.set_text(f"{n_sets} dataset(s) ticked in the tree" if n_sets else
                           "Tick rows to copy them, or tick datasets in the tree.")
        await save_pref(f)

    async def debounced() -> None:
        state["token"] += 1
        mine = state["token"]
        await asyncio.sleep(0.35)
        if mine == state["token"]:
            await reload()

    def on_tick(value: list[str]) -> None:
        tree_state["ticked"] = list(value or [])
        asyncio.ensure_future(debounced())

    async def page_to(step: int) -> None:
        state["offset"] = max(0, state["offset"] + step * PAGE_SIZE)
        await reload(reset_page=False)

    async def toggle(name: str) -> None:
        f = state["filter"]
        on = not getattr(f, name)
        update = {name: on}
        if on and name == "region_agnostic":
            update["region_exclusive"] = False       # a field in region ALL is in another region
        if on and name == "region_exclusive":
            update["region_agnostic"] = False
        state["filter"] = f.model_copy(update=update)
        await reload()

    t_agn.on_click(lambda: toggle("region_agnostic"))
    t_exc.on_click(lambda: toggle("region_exclusive"))
    t_fast.on_click(lambda: toggle("fast_d1"))

    def on_search_change() -> None:
        # a smart search ranks itself until the reader picks a column; an emptied box hands the sort back
        rankable = bool((box.value or "").strip()) and mode.value == SMART
        if rankable and not state["chosen_sort"]:
            sort_sel.set_value(RELEVANCE)
        elif not rankable and sort_sel.value == RELEVANCE:
            sort_sel.set_value("alpha_count")
            state["chosen_sort"] = False
        asyncio.ensure_future(debounced())

    box.on_value_change(lambda _: on_search_change())
    mode.on_value_change(lambda _: on_search_change())

    def on_sort(_e: Any) -> None:
        state["chosen_sort"] = sort_sel.value != RELEVANCE
        asyncio.ensure_future(debounced())

    sort_sel.on_value_change(on_sort)
    for el in (types, cats, desc, month_from, month_to):
        el.on_value_change(lambda _: asyncio.ensure_future(debounced()))
    for lo, hi in ranges.values():
        for el in (lo, hi):
            el.on_value_change(lambda _: asyncio.ensure_future(debounced()))

    async def clear() -> None:
        box.set_value("")
        for el in (types, cats):
            el.set_value([])
        for lo, hi in ranges.values():
            lo.set_value(None)
            hi.set_value(None)
        month_from.set_value(None)
        month_to.set_value(None)
        tree_state["ticked"] = []
        state["filter"] = FieldFilter()
        state["chosen_sort"] = False
        sort_sel.set_value("alpha_count")
        await reload()

    # ---- market selectors
    def universes_for(r: str) -> list[str]:
        held = [x["universe"] for x in synced if x["region"] == r]
        return list(dict.fromkeys([*brain.options.universes(r), *held]))

    def refresh_market_choices() -> None:
        r = region.value
        delays = sorted({*brain.options.delays(r), *(x["delay"] for x in synced if x["region"] == r)})
        delay.options = delays
        if delay.value not in delays:
            delay.value = delays[0] if delays else 1
        delay.update()
        unis = universes_for(r)
        universe.options = unis
        if universe.value not in unis:
            universe.value = unis[0] if unis else None
        universe.update()

    refresh_market_choices()

    async def on_market(_e: Any) -> None:
        refresh_market_choices()
        if not universe.value:
            return
        state["scope"] = Scope(region.value, int(delay.value), universe.value)
        tree_state["ticked"] = []                    # datasets differ between markets
        await reload()

    region.on_value_change(on_market)
    delay.on_value_change(on_market)
    universe.on_value_change(on_market)

    # ---- detail, outline, search everywhere
    async def on_cell(e: Any) -> None:
        row = e.args.get("data") or {}
        if row.get("field_id") and e.args.get("colId") in ("field_id", "description"):
            await field_dialog(scope(), row["field_id"])

    grid.on("cellClicked", on_cell)

    async def copy_outline(download: bool) -> None:
        selected = [r["field_id"] for r in await grid.get_selected_rows()]
        datasets_ticked = ticked_datasets(tree_state["ticked"])
        try:
            if selected:
                out = await outline_mod.field_outline(scope(), field_ids=selected)
            elif datasets_ticked:
                out = await outline_mod.field_outline(scope(), dataset_ids=datasets_ticked)
            else:
                ui.notify("Tick some fields in the table, or some datasets in the tree, first.", type="warning")
                return
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        note = f" ({len(out.missing)} not in {scope().label}: {', '.join(out.missing[:5])})" if out.missing else ""
        if download:
            ui.download(out.text.encode("utf-8"), f"data_fields_{scope().region}_D{scope().delay}.md")
            ui.notify(f"Downloaded {out.fields:,} Data Fields as Markdown{note}.", type="positive")
        else:
            ui.clipboard.write(out.text)
            ui.notify(f"Copied {out.fields:,} Data Fields{note}. Paste them into any chat LLM.", type="positive")

    async def search_everywhere() -> None:
        needle = (anywhere.value or "").strip()
        if not needle:
            ui.notify("Type a field id or a word from its description.", type="warning")
            return
        rows = await queries.fields_anywhere(needle, limit=200)
        with ui.dialog() as dlg, ui.card().classes("ag-card p-6 w-[1100px] max-w-[96vw]"):
            ui.label(f"'{needle}' across every synced market: {len(rows)} field(s)").classes("text-lg")
            ui.label("Each field once, from the market that covers it best; the most used first. Click one to see "
                     "every market it exists in.").classes("ag-muted text-sm")
            g = ui.aggrid(grid_options([
                {"field": "field_id", "headerName": "Field", "width": 220, "cellClass": "ag-mono ag-cyan"},
                {"field": "description", "headerName": "Description", "flex": 1, "minWidth": 240},
                {"field": "dataset_id", "headerName": "Dataset", "width": 120},
                {"field": "region", "headerName": "Best market", "width": 110},
                {"field": "universe", "headerName": "Universe", "width": 130},
                num_col("coverage", "Coverage", 0, 100, pct=True), num_col("alpha_count", "Alphas", 0, 90),
            ], rows, pagination=False), theme="quartz").classes("w-full h-[520px]")

            async def open_row(e: Any) -> None:
                r = e.args.get("data") or {}
                if r.get("field_id"):
                    await field_dialog(Scope(r["region"], int(r["delay"]), r["universe"]), r["field_id"])

            g.on("cellClicked", open_row)
            ui.button("Close", on_click=dlg.close).props("flat").classes("self-end")
        dlg.open()

    anywhere.on("keydown.enter", lambda _: search_everywhere())

    if synced and await search.stale():
        # the index is rebuilt after each all-markets sync; fields stored any other way are indexed here, in the
        # background (substring search answers meanwhile)
        async def build_index() -> None:
            try:
                await search.rebuild()
            except Exception:
                log.warning("field search index rebuild failed; substring search still works", exc_info=True)
        asyncio.ensure_future(build_index())
    await reload()
