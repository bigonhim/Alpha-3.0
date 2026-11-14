"""Alpha Pool (ported from alpha-harness): every alpha stored here, with filters, tags, tag groups, the submittable
shortlist and Power Pool correlation.

Figures are BRAIN's own; those worked out locally (after-cost, K-ratio, the correlations) come from the daily PnL and
turnover each alpha has stored, and say so when it is missing. Green and red only judge a figure against the limit the
alpha's own checks name (or a correlation against its ceiling)."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from nicegui import ui
from sqlalchemy import desc, select

from ...brain import brain
from ...brain.errors import BrainError
from ...brain.platform import (
    POWER_POOL_CORRELATION_CEILING,
    PROD_CORRELATION_CEILING,
    SELF_CORRELATION_CEILING,
    alpha_url,
    is_power_pool,
)
from ...db import db
from ...db.models import Job
from ...engine import metrics, series
from ...jobs.queue import jobs
from ...pool import brainops, correlation, store, submittable
from ...pool import groups as tag_groups
from ..components import (
    card,
    empty_state,
    fmt_int,
    fmt_num,
    fmt_pct,
    grid_options,
    info_box,
    notify_error,
    page_header,
    stat_tile,
)

TONE_CLASS = ("p => { const t = ((p.data && p.data._tone) || {})[p.colDef.field]; return 'ag-num ' + "
              "(t === 'pos' ? 'ag-green' : t === 'neg' ? 'ag-red' : t === 'warn' ? 'ag-amber' : ''); }")


def _fmt(js: str) -> str:
    return f"params => {{ const v = params.value; if (v == null || v === '') return ''; {js} }}"


def fig_col(field: str, header: str, kind: str = "ratio", width: int = 95, tip: str | None = None) -> dict[str, Any]:
    """A figure column toned by the row's `_tone`; kind: ratio | pct | bps | int | corr | imb."""
    js = {
        "ratio": "return Number(v).toFixed(2);",
        "pct": "return (v*100).toFixed(2) + '%';",
        "bps": "return (v*10000).toFixed(2) + ' bps';",
        "int": "return Math.round(v).toLocaleString();",
        "imb": "const x = Math.round(v*100); return (x > 0 ? '+' : '') + x + '%';",
        "corr": ("const a = params.data && params.data._angles; const w = ((params.data && params.data._tone) || {})"
                 "[params.colDef.field] === 'warn' ? ' ≥ limit' : ''; return (a ? Number(v).toFixed(2) + '°' : "
                 "Number(v).toFixed(4)) + w;"),
    }[kind]
    col = {"field": field, "headerName": header, "width": width, "type": "numericColumn",
           ":valueFormatter": _fmt(js), ":cellClass": TONE_CLASS}
    if tip:
        col["headerTooltip"] = tip
    return col


STORED_COLUMNS = [
    {"field": "id", "headerName": "Alpha", "width": 110, "pinned": "left"},
    {"field": "expression", "headerName": "Expression", "minWidth": 260, "flex": 1, "cellClass": "ag-num",
     "tooltipField": "expression"},
    {"field": "region", "headerName": "Region", "width": 85},
    {"field": "delay", "headerName": "D", "width": 58},
    {"field": "universe", "headerName": "Universe", "width": 110},
    fig_col("sharpe", "Sharpe", width=85), fig_col("turnover", "Turnover", "pct", 100),
    fig_col("fitness", "Fitness", width=85), fig_col("returns", "Returns", "pct", 95),
    fig_col("drawdown", "Drawdown", "pct", 100), fig_col("margin", "Margin", "bps", 100),
    fig_col("train_sharpe", "Train Sharpe", width=110), fig_col("test_sharpe", "Test Sharpe", width=105),
    fig_col("test_fitness", "Test Fitness", width=105),
    fig_col("after_cost", "After-Cost Sharpe (5 bps)", width=170,
            tip="Sharpe after 5 bps of trading cost on each day's own turnover. Empty until its daily PnL is stored."),
    fig_col("k_ratio", "K-Ratio", width=90, tip="Kestner's K-ratio (2013): slope of cumulative PnL over its standard "
                                                "error × days, × √250. Rewards steady growth."),
    fig_col("long_count", "Long Count", "int", 100), fig_col("short_count", "Short Count", "int", 105),
    fig_col("imbalance", "Long/Short Imbalance", "imb", 160, tip="(Long − Short) ÷ (Long + Short), in stocks held"),
    fig_col("prod_corr", "Production Correlation", "corr", 170,
            tip=f"BRAIN's Production Correlation; it wants it below {PROD_CORRELATION_CEILING}"),
    fig_col("self_corr", "Self-Correlation (local)", "corr", 175,
            tip="Highest daily PnL correlation with your submitted alphas of its region, Power-Pool-only ones aside, "
                f"over the last 4 years: BRAIN's Self-Correlation as its alpha page shows it today (ceiling "
                f"{SELF_CORRELATION_CEILING})"),
    fig_col("pp_corr", "Power Pool Correlation", "corr", 175,
            tip=f"Against your submitted Power Pool alphas of its region (ceiling {POWER_POOL_CORRELATION_CEILING}); "
                "measured on the Power Pool tab"),
    {"field": "investability", "headerName": "Investability", "width": 115},
    {"field": "classifications", "headerName": "Classifications", "width": 180, "tooltipField": "classifications"},
    {"field": "pyramids", "headerName": "Pyramids", "width": 160, "tooltipField": "pyramids"},
    {"field": "tags", "headerName": "Tags", "width": 140, "tooltipField": "tags"},
    {"field": "ops", "headerName": "Operators", "width": 95, "type": "numericColumn"},
    {"field": "created", "headerName": "Date Created", "width": 115},
    {"field": "pnl", "headerName": "Daily PnL", "width": 95},
    {"field": "failed", "headerName": "Checks Failed", "width": 200, "tooltipField": "failed", "cellClass": "ag-red"},
]
NET_FIELDS = ("sharpe", "fitness", "returns", "drawdown", "margin", "train_sharpe", "test_sharpe", "k_ratio")
CORR_FIELDS = ("prod_corr", "self_corr", "pp_corr")


def angle(rho: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, rho))))


def stored_row(a: Any, has_series: bool, self_corr: float | None) -> dict[str, Any]:
    limits = store.limits_of(a)
    row = {
        "id": a.id, "expression": store.expression_of(a)[:400], "region": a.region, "delay": a.delay,
        "universe": a.universe, "sharpe": a.sharpe, "turnover": a.turnover, "fitness": a.fitness,
        "returns": a.returns, "drawdown": a.drawdown, "margin": a.margin, "train_sharpe": a.train_sharpe,
        "test_sharpe": a.test_sharpe, "test_fitness": a.test_fitness, "after_cost": a.after_cost_5bps,
        "k_ratio": a.k_ratio, "long_count": a.long_count, "short_count": a.short_count,
        "imbalance": store.imbalance(a.long_count, a.short_count), "prod_corr": a.prod_corr, "self_corr": self_corr,
        "pp_corr": a.pp_corr, "investability": store.INVESTABILITY_LABEL[store.investability(a)],
        "classifications": ", ".join(store.classifications_of(a)), "pyramids": ", ".join(store.pyramids_of(a)),
        "tags": ", ".join(store.tags_of(a)), "ops": a.op_count,
        "created": a.date_created.astimezone(store.EASTERN).strftime("%Y-%m-%d") if a.date_created else "",
        "pnl": "stored" if has_series else "", "failed": ", ".join(store.failed_of(a)), "_limits": limits,
    }
    tone_row(row)
    return row


def tone_row(row: dict[str, Any]) -> None:
    limits = row.get("_limits") or {}
    tone = {k: store.figure_tone(k, row.get(k), limits) for k in ("sharpe", "fitness", "turnover", "returns", "margin")}
    for k in ("train_sharpe", "test_sharpe", "after_cost"):
        tone[k] = store.figure_tone("sharpe", row.get(k), limits)
    tone["test_fitness"] = store.figure_tone("fitness", row.get("test_fitness"), limits)
    tone["k_ratio"] = "neg" if (row.get("k_ratio") or 0) < 0 else ""
    for k, ceiling in (("prod_corr", PROD_CORRELATION_CEILING), ("self_corr", SELF_CORRELATION_CEILING),
                       ("pp_corr", POWER_POOL_CORRELATION_CEILING)):
        raw = row.get(f"_{k}", row.get(k))
        tone[k] = store.ceiling_tone(raw, ceiling)
    row["_tone"] = tone


def job_line(kinds: list[str]) -> None:
    """The latest background job of these kinds, live: what it is doing, with Stop while it runs."""
    box = ui.row().classes("w-full items-center gap-3")

    async def refresh() -> None:
        async with db.session() as s:
            job = await s.scalar(select(Job).where(Job.kind.in_(kinds)).order_by(desc(Job.id)).limit(1))
        box.clear()
        if job is None:
            return
        with box:
            kind = {"running": "running", "done": "pass", "failed": "fail", "cancelled": "warning"}.get(job.status, "")
            ui.label(job.status.upper()).classes(f"ag-badge {kind}")
            ui.label(job.title).classes("text-sm")
            text = job.error if job.status == "failed" else job.message
            ui.label(text or "").classes("ag-muted text-sm")
            if job.status in ("running", "queued"):
                ui.linear_progress(value=job.progress or 0, show_value=False).classes("w-40")
                ui.button("Stop", icon="stop", on_click=lambda jid=job.id: jobs.cancel(jid)).props("flat dense")

    ui.timer(2.0, refresh)


def parse_day(text: str | None) -> date | None:
    try:
        return date.fromisoformat(text.strip()) if text and text.strip() else None
    except ValueError:
        return None


async def copy_markdown(alpha_ids: list[str]) -> None:
    """Alphas as Markdown for an LLM (written by the Tools package), copied to the clipboard."""
    if not alpha_ids:
        ui.notify("Tick the alphas to copy first.", type="warning")
        return
    try:
        from ...tools.copy_alphas import alphas_markdown
    except ImportError:
        ui.notify("Copying alphas as Markdown is not available in this build yet.", type="warning")
        return
    rows = await store.by_ids(alpha_ids)
    text = await alphas_markdown([rows[a] for a in alpha_ids if a in rows], glossary=True)
    ui.clipboard.write(text)
    ui.notify(f"Copied {len(rows)} alpha(s) as Markdown, with the metric glossary and the data fields they read.")


# ---------------------------------------------------------------- page

async def page() -> None:
    page_header("Alpha Pool", "Every alpha you have, with filters, tags, tag groups, the submittable shortlist and "
                              "Power Pool correlation.")
    shared: dict[str, Any] = {"ticked": []}
    await _overview()
    with ui.tabs().classes("w-full") as tabs:
        t_stored = ui.tab("Stored Alphas")
        t_sub = ui.tab("Submittable")
        t_groups = ui.tab("Tag Groups")
        t_pp = ui.tab("Power Pool")
    with ui.tab_panels(tabs, value=t_stored).classes("w-full bg-transparent"):
        with ui.tab_panel(t_stored):
            stored_view = await _stored_tab(shared)
        with ui.tab_panel(t_sub):
            await _submittable_tab()
        with ui.tab_panel(t_groups):
            await _groups_tab(shared, stored_view, lambda: tabs.set_value(t_stored))
        with ui.tab_panel(t_pp):
            await _power_pool_tab(shared)


async def _overview() -> None:
    o = await store.overview()
    with ui.row().classes("w-full gap-3 flex-wrap"):
        for label, value, sub in (("Alphas stored", fmt_int(o["alphas"]), f"{fmt_int(o['submitted'])} submitted"),
                                  ("With daily PnL", fmt_int(o["with_pnl"]), "for after-cost and correlations"),
                                  ("Scopes", fmt_int(len(o["scopes"])), f"{len(o['regions'])} regions"),
                                  ("Tags", fmt_int(len(o["tags"])), "carried by stored alphas")):
            v, s = stat_tile(label)
            v.set_text(value)
            s.set_text(sub)
        unsub, unsub_sub = stat_tile("Unsubmitted on BRAIN", tooltip="BRAIN's own count (alphas summary)")
        subm, subm_sub = stat_tile("Submitted on BRAIN", tooltip="BRAIN's Submitted tab: active + decommissioned")

    async def read_summary() -> None:
        if not brain.client.logged_in:
            unsub_sub.set_text("sign in on Sync with BRAIN to read")
            return
        try:
            found = await brainops.brain_summary()
        except BrainError as exc:
            unsub_sub.set_text(f"could not read: {exc.message[:60]}")
            return
        unsub.set_text(fmt_int(found["unsubmitted"]))
        subm.set_text(fmt_int(found["submitted"]))
        subm_sub.set_text(f"{fmt_int(found['active'])} active · {fmt_int(found['decommissioned'])} decommissioned")
    ui.timer(0.1, read_summary, once=True)


class _StoredView:
    """Handles the Stored tab exposes to the other tabs (filter by a tag group, the current filter)."""

    def __init__(self) -> None:
        self.set_tags: Any = None
        self.current: Any = None


async def _stored_tab(shared: dict[str, Any]) -> _StoredView:
    view = _StoredView()
    o = await store.overview()
    f = store.StoredFilter()
    with card("w-full"):
        with ui.row().classes("items-end gap-3 w-full flex-wrap"):
            search = ui.input("Search id, name or expression").classes("w-72").props("clearable")
            submitted = ui.toggle({False: "Unsubmitted", True: "Submitted"}, value=False).props(
                "no-caps unelevated dense toggle-color=primary")
            sort = ui.select(store.SORT_LABELS, value="sharpe", label="Sort by").classes("w-48")
            desc_box = ui.checkbox("Highest first", value=True)
            size = ui.select({n: f"{n} a page" for n in store.PAGE_SIZES}, value=100, label="Page size").classes("w-32")
            angles = ui.checkbox("Show correlation as angles")
            after = ui.checkbox("After-cost mode").tooltip(
                "Replace the figure columns with figures net of a trading cost, worked out from stored daily PnL and "
                "turnover (only alphas with turnover stored)")
            bps = ui.number("Cost (bps)", value=5.0, min=0.5, max=100, step=0.5).classes("w-28")
        with ui.expansion("Filters", icon="filter_alt").classes("w-full") as filters:
            with ui.row().classes("w-full gap-3 flex-wrap"):
                regions = ui.select(o["regions"], multiple=True, label="Region").props("use-chips").classes("min-w-[160px]")
                delays = ui.select({d: f"D{d}" for d in o["delays"]}, multiple=True, label="Delay").props(
                    "use-chips").classes("min-w-[120px]")
                universes = ui.select(o["universes"], multiple=True, label="Universe").props("use-chips").classes(
                    "min-w-[180px]")
                tags = ui.select(o["tags"], multiple=True, label="Tags", new_value_mode="add-unique").props(
                    "use-chips").classes("min-w-[220px]")
                tags_all = ui.toggle({False: "Any of these", True: "All of these"}, value=False).props(
                    "no-caps dense unelevated toggle-color=primary")
                after_day = ui.input("Created on or after (YYYY-MM-DD)").classes("w-52")
                before_day = ui.input("Created on or before (YYYY-MM-DD)").classes("w-52")
            bounds: dict[str, tuple[Any, Any]] = {}
            with ui.row().classes("w-full gap-2 flex-wrap"):
                for key, (label, _, _) in store.BOUNDS.items():
                    with ui.column().classes("gap-0"):
                        ui.label(label).classes("ag-muted text-xs")
                        with ui.row().classes("gap-1 no-wrap"):
                            bounds[key] = (ui.number("min").classes("w-20").props("dense"),
                                           ui.number("max").classes("w-20").props("dense"))
            ui.label("Days count on BRAIN's New York clock, both included, as BRAIN's own date filter reads them.") \
                .classes("ag-muted text-xs")
        grid = ui.aggrid(grid_options(STORED_COLUMNS, [], multi=True, pagination=False, tooltipShowDelay=300),
                         theme="quartz").classes("w-full h-[600px]")
        with ui.row().classes("items-center gap-2 w-full"):
            prev_btn = ui.button(icon="chevron_left").props("flat dense")
            next_btn = ui.button(icon="chevron_right").props("flat dense")
            pager = ui.label("").classes("ag-muted text-sm")
            note = ui.label("").classes("ag-muted text-xs ml-auto")
        with ui.row().classes("items-end gap-2 w-full flex-wrap"):
            tag_text = ui.input("Tags to add (comma-separated)").classes("w-64")
            add_btn = ui.button("Add Tag", icon="sell")
            ui.button("Copy as Markdown", icon="content_copy",
                      on_click=lambda: _with_ticked(grid, copy_markdown)).props("flat")
            ui.button("Download PnL of ticked", icon="download",
                      on_click=lambda: _with_ticked(grid, _download)).props("flat")
            analyse_btn = ui.button("Analyse filter as a Portfolio", icon="donut_large").props("flat")
            ui.button("Refresh", icon="refresh", on_click=lambda: reload()).props("flat")
        with ui.row().classes("items-end gap-2 w-full flex-wrap"):
            import_text = ui.input("Import alphas by BRAIN tag (e.g. ace_tag, harness)").classes("w-80")
            import_btn = ui.button("Import from BRAIN", icon="cloud_download").props("flat")
            job_line(["pool_import_tagged", "fetch_series"])
    analysis = ui.column().classes("w-full gap-3")

    def read_filter() -> None:
        f.search = search.value or ""
        f.submitted = bool(submitted.value)
        f.sort_by, f.sort_desc, f.limit = sort.value, bool(desc_box.value), int(size.value)
        f.regions, f.delays, f.universes = list(regions.value or []), list(delays.value or []), list(
            universes.value or [])
        f.tags, f.tags_all = list(tags.value or []), bool(tags_all.value)
        f.created_after, f.created_before = parse_day(after_day.value), parse_day(before_day.value)
        f.minimum = {k: float(lo.value) for k, (lo, _) in bounds.items() if lo.value is not None}
        f.maximum = {k: float(hi.value) for k, (_, hi) in bounds.items() if hi.value is not None}
        filters.text = f"Filters ({f.active()} active)" if f.active() else "Filters"

    async def reload(reset: bool = False) -> None:
        if reset:
            f.offset = 0
        read_filter()
        total, rows, has_series = await store.page(f)
        try:
            local = await correlation.local_self_correlation([a.id for a in rows])
        except Exception as exc:          # a broken series must not take the table with it
            local = {}
            note.set_text(f"Self-correlation could not be measured: {exc}")
        out = [stored_row(a, a.id in has_series, (local.get(a.id) or (None, 0))[0]) for a in rows]
        msg = ""
        if after.value:
            cost = float(bps.value or 5.0)
            loaded = await series.load([a.id for a in rows if a.id in has_series])
            by_id = {a.id: a for a in rows}
            netted = 0
            for r in out:
                ser = loaded.get(r["id"])
                if ser is None or not ser["has_turnover"]:
                    for k in NET_FIELDS:
                        r[k] = None
                    continue
                fig = metrics.figures(ser["dates"], ser["pnl"], ser["turnover"], split=by_id[r["id"]].test_start,
                                      bps=cost)
                net = fig["after_cost"]
                r.update(sharpe=net.get("sharpe"), fitness=net.get("fitness"), returns=net.get("returns"),
                         drawdown=net.get("drawdown"), margin=net.get("margin"), train_sharpe=net.get("train_sharpe"),
                         test_sharpe=net.get("test_sharpe"), k_ratio=net.get("k_ratio"))
                netted += 1
                tone_row(r)
            msg = (f"After-cost mode: figures net of {cost:g} bps for {netted} of {len(out)} alphas on this page; "
                   "the rest have no turnover stored (Download PnL fills it).")
        if angles.value:
            for r in out:
                r["_angles"] = True
                for k in CORR_FIELDS:
                    if r.get(k) is not None:
                        r[f"_{k}"] = r[k]
                        r[k] = angle(r[k])
                tone_row(r)
        grid.options["rowData"] = out
        grid.update()
        first = f.offset + 1 if total else 0
        pager.set_text(f"{first:,}–{f.offset + len(out):,} of {total:,} alphas")
        prev_btn.set_enabled(f.offset > 0)
        next_btn.set_enabled(f.offset + len(out) < total)
        if msg or not note.text.startswith("Self-correlation"):
            note.set_text(msg)

    async def turn(step: int) -> None:
        f.offset = max(0, f.offset + step * f.limit)
        await reload()

    async def add_tag() -> None:
        ids = [r["id"] for r in await grid.get_selected_rows()]
        try:
            tagged, failed = await brainops.add_tags(ids, tag_text.value or "")
        except ValueError as exc:
            ui.notify(str(exc), type="warning", multi_line=True)
            return
        except BrainError as exc:
            notify_error(exc, "BRAIN refused: ")
            return
        if failed:
            first = next(iter(failed.items()))
            ui.notify(f"Tagged {len(tagged)}; {len(failed)} not tagged ({first[0]}: {first[1]})", type="warning",
                      multi_line=True)
        else:
            ui.notify(f"Tagged {len(tagged)} alpha(s) on BRAIN and here.", type="positive")
        await reload()

    async def import_tagged() -> None:
        try:
            wanted = brainops.parse_tags(import_text.value or "")
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        await jobs.submit("pool_import_tagged", f"Alphas tagged {', '.join(wanted)}", {"tags": wanted})
        ui.notify(f"Bringing in the alphas BRAIN lists under {', '.join(wanted)} (see Tasks).")

    async def analyse() -> None:
        read_filter()
        ids = await store.matching_ids(f)
        await _analyse(analysis, ids, float(bps.value or 5.0))

    async def set_tags(new: list[str], every: bool) -> None:
        tags.options = sorted(set(tags.options) | set(new))
        tags.value = list(new)
        tags_all.value = bool(every)
        await reload(reset=True)

    view.set_tags = set_tags
    view.current = lambda: (read_filter(), f)[1]
    for el in (submitted, sort, desc_box, size, regions, delays, universes, tags, tags_all):
        el.on_value_change(lambda _: reload(reset=True))
    angles.on_value_change(lambda _: reload())
    after.on_value_change(lambda _: reload())
    for el in (search, after_day, before_day, bps, *[b for pair in bounds.values() for b in pair]):
        el.on("keydown.enter", lambda _: reload(reset=True))
        el.on("blur", lambda _: reload(reset=True))
    prev_btn.on_click(lambda: turn(-1))
    next_btn.on_click(lambda: turn(1))
    add_btn.on_click(add_tag)
    import_btn.on_click(import_tagged)
    analyse_btn.on_click(analyse)

    async def on_cell(e) -> None:  # noqa: ANN001
        row = e.args.get("data") or {}
        if row.get("id") and e.args.get("colId") == "id":
            from ..alpha_detail import open_alpha
            await open_alpha(row["id"], on_change=reload)

    async def on_select(_e) -> None:  # noqa: ANN001
        shared["ticked"] = [r["id"] for r in await grid.get_selected_rows()]

    grid.on("cellClicked", on_cell)
    grid.on("selectionChanged", on_select)
    await reload()
    return view


async def _with_ticked(grid: Any, fn: Any) -> None:
    await fn([r["id"] for r in await grid.get_selected_rows()])


async def _download(ids: list[str]) -> None:
    if not ids:
        ui.notify("Tick the alphas first.", type="warning")
        return
    todo = await series.missing(ids, turnover=True)
    if not todo:
        ui.notify("Every ticked alpha already has its daily PnL and turnover stored.")
        return
    await jobs.submit("fetch_series", f"Daily PnL for {len(todo)} alphas", {"alpha_ids": todo, "turnover": True})
    ui.notify(f"Downloading the daily PnL and turnover of {len(todo)} alpha(s) (see Tasks). No simulations are spent.")


async def _analyse(box: Any, ids: list[str], bps: float) -> None:
    from ...pool import portfolio as pf
    from .portfolio import render_results
    box.clear()
    if not ids:
        with box:
            ui.label("Nothing matches, so there is nothing to analyse.").classes("ag-muted")
        return
    more = len(ids) > store.GROUP_LIMIT
    ids = ids[:store.GROUP_LIMIT]
    result = await pf.compute(ids, bps)
    with box:
        with card("w-full"):
            ui.label(f"{len(ids):,} alphas analysed as one portfolio" + (
                f" (the newest {store.GROUP_LIMIT:,}; more matched)" if more else "")).classes("ag-section-title")
            if result["missing"]:
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"{len(result['missing'])} have no daily PnL and turnover stored.").classes("ag-muted")
                    ui.button("Download PnL", icon="download",
                              on_click=lambda: _download(result["missing"])).props("flat dense")
        render_results(result)


# ---------------------------------------------------------------- submittable

def _figure(label: str, value: float | None, tone: str, text: str) -> None:
    color = {"pos": "ag-green", "neg": "ag-red"}.get(tone, "")
    with ui.column().classes("gap-0 min-w-[70px]"):
        ui.label(label).classes("ag-muted text-xs")
        ui.label(text if value is not None else "—").classes(f"ag-mono text-sm {color}")


def _card(a: Any, spark: list[float] | None, refresh: Any) -> None:
    limits = store.limits_of(a)
    with card("w-full"):
        with ui.row().classes("items-center gap-2 w-full"):
            ui.label(a.id).classes("ag-mono font-semibold ag-link cursor-pointer").on(
                "click", lambda: _open(a.id, refresh))
            ui.label(f"{a.region or '—'} D{a.delay if a.delay is not None else '—'} {a.universe or ''}").classes(
                "ag-muted text-sm")
            ui.label(f"{a.neutralization or ''} · decay {a.decay if a.decay is not None else '—'} · truncation "
                     f"{a.truncation if a.truncation is not None else '—'}").classes("ag-muted text-xs")
            ui.space()
            ui.button("Copy expression", icon="content_copy",
                      on_click=lambda: (ui.clipboard.write(store.expression_of(a)),
                                        ui.notify("Expression copied."))).props("flat dense")
            ui.button("Re-check", icon="rule", on_click=lambda: _recheck(a.id)).props("flat dense")
            ui.button("Open in BRAIN", icon="open_in_new",
                      on_click=lambda: ui.navigate.to(alpha_url(a.id), new_tab=True)).props("flat dense")
        ui.label(store.expression_of(a)).classes("ag-mono text-xs break-all")
        with ui.row().classes("gap-4 flex-wrap items-end"):
            _figure("Sharpe", a.sharpe, store.figure_tone("sharpe", a.sharpe, limits), fmt_num(a.sharpe))
            _figure("Fitness", a.fitness, store.figure_tone("fitness", a.fitness, limits), fmt_num(a.fitness))
            _figure("Turnover", a.turnover, store.figure_tone("turnover", a.turnover, limits), fmt_pct(a.turnover, 2))
            _figure("Returns", a.returns, store.figure_tone("returns", a.returns, limits), fmt_pct(a.returns, 2))
            _figure("Drawdown", a.drawdown, "", fmt_pct(a.drawdown, 2))
            _figure("Margin", a.margin, store.figure_tone("margin", a.margin, limits),
                    f"{(a.margin or 0) * 1e4:.2f} bps")
            _figure("Train Sharpe", a.train_sharpe, store.figure_tone("sharpe", a.train_sharpe, limits),
                    fmt_num(a.train_sharpe))
            _figure("Test Sharpe", a.test_sharpe, store.figure_tone("sharpe", a.test_sharpe, limits),
                    fmt_num(a.test_sharpe))
            if spark and len(spark) > 1:
                color = "#4ade80" if spark[-1] >= spark[0] else "#f87171"
                ui.echart({"animation": False, "grid": {"left": 0, "right": 0, "top": 2, "bottom": 2},
                           "xAxis": {"type": "category", "show": False, "data": list(range(len(spark)))},
                           "yAxis": {"type": "value", "show": False, "scale": True},
                           "series": [{"type": "line", "data": spark, "showSymbol": False,
                                       "lineStyle": {"color": color, "width": 1.5}}]}).classes("w-48 h-12")
            else:
                ui.label("PnL not downloaded yet").classes("ag-muted text-xs")


async def _open(alpha_id: str, refresh: Any) -> None:
    from ..alpha_detail import open_alpha
    await open_alpha(alpha_id, on_change=refresh)


async def _recheck(alpha_id: str) -> None:
    await jobs.submit("check_alpha", f"Check {alpha_id}", {"alpha_id": alpha_id})
    ui.notify(f"Asked BRAIN to check {alpha_id} again (see Tasks). It costs no simulations.")


async def _submittable_tab() -> None:
    box = ui.column().classes("w-full gap-3")

    async def refresh() -> None:
        found = await submittable.submittable()
        fams = await submittable.families()
        box.clear()
        with box:
            with ui.row().classes("w-full gap-3 flex-wrap"):
                for label, value, sub in (
                        ("Submittable", found["total"], "every gating check PASS or WARNING"),
                        ("Still being judged", found["pending"], "checks PENDING: not yet, rather than never"),
                        ("Near misses", found["near_misses"], "one or two fixable checks failed"),
                        ("Held-out failed", found["held_out_failed"], "test Sharpe ≤ 0 or < half the train Sharpe"),
                        ("Never validated", found["unvalidated"], "simulated without a test period"),
                        ("Correlated, passed over", found["correlated_pruned"], "|ρ| ≥ 0.7 with a better pick"),
                        ("Implausible, hidden", found["implausible_hidden"], "a Sharpe beyond ±10 in some period")):
                    v, s = stat_tile(label)
                    v.set_text(fmt_int(value))
                    s.set_text(sub)
            if found["markets"]:
                ui.label("Where they are: " + " · ".join(
                    f"{m['region']} D{m['delay']} {m['universe']}: {m['count']}" for m in found["markets"])).classes(
                    "ag-muted text-sm")
            if found["pnl_missing"]:
                ui.label(f"{len(found['pnl_missing'])} held-out alpha(s) have no daily PnL stored, so they could not be "
                         "compared for the shortlist; their download is queued (see Tasks).").classes("ag-amber text-sm")
            ui.label("Shortlist").classes("ag-section-title mt-2")
            ui.label("At most five distinct alphas that held up in the test years, most stable first (test Sharpe and "
                     "sub-universe margin), each kept unless it moves with a better pick.").classes("ag-muted text-xs")
            if not found["shortlist"]:
                empty_state("inventory_2", "No shortlist yet",
                            "Nothing submittable has held up in its test years yet. Pending checks and near misses "
                            "above say what is close.")
            for a in found["shortlist"]:
                _card(a, found["spark"].get(a.id), refresh)
            if found["alphas"]:
                ui.label(f"All submittable, best Sharpe first ({min(len(found['alphas']), 50)} of "
                         f"{found['total']})").classes("ag-section-title mt-2")
                rows = [stored_row(a, a.id in found["spark"], None) for a in found["alphas"]]
                cols = [c for c in STORED_COLUMNS if c["field"] in (
                    "id", "expression", "region", "delay", "universe", "sharpe", "fitness", "turnover", "returns",
                    "train_sharpe", "test_sharpe", "after_cost", "prod_corr", "pp_corr")]
                g = ui.aggrid(grid_options(cols, rows), theme="quartz").classes("w-full h-[420px]")

                async def open_row(e) -> None:  # noqa: ANN001
                    if (e.args.get("data") or {}).get("id"):
                        await _open(e.args["data"]["id"], refresh)
                g.on("cellDoubleClicked", open_row)
            ui.label("Region-agnostic families").classes("ag-section-title mt-2")
            ui.label("BRAIN submits a region-agnostic alpha only when two of its regions pass every check "
                     "(MIN_SUBMITTABLE_CHILDREN); the family stands on its second-best region.").classes(
                "ag-muted text-xs")
            if not fams:
                ui.label("No unsubmitted region-agnostic alphas are stored.").classes("ag-muted text-sm")
            else:
                ui.aggrid(grid_options([
                    {"field": "id", "headerName": "Parent", "width": 110},
                    {"field": "check", "headerName": "MIN_SUBMITTABLE_CHILDREN", "width": 210, ":cellClass": TONE_CLASS},
                    {"field": "passing", "headerName": "Children passing", "width": 140},
                    {"field": "children", "headerName": "Children", "width": 100},
                    {"field": "regions", "headerName": "Scored regions", "width": 160},
                    {"field": "standing", "headerName": "Standing child", "width": 130},
                    fig_col("sharpe", "Standing Sharpe", width=130), fig_col("fitness", "Standing Fitness", width=130),
                    {"field": "expression", "headerName": "Expression", "flex": 1, "minWidth": 240, "cellClass": "ag-num"},
                ], [{
                    "id": fm["parent"].id, "check": fm["check"]["result"], "passing": fm["check"]["value"],
                    "children": len(fm["children"]), "regions": ", ".join(str(r) for r in fm["regions"]),
                    "standing": fm["standing"].id if fm["standing"] else "",
                    "sharpe": fm["standing"].sharpe if fm["standing"] else None,
                    "fitness": fm["standing"].fitness if fm["standing"] else None,
                    "expression": store.expression_of(fm["parent"])[:300],
                    "_tone": {"check": {"PASS": "pos", "FAIL": "neg", "PENDING": "warn"}[fm["check"]["result"]]},
                } for fm in fams]), theme="quartz").classes("w-full h-72")
            ui.button("Refresh", icon="refresh", on_click=refresh).props("flat")

    await refresh()


# ---------------------------------------------------------------- tag groups

async def _groups_tab(shared: dict[str, Any], stored_view: _StoredView, show_stored: Any) -> None:
    o = await store.overview()
    with card("w-full"):
        ui.label("Save a group").classes("ag-section-title")
        with ui.row().classes("items-end gap-3 flex-wrap"):
            name = ui.input("Name (e.g. ACE and Power Pool)").classes("w-64").props("maxlength=60")
            gtags = ui.select(o["tags"], multiple=True, label="Tags", new_value_mode="add-unique").props(
                "use-chips").classes("min-w-[260px]")
            every = ui.toggle({False: "Any of these", True: "All of these"}, value=False).props(
                "no-caps dense unelevated toggle-color=primary")
            save_btn = ui.button("Save as a Group", icon="save")
    listing = ui.column().classes("w-full gap-2")
    analysis = ui.column().classes("w-full gap-3")
    with card("w-full"):
        with ui.row().classes("items-end gap-3"):
            bps = ui.number("Cost for the analysis (bps)", value=5.0, min=0, max=100, step=0.5).classes("w-52")
            ui.button("Analyse the Stored tab's current filter", icon="donut_large",
                      on_click=lambda: _analyse_filter()).props("flat")
        job_line(["fetch_series"])

    async def _analyse_filter() -> None:
        f = stored_view.current()
        await _analyse(analysis, await store.matching_ids(f), float(bps.value or 5.0))

    async def members(g: Any) -> list[str]:
        ids: list[str] = []
        for done in (False, True):
            ids += await store.matching_ids(store.StoredFilter(submitted=done, tags=list(g.tags), tags_all=g.every))
        return ids

    async def draw() -> None:
        found = await tag_groups.all_groups()
        listing.clear()
        with listing:
            if not found:
                info_box("No tag groups yet", "Pick some tags above and save them under a name; then one click filters "
                                              "the pool by them, analyses them as a portfolio or downloads their PnL.")
            for g in found:
                with card("w-full"):
                    with ui.row().classes("items-center gap-3 w-full"):
                        ui.label(g.name).classes("font-semibold")
                        ui.label(f"{'All' if g.every else 'Any'} of: {', '.join(g.tags)}").classes("ag-muted text-sm")
                        ui.space()
                        ui.button("Filter", icon="filter_alt", on_click=lambda g=g: _filter(g)).props("flat dense")
                        ui.button("Analyse", icon="donut_large", on_click=lambda g=g: _analyse_group(g)).props(
                            "flat dense")
                        ui.button("Download PnL", icon="download", on_click=lambda g=g: _pnl(g)).props("flat dense")
                        ui.button(icon="close", on_click=lambda g=g: _forget(g)).props("flat dense").tooltip(
                            f"Forget {g.name}")

    async def _filter(g: Any) -> None:
        show_stored()
        await stored_view.set_tags(list(g.tags), bool(g.every))

    async def _analyse_group(g: Any) -> None:
        await _analyse(analysis, await members(g), float(bps.value or 5.0))

    async def _pnl(g: Any) -> None:
        await _download(await members(g))

    async def _forget(g: Any) -> None:
        await tag_groups.forget_group(g.id)
        await draw()

    async def save() -> None:
        try:
            await tag_groups.save_group(name.value or "", list(gtags.value or []), bool(every.value))
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        name.value = ""
        ui.notify("Group saved.", type="positive")
        await draw()

    save_btn.on_click(save)
    await draw()


# ---------------------------------------------------------------- Power Pool

PP_SCOPES = {"clean": "Unsubmitted Power Pool alphas with no check FAIL or ERROR",
             "all": "Every unsubmitted Power Pool alpha", "ticked": "The alphas ticked on Stored Alphas"}
VERDICT_TONE = {"clear": "pos", "beats": "pos", "blocked": "neg", "unmeasured": "warn"}


async def _pp_ids(scope: str, shared: dict[str, Any]) -> list[str]:
    if scope == "ticked":
        return list(shared.get("ticked") or [])
    alphas = await store.by_ids(await brainops.unsubmitted_ids())
    return [a.id for a in alphas.values() if is_power_pool(a.checks)
            and (scope == "all" or store.verdict_of(a) in ("submittable", "pending"))]


async def _power_pool_tab(shared: dict[str, Any]) -> None:
    with card("w-full"):
        ui.label(f"Measured locally against your submitted Power Pool alphas of the same region, as BRAIN measures it: "
                 f"below {POWER_POOL_CORRELATION_CEILING} is clear; above it an alpha needs a Sharpe of at least 1.1 × "
                 "that of every pool alpha it collides with. Last 4 years, at least 250 shared days. Nothing here "
                 "spends simulations.").classes("ag-muted text-sm")
        with ui.row().classes("items-end gap-3 flex-wrap"):
            scope = ui.select(PP_SCOPES, value="clean", label="Alphas").classes("w-96")
            ui.button("Measure", icon="query_stats", on_click=lambda: measure())
            ui.button("Submission Order", icon="format_list_numbered", on_click=lambda: order()).props("flat")
            ui.button("Power Pool Workflow", icon="account_tree", on_click=lambda: workflow()).props("flat").tooltip(
                "Narrow in three steps (unsubmitted → no check FAIL or ERROR → Power Pool Correlation passed), "
                "downloading the PnL each step needs")
            ui.button("Production Correlation", icon="factory", on_click=lambda: prod()).props("flat").tooltip(
                "Ask BRAIN for the Production Correlation of each alpha it has not given one for; BRAIN limits these "
                "per hour, so a long list takes hours")
        job_line(["pool_power_pool_workflow", "pool_prod_correlation", "fetch_series"])
    out = ui.column().classes("w-full gap-3")

    async def measure() -> None:
        ids = await _pp_ids(scope.value, shared)
        found = await correlation.power_pool_analysis(ids)
        out.clear()
        with out:
            with ui.row().classes("w-full gap-3 flex-wrap"):
                tally: dict[str, int] = {}
                for r in found["rows"]:
                    tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
                for label, value, sub in (
                        ("Power Pool alphas", found["power_pool_alphas"], f"{found['other_alphas']} others not judged"),
                        ("Submitted pool", found["pool_size"], "across every region"),
                        ("Clear", tally.get("clear", 0), "below the ceiling"),
                        ("Beats", tally.get("beats", 0), "above it, Sharpe high enough"),
                        ("Blocked", tally.get("blocked", 0), "above it without the Sharpe"),
                        ("Unmeasured", tally.get("unmeasured", 0) + len(found["without_pnl"]),
                         "too few shared days or no PnL stored")):
                    v, s = stat_tile(label)
                    v.set_text(fmt_int(value))
                    s.set_text(sub)
            for sc in found["scopes"]:
                lacking = sc["pool_without_pnl"]
                ui.label(f"{sc['region']}: pool of {len(sc['pool'])}" + (
                    f", {len(lacking)} without stored PnL (Workflow or a Portfolio sync downloads them)" if lacking
                    else "" if sc["pool"] else " — nothing submitted yet, so the first alpha is clear")).classes(
                    "ag-muted text-sm")
            if found["without_pnl"]:
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"{len(found['without_pnl'])} alpha(s) have no PnL stored.").classes("ag-amber text-sm")
                    ui.button("Download PnL", icon="download",
                              on_click=lambda: _download(found["without_pnl"])).props("flat dense")
            if not found["rows"]:
                empty_state("diamond", "Nothing to measure",
                            "None of these alphas carries BRAIN's POWER_POOL_CORRELATION check.")
                return
            rows = [{**r, "_tone": {"verdict": VERDICT_TONE.get(r["verdict"], ""),
                                    "correlation": store.ceiling_tone(r["correlation"], POWER_POOL_CORRELATION_CEILING)}}
                    for r in found["rows"]]
            ui.aggrid(grid_options([
                {"field": "alpha_id", "headerName": "Alpha", "width": 110},
                {"field": "region", "headerName": "Region", "width": 85},
                {"field": "universe", "headerName": "Universe", "width": 110},
                {"field": "delay", "headerName": "D", "width": 58},
                fig_col("sharpe", "Sharpe", width=85), fig_col("correlation", "Power Pool Correlation", "corr", 175),
                {"field": "verdict", "headerName": "Verdict", "width": 110, ":cellClass": TONE_CLASS},
                fig_col("needed", "Sharpe needed", width=120,
                        tip="The Sharpe that clears every collision at or above the ceiling"),
                {"field": "against", "headerName": "Against", "width": 90},
                {"field": "closest", "headerName": "Closest", "width": 110},
                fig_col("closest_sharpe", "Closest Sharpe", width=125),
            ], rows), theme="quartz").classes("w-full h-[480px]")

    async def order() -> None:
        ids = await _pp_ids(scope.value, shared)
        found = await correlation.order_for(ids)
        out.clear()
        with out:
            ui.label(f"Greedy submission, best After-Cost Sharpe first, of {found['candidates']} candidates: "
                     f"{len(found['picks'])} could be submitted one after another, {len(found['blocked'])} would be "
                     f"blocked by an earlier pick, {len(found['unmeasured'])} could not be measured.").classes("text-sm")
            ui.aggrid(grid_options([
                {"field": "n", "headerName": "#", "width": 60},
                {"field": "alpha_id", "headerName": "Alpha", "width": 110},
                {"field": "region", "headerName": "Region", "width": 85},
                fig_col("sharpe", "Sharpe", width=85), fig_col("after_cost_sharpe", "After-Cost Sharpe", width=140),
                fig_col("correlation", "Correlation at its turn", "corr", 170),
                {"field": "closest", "headerName": "Closest", "width": 110},
                {"field": "verdict", "headerName": "Verdict", "width": 100},
                {"field": "blocks", "headerName": "Blocks", "width": 85},
            ], [{**p, "n": i + 1, "_tone": {"correlation": store.ceiling_tone(p.get("correlation"),
                                                                              POWER_POOL_CORRELATION_CEILING)}}
                for i, p in enumerate(found["picks"])]), theme="quartz").classes("w-full h-80")
            if found["blocked"]:
                ui.label("Blocked: " + ", ".join(f"{b['alpha_id']} (by {b['by']})" for b in found["blocked"])).classes(
                    "ag-muted text-sm")
            if found["unmeasured"]:
                ui.label("Unmeasured: " + ", ".join(found["unmeasured"])).classes("ag-muted text-sm")

    async def workflow() -> None:
        ids = list(shared.get("ticked") or []) if scope.value == "ticked" else []
        if jobs.is_running("pool_power_pool_workflow"):
            ui.notify("The Power Pool Workflow is already running.", type="warning")
            return
        await jobs.submit("pool_power_pool_workflow", "Power Pool Workflow", {"alpha_ids": ids})
        ui.notify("Power Pool Workflow started (see the line above, or Tasks). Stop keeps what it downloaded.")

    async def prod() -> None:
        ids = await _pp_ids(scope.value, shared)
        if not ids:
            ui.notify("No alphas in this choice.", type="warning")
            return
        if jobs.is_running("pool_prod_correlation"):
            ui.notify("Production Correlation is already being asked for.", type="warning")
            return
        await jobs.submit("pool_prod_correlation", f"Production Correlation for {len(ids)} alphas", {"alpha_ids": ids})
        ui.notify("Asking BRAIN for Production Correlation (see Tasks). Every answer is kept.")

