"""Out of Sample — estimated out-of-sample outlook of submitted alphas from their in-sample daily PnL."""

from __future__ import annotations

from nicegui import ui

from ...brain import brain
from ...brain.errors import BrainError
from ...engine.outlook import EXPLAIN, outlook_rows
from ...jobs.queue import jobs
from ..alpha_detail import open_alpha
from ..components import (card, grid_options, info_box, notify_error, num_col, page_header, segmented,
                          sparkline_col, stat_tile)

OUTLOOK_RENDER = """params => {
  const k = {overfit: 'fail', fading: 'warning', fine: '', robust: 'pass'}[params.data.outlook] || '';
  return '<span class="ag-badge ' + k + '">' + params.value + '</span>';
}"""
KIND_RENDER = """params => '<span class="ag-badge ' + (params.value === 'Power Pool' ? 'running' : params.value === 'Super'
  ? 'pending' : '') + '">' + params.value + '</span>'"""

COLUMNS = [
    {"field": "id", "headerName": "Alpha", "width": 110, "pinned": "left"},
    {"field": "kind", "headerName": "Type", "width": 115, ":cellRenderer": KIND_RENDER},
    {"field": "outlook_label", "headerName": "OS outlook", "width": 125, ":cellRenderer": OUTLOOK_RENDER,
     "headerTooltip": EXPLAIN["outlook"], "tooltipField": "reason"},
    {"field": "risk", "headerName": "Risk", "width": 100, "type": "numericColumn", "sort": "desc",
     "headerTooltip": EXPLAIN["risk"]},
    sparkline_col("spark", "IS PnL"),
    num_col("is_sharpe", "IS Sharpe", 2, 100),
    {**num_col("early", "Early IS", 2, 95), "headerTooltip": EXPLAIN["early"]},
    {**num_col("last2", "Last 2y IS", 2, 105), "headerTooltip": EXPLAIN["last2"]},
    {**num_col("holds", "Holds", 0, 90, pct=True), "headerTooltip": EXPLAIN["holds"]},
    {**num_col("est_os", "Est. OS Sharpe", 2, 125), "headerTooltip": EXPLAIN["est_os"]},
    {**num_col("brain_os", "BRAIN OS", 2, 100), "headerTooltip": "Out-of-sample Sharpe reported by BRAIN, if any."},
    num_col("turnover", "Turnover", 0, 100, pct=True),
    {"field": "scope", "headerName": "Scope", "width": 95},
    {"field": "submitted", "headerName": "Submitted", "width": 115},
    {"field": "reason", "headerName": "Main reason", "flex": 1, "minWidth": 320, "tooltipField": "reason"},
]


async def page() -> None:
    with ui.row().classes("w-full items-start no-wrap"):
        page_header("Out of Sample", "How your submitted alphas are likely to hold up out of sample, estimated from "
                                     "their in-sample daily PnL. Riskiest first.")
        ui.space()

        async def refresh() -> None:
            if not brain.client.logged_in:
                ui.notify("Sign in to BRAIN first.", type="warning")
                return
            await jobs.submit("prepare_pnl", "Prepare data (submitted alphas + daily PnL)")
            ui.notify("Refreshing from BRAIN — see Tasks. Reopen this page when it finishes.")

        ui.button("Refresh data", icon="sync", on_click=refresh).props("outline")

    info_box("How this works",
             "BRAIN shows little out-of-sample detail, so this page compares each alpha's early in-sample years with "
             "its last 2 in-sample years. An alpha whose recent years kept most of their Sharpe ('Holds') usually "
             "behaves better after submission. High turnover and complex expressions add risk. Est. OS Sharpe is a "
             "rule of thumb, not a BRAIN number. Hover a column title or an outlook to see what it means.")

    rows = await outlook_rows()
    counts = {k: sum(1 for r in rows if r["outlook"] == k) for k in ("robust", "fine", "fading", "overfit")}
    with ui.row().classes("w-full gap-4 no-wrap"):
        for label, value, color in (("Submitted alphas", len(rows), ""), ("Robust", counts["robust"], "ag-green"),
                                    ("Likely fine", counts["fine"], "ag-cyan"), ("Fading", counts["fading"], "ag-amber"),
                                    ("Overfit risk", counts["overfit"], "ag-red")):
            v, _ = stat_tile(label, color)
            v.set_text(f"{value:,}")
    missing = sum(1 for r in rows if not r["spark"])
    if rows and missing:
        ui.label(f"{missing} submitted alpha(s) have no daily PnL yet, so their outlook is incomplete. "
                 "Press Refresh data.").classes("ag-warn-box text-sm w-full")

    state = {"kind": "all", "outlook": "all"}
    with card("w-full"):
        with ui.row().classes("items-center w-full gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("Alphas").classes("ag-section-title")
                ui.label("Sorted riskiest first. Hover a column title or an outlook to see what it means.").classes(
                    "ag-muted text-xs")
            ui.space()
            segmented({"all": "All", "Regular": "Regular", "Power Pool": "Power Pool", "Super": "Super"}, "all",
                      lambda v: (state.update(kind=v), apply()))
            segmented({"all": "All", "overfit": "Overfit", "fading": "Fading", "fine": "Fine", "robust": "Robust"},
                      "all", lambda v: (state.update(outlook=v), apply()))
        grid = ui.aggrid(grid_options(COLUMNS, rows, tooltipShowDelay=300), theme="quartz",
                         auto_size_columns=False).classes(
            "w-full h-[600px]")
        if not rows:
            ui.label("No submitted alphas yet — sync your alphas (Sync with BRAIN → Sync my alphas), then press "
                     "Refresh data.").classes("ag-muted text-sm")

    def apply() -> None:
        grid.options["rowData"] = [r for r in rows if (state["kind"] == "all" or r["kind"] == state["kind"])
                                   and (state["outlook"] == "all" or r["outlook"] == state["outlook"])]
        grid.update()

    async def open_(e) -> None:  # noqa: ANN001
        aid = (e.args.get("data") or {}).get("id")
        if aid:
            await open_alpha(aid)

    grid.on("cellDoubleClicked", open_)

    with ui.expansion("BRAIN-reported out-of-sample numbers", icon="travel_explore").classes("w-full"):
        pts = [[r["is_sharpe"], r["brain_os"], r["id"]] for r in rows
               if r["is_sharpe"] is not None and r["brain_os"] is not None]
        if pts:
            hi = max(max(p[0] for p in pts), max(p[1] for p in pts), 1)
            ui.echart({
                "backgroundColor": "transparent",
                "tooltip": {":formatter": "p => `${p.value[2]}: IS ${p.value[0].toFixed(2)} / OS ${p.value[1].toFixed(2)}`"},
                "grid": {"left": 50, "right": 20, "top": 20, "bottom": 40},
                "xAxis": {"name": "IS", "type": "value", "axisLabel": {"color": "#8b98ad"},
                          "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
                "yAxis": {"name": "OS", "type": "value", "axisLabel": {"color": "#8b98ad"},
                          "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
                "series": [{"type": "scatter", "data": pts, "symbolSize": 8},
                           {"type": "line", "data": [[0, 0], [hi, hi]], "showSymbol": False,
                            "lineStyle": {"type": "dashed", "color": "#5eead4"}}],
            }).classes("w-full h-72")
        else:
            ui.label("BRAIN has not reported OS Sharpe for your alphas (or they are not synced).").classes(
                "ag-muted text-sm")

        async def before_after() -> None:
            sel = await grid.get_selected_rows()
            if not sel:
                ui.notify("Select an alpha in the table first.", type="warning")
                return
            try:
                data = await brain.alphas.before_after(sel[0]["id"])
                with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[800px]"):
                    ui.label(f"Before/after pool performance — {sel[0]['id']}").classes("font-semibold")
                    ui.json_editor({"content": {"json": data}}).classes("w-full")
                d.open()
            except BrainError as exc:
                notify_error(exc)

        ui.button("Before/after pool performance (selected)", icon="compare", on_click=before_after).props("flat")
