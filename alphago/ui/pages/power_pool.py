"""Power Pool — pick the best Power-Pool-shaped submitted alphas per scope for a month, then tag them on BRAIN."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from nicegui import ui
from sqlalchemy import select

from ...brain import brain
from ...brain.errors import BrainError
from ...config import settings
from ...db import db
from ...db.models import Alpha
from ...engine import actions, portfolio
from ...engine.describe import describe_alpha
from ...engine.programs import PP_TAG, power_pool_eligibility
from ...jobs.queue import jobs
from ..alpha_detail import open_alpha
from ..components import (card, grid_options, info_box, money_col, notify_error, num_col, page_header, segmented,
                          sparkline_col, stat_tile)

RATING_RENDER = """params => {
  const k = {very_good: 'pass', good: 'running', average: 'pending', weak: 'fail'}[params.data.rating] || '';
  return '<span class="ag-badge ' + k + '">' + params.value + '</span>';
}"""
ALPHA_RENDER = """params => params.value + (params.data.tagged ? ' <span class="ag-badge running">tagged</span>' : '')"""

COLUMNS = [
    {"field": "id", "headerName": "Alpha", "width": 200, ":cellRenderer": ALPHA_RENDER, "pinned": "left"},
    {"field": "rating_label", "headerName": "Rating", "width": 130, ":cellRenderer": RATING_RENDER,
     "tooltipField": "why", "headerTooltip": "Hover a rating to see why."},
    {"field": "scope", "headerName": "Scope", "width": 105},
    {"field": "submitted", "headerName": "Submitted", "width": 125},
    {"field": "universe", "headerName": "Universe", "width": 120},
    sparkline_col("spark"),
    num_col("sharpe", "Sharpe", 2, 90),
    num_col("sharpe_2y", "Sharpe 2y", 2, 100),
    num_col("calmar", "Calmar", 2, 90),
    num_col("turnover", "Turnover", 1, 100, pct=True),
    money_col("pnl_1y", "Last 1y PnL", 120),
    {"field": "ops_fields", "headerName": "Ops / Fields", "width": 110},
    {**num_col("max_corr", "Max corr", 2, 100),
     "headerTooltip": "Highest daily-PnL correlation with an alpha picked before it in the same scope."},
]
MIN_RATING = {"very_good": "Very good+", "good": "Good+", "average": "Average+"}


def _month_label(m: str, n: int) -> str:
    return f"{datetime.strptime(m, '%Y-%m'):%b %Y} ({n})"


async def page() -> None:
    p = settings.programs
    months = await portfolio.pp_months()
    pick = await portfolio.saved_pick() or {}
    state = {"month": pick.get("month") or (months[0][0] if months else None), "tab": "picked",
             "rows": [], "picked": set(pick.get("picked") or [])}

    with ui.row().classes("w-full items-start no-wrap"):
        page_header("Power Pool", "Monthly competition for simple alphas (≤ 8 operators, ≤ 3 fields). Pick the best "
                                  "Power-Pool-shaped alphas you submitted that month, per scope, with low correlation "
                                  "to each other, then tag them PowerPoolSelected on BRAIN.")
        ui.space()

        async def refresh() -> None:
            if not brain.client.logged_in:
                ui.notify("Sign in to BRAIN first.", type="warning")
                return
            await jobs.submit("prepare_pnl", "Prepare data (submitted alphas + daily PnL)")
            ui.notify("Reading submitted alphas, tags and daily PnL — reopen this page when the task finishes.")

        with ui.row().classes("gap-2 no-wrap"):
            ui.button("Refresh data", icon="sync", on_click=refresh).props("outline")
            ui.button("Pick best alphas", icon="play_arrow", on_click=lambda: do_pick())

    # ------------------------------------------------------------ tags on BRAIN
    with card("w-full"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("sell").classes("ag-muted")
            ui.label("Tags on BRAIN").classes("ag-section-title")
        ui.label("Only alphas tagged PowerPoolSelected count for the month. Tag changes always ask first and run in "
                 "the background (see Tasks).").classes("ag-muted text-sm")
        with ui.row().classes("gap-2 mt-1"):
            ui.button("Read tags from BRAIN", icon="sync", on_click=lambda: read_tags()).props("outline dense")
            ui.button("Untag all", icon="close", on_click=lambda: confirm_tags([], tagged_ids())).props(
                "outline dense color=negative")
        tag_status = ui.label().classes("ag-muted text-sm")

    # ------------------------------------------------------------ settings
    with card("w-full"):
        with ui.row().classes("items-center gap-3 w-full"):
            ui.icon("tune").classes("ag-muted")
            ui.label("Settings").classes("ag-section-title")
            ui.space()
            month_sel = ui.select({m: _month_label(m, n) for m, n in months}, value=state["month"],
                                  label="Month").classes("w-44")
            segmented({5: "Max 5", 10: "Max 10", 15: "Max 15", 25: "Max 25"}, p.pp_max_per_scope,
                      lambda v: (setattr(p, "pp_max_per_scope", int(v)), settings.save()))
            segmented(MIN_RATING, p.pp_min_rating, lambda v: (setattr(p, "pp_min_rating", v), settings.save()))
        ui.label(f"Max = most alphas picked per scope. Rating floor = the weakest rating allowed. Picked alphas must "
                 f"correlate below {p.pp_max_corr} with each other (daily PnL), as the Power Pool rules require.").classes(
            "ag-muted text-sm")

    with ui.row().classes("w-full gap-4 no-wrap"):
        t_month, _ = stat_tile("Submitted that month")
        t_shape, _ = stat_tile("Power Pool shape")
        t_picked, _ = stat_tile("Picked as best", "ag-green")
        t_tagged, _ = stat_tile("Tagged on BRAIN")
        t_scopes, _ = stat_tile("Scopes picked", icon="public")

    # ------------------------------------------------------------ alphas
    with card("w-full"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("layers").classes("ag-muted")
            with ui.column().classes("gap-0"):
                ui.label("Alphas").classes("ag-section-title")
                picked_lbl = ui.label().classes("ag-muted text-xs")
        with ui.row().classes("items-center gap-2 mt-1 flex-wrap"):
            segmented({"picked": "Picked", "tagged": "Tagged", "shape": "PP shape", "all": "All"}, "picked",
                      lambda v: (state.update(tab=v), apply()))
            ui.button("Replace old with new pick", icon="sync", on_click=lambda: replace()).props("dense")
            tag_btn = ui.button(icon="sell", on_click=lambda: confirm_tags(sorted(state["picked"] - set(tagged_ids())),
                                                                          [])).props("outline dense")
            untag_btn = ui.button(icon="close", on_click=lambda: confirm_tags([], tagged_ids())).props(
                "outline dense")
        chips = ui.row().classes("gap-2 flex-wrap mt-2")
        grid = ui.aggrid(grid_options(COLUMNS, [], tooltipShowDelay=300), theme="quartz",
                         auto_size_columns=False).classes("w-full h-[520px]")

    async def open_(e) -> None:  # noqa: ANN001
        aid = (e.args.get("data") or {}).get("id")
        if aid:
            await open_alpha(aid, on_change=load)

    grid.on("cellClicked", open_)

    def tagged_ids() -> list[str]:
        return [r["id"] for r in state["rows"] if r["tagged"]]

    def apply() -> None:
        rows = state["rows"]
        tab = state["tab"]
        if tab == "picked":
            rows = [r for r in rows if r["id"] in state["picked"]]
        elif tab == "tagged":
            rows = [r for r in rows if r["tagged"]]
        elif tab == "shape":
            rows = [r for r in rows if r["shape"]]
        grid.options["rowData"] = rows
        grid.update()

    async def load() -> None:
        month = state["month"]
        state["rows"] = await portfolio.pp_rows(month) if month else []
        saved = await portfolio.saved_pick() or {}
        if month and saved.get("month") == month:
            state["picked"] = set(saved.get("picked") or [])
            mc = saved.get("max_corr") or {}
            for r in state["rows"]:
                r["max_corr"] = mc.get(r["id"])
            picked_lbl.set_text(f"Picked {datetime.fromisoformat(saved['created']).astimezone():%d/%m/%Y, %H:%M:%S}. "
                                "Click a row for its details; hover a rating to see why.")
        else:
            state["picked"] = set()
            picked_lbl.set_text("Not picked yet for this month — press Pick best alphas.")
        rows = state["rows"]
        t_month.set_text(str(len(rows)))
        t_shape.set_text(str(sum(1 for r in rows if r["shape"])))
        t_picked.set_text(str(len(state["picked"])))
        t_tagged.set_text(str(sum(1 for r in rows if r["tagged"])))
        t_scopes.set_text(str(len({r["scope"] for r in rows if r["id"] in state["picked"]})))
        n_new = len(state["picked"] - set(tagged_ids()))
        tag_btn.set_text(f"Tag all picked ({n_new})")
        untag_btn.set_text(f"Untag all ({len(tagged_ids())})")
        chips.clear()
        by_scope: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_scope[r["scope"]].append(r)
        with chips:
            for scope, items in sorted(by_scope.items()):
                ui.label(f"{scope}: {sum(1 for r in items if r['id'] in state['picked'])} picked · "
                         f"{sum(1 for r in items if r['tagged'])} tagged · {len(items)} total").classes(
                    "ag-src ag-mono text-xs")
        apply()

    async def on_month(e) -> None:  # noqa: ANN001
        state["month"] = e.value
        await load()

    month_sel.on_value_change(on_month)

    async def do_pick() -> None:
        if not state["month"]:
            ui.notify("No submitted alphas yet — press Refresh data.", type="warning")
            return
        res = await portfolio.pp_pick(state["month"], p.pp_max_per_scope, p.pp_min_rating, p.pp_max_corr)
        ui.notify(f"Picked {len(res['picked'])} alphas.", type="positive")
        await load()

    async def read_tags() -> None:
        ids = [r["id"] for r in state["rows"]]
        if not ids:
            return
        await jobs.submit("refresh_alphas", f"Read tags of {len(ids)} alphas", {"alpha_ids": ids})
        ui.notify("Reading tags from BRAIN (see Tasks).")

    def replace() -> None:
        tagged = set(tagged_ids())
        confirm_tags(sorted(state["picked"] - tagged), sorted(tagged - state["picked"]))

    def confirm_tags(tag: list[str], untag: list[str]) -> None:
        if not tag and not untag:
            ui.notify("Nothing to change.", type="info")
            return
        if not brain.client.logged_in:
            ui.notify("Sign in to BRAIN first.", type="warning")
            return
        with ui.dialog() as d, ui.card().classes("ag-card p-6 w-[560px]"):
            ui.label("Change Power Pool tags on BRAIN?").classes("text-lg font-semibold")
            if tag:
                ui.label(f"Tag {len(tag)} alpha(s) {PP_TAG}: {', '.join(tag[:12])}{' …' if len(tag) > 12 else ''}").classes(
                    "text-sm")
            if untag:
                ui.label(f"Remove the tag from {len(untag)} alpha(s): {', '.join(untag[:12])}"
                         f"{' …' if len(untag) > 12 else ''}").classes("text-sm")
            ui.label("Only tagged alphas count toward this month's Power Pool.").classes("ag-muted text-sm")

            async def go() -> None:
                d.close()
                await jobs.submit("pp_tags", f"Power Pool tags (+{len(tag)} / -{len(untag)})",
                                  {"tag": tag, "untag": untag})
                ui.notify("Tag changes queued — see Tasks.", type="positive")

            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=d.close).props("flat")
                ui.button("Apply", icon="check", on_click=go).props("color=primary")
        d.open()

    def tag_status_tick() -> None:
        running = jobs.is_running("pp_tags")
        tag_status.set_text("Tag change running — see Tasks." if running else "No tag change running.")

    ui.timer(2.0, tag_status_tick)

    # ------------------------------------------------------------ v1 eligibility scanner
    with ui.expansion("Eligibility check and descriptions (unsubmitted and submitted alphas)", icon="rule").classes(
            "w-full"):
        info_box("Power Pool rules", "Sharpe ≥ 1.0, ≤ 8 operators (repeats counted), ≤ 3 data fields (grouping fields "
                                     "excluded), turnover 1–70%, sub-universe test, risk-handled neutralization in "
                                     "USA/EUR/ASI/GLB/CHN, and a ≥ 100-character Idea & Rationale description.")
        with ui.row().classes("items-center w-full gap-3"):
            only = ui.checkbox("Eligible only", value=False)
            scope = ui.select({"all": "Unsubmitted + submitted", "UNSUBMITTED": "Unsubmitted", "ACTIVE": "Submitted"},
                              value="UNSUBMITTED", label="Alphas").classes("w-52")
            count = ui.label().classes("ag-muted text-sm")
            ui.space()
            ui.button("Write descriptions (AI) for selected", icon="auto_awesome", on_click=lambda: describe()).props(
                "flat")
            ui.button("Power-pool correlation", icon="hub", on_click=lambda: corr()).props("flat")

            def to_autopilot() -> None:
                settings.research.goal = "power_pool"
                settings.save()
                ui.navigate.to("/autopilot")

            ui.button("Open Autopilot with Power Pool goal", icon="smart_toy", on_click=to_autopilot).props("flat")
        egrid = ui.aggrid(grid_options([
            {"field": "id", "headerName": "Alpha", "width": 110},
            {"field": "eligible", "headerName": "Eligible", "width": 95},
            {"field": "tagged", "headerName": "Tagged", "width": 90},
            {"field": "status", "headerName": "Status", "width": 120},
            {"field": "region", "headerName": "Scope", "width": 90},
            num_col("sharpe", "Sharpe", 2, 90), num_col("turnover", "Turnover", 1, 100, pct=True),
            {"field": "ops", "headerName": "Ops ≤8", "width": 90},
            {"field": "fields", "headerName": "Fields ≤3", "width": 95},
            {"field": "reasons", "headerName": "Blocking reasons", "flex": 1, "minWidth": 300},
            {"field": "code", "headerName": "Expression", "flex": 1, "minWidth": 280, "cellClass": "ag-num"},
        ], [], multi=True), theme="quartz").classes("w-full h-[420px]")

        async def reload_scan() -> None:
            stmt = select(Alpha).where(Alpha.type == "REGULAR", Alpha.code.is_not(None), Alpha.sharpe >= 0.8)
            if scope.value != "all":
                stmt = stmt.where(Alpha.status == scope.value)
            async with db.session() as s:
                alphas = list(await s.scalars(stmt.order_by(Alpha.sharpe.desc()).limit(3000)))
            rows = []
            for a in alphas:
                r = power_pool_eligibility(a)
                if only.value and not r.eligible:
                    continue
                rows.append({"id": a.id, "eligible": "✓" if r.eligible else "", "tagged": "✓" if r.tagged else "",
                             "status": a.status, "region": f"{a.region}/D{a.delay}", "sharpe": a.sharpe,
                             "turnover": a.turnover, "ops": r.ops, "fields": r.fields,
                             "reasons": "; ".join(r.reasons), "code": (a.code or "")[:240]})
            egrid.options["rowData"] = rows
            egrid.update()
            count.set_text(f"{len(rows)} alphas · {sum(1 for r in rows if r['eligible'])} eligible")

        async def describe() -> None:
            ids = [r["id"] for r in await egrid.get_selected_rows()]
            if not ids:
                ui.notify("Select alphas first.", type="warning")
                return
            done = 0
            for aid in ids[:25]:
                try:
                    await actions.set_properties(aid, regular_description=await describe_alpha(aid))
                    done += 1
                except Exception as exc:
                    notify_error(exc, f"{aid}: ")
            ui.notify(f"Descriptions written for {done} alphas.", type="positive")
            await reload_scan()

        async def corr() -> None:
            for r in (await egrid.get_selected_rows())[:10]:
                try:
                    res = await brain.alphas.correlations(r["id"], "power-pool")
                    ui.notify(f"{r['id']}: max power-pool correlation {res.get('max')}", multi_line=True)
                except BrainError as exc:
                    notify_error(exc, f"{r['id']}: ")

        egrid.on("cellDoubleClicked", lambda e: open_alpha((e.args.get("data") or {}).get("id")) if (
            e.args.get("data") or {}).get("id") else None)
        only.on_value_change(lambda _: reload_scan())
        scope.on_value_change(lambda _: reload_scan())
        await reload_scan()

    tag_status_tick()
    await load()
