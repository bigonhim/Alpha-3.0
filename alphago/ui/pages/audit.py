"""Audit Log — every action the app took on your BRAIN account: who triggered it, what it spent, which alphas."""

from __future__ import annotations

from datetime import datetime

from nicegui import ui

from ...engine import audit
from ..components import card, grid_options, info_box, page_header, stat_tile

ACTIONS = ["all", "simulate", "check", "submit", "properties", "tag_list", "sign_in", "sign_out", "verification"]
COLUMNS = [
    {"field": "time", "headerName": "Time", "width": 160},
    {"field": "actor", "headerName": "Who", "width": 150},
    {"field": "action", "headerName": "Action", "width": 110},
    {"field": "status", "headerName": "HTTP", "width": 80},
    {"field": "quota_cost", "headerName": "Simulations", "width": 115, "type": "numericColumn"},
    {"field": "alphas", "headerName": "Alphas", "width": 220},
    {"field": "detail", "headerName": "Detail", "flex": 1, "minWidth": 240},
    {"field": "path", "headerName": "Endpoint", "width": 220},
]


async def page() -> None:
    page_header("Audit Log", "Every action AlphaGo took on your BRAIN account — simulations sent, checks, "
                             "submissions, property and tag changes, sign-ins — and who or what triggered it.")
    info_box("Read-only record", "Recorded at the BRAIN connection itself, so nothing that reaches your account is "
                                 "missed. Reads (GET) are not listed, except submission checks. Entries are never "
                                 "edited or deleted by the app. 'you' = a click in the app; 'you (confirmed)' = a "
                                 "confirmed submit / Osmosis / tag job; 'task:N' and 'batch:N' = research.")
    summary = await audit.summary(30)
    with ui.row().classes("w-full gap-4 no-wrap"):
        v, s = stat_tile("Simulations, last 30 days")
        v.set_text(f"{sum(r['simulations'] for r in summary['by_actor']):,}")
        v, s = stat_tile("Submissions, last 30 days", "ag-green")
        v.set_text(f"{summary['submissions']:,}")
        for r in sorted(summary["by_actor"], key=lambda r: -r["simulations"])[:3]:
            v, s = stat_tile(f"Simulations by {r['actor']}", "ag-cyan")
            v.set_text(f"{r['simulations']:,}")
    state = {"rows": []}
    with card("w-full"):
        with ui.row().classes("items-end gap-3 w-full"):
            action = ui.select(ACTIONS, value="all", label="Action").classes("w-40")
            actor = ui.input("Who (starts with)").classes("w-44")
            days = ui.select({1: "Today", 7: "7 days", 30: "30 days", 90: "90 days", 3650: "All"}, value=30,
                             label="Period").classes("w-32")
            count = ui.label().classes("ag-muted text-sm ml-auto")
        grid = ui.aggrid(grid_options(COLUMNS, []), theme="quartz").classes("w-full h-[600px]")

        async def reload() -> None:
            rows = await audit.entries(days=int(days.value), action=None if action.value == "all" else action.value,
                                       actor=actor.value or None)
            state["rows"] = rows
            grid.options["rowData"] = [{
                "time": r.ts.astimezone().strftime("%Y-%m-%d %H:%M:%S") if r.ts else "", "actor": r.actor,
                "action": r.action, "status": r.status, "quota_cost": r.quota_cost,
                "alphas": " ".join(r.alpha_ids or [])[:300], "detail": r.detail or "", "path": r.path}
                for r in rows]
            grid.update()
            count.set_text(f"{len(rows):,} entries")

        def export() -> None:
            ui.download(audit.to_csv(state["rows"]).encode("utf-8"), f"alphago_audit_{datetime.now():%Y%m%d_%H%M}.csv")

        for el in (action, days):
            el.on_value_change(lambda _: reload())
        actor.on("keydown.enter", lambda _: reload())
        with ui.row().classes("gap-2"):
            ui.button("Refresh", icon="refresh", on_click=reload).props("flat")
            ui.button("Export CSV", icon="download", on_click=export).props("flat")
    await reload()
