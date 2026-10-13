"""Application shell: header status chips, sidebar navigation and client-side routed pages."""

from __future__ import annotations

from typing import Callable

from nicegui import ui

from ..brain import brain
from .components import fmt_duration, fmt_int
from .theme import SIDEBAR_WIDTH, apply_theme


class NavItem:
    def __init__(self, path: str, label: str, icon: str, color: str, builder: Callable) -> None:
        self.path, self.label, self.icon, self.color, self.builder = path, label, icon, color, builder


def _nav() -> list[NavItem]:
    from .pages import (alphas, analytics, autopilot, dashboard, dataset_intel, genius, llm, osmosis, osmosis_review,
                        oos, power_pool, research_history, simulation_matrix, spc_lab, super_alphas, sync, tasks)
    return [
        NavItem("/", "Dashboard", "space_dashboard", "#a78bfa", dashboard.page),
        NavItem("/matrix", "Simulation Matrix", "grid_view", "#818cf8", simulation_matrix.page),
        NavItem("/datasets", "Dataset Intelligence", "psychology", "#f472b6", dataset_intel.page),
        NavItem("/autopilot", "Alpha Autopilot", "smart_toy", "#7dd3fc", autopilot.page),
        NavItem("/genius", "Genius AI", "workspace_premium", "#fbbf24", genius.page),
        NavItem("/osmosis", "Osmosis", "water_drop", "#5eead4", osmosis.page),
        NavItem("/osmosis-review", "Osmosis Review", "fact_check", "#f472b6", osmosis_review.page),
        NavItem("/power-pool", "Power Pool", "diamond", "#c084fc", power_pool.page),
        NavItem("/oos", "Out of Sample", "travel_explore", "#7dd3fc", oos.page),
        NavItem("/spc", "SPC Lab", "emoji_events", "#fbbf24", spc_lab.page),
        NavItem("/super", "Super Alphas", "hub", "#a78bfa", super_alphas.page),
        NavItem("/tasks", "Tasks", "checklist", "#5eead4", tasks.page),
        NavItem("/alphas", "Alphas", "layers", "#818cf8", alphas.page),
        NavItem("/history", "Research History", "history", "#7dd3fc", research_history.page),
        NavItem("/analytics", "Analytics", "insights", "#818cf8", analytics.page),
        NavItem("/llm", "LLM Integration", "auto_awesome", "#c084fc", llm.page),
        NavItem("/sync", "Sync with BRAIN", "sync", "#4ade80", sync.page),
    ]


def _status_chips() -> None:
    from ..engine.simulator import simulator

    session = ui.label().classes("ag-chip")
    left = ui.label().classes("ag-chip")
    queued = ui.label().classes("ag-chip")

    async def stop_all() -> None:
        from ..engine.research import research
        n = await research.stop_all()
        ui.notify(f"Stopped: Autopilot off, {n} task(s) and the simulation queue paused. Run All on Tasks resumes.",
                  type="warning", multi_line=True)

    ui.button("Stop all", icon="stop_circle", on_click=stop_all).props(
        "outline dense no-caps color=negative").classes("ag-stop-all").tooltip(
        "Stop the Autopilot and pause every task and the simulation queue. Nothing is deleted.")
    reset = ui.label().classes("ag-chip")

    def refresh() -> None:
        c = brain.client
        if c.state == "persona_required":
            session.set_text("⚠ BRAIN verification required")
            session.classes(replace="ag-chip warn")
        elif c.logged_in:
            session.set_text(f"🛡 BRAIN Session  {fmt_duration(c.seconds_left)}")
            session.classes(replace="ag-chip" if c.seconds_left > 1800 else "ag-chip warn")
        else:
            session.set_text("🛡 BRAIN  signed out")
            session.classes(replace="ag-chip bad")
        q = c.quota
        left.set_text(f"⚡ {fmt_int(q.effective_remaining)}  Simulations Left Today")
        left.classes(replace="ag-chip" if q.effective_remaining > 500 else "ag-chip warn")
        queued.set_text(f"☰ {fmt_int(simulator.queued_count)}  Queued · {c.slots.in_use}/{c.slots.limit} slots")
        reset.set_text(f"⏱ Simulation Quota Reset in  {fmt_duration(q.seconds_to_reset)}")

    refresh()
    ui.timer(1.0, refresh)


def root() -> None:
    apply_theme()
    items = _nav()
    nav_rows: dict[str, ui.row] = {}

    def go(path: str) -> None:
        ui.navigate.to(path)

    with ui.header().classes("ag-header items-center px-5 py-2 gap-3"):
        ui.space()
        _status_chips()
        ui.button(icon="search", on_click=lambda: go("/alphas")).props("flat round dense color=grey-5")

    with ui.left_drawer(value=True, fixed=True).props(f"width={SIDEBAR_WIDTH} :breakpoint=0").classes("ag-drawer p-3"):
        with ui.row().classes("items-center gap-2 px-2 pb-4 pt-1 no-wrap"):
            ui.icon("change_history").classes("text-3xl").style("color:#8b5cf6")
            ui.label("AlphaGo").classes("text-xl font-semibold")
            ui.label("Quant").classes("text-xl font-semibold ag-cyan")
        badges: dict[str, ui.label] = {}
        with ui.column().classes("w-full gap-1"):
            for item in items:
                with ui.row().classes("ag-nav-item items-center gap-3 no-wrap w-full").on(
                        "click", lambda _e, p=item.path: go(p)) as row:
                    ui.icon(item.icon).style(f"color:{item.color}").classes("text-[20px]")
                    ui.label(item.label).classes("text-[15px]")
                    if item.path == "/alphas":
                        badges[item.path] = ui.label().classes("ag-nav-badge").tooltip("Ready to submit")
                nav_rows[item.path] = row
        ui.space()
        with ui.row().classes("items-center gap-3 no-wrap px-2 pt-4 w-full").style(
                "border-top:1px solid var(--ag-border)"):
            avatar = ui.label().classes("rounded-full w-8 h-8 flex items-center justify-center text-sm font-semibold") \
                .style("background:#5b8def;color:#fff")
            with ui.column().classes("gap-0 min-w-0"):
                who = ui.label().classes("text-sm")
                mail = ui.label().classes("ag-muted text-xs truncate")

        async def refresh_side() -> None:
            from sqlalchemy import func, select

            from .. import credentials
            from ..db import db
            from ..db.models import Alpha
            async with db.session() as s:
                n = await s.scalar(select(func.count(Alpha.id)).where(Alpha.verdict == "ready",
                                                                      Alpha.status == "UNSUBMITTED")) or 0
            if "/alphas" in badges:
                badges["/alphas"].set_text(str(n) if n else "")
            email = credentials.brain_email() or ""
            who.set_text(brain.client.user_id or ("Signed in" if brain.client.logged_in else "Not signed in"))
            mail.set_text(email)
            avatar.set_text((email[:1] or "?").upper())

        ui.timer(0.1, refresh_side, once=True)
        ui.timer(15.0, refresh_side)
        def highlight(path: str) -> None:
            path = (path or "/").split("?", 1)[0]
            for p, row in nav_rows.items():
                if p == path:
                    row.classes(add="active")
                else:
                    row.classes(remove="active")

        router = ui.context.client.sub_pages_router
        highlight(router.current_path)
        router.on_path_changed(highlight)

    with ui.column().classes("w-full px-8 py-6 gap-5"):
        ui.sub_pages({item.path: item.builder for item in items}).classes("w-full")
