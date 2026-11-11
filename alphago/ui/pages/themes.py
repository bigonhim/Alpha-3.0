"""BRAIN Themes (GLB first): the theme's scope and dates, its data, one-click or daily research, and theme matches."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ...brain import brain
from ...config import ThemeDef, settings
from ...db import db
from ...db.models import Job
from ...engine import themes
from ...jobs.queue import jobs
from ..components import badge, card, grid_options, notify_error, num_col, page_header

THEME_HELP = ("BRAIN announces themes in the platform's notifications (bell icon) and publishes no theme API. Copy the "
              "theme's scope, dates and multiplier from the announcement here. BRAIN's check reports a match as "
              "MATCHES_THEMES; that is the only proof an alpha counts.")


async def page() -> None:
    cfg = settings.themes
    state: dict[str, Any] = {"job": None}
    page_header("BRAIN Themes", "Research the current BRAIN themes, starting with GLB: sync the theme's data, start or "
                "schedule theme research, and see which of your alphas BRAIN says match. Nothing is submitted here.")

    with card("w-full"):
        with ui.row().classes("items-center gap-3 w-full"):
            ui.label("Automation").classes("ag-section-title")
            auto = ui.switch("Plan one task per active theme after every quota reset", value=cfg.auto_daily)
            ui.space()
            sims = ui.number("Simulations per theme task", value=cfg.task_sims, min=20, step=20).classes("w-52")
            cores = ui.number("Cores per theme task", value=cfg.task_cores, min=1, max=8).classes("w-44")
        ui.label(f"Mode: {settings.research.mode}. Assisted: theme tasks wait for your approval on Autopilot / Tasks. "
                 "Full Autopilot: they start, within the Daily Autopilot cap. Manual: nothing is planned by itself. "
                 "Every screen on Tasks → Screening before simulating applies.").classes("ag-muted text-xs")

        def save_auto() -> None:
            cfg.auto_daily = bool(auto.value)
            cfg.task_sims, cfg.task_cores = int(sims.value or 300), int(cores.value or 2)
            settings.save()

        for el in (auto, sims, cores):
            el.on_value_change(lambda _: save_auto())

    progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
    progress.set_visibility(False)
    status = ui.label().classes("ag-muted text-sm")
    boxes = ui.column().classes("w-full gap-4")

    async def run_job(kind: str, title: str, params: dict) -> None:
        if state["job"]:
            ui.notify("A theme job is already running.", type="warning")
            return
        if not brain.client.logged_in:
            ui.notify("Sign in to BRAIN first (Sync with BRAIN).", type="warning")
            return
        state["job"] = await jobs.submit(kind, title, params)
        progress.set_visibility(True)
        status.set_text("Queued…")

    async def tick() -> None:
        jid = state["job"]
        if not jid:
            return
        async with db.session() as s:
            job = await s.get(Job, jid)
        if job is None:
            state["job"] = None
            return
        progress.set_value(job.progress or 0)
        status.set_text(job.message or job.status)
        if job.status in ("queued", "running"):
            return
        state["job"] = None
        progress.set_visibility(False)
        status.set_text(f"Failed: {job.error}" if job.status == "failed" else (job.message or "Done."))
        await render()

    ui.timer(1.5, tick)

    async def start_task(t: ThemeDef, start: bool) -> None:
        try:
            tid = await themes.plan_task(t, start=start, source="manual" if start else None)
        except Exception as exc:
            notify_error(exc, "")
            return
        ui.notify(f"Task #{tid} {'started' if start else 'planned'} for the {t.name} theme. Follow it on Tasks.",
                  type="positive")

    def theme_editor(t: ThemeDef) -> None:
        opts = brain.options
        with ui.row().classes("gap-3 items-end flex-wrap"):
            name = ui.input("Name", value=t.name).classes("w-32")
            enabled = ui.switch("Active", value=t.enabled)
            region = ui.select(opts.regions or ["GLB"], value=t.region if t.region in (opts.regions or [t.region])
                               else (opts.regions or ["GLB"])[0], label="Region").classes("w-28")
            delay = ui.select(opts.delays(region.value), value=t.delay if t.delay in opts.delays(region.value)
                              else opts.delays(region.value)[0], label="Delay").classes("w-24")
            unis = opts.universes(region.value)
            universe = ui.select(unis, value=t.universe if t.universe in unis else unis[0], label="Universe").classes(
                "w-36")
            neuts = ["", *opts.neutralizations(region.value)]
            neut = ui.select({n: n or "(category default)" for n in neuts},
                             value=t.neutralization if t.neutralization in neuts else "", label="Neutralization"
                             ).classes("w-52")
            decay = ui.number("Decay (-1 = default)", value=t.decay, min=-1, max=512).classes("w-40")
            start = ui.input("From (YYYY-MM-DD)", value=t.start).classes("w-40")
            end = ui.input("Until", value=t.end).classes("w-36")
            mult = ui.number("Multiplier", value=t.multiplier, min=0, step=0.1, format="%.2f").classes("w-28")
        ds = ui.input("Theme datasets (comma separated; empty = BRAIN's theme=true list)",
                      value=", ".join(t.dataset_ids)).classes("w-full")
        note = ui.input("Notes from the announcement", value=t.note).classes("w-full")

        def on_region(_: Any = None) -> None:
            delay.options = opts.delays(region.value)
            delay.value = delay.options[0]
            universe.options = opts.universes(region.value)
            universe.value = universe.options[0]
            neut.options = {n: n or "(category default)" for n in ["", *opts.neutralizations(region.value)]}
            delay.update(), universe.update(), neut.update()

        def save() -> None:
            t.name = (name.value or t.name).strip()[:40]
            t.enabled, t.region = bool(enabled.value), region.value
            t.delay, t.universe = int(delay.value), universe.value
            t.neutralization, t.decay = neut.value or "", int(decay.value if decay.value is not None else -1)
            t.start, t.end = (start.value or "").strip(), (end.value or "").strip()
            t.multiplier = float(mult.value or 1.0)
            t.dataset_ids = [x.strip() for x in (ds.value or "").split(",") if x.strip()]
            t.note = note.value or ""
            settings.save()

        region.on_value_change(lambda _: (on_region(), save()))
        for el in (name, enabled, delay, universe, neut, decay, start, end, mult, ds, note):
            el.on_value_change(lambda _: save())

    async def render() -> None:
        boxes.clear()
        with boxes:
            for t in list(cfg.items):
                summ = await themes.summary(t)
                ranked = await themes.ranked_datasets(t)
                with card("w-full"):
                    with ui.row().classes("items-center gap-3 w-full"):
                        ui.icon("public").classes("ag-cyan")
                        ui.label(f"{t.name} theme").classes("ag-section-title")
                        badge("active" if summ["active"] else "inactive", "pass" if summ["active"] else "")
                        ui.label(f"{t.region} D{t.delay} {t.universe}"
                                 + (f" · {t.neutralization}" if t.neutralization else "")
                                 + (f" · ×{t.multiplier:g}" if t.multiplier != 1 else "")).classes("ag-muted text-sm")
                        ui.space()
                        ui.button("Sync theme data", icon="sync",
                                  on_click=lambda _e, th=t: run_job("theme_sync", f"Theme data: {th.name}",
                                                                    {"theme": th.name})).props("outline")
                        ui.button("Plan task", icon="event",
                                  on_click=lambda _e, th=t: start_task(th, False)).props("outline").tooltip(
                            "Creates a planned task that waits for your approval on Tasks.")
                        ui.button("Start research now", icon="play_arrow",
                                  on_click=lambda _e, th=t: start_task(th, True)).props("color=primary").tooltip(
                            f"Starts a Search Lab task now: {cfg.task_sims} simulations on {cfg.task_cores} cores.")
                        ui.button("Check theme matches", icon="fact_check",
                                  on_click=lambda _e, th=t: run_job("theme_check", f"Theme check: {th.name}",
                                                                    {"theme": th.name})).props("flat").tooltip(
                            "BRAIN's check on strong / ready unsubmitted alphas of this scope (no simulations).")
                    with ui.row().classes("gap-6 text-sm"):
                        ui.label(f"{summ['alphas']} alphas in scope")
                        ui.label(" · ".join(f"{k} {v}" for k, v in sorted(summ["verdicts"].items())) or "no alphas yet")
                        ui.label(f"{summ['matched']} match a theme ({summ['matched_unsubmitted']} unsubmitted)").classes(
                            "ag-green" if summ["matched"] else "ag-muted")
                        ui.label(f"{summ['datasets']} theme datasets").classes("ag-muted")
                    with ui.expansion("Theme settings (from the announcement)", icon="tune").classes("w-full"):
                        theme_editor(t)
                    grid = ui.aggrid(grid_options([
                        {"field": "id", "headerName": "Dataset", "width": 150},
                        {"field": "name", "headerName": "Name", "flex": 1, "minWidth": 200},
                        {"field": "category", "headerName": "Category", "width": 140},
                        {"field": "fields", "headerName": "Fields", "width": 85},
                        num_col("ai_score", "AI score", 0, 95),
                        {"field": "tested", "headerName": "Tested", "width": 85},
                        num_col("pyramid", "Pyramid ×", 1, 95),
                    ], ranked[:50]), theme="quartz").classes("w-full h-[220px]")
                    if not ranked:
                        ui.label("No theme dataset with synced fields yet: press Sync theme data.").classes(
                            "ag-amber text-sm")
                    _ = grid
                    matched = [a for a in await themes.scope_alphas(t) if themes.theme_hits(a)]
                    if matched:
                        ui.label("Alphas BRAIN says match a theme").classes("text-sm font-semibold mt-2")
                        for a in sorted(matched, key=lambda x: -(x.sharpe or 0))[:12]:
                            with ui.row().classes("items-center gap-2 text-sm"):
                                badge(a.verdict or "—", "pass" if a.verdict in ("ready", "submitted") else "")
                                ui.label(f"{a.id} · Sharpe {a.sharpe or 0:.2f} · {', '.join(themes.theme_hits(a))}")
            ui.label(THEME_HELP).classes("ag-muted text-xs")

            async def add_theme() -> None:
                cfg.items.append(ThemeDef(name=f"Theme {len(cfg.items) + 1}", region="USA", universe="TOP3000",
                                          neutralization=""))
                settings.save()
                await render()

            async def remove_last() -> None:
                if len(cfg.items) > 1:
                    cfg.items.pop()
                    settings.save()
                await render()

            with ui.row().classes("gap-2"):
                ui.button("Add a theme", icon="add", on_click=add_theme).props("flat")
                if len(cfg.items) > 1:
                    ui.button("Remove the last theme", icon="remove", on_click=remove_last).props("flat color=negative")

    await render()
