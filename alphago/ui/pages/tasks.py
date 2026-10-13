"""Tasks — research tasks with their own cores, the Autopilot's daily plan, the simulation queue and background jobs."""

from __future__ import annotations

from nicegui import ui
from sqlalchemy import func, select

from ...brain import brain
from ...config import settings
from ...db import db
from ...db.models import Alpha, Candidate, Job
from ...engine.datasets import rated_datasets
from ...engine.research import KINDS, research, task_label
from ...engine.simulator import simulator
from ...jobs.queue import jobs
from ..alpha_detail import open_alpha
from ..components import badge, card, grid_options, notify_error, num_col, page_header, stat_tile

PROGRESS_RENDER = """params => {
  const d = params.data.done || 0, p = params.data.planned || 0, pct = p ? Math.min(100, d * 100 / p) : 0;
  return '<div style="display:flex;align-items:center;gap:10px"><div style="flex:1;height:6px;border-radius:3px;'
    + 'background:#1e2b40"><div style="width:' + pct + '%;height:6px;border-radius:3px;background:#8b5cf6"></div></div>'
    + '<span style="font-family:var(--ag-mono);min-width:96px;text-align:right">' + d.toLocaleString() + ' / '
    + p.toLocaleString() + '</span></div>';
}"""
STATUS_RENDER = """params => {
  const k = {running: 'running', complete: '', paused: 'pending', queued: 'pending', planned: 'pending',
             stopped: 'fail', failed: 'fail', skipped: 'warning'}[params.value] || '';
  return '<span class="ag-badge ' + k + '">' + params.value.charAt(0).toUpperCase() + params.value.slice(1) + '</span>';
}"""
TASK_RENDER = """params => '<span class="ag-badge running">' + (params.data.source === 'manual' ? 'Manual' : 'Autopilot')
  + '</span> ' + params.value"""

TASK_COLUMNS = [
    {"field": "id", "headerName": "#", "width": 70},
    {"field": "label", "headerName": "Task", "flex": 2, "minWidth": 360, ":cellRenderer": TASK_RENDER,
     "tooltipField": "reason"},
    {"field": "status", "headerName": "Status", "width": 120, ":cellRenderer": STATUS_RENDER, "tooltipField": "message"},
    {"field": "cores", "headerName": "Cores", "width": 85, "type": "numericColumn"},
    {"field": "done", "headerName": "Simulations", "flex": 1, "minWidth": 260, ":cellRenderer": PROGRESS_RENDER},
    {**num_col("best", "Best", 2, 90), "headerTooltip": "Best Sharpe found by the task"},
    {"field": "datasets", "headerName": "Datasets", "width": 220},
    {"field": "created", "headerName": "Created", "width": 120},
]
JOB_COLUMNS = [
    {"field": "id", "headerName": "#", "width": 70},
    {"field": "status", "headerName": "Status", "width": 100},
    {"field": "title", "headerName": "Job", "flex": 1, "minWidth": 260},
    {"field": "progress", "headerName": "Progress", "width": 100},
    {"field": "message", "headerName": "Result / message", "flex": 1, "minWidth": 260},
    {"field": "created", "headerName": "Created", "width": 140},
    {"field": "duration", "headerName": "Duration", "width": 100},
]


async def new_task_dialog(scope: dict | None = None, dataset_ids: list[str] | None = None, kind: str = "single",
                          on_done=None) -> None:  # noqa: ANN001
    """Create (and optionally start) a research task by hand."""
    opts = brain.options
    r = settings.research
    sc = scope or {"region": r.region, "delay": r.delay, "universe": r.universe}
    with ui.dialog() as d, ui.card().classes("ag-card p-6 w-[640px]"):
        ui.label("New research task").classes("text-lg font-semibold")
        k = ui.select({k: v for k, v in KINDS.items()}, value=kind if kind in KINDS else "single",
                      label="Kind").classes("w-full")
        with ui.row().classes("gap-2 w-full no-wrap"):
            region = ui.select(opts.regions, value=sc["region"] if sc["region"] in opts.regions else opts.regions[0],
                               label="Region").classes("w-28")
            delay = ui.select(opts.delays(region.value), value=sc["delay"] if sc["delay"] in opts.delays(region.value)
                              else opts.delays(region.value)[0], label="Delay").classes("w-24")
            universe = ui.select(opts.universes(region.value), label="Universe",
                                 value=sc["universe"] if sc["universe"] in opts.universes(region.value)
                                 else opts.universes(region.value)[0]).classes("grow")
        ds1 = ui.select({}, label="Dataset", with_input=True).classes("w-full")
        ds2 = ui.select({}, label="Second dataset (pairs)", with_input=True).classes("w-full")
        with ui.row().classes("gap-3"):
            sims = ui.number("Simulations", value=r.task_sims, min=10, step=50).classes("w-36")
            cores = ui.number("Cores (of 8)", value=r.task_cores, min=1, max=8).classes("w-32")
        run_now = ui.checkbox("Start now", value=True)
        ui.label("Spends simulations from your daily quota as it runs. Pause or stop it any time on Tasks.").classes(
            "ag-muted text-xs")

        async def load_ds() -> None:
            rows = await rated_datasets(region.value, int(delay.value), universe.value)
            options = {x["id"]: f"{x['id']} · {(x['name'] or '')[:40]} · {x['theme']}"
                                + (f" · AI {x['ai_score']:.0f}" if x["ai_score"] is not None else "")
                       for x in rows if (x["field_count"] or 0) > 0}
            ds1.options, ds2.options = options, options
            ids = dataset_ids or []
            ds1.value = ids[0] if ids and ids[0] in options else None
            ds2.value = ids[1] if len(ids) > 1 and ids[1] in options else None
            ds1.update()
            ds2.update()

        def on_kind() -> None:
            ds2.set_visibility(k.value == "pair")

        k.on_value_change(lambda _: on_kind())
        region.on_value_change(lambda _: load_ds())
        universe.on_value_change(lambda _: load_ds())

        async def create() -> None:
            ids = [x for x in (ds1.value, ds2.value if k.value == "pair" else None) if x]
            try:
                tid = await research.create(kind=k.value, region=region.value, delay=int(delay.value),
                                            universe=universe.value, dataset_ids=ids, planned_sims=int(sims.value or 0),
                                            cores=int(cores.value or 1), source="manual", status="queued",
                                            reason="Created by you.")
                if run_now.value:
                    await research.start(tid)
                d.close()
                ui.notify(f"Task #{tid} {'started' if run_now.value else 'queued'}.", type="positive")
                if on_done:
                    await on_done()
            except Exception as exc:
                notify_error(exc)

        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=d.close).props("flat")
            ui.button("Create", icon="science", on_click=create).props("color=primary")
    on_kind()
    await load_ds()
    d.open()


async def page() -> None:
    with ui.row().classes("w-full items-start no-wrap"):
        page_header("Tasks", "Research tasks share BRAIN's concurrent simulation slots (cores). Pause, stop or delete "
                             "any task; the queue and background jobs are below.")
        ui.space()

        async def stop_all() -> None:
            n = await research.stop_all()
            ui.notify(f"Stopped: Autopilot off, {n} task(s) and the queue paused.", type="warning")
            await refresh(force=True)

        async def run_all() -> None:
            try:
                n = await research.run_all()
                ui.notify(f"Running {n} task(s); simulation queue resumed.", type="positive")
            except Exception as exc:
                notify_error(exc)
            await refresh(force=True)

        with ui.row().classes("gap-2 no-wrap"):
            ui.button("Stop All", icon="stop", on_click=stop_all).props("outline color=negative")
            ui.button("Run All", icon="play_arrow", on_click=run_all)

    with ui.row().classes("w-full gap-4 no-wrap"):
        t_run, t_run_sub = stat_tile("Running Cores", icon="memory")
        t_assigned, _ = stat_tile("Cores Assigned", icon="memory")
        t_planned, _ = stat_tile("Simulations Assigned", icon="bolt")
        t_done, _ = stat_tile("Simulated", icon="bolt")

    with card("w-full"):
        with ui.row().classes("items-center w-full gap-2"):
            ui.label("Autopilot").classes("ag-section-title")
            ap_badge = ui.label().classes("ag-badge")
            ui.space()
            ui.label("Settings").classes("ag-link").on("click", lambda: ui.navigate.to("/autopilot"))
        ap_line = ui.label().classes("ag-muted text-sm")
        plan_box = ui.column().classes("w-full gap-1 mt-1")

    with card("w-full"):
        with ui.row().classes("items-center w-full gap-2"):
            ui.label("Research tasks").classes("ag-section-title")
            ui.label("Tick tasks, then use the buttons. Double-click a task to list its alphas.").classes(
                "ag-muted text-sm")
            ui.space()
            ui.button("New task…", icon="add", on_click=lambda: new_task_dialog(on_done=lambda: refresh(True))).props(
                "dense")
            ui.button("Approve", icon="check", on_click=lambda: act("approve")).props("flat dense")
            ui.button("Pause", icon="pause", on_click=lambda: act("pause")).props("flat dense")
            ui.button("Resume", icon="play_arrow", on_click=lambda: act("resume")).props("flat dense")
            ui.button("Stop", icon="stop", on_click=lambda: act("stop")).props("flat dense color=negative")
            ui.button("Delete", icon="delete", on_click=lambda: act("delete")).props("flat dense")
        grid = ui.aggrid(grid_options(TASK_COLUMNS, [], multi=True, tooltipShowDelay=300), theme="quartz").classes(
            "w-full h-[460px]")
        detail_title = ui.label().classes("ag-section-title mt-2")
        detail = ui.aggrid(grid_options([
            {"field": "alpha_id", "headerName": "Alpha", "width": 110},
            {"field": "verdict", "headerName": "Verdict", "width": 100},
            num_col("sharpe", "Sharpe", 2, 90), num_col("fitness", "Fitness", 2, 90),
            num_col("turnover", "Turnover", 1, 100, pct=True),
            {"field": "origin", "headerName": "Source", "width": 100},
            {"field": "expression", "headerName": "Expression", "flex": 1, "minWidth": 400, "cellClass": "ag-num"},
        ], []), theme="quartz").classes("w-full h-[300px]")
        detail.set_visibility(False)

    with ui.expansion("Simulation queue and background jobs", icon="checklist", value=True).classes("w-full"):
        with ui.row().classes("w-full gap-5"):
            with card("grow"):
                ui.label("Simulation queue").classes("ag-section-title")
                qline = ui.label().classes("text-sm")
                with ui.row().classes("gap-2"):
                    pause_btn = ui.button("Pause queue", icon="pause")
                    ui.button("Cancel all queued", icon="block", on_click=lambda: cancel_all()).props(
                        "flat color=negative")
            with card("w-[420px]"):
                ui.label("Candidate statuses").classes("ag-section-title")
                stats = ui.column().classes("gap-0")
        with ui.row().classes("items-center w-full"):
            ui.label("Jobs").classes("ag-section-title")
            ui.space()
            ui.button("Cancel selected", icon="cancel", on_click=lambda: job_act("cancel")).props("flat")
            ui.button("Retry selected", icon="replay", on_click=lambda: job_act("retry")).props("flat")
        jgrid = ui.aggrid(grid_options(JOB_COLUMNS, []), theme="quartz").classes("w-full h-[360px]")

    def toggle_queue() -> None:
        if simulator.paused:
            simulator.resume()
        else:
            simulator.pause("paused by user")

    pause_btn.on_click(toggle_queue)

    async def cancel_all() -> None:
        n = await simulator.cancel_queued()
        ui.notify(f"Cancelled {n} queued simulations.")

    async def act(what: str) -> None:
        rows = await grid.get_selected_rows()
        if not rows:
            ui.notify("Tick one or more tasks first.", type="warning")
            return
        done = 0
        for r in rows:
            try:
                if what == "approve" and r["status"] == "planned":
                    await research.approve(r["id"])
                elif what == "pause" and r["status"] in ("running", "queued"):
                    await research.pause(r["id"])
                elif what == "resume" and r["status"] in ("paused", "queued"):
                    await research.resume(r["id"])
                elif what == "stop" and r["status"] not in ("complete", "stopped", "skipped", "failed"):
                    await research.stop(r["id"])
                elif what == "delete":
                    if not await research.delete(r["id"]):
                        continue
                else:
                    continue
                done += 1
            except Exception as exc:
                notify_error(exc, f"Task #{r['id']}: ")
        ui.notify(f"{what.title()}: {done} task(s).")
        await refresh(force=True)

    async def job_act(what: str) -> None:
        for r in await jgrid.get_selected_rows():
            await (jobs.cancel(r["id"]) if what == "cancel" else jobs.retry(r["id"]))
        ui.notify(f"{what.title()} requested.")

    async def show_task(e) -> None:  # noqa: ANN001
        tid = (e.args.get("data") or {}).get("id")
        if not tid:
            return
        async with db.session() as s:
            cands = list(await s.scalars(select(Candidate).where(Candidate.task_id == tid, Candidate.alpha_id.is_not(None))
                                         .order_by(Candidate.id.desc()).limit(2000)))
            alphas = {a.id: a for a in await s.scalars(select(Alpha).where(
                Alpha.id.in_([c.alpha_id for c in cands])))} if cands else {}
        rows = []
        for c in cands:
            a = alphas.get(c.alpha_id)
            rows.append({"alpha_id": c.alpha_id, "verdict": a.verdict if a else "", "sharpe": a.sharpe if a else None,
                         "fitness": a.fitness if a else None, "turnover": a.turnover if a else None,
                         "origin": c.origin, "expression": c.expression})
        rows.sort(key=lambda x: -(x["sharpe"] or -99))
        detail.options["rowData"] = rows
        detail.update()
        detail.set_visibility(True)
        detail_title.set_text(f"Task #{tid}: {len(rows)} alphas (best first)")

    grid.on("cellDoubleClicked", show_task)
    detail.on("cellDoubleClicked", lambda e: open_alpha((e.args.get("data") or {}).get("alpha_id"))
              if (e.args.get("data") or {}).get("alpha_id") else None)
    state = {"rv": -1, "jv": -1}

    async def refresh(force: bool = False) -> None:
        c = brain.client
        tasks_ = await research.list()
        active = [t for t in tasks_ if t.status in ("running", "queued", "paused")]
        t_run.set_text(f"{simulator.cores_in_use()} / {c.slots.limit}")
        t_run_sub.set_text(f"{c.slots.in_use} slots busy in total")
        t_assigned.set_text(str(sum(t.cores for t in active if t.status == "running")))
        t_planned.set_text(f"{sum(t.planned_sims for t in active):,}")
        t_done.set_text(f"{sum(t.n_simulated for t in active):,}")
        r = settings.research
        ap_badge.set_text("On" if r.auto_daily else "Off")
        ap_badge.classes(replace="ag-badge " + ("running" if r.auto_daily else "stopped"))
        ap_line.set_text(f"{ {'manual': 'Manual', 'assisted': 'Assisted', 'autopilot': 'Full Autopilot'}.get(r.mode, r.mode)}"
                         f" · {r.region} D{r.delay} {r.universe} · focus {r.research_focus or 'automatic'} · "
                         f"{r.tasks_per_day} task(s)/day, {r.task_sims} simulations each")
        if force or state["rv"] != research.version:
            state["rv"] = research.version
            planned = [t for t in tasks_ if t.source != "manual"][:8]
            plan_box.clear()
            with plan_box:
                if not planned:
                    ui.label("No Autopilot plans yet. Plans appear here after the daily quota reset when the Autopilot "
                             "is on, or when you press 'Plan next task' on the Autopilot page.").classes(
                        "ag-muted text-sm")
                for t in planned:
                    with ui.row().classes("items-center gap-3 no-wrap"):
                        ui.label(t.plan_day or "").classes("ag-mono text-sm ag-muted w-24")
                        badge(t.status, {"skipped": "warning", "planned": "pending", "running": "running",
                                         "complete": ""}.get(t.status, "pending"))
                        ui.label(f"{task_label(t)} · {t.planned_sims:,} simulations").classes("text-sm")
                        for ds in t.dataset_ids or []:
                            ui.label(ds).classes("ag-src ag-mono text-xs")
            grid.options["rowData"] = [{
                "id": t.id, "label": task_label(t), "status": t.status, "cores": t.cores, "done": t.n_simulated,
                "planned": t.planned_sims, "best": t.best_sharpe, "datasets": ", ".join(t.dataset_ids or []),
                "created": f"{t.created_at.astimezone():%m-%d %H:%M}", "source": t.source,
                "reason": t.reason or "", "message": t.message or ""} for t in tasks_]
            grid.update()
        paused = f" · PAUSED ({simulator.pause_reason})" if simulator.paused else ""
        qline.set_text(f"{simulator.queued_count} queued · {simulator.running_count} simulating · slots "
                       f"{c.slots.in_use}/{c.slots.limit} (cap {c.slots.capacity}){paused}")
        pause_btn.set_text("Resume queue" if simulator.paused else "Pause queue")
        pause_btn.props(f"icon={'play_arrow' if simulator.paused else 'pause'}")

    async def refresh_slow() -> None:
        async with db.session() as s:
            counts = dict((await s.execute(select(Candidate.status, func.count()).group_by(Candidate.status))).all())
        stats.clear()
        with stats:
            for k in ("new", "approved", "queued", "simulating", "done", "reused", "error", "invalid", "cancelled"):
                ui.label(f"{k:<11} {counts.get(k, 0):>8,}").classes("ag-mono text-sm whitespace-pre")
        if state["jv"] == jobs.version:
            return
        state["jv"] = jobs.version
        async with db.session() as s:
            rows = list(await s.scalars(select(Job).order_by(Job.id.desc()).limit(300)))
        jgrid.options["rowData"] = [{
            "id": j.id, "status": j.status, "title": j.title, "progress": f"{j.progress * 100:.0f}%",
            "message": j.error or j.message or "", "created": f"{j.created_at:%m-%d %H:%M:%S}",
            "duration": f"{(j.finished_at - j.started_at).total_seconds():.0f}s"
            if j.started_at and j.finished_at else ""} for j in rows]
        jgrid.update()

    await refresh(force=True)
    await refresh_slow()
    ui.timer(2.0, refresh)
    ui.timer(3.0, refresh_slow)
