"""Dashboard — status of the BRAIN session, quota, AI research and recent activity (mirrors the reference app)."""

from __future__ import annotations

from nicegui import ui

from ...brain import brain
from ...catalog.today import simulations_today
from ...config import settings
from ...engine.autopilot import autopilot
from ...engine.programs import genius_status, pyramid_candidates
from ...engine.simulator import simulator
from ...engine.stats import dashboard_numbers, pipeline_today
from ...events import bus
from ..components import ago, fmt_compact, fmt_int, grid_options, kpi_tile, mini_tile


async def page() -> None:
    hello = ui.label().classes("text-[34px] font-semibold leading-tight whitespace-pre-line")
    await getting_started()

    with ui.row().classes("w-full gap-4 no-wrap"):
        sub_v, sub_s = kpi_tile("check_circle", "Submittable Alphas", "ag-teal", ("Open", lambda: ui.navigate.to("/alphas")))
        left_v, left_s, left_bar = sims_tile()
        sent_v, sent_s = kpi_tile("send", "Sent Today", "ag-cyan")
        queue_v, queue_s = kpi_tile("hourglass_empty", "Queued", "ag-violet")
    sub_s.set_text("Passing every submission check")
    await quarter_tiles()

    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        with ui.card().classes("ag-card ag-card-violet p-5 grow"):
            with ui.row().classes("items-center w-full"):
                ui.icon("psychology").classes("text-2xl ag-pink p-1 rounded-lg").style("border:1px solid #5b2140")
                ui.label("AI Research").classes("text-lg")
                state_badge = ui.label().classes("ag-badge")
                ui.space()
                ui.label("Alpha Autopilot").classes("ag-link").on("click", lambda: ui.navigate.to("/autopilot"))
            status_line = ui.label().classes("ag-muted text-sm")
            with ui.grid(columns=4).classes("w-full gap-4 mt-2"):
                ds_v, ds_s = mini_tile("storage", "Datasets Analyzed", "ag-cyan")
                fl_v, fl_s = mini_tile("storage", "Fields Analyzed", "ag-cyan")
                gen_v, gen_s = mini_tile("track_changes", "Alphas Generated Today", "ag-pink")
                sim_v, sim_s = mini_tile("bolt", "Alphas Simulated", "ag-amber")
                opt_v, opt_s = mini_tile("tune", "Optimization Attempts", "ag-cyan")
                strong_v, strong_s = mini_tile("layers", "Strong Candidates", "ag-violet")
                ready_v, ready_s = mini_tile("check_circle", "Ready for Submission", "ag-green")
                subm_v, subm_s = mini_tile("layers", "Submitted Alphas", "ag-violet")
                gs_v, gs_s = mini_tile("workspace_premium", "Genius Signals Completed", "ag-amber")
                gr_v, gr_s = mini_tile("workspace_premium", "Genius Signals Remaining", "ag-amber")
                pyr_v, pyr_s = mini_tile("change_history", "Pyramid Candidates", "ag-amber")
                rate_v, rate_s = mini_tile("track_changes", "AI Success Rate", "ag-pink")
                score_v, score_s = mini_tile("trending_up", "Average Alpha Score", "ag-cyan")
            ready_s.set_text("Review and submit from Alphas")
            rate_s.set_text("Promising or better")
            score_s.set_text("Your alphas, last 7 days (80 = at BRAIN's limits)")
            pipeline = ui.row().classes("gap-2 flex-wrap mt-3 items-center")

        with ui.card().classes("ag-card p-0 w-[520px] min-w-[420px]"):
            with ui.row().classes("items-center gap-3 px-5 pt-5 pb-3"):
                ui.icon("auto_awesome").classes("text-2xl ag-violet p-1 rounded-lg").style("border:1px solid #3b2f6b")
                ui.label("Recent AI activity").classes("text-lg")
            ui.separator().style("background:#1e2b40")
            feed = ui.column().classes("px-5 py-2 gap-0 w-full overflow-y-auto").style("max-height: 640px")

    with ui.card().classes("ag-card p-5 w-full"):
        with ui.row().classes("items-center w-full gap-3"):
            ui.label("Autopilot").classes("text-lg")
            ap_badge = ui.label().classes("ag-badge")
            ui.space()

            async def run_now() -> None:
                from ...engine.research import research
                if settings.research.mode == "manual":
                    ui.notify("Manual mode: switch to Assisted or Full Autopilot on the Autopilot page.", type="info")
                    return
                try:
                    ids = await research.plan_day()
                    ui.notify(f"Planned {len(ids)} research task(s)"
                              + (" and started them." if settings.research.mode == "autopilot"
                                 else " — approve them on the Autopilot page."), type="positive")
                except Exception as exc:
                    ui.notify(str(exc), type="negative")

            def toggle_auto() -> None:
                settings.research.auto_daily = not settings.research.auto_daily
                settings.save()

            ui.button("Run now", on_click=run_now).props("outline")
            toggle_btn = ui.button(on_click=toggle_auto).props("outline")
        ap_line = ui.label().classes("ag-muted text-sm")

    await work_in_flight()

    state = {"feed_version": -1}

    def render_feed() -> None:
        if state["feed_version"] == bus.version:
            return
        state["feed_version"] = bus.version
        feed.clear()
        events = bus.recent(40)
        with feed:
            if not events:
                ui.label("No activity yet. Sign in on 'Sync with BRAIN', sync a catalogue, then start the Autopilot "
                         "or a Simulation Matrix.").classes("ag-muted text-sm py-4")
            for ev in events:
                with ui.column().classes("ag-feed-item gap-1 w-full"):
                    with ui.row().classes("items-center gap-3"):
                        ui.label(ev.source).classes("ag-src")
                        ui.label(ago(ev.ts)).classes("ag-muted text-sm")
                    color = {"error": "ag-red", "warning": "ag-amber", "success": "ag-green"}.get(ev.level, "")
                    ui.label(ev.message).classes(f"text-[15px] leading-snug {color}")

    async def refresh() -> None:
        n = await dashboard_numbers()
        sims = simulations_today()
        hello.set_text(f"Welcome back.\n{sims.headline}")
        sub_v.set_text(fmt_int(n["submittable"]))
        untested = n["submittable"] - n["ready_robust"] - n["ready_fragile"]
        detail = f"{n['ready_robust']} robust · {n['ready_fragile']} fragile · {untested} untested"
        sub_s.set_text("Passed BRAIN's check: " + detail + (f" · {n['recheck']} need a re-check" if n["recheck"]
                                                             else ""))
        ready_s.set_text(detail)
        left_v.set_text(f"{sims.remaining_text} / {fmt_int(sims.limit)}")
        reset = sims.resets_in_seconds
        left_s.set_text(f"Resets in {reset // 3600}h {reset % 3600 // 60:02d}m"
                        + ("" if sims.exact else " · estimate until the first result today"))
        left_bar(sims)
        score_v.set_text("—" if n.get("avg_score") is None else f"{n['avg_score']:.1f}")
        pipe = await pipeline_today()
        pipeline.clear()
        with pipeline:
            ui.label("Today's pipeline:").classes("ag-muted text-sm")
            for k, v in pipe.items():
                kind = ("fail" if k in ("Rejected", "Failed") and v else "pending" if k in ("Simulating", "Optimizing")
                        and v else "pass" if k in ("Submission Ready", "Submitted") and v else "")
                ui.label(f"{k} {v}").classes(f"ag-badge {kind}")
        r = settings.research
        on = r.auto_daily and r.mode != "manual"
        ap_badge.set_text("On" if on else "Off")
        ap_badge.classes(replace="ag-badge " + ("running" if on else "stopped"))
        toggle_btn.set_text("Turn off" if r.auto_daily else "Turn on")
        ap_line.set_text("Plans and queues the day's research by itself right after your allowance resets"
                         + (" and starts it within the daily cap." if r.mode == "autopilot"
                            else "; proposed tasks wait for your approval." if r.mode == "assisted"
                            else " (Manual mode: nothing runs by itself).")
                         + " It never submits: you choose what to submit.")
        sent_v.set_text(fmt_int(sims.used))
        pct = sims.used_share * 100
        sent_s.set_text(f"{pct:.0f}% of the allowance")
        queue_v.set_text(fmt_int(simulator.queued_count))
        c = brain.client
        queue_s.set_text(f"{c.slots.in_use}/{c.slots.limit} cores busy"
                         + (f" · paused ({simulator.pause_reason})" if simulator.paused else ""))
        running = autopilot.running
        state_badge.set_text("Running" if running else "Stopped")
        state_badge.classes(replace="ag-badge " + ("running" if running else "stopped"))
        status_line.set_text(autopilot.status_line)
        ds_v.set_text(fmt_int(n["dataset_scopes"]))
        ds_s.set_text(f"{fmt_int(n['insights'])} with AI briefs")
        fl_v.set_text(fmt_compact(n["fields"]))
        gen_v.set_text(fmt_int(n["generated_today"]))
        sim_v.set_text(fmt_int(n["simulated_total"]))
        sim_s.set_text(f"{fmt_int(n['simulated_today'])} today")
        opt_v.set_text(fmt_int(n["optimizations"]))
        strong_v.set_text(fmt_int(n["strong"]))
        ready_v.set_text(fmt_int(n["submittable"]))
        subm_v.set_text(fmt_int(n["submitted"]))
        rate_v.set_text(f"{n['success_rate'] * 100:.2f}%")
        render_feed()

    async def refresh_slow() -> None:
        try:
            g = await genius_status()
            gs_v.set_text(fmt_int(g["signals"]))
            gs_s.set_text(f"{g['level'].title()} → {g['target'].title()}")
            gr_v.set_text(fmt_int(g["signals_remaining"]))
            gr_s.set_text(f"{g['days_left']} days left in quarter")
            pyr_v.set_text(fmt_int(len(await pyramid_candidates())))
            pyr_s.set_text("Would light an unlit pyramid")
        except Exception:
            pass

    await refresh()
    await refresh_slow()
    ui.timer(2.0, refresh)
    ui.timer(30.0, refresh_slow)


# ---------------------------------------------------------------- sections ported from alpha-harness' Dashboard

def sims_tile():
    """Simulations Left Today: "~" while the figure is estimated, a depletion bar, and a mark for how much of BRAIN's
    quota day has gone (a fill left of the mark is allowance falling behind the day). Returns (value, sub, update)."""
    with ui.card().classes("ag-card p-5 grow min-w-[220px]"):
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.icon("bolt").classes("ag-amber text-xl")
            ui.label("Simulations Left Today").classes("text-[15px]")
        value = ui.label("—").classes("ag-kpi mt-2")
        with ui.element("div").style("position:relative;height:6px;border-radius:3px;background:#1e2b40;"
                                     "margin:6px 0 4px 0") as track:
            fill = ui.element("div").style("height:6px;border-radius:3px;background:#8b5cf6;width:0%")
            mark = ui.element("div").style("position:absolute;top:-3px;left:0%;width:2px;height:12px;"
                                           "background:#e6edf7")
        sub = ui.label("").classes("ag-muted text-sm")

    def update(sims) -> None:  # noqa: ANN001  (catalog.today.Simulations)
        fill.style(replace=f"height:6px;border-radius:3px;background:#8b5cf6;width:{sims.used_share * 100:.1f}%")
        mark.style(replace=f"position:absolute;top:-3px;left:{sims.day_gone * 100:.1f}%;width:2px;height:12px;"
                           "background:#e6edf7")
        track.tooltip(f"{sims.used_share * 100:.0f}% of the allowance used · {sims.day_gone * 100:.0f}% of the quota "
                      "day gone")

    return value, sub, update


async def getting_started() -> None:
    """What a new consultant sets up first, ticked off from real state; hidden once everything is done."""
    from ...catalog.today import getting_started as steps_now
    box = ui.column().classes("w-full")
    seen = {"sig": None}

    async def refresh() -> None:
        steps = await steps_now()
        sig = tuple((s.key, s.done, s.detail) for s in steps)
        if sig == seen["sig"]:
            return
        seen["sig"] = sig
        box.clear()
        done = sum(s.done for s in steps)
        if done == len(steps):
            return
        nxt = next(s for s in steps if not s.done)
        with box, ui.card().classes("ag-card p-0 w-full"):
            with ui.row().classes("items-center gap-3 px-5 pt-4 pb-2"):
                ui.label("Getting Started").classes("text-lg")
                ui.label(f"{done} of {len(steps)} done").classes("ag-muted text-sm")
            for i, s in enumerate(steps):
                with ui.row().classes("items-center gap-4 px-5 py-3 w-full no-wrap").style(
                        "background:rgba(255,255,255,.03)" if s is nxt else ""):
                    ui.label("✓" if s.done else str(i + 1)).classes(
                        "w-7 h-7 rounded flex items-center justify-center text-sm "
                        + ("ag-green" if s.done else "")).style("border:1px solid #2a3850")
                    with ui.column().classes("gap-0 grow"):
                        ui.label(s.title).classes("text-[15px]" + (" ag-muted line-through" if s.done else ""))
                        if not s.done and s.detail:
                            ui.label(s.detail).classes("ag-muted text-xs")
                    if not s.done:
                        ui.button(s.action, on_click=lambda _e, t=s.target: ui.navigate.to(t)).props(
                            ("unelevated" if s is nxt else "outline") + " no-caps dense")

    await refresh()
    ui.timer(5.0, refresh)


async def quarter_tiles() -> None:
    """The quarter BRAIN judges on: alphas submitted, pyramids formulated, distinct operators used."""
    from ...catalog import quarter as quarter_mod
    held: dict = {"usage": None}

    def show_operators() -> None:
        use = held["usage"]
        if use is None:
            ui.notify("Operator usage is still loading (or you are signed out).", type="info")
            return
        with ui.dialog() as dlg, ui.card().classes("ag-card p-6 w-[760px] max-w-[95vw]"):
            ui.label("Operator Usage").classes("text-lg")
            ui.label(f"How many of the {use.alphas:,} alphas submitted in {use.label} use each operator, unused "
                     "first. ts_backfill and group_backfill are left out: BRAIN does not count them.").classes(
                "ag-muted text-sm")
            if use.unreadable:
                ui.label(f"{use.unreadable:,} submitted alpha(s) have an expression AlphaGo could not read, so they "
                         "are not counted.").classes("ag-amber text-sm")
            if use.unlisted:
                ui.label(f"{use.unlisted:,} submitted alpha(s) are past the most BRAIN lists at once, so they are not "
                         "counted.").classes("ag-amber text-sm")
            ui.aggrid(grid_options([
                {"field": "name", "headerName": "Operator", "flex": 2, "cellClass": "ag-mono"},
                {"field": "category", "headerName": "Category", "flex": 1},
                {"field": "alphas", "headerName": "Alphas", "width": 100, "type": "numericColumn"},
            ], [{"name": o.name, "category": o.category, "alphas": o.alphas} for o in use.operators],
                pagination=False), theme="quartz").classes("w-full h-[420px]")
            ui.label("BRAIN breaks level ties partly on the distinct operators your quarter's alphas use: the unused "
                     "ones are worth a try.").classes("ag-muted text-xs")
            ui.button("Close", on_click=dlg.close).props("flat").classes("self-end")
        dlg.open()

    with ui.row().classes("w-full gap-4 no-wrap"):
        q_v, q_s = kpi_tile("workspace_premium", "Submitted This Quarter", "ag-green")
        p_v, p_s = kpi_tile("change_history", "Pyramids Completed", "ag-amber",
                            ("Pyramids", lambda: ui.navigate.to("/pyramids")))
        o_v, o_s = kpi_tile("functions", "Operators Used", "ag-cyan", ("All Operators", show_operators))

    async def load() -> None:
        if not brain.client.logged_in:
            for sub in (q_s, p_s, o_s):
                sub.set_text("Sign in to BRAIN to count this quarter.")
            return
        st = await quarter_mod.standing()
        if st.errors and not st.submitted and not st.pyramids_formulated:
            q_s.set_text("The count could not load: " + "; ".join(st.errors))
        else:
            q_v.set_text(fmt_int(st.submitted))
            q_s.set_text(f"Submitted in {st.label} (BRAIN's own daily counts)")
            p_v.set_text(fmt_int(st.pyramids_formulated))
            p_s.set_text(f"{st.alphas_per_pyramid} alphas complete one"
                         + (f" · {st.pyramids_started} started" if st.pyramids_started else ""))
        try:
            use = await quarter_mod.operator_usage()
        except Exception as exc:
            o_s.set_text(f"Could not read this quarter's alphas: {getattr(exc, 'message', exc)}")
            return
        held["usage"] = use
        used = sum(1 for o in use.operators if o.alphas > 0)
        o_v.set_text(f"{used:,} / {len(use.operators):,}")
        o_s.set_text(f"used yet in {use.label}")

    ui.timer(0.2, load, once=True)
    ui.timer(600.0, load)


async def work_in_flight() -> None:
    """Every task holding queued simulations or cores, with a per-task Pause, and the background tasks."""
    from sqlalchemy import select

    from ...catalog import work
    from ...db import db
    from ...db.models import Job
    with ui.card().classes("ag-card p-5 w-full gap-2"):
        with ui.row().classes("items-center w-full"):
            ui.label("Work in Flight").classes("text-lg")
            ui.space()
            pause_all_btn = ui.button("Pause all", icon="pause").props("outline no-caps dense")
        table = ui.column().classes("w-full gap-1")
        ui.separator().style("background:#1e2b40")
        jobs_line = ui.label().classes("text-sm")
        jobs_box = ui.column().classes("w-full gap-1")
    seen: dict = {"sig": None, "waiting": 0}

    def confirm(title: str, waiting: int, action) -> None:  # noqa: ANN001
        with ui.dialog() as dlg, ui.card().classes("ag-card p-5"):
            ui.label(title).classes("text-lg")
            ui.label(f"Takes {waiting:,} waiting simulation(s) off the queue; they are sent when you run the task "
                     "again from Tasks. Anything already sent to BRAIN keeps running.").classes("text-sm")
            with ui.row().classes("self-end"):
                ui.button("Keep going", on_click=dlg.close).props("flat")

                async def go() -> None:
                    dlg.close()
                    await action()
                    seen["sig"] = None
                    await refresh()
                ui.button("Pause", on_click=go).props("unelevated")
        dlg.open()

    async def pause_all() -> None:
        n = await work.pause_all()
        ui.notify(f"Paused {n} task(s). Their queued simulations wait for Run on the Tasks page.", type="positive")

    pause_all_btn.on_click(lambda: confirm("Pause every task?", seen["waiting"], pause_all))

    async def refresh() -> None:
        rows = await work.rows()
        async with db.session() as s:
            jobs = list(await s.scalars(select(Job).where(Job.status.in_(["queued", "running", "failed"]))
                                        .order_by(Job.id.desc()).limit(8)))
        sig = (tuple((r.kind, r.task_id, r.status, r.in_flight, r.waiting) for r in rows),
               tuple((j.id, j.status, round(j.progress or 0, 2), j.message) for j in jobs))
        if sig == seen["sig"]:
            return
        seen["sig"] = sig
        seen["waiting"] = sum(r.waiting for r in rows if r.pausable)
        pause_all_btn.set_visibility(any(r.pausable for r in rows))
        table.clear()
        with table:
            if not rows:
                ui.label("Nothing is queued right now.").classes("ag-muted text-sm")
            else:
                with ui.row().classes("w-full ag-muted text-xs no-wrap"):
                    ui.label("Task").classes("grow")
                    ui.label("Status").classes("w-28")
                    ui.label("On BRAIN").classes("w-24 text-right")
                    ui.label("Waiting").classes("w-20 text-right")
                    ui.label("").classes("w-20")
            for r in rows:
                with ui.row().classes("w-full items-center no-wrap"):
                    ui.label(r.name).classes("grow text-sm truncate")
                    label = "Paused" if r.status.lower() == "paused" else "Running" if r.running else "Queued"
                    ui.label(label).classes("w-28 ag-badge " + ("running" if r.running else "stopped"))
                    ui.label(f"{r.in_flight:,} cores").classes("w-24 text-right ag-mono text-sm")
                    ui.label(f"{r.waiting:,}").classes("w-20 text-right ag-mono text-sm")
                    with ui.element("div").classes("w-20 text-right"):
                        if r.pausable:
                            ui.button("Pause", on_click=lambda _e, row=r: confirm(
                                f"Pause {row.name}?", row.waiting, lambda: work.pause(row))).props("flat dense no-caps")
        running = [j for j in jobs if j.status == "running"]
        queued = [j for j in jobs if j.status == "queued"]
        failed = [j for j in jobs if j.status == "failed"]
        jobs_line.set_text(f"Background Tasks: {len(running)} running · {len(queued)} queued"
                           + (f" · {len(failed)} failed recently" if failed else ""))
        jobs_box.clear()
        with jobs_box:
            for j in running + queued + failed[:3]:
                with ui.row().classes("w-full items-center gap-3 no-wrap"):
                    ui.label(j.status).classes("ag-badge " + ("fail" if j.status == "failed" else
                                                             "running" if j.status == "running" else "stopped"))
                    ui.label(j.title).classes("text-sm grow truncate")
                    if j.status == "running":
                        ui.linear_progress(value=j.progress or 0, show_value=False).classes("w-40")
                    ui.label(j.error or j.message or "").classes(
                        ("ag-red" if j.error else "ag-muted") + " text-xs max-w-[420px] truncate")

    await refresh()
    ui.timer(3.0, refresh)
