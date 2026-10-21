"""Dashboard — status of the BRAIN session, quota, AI research and recent activity (mirrors the reference app)."""

from __future__ import annotations

from nicegui import ui

from ...brain import brain
from ...config import settings
from ...engine.autopilot import autopilot
from ...engine.programs import genius_status, pyramid_candidates
from ...engine.simulator import simulator
from ...engine.stats import dashboard_numbers, pipeline_today
from ...events import bus
from ..components import ago, fmt_compact, fmt_int, kpi_tile, mini_tile


async def page() -> None:
    hello = ui.label().classes("text-[34px] font-semibold leading-tight whitespace-pre-line")

    with ui.row().classes("w-full gap-4 no-wrap"):
        sub_v, sub_s = kpi_tile("check_circle", "Submittable Alphas", "ag-teal", ("Open", lambda: ui.navigate.to("/alphas")))
        left_v, left_s = kpi_tile("bolt", "Simulations Left Today", "ag-amber")
        sent_v, sent_s = kpi_tile("send", "Sent Today", "ag-cyan")
        queue_v, queue_s = kpi_tile("hourglass_empty", "Queued", "ag-violet")
    sub_s.set_text("Passing every submission check")

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
        queued = simulator.queued_count
        hello.set_text(f"Welcome back.\n{fmt_int(n['sims_left'])} simulations left today."
                       + (f" {fmt_int(queued)} more are already queued." if queued else ""))
        sub_v.set_text(fmt_int(n["submittable"]))
        untested = n["submittable"] - n["ready_robust"] - n["ready_fragile"]
        detail = f"{n['ready_robust']} robust · {n['ready_fragile']} fragile · {untested} untested"
        sub_s.set_text("Passed BRAIN's check: " + detail + (f" · {n['recheck']} need a re-check" if n["recheck"]
                                                             else ""))
        ready_s.set_text(detail)
        left_v.set_text(f"{fmt_int(n['sims_left'])} / {fmt_int(n['sims_limit'])}")
        left_s.set_text(f"Resets in {int(brain.client.quota.seconds_to_reset // 3600)}h "
                        f"{int(brain.client.quota.seconds_to_reset % 3600 // 60)}m")
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
        sent_v.set_text(fmt_int(n["sent_today"]))
        pct = n["sent_today"] / n["sims_limit"] * 100 if n["sims_limit"] else 0
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
