"""Alpha Autopilot — modes, daily research planning, proposed tasks, the current task, and AI idea batches."""

from __future__ import annotations

from nicegui import ui
from sqlalchemy import select

from ...brain import brain
from ...config import settings
from ...db import db
from ...db.models import Batch, Candidate, ResearchTask
from ...engine.autopilot import GOALS, AutopilotConfig, autopilot
from ...engine.datasets import scored_datasets
from ...engine.research import KINDS, STAGES, autopilot_sims_today, research
from ...llm.schemas import THEMES
from ..components import badge, card, grid_options, notify_error, page_header, stage_bar, stat_tile
from .simulation_matrix import RESULT_COLUMNS, batch_rows

AUTOPILOT_WARNING = (
    "Full Autopilot plans and runs research and simulates alpha ideas without your review. A copy of BRAIN's Consultant "
    "Do's and Don'ts lists 'using automation to generate Alpha ideas' as a research don't, and community reports "
    "describe accounts locked after fully automated mining. Check the current rules on the platform before using "
    "this mode. Submission always stays manual.")
MODE_LABEL = {"manual": "Manual", "assisted": "Assisted", "autopilot": "Full Autopilot"}
MODE_SUB = {"manual": "You start everything", "assisted": "Proposes; waits for your approval",
            "autopilot": "Running by itself"}

HISTORY_COLUMNS = [
    {"field": "id", "headerName": "Batch", "width": 80},
    {"field": "created", "headerName": "Created", "width": 130},
    {"field": "dataset", "headerName": "Dataset", "width": 150},
    {"field": "strategy", "headerName": "Move / goal", "width": 170},
    {"field": "status", "headerName": "Status", "width": 150},
    {"field": "generated", "headerName": "Gen", "width": 75},
    {"field": "simulated", "headerName": "Sim", "width": 75},
    {"field": "reused", "headerName": "Reused", "width": 85},
    {"field": "promising", "headerName": "Promising+", "width": 110},
    {"field": "strong", "headerName": "Strong", "width": 85},
    {"field": "ready", "headerName": "Ready", "width": 80},
    {"field": "rejected", "headerName": "Rejected", "width": 95},
]
APPROVAL_COLUMNS = [
    {"field": "cid", "headerName": "#", "width": 80},
    {"field": "origin", "headerName": "Source", "width": 100},
    {"field": "expression", "headerName": "Expression (editable)", "flex": 2, "minWidth": 360, "editable": True,
     "cellClass": "ag-num"},
    {"field": "hypothesis", "headerName": "Hypothesis", "flex": 1, "minWidth": 260},
    {"field": "neutralization", "headerName": "Neutralization", "width": 140},
    {"field": "decay", "headerName": "Decay", "width": 80},
    {"field": "ops", "headerName": "Ops", "width": 70},
]


async def confirm_full_autopilot() -> bool:
    if settings.autopilot_warning_accepted:
        return True
    with ui.dialog() as dialog, ui.card().classes("ag-card p-6 w-[640px]"):
        ui.label("Enable Full Autopilot?").classes("text-lg font-semibold")
        ui.label(AUTOPILOT_WARNING).classes("text-sm")
        accept = ui.checkbox("I have checked BRAIN's current rules and accept the risk for my account.")
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat")
            ui.button("Enable Full Autopilot", on_click=lambda: dialog.submit(bool(accept.value))).props(
                "color=warning")
    ok = await dialog
    if ok:
        settings.autopilot_warning_accepted = True
        settings.save()
    return bool(ok)


def autopilot_active() -> bool:
    return settings.research.auto_daily or autopilot.running


async def page() -> None:
    r = settings.research
    opts = brain.options

    with ui.row().classes("w-full items-start no-wrap"):
        page_header("Alpha Autopilot", "Plans research by itself — analyse, rank, generate, simulate, optimise, select — "
                                       "within the safety limits below. Best candidates end at Ready for Manual "
                                       "Submission.")
        ui.space()
        with ui.row().classes("gap-2 no-wrap"):
            resume_btn = ui.button("Resume Autopilot", icon="replay").props("outline")
            ui.button("STOP AUTOPILOT", icon="cancel", on_click=lambda: stop_autopilot()).props(
                "outline color=negative")

    banner = ui.row().classes("w-full items-start gap-3 p-4 rounded-xl")
    with ui.row().classes("w-full gap-4 no-wrap"):
        t_mode, t_mode_sub = stat_tile("Mode")
        t_cap, t_cap_sub = stat_tile("Autopilot simulations today", "ag-amber", icon="bolt")
        t_active, _ = stat_tile("Active tasks")
        t_last, t_last_sub = stat_tile("Last task")
        t_sub, t_sub_sub = stat_tile("Submissions")
    t_sub.set_text("Manual")
    t_sub_sub.set_text("AI never submits")

    # ------------------------------------------------------------ mode + settings
    with card("w-full"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("smart_toy").classes("ag-cyan")
            with ui.column().classes("gap-0"):
                ui.label("Autopilot mode").classes("ag-section-title")
                ui.label("Manual: you start everything. Assisted: the Autopilot proposes research tasks and AI batches "
                         "and waits for your approval. Full Autopilot: proposed tasks start by themselves within the "
                         "daily cap.").classes("ag-muted text-xs")
        with ui.row().classes("items-center gap-4 mt-2"):
            mode = ui.toggle(MODE_LABEL, value=r.mode if r.mode in MODE_LABEL else "assisted").props(
                "no-caps unelevated toggle-color=primary").classes("ag-seg")
            auto = ui.checkbox("Run research batches automatically (plan the day after each quota reset)",
                               value=r.auto_daily)
        with ui.row().classes("items-end gap-3 mt-2 flex-wrap"):
            region = ui.select(opts.regions, value=r.region if r.region in opts.regions else opts.regions[0],
                               label="Region").classes("w-28")
            delay = ui.select(opts.delays(region.value), value=r.delay if r.delay in opts.delays(region.value)
                              else opts.delays(region.value)[0], label="Delay").classes("w-24")
            universe = ui.select(opts.universes(region.value), label="Universe",
                                 value=r.universe if r.universe in opts.universes(region.value)
                                 else opts.universes(region.value)[0]).classes("w-36")
            focus = ui.select({"": "(automatic)", **{t: t for t in THEMES}}, value=r.research_focus,
                              label="Research focus").classes("w-44")
            per_day = ui.number("Tasks per day", value=r.tasks_per_day, min=1, max=10).classes("w-32")
            sims = ui.number("Simulations per task", value=r.task_sims, min=10, step=50).classes("w-44")
            cores = ui.number("Cores per task", value=r.task_cores, min=1, max=8).classes("w-32")
            cap = ui.number("Daily Autopilot cap", value=r.autopilot_daily_cap, min=0, step=50).classes("w-40")
            robust = ui.checkbox("Robustness gate", value=r.robustness_gate).tooltip(
                "Neighbouring settings must keep ≥ 80% of the Sharpe before an alpha counts as ready.")
        with ui.row().classes("gap-2 mt-2"):
            plan_btn = ui.button("Plan next task", icon="auto_awesome", on_click=lambda: plan_next())
            ui.label("Manual tasks: Tasks → New task…, or Dataset Intelligence → Research task….").classes(
                "ag-muted text-xs self-center")

    def save_settings() -> None:
        r.region, r.delay, r.universe = region.value, int(delay.value), universe.value
        r.research_focus = focus.value or ""
        r.tasks_per_day, r.task_sims = int(per_day.value or 1), int(sims.value or 500)
        r.task_cores, r.autopilot_daily_cap = int(cores.value or 4), int(cap.value or 0)
        r.robustness_gate = bool(robust.value)
        settings.save()

    def on_region(_=None) -> None:  # noqa: ANN001
        delay.options = opts.delays(region.value)
        delay.value = delay.options[0]
        universe.options = opts.universes(region.value)
        universe.value = universe.options[0]
        delay.update()
        universe.update()
        save_settings()

    region.on_value_change(on_region)
    for el in (delay, universe, focus, per_day, sims, cores, cap, robust):
        el.on_value_change(lambda _: save_settings())

    async def on_mode(e) -> None:  # noqa: ANN001
        if e.value == "autopilot" and not await confirm_full_autopilot():
            mode.value = r.mode if r.mode != "autopilot" else "assisted"
            return
        r.mode = e.value
        settings.save()
        await refresh(force=True)

    mode.on_value_change(on_mode)

    async def on_auto(e) -> None:  # noqa: ANN001
        r.auto_daily = bool(e.value)
        settings.save()
        await refresh(force=True)

    auto.on_value_change(on_auto)

    # ------------------------------------------------------------ proposals
    proposals = card("w-full")

    # ------------------------------------------------------------ strategy + current task
    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        strat = card("w-1/2")
        current = card("grow")

    # ------------------------------------------------------------ actions
    async def plan_next() -> None:
        if r.mode == "manual":
            ui.notify("Manual mode: create tasks yourself (Tasks → New task…), or switch to Assisted.", type="info")
            return
        plan_btn.props("loading")
        try:
            start = r.mode == "autopilot"
            tid = await research.plan_and_create(source="autopilot" if start else "assisted", start=start)
            t = await research.get(tid)
            ui.notify(f"Task #{tid}: {KINDS[t.kind]} on {', '.join(t.dataset_ids or [])} — "
                      + ("started." if start else "waiting for your approval."), type="positive")
        except Exception as exc:
            notify_error(exc, "Planning failed: ")
        finally:
            plan_btn.props(remove="loading")
        await refresh(force=True)

    async def resume_autopilot() -> None:
        if r.mode == "manual":
            ui.notify("Manual mode: switch to Assisted or Full Autopilot first.", type="warning")
            return
        if not brain.client.logged_in:
            ui.notify("Sign in to BRAIN first.", type="warning")
            return
        r.auto_daily = True
        auto.value = True
        settings.save()
        from ...brain.quota import eastern_day
        if await db.kv_get("autopilot:planned_day") != eastern_day():
            ids = await research.plan_day()
            ui.notify(f"Planned {len(ids)} task(s) for today.", type="positive")
        for t in await research.list():
            if t.source == "autopilot" and t.status == "paused":
                await research.resume(t.id)
        await refresh(force=True)

    resume_btn.on_click(resume_autopilot)

    async def stop_autopilot() -> None:
        r.auto_daily = False
        auto.value = False
        settings.save()
        await autopilot.stop("stopped by user")
        n = 0
        for t in await research.list():
            if t.source != "manual" and t.status in ("running", "queued"):
                await research.pause(t.id)
                n += 1
        ui.notify(f"Autopilot stopped; {n} task(s) paused.", type="warning")
        await refresh(force=True)

    state = {"v": -1}

    async def refresh(force: bool = False) -> None:
        on = autopilot_active() and r.mode != "manual"
        banner.clear()
        banner.classes(replace="w-full items-start gap-3 p-4 rounded-xl " + ("ag-bad-box" if not on else "ag-warn-box"))
        with banner:
            ui.icon("error_outline" if not on else "play_circle").classes("text-xl " + ("ag-red" if not on else "ag-green"))
            with ui.column().classes("gap-0"):
                if not on:
                    ui.label("Autopilot is stopped").classes("font-semibold")
                    ui.label("Nothing starts or runs by itself until you press Resume Autopilot. Your own tasks and "
                             "AI batches still run.").classes("text-sm")
                else:
                    ui.label(f"Autopilot is on — {MODE_LABEL[r.mode]}").classes("font-semibold ag-green")
                    ui.label("It plans research after each quota reset" + (
                        " and starts it within the daily cap." if r.mode == "autopilot"
                        else "; each proposed task waits for your approval.")).classes("text-sm ag-muted")
        t_mode.set_text(MODE_LABEL.get(r.mode, r.mode))
        t_mode_sub.set_text(MODE_SUB.get(r.mode, ""))
        used = await autopilot_sims_today()
        t_cap.set_text(f"{used} / {r.autopilot_daily_cap}")
        t_cap_sub.set_text("simulations the Autopilot started today")
        tasks_ = await research.list(100)
        auto_tasks = [t for t in tasks_ if t.source != "manual"]
        active = [t for t in tasks_ if t.status in ("running", "queued", "paused")]
        t_active.set_text(str(len(active)))
        last = next((t for t in tasks_ if t.status in ("complete", "stopped", "failed")), None)
        t_last.set_text(last.status.title() if last else "—")
        t_last_sub.set_text((last.message or "")[:140] if last else "No finished task yet")
        if not force and state["v"] == research.version:
            return
        state["v"] = research.version
        # proposals
        proposals.clear()
        planned = [t for t in auto_tasks if t.status == "planned"]
        proposals.set_visibility(bool(planned))
        with proposals:
            ui.label(f"Proposed tasks waiting for your approval ({len(planned)})").classes("ag-section-title")
            for t in planned:
                with ui.row().classes("ag-tile w-full items-center gap-3 no-wrap"):
                    with ui.column().classes("gap-0 grow"):
                        ui.label(f"#{t.id} · {KINDS[t.kind]} · {', '.join(t.dataset_ids or [])} · {t.region} D{t.delay} "
                                 f"{t.universe} · {t.planned_sims:,} simulations · {t.cores} cores").classes(
                            "font-semibold text-sm")
                        ui.label(t.reason or "").classes("text-sm ag-muted")

                    async def approve(tid=t.id) -> None:
                        try:
                            await research.approve(tid)
                            ui.notify(f"Task #{tid} started.", type="positive")
                        except Exception as exc:
                            notify_error(exc)
                        await refresh(force=True)

                    async def skip(tid=t.id) -> None:
                        await research.stop(tid, "skipped by user")
                        await refresh(force=True)

                    ui.button("Approve", icon="check", on_click=approve).props("color=positive dense")
                    ui.button("Skip", icon="close", on_click=skip).props("flat dense")
        # strategy
        strat.clear()
        strong = await research.strong_by_theme(r.region, r.delay, r.universe)
        latest = next((t for t in auto_tasks if t.status != "skipped"), None)
        with strat:
            with ui.row().classes("items-center gap-2"):
                ui.icon("psychology").classes("ag-teal")
                with ui.column().classes("gap-0"):
                    ui.label("Research strategy").classes("ag-section-title")
                    ui.label("What the agent is researching and why.").classes("ag-muted text-xs")
            with ui.grid(columns=2).classes("gap-x-6 gap-y-1 mt-2"):
                ui.label("Region · Delay").classes("ag-muted text-sm")
                ui.label(f"{r.region} · D{r.delay} · {r.universe}").classes("ag-mono")
                ui.label("Research focus").classes("ag-muted text-sm")
                ui.label(r.research_focus or (latest.theme if latest and latest.theme else "automatic")).classes(
                    "ag-mono")
            with ui.column().classes("ag-tile w-full gap-0 mt-2"):
                ui.label("Reason").classes("text-sm")
                ui.label("• " + ((latest.reason if latest else None) or (
                    f"Focus set to {r.research_focus} in the settings." if r.research_focus
                    else "No plan yet — press Plan next task."))).classes("text-sm ag-muted")
            with ui.row().classes("items-center gap-2 flex-wrap mt-2"):
                ui.label("Strong alphas by category:").classes("ag-muted text-sm")
                if not strong:
                    ui.label("none yet").classes("ag-muted text-sm")
                for th, n in strong.items():
                    ui.label(f"{th} {n}").classes("ag-src text-xs")
        # current task
        current.clear()
        cur = next((t for t in tasks_ if t.status in ("running", "paused", "queued")), None) or next(
            (t for t in tasks_ if t.status not in ("planned", "skipped")), None)
        with current:
            if cur is None:
                ui.label("No research task yet").classes("ag-section-title")
                ui.label("Plan one here, or create one on Tasks.").classes("ag-muted text-sm")
            else:
                await task_card(cur)

    async def task_card(t: ResearchTask) -> None:
        with ui.row().classes("items-center gap-2"):
            ui.icon("science").classes("ag-teal")
            with ui.column().classes("gap-0"):
                ui.label(f"Task #{t.id} · {t.status.title()}").classes("ag-section-title")
                ui.label(f"Researching {', '.join(t.dataset_ids or [])} in EQUITY/{t.region}/D{t.delay}/{t.universe}"
                         + (f", prioritising {t.theme}." if t.theme else ".")).classes("ag-muted text-xs")
        stage_bar(STAGES, t.stage, done=t.status == "complete")
        if t.message:
            ui.label(t.message).classes("text-sm")
        with ui.row().classes("gap-6 mt-1"):
            for label, value, color in (("Generated", t.n_generated, ""), ("Simulated", t.n_simulated, ""),
                                        ("Optimized", t.n_optimized, ""), ("Strong", t.n_strong, "ag-green"),
                                        ("Ready", t.n_ready, "ag-green"), ("Rejected", t.n_rejected, "ag-red")):
                with ui.column().classes("gap-0"):
                    ui.label(label).classes("ag-muted text-xs")
                    ui.label(f"{value:,}").classes(f"ag-kpi-sm {color}")
        if t.left_out:
            with ui.column().classes("ag-tile w-full gap-0 mt-1"):
                ui.label("Left out on purpose").classes("text-sm")
                for reason, n in sorted(t.left_out.items(), key=lambda kv: -kv[1]):
                    ui.label(f"• {n} candidate(s): {reason}.").classes("text-sm ag-muted")
        with ui.column().classes("ag-tile w-full gap-0 mt-1"):
            ui.label("Next batch").classes("text-sm")
            ui.label("• " + (f"Focus set to {r.research_focus} in the settings." if r.research_focus
                             else "Focus chosen automatically: the theme with the fewest strong alphas.")).classes(
                "text-sm ag-muted")

    await refresh(force=True)
    ui.timer(3.0, refresh)

    # ------------------------------------------------------------ AI idea batches (v1)
    with ui.expansion("AI idea batches (Gemini, templates and mutations, ~12 ideas per batch)", icon="lightbulb",
                      value=False).classes("w-full"):
        with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
            with card("w-[60%]"):
                with ui.row().classes("gap-3 items-end w-full"):
                    goal = ui.select({k: k.replace("_", " ").title() for k in GOALS},
                                     value=r.goal if r.goal in GOALS else "general", label="Goal").classes("w-40")
                    neut = ui.select(["(per idea)"] + opts.neutralizations(region.value), value="(per idea)",
                                     label="Neutralization").classes("w-44")
                    strategy = ui.select({"opportunity": "Best opportunity (bandit)", "pyramid_gap": "Pyramid gaps first",
                                          "under_used": "Under-used datasets", "pinned": "Pinned datasets"},
                                         value=r.dataset_strategy, label="Dataset strategy").classes("w-60")
                    pinned = ui.select([], multiple=True, value=list(r.pinned_datasets), label="Pinned datasets",
                                       with_input=True).classes("grow").props("use-chips")
                with ui.grid(columns=3).classes("w-full gap-x-6 gap-y-1"):
                    with ui.column().classes("gap-0"):
                        batch_lbl = ui.label().classes("text-sm")
                        batch = ui.slider(min=4, max=30, value=r.batch_size)
                    with ui.column().classes("gap-0"):
                        rounds_lbl = ui.label().classes("text-sm")
                        rounds = ui.slider(min=1, max=5, value=r.max_rounds)
                    with ui.column().classes("gap-0"):
                        share_lbl = ui.label().classes("text-sm")
                        share = ui.slider(min=0.05, max=1.0, step=0.05, value=r.quota_share)
                    with ui.column().classes("gap-0"):
                        llm_lbl = ui.label().classes("text-sm")
                        w_llm = ui.slider(min=0, max=1, step=0.1, value=r.generator_mix.get("llm", 0.5))
                    with ui.column().classes("gap-0"):
                        tpl_lbl = ui.label().classes("text-sm")
                        w_tpl = ui.slider(min=0, max=1, step=0.1, value=r.generator_mix.get("template", 0.3))
                    with ui.column().classes("gap-0"):
                        mut_lbl = ui.label().classes("text-sm")
                        w_mut = ui.slider(min=0, max=1, step=0.1, value=r.generator_mix.get("mutation", 0.2))
                with ui.row().classes("gap-4 items-center"):
                    empty = ui.number("Stop after N empty batches", value=r.stop_after_empty_batches, min=1).classes(
                        "w-52")
                    max_b = ui.number("Max batches (0 = ∞)", value=0, min=0).classes("w-44")

                def labels() -> None:
                    batch_lbl.set_text(f"Batch size: {int(batch.value)} ideas")
                    rounds_lbl.set_text(f"Max rounds (incl. optimisation): {int(rounds.value)}")
                    share_lbl.set_text(f"Use ≤ {share.value:.0%} of remaining quota")
                    llm_lbl.set_text(f"Gemini ideas weight: {w_llm.value:.1f}")
                    tpl_lbl.set_text(f"Template factory weight: {w_tpl.value:.1f}")
                    mut_lbl.set_text(f"Mutation / recombination weight: {w_mut.value:.1f}")

                for s in (batch, rounds, share, w_llm, w_tpl, w_mut):
                    s.on_value_change(lambda _: labels())
                labels()

                async def load_pinned_options() -> None:
                    ds = await scored_datasets(region.value, int(delay.value), universe.value)
                    pinned.options = {d.id: f"{d.id} · {d.name[:40]} · opp {d.opportunity}" for d in ds}
                    pinned.update()

                for el in (region, delay, universe):
                    el.on_value_change(lambda _: load_pinned_options())
                await load_pinned_options()

            with card("grow"):
                ui.label("Control").classes("ag-section-title")
                state_lbl = ui.label().classes("text-lg")
                line = ui.label().classes("ag-muted text-sm")
                with ui.row().classes("gap-2"):
                    start_btn = ui.button("Start", icon="play_arrow").props("color=positive")
                    stop_btn = ui.button("Stop", icon="stop").props("color=negative")
                ui.label("Manual and Assisted: each batch appears below for review — untick ideas, edit expressions "
                         "inline (they are re-checked), then approve. Full Autopilot: batches run without review, "
                         "within the daily cap.").classes("ag-muted text-sm")

        async def do_start() -> None:
            if r.mode == "autopilot" and not await confirm_full_autopilot():
                return
            cfg = AutopilotConfig(
                mode="autopilot" if r.mode == "autopilot" else "assisted", region=region.value,
                universe=universe.value, delay=int(delay.value), batch_size=int(batch.value),
                max_rounds=int(rounds.value), quota_share=float(share.value),
                stop_after_empty_batches=int(empty.value or 6),
                generator_mix={"llm": float(w_llm.value), "template": float(w_tpl.value), "mutation": float(w_mut.value)},
                dataset_strategy=strategy.value, pinned_datasets=list(pinned.value or []), goal=goal.value,
                robustness_gate=bool(robust.value), max_batches=int(max_b.value or 0),
                neutralization=None if neut.value == "(per idea)" else neut.value)
            r.goal, r.batch_size, r.max_rounds, r.quota_share = cfg.goal, cfg.batch_size, cfg.max_rounds, cfg.quota_share
            r.generator_mix, r.dataset_strategy, r.pinned_datasets = cfg.generator_mix, cfg.dataset_strategy, \
                cfg.pinned_datasets
            r.stop_after_empty_batches = cfg.stop_after_empty_batches
            settings.save()
            try:
                await autopilot.start(cfg)
                ui.notify("AI batches started.", type="positive")
            except Exception as exc:
                ui.notify(str(exc), type="negative")

        start_btn.on_click(do_start)
        stop_btn.on_click(lambda: autopilot.stop("stopped by user"))

        approval_card = card("w-full")
        with approval_card:
            with ui.row().classes("items-center w-full"):
                approval_title = ui.label().classes("ag-section-title")
                ui.space()
                approve_btn = ui.button("Approve selected & simulate", icon="play_arrow").props("color=positive")
                reject_btn = ui.button("Reject batch", icon="close").props("flat color=negative")
            approval_grid = ui.aggrid(grid_options(APPROVAL_COLUMNS, [], multi=True), theme="quartz").classes(
                "w-full h-[380px]")
        approval_card.set_visibility(False)
        approval_state = {"batch": None}

        async def load_approval() -> None:
            bid = autopilot.batch_id
            if autopilot.state != "awaiting_approval" or bid is None:
                approval_card.set_visibility(False)
                approval_state["batch"] = None
                return
            if approval_state["batch"] == bid:
                return
            approval_state["batch"] = bid
            async with db.session() as s:
                cands = list(await s.scalars(select(Candidate).where(Candidate.batch_id == bid,
                                                                     Candidate.status == "new").order_by(Candidate.id)))
            approval_grid.options["rowData"] = [{
                "cid": c.id, "origin": c.origin, "expression": c.expression, "hypothesis": c.hypothesis or "",
                "neutralization": (c.settings or {}).get("neutralization"), "decay": (c.settings or {}).get("decay"),
                "ops": c.op_count} for c in cands]
            approval_grid.update()
            approval_title.set_text(f"Batch #{bid} awaiting approval — {len(cands)} ideas (≈ {len(cands)} simulations)")
            approval_card.set_visibility(True)
            await ui.run_javascript("await new Promise(r => setTimeout(r, 300));")
            approval_grid.run_grid_method("selectAll")

        async def do_approve() -> None:
            rows = await approval_grid.get_selected_rows()
            all_rows = {x["cid"]: x["expression"] for x in await approval_grid.get_client_data()}
            selected = [x["cid"] for x in rows]
            mapping, problems = await autopilot.replace_edited({cid: all_rows[cid] for cid in selected if cid in all_rows})
            final = [mapping.get(cid, cid) for cid in selected if not any(p.startswith(f"#{cid}:") for p in problems)]
            autopilot.approve(final)
            approval_card.set_visibility(False)
            ui.notify(f"Approved {len(final)} ideas" + (f" ({len(mapping)} edited, re-checked)" if mapping else "")
                      + (f"; {len(problems)} edited expression(s) failed the check and were left out" if problems
                         else "") + ".", type="positive" if not problems else "warning", multi_line=True)

        approve_btn.on_click(do_approve)
        reject_btn.on_click(lambda: (autopilot.approve([]), approval_card.set_visibility(False)))

        with card("w-full"):
            with ui.row().classes("items-center w-full"):
                ui.label("Batch history").classes("ag-section-title")
                ui.label("Double-click a batch to see its alphas").classes("ag-muted text-sm")
            hist = ui.aggrid(grid_options(HISTORY_COLUMNS, []), theme="quartz").classes("w-full h-[320px]")
            detail_title = ui.label().classes("ag-section-title mt-2")
            detail = ui.aggrid(grid_options(RESULT_COLUMNS, []), theme="quartz").classes("w-full h-[360px]")

        async def show_batch(e) -> None:  # noqa: ANN001
            bid = (e.args.get("data") or {}).get("id")
            if bid:
                detail.options["rowData"] = await batch_rows(int(bid))
                detail.update()
                detail_title.set_text(f"Batch #{bid}")

        hist.on("cellDoubleClicked", show_batch)

        async def open_from_detail(e) -> None:  # noqa: ANN001
            from ..alpha_detail import open_alpha
            aid = (e.args.get("data") or {}).get("alpha_id")
            if aid:
                await open_alpha(aid)

        detail.on("cellDoubleClicked", open_from_detail)
        hist_state = {"sig": None}

        async def refresh_batches() -> None:
            state_lbl.set_text({"running": "● Running", "awaiting_approval": "● Waiting for your approval",
                                "stopping": "● Stopping…", "stopped": "● Stopped"}.get(autopilot.state, autopilot.state))
            state_lbl.classes(replace="text-lg " + {"running": "ag-green", "awaiting_approval": "ag-amber",
                                                     "stopping": "ag-amber"}.get(autopilot.state, "ag-muted"))
            line.set_text(autopilot.status_line)
            start_btn.set_enabled(not autopilot.running)
            stop_btn.set_enabled(autopilot.running)
            await load_approval()
            async with db.session() as s:
                rows = list(await s.scalars(select(Batch).where(Batch.source == "autopilot").order_by(Batch.id.desc())
                                            .limit(200)))
            data = [{"id": b.id, "created": f"{b.created_at:%m-%d %H:%M}", "dataset": b.dataset_id or "pv",
                     "strategy": b.strategy, "status": b.status, "generated": b.n_generated, "simulated": b.n_simulated,
                     "reused": b.n_reused, "promising": b.n_promising, "strong": b.n_strong, "ready": b.n_ready,
                     "rejected": b.n_rejected} for b in rows]
            sig = str(data[:5]) + str(len(data))
            if sig != hist_state["sig"]:
                hist_state["sig"] = sig
                hist.options["rowData"] = data
                hist.update()

        await refresh_batches()
        ui.timer(2.0, refresh_batches)
        if not brain.client.logged_in:
            badge("Sign in to BRAIN first (Sync with BRAIN).", "warning")
