"""The LLM Power Pool Lab and Super Alpha Lab tabs of the Research Labs page.

Each tab holds the whole lab: the form, a free preview (the Power Pool tab shows the exact first prompt with a token
estimate; the Super tab the grid it would run), "Add to Tasks" / "Run now", and the lab's tasks with their progress
and actions. The /labs page calls `power_pool_tab()` and `super_tab()`.
"""

from __future__ import annotations

from typing import Any, Callable

from nicegui import ui

from ...brain import brain
from ...brain.platform import FULL_MODE, REGION_AGNOSTIC, SIMULATION_SLOTS, SUPER_ALPHA_MIN_SELECTION, SUPER_ALPHA_SLOTS
from ...db import db
from ...labs import power_pool_lab as pp
from ...labs import super_lab as sl
from ..components import badge, card, fmt_int, fmt_num, grid_options, notify_error, num_col

PP_DRAFT = "pref:power_pool_lab"
STATUS_KIND = {"RUNNING": "running", "QUEUED": "pending", "PAUSED": "pending", "IDLE": "", "COMPLETE": "pass",
               "FAILED": "fail"}
RESULT_COLUMNS = [
    {"field": "trial", "headerName": "#", "width": 70},
    {"field": "alpha_id", "headerName": "Alpha", "width": 110},
    num_col("sharpe", "Sharpe", 2, 90), num_col("fitness", "Fitness", 2, 90),
    num_col("turnover", "Turnover", 1, 100, pct=True), num_col("value", "Score", 2, 85),
    {"field": "verdict", "headerName": "BRAIN verdict", "width": 130},
    {"field": "universe", "headerName": "Universe", "width": 110},
    {"field": "neutralization", "headerName": "Neutralization", "width": 130},
    {"field": "decay", "headerName": "Decay", "width": 80, "type": "numericColumn"},
    {"field": "expression", "headerName": "Expression", "flex": 1, "minWidth": 320, "cellClass": "ag-num"},
]
CALL_COLUMNS = [
    {"field": "at", "headerName": "When (UTC)", "width": 165},
    {"field": "dataset", "headerName": "Dataset", "width": 140},
    {"field": "model", "headerName": "Model", "width": 220},
    {"field": "fields", "headerName": "Fields shown", "width": 115, "type": "numericColumn"},
    {"field": "tokens", "headerName": "~Tokens", "width": 100, "type": "numericColumn"},
    {"field": "valid", "headerName": "Valid", "width": 80, "type": "numericColumn"},
    {"field": "rejected", "headerName": "Rejected", "width": 95, "type": "numericColumn"},
    {"field": "error", "headerName": "Error", "flex": 1, "minWidth": 240},
]


def _runner():  # noqa: ANN202 - the module attribute, so a test's runner is the one shown
    from ...labs import runner as runner_mod
    return runner_mod.runner


def _notice(text: str, kind: str = "fail") -> None:
    icon = {"fail": "error_outline", "warning": "warning_amber", "info": "info"}.get(kind, "info")
    color = {"fail": "ag-red", "warning": "ag-amber", "info": "ag-cyan"}.get(kind, "ag-cyan")
    with ui.row().classes("items-start gap-2 no-wrap"):
        ui.icon(icon).classes(f"{color} text-lg")
        ui.label(text).classes("text-sm")


def _metric(label: str, value: str, hint: str = "") -> None:
    with ui.card().classes("ag-tile grow min-w-[150px]"):
        ui.label(label).classes("ag-muted text-xs")
        ui.label(value).classes("ag-kpi-sm")
        if hint:
            ui.label(hint).classes("ag-muted text-xs")


# ---------------------------------------------------------------- the task list (both labs)

async def _results_dialog(task_id: int) -> None:
    rows = await _runner().top(task_id, limit=200)
    for r in rows:
        st = r.get("settings") or {}
        r.update(universe=st.get("universe"), neutralization=st.get("neutralization"), decay=st.get("decay"),
                 verdict=r.get("brain_verdict") or "")
    with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[1100px] max-w-[95vw]"):
        ui.label(f"Results of task {task_id}, best first").classes("text-lg font-semibold")
        ui.label("Score is the objective minus 1 per failed structural check. Click a row to open the alpha.").classes(
            "ag-muted text-xs")
        if not rows:
            ui.label("Nothing scored yet.").classes("ag-muted")
        grid = ui.aggrid(grid_options(RESULT_COLUMNS, rows), theme="quartz").classes("w-full h-[480px]")

        async def opened(e: Any) -> None:
            alpha_id = (e.args.get("data") or {}).get("alpha_id")
            if alpha_id:
                from ..alpha_detail import open_alpha
                await open_alpha(alpha_id)

        grid.on("rowClicked", opened)
        ui.button("Close", on_click=d.close).props("flat")
    d.open()


async def _calls_dialog(task_id: int) -> None:
    task = await _runner().get(task_id)
    state = (task.state if task else None) or {}
    calls = list(reversed(state.get("calls") or []))
    tally = state.get("llm") or {}
    with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[1100px] max-w-[95vw]"):
        ui.label(f"LLM calls of task {task_id} (last 100)").classes("text-lg font-semibold")
        by_model = ", ".join(f"{m}: {n}" for m, n in (tally.get("byModel") or {}).items()) or "none yet"
        by_dataset = ", ".join(f"{k}: {v.get('calls', 0)}" for k, v in (tally.get("byDataset") or {}).items())
        ui.label(f"Calls {fmt_int(tally.get('calls') or 0)} · valid {fmt_int(tally.get('valid') or 0)} · rejected "
                 f"{fmt_int(tally.get('rejected') or 0)} · empty in a row {fmt_int(tally.get('empty') or 0)} · "
                 f"by model: {by_model}").classes("text-sm")
        if by_dataset:
            ui.label(f"By dataset: {by_dataset}").classes("ag-muted text-xs")
        ui.aggrid(grid_options(CALL_COLUMNS, calls), theme="quartz").classes("w-full h-[420px]")
        ui.button("Close", on_click=d.close).props("flat")
    d.open()


async def _act(action: str, task_id: int, refresh: Callable[[], Any]) -> None:
    r = _runner()
    try:
        if action == "run":
            await r.run(task_id)
            ui.notify("Queued: it starts as soon as its cores are free.", type="positive")
        elif action == "pause":
            await r.pause(task_id)
            ui.notify("Paused. Simulations already on BRAIN finish and are scored when you resume.")
        elif action == "stop":
            result = await r.stop(task_id)
            ui.notify("Stopping: nothing more is sent; what is out is scored. Press Stop again to cancel it on "
                      "BRAIN." if result == "stopping" else "Stopped.")
        elif action == "delete":
            await r.delete(task_id)
            ui.notify("Task removed.")
    except ValueError as exc:
        notify_error(exc)
    except Exception as exc:  # noqa: BLE001 - shown to the user, never swallowed silently
        notify_error(exc, "Could not do that: ")
    await refresh()


async def _task_row(t: Any, lab: str, refresh: Callable[[], Any]) -> None:
    r = _runner()
    s = await r.summary(t.id)
    state = t.state or {}
    with ui.card().classes("ag-card p-4 w-full"):
        with ui.row().classes("items-center gap-3 w-full no-wrap"):
            badge(t.status.title(), STATUS_KIND.get(t.status, ""))
            ui.label(t.name or f"Task {t.id}").classes("font-semibold")
            ui.label(f"#{t.id} · {t.region} D{t.delay} · {t.cores} core{'s' if t.cores != 1 else ''}").classes(
                "ag-muted text-xs")
            if t.stopping:
                badge("Stopping", "warning")
            ui.space()
            if t.status in ("IDLE", "PAUSED", "FAILED"):
                ui.button("Run", icon="play_arrow", on_click=lambda: _act("run", t.id, refresh)).props("dense flat")
            if t.status in ("RUNNING", "QUEUED"):
                ui.button("Pause", icon="pause", on_click=lambda: _act("pause", t.id, refresh)).props("dense flat")
            if t.status in ("RUNNING", "QUEUED", "PAUSED"):
                ui.button("Stop", icon="stop", on_click=lambda: _act("stop", t.id, refresh)).props("dense flat")
            ui.button("Results", icon="leaderboard", on_click=lambda: _results_dialog(t.id)).props("dense flat")
            if lab == "power_pool":
                ui.button("LLM calls", icon="forum", on_click=lambda: _calls_dialog(t.id)).props("dense flat")
            if t.status not in ("RUNNING", "QUEUED"):
                ui.button(icon="delete_outline", on_click=lambda: _act("delete", t.id, refresh)).props(
                    "dense flat").tooltip("Remove this task and its trials")
        done = int(s["simulated"])
        ui.linear_progress(value=min(1.0, done / max(1, t.max_trials)), show_value=False).props("rounded size=6px")
        parts = [f"{fmt_int(done)} / {fmt_int(t.max_trials)} simulations", f"{fmt_int(s['queued'])} waiting",
                 f"{fmt_int(s['running'])} running", f"{fmt_int(s['failed'])} failed"]
        if s["cached"]:
            parts.append(f"{fmt_int(s['cached'])} answered free")
        if lab == "power_pool":
            tally = state.get("llm") or {}
            parts += [f"{fmt_int(tally.get('calls') or 0)} / {fmt_int(pp.call_cap(t.max_trials))} LLM calls",
                      f"{fmt_int(tally.get('valid') or 0)} valid drafts",
                      f"{fmt_int(tally.get('rejected') or 0)} rejected"]
        if t.best_value is not None:
            parts.append(f"best score {fmt_num(t.best_value)}" + (f" ({t.best_alpha_id})" if t.best_alpha_id else ""))
        ui.label(" · ".join(parts)).classes("text-sm ag-mono")
        if lab == "super":
            space = t.space or {}
            ui.label(f"{len(space.get('recipes') or [])} pairings × decays {', '.join(map(str, sl.DECAYS))} × "
                     f"{' and '.join(space.get('activation') or [])} · selection limit "
                     f"{space.get('selection_limit')} · {space.get('neutralization')}").classes("ag-muted text-xs")
        if t.message:
            ui.label(t.message).classes("text-sm ag-amber" if t.status in ("PAUSED", "FAILED") else "text-sm ag-muted")
        for reason, n in s["reasons"][:3]:
            ui.label(f"{n} × {reason}").classes("ag-muted text-xs")


def _task_list(lab: str) -> Callable[[], Any]:
    @ui.refreshable
    async def tasks() -> None:
        rows = await _runner().list(lab=lab, limit=30)
        if not rows:
            ui.label("No tasks yet. Add one above.").classes("ag-muted text-sm")
            return
        for t in rows:
            await _task_row(t, lab, tasks.refresh)

    return tasks


async def _tasks_card(lab: str) -> Any:
    with card("w-full"):
        with ui.row().classes("items-center w-full"):
            ui.label("Tasks").classes("ag-section-title")
            ui.space()
            ui.label("Refreshes every 5 seconds; every lab's tasks are also on the Tasks page.").classes(
                "ag-muted text-xs")
        tasks = _task_list(lab)
        await tasks()
        ui.timer(5.0, tasks.refresh)
    return tasks


# ---------------------------------------------------------------- LLM Power Pool Lab

def _default_region() -> str:
    from ...config import settings
    regions = brain.options.regions
    return settings.research.region if settings.research.region in regions else (regions[0] if regions else "USA")


async def power_pool_tab() -> None:
    """The LLM Power Pool Lab: datasets, a model, cores and simulations, then add the task to Tasks."""
    from ...config import settings
    opts = brain.options
    draft = dict(await db.kv_get(PP_DRAFT) or {})
    region0 = draft.get("region") if draft.get("region") in opts.regions else _default_region()
    delays0 = opts.delays(region0)
    delay0 = draft.get("delay") if draft.get("delay") in delays0 else (settings.research.delay if settings.research
                                                                         .delay in delays0 else delays0[0])
    models = pp.model_options()
    model_items = {m["ref"]: m["label"] for m in models}
    lists: dict[str, Any] = {"unused": None}

    with card("w-full"):
        ui.label("LLM Power Pool Lab").classes("ag-section-title")
        ui.label("While the task runs, the model writes 20 Power Pool Alphas per call for one of your datasets. Each "
                 "is checked here first (at most 8 operators and 3 data fields, counted as BRAIN counts them, and at "
                 "least one field of the dataset) and given a random universe, neutralization and decay; truncation "
                 f"{pp.TRUNCATION}. Rejected drafts are kept with the reason and shown to the next call.").classes(
            "ag-muted text-sm")
        if not models:
            _notice("This lab needs a model. Set up a provider and choose a model on the LLM Integration page.",
                    "warning")
        with ui.row().classes("gap-3 w-full items-end"):
            region = ui.select(opts.regions, value=region0, label="Region").classes("w-28")
            delay = ui.select(delays0, value=delay0, label="Delay").classes("w-24")
            universe = ui.select([], label="Universe").classes("w-36")
            datasets = ui.select({}, label=f"Datasets (up to {pp.MAX_DATASETS})", multiple=True,
                                 with_input=True).props("use-chips").classes("grow min-w-[320px]")
        with ui.row().classes("gap-3 w-full items-end"):
            search = ui.input("Field filter: text in id or description",
                              value=(draft.get("field_filter") or {}).get("search", "")).classes("w-72")
            min_cov = ui.number("Min coverage %", min=0, max=100, step=5,
                                value=((draft.get("field_filter") or {}).get("min_coverage") or 0) * 100).classes(
                "w-36")
            types = ui.select(["MATRIX", "VECTOR"], multiple=True, label="Field types (none = all)",
                              value=(draft.get("field_filter") or {}).get("types") or []).classes("w-56")
        with ui.row().classes("gap-3 w-full items-end"):
            default_model = draft.get("model") if draft.get("model") in model_items else next(iter(model_items), None)
            model = ui.select(model_items, value=default_model, label="Model").classes("w-96")
            only = ui.checkbox("Only use this model", value=bool(draft.get("only_model")))
            cores = ui.number(f"Cores (of {SIMULATION_SLOTS})", value=draft.get("cores") or settings.research.task_cores,
                              min=1, max=SIMULATION_SLOTS, step=1).classes("w-32")
            sims = ui.number("Simulations", value=draft.get("simulations") or 500, min=1, max=100_000,
                             step=20).classes("w-32")
        only_hint = ui.label().classes("ag-muted text-xs")

        def hint_only() -> None:
            only_hint.text = ("When it cannot answer (daily budget, provider down), the task waits for it." if
                              only.value else "When it cannot answer, the task carries on with the next provider set "
                              "up on the LLM Integration page (its fallbacks, in order). Every task shares the daily "
                              "LLM call budget.")

        only.on_value_change(lambda _: hint_only())
        hint_only()
        hypothesis = ui.textarea("Hypothesis · optional", value=draft.get("hypothesis") or "",
                                 placeholder="e.g. Stocks whose analysts keep revising estimates up outperform for the "
                                             "next month.").props(f"maxlength={pp.MAX_HYPOTHESIS} counter autogrow"
                                                                  ).classes("w-full")
        ui.label("Every LLM call writes its Alphas to test this idea on the chosen datasets. Left empty, the LLM "
                 "explores on its own.").classes("ag-muted text-xs")
        with ui.row().classes("items-center gap-3"):
            favour = ui.checkbox("Favour operators unused this quarter", value=bool(draft.get("favour")))
            favour_info = ui.label().classes("ag-muted text-xs")
        with ui.row().classes("gap-3 w-full items-end"):
            universes = ui.select([], multiple=True, label="Universes (none ticked = every downloaded one)").props(
                "use-chips").classes("w-[420px]")
            neutralizations = ui.select([], multiple=True, label="Neutralizations").props("use-chips").classes(
                "grow min-w-[320px]")
        with ui.row().classes("gap-3 w-full items-end"):
            name = ui.input("Task name (optional)").classes("w-72")
            ui.space()
            preview_btn = ui.button("Preview", icon="visibility").props("flat")
            add_btn = ui.button("Add to Tasks", icon="playlist_add").props("outline")
            run_btn = ui.button("Run now", icon="play_arrow")
        preview_box = ui.column().classes("w-full gap-2")

    async def load_market() -> None:
        uni = await pp.synced_universes(region.value, delay.value)
        universe.options = uni or opts.universes(region.value)
        universe.value = draft.get("universe") if draft.get("universe") in universe.options else (
            universe.options[0] if universe.options else None)
        universe.update()
        universes.options = uni
        universes.value = [u for u in (draft.get("universes") or []) if u in uni]
        universes.update()
        neut = [n for n in opts.neutralizations(region.value) if n != "NONE"]
        neutralizations.options = neut
        kept = [n for n in (draft.get("neutralizations") or []) if n in neut]
        neutralizations.value = kept or neut
        neutralizations.update()
        rows = await pp.datasets_of(region.value, delay.value)
        datasets.options = {d["id"]: f"{d['id']} · {d['name']}" for d in rows}
        datasets.value = [d for d in (draft.get("dataset_ids") or []) if d in datasets.options][:pp.MAX_DATASETS]
        datasets.update()

    async def region_changed() -> None:
        delay.options = opts.delays(region.value)
        if delay.value not in delay.options:
            delay.value = delay.options[0]
        delay.update()
        await load_market()

    region.on_value_change(lambda _: region_changed())
    delay.on_value_change(lambda _: load_market())

    async def favour_changed() -> None:
        if not favour.value:
            favour_info.text = ("Total distinct operators over the quarter is one of BRAIN Genius's tiebreakers. On, "
                                "the LLM is asked to reach for the ones your submitted Alphas have not used yet.")
            return
        lists["unused"] = await pp.unused_operators()
        u = lists["unused"]
        favour_info.text = (f"{len(u['unused'])} of {len(u['operators'])} operators are unused in {u['label']} "
                            f"(read from {u['alphas']} submitted alphas synced locally). Every call asks for them where "
                            "they fit the idea.")

    favour.on_value_change(lambda _: favour_changed())

    def params() -> dict[str, Any]:
        ff = {"search": (search.value or "").strip(), "min_coverage": (float(min_cov.value or 0) / 100) or None,
              "types": list(types.value or [])}
        ff = {k: v for k, v in ff.items() if v} or None
        favour_ops = (lists["unused"] or {}).get("unused") or [] if favour.value else []
        return {"region": region.value, "delay": int(delay.value), "universe": universe.value,
                "dataset_ids": list(datasets.value or [])[:pp.MAX_DATASETS], "model": model.value,
                "only_model": bool(only.value), "neutralizations": list(neutralizations.value or []),
                "universes": list(universes.value or []), "cores": int(cores.value or 1),
                "simulations": int(sims.value or 0), "field_filter": ff,
                "hypothesis": (hypothesis.value or "").strip()[:pp.MAX_HYPOTHESIS], "favour_operators": favour_ops}

    async def save_draft(p: dict[str, Any]) -> None:
        await db.kv_set(PP_DRAFT, {**{k: v for k, v in p.items() if k != "favour_operators"},
                                   "favour": bool(favour.value)})

    async def show_preview() -> dict[str, Any] | None:
        p = params()
        if favour.value and lists["unused"] is None:
            await favour_changed()
            p = params()
        try:
            plan = await pp.plan_task(p)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, "Could not plan the task: ")
            return None
        await save_draft(p)
        preview_box.clear()
        with preview_box:
            with ui.row().classes("gap-3 w-full"):
                _metric("Datasets", fmt_int(len(p["dataset_ids"])))
                _metric("Fields", fmt_int(plan["fields"]))
                _metric("LLM calls", fmt_int(plan["llmCalls"]), f"{pp.PER_CALL} Alphas each")
                _metric("Universes", fmt_int(len(plan["universes"])),
                        f"{len(plan['neutralizations'])} neutralizations")
            ui.label("Each Alpha gets a random universe, neutralization and decay (0, 3, 5, 7 or 10); truncation "
                     f"{pp.TRUNCATION}.").classes("ag-muted text-xs")
            for m in plan["problems"]:
                _notice(m, "fail")
            for m in plan["warnings"]:
                _notice(m, "warning")
            if plan["prompt"]:
                pr = plan["prompt"]
                with ui.expansion(f"Prompt · ~{fmt_int(pr['tokens'])} tokens · {pr['fields']} fields of the first "
                                  "dataset shown", icon="article").classes("w-full"):
                    ui.label("System prompt").classes("ag-muted text-xs")
                    ui.html(_pre(pr["system"]))
                    ui.label("User prompt (first call)").classes("ag-muted text-xs")
                    ui.html(_pre(pr["user"]))
        return plan

    async def add(run: bool) -> None:
        plan = await show_preview()
        if plan is None:
            return
        if plan["problems"]:
            notify_error(ValueError(plan["problems"][0]))
            return
        p = params()
        try:
            tid = await _runner().create("power_pool", p, cores=p["cores"], max_trials=p["simulations"],
                                         name=name.value or None, run=run)
        except ValueError as exc:
            notify_error(exc)
            return
        ui.notify(f"Task {tid} " + ("queued: it starts as soon as its cores are free." if run else
                                    "added. Run it here or on the Tasks page."), type="positive")
        await tasks.refresh()

    preview_btn.on_click(show_preview)
    add_btn.on_click(lambda: add(False))
    run_btn.on_click(lambda: add(True))
    await load_market()
    await favour_changed()
    tasks = await _tasks_card("power_pool")


def _pre(text: str) -> str:
    import html
    return ("<pre style=\"max-height:320px;overflow:auto;white-space:pre-wrap;font-family:var(--ag-mono);"
            "font-size:12px;border:1px solid #1e2b40;border-radius:6px;padding:10px;margin:0\">"
            f"{html.escape(text)}</pre>")


# ---------------------------------------------------------------- Super Alpha Lab

async def super_tab() -> None:
    """The Super Alpha Lab: SuperAlphas built from your own submitted Alphas, every pairing in one task."""
    opts = brain.options
    prefs = await sl.load_prefs()
    regions = [r for r in opts.regions if r != REGION_AGNOSTIC] or ["USA"]
    region0 = _default_region() if _default_region() in regions else regions[0]
    state: dict[str, Any] = {"plan": None}

    with card("w-full"):
        ui.label("Super Alpha Lab").classes("ag-section-title")
        ui.label("Combines your submitted ACTIVE Alphas of one region and delay into SuperAlphas, ready to submit. "
                 "Every ticked selection is crossed with every ticked combo, at two decays to land Turnover between "
                 f"2% and 40%, each run in {FULL_MODE.title()} mode so BRAIN's own checks say which are "
                 "submittable. Selection handling is Positive: an Alpha a selection scores zero is left out.").classes(
            "ag-muted text-sm")
        with ui.row().classes("gap-3 w-full items-end"):
            region = ui.select(regions, value=region0, label="Region").classes("w-28")
            delay = ui.select(opts.delays(region0), value=opts.delays(region0)[0], label="Delay").classes("w-24")
            universe = ui.select(opts.universes(region0), value=opts.universes(region0)[0],
                                 label="Universe").classes("w-36")
            neut_opts = list(opts.neutralizations(region0))
            neutralization = ui.select(neut_opts, label="Neutralization",
                                       value="SUBINDUSTRY" if "SUBINDUSTRY" in neut_opts else neut_opts[0]).classes(
                "w-44")
            limit = ui.number(f"Selection limit ({SUPER_ALPHA_MIN_SELECTION}–{sl.MAX_LIMIT})", value=sl.DEFAULT_LIMIT,
                              min=SUPER_ALPHA_MIN_SELECTION, max=sl.MAX_LIMIT, step=1).classes("w-44")
            activation = ui.toggle(sl.ACTIVATIONS, value="BOTH").props("dense")
            cores = ui.toggle({i: str(i) for i in range(1, SUPER_ALPHA_SLOTS + 1)}, value=SUPER_ALPHA_SLOTS).props(
                "dense").tooltip("BRAIN runs three SuperAlpha simulations at a time")
        with ui.row().classes("gap-3 w-full items-end"):
            name = ui.input("Task name (optional)").classes("w-72")
            ui.space()
            add_btn = ui.button("Add to Tasks", icon="playlist_add").props("outline")
            run_btn = ui.button("Run now", icon="play_arrow")
        summary = ui.column().classes("w-full gap-2")
    with card("w-full"):
        with ui.row().classes("items-center w-full gap-2"):
            ui.label("What runs").classes("ag-section-title")
            ui.space()
            ui.button("Write your own selection", icon="edit",
                      on_click=lambda: new_own("selections")).props("flat dense")
            ui.button("Write your own combo", icon="edit", on_click=lambda: new_own("combos")).props("flat dense")
        ui.label("Tick what to cross. Copy a built-in one to edit it; yours are kept on this computer.").classes(
            "ag-muted text-xs")
        recipes_box = ui.column().classes("w-full gap-3")
        _notice("Each built-in selection caps an Alpha's operators at 8,000 divided by the Selection Limit, so the "
                "selected Alphas never pass BRAIN's 8,000-operator ceiling. Submitting stays on BRAIN, with a "
                "description of the selection and the combo.", "info")

    def form() -> dict[str, Any]:
        return {"region": region.value, "delay": int(delay.value or 0), "universe": universe.value,
                "neutralization": neutralization.value, "selection_limit": limit.value,
                "activation": activation.value, "cores": int(cores.value or 1)}

    def chosen_lists() -> tuple[list[sl.Recipe], list[sl.Recipe]]:
        plan = state["plan"] or {"selections": [], "combos": []}
        return (sl.chosen(plan["selections"], prefs["selections"], prefs["off"]),
                sl.chosen(plan["combos"], prefs["combos"], prefs["off"]))

    def task_params() -> dict[str, Any]:
        sel, com = chosen_lists()
        return {**form(), "selections": [{"name": r.name, "code": r.code} for r in sel],
                "combos": [{"name": r.name, "code": r.code} for r in com]}

    async def persist() -> None:
        await sl.save_prefs(prefs)

    async def toggle(key: str, on: bool) -> None:
        prefs["off"] = [k for k in prefs["off"] if k != key] + ([] if on else [key])
        await persist()
        await render_summary()

    async def new_own(kind: str, name_: str | None = None, code: str | None = None) -> None:
        sl.add_own(prefs, kind, name_, code)
        await persist()
        await refresh_all()

    async def edit_own(kind: str, rid: str, field: str, value: str) -> None:
        for r in prefs[kind]:
            if r["id"] == rid:
                r[field] = value
        await persist()
        await render_summary()

    async def remove_own(kind: str, rid: str) -> None:
        prefs[kind] = [r for r in prefs[kind] if r["id"] != rid]
        prefs["off"] = [k for k in prefs["off"] if k != f"custom:{rid}"]
        await persist()
        await refresh_all()

    def recipe_list(kind: str, builtins: list[sl.Recipe]) -> None:
        ui.label("Selections" if kind == "selections" else "Combos").classes("font-semibold")
        for r in builtins:
            key = f"builtin:{r.name}"
            with ui.row().classes("items-start gap-3 w-full no-wrap"):
                ui.checkbox(value=key not in prefs["off"],
                            on_change=lambda e, k=key: toggle(k, bool(e.value)))
                with ui.column().classes("gap-0 grow"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(r.name)
                        badge("Built-in")
                    ui.html(_pre(r.code))
                ui.button("Copy to edit", icon="content_copy",
                          on_click=lambda rr=r: new_own(kind, f"{rr.name} (edited)", rr.code)).props("flat dense")
        for own in prefs[kind]:
            key = f"custom:{own['id']}"
            with ui.row().classes("items-start gap-3 w-full no-wrap"):
                ui.checkbox(value=key not in prefs["off"], on_change=lambda e, k=key: toggle(k, bool(e.value)))
                with ui.column().classes("gap-1 grow"):
                    ui.input("Name", value=own["name"]).props("maxlength=120").classes("w-72").on(
                        "blur", lambda e, rid=own["id"]: edit_own(kind, rid, "name", e.sender.value))
                    ui.textarea("Code", value=own["code"]).props("autogrow maxlength=8000").classes(
                        "w-full ag-mono").on("blur", lambda e, rid=own["id"]: edit_own(kind, rid, "code",
                                                                                       e.sender.value))
                ui.button(icon="delete_outline", on_click=lambda rid=own["id"]: remove_own(kind, rid)).props(
                    "flat dense").tooltip("Remove")

    async def render_summary() -> None:
        sel, com = chosen_lists()
        f = form()
        found, problems = await sl.build({**f, "selections": [r.__dict__ for r in sel],
                                          "combos": [r.__dict__ for r in com]})
        plan = state["plan"] or {}
        summary.clear()
        with summary:
            with ui.row().classes("gap-3 w-full"):
                _metric("SuperAlpha simulations", fmt_int(len(found)))
                _metric("Decay", " and ".join(map(str, plan.get("decays") or sl.DECAYS)))
                _metric("Truncation", str(plan.get("truncation", sl.TRUNCATION)))
                _metric("Built-ins", f"{len(plan.get('selections') or [])} selections",
                        f"{len(plan.get('combos') or [])} combos")
            for m in problems:
                _notice(m, "fail")
            if not problems and not sl.runner_hooks():
                _notice(sl.NO_HOOK, "fail")
            reason = sl.backoff()
            if reason:
                _notice(f"SuperAlpha sends are waiting: {reason}.", "warning")

    async def refresh_all() -> None:
        state["plan"] = await sl.plan_summary(form())
        recipes_box.clear()
        with recipes_box:
            recipe_list("selections", state["plan"]["selections"])
            recipe_list("combos", state["plan"]["combos"])
        await render_summary()

    async def region_changed() -> None:
        delay.options = opts.delays(region.value)
        delay.value = delay.value if delay.value in delay.options else delay.options[0]
        delay.update()
        universe.options = opts.universes(region.value)
        universe.value = universe.value if universe.value in universe.options else universe.options[0]
        universe.update()
        neutralization.options = list(opts.neutralizations(region.value))
        if neutralization.value not in neutralization.options:
            neutralization.value = "SUBINDUSTRY" if "SUBINDUSTRY" in neutralization.options else \
                neutralization.options[0]
        neutralization.update()
        await render_summary()

    region.on_value_change(lambda _: region_changed())
    for w in (delay, universe, neutralization, activation):
        w.on_value_change(lambda _: render_summary())
    limit.on_value_change(lambda _: refresh_all())

    async def add(run: bool) -> None:
        p = task_params()
        found, problems = await sl.build(p)
        if problems:
            notify_error(ValueError(problems[0]))
            return
        try:
            tid = await _runner().create("super", p, cores=p["cores"], max_trials=len(found),
                                         name=name.value or f"Super Alpha Lab · {p['region']} D{p['delay']} "
                                                            f"{p['universe']}", run=run)
        except ValueError as exc:
            notify_error(exc)
            return
        ui.notify(f"Task {tid}: {len(found)} SuperAlpha simulations " + (
            "queued." if run else "added. Run it here or on the Tasks page."), type="positive")
        await tasks.refresh()

    add_btn.on_click(lambda: add(False))
    run_btn.on_click(lambda: add(True))
    await refresh_all()
    tasks = await _tasks_card("super")
