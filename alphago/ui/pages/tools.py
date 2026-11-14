"""Tools (ported from alpha-harness): Settings Sampler, Correlation Breaker, Fields Sampler, Paste Alphas, and the
Copy Alphas / Power Pool description helpers.

Every tool works the same way: read what it is given (an Alpha ID, or pasted text), show a free preview — counts,
sample requests, problems and warnings, cost with All Regions counted at four, batches — then "Add to Tasks" (waits
in the list below) or "Run now". Each tool's tasks are listed under it with their progress and the usual actions.
Nothing here calls BRAIN's /simulations: every simulation goes through the lab task runner.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from nicegui import ui

from ...brain.platform import MAX_DECAY, REGION_AGNOSTIC, SIMULATION_SLOTS
from ...labs import runner as runner_mod
from ...tools import correlation_breaker as cb
from ...tools import fields_sampler as fs
from ...tools import paste_alphas as paste
from ...tools import settings_sampler as ss
from ...tools.common import NOT_AN_ID, ToolError, looks_like_id, settings_line
from ..components import badge, card, fmt_int, fmt_num, fmt_pct, grid_options, info_box, notify_error, num_col, page_header

TOOLS = (
    ("settings", "Settings Sampler", "tune", "Sweep one Alpha across Simulation Settings to find where it works best."),
    ("breaker", "Correlation Breaker", "link_off", "Re-shape an Alpha that is already in the Production Pool."),
    ("fields", "Fields Sampler", "call_split",
     "Swap one Alpha's data fields for others, one at a time, and grow from every swap that passes."),
    ("paste", "Paste Alphas", "content_paste",
     "Simulate the Alphas an LLM wrote, each at its own Simulation Settings, and copy their results back."),
    ("copy", "Copy & Describe", "description",
     "Copy Alphas as Markdown for an LLM, or write a Power Pool description in BRAIN's template."),
)
TASK_COLUMNS = [
    {"field": "id", "headerName": "#", "width": 70},
    {"field": "name", "headerName": "Task", "flex": 1, "minWidth": 220},
    {"field": "status", "headerName": "Status", "width": 110},
    {"field": "progress", "headerName": "Simulations", "width": 130},
    {"field": "cached", "headerName": "Free", "width": 80, "type": "numericColumn"},
    {"field": "failed", "headerName": "Failed", "width": 85, "type": "numericColumn"},
    {"field": "cores", "headerName": "Cores", "width": 80, "type": "numericColumn"},
    num_col("best", "Best", 2, 90),
    {"field": "message", "headerName": "Note", "flex": 1, "minWidth": 200},
]
TOP_COLUMNS = [
    {"field": "alpha_id", "headerName": "Alpha", "width": 110},
    num_col("sharpe", "Sharpe", 2, 90), num_col("fitness", "Fitness", 2, 90),
    num_col("turnover", "Turnover", 1, 100, pct=True), {"field": "brain_verdict", "headerName": "BRAIN", "width": 120},
    {"field": "market", "headerName": "Settings", "width": 260},
    {"field": "expression", "headerName": "Expression", "flex": 1, "minWidth": 260},
]


# ---------------------------------------------------------------- shared pieces

def _notify(exc: BaseException) -> None:
    if isinstance(exc, ToolError):
        ui.notify(str(exc), type="warning", multi_line=True, close_button=True, timeout=8000)
    else:
        notify_error(exc)


def problems_box(container: ui.element, problems: list[str], warnings: list[str] | None = None) -> None:
    with container:
        for p in problems:
            with ui.row().classes("items-start gap-2 no-wrap"):
                ui.icon("error").classes("ag-red")
                ui.label(f"Cannot run: {p}").classes("text-sm ag-red")
        for w in warnings or []:
            with ui.row().classes("items-start gap-2 no-wrap"):
                ui.icon("warning").classes("ag-amber")
                ui.label(w).classes("text-sm ag-amber")


def alpha_id_box(on_analyse: Callable[[str], Any], label: str = "Alpha ID", mark: str = "") -> ui.input:
    """The Alpha ID box: says so at once when an expression was pasted instead (it would come back "not found")."""
    with ui.row().classes("items-end gap-3 w-full"):
        box = ui.input(label, placeholder="e.g. aB12cD3 or its BRAIN link").props("dense clearable").classes(
            "w-80 ag-mono")
        button = ui.button("Analyse", icon="search").props("unelevated")
    if mark:
        box.mark(f"alpha-{mark}")
        button.mark(f"analyse-{mark}")
    hint = ui.label(NOT_AN_ID).classes("text-sm ag-amber")
    hint.set_visibility(False)

    def check(_e: Any = None) -> None:
        text = (box.value or "").strip()
        bad = bool(text) and not looks_like_id(text)
        hint.set_visibility(bad)
        button.set_enabled(bool(text) and not bad)

    box.on_value_change(check)
    check()

    async def go() -> None:
        try:
            await on_analyse((box.value or "").strip())
        except Exception as exc:
            _notify(exc)
    button.on_click(go)
    box.on("keydown.enter", go)
    return box


def held_badges(settings: dict[str, Any]) -> None:
    with ui.row().classes("gap-2 flex-wrap"):
        for key, title in (("region", "Region"), ("delay", "Delay"), ("universe", "Universe"),
                           ("neutralization", "Neutralization"), ("decay", "Decay"), ("truncation", "Truncation")):
            value = settings.get(key)
            text = "—" if value is None else (f"D{value}" if key == "delay" else str(value))
            badge(f"{title} {text}")


def stat(label: str, value: str, hint: str = "") -> None:
    with ui.column().classes("gap-0 min-w-[120px]"):
        ui.label(label).classes("text-xs ag-muted")
        ui.label(value).classes("ag-kpi-sm")
        if hint:
            ui.label(hint).classes("text-xs ag-muted")


def add_buttons(create: Callable[[bool, int, str | None], Any], default_name: Callable[[], str]) -> None:
    """Cores, an optional name, then Add to Tasks (waits) or Run now (queues for cores at once)."""
    with ui.row().classes("items-end gap-3"):
        cores = ui.number("Cores", value=SIMULATION_SLOTS, min=1, max=SIMULATION_SLOTS, step=1).props(
            "dense").classes("w-24")
        name = ui.input("Task name (optional)").props("dense").classes("w-72")

        async def go(run: bool) -> None:
            try:
                tid = await create(run, int(cores.value or 1), (name.value or "").strip() or default_name())
            except Exception as exc:
                _notify(exc)
                return
            ui.notify(f"Task #{tid} {'queued to run' if run else 'added to Tasks'}.", type="positive")
        ui.button("Add to Tasks", icon="playlist_add", on_click=lambda: go(False)).props("outline")
        ui.button("Run now", icon="play_arrow", on_click=lambda: go(True)).props("unelevated")


async def _rename(tid: int, current: str) -> None:
    with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[420px]"):
        ui.label(f"Rename task #{tid}").classes("ag-section-title")
        box = ui.input("Name", value=current).classes("w-full")

        async def save() -> None:
            await runner_mod.runner.rename(tid, box.value)
            d.close()
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=d.close).props("flat")
            ui.button("Save", on_click=save).props("unelevated")
    d.open()


async def _resize(tid: int, cores: int, max_trials: int) -> None:
    with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[420px]"):
        ui.label(f"Resize task #{tid}").classes("ag-section-title")
        c = ui.number("Cores", value=cores, min=1, max=SIMULATION_SLOTS, step=1).classes("w-full")
        m = ui.number("Simulations to spend", value=max_trials, min=1, step=10).classes("w-full")

        async def save() -> None:
            try:
                await runner_mod.runner.resize(tid, cores=int(c.value or 1), max_trials=int(m.value or 1))
                d.close()
            except Exception as exc:
                _notify(exc)
        with ui.row().classes("justify-end w-full"):
            ui.button("Cancel", on_click=d.close).props("flat")
            ui.button("Save", on_click=save).props("unelevated")
    d.open()


async def _results(tid: int) -> None:
    from ..alpha_detail import open_alpha
    rows = await runner_mod.runner.top(tid, limit=200)
    for r in rows:
        r["market"] = settings_line(r.get("settings") or {})
    with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[1100px] max-w-[95vw]"):
        ui.label(f"Task #{tid}: {len(rows)} scored alphas, best first (double-click to open)").classes(
            "ag-section-title")
        g = ui.aggrid(grid_options(TOP_COLUMNS, rows), theme="quartz").classes("w-full h-[460px]")
        g.on("cellDoubleClicked", lambda e: open_alpha((e.args.get("data") or {}).get("alpha_id"))
             if (e.args.get("data") or {}).get("alpha_id") else None)
        ui.button("Close", on_click=d.close).props("flat")
    d.open()


def task_list(lab: str, extra: list[tuple[str, str, Callable[[int], Any]]] | None = None) -> None:
    """This tool's tasks with status and progress, and the lifecycle actions on the selected one."""
    with card("w-full"):
        with ui.row().classes("items-center w-full gap-2"):
            ui.label("Tasks of this tool").classes("ag-section-title")
            ui.space()
            buttons = ui.row().classes("gap-1")
        grid = ui.aggrid(grid_options(TASK_COLUMNS, []), theme="quartz").classes("w-full h-[260px]")

    async def selected() -> dict[str, Any] | None:
        rows = await grid.get_selected_rows()
        if not rows:
            ui.notify("Select a task first.", type="warning")
            return None
        return rows[0]

    def action(label: str, icon: str, fn: Callable[[dict[str, Any]], Any]) -> None:
        async def go() -> None:
            row = await selected()
            if row is None:
                return
            try:
                out = await fn(row)
                if isinstance(out, str):
                    ui.notify(out)
            except Exception as exc:
                _notify(exc)
            await refresh()
        with buttons:
            ui.button(label, icon=icon, on_click=go).props("flat dense no-caps")

    r = runner_mod.runner
    action("Run", "play_arrow", lambda row: r.run(row["id"]))
    action("Pause", "pause", lambda row: r.pause(row["id"]))

    async def stop(row: dict[str, Any]) -> str:
        said = await r.stop(row["id"])
        return {"stopping": "Stopping: what is out is scored; press Stop again to cancel it on BRAIN.",
                "stopped": "Stopped."}.get(said, "Nothing to stop.")
    action("Stop", "stop", stop)
    action("Rename", "edit", lambda row: _rename(row["id"], row.get("name") or ""))
    action("Resize", "tune", lambda row: _resize(row["id"], int(row.get("cores") or 1), int(row.get("max") or 1)))

    async def delete(row: dict[str, Any]) -> str:
        await r.delete(row["id"])
        return f"Task #{row['id']} removed."
    action("Delete", "delete", delete)
    action("Results", "leaderboard", lambda row: _results(row["id"]))
    for label, icon, fn in extra or []:
        action(label, icon, lambda row, fn=fn: fn(row["id"]))

    async def refresh() -> None:
        rows = []
        for t in await r.list(lab):
            s = await r.summary(t.id)
            rows.append({"id": t.id, "name": t.name or f"{r.lab(t.lab).title} #{t.id}", "status": t.status.title(),
                         "progress": f"{s['simulated']:,} / {t.max_trials:,}" + (
                             f" · {s['running']} out" if s["running"] else ""),
                         "cached": s["cached"], "failed": s["failed"], "cores": t.cores, "max": t.max_trials,
                         "best": t.best_value, "message": t.message or ""})
        grid.options["rowData"] = rows
        grid.update()

    ui.timer(5.0, refresh)
    ui.timer(0.1, refresh, once=True)


# ---------------------------------------------------------------- Settings Sampler

def _market_key(m: dict[str, Any]) -> tuple[str, int, str]:
    return str(m["region"]), int(m["delay"]), str(m["universe"])


def settings_sampler_panel() -> None:
    state: dict[str, Any] = {"found": None, "chosen": set(), "neutralizations": set(), "pairs": set(), "decays": []}
    info_box("Settings Sampler", "The expression, decay and truncation are held as the Alpha has them; what varies is "
             "the market (where every field it reads is downloaded), the neutralization and the Max Trade / Max "
             "Position pair. Coverage is the thinnest field's and is shown only: BRAIN reports zero for some markets "
             "that simulate fine.")
    with card("w-full"):
        source = ui.toggle({"alpha": "Alpha ID", "expression": "Expression"}, value="alpha").props(
            "no-caps unelevated dense toggle-color=primary").classes("ag-seg")
        alpha_area = ui.column().classes("w-full gap-1")
        expr_area = ui.column().classes("w-full gap-1")
        with expr_area:
            expr_box = ui.textarea("Expression", placeholder="rank(-ts_delta(close, 5))").classes("w-full ag-mono")
            ui.button("Analyse", icon="search", on_click=lambda: analyse(None)).props("unelevated")
        expr_area.set_visibility(False)
        source.on_value_change(lambda e: (alpha_area.set_visibility(e.value == "alpha"),
                                          expr_area.set_visibility(e.value == "expression")))
        with ui.expansion("Override the held settings", icon="tune").classes("w-full"):
            with ui.row().classes("gap-3 items-end"):
                o_decay = ui.number("Decay", min=0, max=MAX_DECAY, step=1).props("dense clearable").classes("w-28")
                o_trunc = ui.number("Truncation", min=0, max=1, step=0.01).props("dense clearable").classes("w-28")
                o_past = ui.select({None: "Held", "ON": "On", "OFF": "Off"}, value=None, label="Pasteurization").props(
                    "dense").classes("w-36")
                o_nan = ui.select({None: "Held", "ON": "On", "OFF": "Off"}, value=None, label="NaN Handling").props(
                    "dense").classes("w-36")
                o_test = ui.input("Test Period", placeholder="P2Y0M0D").props("dense clearable").classes("w-32")
    body = ui.column().classes("w-full gap-3")

    def params() -> dict[str, Any]:
        found = state["found"] or {}
        return {"alpha_id": found.get("alpha_id") or "",
                "expression": None if found.get("alpha_id") else found.get("expression"),
                "decay": None if o_decay.value in (None, "") else int(o_decay.value),
                "truncation": None if o_trunc.value in (None, "") else float(o_trunc.value),
                "pasteurization": o_past.value, "nan_handling": o_nan.value, "test_period": o_test.value or None,
                "markets": sorted(state["chosen"]), "neutralizations": sorted(state["neutralizations"]),
                "pairs": sorted(state["pairs"]), "decays": state["decays"],
                "market_neutral_only": state.get("market_neutral_only", True),
                "truncation_agent": state.get("truncation_agent", False),
                "visualization": state.get("visualization", False)}

    async def analyse(alpha_id: str | None) -> None:
        if alpha_id is None and not (expr_box.value or "").strip():
            ui.notify("Write an expression first.", type="warning")
            return
        try:
            found = await ss.plan(alpha_id or "", expression=None if alpha_id else expr_box.value,
                                  decay=None if o_decay.value in (None, "") else int(o_decay.value),
                                  truncation=None if o_trunc.value in (None, "") else float(o_trunc.value),
                                  pasteurization=o_past.value, nan_handling=o_nan.value,
                                  test_period=o_test.value or None)
        except Exception as exc:
            _notify(exc)
            return
        state["found"] = found
        apply(ss.defaults_choice(found))

    with alpha_area:
        alpha_id_box(lambda text: analyse(text), mark="settings")

    def apply(choice: dict[str, Any]) -> None:
        state["chosen"] = {tuple(m) for m in choice["markets"]}
        state["neutralizations"] = set(choice["neutralizations"])
        state["pairs"] = {tuple(p) for p in choice["pairs"]}
        state["decays"] = list(choice["decays"])
        render()

    def toggle_group(keys: list[tuple[str, int, str]]) -> None:
        if any(k in state["chosen"] for k in keys):
            state["chosen"] -= set(keys)
        else:
            state["chosen"] |= set(keys)
        render()

    def chip(label: str, keys: list[tuple[str, int, str]]) -> None:
        on = sum(k in state["chosen"] for k in keys)
        text = f"{label} {on}/{len(keys)}" if 0 < on < len(keys) else label
        props = "unelevated dense no-caps" if on == len(keys) else "outline dense no-caps" if on else "flat dense no-caps"
        ui.button(text, on_click=lambda: toggle_group(keys)).props(props).tooltip(
            "All chosen: click to clear" if on else "None chosen: click to choose all")

    def render() -> None:
        body.clear()
        found = state["found"]
        if not found:
            return
        markets = [m for r in found["regions"] for m in r["markets"]]
        with body:
            with card("w-full"):
                ui.label("The expression").classes("ag-section-title")
                ui.label(found["expression"]).classes("ag-mono text-sm break-all")
                with ui.row().classes("gap-2 items-center flex-wrap"):
                    ui.label("Data fields:").classes("text-sm ag-muted")
                    for f in found["data_fields"]:
                        badge(f)
                    for f in found["grouping_fields"]:
                        badge(f"{f} (grouping)")
                held_badges(found["settings"])
                problems_box(ui.column().classes("gap-1"), found["problems"], found["warnings"])
            if found["problems"]:
                return
            with card("w-full"):
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label("Where to run it").classes("ag-section-title")
                    ui.space()
                    ui.label("Presets:").classes("text-sm ag-muted")
                    ui.button("Defaults", on_click=lambda: apply(ss.defaults_choice(found))).props("flat dense no-caps")
                    ui.button("Own settings", on_click=lambda: apply(ss.own_choice(found))).props("flat dense no-caps")
                    ui.button("Decay ladder", on_click=lambda: apply(ss.ladder_choice(found))).props(
                        "flat dense no-caps").tooltip("The own settings at decay +0, +2, +5, +10, +20 (max 512)")
                for title, key in (("Region", lambda m: m["region"]), ("Delay", lambda m: f"D{m['delay']}"),
                                   ("Universe", lambda m: m["universe"])):
                    with ui.row().classes("items-center gap-1 flex-wrap"):
                        ui.label(title).classes("text-sm ag-muted w-20")
                        for value in sorted({key(m) for m in markets}):
                            chip(value, [_market_key(m) for m in markets if key(m) == value])
                with ui.expansion(f"Markets ({len(state['chosen'])} of {len(markets)} chosen)", icon="public").classes(
                        "w-full"):
                    for region in found["regions"]:
                        with ui.row().classes("items-center gap-3 flex-wrap"):
                            ui.label(region["region"] + (" (4 simulations each)" if region["region"] == REGION_AGNOSTIC
                                                         else "")).classes("w-40 font-semibold text-sm")
                            for m in region["markets"]:
                                k = _market_key(m)

                                def flip(e: Any, k: tuple = k) -> None:
                                    (state["chosen"].add if e.value else state["chosen"].discard)(k)
                                    render()
                                ui.checkbox(f"D{m['delay']} {m['universe']} · cov {fmt_pct(m['coverage'], 0)} · "
                                            f"agent {m['agent_truncation']}", value=k in state["chosen"],
                                            on_change=flip).props("dense")
                every_neut = sorted({n for r in found["regions"] for n in r["neutralizations"]})
                ui.label("Neutralizations (each region runs the ones BRAIN offers it)").classes("text-sm ag-muted")
                with ui.row().classes("gap-1 flex-wrap"):
                    for n in every_neut:
                        def neut(e: Any, n: str = n) -> None:
                            (state["neutralizations"].add if e.value else state["neutralizations"].discard)(n)
                            render()
                        ui.checkbox(paste.label("neutralization", n), value=n in state["neutralizations"],
                                    on_change=neut).props("dense")
                ui.label("Investability (Max Trade / Max Position; never both on)").classes("text-sm ag-muted")
                with ui.row().classes("gap-3"):
                    for pair, title in ((("OFF", "OFF"), "Neither"), (("ON", "OFF"), "Max Trade"),
                                        (("OFF", "ON"), "Max Position (USA, EUR, ASI, GLB)")):
                        def pr(e: Any, pair: tuple = pair) -> None:
                            (state["pairs"].add if e.value else state["pairs"].discard)(pair)
                            render()
                        ui.checkbox(title, value=pair in state["pairs"], on_change=pr).props("dense")
                with ui.row().classes("gap-4 items-center flex-wrap"):
                    def opt(name: str) -> Callable[[Any], None]:
                        def set_(e: Any) -> None:
                            state[name] = bool(e.value)
                            render()
                        return set_
                    ui.switch("Market neutral only", value=state.get("market_neutral_only", True),
                              on_change=opt("market_neutral_only")).tooltip(
                        "Drop NONE neutralization with neither Max Trade nor Max Position: that book carries the "
                        "market's own direction.")
                    ui.switch("Truncation Agent", value=state.get("truncation_agent", False),
                              on_change=opt("truncation_agent")).tooltip(
                        "Truncation set per market: 0.08 for 2000+ names, 0.06 for 1000+, else 0.05; never above "
                        "0.05 in multi-country regions or delay 0.")
                    ui.switch("Visualization", value=state.get("visualization", False),
                              on_change=opt("visualization")).tooltip("Slows every simulation; off unless asked.")
                    decays = ui.input("Decays (comma separated; empty = held)",
                                      value=", ".join(map(str, state["decays"]))).props("dense").classes("w-72")

                    def set_decays() -> None:
                        try:
                            values = sorted({int(x) for x in (decays.value or "").replace(";", ",").split(",")
                                             if x.strip()})
                        except ValueError:
                            ui.notify("Decays are whole numbers separated by commas.", type="warning")
                            return
                        if any(not 0 <= v <= MAX_DECAY for v in values) or len(values) > 10:
                            ui.notify(f"Up to 10 decays, each from 0 to {MAX_DECAY}.", type="warning")
                            return
                        state["decays"] = values
                        render()
                    decays.on("blur", set_decays)
                    decays.on("keydown.enter", set_decays)
            preview_card()

    def preview_card() -> None:
        found = state["found"]
        with card("w-full"):
            ui.label("Preview (free: nothing is simulated)").classes("ag-section-title")
            try:
                reqs = ss.requests_for(found, params(), random.Random(0))
                problems: list[str] = [] if reqs else [
                    "Nothing to run: choose a market, a neutralization and a Max Trade / Max Position pair."]
            except ToolError as exc:
                reqs, problems = [], [str(exc)]
            cost = sum(ss.simulation_cost(r["settings"]["region"]) for r in reqs)
            per: dict[tuple, int] = {}
            for r in reqs:
                per[(r["settings"]["region"], r["settings"]["delay"])] = per.get(
                    (r["settings"]["region"], r["settings"]["delay"]), 0) + 1
            batches = sum(-(-n // ss.batch_size(k[0])) for k, n in per.items())
            with ui.row().classes("gap-6"):
                stat("Simulations", fmt_int(len(reqs)))
                stat("Of today's quota", fmt_int(cost), "All Regions counts 4 each")
                stat("Batches", fmt_int(batches), "of up to 10 sharing region and delay")
                stat("Markets", fmt_int(len({(r['settings']['region'], r['settings']['delay'],
                                              r['settings']['universe']) for r in reqs})))
            problems_box(ui.column(), problems)
            sample = [{"market": settings_line(r["settings"]), "pair": f"{r['settings']['maxTrade']}/"
                       f"{r['settings']['maxPosition']}", "decay": r["settings"]["decay"]} for r in reqs[:8]]
            if sample:
                ui.aggrid(grid_options([{"field": "market", "headerName": "Sample request", "flex": 1},
                                        {"field": "pair", "headerName": "Max Trade / Position", "width": 170},
                                        {"field": "decay", "headerName": "Decay", "width": 90}], sample,
                                       pagination=False), theme="quartz").classes("w-full h-[220px]")
            if reqs:
                add_buttons(create, lambda: f"Settings Sampler · {found.get('alpha_id') or 'Expression'}")

    async def create(run: bool, cores: int, name: str | None) -> int:
        return await runner_mod.runner.create(ss.NAME, params(), cores=cores, name=name, run=run)

    task_list(ss.NAME)


# ---------------------------------------------------------------- Correlation Breaker

def breaker_panel() -> None:
    state: dict[str, Any] = {"found": None, "ticked": set(), "custom": []}
    info_box("Correlation Breaker", "For an Alpha failing Production Correlation: its settings are held whole and run "
             "in Full mode; only the expression is re-shaped around `alpha` (its own expression). Recipes this "
             "account has run three times without one success start unticked.")
    with card("w-full"):
        alpha_id_box(lambda text: analyse(text), mark="breaker")
    body = ui.column().classes("w-full gap-3")

    async def analyse(alpha_id: str) -> None:
        state["custom"] = await cb.saved_custom()
        found = await cb.plan(alpha_id, [(c["id"], c.get("name", ""), c.get("code", "")) for c in state["custom"]])
        state["found"] = found
        state["ticked"] = set(cb.default_ticks(found))
        render()

    def render() -> None:
        body.clear()
        found = state["found"]
        if not found:
            return
        with body:
            with card("w-full"):
                ui.label(f"Alpha {found['alpha_id']}").classes("ag-section-title")
                held_badges(found["settings"])
                with ui.row().classes("gap-2 flex-wrap items-center"):
                    corr = found.get("correlation") or {}
                    if corr:
                        value = corr.get("value")
                        badge(f"Production Correlation {corr.get('result')}"
                              + (f" {fmt_num(value)}" if value is not None else ""),
                              "fail" if str(corr.get("result")).upper() == "FAIL" else "")
                    if found["power_pool"]:
                        badge("Power Pool: re-shapes are held to 8 operators, 3 fields")
                    if found["single_dataset"]:
                        badge("Single data set: re-shapes reading another dataset are flagged")
                    if found["operators"] is not None:
                        badge(f"{found['operators']} operators · {found['data_fields']} data fields")
                if found["bound"]:
                    ui.label("What `alpha` stands for:").classes("text-sm ag-muted")
                    ui.code(found["bound"]).classes("w-full text-xs")
                problems_box(ui.column(), found["problems"])
            if found["problems"]:
                return
            with card("w-full"):
                ui.label("Re-shapes (ranked by what worked on this account)").classes("ag-section-title")
                for r in found["recipes"]:
                    with ui.row().classes("w-full items-start gap-3 no-wrap border-b border-[#1e2b40] py-2"):
                        def tick(e: Any, rid: str = r["id"]) -> None:
                            (state["ticked"].add if e.value else state["ticked"].discard)(rid)
                            render_preview()
                        ui.checkbox(value=r["id"] in state["ticked"], on_change=tick).props("dense").set_enabled(
                            not r["blocked"])
                        with ui.column().classes("gap-1 grow"):
                            with ui.row().classes("items-center gap-2 flex-wrap"):
                                ui.label(r["name"] or r["id"]).classes("font-semibold text-sm")
                                if r["custom"]:
                                    badge("yours")
                                if r["judged"]:
                                    badge(f"worked {r['worked']} of {r['judged']}", "pass" if r["worked"] else "")
                                if r["gave_up"]:
                                    badge("gave up after 3", "warning")
                                if r["over_power_pool"]:
                                    badge(f"leaves Power Pool: {r['over_power_pool']}", "warning")
                                if r["over_single_dataset"]:
                                    badge(f"second dataset: {r['over_single_dataset']}", "warning")
                                if not r["blocked"]:
                                    badge(f"{r['operators']} ops · {r['data_fields']} fields")
                            if r["why"]:
                                ui.label(r["why"]).classes("text-sm ag-muted")
                            if r["caution"]:
                                ui.label(f"Caution: {r['caution']}").classes("text-xs ag-amber")
                            if r["transform"]:
                                ui.code(r["transform"]).classes("text-xs w-full")
                            if r["blocked"]:
                                ui.label(f"Cannot run here: {r['blocked']}").classes("text-xs ag-amber")
            custom_card()
            preview = ui.column().classes("w-full")
            state["preview"] = preview
            render_preview()

    def custom_card() -> None:
        with card("w-full"):
            ui.label("Your own re-shapes (kept on this computer)").classes("ag-section-title")
            ui.label("Write any statements you need, then the line that runs, around `alpha`. E.g. "
                     "`group_rank(alpha, sector)`.").classes("text-xs ag-muted")
            rows = state["custom"]
            for c in rows:
                with ui.row().classes("w-full items-start gap-2"):
                    name = ui.input("Name", value=c.get("name", "")).props("dense").classes("w-60")
                    code = ui.textarea("Re-shape", value=c.get("code", "")).props("dense autogrow").classes(
                        "grow ag-mono")
                    name.on_value_change(lambda e, c=c: c.__setitem__("name", e.value))
                    code.on_value_change(lambda e, c=c: c.__setitem__("code", e.value))
                    ui.button(icon="delete", on_click=lambda c=c: (rows.remove(c), render())).props("flat dense")

            def add() -> None:
                rows.append({"id": f"r{random.randrange(16 ** 6):06x}", "name": "", "code": ""})
                render()

            async def save() -> None:
                try:
                    await cb.save_custom(rows)
                    await analyse(state["found"]["alpha_id"])
                    ui.notify("Re-shapes saved.", type="positive")
                except Exception as exc:
                    _notify(exc)
            with ui.row().classes("gap-2"):
                ui.button("Add re-shape", icon="add", on_click=add).props("flat dense no-caps")
                ui.button("Save and re-check", icon="save", on_click=save).props("outline dense no-caps")

    def render_preview() -> None:
        preview = state.get("preview")
        found = state["found"]
        if preview is None or not found:
            return
        preview.clear()
        with preview, card("w-full"):
            ui.label("Preview (free: nothing is simulated)").classes("ag-section-title")
            try:
                reqs = cb.chosen_requests(found, sorted(state["ticked"])) if state["ticked"] else []
                problems = [] if reqs else ["Tick at least one re-shape that can run here."]
            except ToolError as exc:
                reqs, problems = [], [str(exc)]
            with ui.row().classes("gap-6"):
                stat("Simulations", fmt_int(len(reqs)), "one per re-shape, Full mode")
                stat("Batches", fmt_int(-(-len(reqs) // 10)))
            problems_box(ui.column(), problems)
            if reqs:
                ui.code(reqs[0]["expression"]).classes("text-xs w-full")
                add_buttons(create, lambda: f"Correlation Breaker · {found['alpha_id']}")

    async def create(run: bool, cores: int, name: str | None) -> int:
        found = state["found"]
        return await runner_mod.runner.create(cb.NAME, {"alpha_id": found["alpha_id"],
                                                        "recipes": sorted(state["ticked"])},
                                              cores=cores, name=name, run=run)

    task_list(cb.NAME)


# ---------------------------------------------------------------- Fields Sampler

def fields_panel() -> None:
    state: dict[str, Any] = {"found": None, "fields": set(), "alphas": set()}
    info_box("Fields Sampler", "One field changes at a time, everything else held. Sweep: every other field of its "
             "dataset. Similar: the fields whose descriptions read closest (k per tier: dataset, subcategory, "
             "category, market). Buckets: every field swapped at once, each into its own kind. In USA D1 the Fast D1 "
             "twin and its overnight delta lead. A swap that passes is swept the same way (a tree).")
    with card("w-full"):
        with ui.row().classes("items-end gap-3"):
            mode = ui.toggle({fs.SWEEP: "Sweep", fs.SIMILAR: "Similar", fs.BUCKETS: "Buckets"}, value=fs.SWEEP).props(
                "no-caps unelevated dense toggle-color=primary").classes("ag-seg")
            k = ui.number("k (per tier / bucket picks)", value=fs.DEFAULT_K, min=1, max=50, step=1).props(
                "dense").classes("w-48")
            budget = ui.number("Budget (simulations, empty = until nothing new)", min=1, step=10).props(
                "dense clearable").classes("w-80")
        alpha_id_box(lambda text: analyse(text), mark="fields")
    body = ui.column().classes("w-full gap-3")

    async def analyse(alpha_id: str) -> None:
        found = await fs.plan(alpha_id, mode.value, int(k.value or fs.DEFAULT_K))
        state.update(found=found, fields={s["field"] for s in found["slots"]}, alphas=set())
        await render()

    def params() -> dict[str, Any]:
        found = state["found"]
        return {"alpha_id": found["alpha_id"], "mode": found["mode"], "k": found["k"],
                "fields": sorted(state["fields"]), "alphas": sorted(state["alphas"]),
                "budget": int(budget.value) if budget.value else None}

    async def render() -> None:
        body.clear()
        found = state["found"]
        if not found:
            return
        with body:
            with card("w-full"):
                ui.label(f"Alpha {found['alpha_id']} · {found['mode'].title()}").classes("ag-section-title")
                ui.label(found["expression"]).classes("ag-mono text-sm break-all")
                held_badges(found["settings"])
                if found["bucketable"] and found["mode"] != fs.BUCKETS:
                    ui.label("It reads several fields of one dataset: Buckets can swap them together.").classes(
                        "text-xs ag-muted")
                problems_box(ui.column(), found["problems"])
            if found["problems"]:
                return
            with card("w-full"):
                ui.label("Its fields and what each could become").classes("ag-section-title")
                for slot in found["slots"]:
                    with ui.row().classes("items-center gap-2"):
                        def tick(e: Any, f: str = slot["field"]) -> None:
                            (state["fields"].add if e.value else state["fields"].discard)(f)
                        cbx = ui.checkbox(f"{slot['field']} · {len(slot['candidates'])} swaps",
                                          value=slot["field"] in state["fields"], on_change=tick).props("dense")
                        cbx.set_enabled(found["mode"] != fs.BUCKETS)
                        if slot["problem"]:
                            ui.label(slot["problem"]).classes("text-xs ag-amber")
                    rows = [{**c, "coverage": c["coverage"]} for c in slot["candidates"][:200]]
                    if rows:
                        ui.aggrid(grid_options([{"field": "field", "headerName": "Swap", "flex": 1},
                                                {"field": "dataset", "headerName": "Dataset", "width": 140},
                                                {"field": "tier", "headerName": "Found by", "width": 120},
                                                num_col("coverage", "Coverage", 0, 110, pct=True)], rows,
                                               pagination=False), theme="quartz").classes("w-full h-[180px]")
            if found["others"]:
                with card("w-full"):
                    ui.label("Also submitted in (run each swap at these settings too)").classes("ag-section-title")
                    for o in found["others"]:
                        def other(e: Any, a: str = o["alpha_id"]) -> None:
                            (state["alphas"].add if e.value else state["alphas"].discard)(a)
                        counts = ", ".join(f"{s['field']} {s['count']}" for s in o["swaps"])
                        ui.checkbox(f"{o['alpha_id']} ({o['status']}) · {settings_line(o['raw_settings'])} · swaps: "
                                    f"{counts}", value=False, on_change=other).props("dense")
            with card("w-full"):
                ui.label("Preview (free: nothing is simulated)").classes("ag-section-title")
                out = ui.column().classes("w-full")

                async def show() -> None:
                    out.clear()
                    with out:
                        try:
                            chosen, cap, _fields = await fs.chosen_variants(found, params())
                        except ToolError as exc:
                            problems_box(ui.column(), [str(exc)])
                            return
                        with ui.row().classes("gap-6"):
                            stat("First sweep", fmt_int(len(chosen)))
                            stat("Budget", "until nothing new" if cap >= 100_000 else fmt_int(cap),
                                 "winners' sweeps included")
                            stat("Batches", fmt_int(-(-len(chosen) // 10)))
                        sample = [{"expression": v.expression, "swap": f"{v.field} → {v.swap}", "tier": v.tier}
                                  for v in chosen[:8]]
                        ui.aggrid(grid_options([{"field": "swap", "headerName": "Swap", "width": 260},
                                                {"field": "tier", "headerName": "Found by", "width": 110},
                                                {"field": "expression", "headerName": "Sample request", "flex": 1}],
                                               sample, pagination=False), theme="quartz").classes("w-full h-[220px]")
                        add_buttons(create, lambda: f"Fields Sampler · {found['alpha_id']}"
                                    + (f" +{len(state['alphas'])}" if state["alphas"] else ""))
                ui.button("Update preview", icon="refresh", on_click=show).props("flat dense no-caps")
                await show()

    async def create(run: bool, cores: int, name: str | None) -> int:
        return await runner_mod.runner.create(fs.NAME, params(), cores=cores, name=name, run=run)

    async def show_tree(tid: int) -> None:
        trials = await runner_mod.runner.trials(tid)
        with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[900px] max-w-[95vw]"):
            ui.label(f"Task #{tid}: each swap under the winner it grew from").classes("ag-section-title")
            ui.tree(fs.tree(trials), label_key="label").classes("w-full max-h-[600px] overflow-auto text-sm")
            ui.button("Close", on_click=d.close).props("flat")
        d.open()

    task_list(fs.NAME, [("Tree", "account_tree", show_tree)])


# ---------------------------------------------------------------- Paste Alphas

def paste_panel() -> None:
    state: dict[str, Any] = {"found": None}
    info_box("Paste Alphas", "Paste an LLM's answer. Each Alpha starts with 'Alpha Expression:' (or a code block) "
             "with its settings around it, as Copy Alphas writes them; Markdown is fine. Region, Delay and Universe "
             "are required; anything else left out takes BRAIN's default where BRAIN offers it. A repeat runs once.")
    with card("w-full"):
        text = ui.textarea("The LLM's output").props("autogrow").classes("w-full ag-mono")
        ui.button("Read it", icon="fact_check", on_click=lambda: read()).props("unelevated")
    body = ui.column().classes("w-full gap-3")

    async def read() -> None:
        try:
            state["found"] = await paste.plan(text.value or "")
        except Exception as exc:
            _notify(exc)
            return
        render()

    def render() -> None:
        body.clear()
        found = state["found"]
        with body, card("w-full"):
            ui.label("Preview (free: nothing is simulated)").classes("ag-section-title")
            with ui.row().classes("gap-6"):
                stat("Alphas read", fmt_int(len(found["alphas"])))
                stat("Simulations", fmt_int(found["simulations"]), "every one that can run, once")
                stat("Of today's quota", fmt_int(found["cost"]), "All Regions counts 4 each")
                stat("Batches", fmt_int(found["batches"]))
            problems_box(ui.column(), found["problems"])
            rows = [{"number": a["number"], "expression": a["expression"],
                     "settings": " · ".join(f"{paste.TITLES[k]} {v}" for k, v in a["labels"].items()),
                     "categories": ", ".join(a["categories"]) or "—",
                     "defaulted": ", ".join(a["defaulted"]) or "—",
                     "state": ("Cannot run: " + "; ".join(a["problems"])) if a["problems"]
                     else (f"Same as #{a['duplicate_of']}: runs once" if a["duplicate_of"] else "Ready"),
                     "many": len(a["categories"]) > 2} for a in found["alphas"]]
            ui.aggrid(grid_options([{"field": "number", "headerName": "#", "width": 60},
                                    {"field": "state", "headerName": "State", "width": 260},
                                    {"field": "expression", "headerName": "Expression", "flex": 1, "minWidth": 260},
                                    {"field": "settings", "headerName": "Settings", "width": 420},
                                    {"field": "categories", "headerName": "Categories read", "width": 180,
                                     "tooltipValueGetter": "(p) => 'Over 2 categories counts toward no Genius "
                                                           "pyramid'"},
                                    {"field": "defaulted", "headerName": "Defaulted", "width": 200}], rows),
                      theme="quartz").classes("w-full h-[320px]")
            if found["requests"] and not found["problems"]:
                add_buttons(create, lambda: f"Paste Alphas · {found['simulations']} Alphas")

    async def create(run: bool, cores: int, name: str | None) -> int:
        found = await paste.plan(text.value or "")
        if found["problems"]:
            raise ToolError(found["problems"][0])
        return await runner_mod.runner.create(paste.NAME, {"text": text.value or ""}, cores=cores,
                                              max_trials=found["simulations"], name=name, run=run)

    async def copy_back(tid: int) -> None:
        try:
            out = await paste.copy_results(tid)
        except Exception as exc:
            _notify(exc)
            return
        with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[900px] max-w-[95vw]"):
            ui.label(f"Results of task #{tid} in Copy Alphas format").classes("ag-section-title")
            notes = [f"{out['alphas']} alpha(s)", f"{out['failed']} failed (listed with BRAIN's reason)"]
            if out["waiting"]:
                notes.append(f"{out['waiting']} not back yet")
            if out["downloading"]:
                notes.append(f"{out['downloading']} PnL download(s) queued: their K-Ratio and correlations read — "
                             "until done")
            ui.label(" · ".join(notes)).classes("text-sm ag-muted")
            ui.textarea(value=out["text"]).props("readonly autogrow").classes("w-full ag-mono text-xs max-h-[520px]")
            with ui.row().classes("justify-end w-full"):
                ui.button("Copy", icon="content_copy", on_click=lambda: (ui.clipboard.write(out["text"]),
                                                                         ui.notify("Copied.", type="positive")))
                ui.button("Close", on_click=d.close).props("flat")
        d.open()

    task_list(paste.NAME, [("Copy results", "content_copy", copy_back)])


# ---------------------------------------------------------------- Copy Alphas and the description writer

def copy_panel() -> None:
    from sqlalchemy import select

    from ...db import db
    from ...db.models import Alpha
    from ...tools.copy_alphas import alphas_markdown
    from ...tools.description import description_problems, write_description
    with card("w-full"):
        ui.label("Copy Alphas (Markdown for an LLM)").classes("ag-section-title")
        ui.label("Alpha IDs separated by spaces, commas or lines. Paste the LLM's answer into Paste Alphas to run "
                 "what it proposes.").classes("text-xs ag-muted")
        ids_box = ui.textarea("Alpha IDs").props("autogrow").classes("w-full ag-mono")
        glossary = ui.switch("Include the metric and field glossaries", value=True)
        out = ui.textarea("Markdown").props("readonly autogrow").classes("w-full ag-mono text-xs")
        out.set_visibility(False)

        async def make() -> None:
            ids = [x for x in (ids_box.value or "").replace(",", " ").split() if x]
            if not ids:
                ui.notify("Enter at least one Alpha ID.", type="warning")
                return
            bad = [x for x in ids if not looks_like_id(x)]
            if bad:
                ui.notify(f"{bad[0][:40]}: {NOT_AN_ID}", type="warning", multi_line=True)
                return
            async with db.session() as s:
                alphas = list(await s.scalars(select(Alpha).where(Alpha.id.in_(ids))))
            missing = sorted(set(ids) - {a.id for a in alphas})
            if missing:
                ui.notify(f"Not stored on this computer (sync first): {', '.join(missing[:5])}", type="warning")
            if not alphas:
                return
            try:
                text = await alphas_markdown(alphas, glossary=bool(glossary.value))
            except Exception as exc:
                _notify(exc)
                return
            out.value = text
            out.set_visibility(True)
            ui.clipboard.write(text)
            ui.notify(f"{len(alphas)} alpha(s) copied as Markdown.", type="positive")
        ui.button("Copy as Markdown", icon="content_copy", on_click=make).props("unelevated")
    with card("w-full"):
        ui.label("Power Pool description (written by AI)").classes("ag-section-title")
        ui.label("The model sees the expression and only its own fields and operators with BRAIN's descriptions, and "
                 "fills BRAIN's Idea / Rationale template, each part at least 100 characters. Nothing is saved to "
                 "BRAIN: edit it, then save it on the Alpha's page.").classes("text-xs ag-muted")
        written = ui.textarea("Description").props("autogrow").classes("w-full")
        problems = ui.column().classes("gap-1")

        def check(_e: Any = None) -> None:
            problems.clear()
            if written.value:
                found = description_problems(written.value)
                with problems:
                    if found:
                        for p in found:
                            ui.label(f"BRAIN would refuse it: {p}").classes("text-sm ag-amber")
                    else:
                        ui.label("✓ In BRAIN's template").classes("text-sm ag-green")
        written.on_value_change(check)

        async def write(alpha_id: str) -> None:
            ui.notify("Writing…")
            written.value = await write_description(alpha_id)
            check()
        alpha_id_box(write, "Alpha ID to describe")


PANELS = {"settings": settings_sampler_panel, "breaker": breaker_panel, "fields": fields_panel, "paste": paste_panel,
          "copy": copy_panel}


async def page() -> None:
    page_header("Tools", "Work on an Alpha you already have: sweep its settings, break its correlation, swap its "
                         "fields, or run what an LLM wrote. Every preview is free; simulations run as tasks.")
    with ui.row().classes("w-full gap-3 flex-wrap"):
        hub = {}
        for key, title, icon, about in TOOLS:
            with ui.card().classes("ag-card p-4 grow min-w-[220px] max-w-[320px] cursor-pointer") as c:
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.icon(icon).classes("ag-cyan text-xl")
                    ui.label(title).classes("font-semibold")
                ui.label(about).classes("text-sm ag-muted")
            hub[key] = c
    with ui.tabs().classes("w-full") as tabs:
        tab = {key: ui.tab(key, label=title, icon=icon) for key, title, icon, _ in TOOLS}
    for key, c in hub.items():
        c.on("click", lambda _e, key=key: tabs.set_value(tab[key]))
    with ui.tab_panels(tabs, value=tab["settings"]).classes("w-full bg-transparent"):
        for key, *_ in TOOLS:
            with ui.tab_panel(tab[key]).classes("gap-3 px-0"):
                PANELS[key]()
