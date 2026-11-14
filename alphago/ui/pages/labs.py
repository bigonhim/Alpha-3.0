"""Research Labs (ported from alpha-harness research-labs and tasks screens).

Search, Template and Evolution labs plan a task for free (Preview), then add it to Tasks, where it runs on its share
of the eight simulation cores. Power Pool and Super Alpha tabs come from `labs_llm`. Tasks lists every lab task with
its progress and controls; Submittable lists every Alpha any task found that BRAIN would take.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from nicegui import ui

from ...brain import brain
from ...brain.platform import MAX_TRADE_REGIONS, SIMULATION_SLOTS, TEST_PERIOD
from ...db import db
from ...labs import ga, launch, results, templates_gallery
from ...labs import runner as runner_mod
from ...labs import search as search_lab
from ...labs import template as template_lab
from ...labs.tree import operator_table
from ..alpha_detail import open_alpha
from ..components import (
    badge,
    card,
    empty_state,
    fmt_duration,
    fmt_int,
    fmt_num,
    grid_options,
    info_box,
    notify_error,
    num_col,
    page_header,
    segmented,
    stat_tile,
)

TABS = ("Search", "Template", "Evolution", "Power Pool", "Super Alpha", "Tasks", "Submittable")
CORES = {i: str(i) for i in range(1, SIMULATION_SLOTS + 1)}
RISK_TITLE = "Risk (FAST, SLOW, … RAM)"
HANDOFF_KEY = "handoff:search"          # {region, delay, universe, dataset_ids, field_ids} from the assistant
TRUNCATION_EXAMPLES = [("USA · Delay 1 · TOP3000", "0.08"), ("USA · Delay 1 · TOP1000", "0.06"),
                       ("USA · Delay 1 · TOP200", "0.05"), ("USA · Delay 0 · TOP3000", "0.05"),
                       ("JPN · Delay 1 · TOP1600", "0.06"), ("GLB · Delay 1 · TOP3000", "0.05")]


def runner() -> runner_mod.LabRunner:
    return runner_mod.runner


def neutral_label(value: str) -> str:
    return " ".join(w.capitalize() for w in value.split("_") if w)


# ----------------------------------------------------------------- shared pieces

class NeutralizationPicker:
    """Every neutralization the market offers, in BRAIN's two families, each with a button that ticks or unticks
    the whole family and an "n of m" count. The families are a shortcut, never a restriction."""

    def __init__(self, chosen: list[str] | None = None, legend: str = "Neutralization") -> None:
        self.chosen: list[str] = list(chosen or ["INDUSTRY"])
        self.legend = legend
        self.box = ui.column().classes("w-full gap-2")
        self.region = ""

    def show(self, region: str) -> None:
        self.region = region
        offered = list(brain.options.neutralizations(region))
        self.chosen = [n for n in self.chosen if n in offered]
        self.box.clear()
        with self.box:
            ui.label(self.legend).classes("text-sm ag-muted")
            with ui.row().classes("w-full gap-3 items-start"):
                for family, title in (("risk", RISK_TITLE), ("group", "Group")):
                    items = [n for n in offered if launch.neutralization_family(n) == family]
                    if items:
                        self._family(title, items)

    def _family(self, title: str, items: list[str]) -> None:
        with ui.card().classes("ag-card p-3 grow min-w-[260px]"):
            with ui.row().classes("items-center justify-between w-full no-wrap"):
                with ui.row().classes("items-baseline gap-2"):
                    ui.label(title).classes("font-medium")
                    count = ui.label("").classes("ag-muted text-xs ag-mono")
                button = ui.button("").props("flat dense no-caps size=sm")
            boxes: dict[str, ui.checkbox] = {}

            def refresh() -> None:
                on = sum(1 for n in items if n in self.chosen)
                count.text = f"{on} of {len(items)}"
                button.text = "Clear all" if on == len(items) else "Select all"

            def toggle(name: str, value: bool) -> None:
                if value and name not in self.chosen:
                    self.chosen.append(name)
                elif not value and name in self.chosen:
                    self.chosen.remove(name)
                refresh()

            def whole() -> None:
                everything = all(n in self.chosen for n in items)
                for n, b in boxes.items():
                    b.value = not everything
                    toggle(n, not everything)

            with ui.row().classes("gap-x-3 gap-y-0 flex-wrap"):
                for n in items:
                    boxes[n] = ui.checkbox(neutral_label(n), value=n in self.chosen,
                                           on_change=lambda e, n=n: toggle(n, e.value)).props("dense")
            button.on("click", lambda _: whole())
            refresh()


def truncation_agent_info() -> None:
    with ui.expansion("Truncation Agent", icon="info").classes("w-full ag-muted text-sm"):
        ui.markdown(
            "Truncation sets the maximum weight of one stock in the Alpha. BRAIN fails the Weight test when one stock "
            "has too much weight: for consultants the limit is 8% in USA and 10% elsewhere.\n\n"
            "The agent sets Truncation for each market with fixed rules: a Universe of 2,000 names or more takes "
            "**0.08**, 1,000 or more **0.06**, fewer **0.05**. A region of many countries, or Delay 0, never takes "
            "more than **0.05**. Nothing goes above 0.08.").classes("text-sm")
        with ui.row().classes("gap-4 flex-wrap"):
            for market, value in TRUNCATION_EXAMPLES:
                ui.label(f"{market} → {value}").classes("ag-mono text-xs")


class TaskControls:
    """Cores, simulations and name: what every lab's task needs, and the free Preview / Add to Tasks / Run now."""

    def __init__(self, sims: int = 500) -> None:
        with ui.row().classes("items-end gap-4 flex-wrap w-full"):
            with ui.column().classes("gap-0"):
                ui.label("Cores").classes("text-sm ag-muted")
                self.cores = segmented(CORES, 4)
            self.sims = ui.number("Simulations", value=sims, min=1, max=search_lab.MAX_SIMULATIONS, step=50,
                                  format="%d").classes("w-36")
            self.name = ui.input("Task name (optional)").classes("w-64")
        ui.label("Each core keeps 10 simulations in flight. A task waits in Tasks until its cores are free, and "
                 "may carry on through the next days' allowances.").classes("ag-muted text-xs")

    def values(self) -> tuple[int, int, str | None]:
        return int(self.cores.value or 1), int(self.sims.value or 0), (self.name.value or "").strip() or None

    def buttons(self, preview: Callable[[], Awaitable[None]], add: Callable[[bool], Awaitable[None]]) -> None:
        with ui.row().classes("gap-2 mt-2"):
            ui.button("Preview", icon="visibility", on_click=preview).props("outline no-caps")
            ui.button("Add to Tasks", icon="playlist_add", on_click=lambda: add(False)).props("outline no-caps")
            ui.button("Run now", icon="play_arrow", on_click=lambda: add(True)).props("unelevated no-caps")


def show_problems(problems: list[str], warnings: list[str]) -> None:
    for p in problems:
        with ui.row().classes("items-start gap-2 no-wrap"):
            ui.icon("error").classes("ag-red")
            ui.label(p).classes("text-sm")
    for w in warnings:
        with ui.row().classes("items-start gap-2 no-wrap"):
            ui.icon("warning").classes("ag-amber")
            ui.label(w).classes("text-sm")


def show_samples(sample: list[dict[str, Any]]) -> None:
    if not sample:
        return
    ui.label("Example Alphas this task could send").classes("ag-section-title mt-2")
    for s in sample:
        st = s.get("settings") or {}
        with ui.column().classes("gap-0 w-full"):
            ui.label(s.get("expression") or "").classes("ag-mono text-sm break-all")
            ui.label(" · ".join(str(x) for x in (st.get("universe"), st.get("neutralization"), f"decay {st.get('decay')}",
                                                 f"trunc {st.get('truncation')}") if x)).classes("ag-muted text-xs")


async def add_task(lab: str, params: dict[str, Any], controls: TaskControls, run: bool) -> None:
    cores, sims, name = controls.values()
    if sims < 1:
        ui.notify(launch.NO_SIMULATIONS, type="warning")
        return
    try:
        tid = await runner().create(lab, {**params, "cores": cores}, cores=cores, max_trials=sims, name=name, run=run)
    except ValueError as exc:
        ui.notify(f"Not added: {exc}", type="negative", multi_line=True, close_button=True, timeout=10000)
        return
    except Exception as exc:
        notify_error(exc, "Could not add the task: ")
        return
    ui.notify(f"Task #{tid} {'is running (or waiting for cores)' if run else 'added to Tasks'}.", type="positive")


class MarketPicker:
    """Region, delay and universe from BRAIN's options for this account."""

    def __init__(self, on_change: Callable[[], Any], universe: bool = True) -> None:
        from ...config import settings
        opts = brain.options
        r = settings.research
        regions = list(opts.regions) or ["USA"]
        with ui.row().classes("gap-2 items-end"):
            self.region = ui.select(regions, value=r.region if r.region in regions else regions[0],
                                    label="Region").classes("w-28")
            delays = opts.delays(self.region.value)
            self.delay = ui.select(delays, value=r.delay if r.delay in delays else delays[0], label="Delay") \
                .classes("w-24")
            self.universe = None
            if universe:
                us = opts.universes(self.region.value)
                self.universe = ui.select(us, value=r.universe if r.universe in us else us[0], label="Universe") \
                    .classes("w-40")
        self._on_change = on_change

        def region_changed() -> None:
            self.delay.options = opts.delays(self.region.value)
            self.delay.value = self.delay.options[0]
            if self.universe is not None:
                self.universe.options = opts.universes(self.region.value)
                self.universe.value = self.universe.options[0]
            self.delay.update()
            on_change()

        self.region.on_value_change(lambda _: region_changed())
        self.delay.on_value_change(lambda _: on_change())
        if self.universe is not None:
            self.universe.on_value_change(lambda _: on_change())

    def values(self) -> dict[str, Any]:
        out = {"region": self.region.value, "delay": int(self.delay.value)}
        if self.universe is not None:
            out["universe"] = self.universe.value
        return out


async def dataset_options(region: str, delay: int) -> dict[str, str]:
    rows = await launch.datasets_for(region, delay)
    return {d["id"]: f"{d['id']} — {d['name']} ({fmt_int(d['fields'])} fields)" for d in rows}


# ----------------------------------------------------------------- Search

async def search_tab() -> None:
    info_box("Search Lab", "First- and second-degree Alphas over the datasets you choose: every field is cleaned "
             "with ts_backfill, then written in six shapes (cs, ts, group and their compositions). A TPE search "
             "gives every field one trial first, then concentrates on what scores. Every simulation holds the last "
             "two years out and is scored on its train-years Sharpe.")
    operators = await launch.account_operators()
    vector_ops = list(search_lab.catalogue(operators).vector)
    with card("w-full"):
        datasets = ui.select({}, multiple=True, label="Datasets", with_input=True).props("use-chips").classes("w-full")

        async def market_changed() -> None:
            v = market.values()
            datasets.options = await dataset_options(v["region"], v["delay"])
            datasets.value = [d for d in (datasets.value or []) if d in datasets.options]
            datasets.update()
            neut.show(v["region"])

        market = MarketPicker(lambda: ui.timer(0, market_changed, once=True))
        only = ui.checkbox("Only this universe (keep the market's other downloaded universes out)")
        handoff = await db.kv_get(HANDOFF_KEY)
        field_ids: list[str] = []
        fields_note = ui.row().classes("items-center gap-2")
        with ui.row().classes("gap-4 items-end w-full"):
            vector = ui.select(vector_ops, multiple=True, label="Vector operators (to use vector fields)") \
                .props("use-chips").classes("min-w-[260px]")
            types = ui.select(["MATRIX", "VECTOR"], multiple=True, label="Field types (all when empty)") \
                .props("use-chips").classes("w-60")
            coverage = ui.number("Min coverage (0–1)", min=0, max=1, step=0.05).classes("w-40")
        neut = NeutralizationPicker(["INDUSTRY", "SUBINDUSTRY"])
        with ui.row().classes("gap-4 items-end"):
            decays = ui.input("Decays", value="0").classes("w-48").tooltip(
                "Whole numbers from 0 to 512, separated by commas. With several, each Alpha gets one and the search "
                "steers towards the best.")
            agent = ui.checkbox("Truncation Agent (else 0.08)")
            quick = ui.checkbox("Quick mode (passing Alphas are re-run in Full automatically)")
        truncation_agent_info()
        controls = TaskControls(500)
        out = ui.column().classes("w-full gap-1")

        def params() -> dict[str, Any]:
            return {**market.values(), "only_universe": bool(only.value), "dataset_ids": list(datasets.value or []),
                    "vector_operators": list(vector.value or []), "field_types": list(types.value or []),
                    "min_coverage": coverage.value, "neutralizations": list(neut.chosen),
                    "decays": decays.value or "0", "truncation_agent": bool(agent.value), "quick": bool(quick.value),
                    "field_ids": list(field_ids),
                    "cores": controls.values()[0]}

        async def preview() -> None:
            try:
                plan = await search_lab.plan(params())
            except Exception as exc:
                notify_error(exc, "Could not plan the task: ")
                return
            out.clear()
            with out:
                with ui.row().classes("gap-3 w-full"):
                    for label, value in (("Round", f"{plan['round']} simulations"),
                                         ("Fields", f"{fmt_int(plan['fields']['total'])} "
                                                    f"({fmt_int(plan['fields']['matrix'])} matrix, "
                                                    f"{fmt_int(plan['fields']['vector'])} vector)"),
                                         ("Universes", ", ".join(plan["universes"]) or "—"),
                                         ("Vector fields left out", fmt_int(plan["left_out"]["vector"]))):
                        v, _ = stat_tile(label)
                        v.text = value
                show_problems(plan["problems"], plan["warnings"])
                show_samples(plan["sample"])

        controls.buttons(preview, lambda run: add_task("search", params(), controls, run))

        def show_fields_note() -> None:
            fields_note.clear()
            if field_ids:
                with fields_note:
                    badge(f"Only {len(field_ids)} chosen field{'s' if len(field_ids) != 1 else ''}", "pending") \
                        .tooltip(", ".join(field_ids[:40]))
                    ui.button("Search every field", on_click=lambda: (field_ids.clear(), show_fields_note())) \
                        .props("flat dense no-caps")

        if isinstance(handoff, dict) and handoff:
            # handed over by the assistant or the Data Explorer: read once, then cleared
            await db.kv_set(HANDOFF_KEY, None)
            if handoff.get("region") in (market.region.options or []):
                market.region.value = handoff["region"]
            if handoff.get("delay") is not None and int(handoff["delay"]) in (market.delay.options or []):
                market.delay.value = int(handoff["delay"])
            if market.universe is not None and handoff.get("universe") in (market.universe.options or []):
                market.universe.value = handoff["universe"]
            field_ids.extend(str(f) for f in handoff.get("field_ids") or [])
            await market_changed()
            datasets.value = [d for d in handoff.get("dataset_ids") or [] if d in datasets.options]
            show_fields_note()
            ui.notify("The Search Lab is filled in from the hand-off: check it, then Preview.", type="info")
        else:
            await market_changed()


# ----------------------------------------------------------------- Template

class TemplateEditor:
    """The template editor: code with live lint, the variables panel, settings, saved templates and the gallery."""

    def __init__(self, operators: list[dict[str, Any]]) -> None:
        self.table = operator_table(operators)
        self.vector_ops = list(search_lab.catalogue(operators).vector)
        self.given: dict[str, dict[str, Any]] = {}
        self.datasets: list[dict[str, Any]] = []
        self.saved_id: int | None = None
        self.dirty_lint = True
        self.names: list[str] = []

    async def build(self) -> None:
        info_box("Template Lab", "Write an Alpha with $variables where the search chooses: fields from datasets you "
                 "pick, or values you type (numbers, groups, operator names, expressions). $name(...) is an operator. "
                 "... marks a signal still to write. Preview is free: it counts the combinations, checks every "
                 "variable and shows example Alphas.")
        with ui.expansion("Templates", icon="collections_bookmark", value=True).classes("w-full ag-card"):
            self.gallery = ui.column().classes("w-full gap-2")
        with card("w-full"):
            with ui.row().classes("gap-3 items-end w-full"):
                self.tname = ui.input("Template name").classes("w-64")
                self.desc = ui.input("Description").classes("grow")
                ui.button("Save", icon="save", on_click=self.save).props("outline no-caps dense")
                ui.button("Save As", icon="save_as", on_click=lambda: self.save(as_new=True)).props("outline no-caps dense")
                ui.button("New", icon="add", on_click=self.new).props("flat no-caps dense")
                self.delete_btn = ui.button("Delete", icon="delete", on_click=self.delete).props("flat no-caps dense")
            self.text = ui.textarea("Template", placeholder="e.g. group_rank($ts_op($field, $lookback), $group)") \
                .props("outlined autogrow input-style='font-family: var(--ag-mono)'").classes("w-full")
            self.text.on_value_change(lambda _: self._touched())
            self.lint_box = ui.column().classes("w-full gap-0")
            ui.label("Variables").classes("ag-section-title mt-2")
            self.vars_box = ui.column().classes("w-full gap-2")
        with card("w-full"):
            ui.label("Settings").classes("ag-section-title")
            self.market = MarketPicker(lambda: ui.timer(0, self.market_changed, once=True), universe=False)
            self.universes = ui.select([], multiple=True, label="Universes").props("use-chips").classes("w-full")
            self.neut = NeutralizationPicker(["INDUSTRY"])
            with ui.row().classes("gap-4 items-end"):
                self.invest_box = ui.row().classes("gap-2 items-center")
                self.decay = ui.number("Decay", value=0, min=0, max=512, format="%d").classes("w-24")
                self.trunc = ui.number("Truncation", value=search_lab.TRUNCATION, min=0, max=1, step=0.01) \
                    .classes("w-28")
                self.agent = ui.checkbox("Truncation Agent")
                self.paste = ui.select(["ON", "OFF"], value="ON", label="Pasteurization").classes("w-32")
                self.nan = ui.select(["ON", "OFF"], value="ON", label="NaN handling").classes("w-32")
                self.test_period = ui.input("Test period", value=TEST_PERIOD).classes("w-32").tooltip(
                    "P{years}Y{months}M0D, at most P6Y0M0D. P0Y0M0D scores the whole period on Sharpe.")
                self.quick = ui.checkbox("Quick mode")
            self.invest_warning = ui.label("").classes("ag-amber text-sm")
            truncation_agent_info()
            self.controls = TaskControls(500)
            self.out = ui.column().classes("w-full gap-1")
            self.controls.buttons(self.preview, lambda run: add_task("template", self.params(), self.controls, run))
        ui.timer(0.7, self._lint_tick)
        await self.market_changed()
        await self.show_gallery()

    # -- state
    def _touched(self) -> None:
        self.dirty_lint = True

    def _resolved(self) -> dict[str, dict[str, Any]]:
        return template_lab.resolve(self.text.value or "", self.given, self.datasets)

    def params(self) -> dict[str, Any]:
        cores, _, _ = self.controls.values()
        return {**self.market.values(), "template": self.text.value or "", "variables": self._resolved(),
                "universes": list(self.universes.value or []), "neutralizations": list(self.neut.chosen),
                "investability": [k for k, b in self.invest.items() if b.value], "decay": int(self.decay.value or 0),
                "truncation": float(self.trunc.value if self.trunc.value is not None else search_lab.TRUNCATION),
                "truncation_agent": bool(self.agent.value), "pasteurization": self.paste.value,
                "nan_handling": self.nan.value, "test_period": (self.test_period.value or TEST_PERIOD).strip(),
                "quick": bool(self.quick.value), "cores": cores, "template_name": self.tname.value or "Template"}

    async def market_changed(self) -> None:
        v = self.market.values()
        offered = await launch.synced_universes(v["region"], v["delay"])
        self.universes.options = offered
        self.universes.value = [u for u in (self.universes.value or []) if u in offered] or offered[:1]
        self.universes.update()
        self.datasets = await launch.datasets_for(v["region"], v["delay"])
        self.neut.show(v["region"])
        self.invest_box.clear()
        self.invest = {}
        with self.invest_box:
            ui.label("Investability").classes("text-sm ag-muted")
            for key, labelled in template_lab.INVESTABILITY_LABELS.items():
                if key == "max_position" and not template_lab.takes_max_position(v["region"]):
                    continue
                default = key == ("max_trade" if v["region"] in MAX_TRADE_REGIONS else "none")
                self.invest[key] = ui.checkbox(labelled, value=default, on_change=lambda _: self._invest_note())
        self._invest_note()
        self.show_variables()

    def _invest_note(self) -> None:
        region = self.market.values()["region"]
        chosen = [k for k, b in self.invest.items() if b.value]
        self.invest_warning.text = (f"In {region}, an Alpha without Max Trade is only submittable if it keeps 70% of "
                                    "its Sharpe under investability constraints.") \
            if region in MAX_TRADE_REGIONS and chosen and chosen != ["max_trade"] else ""

    # -- lint and variables
    async def _lint_tick(self) -> None:
        if not self.dirty_lint:
            return
        self.dirty_lint = False
        text = self.text.value or ""
        self.lint_box.clear()
        with self.lint_box:
            for item in template_lab.lint(text, self.table):
                where = f"line {item['line']}, col {item['column']}: " if item.get("line") else ""
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.icon("error" if item["level"] == "error" else "warning") \
                        .classes("ag-red" if item["level"] == "error" else "ag-amber")
                    ui.label(where + item["message"]).classes("text-sm")
        names = template_lab.names_in(text)
        if names != self.names:
            self.names = names
            self.show_variables()

    def show_variables(self) -> None:
        self.vars_box.clear()
        resolved = self._resolved()
        ds_options = {d["id"]: f"{d['id']} — {d['name']}" for d in self.datasets}
        with self.vars_box:
            if not self.names:
                ui.label("Write $name in the template where the search should choose.").classes("ag-muted text-sm")
            for name in self.names:
                d = resolved.get(name) or {"kind": "values", "values": ""}
                defined_here = name in self.given
                with ui.card().classes("ag-card p-3 w-full"):
                    with ui.row().classes("items-center gap-3 w-full"):
                        ui.label(f"${name}").classes("ag-mono font-semibold")
                        if not defined_here and resolved.get(name):
                            badge("default", "pending").tooltip("Named after a preset, dataset, subcategory or "
                                                                "category; change it to make it your own.")
                        kind = segmented({"fields": "Fields", "values": "Values"}, d["kind"],
                                         lambda v, n=name: self._switch(n, v))
                        kind.classes("ml-2")
                        if defined_here:
                            ui.button("Use default", on_click=lambda n=name: self._reset(n)).props("flat dense no-caps")
                    if d["kind"] == "values":
                        ui.input("Values, separated by commas", value=str(d.get("values") or ""),
                                 on_change=lambda e, n=name: self._update(n, {"kind": "values", "values": e.value})) \
                            .classes("w-full ag-mono")
                    else:
                        with ui.row().classes("gap-3 w-full items-end"):
                            ui.select(ds_options, multiple=True, value=[x for x in d.get("dataset_ids") or []
                                                                        if x in ds_options],
                                      label="Datasets", with_input=True,
                                      on_change=lambda e, n=name: self._update(n, {"dataset_ids": list(e.value or [])})) \
                                .props("use-chips").classes("grow min-w-[280px]")
                            ui.select(["MATRIX", "VECTOR"], multiple=True, value=d.get("field_types") or [],
                                      label="Field types (from where it is written)",
                                      on_change=lambda e, n=name: self._update(n, {"field_types": list(e.value or [])})) \
                                .props("use-chips").classes("w-64")
                            ui.select(self.vector_ops, multiple=True, value=d.get("vector_operators") or [],
                                      label="Vector operators",
                                      on_change=lambda e, n=name: self._update(
                                          n, {"vector_operators": list(e.value or [])})) \
                                .props("use-chips").classes("w-56")

    def _switch(self, name: str, kind: str) -> None:
        """Fields or values: a name switched to values starts from its preset, if it has one."""
        self._set(name, {"kind": "values", "values": template_lab.PRESETS.get(name, "")} if kind == "values"
                  else {"kind": "fields", "dataset_ids": []})

    def _set(self, name: str, definition: dict[str, Any]) -> None:
        self.given[name] = definition
        self.show_variables()

    def _update(self, name: str, change: dict[str, Any]) -> None:
        base = dict(self.given.get(name) or self._resolved().get(name) or {"kind": "fields"})
        base.update(change)
        self.given[name] = base

    def _reset(self, name: str) -> None:
        self.given.pop(name, None)
        self.show_variables()

    # -- preview
    async def preview(self) -> None:
        try:
            plan = await template_lab.plan(self.params())
        except Exception as exc:
            notify_error(exc, "Could not plan the task: ")
            return
        self.out.clear()
        with self.out:
            stats = plan.get("stats")
            with ui.row().classes("gap-3 w-full"):
                v, sub = stat_tile("Combinations")
                v.text = fmt_int(plan["combinations"]) if plan["combinations"] else "—"
                sub.text = "different simulations the space holds"
                if stats and not stats.get("problem"):
                    v, sub = stat_tile("Operators")
                    v.text = _span(stats["operators"]) + f" / {stats['limits'][0]}"
                    v, sub = stat_tile("Fields")
                    v.text = _span(stats["fields"]) + f" / {stats['limits'][1]}"
            if stats and not stats.get("problem"):
                facts(stats)
            show_problems(plan["problems"], plan["warnings"])
            if plan["variables"]:
                rows = [{"name": f"${i['name']}", "kind": i["kind"] or "—", "operator": "yes" if i["operator"] else "",
                         "fields": (f"{i['fields']['total']} ({i['fields']['matrix']} M / {i['fields']['vector']} V)"
                                    if i.get("fields") else ""), "types": ", ".join(i["field_types"]),
                         "values": i["values"] or "", "problems": "; ".join(i["problems"])} for i in plan["variables"]]
                cols = [{"field": "name", "headerName": "Variable", "width": 140},
                        {"field": "kind", "headerName": "Kind", "width": 90},
                        {"field": "operator", "headerName": "Operator", "width": 95},
                        {"field": "fields", "headerName": "Fields", "width": 150},
                        {"field": "types", "headerName": "Field types", "width": 140},
                        {"field": "values", "headerName": "Values", "width": 85},
                        {"field": "problems", "headerName": "Problems", "flex": 1, "minWidth": 280,
                         "tooltipField": "problems"}]
                ui.aggrid(grid_options(cols, rows, pagination=False, domLayout="autoHeight"), theme="quartz") \
                    .classes("w-full")
            show_samples(plan["sample"])

    # -- saved templates and the gallery
    async def show_gallery(self) -> None:
        self.gallery.clear()
        saved = await templates_gallery.saved()
        v = self.market.values()
        items = [{"text": f.text, "variables": {}} for f in templates_gallery.GALLERY] + \
                [{"text": s["text"], "variables": s["variables"]} for s in saved]
        stats = await template_lab.stats(items, v["region"], v["delay"])
        with self.gallery:
            ui.label("Built-in").classes("ag-section-title")
            with ui.row().classes("w-full gap-2"):
                for i, f in enumerate(templates_gallery.GALLERY):
                    with ui.card().classes("ag-card p-3 cursor-pointer w-[420px]").on(
                            "click", lambda _, f=f: self.family_dialog(f)):
                        if f.title:
                            ui.label(f.title).classes("font-medium")
                        ui.label(f.shown).classes("ag-mono text-sm break-all")
                        facts(stats[i])
                        ui.label(f"{len(f.variations)} variation{'s' if len(f.variations) != 1 else ''} ›") \
                            .classes("ag-muted text-xs")
            ui.label("Saved").classes("ag-section-title mt-2")
            if not saved:
                ui.label("Nothing saved yet. Save As keeps the template you are writing here.").classes("ag-muted text-sm")
            with ui.row().classes("w-full gap-2"):
                for s, st in zip(saved, stats[len(templates_gallery.GALLERY):]):
                    with ui.card().classes("ag-card p-3 cursor-pointer w-[320px]").on(
                            "click", lambda _, s=s: self.open_saved(s)):
                        ui.label(s["name"]).classes("font-medium")
                        if s.get("description"):
                            ui.label(s["description"]).classes("ag-muted text-xs")
                        ui.label(s["text"][:240]).classes("ag-mono text-xs break-all")
                        facts(st)

    def family_dialog(self, family: templates_gallery.Family) -> None:
        with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[760px] max-w-full"):
            ui.label(family.title or "Built-in template").classes("text-lg font-semibold")
            for text in [family.text, *family.variations]:
                with ui.row().classes("items-start justify-between w-full no-wrap gap-3"):
                    ui.label(text).classes("ag-mono text-sm break-all whitespace-pre-wrap")
                    ui.button("Open", on_click=lambda t=text: (d.close(), self.open_text(t))).props("outline dense no-caps")
            ui.button("Close", on_click=d.close).props("flat no-caps")
        d.open()

    def open_text(self, text: str) -> None:
        self.saved_id = None
        self.given = {}
        self.tname.value, self.desc.value = "", ""
        self.text.value = text
        self.dirty_lint = True

    def open_saved(self, s: dict[str, Any]) -> None:
        self.saved_id = s["id"]
        self.given = dict(s.get("variables") or {})
        self.tname.value, self.desc.value = s["name"], s.get("description") or ""
        self.text.value = s["text"]
        self.names = []
        self.dirty_lint = True

    def new(self) -> None:
        self.open_text("")

    async def save(self, as_new: bool = False) -> None:
        try:
            if self.saved_id is not None and not as_new:
                row = await templates_gallery.update(self.saved_id, self.tname.value or "", self.text.value or "",
                                                     self._resolved(), self.desc.value)
            else:
                row = await templates_gallery.create(self.tname.value or "", self.text.value or "",
                                                     self._resolved(), self.desc.value)
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        self.saved_id = row["id"]
        ui.notify(f"Saved {row['name']}.", type="positive")
        await self.show_gallery()

    async def delete(self) -> None:
        if self.saved_id is None:
            ui.notify("Open a saved template to delete it.", type="warning")
            return
        try:
            await templates_gallery.delete(self.saved_id)
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        self.saved_id = None
        ui.notify("Template deleted.", type="positive")
        await self.show_gallery()


def _span(pair: list[int]) -> str:
    a, b = (pair + [0, 0])[:2]
    return f"{a}" if a == b else f"{a}–{b}"


def facts(stats: dict[str, Any] | None) -> None:
    """Power Pool budgets and the kinds of Alpha a template makes."""
    if not stats or stats.get("problem"):
        if stats and stats.get("problem"):
            ui.label(stats["problem"]).classes("ag-amber text-xs")
        return
    with ui.row().classes("gap-2 items-center flex-wrap"):
        ops_over = stats["operators"][1] > stats["limits"][0]
        fields_over = stats["fields"][1] > stats["limits"][1]
        badge(f"Operators {_span(stats['operators'])}/{stats['limits'][0]}", "warning" if ops_over else "")
        badge(f"Fields {_span(stats['fields'])}/{stats['limits'][1]}", "warning" if fields_over else "")
        if stats.get("power_pool") and stats.get("holes") == 0:
            badge("Power Pool", "pass").tooltip("Every Alpha it makes is within Power Pool's operators and fields; "
                                                "ts_backfill and grouping fields are not counted")
        if stats.get("single_dataset"):
            badge("Single Dataset", "pass")
        if stats.get("pyramids") is not None and stats["pyramids"] > 2:
            badge(f"Up to {stats['pyramids']} Dataset Categories", "warning").tooltip(
                "An Alpha reading more than 2 Dataset Categories counts toward no Genius pyramid.")
        if stats.get("holes"):
            badge(f"{stats['holes']} Signal{'s' if stats['holes'] != 1 else ''} to Write", "pending")
        if stats.get("missing"):
            badge(f"Needs {', '.join(stats['missing'])}", "fail")


async def template_tab() -> None:
    editor = TemplateEditor(await launch.account_operators())
    await editor.build()


# ----------------------------------------------------------------- Evolution

SEED_COLUMNS = [
    {"field": "alpha_id", "headerName": "Alpha", "width": 120},
    num_col("score", "Seed score", 3, 110),
    num_col("sharpe", "Sharpe", 2, 90), num_col("fitness", "Fitness", 2, 90),
    num_col("train_fitness", "Train Fitness", 2, 115), num_col("turnover", "Turnover", 1, 100, pct=True),
    num_col("test_sharpe", "Test Sharpe", 2, 105),
    {"field": "expression", "headerName": "Expression", "flex": 1, "minWidth": 320, "tooltipField": "expression"},
]


async def evolution_tab() -> None:
    info_box("Evolution Lab", "Breed new Alphas from seed Alphas you already have. Children mutate operators, "
             "lookbacks, fields within a dataset, groups, neutralization and decay, and cross gene by gene when two "
             "parents share a shape. Every child holds the last two years out and is scored on Train Fitness; "
             "parents are kept apart by the correlation of their daily PnL. A task stops early after 1,000 "
             "children without a new best.")
    markets = await ga.evolvable_markets()
    state: dict[str, Any] = {"job": None}
    with card("w-full"):
        if not markets:
            empty_state("science", "No market holds unsubmitted Alphas yet",
                        "Sync your Alphas from BRAIN, or simulate some first: Evolution breeds from stored Alphas.")
            return
        options = {f"{m['region']}|{m['delay']}|{m['universe']}":
                   f"{m['region']} · D{m['delay']} · {m['universe']} — {fmt_int(m['alphas'])} Alphas" for m in markets}
        market = ui.select(options, value=next(iter(options)), label="Market").classes("w-96")
        with ui.row().classes("items-end gap-3"):
            count = ui.number("Auto Select: seeds", value=20, min=2, max=ga.MAX_SEEDS, format="%d").classes("w-40")
            auto_btn = ui.button("Auto Select", icon="auto_awesome").props("outline no-caps")
            auto_status = ui.label("").classes("ag-muted text-sm")
        ui.label("Seeds: tick at least two (the best by seed score first). Auto Select picks the best mutually "
                 "uncorrelated ones, downloading daily PnL where missing; it never simulates.").classes("ag-muted text-xs")
        grid = ui.aggrid(grid_options(SEED_COLUMNS, [], multi=True), theme="quartz").classes("w-full h-[380px]")
        neut = NeutralizationPicker(["INDUSTRY", "SUBINDUSTRY"], legend="Neutralizations a gene may take")
        with ui.row().classes("gap-4 items-end"):
            with ui.column().classes("gap-0"):
                ui.label("Population").classes("text-sm ag-muted")
                population = segmented({0: "Auto", **{p: str(p) for p in ga.POPULATIONS}}, 0)
            with ui.column().classes("gap-0"):
                ui.label("Mutation rate").classes("text-sm ag-muted")
                rate = segmented({r: f"{r:.2f}" for r in ga.MUTATION_RATES}, ga.MUTATION_RATE)
            keep = ui.checkbox("Keep a parent's decay (mutation never changes it)")
        controls = TaskControls(1000)
        out = ui.column().classes("w-full gap-1")

        def parts() -> tuple[str, int, str]:
            region, delay, universe = str(market.value).split("|")
            return region, int(delay), universe

        async def load_candidates(select_ids: list[str] | None = None) -> None:
            region, delay, universe = parts()
            rows = [ga.seed_view(r) for r in await ga.candidates(region, delay, universe)]
            grid.options["rowData"] = rows
            grid.update()
            neut.show(region)
            if select_ids:
                wanted = json.dumps(select_ids)
                grid.run_grid_method("forEachNode",
                                     f"(node) => node.setSelected({wanted}.includes(node.data.alpha_id))")

        async def chosen() -> list[str]:
            return [r["alpha_id"] for r in await grid.get_selected_rows()]

        async def params() -> dict[str, Any]:
            region, delay, universe = parts()
            return {"region": region, "delay": delay, "universe": universe, "alpha_ids": await chosen(),
                    "neutralizations": list(neut.chosen), "population": int(population.value) or None,
                    "mutation_rate": float(rate.value), "keep_decay": bool(keep.value)}

        async def preview() -> None:
            body = await params()
            try:
                plan = await ga.plan({**body, "simulations": controls.values()[1]})
            except Exception as exc:
                notify_error(exc, "Could not plan the task: ")
                return
            out.clear()
            with out:
                with ui.row().classes("gap-3"):
                    for label, value in (("Seeds", fmt_int(len(plan["seeds"]))),
                                         ("Population", fmt_int(plan["population"])),
                                         ("Generations", fmt_int(plan["generations"]))):
                        v, _ = stat_tile(label)
                        v.text = value
                for s in plan["skipped"]:
                    ui.label(f"{s['alpha_id']}: {s['reason']}").classes("ag-muted text-sm")
                show_problems(plan["problems"], [])
                show_samples(plan["sample"])

        async def add(run: bool) -> None:
            await add_task("evolution", await params(), controls, run)

        async def auto() -> None:
            from ...jobs.queue import jobs
            region, delay, universe = parts()
            try:
                state["job"] = await jobs.submit("evolution_seeds", f"Auto Select seeds · {region} D{delay} {universe}",
                                                 {"region": region, "delay": delay, "universe": universe,
                                                  "count": int(count.value or 20)})
            except Exception as exc:
                notify_error(exc, "Could not start Auto Select: ")
                return
            auto_status.text = "Choosing seeds…"

        async def poll() -> None:
            if state["job"] is None:
                return
            from ...db.models import Job
            async with db.session() as s:
                job = await s.get(Job, state["job"])
            if job is None:
                return
            auto_status.text = job.message or job.status
            if job.status in ("done", "failed", "cancelled"):
                state["job"] = None
                if job.status == "done":
                    region, delay, universe = parts()
                    found = await db.kv_get(ga.AUTO_KEY.format(region=region, delay=delay, universe=universe)) or {}
                    ids = [s["alpha_id"] for s in found.get("seeds") or []]
                    await load_candidates(ids)
                    reasons = found.get("reasons") or []
                    auto_status.text = (f"Chose {len(ids)} of {found.get('wanted')} seeds "
                                        f"({found.get('examined')} examined)."
                                        + (f" Not chosen: {reasons[0]['alpha_id']} — {reasons[0]['reason']}"
                                           if reasons else ""))
                else:
                    auto_status.text = f"Auto Select {job.status}: {job.error or ''}"

        auto_btn.on("click", auto)
        ui.timer(1.5, poll)
        market.on_value_change(lambda _: load_candidates())
        controls.buttons(preview, add)
    await load_candidates()


# ----------------------------------------------------------------- Tasks

STATUS_RENDER = """params => {
  const k = {RUNNING: 'running', COMPLETE: 'pass', PAUSED: 'pending', QUEUED: 'pending', IDLE: '',
             FAILED: 'fail'}[params.data.status] || '';
  return '<span class="ag-badge ' + k + '">' + params.value + '</span>';
}"""
TASK_COLUMNS = [
    {"field": "id", "headerName": "#", "width": 70},
    {"field": "label", "headerName": "Task", "flex": 2, "minWidth": 220, "tooltipField": "message"},
    {"field": "lab_name", "headerName": "Lab", "width": 140},
    {"field": "status_label", "headerName": "Status", "width": 150, ":cellRenderer": STATUS_RENDER,
     "tooltipField": "message"},
    {"field": "cores", "headerName": "Cores", "width": 80, "type": "numericColumn"},
    {"field": "progress", "headerName": "Simulated / target", "width": 160},
    {"field": "cached", "headerName": "Cached (free)", "width": 115, "type": "numericColumn"},
    {"field": "queued", "headerName": "Queued", "width": 90, "type": "numericColumn"},
    {"field": "running", "headerName": "Running", "width": 95, "type": "numericColumn"},
    {"field": "failed", "headerName": "Failed", "width": 85, "type": "numericColumn"},
    {"field": "best_text", "headerName": "Best", "width": 150},
    {"field": "waiting", "headerName": "Waiting", "width": 95},
    {"field": "running_for", "headerName": "Running for", "width": 110},
    {"field": "reasons", "headerName": "Top failure reasons", "flex": 1, "minWidth": 240, "tooltipField": "reasons"},
    {"field": "market", "headerName": "Market", "width": 160},
    {"field": "created", "headerName": "Created", "width": 140},
]


async def task_view_rows() -> list[dict[str, Any]]:
    rows = await results.task_rows()
    for r in rows:
        r["progress"] = f"{fmt_int(r['simulated'])} / {fmt_int(r['target'])}"
        r["best_text"] = f"{fmt_num(r['best'])} {r['objective']}" if r["best"] is not None else "—"
        r["waiting"] = fmt_duration(r["waiting_s"]) if r["waiting_s"] else "—"
        r["running_for"] = fmt_duration(r["running_s"]) if r["running_s"] else "—"
        r["market"] = " · ".join(str(x) for x in (r["region"], f"D{r['delay']}" if r["delay"] is not None else None,
                                                  r["universe"]) if x)
    return rows


async def tasks_tab(open_task: int | None = None) -> None:
    selected: dict[str, Any] = {"id": None, "row": None}
    with card("w-full"):
        with ui.row().classes("items-center gap-2 w-full flex-wrap"):
            ui.button("Run", icon="play_arrow", on_click=lambda: act("run")).props("unelevated no-caps dense")
            ui.button("Pause", icon="pause", on_click=lambda: act("pause")).props("outline no-caps dense")
            ui.button("Stop", icon="stop", on_click=lambda: act("stop")).props("outline no-caps dense").tooltip(
                "First press: send nothing more and score what is out. Pressed again on a stopping task: cancel on "
                "BRAIN and finish now.")
            ui.button("Rename", icon="edit", on_click=lambda: rename()).props("flat no-caps dense")
            ui.button("Resize", icon="tune", on_click=lambda: resize()).props("flat no-caps dense")
            ui.button("Results", icon="leaderboard", on_click=lambda: open_results()).props("flat no-caps dense")
            ui.button("Remove", icon="delete", on_click=lambda: remove()).props("flat no-caps dense")
            ui.space()
            lend = ui.checkbox("Lend idle cores").tooltip(
                "A task that has sent everything holds no cores, so a waiting task starts beside its last batches, "
                "and free slots are lent to the tasks still sending.")
            ui.button("Run All", icon="playlist_play", on_click=lambda: run_all()).props("outline no-caps dense")
        picked = ui.label("Click a task to act on it.").classes("ag-muted text-sm")
        grid = ui.aggrid(grid_options(TASK_COLUMNS, await task_view_rows(), getRowId=None), theme="quartz") \
            .classes("w-full h-[520px]")
        pref = await db.kv_get(runner_mod.PREF_KEY) or {}
        lend.value = bool(pref.get("lend_idle_cores"))

        async def lend_changed(e) -> None:  # noqa: ANN001
            current = await db.kv_get(runner_mod.PREF_KEY) or {}
            await db.kv_set(runner_mod.PREF_KEY, {**current, "lend_idle_cores": bool(e.value)})
            await runner().start_waiting()

        lend.on_value_change(lend_changed)

    def on_click(e) -> None:  # noqa: ANN001
        row = (e.args or {}).get("data") or {}
        if row.get("id"):
            selected["id"], selected["row"] = row["id"], row
            picked.text = f"Task #{row['id']} · {row['label']} · {row['status_label']}" + \
                (f" — {row['message']}" if row.get("message") else "")

    grid.on("cellClicked", on_click)
    grid.on("cellDoubleClicked", lambda e: (on_click(e), open_results()))

    async def refresh() -> None:
        grid.options["rowData"] = await task_view_rows()
        grid.update()

    def need() -> int | None:
        if selected["id"] is None:
            ui.notify("Click a task first.", type="warning")
        return selected["id"]

    async def act(what: str) -> None:
        tid = need()
        if tid is None:
            return
        try:
            if what == "run":
                await runner().run(tid)
            elif what == "pause":
                await runner().pause(tid)
            else:
                said = await runner().stop(tid)
                ui.notify({"stopping": "Stopping: what is out will be scored. Press Stop again to cancel it on BRAIN.",
                           "stopped": "Stopped."}.get(said, "Nothing to stop."))
        except Exception as exc:
            notify_error(exc)
        await refresh()

    async def run_all() -> None:
        n = await runner().run_all()
        ui.notify(f"{n} task{'s' if n != 1 else ''} started or waiting for cores.")
        await refresh()

    async def rename() -> None:
        tid = need()
        if tid is None:
            return
        with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[420px]"):
            name = ui.input("Name (blank clears it)", value=selected["row"].get("name") or "").classes("w-full")

            async def save() -> None:
                await runner().rename(tid, name.value)
                d.close()
                await refresh()

            with ui.row():
                ui.button("Save", on_click=save).props("unelevated no-caps")
                ui.button("Cancel", on_click=d.close).props("flat no-caps")
        d.open()

    async def resize() -> None:
        tid = need()
        if tid is None:
            return
        row = selected["row"]
        with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[460px]"):
            ui.label("A running task takes its new cores and simulations from its next round.").classes("ag-muted text-sm")
            cores = segmented(CORES, int(row.get("cores") or 1))
            sims = ui.number("Simulations", value=row.get("target"), min=1, format="%d").classes("w-40")

            async def save() -> None:
                try:
                    await runner().resize(tid, cores=int(cores.value), max_trials=int(sims.value or 1))
                except ValueError as exc:
                    ui.notify(str(exc), type="warning")
                    return
                d.close()
                await refresh()

            with ui.row():
                ui.button("Save", on_click=save).props("unelevated no-caps")
                ui.button("Cancel", on_click=d.close).props("flat no-caps")
        d.open()

    async def remove() -> None:
        tid = need()
        if tid is None:
            return
        try:
            await runner().delete(tid)
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        selected["id"] = None
        picked.text = "Removed. The Alphas it found stay in Alphas."
        await refresh()

    async def open_results() -> None:
        tid = need()
        if tid is not None:
            await results_dialog(tid, selected["row"])

    ui.timer(5.0, refresh)
    if open_task:
        row = next((r for r in grid.options["rowData"] if r["id"] == int(open_task)), None)
        if row is None:
            ui.notify(f"Task #{open_task} no longer exists.", type="warning")
        else:
            selected["id"], selected["row"] = row["id"], row
            picked.text = f"Task #{row['id']} · {row['label']} · {row['status_label']}"
            await results_dialog(row["id"], row)


RESULT_COLUMNS = [
    {"field": "alpha_id", "headerName": "Alpha", "width": 115},
    num_col("value", "Score", 2, 85),
    num_col("sharpe", "Sharpe", 2, 85), num_col("fitness", "Fitness", 2, 85),
    num_col("turnover", "Turnover", 1, 95, pct=True), num_col("returns", "Returns", 1, 90, pct=True),
    num_col("test_sharpe", "Test Sharpe", 2, 100), num_col("after_cost_5bps", "After-cost Sharpe", 2, 130),
    num_col("prod_corr", "Prod corr", 2, 95),
    {"field": "failed_checks", "headerName": "Checks failed", "width": 220, "tooltipField": "failed_checks"},
    {"field": "region", "headerName": "Region", "width": 85}, {"field": "universe", "headerName": "Universe",
                                                                "width": 105},
    {"field": "neutralization", "headerName": "Neutralization", "width": 135},
    {"field": "investability", "headerName": "Investability", "width": 120},
    {"field": "expression", "headerName": "Expression", "flex": 1, "minWidth": 300, "tooltipField": "expression"},
]


async def results_dialog(task_id: int, row: dict[str, Any] | None) -> None:
    found = await results.results(task_id)
    rows = found["rows"]
    with ui.dialog().props("maximized") as d, ui.card().classes("ag-card p-5 w-full"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(f"{(row or {}).get('label') or f'Task {task_id}'} — results").classes("text-lg font-semibold")
            ui.button("Close", icon="close", on_click=d.close).props("flat no-caps")
        green = sum(1 for r in rows if r["submittable"] or r["pending"])
        pending = sum(1 for r in rows if r["pending"])
        with ui.row().classes("gap-3 w-full"):
            for label, value, color in (("Alphas", fmt_int(len(rows)), ""),
                                        ("Submittable", f"{'~' if pending else ''}{fmt_int(green)}", "ag-green"),
                                        ("Refused", fmt_int(len(rows) - green), "ag-red" if len(rows) - green else ""),
                                        ("Regions", fmt_int(len(found["regions"])), "")):
                v, _ = stat_tile(label, color)
                v.text = value
        if not rows:
            empty_state("hourglass_empty", "No Alphas back yet.")
        else:
            ui.label("Across Regions").classes("ag-section-title mt-2")
            ui.label("Where the Alphas are worth submitting, and the best each metric reaches there. Refused Alphas "
                     "are counted, never counted as a best.").classes("ag-muted text-xs")
            region_cols = [{"field": "region", "headerName": "Region", "width": 100},
                           {"field": "alphas", "headerName": "Alphas", "width": 90, "type": "numericColumn"},
                           {"field": "submittable", "headerName": "Submittable", "width": 115, "type": "numericColumn"},
                           {"field": "refused", "headerName": "Refused", "width": 95, "type": "numericColumn"},
                           num_col("best_sharpe", "Best Sharpe", 2, 110), num_col("best_fitness", "Best Fitness", 2, 115),
                           num_col("best_turnover", "Lowest Turnover", 1, 135, pct=True),
                           num_col("best_test_sharpe", "Best Test Sharpe", 2, 135),
                           num_col("best_after_cost_5bps", "Best After-cost", 2, 125)]
            ui.aggrid(grid_options(region_cols, found["regions"], pagination=False, domLayout="autoHeight"),
                      theme="quartz").classes("w-full")
            ui.label("Checks Failed Together").classes("ag-section-title mt-2")
            ui.label("Every distinct set of refusing checks, and how many Alphas hit exactly that set: the ones a "
                     "single problem is holding back show up with one check.").classes("ag-muted text-xs")
            set_cols = [{"field": "checks", "headerName": "Checks failed", "flex": 1, "minWidth": 260},
                        {"field": "count", "headerName": "Checks", "width": 90, "type": "numericColumn"},
                        {"field": "alphas", "headerName": "Alphas", "width": 90, "type": "numericColumn"},
                        num_col("share", "Share", 0, 90, pct=True)]
            ui.aggrid(grid_options(set_cols, found["check_sets"], pagination=False, domLayout="autoHeight"),
                      theme="quartz").classes("w-full")
            ui.label("Every Alpha, best first (double-click to open)").classes("ag-section-title mt-2")
            grid = ui.aggrid(grid_options(RESULT_COLUMNS, rows), theme="quartz").classes("w-full h-[520px]")
            ids = [r["alpha_id"] for r in rows if r.get("alpha_id")]

            async def open_row(e) -> None:  # noqa: ANN001
                aid = ((e.args or {}).get("data") or {}).get("alpha_id")
                if aid:
                    await open_alpha(aid, ids)

            grid.on("cellDoubleClicked", open_row)
    d.open()


# ----------------------------------------------------------------- Submittable

SUB_COLUMNS = [
    {"field": "alpha_id", "headerName": "Alpha", "width": 115}, {"field": "task_name", "headerName": "Task",
                                                                  "width": 170},
    {"field": "region", "headerName": "Region", "width": 85}, {"field": "delay", "headerName": "Delay", "width": 75},
    {"field": "universe", "headerName": "Universe", "width": 105},
    {"field": "neutralization", "headerName": "Neutralization", "width": 130},
    {"field": "investability", "headerName": "Investability", "width": 115},
    num_col("pp_corr", "Power Pool corr", 2, 125), num_col("prod_corr", "Prod corr", 2, 95),
    num_col("sharpe", "Sharpe", 2, 85), num_col("fitness", "Fitness", 2, 85),
    num_col("turnover", "Turnover", 1, 95, pct=True), num_col("returns", "Returns", 1, 90, pct=True),
    num_col("drawdown", "Drawdown", 1, 100, pct=True), num_col("margin", "Margin", 5, 95),
    num_col("after_cost_5bps", "After-cost Sharpe", 2, 130), num_col("test_sharpe", "Test Sharpe", 2, 100),
    num_col("test_fitness", "Test Fitness", 2, 105),
    {"field": "expression", "headerName": "Expression", "flex": 1, "minWidth": 300, "tooltipField": "expression"},
]


async def submittable_tab() -> None:
    state: dict[str, Any] = {"all": await results.submittable()}
    with card("w-full"):
        ui.label("Every Alpha from every lab task that BRAIN would take: its checks pass (Quick Alphas wait for "
                 "their Full run), each listed once, from the task that found it first.").classes("ag-muted text-sm")
        with ui.row().classes("gap-3 items-end w-full flex-wrap"):
            search = ui.input("Search (Alpha id or expression)").classes("w-72")
            picks = {k: ui.select([], multiple=True, label=k.capitalize()).props("use-chips dense").classes("w-40")
                     for k in results.PICKS}
            implausible = ui.checkbox("Show implausible (|Sharpe| > 10)")
            ui.button("Refresh", icon="refresh", on_click=lambda: reload()).props("flat no-caps dense")
        bounds: dict[str, tuple[ui.number, ui.number]] = {}
        with ui.expansion("Bounds", icon="filter_alt").classes("w-full"):
            with ui.row().classes("gap-4 flex-wrap"):
                for key, label, _ in results.BOUNDS:
                    with ui.column().classes("gap-0"):
                        ui.label(label).classes("text-xs ag-muted")
                        with ui.row().classes("gap-1 no-wrap"):
                            lo = ui.number(placeholder="Min").props("dense").classes("w-20")
                            hi = ui.number(placeholder="Max").props("dense").classes("w-20")
                        bounds[key] = (lo, hi)
        count = ui.label("").classes("ag-muted text-sm")
        grid = ui.aggrid(grid_options(SUB_COLUMNS, []), theme="quartz").classes("w-full h-[560px]")

    def offered() -> None:
        for k, sel in picks.items():
            sel.options = sorted({str(r.get(k)) for r in state["all"] if r.get(k) is not None})
            sel.update()

    def apply() -> None:
        rows = results.filter_submittable(
            state["all"], search=search.value or "", picked={k: list(s.value or []) for k, s in picks.items()},
            bounds={k: (lo.value, hi.value) for k, (lo, hi) in bounds.items()}, implausible=bool(implausible.value))
        grid.options["rowData"] = rows
        grid.update()
        unmeasured = [f"{sum(1 for r in state['all'] if r.get(k) is None)} without {label} yet"
                      for k, label, _ in results.BOUNDS if (bounds[k][0].value is not None or
                                                            bounds[k][1].value is not None)
                      and any(r.get(k) is None for r in state["all"])]
        count.text = f"{fmt_int(len(rows))} of {fmt_int(len(state['all']))} submittable Alphas" + \
            (f" · {'; '.join(unmeasured)}" if unmeasured else "")

    async def reload() -> None:
        state["all"] = await results.submittable()
        offered()
        apply()

    for el in (search, implausible, *picks.values(), *(x for pair in bounds.values() for x in pair)):
        el.on_value_change(lambda _: apply())

    async def open_row(e) -> None:  # noqa: ANN001
        aid = ((e.args or {}).get("data") or {}).get("alpha_id")
        if aid:
            await open_alpha(aid, [r["alpha_id"] for r in grid.options["rowData"]])

    grid.on("cellDoubleClicked", open_row)
    offered()
    apply()


# ----------------------------------------------------------------- the page

async def _other_lab(function: str, title: str) -> None:
    """The Power Pool and Super Alpha tabs come from labs_llm; until that module is there, say so."""
    try:
        from . import labs_llm
        tab = getattr(labs_llm, function)
    except (ImportError, AttributeError):
        info_box(title, f"The {title} is loading: its screen is not installed yet. Restart the app once it is.",
                 icon="hourglass_top")
        return
    await tab()


async def page(task: int | None = None) -> None:
    """/labs, or /labs?task=<id> to open the Tasks tab on that task's results."""
    page_header("Research Labs", "Search, Template, Evolution, Power Pool and Super Alpha labs: tasks that search for "
                "alphas on their share of the simulation cores.")
    with ui.tabs().classes("w-full") as tabs:
        tab = {name: ui.tab(name) for name in TABS}
    builders: dict[str, Callable[[], Awaitable[None]]] = {
        "Search": search_tab, "Template": template_tab, "Evolution": evolution_tab,
        "Power Pool": lambda: _other_lab("power_pool_tab", "LLM Power Pool Lab"),
        "Super Alpha": lambda: _other_lab("super_tab", "Super Alpha Lab"),
        "Tasks": lambda: tasks_tab(task), "Submittable": submittable_tab}
    built: set[str] = set()
    panels_by: dict[str, Any] = {}
    first = "Tasks" if task else "Search"
    with ui.tab_panels(tabs, value=tab[first]).classes("w-full bg-transparent") as panels:
        for name in TABS:
            with ui.tab_panel(tab[name]).classes("p-0 pt-3") as p:
                panels_by[name] = ui.column().classes("w-full gap-3")
            del p

    async def build(name: str) -> None:
        if name in built:
            return
        built.add(name)
        with panels_by[name]:
            try:
                await builders[name]()
            except Exception as exc:
                notify_error(exc, f"The {name} tab could not be built: ")

    async def changed(e) -> None:  # noqa: ANN001
        value = e.value if isinstance(e.value, str) else getattr(e.value, "_props", {}).get("name", "")
        if value in builders:
            await build(value)

    tabs.on_value_change(changed)
    panels.on_value_change(changed)
    await build(first)
