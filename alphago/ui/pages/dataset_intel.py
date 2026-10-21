"""Dataset Intelligence — AI-rated datasets (single field / double / interaction), top picks, briefs, fields, ideas."""

from __future__ import annotations

from collections import Counter

from nicegui import ui
from sqlalchemy import func, select

from ...brain import brain
from ...config import settings
from ...db import db
from ...db.models import Batch, DataField, Dataset
from ...engine.datasets import USAGES, analyze_dataset, get_insight, probe_field, rated_datasets
from ...engine.generators import llm_ideas, template_ideas
from ...engine.simulator import simulator
from ...events import bus
from ...jobs.queue import jobs
from ...llm import LLMError
from ..components import badge, bar_col, card, fmt_int, grid_options, notify_error, num_col, page_header, segmented, \
    stat_tile

REC_RENDER = """params => {
  if (!params.value) return '';
  const k = params.data.recommended === 'single' ? '' : 'pending';
  return '<span class="ag-badge ' + k + '">' + params.value + '</span>';
}"""
DS_RENDER = """params => '<span class="ag-num">' + params.data.id + '</span> <span style="color:#8b98ad">· '
  + (params.data.name || '') + '</span>'"""

DS_COLUMNS = [
    {"field": "rank", "headerName": "#", "width": 65},
    {"field": "id", "headerName": "Dataset", "flex": 2, "minWidth": 300, ":cellRenderer": DS_RENDER,
     "tooltipField": "reason"},
    {"field": "theme", "headerName": "Category", "width": 140},
    {**num_col("ai_score", "AI Score", 0, 105), "headerTooltip": "AI rating blended with your own hit rate once you "
                                                                "have 20+ simulations on the dataset."},
    bar_col("single", "Single"),
    bar_col("double", "Double"),
    bar_col("interaction", "Interaction"),
    {"field": "recommended_label", "headerName": "Recommended", "width": 135, ":cellRenderer": REC_RENDER},
    {"field": "field_count", "headerName": "Fields", "width": 90, "type": "numericColumn"},
    {"field": "tried", "headerName": "Tested", "width": 95, "type": "numericColumn",
     "headerTooltip": "Your first-pass simulations on this dataset (repairs and robustness tests not counted)"},
    {"field": "passed", "headerName": "Passed", "width": 95, "type": "numericColumn",
     "headerTooltip": "Ideas from this dataset that passed BRAIN's submission check (directly or after repair)"},
    {**num_col("pyramid_multiplier", "Pyramid ×", 1, 105)},
    {**num_col("opportunity", "Opportunity", 0, 115), "headerTooltip": "v1 score: pyramid multiplier, value score, "
                                                                      "low crowding, coverage, your hit rate"},
]
FIELD_COLUMNS = [
    {"field": "id", "headerName": "Field", "width": 260, "cellClass": "ag-num"},
    {"field": "type", "headerName": "Type", "width": 95},
    num_col("coverage", "Coverage", 0, 105, pct=True),
    {"field": "alpha_count", "headerName": "Alphas", "width": 95, "type": "numericColumn"},
    {"field": "user_count", "headerName": "Users", "width": 85, "type": "numericColumn"},
    {"field": "description", "headerName": "Description", "flex": 1, "minWidth": 300},
]


async def ideas_dialog(specs, title: str, scope: dict) -> None:  # noqa: ANN001
    """Review generated ideas, then simulate the selected ones (assisted flow)."""
    rows = [{"i": i, "expression": s.expression, "hypothesis": s.hypothesis or "",
             "neutralization": s.settings.get("neutralization"), "decay": s.settings.get("decay")}
            for i, s in enumerate(specs)]
    with ui.dialog() as dialog, ui.card().classes("ag-card p-5 w-[1200px] max-w-[96vw]"):
        ui.label(title).classes("text-lg font-semibold")
        grid = ui.aggrid(grid_options([
            {"field": "expression", "headerName": "Expression", "flex": 2, "minWidth": 380, "cellClass": "ag-num"},
            {"field": "hypothesis", "headerName": "Hypothesis", "flex": 1, "minWidth": 260},
            {"field": "neutralization", "headerName": "Neut.", "width": 130},
            {"field": "decay", "headerName": "Decay", "width": 80}], rows, multi=True), theme="quartz").classes(
            "w-full h-[420px]")

        async def simulate() -> None:
            sel = {r["i"] for r in await grid.get_selected_rows()}
            chosen = [s for i, s in enumerate(specs) if i in sel]
            if not chosen:
                ui.notify("Select ideas first.", type="warning")
                return
            async with db.write() as s:
                b = Batch(source="manual", region=scope["region"], delay=scope["delay"], universe=scope["universe"],
                          dataset_id=chosen[0].dataset_id, strategy="dataset-intel", status="running",
                          n_generated=len(specs))
                s.add(b)
                await s.flush()
                bid = b.id
            res = await simulator.submit(chosen, batch_id=bid)
            bus.publish("simulate", f"{len(res.created)} simulations queued from Dataset Intelligence "
                                    f"(batch #{bid}); {len(res.reused)} reused.")
            ui.notify(f"Queued {len(res.created)} simulations.", type="positive")
            dialog.close()

        with ui.row().classes("justify-end w-full"):
            ui.button("Close", on_click=dialog.close).props("flat")
            ui.button("Simulate selected", icon="play_arrow", on_click=simulate).props("color=positive")
    dialog.open()
    await ui.run_javascript("await new Promise(r => setTimeout(r, 300));")
    grid.run_grid_method("selectAll")


async def page() -> None:
    opts = brain.options
    r = settings.research
    with ui.row().classes("w-full items-start no-wrap"):
        page_header("Dataset Intelligence", "Where to look next: every dataset rated by AI for single-field alphas, "
                                            "pairing with another dataset, and field interactions — blended with your "
                                            "own results.")
        ui.space()
        with ui.row().classes("items-end gap-2 no-wrap"):
            region = ui.select(opts.regions, value=r.region if r.region in opts.regions else opts.regions[0],
                               label="Region").classes("w-28")
            delay = ui.select(opts.delays(region.value), value=r.delay if r.delay in opts.delays(region.value)
                              else opts.delays(region.value)[0], label="Delay").classes("w-24")
            universe = ui.select(opts.universes(region.value), label="Universe",
                                 value=r.universe if r.universe in opts.universes(region.value)
                                 else opts.universes(region.value)[0]).classes("w-36")

    with ui.row().classes("w-full gap-4 no-wrap"):
        t_ds, t_ds_sub = stat_tile("Datasets analysed", icon="storage")
        t_fl, _ = stat_tile("Fields analysed", icon="storage")
        t_single, _ = stat_tile("Best for Single")
        t_double, _ = stat_tile("Best for Double")
        t_inter, _ = stat_tile("Best for Interaction")

    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        with card("w-[62%]"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("emoji_events").classes("ag-amber")
                with ui.column().classes("gap-0"):
                    ui.label("Top recommendations").classes("ag-section-title")
                    ui.label("The datasets the research agent picks first in this market.").classes("ag-muted text-xs")
            top_box = ui.row().classes("w-full gap-3 no-wrap mt-2")
        with card("grow"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("sell").classes("ag-pink")
                ui.label("Categories").classes("ag-section-title")
            cat_box = ui.column().classes("w-full")

    state = {"rows": [], "usage": "any", "theme": None}
    with card("w-full"):
        with ui.row().classes("items-center w-full gap-3"):
            ui.icon("storage").classes("ag-muted")
            count_lbl = ui.label().classes("ag-section-title")
            ui.space()
            search = ui.input("Search datasets").props("clearable dense").classes("w-64")

            def scope() -> dict:
                return {"region": region.value, "delay": int(delay.value), "universe": universe.value}

            async def rate_ai() -> None:
                await jobs.submit("rate_datasets", f"AI-rate datasets {region.value} D{delay.value} {universe.value}",
                                  scope())
                ui.notify("Rating datasets with AI in the background (see Tasks).")

            async def sync_scope() -> None:
                if not brain.client.logged_in:
                    ui.notify("Sign in to BRAIN first.", type="warning")
                    return
                await jobs.submit("sync_scope", f"Catalogue {region.value} D{delay.value} {universe.value}", scope())
                ui.notify("Catalogue sync queued (see Tasks).")

            async def analyze_top() -> None:
                await jobs.submit("analyze_top_datasets", f"AI briefs for top datasets {region.value}",
                                  {**scope(), "n": 5})
                ui.notify("Writing AI briefs for the top 5 datasets without one (see Tasks).")

            ui.button("Rate datasets with AI", icon="auto_awesome", on_click=rate_ai).props("dense")
            ui.button("Sync catalogue", icon="cloud_download", on_click=sync_scope).props("flat dense")
            ui.button("AI-brief top 5", icon="description", on_click=analyze_top).props("flat dense")
        with ui.row().classes("items-center gap-2 mt-1"):
            segmented({"any": "Any usage", "single": "Single", "double": "Double", "interaction": "Interaction"}, "any",
                      lambda v: (state.update(usage=v), apply()))
        theme_row = ui.row().classes("gap-2 flex-wrap")
        grid = ui.aggrid(grid_options(DS_COLUMNS, [], tooltipShowDelay=300), theme="quartz").classes(
            "w-full h-[460px]")

    detail = ui.column().classes("w-full gap-4")

    def apply() -> None:
        rows = state["rows"]
        if state["usage"] != "any":
            rows = [x for x in rows if x["recommended"] == state["usage"]]
        if state["theme"]:
            rows = [x for x in rows if x["theme"] == state["theme"]]
        term = (search.value or "").lower()
        if term:
            rows = [x for x in rows if term in (x["id"] + " " + (x["name"] or "") + " " + (x["theme"] or "")).lower()]
        for i, x in enumerate(rows, 1):
            x["rank"] = i
        grid.options["rowData"] = rows
        grid.update()
        count_lbl.set_text(f"{len(rows)} datasets")

    def draw_themes() -> None:
        theme_row.clear()
        themes = sorted({x["theme"] for x in state["rows"] if x["theme"]})
        with theme_row:
            for t in themes:
                active = state["theme"] == t
                ui.button(t, on_click=lambda _e, t=t: (state.update(theme=None if state["theme"] == t else t),
                                                       draw_themes(), apply())).props(
                    f"dense no-caps {'unelevated color=primary' if active else 'outline color=grey-5'}")

    async def reload() -> None:
        sc = scope()
        rows = await rated_datasets(sc["region"], sc["delay"], sc["universe"])
        state["rows"] = rows
        rated = [x for x in rows if x["rated"]]
        async with db.session() as s:
            n_fields = await s.scalar(select(func.count()).select_from(DataField).where(
                DataField.region == sc["region"], DataField.delay == sc["delay"], DataField.universe == sc["universe"]))
        t_ds.set_text(fmt_int(len(rated)))
        t_ds_sub.set_text(f"of {len(rows)} datasets" if rows else "Sync the catalogue first")
        t_fl.set_text(fmt_int(n_fields or 0))
        for tile, key in ((t_single, "single"), (t_double, "double"), (t_inter, "interaction")):
            tile.set_text(fmt_int(sum(1 for x in rated if x["recommended"] == key)))
        top_box.clear()
        with top_box:
            top = [x for x in rows if x["rated"]][:3]
            if not top:
                ui.label("No AI ratings yet — press 'Rate datasets with AI' (sync the catalogue first).").classes(
                    "ag-muted text-sm")
            for x in top:
                with ui.card().classes("ag-tile grow min-w-[200px] cursor-pointer").on(
                        "click", lambda _e, d=x["id"]: open_dataset(d)):
                    with ui.row().classes("items-center w-full"):
                        ui.label(x["id"]).classes("ag-mono")
                        ui.space()
                        ui.label(f"{x['ai_score']:.0f}").classes("ag-kpi-sm ag-green")
                    with ui.row().classes("items-center gap-2"):
                        ui.label(x["theme"] or "").classes("text-sm")
                        badge(USAGES.get(x["recommended"], ""), "" if x["recommended"] == "single" else "pending")
                    ui.label(x["reason"] or x["name"] or "").classes("ag-muted text-xs")
        cat_box.clear()
        counts = Counter(x["theme"] for x in rated)
        with cat_box:
            if counts:
                items = counts.most_common(12)[::-1]
                ui.echart({"backgroundColor": "transparent", "grid": {"left": 130, "right": 20, "top": 10, "bottom": 25},
                           "xAxis": {"type": "value", "minInterval": 1, "axisLabel": {"color": "#8b98ad"},
                                     "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
                           "yAxis": {"type": "category", "data": [k for k, _ in items],
                                     "axisLabel": {"color": "#c9d4e5", "fontFamily": "monospace"}},
                           "series": [{"type": "bar", "data": [v for _, v in items],
                                       "itemStyle": {"color": "#8b8fb8"}}]}).classes("w-full h-64")
            else:
                ui.label("Categories appear after the AI rating.").classes("ag-muted text-sm")
        draw_themes()
        apply()

    def on_region(_=None) -> None:  # noqa: ANN001
        delay.options = opts.delays(region.value)
        delay.value = delay.options[0]
        universe.options = opts.universes(region.value)
        universe.value = universe.options[0]
        delay.update()
        universe.update()

    region.on_value_change(on_region)
    for el in (delay, universe):
        el.on_value_change(lambda _: reload())
    search.on("keydown.enter", lambda _: apply())
    search.on_value_change(lambda e: apply() if not e.value else None)

    async def open_dataset(dataset_id: str) -> None:
        sc = scope()
        async with db.session() as s:
            ds = await s.scalar(select(Dataset).where(Dataset.id == dataset_id, Dataset.region == sc["region"],
                                                      Dataset.delay == sc["delay"], Dataset.universe == sc["universe"]))
            fields = list(await s.scalars(select(DataField).where(
                DataField.dataset_id == dataset_id, DataField.region == sc["region"], DataField.delay == sc["delay"],
                DataField.universe == sc["universe"]).order_by(DataField.alpha_count)))
        insight = await get_insight(dataset_id, sc["region"], sc["delay"], sc["universe"])
        rating = next((x for x in state["rows"] if x["id"] == dataset_id), None)
        detail.clear()
        with detail:
            with card("w-full"):
                with ui.row().classes("items-center w-full gap-2"):
                    ui.label(f"{dataset_id} — {ds.name if ds else ''}").classes("text-lg font-semibold")
                    ui.label(ds.category_name if ds else "").classes("ag-badge")
                    if rating and rating["rated"]:
                        badge(f"AI {rating['ai_score']:.0f} · {USAGES.get(rating['recommended'], '')}", "pending")
                    ui.space()

                    async def analyze() -> None:
                        btn.props("loading")
                        try:
                            await analyze_dataset(dataset_id, sc["region"], sc["delay"], sc["universe"])
                            await open_dataset(dataset_id)
                        except LLMError as exc:
                            notify_error(exc)
                        finally:
                            btn.props(remove="loading")

                    async def gen_ai() -> None:
                        gbtn.props("loading")
                        try:
                            specs = await llm_ideas(sc, dataset_id=dataset_id, n=settings.llm.ideas_per_call,
                                                    goal="Submittable, uncorrelated alphas from this dataset.")
                            await ideas_dialog(specs, f"{len(specs)} AI ideas for {dataset_id}", sc)
                        except LLMError as exc:
                            notify_error(exc)
                        finally:
                            gbtn.props(remove="loading")

                    async def gen_tpl() -> None:
                        specs = await template_ideas(sc, dataset_id=dataset_id, n=12)
                        await ideas_dialog(specs, f"{len(specs)} template ideas for {dataset_id}", sc)

                    def seed_autopilot() -> None:
                        settings.research.dataset_strategy = "pinned"
                        settings.research.pinned_datasets = [dataset_id]
                        settings.research.region, settings.research.delay = sc["region"], sc["delay"]
                        settings.research.universe = sc["universe"]
                        settings.save()
                        ui.navigate.to("/autopilot")

                    async def research() -> None:
                        from .tasks import new_task_dialog
                        kind = (rating or {}).get("recommended") or "single"
                        await new_task_dialog(scope=sc, dataset_ids=[dataset_id],
                                              kind={"double": "pair"}.get(kind, kind))

                    btn = ui.button("Analyze with AI", icon="auto_awesome", on_click=analyze).props("flat")
                    gbtn = ui.button("Generate AI ideas", icon="lightbulb", on_click=gen_ai)
                    ui.button("Template ideas", icon="widgets", on_click=gen_tpl).props("flat")
                    ui.button("Research task…", icon="science", on_click=research).props("flat")
                    ui.button("Seed Autopilot", icon="smart_toy", on_click=seed_autopilot).props("flat")
                if rating and rating["reason"]:
                    ui.label(f"AI: {rating['reason']}").classes("text-sm ag-cyan")
                if ds:
                    ui.label(ds.description or "").classes("text-sm ag-muted")
                    ui.label(f"Coverage {ds.coverage} · value score {ds.value_score} · {fmt_int(ds.alpha_count)} alphas "
                             f"by {fmt_int(ds.user_count)} users · pyramid multiplier {ds.pyramid_multiplier} · "
                             f"themes {ds.themes or '—'}").classes("text-sm")
            if insight:
                with card("w-full"):
                    ui.label("AI brief").classes("ag-section-title")
                    ui.label(insight.summary or "").classes("text-sm")
                    notes = insight.field_notes or {}
                    if notes.get("themes"):
                        ui.label("Themes: " + ", ".join(notes["themes"])).classes("text-sm ag-cyan")
                    if notes.get("pitfalls"):
                        ui.label("Pitfalls: " + "; ".join(notes["pitfalls"])).classes("text-sm ag-amber")
                    with ui.row().classes("w-full gap-4 items-start no-wrap"):
                        with ui.column().classes("w-1/2 gap-1"):
                            ui.label("Key fields").classes("font-semibold text-sm")
                            for kf in notes.get("key_fields", [])[:12]:
                                ui.label(f"• {kf.get('field')}: {kf.get('meaning')} — {kf.get('usage')}").classes(
                                    "text-xs")
                        with ui.column().classes("w-1/2 gap-1"):
                            ui.label("Idea families (copy a template into the Simulation Matrix → Template tab)"
                                     ).classes("font-semibold text-sm")
                            for fam in insight.ideas or []:
                                with ui.column().classes("gap-0 ag-tile w-full"):
                                    ui.label(fam.get("name", "")).classes("ag-violet text-sm")
                                    ui.label(fam.get("hypothesis", "")).classes("text-xs")
                                    ui.label(fam.get("template", "")).classes("ag-mono text-xs ag-cyan")
            with card("w-full"):
                with ui.row().classes("items-center w-full"):
                    ui.label(f"Fields ({len(fields)})").classes("ag-section-title")
                    ui.label("Double-click a field to queue exploration probes (coverage / update frequency; "
                             "4 simulations).").classes("ag-muted text-sm")
                fgrid = ui.aggrid(grid_options(FIELD_COLUMNS, [{
                    "id": f.id, "type": f.type, "coverage": f.coverage, "alpha_count": f.alpha_count,
                    "user_count": f.user_count, "description": f.description} for f in fields]),
                    theme="quartz").classes("w-full h-[420px]")

                async def probe(e) -> None:  # noqa: ANN001
                    row = e.args.get("data") or {}
                    if not row.get("id") or row.get("type") == "GROUP":
                        return
                    n = await probe_field(row["id"], row["type"], sc["region"], sc["delay"], sc["universe"])
                    ui.notify(f"Queued {n} probe simulations for {row['id']}.")

                fgrid.on("cellDoubleClicked", probe)

    async def on_pick(e) -> None:  # noqa: ANN001
        did = (e.args.get("data") or {}).get("id")
        if did:
            await open_dataset(did)

    grid.on("cellClicked", on_pick)
    await reload()
