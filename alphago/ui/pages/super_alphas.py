"""Super Alphas — combine your ACTIVE alphas (selection + combo expressions), simulate, describe and submit."""

from __future__ import annotations

from nicegui import ui
from sqlalchemy import select

from ...brain import brain
from ...brain.errors import BrainError
from ...db import db
from ...db.models import Alpha
from ...engine import superalpha
from ...llm import LLMError
from ..alpha_detail import confirm_submit, open_alpha
from ..components import card, grid_options, notify_error, num_col, page_header

COLUMNS = [
    {"field": "id", "headerName": "Alpha", "width": 110},
    {"field": "name", "headerName": "Name", "width": 120},
    num_col("sharpe", "Sharpe", 2, 90), num_col("fitness", "Fitness", 2, 90),
    num_col("turnover", "Turnover", 1, 100, pct=True), num_col("self_corr", "Self corr", 2, 95),
    {"field": "universe", "headerName": "Universe", "width": 110},
    {"field": "code", "headerName": "Expression", "flex": 1, "minWidth": 300, "cellClass": "ag-num"},
]

SELECTION_PRESETS = {
    "name == 'CODE'": "Alphas renamed to the code (guide method)",
    "turnover < 0.3 && self_correlation < 0.6": "Low turnover, low self-correlation",
    "operator_count <= 6": "Simple alphas only",
}
COMBO_PRESETS = {"1": "Equal weight (guide method)"}


async def page() -> None:
    page_header("Super Alphas", "Following the Super Alpha guide: pick ~20 strong ACTIVE alphas from one region, give them "
                                "a code name, select them with `name == 'CODE'` and combine with `1` (equal weight). "
                                "SuperAlphas need turnover 2–40%, selection & combo descriptions (≥ 100 chars each), and "
                                "count toward a separate daily limit (1/day).")
    opts = brain.options
    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        with card("w-[58%]"):
            with ui.row().classes("items-end gap-3 w-full"):
                ui.label("1 · Choose components").classes("ag-section-title")
                region = ui.select(opts.regions, value=opts.regions[0], label="Region").classes("w-28")
                count = ui.label().classes("ag-muted text-sm")
            grid = ui.aggrid(grid_options(COLUMNS, [], multi=True), theme="quartz").classes("w-full h-[460px]")
            with ui.row().classes("items-end gap-3"):
                code = ui.input("Code name", value="321").classes("w-32")
                ui.button("Rename selected to code", icon="badge", on_click=lambda: rename()).props("flat")
                ui.button("Tag selected with code", icon="sell", on_click=lambda: tag()).props("flat")
        with card("grow"):
            ui.label("2 · Expressions & settings").classes("ag-section-title")
            selection = ui.input("Selection expression", value="name == '321'").classes("w-full ag-mono")
            with ui.row().classes("gap-1 flex-wrap"):
                for expr, tip in SELECTION_PRESETS.items():
                    ui.button(expr if len(expr) < 30 else expr[:28] + "…",
                              on_click=lambda e=expr: selection.set_value(e.replace("CODE", code.value or "321"))).props(
                        "flat dense no-caps").tooltip(tip)
            combo = ui.input("Combo expression", value="1").classes("w-full ag-mono")
            with ui.row().classes("gap-3"):
                universe = ui.select(opts.universes(region.value), value=opts.universes(region.value)[0],
                                     label="Universe").classes("w-36")
                delay = ui.select(opts.delays(region.value), value=opts.delays(region.value)[0], label="Delay").classes("w-24")
                neut = ui.select(opts.neutralizations(region.value), value="SUBINDUSTRY" if "SUBINDUSTRY" in
                                 opts.neutralizations(region.value) else opts.neutralizations(region.value)[0],
                                 label="Neutralization").classes("w-40")
            with ui.row().classes("gap-3"):
                decay = ui.number("Decay", value=0, min=0).classes("w-24")
                trunc = ui.number("Truncation", value=0.08, step=0.01).classes("w-28")
                limit = ui.number("Selection limit (≥10)", value=20, min=10).classes("w-40")
                handling = ui.select(["POSITIVE", "NON_ZERO", "NON_NAN"], value="POSITIVE",
                                     label="Selection handling").classes("w-40")
            result = ui.label().classes("text-sm ag-cyan")
            with ui.row().classes("gap-2"):
                ui.button("Preview selection", icon="visibility", on_click=lambda: do_preview()).props("flat")
                sim_btn = ui.button("Simulate SuperAlpha", icon="play_arrow", on_click=lambda: do_sim())
            ui.label("3 · Describe & submit").classes("ag-section-title mt-3")
            sa_id = ui.input("SuperAlpha id (after simulation)").classes("w-56")
            with ui.row().classes("gap-2"):
                ui.button("Open details", icon="open_in_full",
                          on_click=lambda: open_alpha(sa_id.value) if sa_id.value else None).props("flat")
                ui.button("Write descriptions (AI)", icon="auto_awesome", on_click=lambda: describe()).props("flat")
                ui.button("Submit…", icon="send", on_click=lambda: confirm_submit(sa_id.value) if sa_id.value else None
                          ).props("color=positive")

    def settings() -> dict:
        return {"region": region.value, "universe": universe.value, "delay": int(delay.value),
                "neutralization": neut.value, "decay": int(decay.value or 0), "truncation": float(trunc.value or 0.08)}

    async def reload() -> None:
        async with db.session() as s:
            rows = list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE", Alpha.type == "REGULAR",
                                                            Alpha.region == region.value)
                                        .order_by(Alpha.sharpe.desc().nulls_last())))
        grid.options["rowData"] = [{"id": a.id, "name": a.name, "sharpe": a.sharpe, "fitness": a.fitness,
                                    "turnover": a.turnover, "self_corr": a.self_corr, "universe": a.universe,
                                    "code": (a.code or "")[:200]} for a in rows]
        grid.update()
        count.set_text(f"{len(rows)} ACTIVE alphas in {region.value}")
        universe.options, delay.options = opts.universes(region.value), opts.delays(region.value)
        if universe.value not in universe.options:
            universe.value = universe.options[0]
        if delay.value not in delay.options:
            delay.value = delay.options[0]
        universe.update()
        delay.update()

    region.on_value_change(lambda _: reload())

    async def chosen() -> list[str]:
        return [r["id"] for r in await grid.get_selected_rows()]

    async def rename() -> None:
        ids = await chosen()
        if not ids:
            ui.notify("Select component alphas first.", type="warning")
            return
        with ui.dialog() as d, ui.card().classes("ag-card p-5"):
            ui.label(f"Rename {len(ids)} ACTIVE alphas to '{code.value}' on BRAIN?").classes("font-semibold")
            ui.label("Only the alpha name changes; this lets the selection expression pick them.").classes("text-sm")
            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=d.close).props("flat")

                async def go() -> None:
                    d.close()
                    try:
                        n = await superalpha.rename_alphas(ids, code.value)
                        selection.value = f"name == '{code.value}'"
                        ui.notify(f"Renamed {n} alphas.", type="positive")
                        await reload()
                    except BrainError as exc:
                        notify_error(exc)

                ui.button("Rename", on_click=go)
        d.open()

    async def tag() -> None:
        ids = await chosen()
        if ids:
            try:
                n = await superalpha.tag_alphas(ids, code.value)
                ui.notify(f"Tagged {n} alphas with '{code.value}'.", type="positive")
            except BrainError as exc:
                notify_error(exc)

    async def do_preview() -> None:
        try:
            data = await superalpha.preview(selection.value, settings(), int(limit.value or 20), handling.value)
            result.set_text(f"Selection matches {data.get('count', '?')} alphas. {data.get('message') or ''}")
        except BrainError as exc:
            notify_error(exc)

    async def do_sim() -> None:
        if not brain.client.logged_in:
            ui.notify("Sign in to BRAIN first.", type="warning")
            return
        sim_btn.props("loading")
        try:
            aid = await superalpha.simulate_super(selection.value, combo.value, settings(), int(limit.value or 20),
                                                  handling.value)
            sa_id.value = aid
            result.set_text(f"SuperAlpha {aid} simulated — open details to see Pnl_combo vs Pnl_eqw.")
            await open_alpha(aid)
        except Exception as exc:
            notify_error(exc, "SuperAlpha simulation failed: ")
        finally:
            sim_btn.props(remove="loading")

    async def describe() -> None:
        if not sa_id.value:
            ui.notify("Simulate a SuperAlpha first.", type="warning")
            return
        ids = await chosen()
        try:
            desc = await superalpha.write_descriptions(sa_id.value, selection.value, combo.value,
                                                       f"{len(ids)} selected components in {region.value}")
            ui.notify("Descriptions saved to BRAIN.", type="positive")
            with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[700px]"):
                ui.label("Selection description").classes("font-semibold")
                ui.label(desc.selection_description).classes("text-sm")
                ui.label("Combo description").classes("font-semibold mt-2")
                ui.label(desc.combo_description).classes("text-sm")
            d.open()
        except (LLMError, BrainError) as exc:
            notify_error(exc)

    await reload()
