"""Pyramids (ported from alpha-harness screens/pyramids): BRAIN's Pyramid Multiplier for every Region · Delay ·
Dataset Category, and the alphas you submitted in each, in a quarter or ever. A cell whose fields are synced opens
that market in the Data Explorer, filtered to the category."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ...brain import brain
from ...brain.errors import BrainError
from ...catalog import pyramids as pyr
from ...catalog import queries
from ..components import card, empty_state, notify_error, page_header, segmented
from .data_explorer import open_market

TOTAL_NOTE = "counts an alpha once in every pyramid it is in"


async def page() -> None:
    page_header("Pyramids", "Pyramid multipliers and your alphas per pyramid, by quarter or all time.")
    state: dict[str, Any] = {"view": "multiplier", "period": None, "grid": None}
    with card("w-full gap-3"):
        with ui.row().classes("items-center w-full gap-3"):
            title = ui.label().classes("ag-section-title")
            ui.space()
            period = ui.select({}, label="Period").props("dense").classes("w-56")
            view = segmented({"multiplier": "Multiplier", "alphas": "Alphas"}, "multiplier")
            ui.button(icon="refresh", on_click=lambda: load(refresh=True)).props("flat dense").tooltip(
                "Read again from BRAIN (multipliers are kept 6 hours, counts 10 minutes)")
        subtitle = ui.label().classes("ag-muted text-sm")
        body = ui.column().classes("w-full gap-2")
        legend = ui.row().classes("gap-4 flex-wrap text-xs ag-muted items-center")

    async def load(refresh: bool = False) -> None:
        if refresh:
            pyr.clear_cache()
        if not brain.client.logged_in:
            body.clear()
            with body:
                empty_state("login", "Sign in to BRAIN to see your pyramids",
                            "Multipliers and your submitted alphas per pyramid come from your BRAIN account.")
                ui.button("Open Sync with BRAIN", on_click=lambda: ui.navigate.to("/sync")).props("flat")
            return
        try:
            state["grid"] = await pyr.pyramid_grid(state["period"])
        except pyr.UnknownPeriodError as exc:
            ui.notify(str(exc), type="warning")
            state["period"] = None
            state["grid"] = await pyr.pyramid_grid(None)
        except BrainError as exc:
            body.clear()
            with body:
                empty_state("cloud_off", "Could not load pyramids from BRAIN",
                            f"{exc.message} — sign in on Sync with BRAIN, then refresh.")
            return
        except Exception as exc:
            notify_error(exc, "Pyramids: ")
            return
        grid = state["grid"]
        options = {pyr.ALL_TIME: "All Time"}
        for i, q in enumerate(grid["quarters"]):
            options[q["name"]] = f"{q['name']} (Current)" if i == 0 else q["name"]
        period.options = options
        period.value = grid["period"]
        period.update()
        render()

    def render() -> None:
        grid = state["grid"]
        if grid is None:
            return
        alphas_view = state["view"] == "alphas"
        all_time = grid["period"] == pyr.ALL_TIME
        during = "across all time" if all_time else f"in {grid['period']}"
        period.set_visibility(alphas_view)
        title.set_text("Pyramid Alpha Distribution" if alphas_view else "Pyramid Multiplier Map")
        subtitle.set_text(f"Alphas you submitted in each pyramid {during}." if alphas_view
                          else "What BRAIN pays for a submission in each pyramid. Both axes run richest first, so "
                               "the strongest pyramids gather top-left.")
        needed = int(grid["alphas_per_pyramid"])
        cells = {(c["category_id"], c["region"], c["delay"]): c for c in grid["cells"]}
        by_row: dict[str, list[float]] = {}
        by_col: dict[tuple[str, int], list[float]] = {}
        row_totals: dict[str, int] = {}
        col_totals: dict[tuple[str, int], int] = {}
        for c in grid["cells"]:
            row_totals[c["category_id"]] = row_totals.get(c["category_id"], 0) + c["alpha_count"]
            col = (c["region"], c["delay"])
            col_totals[col] = col_totals.get(col, 0) + c["alpha_count"]
            if c["multiplier"] is not None:
                by_row.setdefault(c["category_id"], []).append(float(c["multiplier"]))
                by_col.setdefault(col, []).append(float(c["multiplier"]))
        best_row, best_col = pyr.peaks(by_row), pyr.peaks(by_col)
        ceiling = max([m for v in by_row.values() for m in v] or [pyr.TOP_MULTIPLIER])
        body.clear()
        legend.clear()
        with body:
            if not grid["categories"]:
                empty_state("change_history", "No Dataset Categories", "BRAIN returned no pyramids to show.")
                return
            with ui.element("div").classes("w-full overflow-x-auto"):
                with ui.element("table").classes("ag-pyramids").style("border-collapse:separate;border-spacing:3px"):
                    with ui.element("tr"):
                        with ui.element("th").classes("text-left text-xs ag-muted px-2"):
                            ui.label("Category")
                        for col in grid["columns"]:
                            with ui.element("th").classes("text-xs ag-mono px-2"):
                                ui.label(f"{col['region']} D{col['delay']}")
                        if alphas_view:
                            with ui.element("th").classes("text-xs px-2").props(f"title='{TOTAL_NOTE}'"):
                                ui.label("Total")
                    for cat in grid["categories"]:
                        with ui.element("tr"):
                            with ui.element("th").classes("text-left text-sm px-2 whitespace-nowrap"):
                                ui.label(cat["name"])
                            for col in grid["columns"]:
                                cell = cells.get((cat["id"], col["region"], col["delay"]))
                                render_cell(cat, col, cell, alphas_view, all_time, needed, best_row, best_col,
                                            ceiling, during)
                            if alphas_view:
                                with ui.element("td").classes("text-center ag-mono text-sm ag-muted px-2"):
                                    ui.label(f"{row_totals.get(cat['id'], 0):,}")
                    if alphas_view:
                        with ui.element("tr"):
                            with ui.element("th").classes("text-left text-xs ag-muted px-2").props(
                                    f"title='{TOTAL_NOTE}'"):
                                ui.label("Total")
                            for col in grid["columns"]:
                                with ui.element("td").classes("text-center ag-mono text-sm ag-muted"):
                                    ui.label(f"{col_totals.get((col['region'], col['delay']), 0):,}")
                            with ui.element("td").classes("text-center ag-mono text-sm"):
                                ui.label(f"{sum(row_totals.values()):,}")
        with legend:
            if alphas_view and all_time:
                ui.label("— none submitted · a number: submitted, counted across every quarter")
            elif alphas_view:
                ui.label(f"— none submitted · n/{needed}: started, not yet formulated · "
                         f"bold: formulated ({needed}+ alphas this quarter)")
            else:
                for i in (0, 3, 6, 10):
                    ui.element("span").style(f"display:inline-block;width:22px;height:12px;border-radius:2px;"
                                             f"background:{pyr.RAMP[i]}")
                ui.label("lower → higher Multiplier · ▲ best market for this category · ◀ best category in this "
                         "market")
            ui.label("Click a synced cell to browse that pyramid's Data Fields; a dimmed cell's market is not "
                     "synced yet.")

    def render_cell(cat: dict[str, Any], col: dict[str, Any], cell: dict[str, Any] | None, alphas_view: bool,
                    all_time: bool, needed: int, best_row: dict[str, float], best_col: dict[tuple[str, int], float],
                    ceiling: float, during: str) -> None:
        if cell is None:
            with ui.element("td").classes("text-center ag-muted text-sm").style("min-width:64px;height:34px"):
                ui.label("—")
            return
        m = cell["multiplier"]
        top_row = not alphas_view and m is not None and m == best_row.get(cat["id"])
        top_col = not alphas_view and m is not None and m == best_col.get((col["region"], col["delay"]))
        label, note = pyr.standing(int(cell["alpha_count"]), needed, all_time)
        if alphas_view:
            formulated = not all_time and cell["alpha_count"] >= needed
            face = ("background:#5b8def;color:#fff;font-weight:600" if formulated else
                    "background:#1e2b40" if cell["alpha_count"] else "background:#121a28;color:#6b7a90")
            text = label
        else:
            face = f"background:{pyr.ramp_colour(m, ceiling)};color:#fff"
            text = "—" if m is None else f"×{float(m):.1f}" + (" ▲" if top_row else "") + (" ◀" if top_col else "")
        notes = [n for n, on in (("best market for this category", top_row),
                                 ("best category in this market", top_col)) if on]
        synced = bool(cell.get("synced"))
        tip = (f"{cat['name']} · {col['region']} D{col['delay']} · Pyramid Multiplier "
               f"{'—' if m is None else f'×{float(m):.1f}'} · {cell['alpha_count']:,} of your alphas {during}, {note}"
               + (f" · {' · '.join(notes)}" if notes else "")
               + ("" if synced else " · not synced: download this market on Sync with BRAIN to browse it"))
        td = ui.element("td").classes("text-center ag-mono text-sm rounded").style(
            f"{face};min-width:64px;height:34px;cursor:{'pointer' if synced else 'default'};"
            f"opacity:{1 if synced else 0.55}")
        td.tooltip(tip)
        with td:
            ui.label(text)
        if synced:
            td.on("click", lambda _e, c=cat["id"], r=col["region"], d=col["delay"]: open_cell(c, r, d))

    async def open_cell(category_id: str, region: str, delay: int) -> None:
        best = await queries.best_universes()
        universe = best.get((region, int(delay)))
        if universe is None:
            ui.notify(f"{region} D{delay} is not synced yet. Download it on Sync with BRAIN.", type="warning")
            return
        await open_market(region, int(delay), universe, category_ids=[category_id])

    def on_view(value: str) -> None:
        state["view"] = value
        render()

    async def on_period(e: Any) -> None:
        if e.value and state["grid"] is not None and e.value != state["grid"]["period"]:
            state["period"] = e.value
            await load()

    view.on_value_change(lambda e: on_view(e.value))
    period.on_value_change(on_period)
    await load()
    ui.timer(600, lambda: load(), active=True)
