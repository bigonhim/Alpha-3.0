"""Quota Planner — what 100 simulations bought so far by dataset and strategy, and a suggested split of a quota."""

from __future__ import annotations

from nicegui import ui

from ...config import settings
from ...engine.planner import allocate, yield_table
from ..components import card, grid_options, info_box, num_col, page_header

COLUMNS = [
    {"field": "dataset", "headerName": "Dataset", "width": 170},
    {"field": "strategy_label", "headerName": "Strategy", "width": 170},
    {"field": "sims", "headerName": "Simulations", "width": 115, "type": "numericColumn"},
    num_col("promising_per_100", "Promising / 100", 1, 135),
    num_col("strong_per_100", "Strong / 100", 1, 120),
    num_col("passed_per_100", "Passed check / 100", 2, 150),
    {"field": "interval", "headerName": "80% range", "width": 120},
    {"field": "submitted", "headerName": "Submitted", "width": 105, "type": "numericColumn"},
]


async def page() -> None:
    page_header("Quota Planner", "Expected results per 100 simulations, by dataset and strategy, from your own "
                                 "history — so quota goes where submittable alphas actually came from.")
    info_box("How to read it", "Counts first-pass ideas only (repairs and robustness tests are not new ideas); a "
                               "near miss the repair loop made submittable counts for the idea it came from. 'Passed "
                               "check' = BRAIN's submission check passed. The 80% range is wide when there are few "
                               "simulations — treat small rows as unknown, not bad.")
    r = settings.research
    rows: list[dict] = []
    with card("w-full"):
        with ui.row().classes("items-end gap-3 w-full"):
            region = ui.input("Region", value=r.region).classes("w-24")
            delay = ui.select([0, 1], value=r.delay, label="Delay").classes("w-24")
            min_sims = ui.number("Min simulations", value=20, min=1).classes("w-36")
            total = ui.number("Simulations to allocate", value=r.autopilot_daily_cap, min=10, step=50).classes("w-48")
            ui.space()
        grid = ui.aggrid(grid_options(COLUMNS, []), theme="quartz").classes("w-full h-[420px]")
    with card("w-full"):
        ui.label("Suggested split").classes("ag-section-title")
        ui.label("80% in proportion to each row's estimated pass rate, 20% spread over rows with under 50 "
                 "simulations (exploring). Research-task rows can be created as planned tasks — they wait for your "
                 "approval on the Autopilot page.").classes("ag-muted text-xs")
        alloc_grid = ui.aggrid(grid_options([
            {"field": "dataset", "headerName": "Dataset", "width": 170},
            {"field": "strategy_label", "headerName": "Strategy", "width": 170},
            {"field": "allocate", "headerName": "Simulations", "width": 115, "type": "numericColumn"},
            num_col("expected_passed", "Expected passed", 2, 140),
            {"field": "task_kind", "headerName": "Task kind", "width": 120},
        ], []), theme="quartz").classes("w-full h-64")

        async def reload() -> None:
            rows.clear()
            rows.extend(await yield_table(region.value or r.region, int(delay.value), int(min_sims.value or 1)))
            for x in rows:
                x["interval"] = f"{x['passed_lo']:.1f}–{x['passed_hi']:.1f}"
            grid.options["rowData"] = rows
            grid.update()
            alloc_grid.options["rowData"] = allocate(rows, int(total.value or 0))
            alloc_grid.update()

        async def create_tasks() -> None:
            from ...engine.research import research
            made = 0
            for x in alloc_grid.options.get("rowData") or []:
                if x.get("task_kind") and x["dataset"] != "—" and (x["task_kind"] != "pair" or "," in x["dataset"]):
                    ids = x["dataset"].split(",")
                    try:
                        await research.create(kind=x["task_kind"], region=region.value or r.region,
                                              delay=int(delay.value), universe=r.universe, dataset_ids=ids,
                                              planned_sims=x["allocate"], source="assisted", status="planned",
                                              reason=f"Quota Planner: {x['passed_per_100']:.2f} passed per 100 so "
                                                     f"far ({x['sims']} simulations).")
                        made += 1
                    except ValueError:
                        continue
            ui.notify(f"{made} planned task(s) created — approve them on the Autopilot page." if made else
                      "No row in the split can become a research task (AI batches run from the Autopilot page).",
                      type="positive" if made else "info")

        for el in (delay,):
            el.on_value_change(lambda _: reload())
        with ui.row().classes("gap-2"):
            ui.button("Recalculate", icon="calculate", on_click=reload).props("flat")
            ui.button("Create planned tasks", icon="playlist_add", on_click=create_tasks).props("flat")
    await reload()
