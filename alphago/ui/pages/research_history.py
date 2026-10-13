"""Research History — every idea tried and its verdict (the journal), plus runs and batches over time."""

from __future__ import annotations

from nicegui import ui
from sqlalchemy import or_, select

from ...db import db
from ...db.models import AutopilotRun, Batch, JournalEntry
from ..alpha_detail import open_alpha
from ..components import card, grid_options, num_col, page_header

JOURNAL_COLUMNS = [
    {"field": "ts", "headerName": "When", "width": 130},
    {"field": "verdict", "headerName": "Verdict", "width": 100},
    num_col("sharpe", "Sharpe", 2, 90), num_col("fitness", "Fitness", 2, 90),
    num_col("turnover", "Turnover", 1, 100, pct=True),
    {"field": "dataset", "headerName": "Dataset", "width": 130},
    {"field": "failing", "headerName": "Failing checks", "width": 240},
    {"field": "expression", "headerName": "Expression", "flex": 2, "minWidth": 320, "cellClass": "ag-num"},
    {"field": "hypothesis", "headerName": "Hypothesis", "flex": 1, "minWidth": 240},
    {"field": "alpha_id", "headerName": "Alpha", "width": 105},
]


async def page() -> None:
    page_header("Research History", "The idea journal: every simulated idea, its hypothesis and verdict. Failed shapes are "
                                    "fed back to Gemini as 'don't repeat' memory.")
    with card("w-full"):
        ui.label("Autopilot runs").classes("ag-section-title")
        async with db.session() as s:
            runs = list(await s.scalars(select(AutopilotRun).order_by(AutopilotRun.id.desc()).limit(50)))
            batches = list(await s.scalars(select(Batch).order_by(Batch.id.desc()).limit(500)))
        by_run: dict[int | None, list[Batch]] = {}
        for b in batches:
            by_run.setdefault(b.run_id, []).append(b)
        if not runs:
            ui.label("No runs yet.").classes("ag-muted text-sm")
        with ui.column().classes("w-full gap-1"):
            for r in runs:
                bs = by_run.get(r.id, [])
                ui.label(f"Run #{r.id} · {r.started_at:%Y-%m-%d %H:%M} · {r.mode} · goal {r.goal} · {r.status} · "
                         f"{len(bs)} batches · strong {sum(b.n_strong for b in bs)} · ready {sum(b.n_ready for b in bs)}"
                         + (f" · {r.reason}" if r.reason else "")).classes("ag-mono text-sm")

    with card("w-full"):
        with ui.row().classes("items-end gap-3 w-full"):
            ui.label("Idea journal").classes("ag-section-title")
            search = ui.input("Search expression / hypothesis").classes("w-80").props("clearable")
            verdict = ui.select(["all", "ready", "strong", "promising", "weak", "rejected"], value="all",
                                label="Verdict").classes("w-36")
            count = ui.label().classes("ag-muted text-sm")
        grid = ui.aggrid(grid_options(JOURNAL_COLUMNS, []), theme="quartz").classes("w-full h-[600px]")

    async def reload() -> None:
        stmt = select(JournalEntry)
        if verdict.value != "all":
            stmt = stmt.where(JournalEntry.verdict == verdict.value)
        if search.value:
            term = f"%{search.value}%"
            stmt = stmt.where(or_(JournalEntry.expression.like(term), JournalEntry.hypothesis.like(term)))
        async with db.session() as s:
            rows = list(await s.scalars(stmt.order_by(JournalEntry.id.desc()).limit(3000)))
        grid.options["rowData"] = [{
            "ts": f"{r.ts:%m-%d %H:%M}", "verdict": r.verdict, "sharpe": r.sharpe, "fitness": r.fitness,
            "turnover": r.turnover, "dataset": r.dataset_id or "", "failing": ", ".join(r.failing_checks or []),
            "expression": r.expression, "hypothesis": r.hypothesis or "", "alpha_id": r.alpha_id or ""} for r in rows]
        grid.update()
        count.set_text(f"{len(rows):,} ideas")

    async def open_(e) -> None:  # noqa: ANN001
        aid = (e.args.get("data") or {}).get("alpha_id")
        if aid:
            await open_alpha(aid)

    grid.on("cellDoubleClicked", open_)
    verdict.on_value_change(lambda _: reload())
    search.on("keydown.enter", lambda _: reload())
    await reload()
