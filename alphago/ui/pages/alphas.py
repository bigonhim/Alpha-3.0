"""Alphas — every alpha the app simulated or synced from BRAIN, with filters, details and actions."""

from __future__ import annotations

import csv
import io
from datetime import datetime

from nicegui import ui
from sqlalchemy import or_, select

from ...db import db
from ...db.models import Alpha
from ...jobs.queue import jobs
from ..alpha_detail import open_alpha, robust_label
from ..components import card, grid_options, num_col, page_header

VERDICTS = ["all", "ready", "strong", "promising", "weak", "rejected", "submitted"]


async def query_alphas(f: dict) -> list[Alpha]:
    stmt = select(Alpha)
    if f.get("verdict") and f["verdict"] != "all":
        stmt = stmt.where(Alpha.verdict == f["verdict"])
    if f.get("status") and f["status"] != "all":
        stmt = stmt.where(Alpha.status == f["status"])
    if f.get("region") and f["region"] != "all":
        stmt = stmt.where(Alpha.region == f["region"])
    if f.get("origin") and f["origin"] != "all":
        stmt = stmt.where(Alpha.origin == f["origin"])
    if f.get("check") and f["check"] != "all":
        stmt = stmt.where(Alpha.check_state.is_(None) if f["check"] == "unchecked" else Alpha.check_state == f["check"])
    if f.get("robust") and f["robust"] != "all":
        stmt = stmt.where(Alpha.robust_state == f["robust"])
    if f.get("min_sharpe") not in (None, ""):
        stmt = stmt.where(Alpha.sharpe >= float(f["min_sharpe"]))
    if f.get("search"):
        term = f"%{f['search']}%"
        stmt = stmt.where(or_(Alpha.code.like(term), Alpha.id.like(term), Alpha.name.like(term)))
    order = {"score": Alpha.score.desc(), "sharpe": Alpha.sharpe.desc(), "newest": Alpha.date_created.desc(),
             "fitness": Alpha.fitness.desc()}.get(f.get("order", "score"), Alpha.score.desc())
    stmt = stmt.order_by(order.nulls_last()).limit(int(f.get("limit", 2000)))
    async with db.session() as s:
        return list(await s.scalars(stmt))


def alpha_rows(alphas: list[Alpha]) -> list[dict]:
    return [{
        "id": a.id, "verdict": a.verdict, "status": a.status, "sharpe": a.sharpe, "fitness": a.fitness,
        "turnover": a.turnover, "returns": a.returns, "drawdown": a.drawdown, "margin": a.margin,
        "region": a.region, "universe": a.universe, "delay": a.delay, "neutralization": a.neutralization,
        "decay": a.decay, "truncation": a.truncation, "ops": a.op_count, "check": a.check_state or "",
        "robust": robust_label(a), "est_corr": a.est_self_corr, "cluster": a.cluster or "",
        "created": a.date_created.strftime("%Y-%m-%d %H:%M") if a.date_created else "",
        "code": (a.code or a.selection or "")[:300], "origin": a.origin, "score": a.score,
    } for a in alphas]


ALPHA_COLUMNS = [
    {"field": "id", "headerName": "Alpha", "width": 110, "pinned": "left"},
    {"field": "verdict", "headerName": "Verdict", "width": 105},
    num_col("sharpe", "Sharpe", 2, 90),
    num_col("fitness", "Fitness", 2, 90),
    num_col("turnover", "Turnover", 1, 100, pct=True),
    num_col("returns", "Returns", 1, 95, pct=True),
    num_col("drawdown", "Drawdown", 1, 100, pct=True),
    {"field": "universe", "headerName": "Universe", "width": 115},
    {"field": "neutralization", "headerName": "Neutralization", "width": 140},
    {"field": "decay", "headerName": "Decay", "width": 80},
    {"field": "truncation", "headerName": "Trunc", "width": 80},
    {"field": "region", "headerName": "Region", "width": 85},
    {"field": "delay", "headerName": "D", "width": 60},
    {"field": "check", "headerName": "BRAIN check", "width": 115,
     "headerTooltip": "pass / fail / pending; stale = you submitted another alpha since, so check again"},
    {"field": "robust", "headerName": "Robust", "width": 115,
     "headerTooltip": "AlphaGo's robustness test: share of Sharpe the worst neighbouring setting keeps (< 80% = fragile)"},
    num_col("est_corr", "Est. self-corr", 2, 120),
    {"field": "cluster", "headerName": "Idea", "width": 110},
    {"field": "status", "headerName": "Status", "width": 120},
    {"field": "ops", "headerName": "Ops", "width": 70},
    {"field": "created", "headerName": "Created", "width": 140},
    {"field": "code", "headerName": "Expression", "minWidth": 380, "flex": 1, "cellClass": "ag-num"},
]


async def page() -> None:
    page_header("Alphas", "Everything simulated by the app or synced from your BRAIN account. Click a row for details.")
    f = {"verdict": "all", "status": "all", "region": "all", "origin": "all", "min_sharpe": None, "search": "",
         "order": "score", "limit": 2000, "check": "all", "robust": "all"}

    with card("w-full"):
        with ui.row().classes("items-end gap-3 w-full flex-wrap"):
            search = ui.input("Search expression / id / name").classes("w-72").props("clearable")
            verdict = ui.select(VERDICTS, value="all", label="Verdict").classes("w-36")
            status = ui.select(["all", "UNSUBMITTED", "ACTIVE", "DECOMMISSIONED"], value="all", label="Status").classes("w-40")
            region = ui.select(["all", "USA", "GLB", "EUR", "ASI", "CHN", "IND", "JPN", "KOR", "TWN", "HKG", "AMR", "GBR"],
                               value="all", label="Region").classes("w-28")
            origin = ui.select(["all", "app", "synced"], value="all", label="Origin").classes("w-28")
            check = ui.select({"all": "all", "pass": "passed", "fail": "failed", "pending": "pending",
                               "stale": "re-check needed", "unchecked": "not checked"}, value="all",
                              label="BRAIN check").classes("w-40")
            robust = ui.select({"all": "all", "robust": "robust", "fragile": "fragile", "unknown": "untested"},
                               value="all", label="Robust").classes("w-32")
            min_sharpe = ui.number("Min Sharpe", value=None, step=0.1).classes("w-28")
            order = ui.select({"score": "Score", "sharpe": "Sharpe", "fitness": "Fitness", "newest": "Newest"},
                              value="score", label="Sort").classes("w-32")
            count = ui.label().classes("ag-muted text-sm ml-auto")

        grid = ui.aggrid(grid_options(ALPHA_COLUMNS, [], multi=True), theme="quartz").classes("w-full h-[640px]")

        async def reload() -> None:
            f.update(search=search.value or "", verdict=verdict.value, status=status.value, region=region.value,
                     origin=origin.value, min_sharpe=min_sharpe.value, order=order.value, check=check.value,
                     robust=robust.value)
            rows = alpha_rows(await query_alphas(f))
            grid.options["rowData"] = rows
            grid.update()
            count.set_text(f"{len(rows):,} alphas")

        for el in (verdict, status, region, origin, order, check, robust):
            el.on_value_change(lambda _: reload())
        search.on("keydown.enter", lambda _: reload())
        min_sharpe.on("keydown.enter", lambda _: reload())

        async def on_row(e) -> None:  # noqa: ANN001
            row = e.args.get("data") or {}
            if row.get("id"):
                await open_alpha(row["id"], on_change=reload)

        grid.on("cellDoubleClicked", on_row)
        grid.on("cellClicked", lambda e: on_row(e) if (e.args.get("colId") == "id") else None)

        async def selected_ids() -> list[str]:
            rows = await grid.get_selected_rows()
            return [r["id"] for r in rows]

        async def check_selected() -> None:
            ids = await selected_ids()
            if not ids:
                ui.notify("Select alphas first (checkbox column).", type="warning")
                return
            await jobs.submit("check_many", f"Check {len(ids)} alphas", {"alpha_ids": ids})
            ui.notify(f"Queued checks for {len(ids)} alphas (see Tasks).")

        async def refresh_selected() -> None:
            ids = await selected_ids()
            if ids:
                await jobs.submit("refresh_alphas", f"Refresh {len(ids)} alphas", {"alpha_ids": ids})
                ui.notify("Refresh queued.")

        async def export_csv() -> None:
            rows = grid.options.get("rowData") or []
            if not rows:
                return
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            ui.download(buf.getvalue().encode("utf-8"), f"alphas_{datetime.now():%Y%m%d_%H%M}.csv")

        with ui.row().classes("gap-2"):
            ui.button("Refresh list", icon="refresh", on_click=reload).props("flat")
            ui.button("Run submission check on selected", icon="rule", on_click=check_selected).props("flat")
            ui.button("Re-fetch selected from BRAIN", icon="cloud_sync", on_click=refresh_selected).props("flat")
            ui.button("Export CSV", icon="download", on_click=export_csv).props("flat")
            ui.label("Tip: double-click a row (or click the Alpha id) to open details, check and submit.").classes(
                "ag-muted text-sm self-center")

    await reload()
