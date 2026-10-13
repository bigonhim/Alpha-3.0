"""Osmosis Review — check the points currently on BRAIN, spot mistakes, and ask Gemini what to expect and fix."""

from __future__ import annotations

from collections import defaultdict

from nicegui import ui
from sqlalchemy import select

from ...brain import brain
from ...brain.errors import BrainError
from ...config import settings
from ...db import db
from ...db.models import Alpha, ConsultantSnapshot, OsmosisAllocation
from ...engine import portfolio
from ...engine.sync import last_synced
from ...jobs.queue import jobs
from ..components import (card, fmt_int, fmt_num, grid_options, info_box, lang_toggle, notify_error, num_col,
                          page_header, stat_tile)


async def page() -> None:
    with ui.row().classes("w-full items-start no-wrap"):
        page_header("Osmosis Review", "How the points now on BRAIN are spread, what looks wrong, and what to expect.")
        ui.space()

        async def refresh() -> None:
            if not brain.client.logged_in:
                ui.notify("Sign in to BRAIN first.", type="warning")
                return
            await jobs.submit("prepare_pnl", "Prepare data (submitted alphas + daily PnL)")
            ui.notify("Reading points and daily PnL from BRAIN — reopen this page when the task finishes.")

        ui.button("Refresh data", icon="sync", on_click=refresh)

    seen = await last_synced("active")
    info_box("Good to know",
             "Your real Osmosis result is out of sample, which BRAIN does not show in detail. This review uses each "
             "alpha's in-sample daily PnL (\"last year\" means the last year of that data), so it shows the most likely "
             "causes, not a certainty. Press Refresh data first so the points are read fresh from BRAIN"
             + (f" (last read {seen.astimezone():%d/%m/%Y, %H:%M:%S})." if seen else " (not read yet)."))

    scopes = await portfolio.review()
    with ui.row().classes("w-full gap-4 no-wrap"):
        for label, value, color in (("Scopes with points", len(scopes), "ag-amber"),
                                    ("Alphas with points", sum(s.alphas for s in scopes), "ag-amber"),
                                    ("Scopes with big mistakes", sum(1 for s in scopes if s.big), "ag-red"),
                                    ("Scopes with some mistakes", sum(1 for s in scopes if s.some), "ag-amber")):
            v, _ = stat_tile(label, color, icon="public" if "Scopes" in label else "link")
            v.set_text(fmt_int(value))

    lang = {"value": settings.llm.ai_language}

    def set_lang(v: str) -> None:
        lang["value"] = v
        settings.llm.ai_language = v
        settings.save()

    # ------------------------------------------------------------ prediction
    with card("w-full"):
        ui.label("Prediction: your next Osmosis result").classes("ag-section-title")
        ui.label("Type the Osmosis score BRAIN shows you now. Gemini, acting as a quant finance expert, looks at the points "
                 "on BRAIN now and predicts your next Osmosis result. It is an estimate from in-sample data, not a "
                 "promise. Only numbers are sent, never expressions.").classes("ag-muted text-sm")
        with ui.row().classes("items-center gap-3 mt-2"):
            score = ui.number("Your Osmosis score now (from BRAIN)", placeholder="e.g. -0.8", step=0.1).classes("w-72")
            lang_toggle(lang["value"], set_lang)
            pbtn = ui.button("Predict with AI", icon="auto_awesome")
        pred_box = ui.column().classes("w-full gap-1")
        with ui.card().classes("ag-tile w-full mt-2"):
            ui.label("Your points on BRAIN now").classes("font-semibold")
            combos = [s.combo_sharpe for s in scopes if s.combo_sharpe is not None]
            last2 = [s.combo_2y for s in scopes if s.combo_2y is not None]
            ests = [(s.est_os, s.points) for s in scopes if s.est_os is not None]
            est = sum(e * w for e, w in ests) / sum(w for _, w in ests) if ests else None
            with ui.grid(columns=3).classes("w-full gap-x-10 gap-y-2 mt-1"):
                for label, value, color in (("Scopes", str(len(scopes)), ""),
                                            ("Alphas", str(sum(s.alphas for s in scopes)), ""),
                                            ("Est. OS Sharpe", fmt_num(est), "ag-green"),
                                            ("Avg combo Sharpe", fmt_num(sum(combos) / len(combos) if combos else None), ""),
                                            ("Avg last 2 years", fmt_num(sum(last2) / len(last2) if last2 else None), "")):
                    with ui.column().classes("gap-0"):
                        ui.label(label).classes("ag-muted text-sm")
                        ui.label(value).classes(f"ag-kpi-sm {color}")
            ui.label(" · ".join(f"{s.key} {s.alphas}" for s in scopes) or "No points on BRAIN yet.").classes(
                "ag-muted text-xs ag-mono")

    async def predict() -> None:
        if not scopes:
            ui.notify("No points on BRAIN yet (or not read). Press Refresh data first.", type="warning")
            return
        pbtn.props("loading")
        try:
            res = await portfolio.ai_predict(scopes, score.value, lang["value"])
            pred_box.clear()
            with pred_box:
                arrow = {"up": "↑", "down": "↓"}.get(res.direction.lower(), "→")
                ui.label(f"{arrow} Predicted next score: {res.predicted_low:.2f} to {res.predicted_high:.2f}").classes(
                    "text-lg ag-teal")
                ui.label(res.summary).classes("text-sm")
                for d in res.drivers:
                    ui.label(f"• {d}").classes("text-sm ag-muted")
        except Exception as exc:
            notify_error(exc, "Prediction failed: ")
        finally:
            pbtn.props(remove="loading")

    pbtn.on_click(predict)

    # ------------------------------------------------------------ explanation
    with card("w-full"):
        ui.label("AI explanation").classes("ag-section-title")
        ui.label("Gemini (or your model in LLM Integration) reads every scope's numbers and explains the main mistakes "
                 "and what to do next time. Only numbers are sent, never expressions.").classes("ag-muted text-sm")
        with ui.row().classes("items-center gap-3 mt-2"):
            lang_toggle(lang["value"], set_lang)
            ebtn = ui.button("Explain my mistakes", icon="auto_awesome")
        exp_box = ui.column().classes("w-full gap-1")

    async def explain() -> None:
        if not scopes:
            ui.notify("No points on BRAIN yet (or not read). Press Refresh data first.", type="warning")
            return
        ebtn.props("loading")
        try:
            res = await portfolio.ai_explain(scopes, lang["value"])
            exp_box.clear()
            with exp_box:
                ui.label(res.summary).classes("text-sm")
                for m in res.mistakes:
                    with ui.column().classes("ag-tile w-full gap-0"):
                        ui.label(f"{m.scope}: {m.issue}").classes("text-sm ag-amber")
                        ui.label(f"Fix: {m.fix}").classes("text-sm")
                if res.next_steps:
                    ui.label("Next time").classes("font-semibold text-sm mt-1")
                    for s in res.next_steps:
                        ui.label(f"• {s}").classes("text-sm ag-muted")
        except Exception as exc:
            notify_error(exc, "Explanation failed: ")
        finally:
            ebtn.props(remove="loading")

    ebtn.on_click(explain)

    # ------------------------------------------------------------ per scope
    with card("w-full"):
        ui.label("Scopes on BRAIN").classes("ag-section-title")
        if not scopes:
            ui.label("No points on BRAIN yet, or they have not been read. Press Refresh data.").classes(
                "ag-muted text-sm")
        for s in scopes:
            with ui.column().classes("ag-tile w-full gap-1"):
                with ui.row().classes("items-center gap-6 w-full"):
                    ui.label(s.key).classes("ag-mono text-lg")
                    for label, value in (("Alphas", str(s.alphas)), ("Points", f"{s.points:,}"),
                                         ("Combo Sharpe", fmt_num(s.combo_sharpe)), ("Last 2 years", fmt_num(s.combo_2y)),
                                         ("Est. OS Sharpe", fmt_num(s.est_os)), ("Avg corr", fmt_num(s.avg_corr))):
                        with ui.column().classes("gap-0"):
                            ui.label(label).classes("ag-muted text-xs")
                            ui.label(value).classes("ag-mono ag-teal")
                for m in s.big:
                    ui.label(f"Big: {m}").classes("ag-red text-sm")
                for m in s.some:
                    ui.label(f"Some: {m}").classes("ag-amber text-sm")
                if not s.big and not s.some:
                    ui.label("No mistakes found.").classes("ag-green text-sm")

    # ------------------------------------------------------------ history (v1)
    async with db.session() as ss:
        allocs = list(await ss.scalars(select(OsmosisAllocation).order_by(OsmosisAllocation.id.desc()).limit(3000)))
        ids = list({a.alpha_id for a in allocs})
        alphas = {a.id: a for a in await ss.scalars(select(Alpha).where(Alpha.id.in_(ids)))} if ids else {}
        snaps = list(await ss.scalars(select(ConsultantSnapshot).order_by(ConsultantSnapshot.id)))

    with ui.expansion("Combined Osmosis performance and allocation history", icon="history").classes("w-full"):
        pts = [(s.ts.strftime("%m-%d"), s.payload.get("combinedOsmosisPerformance")) for s in snaps
               if isinstance(s.payload, dict) and s.payload.get("combinedOsmosisPerformance") is not None]
        if pts:
            ui.echart({"backgroundColor": "transparent", "tooltip": {"trigger": "axis"},
                       "grid": {"left": 50, "right": 20, "top": 20, "bottom": 30},
                       "xAxis": {"type": "category", "data": [p[0] for p in pts], "axisLabel": {"color": "#8b98ad"}},
                       "yAxis": {"type": "value", "axisLabel": {"color": "#8b98ad"},
                                 "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
                       "series": [{"type": "line", "data": [p[1] for p in pts], "smooth": True}]}).classes("w-full h-56")
        else:
            ui.label("Combined Osmosis performance appears here after a few 'Pyramids & Genius' syncs.").classes(
                "ag-muted text-sm")

        async def fetch_summary() -> None:
            try:
                data = await brain.users.osmosis_summary(brain.client.user_id or "self")
                with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[800px]"):
                    ui.label("BRAIN Osmosis summary").classes("font-semibold")
                    ui.json_editor({"content": {"json": data}}).classes("w-full")
                d.open()
            except BrainError as exc:
                notify_error(exc, "Osmosis summary unavailable: ")

        ui.button("Fetch Osmosis summary from BRAIN", icon="cloud_download", on_click=fetch_summary).props("flat")
        by_week: dict[tuple[str, str, int], list[OsmosisAllocation]] = defaultdict(list)
        for a in allocs:
            by_week[(a.week, a.region, a.delay)].append(a)
        rows = []
        for (week, region, delay), items in sorted(by_week.items(), reverse=True):
            perf = [((alphas[i.alpha_id].os_sharpe if i.alpha_id in alphas else None), i.points) for i in items]
            known = [(s_, p_) for s_, p_ in perf if s_ is not None]
            rows.append({"week": week, "pool": f"{region}/D{delay}", "alphas": len(items),
                         "points": sum(i.points for i in items), "method": items[0].method, "status": items[0].status,
                         "os_weighted": sum(s_ * p_ for s_, p_ in known) / sum(p_ for _, p_ in known) if known else None})
        ui.aggrid(grid_options([
            {"field": "week", "headerName": "Week", "width": 110}, {"field": "pool", "headerName": "Scope", "width": 110},
            {"field": "alphas", "headerName": "Alphas", "width": 90},
            {"field": "points", "headerName": "Points", "width": 110},
            {"field": "method", "headerName": "Method", "width": 140},
            {"field": "status", "headerName": "Status", "width": 100},
            num_col("os_weighted", "Points-weighted BRAIN OS Sharpe", 2, 250)], rows), theme="quartz").classes(
            "w-full h-[300px]")
