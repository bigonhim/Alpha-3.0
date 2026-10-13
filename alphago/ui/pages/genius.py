"""Genius AI — quarterly Genius-level progress, pyramid map, pyramid candidates and AI "Genius signals"."""

from __future__ import annotations

from nicegui import ui

from ...brain import brain
from ...config import settings
from ...db import db
from ...engine.programs import genius_plan, genius_status, pyramid_candidates, pyramid_cells
from ...jobs.queue import jobs
from ...llm import LLMError
from ..alpha_detail import open_alpha
from ..components import card, fmt_num, grid_options, notify_error, num_col, page_header


def _progress(label: str, value: float | None, target: float | None, fmt=lambda v: f"{v:g}") -> None:  # noqa: ANN001
    v, t = value or 0, target or 0
    ratio = min(1.0, v / t) if t else 0.0
    with ui.column().classes("gap-1 grow min-w-[220px]"):
        with ui.row().classes("justify-between w-full"):
            ui.label(label).classes("text-sm")
            ui.label(f"{fmt(v) if value is not None else '—'} / {fmt(t)}").classes("ag-mono text-sm")
        ui.linear_progress(value=ratio, show_value=False).props(
            f"color={'positive' if ratio >= 1 else 'warning' if ratio >= 0.5 else 'negative'} rounded size=10px")


async def page() -> None:
    page_header("Genius AI", "Quarterly Genius level: signals (submissions), pyramids lit (≥ 3 alphas per region × delay × "
                             "category) and combined OS performance. Thresholds are community-reported and editable.")
    status_box = card("w-full")
    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        map_box = card("w-[58%]")
        cand_box = card("grow")
    plan_box = card("w-full")

    async def render_status() -> None:
        st = await genius_status()
        status_box.clear()
        with status_box:
            with ui.row().classes("items-center w-full gap-4"):
                ui.icon("workspace_premium").classes("text-3xl ag-amber")
                ui.label(f"{st['level'].title()}").classes("text-2xl font-semibold")
                ui.label(f"→ target {st['target'].title()} · {st['days_left']} days left this quarter").classes(
                    "ag-muted")
                ui.space()

                async def sync() -> None:
                    if not brain.client.logged_in:
                        ui.notify("Sign in to BRAIN first.", type="warning")
                        return
                    await jobs.submit("sync_programs", "Sync pyramids & Genius summary")
                    ui.notify("Sync queued — refresh in a moment.")

                ui.button("Sync from BRAIN", icon="sync", on_click=sync).props("flat")
                ui.button("Refresh", icon="refresh", on_click=refresh_all).props("flat")
            with ui.row().classes("w-full gap-8 mt-2"):
                _progress("Signals this quarter", st["signals"], st["signals_target"])
                _progress("Pyramids lit", st["pyramids"], st["pyramids_target"])
                _progress("Combined alpha performance", st["performance"], st["performance_target"],
                          lambda v: f"{v:.2f}")
            tb = st["tiebreak"]
            ui.label("Tie-breakers — avg operators/alpha {} (lower better) · avg fields/alpha {} (lower better) · "
                     "distinct operators {} · distinct fields {} · max simulation streak {}".format(
                         fmt_num(tb.get("operatorAvg")), fmt_num(tb.get("fieldAvg")), tb.get("operatorCount") or "—",
                         tb.get("fieldCount") or "—", tb.get("maxSimulationStreak") or "—")).classes("ag-muted text-sm")
            with ui.expansion("Edit level thresholds (signals, pyramids, performance)").classes("w-full"):
                th = settings.programs.genius_thresholds
                inputs = {}
                with ui.row().classes("gap-6"):
                    for lvl in ("EXPERT", "MASTER", "GRANDMASTER"):
                        with ui.column().classes("gap-0"):
                            ui.label(lvl.title()).classes("text-sm font-semibold")
                            inputs[lvl] = [ui.number(lab, value=v).classes("w-36")
                                           for lab, v in zip(("signals", "pyramids", "performance"), th[lvl])]

                def save_th() -> None:
                    for lvl, ins in inputs.items():
                        th[lvl] = [float(i.value or 0) for i in ins]
                    settings.save()
                    ui.notify("Thresholds saved.")

                ui.button("Save thresholds", on_click=save_th).props("flat")

    async def render_map() -> None:
        cells = await pyramid_cells()
        map_box.clear()
        with map_box:
            ui.label("Pyramid map (this quarter)").classes("ag-section-title")
            if not cells:
                ui.label("No pyramid data yet — click 'Sync from BRAIN'.").classes("ag-muted text-sm")
                return
            cols = sorted({f"{c.region}/D{c.delay}" for c in cells})
            rows = sorted({c.category_name or c.category_id for c in cells})
            data = [[cols.index(f"{c.region}/D{c.delay}"), rows.index(c.category_name or c.category_id),
                     c.alpha_count, c.multiplier or 0] for c in cells]
            ui.echart({
                "backgroundColor": "transparent",
                "tooltip": {":formatter": "p => `${p.value[2]} alphas · multiplier ${p.value[3]}`"},
                "grid": {"left": 170, "right": 20, "top": 10, "bottom": 70},
                "xAxis": {"type": "category", "data": cols, "axisLabel": {"color": "#8b98ad", "rotate": 30}},
                "yAxis": {"type": "category", "data": rows, "axisLabel": {"color": "#8b98ad"}},
                "visualMap": {"min": 0, "max": 3, "dimension": 2, "show": False,
                              "inRange": {"color": ["#3b1220", "#7c5a12", "#14532d"]}},
                "series": [{"type": "heatmap", "data": data,
                            "label": {"show": True, "color": "#fff", ":formatter": "p => p.value[2]"}}],
            }).classes("w-full").style(f"height:{max(260, 22 * len(rows) + 90)}px")
            ui.label("Green = lit (≥ 3 submissions). Lower utilisation pyramids carry higher multipliers.").classes(
                "ag-muted text-xs")

    async def render_candidates() -> None:
        rows = await pyramid_candidates()
        cand_box.clear()
        with cand_box:
            ui.label(f"Pyramid candidates ({len(rows)})").classes("ag-section-title")
            ui.label("Ready/strong unsubmitted alphas that would advance an unlit pyramid.").classes("ag-muted text-sm")
            grid = ui.aggrid(grid_options([
                {"field": "alpha_id", "headerName": "Alpha", "width": 110},
                {"field": "pyramid", "headerName": "Pyramid", "flex": 1},
                {"field": "current", "headerName": "Now", "width": 70},
                num_col("multiplier", "×", 2, 70), num_col("sharpe", "Sharpe", 2, 85),
                {"field": "verdict", "headerName": "Verdict", "width": 95}], rows), theme="quartz").classes(
                "w-full h-[360px]")

            async def open_(e) -> None:  # noqa: ANN001
                aid = (e.args.get("data") or {}).get("alpha_id")
                if aid:
                    await open_alpha(aid)

            grid.on("cellDoubleClicked", open_)

    async def render_plan() -> None:
        plan = await db.kv_get("genius_plan")
        plan_box.clear()
        with plan_box:
            with ui.row().classes("items-center w-full"):
                ui.label("Genius signals (AI plan)").classes("ag-section-title")
                ui.space()

                async def make_plan() -> None:
                    btn.props("loading")
                    try:
                        await genius_plan()
                        await render_plan()
                    except LLMError as exc:
                        notify_error(exc)
                    finally:
                        btn.props(remove="loading")

                btn = ui.button("Generate plan with Gemini", icon="auto_awesome", on_click=make_plan)
            if not plan:
                ui.label("No plan yet. The plan turns your gaps into concrete goals you can launch as targeted "
                         "Autopilot runs.").classes("ag-muted text-sm")
                return
            ui.label(plan.get("summary", "")).classes("text-sm")
            for i, sig in enumerate(plan.get("signals", [])):
                with ui.row().classes("ag-tile w-full items-center gap-3 no-wrap"):
                    done = ui.checkbox(value=sig.get("done", False))

                    async def toggle(e, idx=i) -> None:  # noqa: ANN001
                        p = await db.kv_get("genius_plan")
                        p["signals"][idx]["done"] = bool(e.value)
                        await db.kv_set("genius_plan", p)

                    done.on_value_change(toggle)
                    with ui.column().classes("gap-0 grow"):
                        ui.label(sig.get("title", "")).classes("font-semibold")
                        ui.label(f"{sig.get('why', '')} → {sig.get('action', '')}").classes("text-sm ag-muted")
                        meta = " · ".join(x for x in (sig.get("region"), f"D{sig.get('delay')}" if sig.get("region") else "",
                                                       sig.get("category"), sig.get("constraint")) if x)
                        if meta:
                            ui.label(meta).classes("text-xs ag-cyan")

                    def launch(s=sig) -> None:
                        r = settings.research
                        if s.get("region") and s["region"] in brain.options.regions:
                            r.region = s["region"]
                            r.delay = int(s.get("delay") or 1)
                            r.universe = brain.options.universes(r.region)[0]
                        r.dataset_strategy = "pyramid_gap"
                        settings.save()
                        ui.navigate.to("/autopilot")

                    ui.button("Launch run", icon="smart_toy", on_click=launch).props("flat")

    async def refresh_all() -> None:
        await render_status()
        await render_map()
        await render_candidates()

    await refresh_all()
    await render_plan()
