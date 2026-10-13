"""Osmosis — plan 100,000 points per scope across your submitted alphas from their daily PnL, then send to BRAIN."""

from __future__ import annotations

import csv
import io
from datetime import datetime

from nicegui import ui
from sqlalchemy import select

from ...brain import brain
from ...config import settings
from ...db import db
from ...db.models import Alpha
from ...engine import pnl, portfolio
from ...engine.programs import next_osmosis_lock
from ...engine.sync import last_synced
from ...jobs.queue import jobs
from ..components import (ago, badge, card, fmt_int, fmt_num, grid_options, notify_error, num_col, page_header,
                          segmented, sparkline_col, stat_tile)

METHODS = {"risk": "Equal risk (default)", "score": "Quality-weighted", "kelly": "Half-Kelly (Chan)",
           "inverse_corr": "Inverse correlation", "equal": "Equal points"}

RATING_RENDER = """params => {
  const k = {very_good: 'pass', good: 'running', average: 'pending', weak: 'fail'}[params.data.rating] || '';
  return '<span class="ag-badge ' + k + '">' + params.value + '</span>';
}"""


def _local_lock() -> str:
    t = next_osmosis_lock().astimezone()
    return t.strftime("%a, %b %d, ") + t.strftime("%I:%M %p").lstrip("0")


async def page() -> None:
    p = settings.programs
    with ui.row().classes("w-full items-start no-wrap"):
        page_header("Osmosis", "Spread 100,000 points per region/delay over your submitted alphas. The plan picks the "
                               "alphas whose daily PnL combine best and leaves out weak ones.")
        ui.space()
        with ui.row().classes("gap-2 no-wrap"):
            ui.button("Prepare data", icon="sync", on_click=lambda: prepare()).props("outline")
            make_btn = ui.button("Make best plan", icon="play_arrow", on_click=lambda: make())

    with ui.row().classes("w-full gap-4 no-wrap"):
        t_sub, _ = stat_tile("Submitted alphas")
        t_pnl, _ = stat_tile("With daily PnL", icon="show_chart")
        t_lock, t_lock_sub = stat_tile("Next weekly lock (your time)", "ag-amber", icon="event")
        t_read, t_read_sub = stat_tile("Current points read from BRAIN", "ag-green", icon="check_circle")

    plan_box = ui.column().classes("w-full gap-4")

    with ui.expansion("Lock time, limits and weighting", icon="tune").classes("w-full"):
        with ui.row().classes("gap-3 items-end flex-wrap"):
            wd = ui.select({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"},
                           value=p.osmosis_lock_weekday, label="Lock day (New York)").classes("w-40")
            hr = ui.number("Hour (New York)", value=p.osmosis_lock_hour_et, min=0, max=23).classes("w-36")
            mi = ui.number("Minute", value=p.osmosis_lock_minute, min=0, max=59).classes("w-28")
            minp = ui.number("Min pts/alpha", value=p.osmosis_min_points, min=0).classes("w-32")
            maxp = ui.number("Max pts/alpha", value=p.osmosis_max_points, min=1).classes("w-32")
            mina = ui.number("Min alphas/scope", value=p.osmosis_min_alphas, min=1).classes("w-36")
            lo_s = ui.number("Leave out: last-2y Sharpe below", value=p.osmosis_leave_out_sharpe2y, step=0.1).classes(
                "w-56")
            lo_t = ui.number("… or turnover above", value=p.osmosis_leave_out_turnover, step=0.05).classes("w-44")
            method = ui.select(METHODS, value="risk", label="Weighting").classes("w-52")
        ui.label("BRAIN does not publish the lock time; the default (Sunday 23:59 New York) is inferred from the "
                 "reference app, which showed it as Monday 9:29 AM India time.").classes("ag-muted text-xs")

        def save_settings() -> None:
            p.osmosis_lock_weekday, p.osmosis_lock_hour_et = int(wd.value), int(hr.value or 0)
            p.osmosis_lock_minute = int(mi.value or 0)
            p.osmosis_min_points, p.osmosis_max_points = int(minp.value or 0), int(maxp.value or 15000)
            p.osmosis_min_alphas = int(mina.value or 10)
            p.osmosis_leave_out_sharpe2y, p.osmosis_leave_out_turnover = float(lo_s.value or 0), float(lo_t.value or 1)
            settings.save()
            t_lock.set_text(_local_lock())

        for el in (wd, hr, mi, minp, maxp, mina, lo_s, lo_t):
            el.on_value_change(lambda _: save_settings())

    async def refresh_tiles() -> None:
        have, total = await pnl.coverage()
        t_sub.set_text(fmt_int(total))
        t_pnl.set_text(f"{have} of {total}")
        t_lock.set_text(_local_lock())
        t_lock_sub.set_text("You can change points any time before the lock.")
        seen = await last_synced("active")
        fresh = seen is not None and (datetime.now(seen.tzinfo) - seen).total_seconds() < 12 * 3600
        t_read.set_text("Yes" if fresh else "No")
        t_read.classes(replace="ag-kpi-sm mt-1 " + ("ag-green" if fresh else "ag-amber"))
        t_read_sub.set_text(f"read {ago(seen)}" if seen else "press Prepare data")

    async def prepare() -> None:
        if not brain.client.logged_in:
            ui.notify("Sign in to BRAIN first.", type="warning")
            return
        await jobs.submit("prepare_pnl", "Prepare data (submitted alphas + daily PnL)")
        ui.notify("Reading your submitted alphas and their daily PnL — follow it on Tasks.")

    async def make(max_n: int | None = None) -> None:
        make_btn.props("loading")
        try:
            plan = await portfolio.make_plan(max_n or p.osmosis_plan_max, method.value)
            await render(plan)
        except Exception as exc:
            notify_error(exc, "Plan failed: ")
        finally:
            make_btn.props(remove="loading")

    async def render(plan: dict | None) -> None:
        plan_box.clear()
        have, total = await pnl.coverage()
        with plan_box:
            if not plan:
                with card("w-full"):
                    ui.label("No plan yet.").classes("ag-section-title")
                    ui.label("1) Prepare data reads your submitted alphas and their daily PnL from BRAIN. "
                             "2) Make best plan picks the alphas per scope. 3) Check it, then Send to BRAIN.").classes(
                        "ag-muted text-sm")
                return
            scopes = plan["scopes"]
            with card("w-full"):
                with ui.row().classes("items-center w-full gap-3"):
                    ui.icon("checklist").classes("ag-teal text-xl")
                    with ui.column().classes("gap-0"):
                        ui.label("Plan").classes("ag-section-title")
                        made = datetime.fromisoformat(plan["created"]).astimezone()
                        ui.label(f"Made {made:%d/%m/%Y, %H:%M:%S}. Points come from each alpha's in-sample daily PnL: "
                                 "a guide, not a promise of Osmosis rank.").classes("ag-muted text-xs")
                with ui.row().classes("items-center gap-2 mt-2"):
                    segmented({15: "Max 15", 20: "Max 20", 30: "Max 30", 40: "Max 40"}, plan.get("max_n", 30),
                              lambda v: (setattr(p, "osmosis_plan_max", int(v)), settings.save()))
                    ui.button("Update plan", on_click=lambda: make(p.osmosis_plan_max)).props("outline dense")
                    ui.button("CSV", icon="download", on_click=lambda: export_csv(plan)).props("outline dense")
                    ui.space()
                    ui.button("Send to BRAIN…", icon="send", on_click=lambda: confirm_send(plan)).props(
                        "color=positive")
                totals = ui.row().classes("w-full gap-10 mt-3")

                def draw_totals() -> None:
                    chosen = [s for s in scopes if s["include"]]
                    totals.clear()
                    with totals:
                        for label, value, color in (
                                ("Scopes chosen", len(chosen), "ag-teal"),
                                ("Alphas with points", sum(len(s["points"]) for s in chosen), "ag-teal"),
                                ("Points in total", f"{sum(sum(s['points'].values()) for s in chosen):,}", "ag-teal"),
                                ("Rules met", "Yes" if chosen and all(s["rules_ok"] for s in chosen) else "No",
                                 "ag-green" if chosen and all(s["rules_ok"] for s in chosen) else "ag-amber")):
                            with ui.column().classes("gap-0"):
                                ui.label(label).classes("ag-muted text-sm")
                                ui.label(str(value)).classes(f"ag-kpi-sm {color}")

                draw_totals()
                if have < total:
                    ui.label("Some alphas have no daily PnL yet, so they are left out. Press Prepare data.").classes(
                        "ag-warn-box text-sm w-full mt-2")
            for sp in scopes:
                scope_card(plan, sp, draw_totals)

    def scope_card(plan: dict, sp: dict, draw_totals) -> None:  # noqa: ANN001
        with card("w-full ag-card-violet" if sp["include"] else "w-full"):
            with ui.row().classes("items-center w-full gap-3 no-wrap"):
                inc = ui.checkbox(value=sp["include"])

                async def toggle(e, s=sp) -> None:  # noqa: ANN001
                    s["include"] = bool(e.value)
                    await db.kv_set("osmosis_plan", plan)
                    draw_totals()

                inc.on_value_change(toggle)
                ui.icon("public").classes("ag-cyan")
                ui.label(f"{sp['region']}/D{sp['delay']}").classes("text-lg ag-mono")
                badge("Recommended", "pass") if sp["recommended"] else badge("Check", "warning")
                ui.label(f"{len(sp['picked'])} picked of {sp['submitted']} submitted").classes("ag-muted text-sm")
                ui.space()
                for label, value in (("Combo Sharpe", fmt_num(sp["combo_sharpe"])),
                                     ("Last 2 years", fmt_num(sp["combo_2y"])),
                                     ("Best single", fmt_num(sp["best_single"])),
                                     ("Avg corr", fmt_num(sp["avg_corr"])),
                                     ("Datasets", str(sp["datasets"])),
                                     ("Now on BRAIN", f"{sp['now_points']:,} / {sp['now_alphas']}")):
                    with ui.column().classes("gap-0 items-end min-w-[90px]"):
                        ui.label(label).classes("ag-muted text-xs")
                        ui.label(value).classes("ag-mono ag-teal")
            weak = [x for x in sp["left_out"] if x["reason"] != "no daily PnL yet"]
            nopnl = len(sp["left_out"]) - len(weak)
            if weak:
                ui.label(f"Left out {len(weak)} weak alpha(s): last-2-year Sharpe below "
                         f"{p.osmosis_leave_out_sharpe2y} or turnover above {p.osmosis_leave_out_turnover:.0%}.").classes(
                    "ag-amber text-sm")
            if nopnl:
                ui.label(f"{nopnl} alpha(s) have no daily PnL yet.").classes("ag-amber text-sm")
            for note in sp["notes"]:
                ui.label(note).classes("ag-amber text-sm")
            with ui.row().classes("items-center gap-2"):
                ui.label("Alphas in plan:").classes("ag-muted text-sm")
                for r, kind in (("very_good", "pass"), ("good", "running"), ("average", ""), ("weak", "fail")):
                    badge(f"{portfolio.RATING_LABEL[r]} {sp['ratings'].get(r, 0)}", kind)
            chart_exp = ui.expansion("PnL chart", icon="show_chart").classes("w-full")
            alphas_exp = ui.expansion(f"Alphas: PnL and points ({len(sp['picked'])} in plan, "
                                      f"{sp['submitted'] - len(sp['picked'])} left out)", icon="trending_up").classes(
                "w-full")
            loaded = {"chart": False, "alphas": False}

            async def load_chart(e, s=sp) -> None:  # noqa: ANN001
                if not e.value or loaded["chart"]:
                    return
                loaded["chart"] = True
                data = await portfolio.scope_chart(s["region"], s["delay"], s["points"], s["best_single_id"])
                with chart_exp:
                    if not data["dates"]:
                        ui.label("No daily PnL to chart.").classes("ag-muted text-sm")
                        return
                    ui.echart({
                        "backgroundColor": "transparent", "tooltip": {"trigger": "axis"},
                        "legend": {"top": 0, "textStyle": {"color": "#8b98ad"}},
                        "grid": {"left": 70, "right": 20, "top": 36, "bottom": 30},
                        "xAxis": {"type": "category", "data": data["dates"], "axisLabel": {"color": "#8b98ad"}},
                        "yAxis": {"type": "value", "axisLabel": {"color": "#8b98ad"},
                                  "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
                        "series": [{"name": "Plan (combined)", "type": "line", "showSymbol": False,
                                    "data": data["combo"]},
                                   {"name": f"Best single ({s['best_single_id']})", "type": "line", "showSymbol": False,
                                    "data": data["best"]}]}).classes("w-full h-64")

            async def load_alphas(e, s=sp) -> None:  # noqa: ANN001
                if not e.value or loaded["alphas"]:
                    return
                loaded["alphas"] = True
                async with db.session() as ss:
                    alphas = list(await ss.scalars(select(Alpha).where(
                        Alpha.status == "ACTIVE", Alpha.type == "REGULAR", Alpha.region == s["region"],
                        Alpha.delay == s["delay"])))
                rows = await pnl.stored(a.id for a in alphas)
                left = {x["id"]: x["reason"] for x in s["left_out"]}
                data = []
                for a in alphas:
                    pr_ = rows.get(a.id)
                    r, why = portfolio.rate(a, pr_)
                    data.append({"id": a.id, "points": s["points"].get(a.id, 0), "in_plan": a.id in s["points"],
                                 "rating": r, "rating_label": portfolio.RATING_LABEL[r], "why": why,
                                 "sharpe": a.sharpe, "sharpe_2y": pr_.sharpe_2y if pr_ else None,
                                 "turnover": a.turnover, "spark": (pr_.spark if pr_ else None) or [],
                                 "now": a.osmosis_points or 0, "left": left.get(a.id, "")})
                data.sort(key=lambda r: (-r["points"], r["id"]))
                cols = [
                    {"field": "id", "headerName": "Alpha", "width": 110},
                    {"field": "points", "headerName": "Points (editable)", "width": 140, "editable": True,
                     "type": "numericColumn"},
                    {"field": "rating_label", "headerName": "Rating", "width": 110, ":cellRenderer": RATING_RENDER,
                     "tooltipField": "why"},
                    sparkline_col("spark"),
                    num_col("sharpe", "Sharpe", 2, 90), num_col("sharpe_2y", "Last 2y", 2, 90),
                    num_col("turnover", "Turnover", 0, 95, pct=True),
                    {"field": "now", "headerName": "Now on BRAIN", "width": 120, "type": "numericColumn"},
                    {"field": "left", "headerName": "Why left out", "flex": 1, "minWidth": 260, "tooltipField": "left"},
                ]
                with alphas_exp:
                    grid = ui.aggrid(grid_options(cols, data, tooltipShowDelay=300), theme="quartz").classes(
                        "w-full h-[380px]")
                    total_lbl = ui.label(f"Total {sum(s['points'].values()):,} points").classes("ag-cyan text-sm")

                async def edited(ev, s2=s) -> None:  # noqa: ANN001
                    row = ev.args.get("data") or {}
                    try:
                        pts = max(0, int(float(row.get("points") or 0)))
                    except (TypeError, ValueError):
                        pts = 0
                    if pts:
                        s2["points"][row["id"]] = pts
                    else:
                        s2["points"].pop(row["id"], None)
                    total_pts = sum(s2["points"].values())
                    s2["rules_ok"] = total_pts == p.osmosis_points_total and all(
                        p.osmosis_min_points <= v <= p.osmosis_max_points for v in s2["points"].values())
                    total_lbl.set_text(f"Total {total_pts:,} points" + ("" if total_pts == p.osmosis_points_total
                                                                         else f" (needs {p.osmosis_points_total:,})"))
                    await db.kv_set("osmosis_plan", plan)
                    draw_totals()

                grid.on("cellValueChanged", edited)

            chart_exp.on_value_change(load_chart)
            alphas_exp.on_value_change(load_alphas)

    def export_csv(plan: dict) -> None:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["scope", "alpha", "points", "included"])
        for s in plan["scopes"]:
            for aid, pts in s["points"].items():
                w.writerow([f"{s['region']}/D{s['delay']}", aid, pts, s["include"]])
        ui.download.content(buf.getvalue(), f"osmosis_plan_{datetime.now():%Y%m%d_%H%M}.csv")

    async def confirm_send(plan: dict) -> None:
        chosen = [s for s in plan["scopes"] if s["include"] and s["points"]]
        if not chosen:
            ui.notify("Tick at least one scope with points.", type="warning")
            return
        if not brain.client.logged_in:
            ui.notify("Sign in to BRAIN first.", type="warning")
            return
        with ui.dialog() as dialog, ui.card().classes("ag-card p-6 w-[600px]"):
            ui.label("Send these Osmosis points to BRAIN?").classes("text-lg font-semibold")
            ui.separator()
            for s in chosen:
                total = sum(s["points"].values())
                bad = total != p.osmosis_points_total
                ui.label(f"• {s['region']}/D{s['delay']}: {len(s['points'])} alphas, {total:,} points"
                         + (f"  ⚠ not {p.osmosis_points_total:,}" if bad else "")).classes(
                    "text-[15px] " + ("ag-red" if bad else ""))
            ui.label("Alphas in these scopes that are not in the plan get their points cleared. You can change points "
                     "again any time before the lock.").classes("ag-muted text-sm")

            async def go() -> None:
                for s in chosen:
                    await jobs.submit("osmosis_push", f"Osmosis {s['region']}/D{s['delay']} ({len(s['points'])} alphas)",
                                      {"region": s["region"], "delay": s["delay"], "allocation": s["points"],
                                       "method": f"plan:{plan.get('method', 'risk')}"})
                dialog.close()
                ui.notify("Osmosis updates queued — see Tasks.", type="positive")

            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Apply", icon="check", on_click=go).props("color=primary")
        dialog.open()

    await refresh_tiles()
    await render(await portfolio.saved_plan())
    ui.timer(30, refresh_tiles)
