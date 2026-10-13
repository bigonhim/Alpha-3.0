"""Simulation Matrix — expressions × settings grid, local validation, packed multi-simulations, live results."""

from __future__ import annotations

import itertools
import re
from typing import Any

from nicegui import ui
from sqlalchemy import select

from ...brain import brain
from ...config import settings as app_settings
from ...db import db
from ...db.models import Alpha, Batch, Candidate
from ...engine.catalogs import get_catalog
from ...engine.simulator import CandidateSpec, simulator
from ...events import bus
from ...fastexpr import validate
from ..alpha_detail import open_alpha
from ..components import card, fmt_int, grid_options, num_col, page_header

PLACEHOLDER = re.compile(r"\{(field|window|group|field2)\}")


def _parse_list(text: str, cast=str) -> list:  # noqa: ANN001
    out = []
    for part in re.split(r"[,\s]+", text or ""):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(cast(part))
        except ValueError:
            pass
    return out


def expand_template(template: str, fields: list[str], windows: list[int], groups: list[str]) -> list[str]:
    needed = set(PLACEHOLDER.findall(template))
    dims = {
        "field": fields if "field" in needed else [None],
        "field2": fields if "field2" in needed else [None],
        "window": windows if "window" in needed else [None],
        "group": groups if "group" in needed else [None],
    }
    out = []
    for f, f2, w, g in itertools.product(dims["field"], dims["field2"], dims["window"], dims["group"]):
        if f is not None and f2 is not None and f == f2:
            continue
        expr = template
        for key, val in (("field", f), ("field2", f2), ("window", w), ("group", g)):
            if val is not None:
                expr = expr.replace("{" + key + "}", str(val))
        out.append(expr)
    return list(dict.fromkeys(out))


RESULT_COLUMNS = [
    {"field": "status", "headerName": "Status", "width": 105},
    {"field": "verdict", "headerName": "Verdict", "width": 100},
    num_col("sharpe", "Sharpe", 2, 90),
    num_col("fitness", "Fitness", 2, 90),
    num_col("turnover", "Turnover", 1, 100, pct=True),
    num_col("returns", "Returns", 1, 95, pct=True),
    {"field": "universe", "headerName": "Universe", "width": 115},
    {"field": "neutralization", "headerName": "Neutralization", "width": 140},
    {"field": "decay", "headerName": "Decay", "width": 80},
    {"field": "truncation", "headerName": "Trunc", "width": 80},
    {"field": "alpha_id", "headerName": "Alpha", "width": 105},
    {"field": "expression", "headerName": "Expression", "flex": 1, "minWidth": 320, "cellClass": "ag-num"},
    {"field": "error", "headerName": "Error", "width": 260},
]


async def batch_rows(batch_id: int) -> list[dict[str, Any]]:
    async with db.session() as s:
        cands = list(await s.scalars(select(Candidate).where(Candidate.batch_id == batch_id).order_by(Candidate.id)))
        alpha_ids = [c.alpha_id for c in cands if c.alpha_id]
        alphas = {a.id: a for a in await s.scalars(select(Alpha).where(Alpha.id.in_(alpha_ids)))} if alpha_ids else {}
    rows = []
    for c in cands:
        a = alphas.get(c.alpha_id) if c.alpha_id else None
        st = c.settings or {}
        rows.append({
            "cid": c.id, "status": c.status, "verdict": a.verdict if a else "", "sharpe": a.sharpe if a else None,
            "fitness": a.fitness if a else None, "turnover": a.turnover if a else None,
            "returns": a.returns if a else None, "universe": st.get("universe"), "neutralization": st.get("neutralization"),
            "decay": st.get("decay"), "truncation": st.get("truncation"), "alpha_id": c.alpha_id or "",
            "expression": c.expression, "error": c.error or "",
        })
    return rows


async def page() -> None:
    page_header("Simulation Matrix",
                "Run expressions across a grid of settings. Expressions are validated locally first, then packed into "
                "multi-simulations of up to 10 and run under your concurrency slots and daily quota.")
    opts = brain.options
    r0 = app_settings.research

    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        with card("w-[56%]"):
            with ui.tabs().classes("w-full") as tabs:
                t_expr = ui.tab("Expressions")
                t_tmpl = ui.tab("Template")
            with ui.tab_panels(tabs, value=t_expr).classes("w-full"):
                with ui.tab_panel(t_expr):
                    exprs = ui.textarea("One expression per line (multi-statement allowed with ';')",
                                        value="rank(-ts_delta(close, 5))\n"
                                              "group_rank(ts_rank(operating_income/cap, 126), subindustry)").classes(
                        "w-full ag-mono").props("autogrow outlined input-style='min-height:160px'")
                with ui.tab_panel(t_tmpl):
                    template = ui.input("Template — placeholders {field} {field2} {window} {group}",
                                        value="group_rank(ts_rank({field}, {window}), {group})").classes("w-full ag-mono")
                    with ui.row().classes("w-full gap-3 items-end"):
                        field_search = ui.input("Search fields (synced catalogue)").classes("w-64")
                        fields_sel = ui.select([], multiple=True, label="Fields", with_input=True).classes(
                            "grow").props("use-chips")
                    with ui.row().classes("w-full gap-3"):
                        windows = ui.input("Windows", value="22, 66, 126, 252").classes("w-56")
                        groups = ui.input("Groups", value="subindustry, industry").classes("w-72")

        with card("grow"):
            ui.label("Settings matrix").classes("ag-section-title")
            with ui.row().classes("gap-3 items-end"):
                region = ui.select(opts.regions, value=r0.region if r0.region in opts.regions else opts.regions[0],
                                   label="Region").classes("w-28")
                delay = ui.select(opts.delays(region.value), value=r0.delay if r0.delay in opts.delays(region.value)
                                  else opts.delays(region.value)[0], label="Delay").classes("w-24")
            universes = ui.select(opts.universes(region.value), multiple=True, label="Universes",
                                  value=[r0.universe] if r0.universe in opts.universes(region.value)
                                  else opts.universes(region.value)[:1]).classes("w-full").props("use-chips")
            neuts = ui.select(opts.neutralizations(region.value), multiple=True, label="Neutralizations",
                              value=[n for n in ("SUBINDUSTRY", "INDUSTRY") if n in opts.neutralizations(region.value)]
                              ).classes("w-full").props("use-chips")
            with ui.row().classes("gap-3 w-full"):
                decays = ui.input("Decays", value="0, 4").classes("w-36")
                truncs = ui.input("Truncations", value="0.08").classes("w-36")
                nan = ui.select(["OFF", "ON"], value="OFF", label="NaN handling").classes("w-32")
                past = ui.select(["ON", "OFF"], value="ON", label="Pasteurize").classes("w-28")
            max_trade = None
            if opts.has_setting("maxTrade"):
                max_trade = ui.select(["OFF", "ON"], value="OFF", label="Max trade").classes("w-28")

            def on_region(_=None) -> None:  # noqa: ANN001
                r = region.value
                delay.options = opts.delays(r)
                delay.value = delay.options[0]
                universes.options = opts.universes(r)
                universes.value = universes.options[:1]
                neuts.options = opts.neutralizations(r)
                neuts.value = [n for n in ("SUBINDUSTRY", "INDUSTRY") if n in neuts.options] or neuts.options[:1]
                for el in (delay, universes, neuts):
                    el.update()

            region.on_value_change(on_region)
            preview = ui.label().classes("ag-cyan text-sm mt-2")
            with ui.row().classes("gap-2 mt-1"):
                validate_btn = ui.button("Validate", icon="fact_check").props("flat")
                run_btn = ui.button("Run matrix", icon="play_arrow")

    issues_box = ui.column().classes("w-full gap-1")

    # ------------------------------------------------------------------ results
    with card("w-full"):
        with ui.row().classes("items-center w-full gap-3"):
            ui.label("Results").classes("ag-section-title")
            batch_sel = ui.select({}, label="Matrix batch").classes("w-80")
            batch_info = ui.label().classes("ag-muted text-sm")
            ui.space()
            pivot_x = ui.select(["neutralization", "decay", "truncation", "universe"], value="neutralization",
                                label="Heatmap X").classes("w-40")
            pivot_y = ui.select(["decay", "neutralization", "truncation", "universe"], value="decay",
                                label="Heatmap Y").classes("w-40")
            pivot_m = ui.select(["sharpe", "fitness", "turnover"], value="sharpe", label="Metric").classes("w-32")
        grid = ui.aggrid(grid_options(RESULT_COLUMNS, []), theme="quartz").classes("w-full h-[460px]")
        heat_box = ui.column().classes("w-full")

    async def on_cell(e) -> None:  # noqa: ANN001
        aid = (e.args.get("data") or {}).get("alpha_id")
        if aid:
            await open_alpha(aid)

    grid.on("cellDoubleClicked", on_cell)

    def current_expressions() -> list[str]:
        if tabs.value in (t_tmpl, "Template"):
            return expand_template(template.value or "", list(fields_sel.value or []),
                                   _parse_list(windows.value, int), _parse_list(groups.value))
        return [line.strip() for line in (exprs.value or "").splitlines() if line.strip()]

    def settings_grid() -> list[dict[str, Any]]:
        grid_ = []
        for uni, neu, dec, tr in itertools.product(universes.value or [], neuts.value or [],
                                                   _parse_list(decays.value, int) or [0],
                                                   _parse_list(truncs.value, float) or [0.08]):
            s = {"region": region.value, "delay": int(delay.value), "universe": uni, "neutralization": neu,
                 "decay": dec, "truncation": tr, "nanHandling": nan.value, "pasteurization": past.value}
            if max_trade is not None:
                s["maxTrade"] = max_trade.value
            grid_.append(s)
        return grid_

    def update_preview() -> None:
        n_expr, n_set = len(current_expressions()), len(settings_grid())
        total = n_expr * n_set
        remaining = brain.client.quota.effective_remaining
        preview.set_text(f"{n_expr} expression(s) × {n_set} setting combination(s) = {fmt_int(total)} simulations "
                         f"· {fmt_int(remaining)} left today")
        preview.classes(replace="text-sm mt-2 " + ("ag-red" if total > remaining else "ag-cyan"))

    ui.timer(1.0, update_preview)

    async def search_fields() -> None:
        cat = await get_catalog(region.value, int(delay.value), (universes.value or ["TOP3000"])[0])
        term = (field_search.value or "").lower()
        matches = [fid for fid, fi in cat.fields.items()
                   if fi.type == "MATRIX" and (term in fid.lower() or term in (fi.description or "").lower())][:400]
        fields_sel.options = matches
        fields_sel.update()
        if not cat.fields:
            ui.notify("No fields synced for this scope yet — use 'Sync with BRAIN' first.", type="warning")

    field_search.on("keydown.enter", lambda _: search_fields())

    async def do_validate() -> tuple[list[str], list[tuple[str, str]]]:
        exprs_ = current_expressions()
        good, bad = [], []
        sg = settings_grid()
        if not sg:
            return [], []
        s0 = sg[0]
        cat = await get_catalog(s0["region"], s0["delay"], s0["universe"])
        for e in exprs_:
            r = validate(e, cat)
            if r.ok:
                good.append(e)
            else:
                bad.append((e, "; ".join(i.message for i in r.errors)))
        issues_box.clear()
        with issues_box:
            if bad:
                with card("w-full"):
                    ui.label(f"{len(bad)} expression(s) failed local validation (no quota spent):").classes("ag-red")
                    for e, msg in bad[:30]:
                        ui.label(f"• {e}  →  {msg}").classes("ag-mono text-xs")
            elif exprs_:
                ui.label(f"✓ All {len(good)} expressions passed local validation"
                         + ("" if cat.strict_fields else " (fields not synced: identifiers checked loosely)")).classes(
                    "ag-green text-sm")
        return good, bad

    async def do_run() -> None:
        if not brain.client.logged_in:
            ui.notify("Sign in to BRAIN first (Sync with BRAIN).", type="warning")
            return
        good, bad = await do_validate()
        sg = settings_grid()
        specs = [CandidateSpec(expression=e, settings=s, origin="matrix") for e in good for s in sg]
        if not specs:
            ui.notify("Nothing valid to simulate.", type="warning")
            return
        if len(specs) > brain.client.quota.effective_remaining:
            ui.notify("This matrix exceeds today's remaining simulations.", type="negative")
            return
        async with db.write() as s:
            batch = Batch(source="matrix", region=region.value, delay=int(delay.value),
                          universe=(universes.value or [None])[0], strategy="matrix", status="running",
                          n_generated=len(specs))
            s.add(batch)
            await s.flush()
            bid = batch.id
        result = await simulator.submit(specs, batch_id=bid)
        async with db.write() as s:
            b = await s.get(Batch, bid)
            b.n_valid, b.n_reused = len(result.created) + len(result.reused), len(result.reused)
        bus.publish("simulate", f"Matrix #{bid}: queued {len(result.created)} simulations"
                                f" ({len(result.reused)} reused, {result.skipped} duplicates skipped).")
        ui.notify(f"Queued {len(result.created)} simulations ({len(result.reused)} reused from cache).",
                  type="positive")
        await load_batches(select_id=bid)

    validate_btn.on_click(do_validate)
    run_btn.on_click(do_run)

    async def load_batches(select_id: int | None = None) -> None:
        async with db.session() as s:
            rows = list(await s.scalars(select(Batch).where(Batch.source == "matrix").order_by(Batch.id.desc()).limit(40)))
        batch_sel.options = {b.id: f"#{b.id} · {b.created_at:%m-%d %H:%M} · {b.region} {b.universe} · {b.n_generated} sims"
                             for b in rows}
        batch_sel.update()
        if select_id or (rows and batch_sel.value is None):
            batch_sel.value = select_id or rows[0].id

    def render_heatmap(rows: list[dict[str, Any]]) -> None:
        heat_box.clear()
        done = [r for r in rows if r.get(pivot_m.value) is not None]
        if not done or pivot_x.value == pivot_y.value:
            return
        xs = sorted({str(r[pivot_x.value]) for r in done})
        ys = sorted({str(r[pivot_y.value]) for r in done})
        cells: dict[tuple[int, int], list[float]] = {}
        for r in done:
            cells.setdefault((xs.index(str(r[pivot_x.value])), ys.index(str(r[pivot_y.value]))), []).append(
                float(r[pivot_m.value]))
        data = [[x, y, round(sum(v) / len(v), 3)] for (x, y), v in cells.items()]
        vals = [d[2] for d in data]
        with heat_box:
            ui.label(f"Mean {pivot_m.value} by {pivot_x.value} × {pivot_y.value} — robust ideas show a plateau, "
                     "not a single spike").classes("ag-muted text-sm")
            ui.echart({
                "backgroundColor": "transparent",
                "tooltip": {"position": "top"},
                "grid": {"left": 110, "right": 30, "top": 10, "bottom": 60},
                "xAxis": {"type": "category", "data": xs, "axisLabel": {"color": "#8b98ad"}},
                "yAxis": {"type": "category", "data": ys, "axisLabel": {"color": "#8b98ad"}},
                "visualMap": {"min": min(vals), "max": max(vals), "calculable": True, "orient": "horizontal",
                              "left": "center", "bottom": 0, "textStyle": {"color": "#8b98ad"},
                              "inRange": {"color": ["#3b1d4f", "#6d28d9", "#2dd4bf", "#a7f3d0"]}},
                "series": [{"type": "heatmap", "data": data, "label": {"show": True, "color": "#fff"}}],
            }).classes("w-full h-72")

    state = {"rows": None}

    async def refresh_results() -> None:
        if batch_sel.value is None:
            return
        rows = await batch_rows(int(batch_sel.value))
        if rows != state["rows"]:
            state["rows"] = rows
            grid.options["rowData"] = rows
            grid.update()
            render_heatmap(rows)
        done = sum(r["status"] in ("done", "reused") for r in rows)
        failed = sum(r["status"] in ("error", "invalid") for r in rows)
        batch_info.set_text(f"{done}/{len(rows)} done · {failed} failed")

    batch_sel.on_value_change(lambda _: refresh_results())
    for el in (pivot_x, pivot_y, pivot_m):
        el.on_value_change(lambda _: render_heatmap(state["rows"] or []))
    await load_batches()
    await refresh_results()
    ui.timer(3.0, refresh_results)
