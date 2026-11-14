"""Portfolio (ported from alpha-harness): alphas combined at equal weight, as BRAIN combines its own pool.

Pick the alphas (your submitted ones by default, a tag group, or pasted ids), narrow them with a Selection
Expression, facet chips and the pyramid grid, then see the combined figures gross and net of a trading cost, the
yearly rows, the curves, the correlations and what each alpha adds."""

from __future__ import annotations

import math
from typing import Any

from nicegui import ui

from ...jobs.queue import jobs
from ...pool import groups as tag_groups
from ...pool import portfolio as pf
from ...pool import store
from ...pool.selection import FUNCTIONS, SelectionError
from ..components import card, fmt_int, fmt_num, fmt_pct, grid_options, num_col, page_header, stat_tile
from .pool import TONE_CLASS, job_line

SOURCES = {"submitted": "Submitted alphas", "group": "A tag group", "ids": "Alpha ids I paste"}


def _stats_block(title: str, block: dict[str, Any] | None) -> None:
    """One window's figures: Sharpe, Fitness, Returns, Turnover, Drawdown, Margin, days."""
    with ui.column().classes("gap-1 min-w-[180px]"):
        ui.label(title).classes("ag-section-title")
        if not block or block.get("days", 0) == 0:
            ui.label("No days in this window.").classes("ag-muted text-sm")
            return
        for label, value in (("Sharpe", fmt_num(block.get("sharpe"))), ("Fitness", fmt_num(block.get("fitness"))),
                             ("Returns", fmt_pct(block.get("returns"), 2)), ("Turnover", fmt_pct(block.get("turnover"), 2)),
                             ("Drawdown", fmt_pct(block.get("drawdown"), 2)),
                             ("Margin", "—" if block.get("margin") is None else f"{block['margin'] * 1e4:.2f} bps"),
                             ("Days", fmt_int(block.get("days")))):
            tone = ""
            if label == "Sharpe" and block.get("sharpe") is not None and block["sharpe"] < 0:
                tone = "ag-red"
            if label == "Returns" and block.get("returns") is not None:
                tone = "ag-green" if block["returns"] > 0 else "ag-red" if block["returns"] < 0 else ""
            with ui.row().classes("w-full justify-between gap-4 no-wrap"):
                ui.label(label).classes("ag-muted text-sm")
                ui.label(value).classes(f"ag-mono text-sm {tone}")


def _angle(rho: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, rho))))


MEMBER_COLUMNS = [
    {"field": "alpha_id", "headerName": "Alpha", "width": 110, "pinned": "left"},
    {"field": "region", "headerName": "Region", "width": 85},
    {"field": "delay", "headerName": "D", "width": 60},
    num_col("sharpe", "Sharpe", 2, 90), num_col("train_sharpe", "Train Sharpe", 2, 110),
    num_col("test_sharpe", "Test Sharpe", 2, 105), num_col("k_ratio", "K-Ratio", 2, 95),
    num_col("net_sharpe", "Net Sharpe", 2, 100), num_col("net_train_sharpe", "Net Train Sharpe", 2, 130),
    num_col("net_test_sharpe", "Net Test Sharpe", 2, 125), num_col("net_returns", "Net Returns", 2, 110, pct=True),
    num_col("max_pairwise", "Max Pairwise Corr", 4, 140), num_col("portfolio_corr", "Portfolio Corr", 4, 125),
    num_col("incremental_sharpe", "Incremental Sharpe", 4, 145),
]
for _col in MEMBER_COLUMNS:
    if _col.get("field") not in ("alpha_id", "region", "delay"):
        _col[":cellClass"] = TONE_CLASS


def _member_rows(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for m in members:
        tone = {k: store.figure_tone("sharpe", m.get(k), m.get("limits") or {})
                for k in ("sharpe", "train_sharpe", "test_sharpe", "net_sharpe", "net_train_sharpe", "net_test_sharpe")}
        tone["net_returns"] = store.figure_tone("returns", m.get("net_returns"), {})
        inc = m.get("incremental_sharpe")
        tone["incremental_sharpe"] = "" if inc is None or abs(inc) < 5e-5 else "pos" if inc > 0 else "neg"
        tone["max_pairwise"] = ("neg" if (m.get("max_pairwise") or 0) >= 0.7 else
                                "warn" if (m.get("max_pairwise") or 0) >= 0.5 else "pos") \
            if m.get("max_pairwise") is not None else ""
        rows.append({**{k: v for k, v in m.items() if k != "limits"}, "_tone": tone})
    return rows


def render_results(result: dict[str, Any], *, angles: bool = False) -> None:
    """Everything a computed portfolio shows (also used by the Alpha Pool's tag group analysis)."""
    if not result.get("alphas"):
        missing = len(result.get("missing") or [])
        ui.label("None of these alphas has a daily PnL and turnover stored yet." + (
            f" {missing} need downloading: press Download missing PnL, then compute again." if missing else ""
        )).classes("ag-muted")
        return
    bps = result.get("bps", 5.0)
    n = len(result["alphas"])
    with ui.row().classes("w-full gap-3 flex-wrap"):
        for label, value, sub in (
                ("Alphas combined", fmt_int(n), f"{len(result.get('missing') or [])} without a stored series"),
                ("Sharpe (IS)", fmt_num((result.get("is") or {}).get("sharpe")), "gross"),
                (f"Sharpe net of {bps:g} bps", fmt_num(((result.get("after_cost") or {}).get("is") or {}).get("sharpe")),
                 "each day pays its own turnover"),
                ("K-Ratio", fmt_num(result.get("k_ratio")), "steadiness of the cumulative PnL"),
                ("Measured pairs", fmt_int(result.get("measured_pairs")), "≥ 250 shared days, last 4 years")):
            v, s = stat_tile(label)
            v.set_text(value)
            s.set_text(sub)
    with card("w-full"):
        ui.label("In sample, train and test").classes("ag-section-title")
        split = result.get("split")
        ui.label(f"Test period starts {split}" if split else "No test period: these alphas were simulated without one.") \
            .classes("ag-muted text-xs")
        with ui.row().classes("gap-10 flex-wrap mt-2"):
            ac = result.get("after_cost") or {}
            _stats_block("In sample (gross)", result.get("is"))
            _stats_block(f"In sample (net {bps:g} bps)", ac.get("is"))
            if split:
                _stats_block("Train (gross)", result.get("train"))
                _stats_block("Test (gross)", result.get("test"))
                _stats_block(f"Test (net {bps:g} bps)", ac.get("test"))
    with card("w-full"):
        ui.label("Cumulative PnL").classes("ag-section-title")
        ui.echart({
            "backgroundColor": "transparent", "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Gross", f"Net of {bps:g} bps"], "textStyle": {"color": "#9fb0c8"}},
            "grid": {"left": 70, "right": 20, "top": 30, "bottom": 30},
            "xAxis": {"type": "category", "data": result["dates"], "axisLabel": {"color": "#9fb0c8"}},
            "yAxis": {"type": "value", "axisLabel": {"color": "#9fb0c8"}, "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
            "series": [
                {"name": "Gross", "type": "line", "showSymbol": False, "data": [round(x, 2) for x in result["curve"]],
                 "lineStyle": {"color": "#7dd3fc"}, "itemStyle": {"color": "#7dd3fc"}},
                {"name": f"Net of {bps:g} bps", "type": "line", "showSymbol": False,
                 "data": [round(x, 2) for x in result["after_cost_curve"]],
                 "lineStyle": {"color": "#c084fc"}, "itemStyle": {"color": "#c084fc"}},
            ],
        }).classes("w-full h-72")
    with card("w-full"):
        ui.label("Year by year (gross)").classes("ag-section-title")
        ui.aggrid(grid_options([
            {"field": "year", "headerName": "Year", "width": 80}, {"field": "stage", "headerName": "Stage", "width": 90},
            num_col("sharpe", "Sharpe", 2, 90), num_col("fitness", "Fitness", 2, 90),
            num_col("returns", "Returns", 2, 100, pct=True), num_col("turnover", "Turnover", 2, 100, pct=True),
            num_col("drawdown", "Drawdown", 2, 105, pct=True), {"field": "days", "headerName": "Days", "width": 80},
        ], [{k: r.get(k) for k in ("year", "stage", "sharpe", "fitness", "returns", "turnover", "drawdown", "days")}
            for r in result.get("yearly", [])], pagination=False), theme="quartz").classes("w-full h-64")
    with card("w-full"):
        ui.label("Correlation").classes("ag-section-title")
        hi, lo = result.get("highest"), result.get("lowest")

        def shown(v: float) -> str:
            return f"{_angle(v):.2f}°" if angles else f"{v:.4f}"
        if hi:
            ui.label(f"Most correlated pair: {hi[0]} and {hi[1]} at {shown(hi[2])}"
                     + ("  (at or above the 0.7 self-correlation ceiling)" if hi[2] >= 0.7 else "")).classes("text-sm")
        if lo:
            ui.label(f"Least correlated pair: {lo[0]} and {lo[1]} at {shown(lo[2])}").classes("text-sm ag-muted")
        if not hi:
            ui.label("No pair shares 250 days in the last four years, so nothing could be measured.").classes("ag-muted")
        ids = result["alphas"]
        if result.get("correlation") is not None and len(ids) > 1:
            cells = [[j, i, v] for i, row in enumerate(result["correlation"]) for j, v in enumerate(row) if v is not None]
            ui.echart({
                "backgroundColor": "transparent", "tooltip": {"position": "top"},
                "grid": {"left": 90, "right": 20, "top": 10, "bottom": 80},
                "xAxis": {"type": "category", "data": ids, "axisLabel": {"rotate": 60, "color": "#9fb0c8", "fontSize": 9}},
                "yAxis": {"type": "category", "data": ids, "axisLabel": {"color": "#9fb0c8", "fontSize": 9}},
                "visualMap": {"min": -1, "max": 1, "calculable": True, "orient": "horizontal", "left": "center",
                              "bottom": 0, "textStyle": {"color": "#9fb0c8"},
                              "inRange": {"color": ["#1e3a8a", "#0f172a", "#b45309"]}},
                "series": [{"type": "heatmap", "data": cells}],
            }).classes("w-full").style(f"height: {min(900, 160 + 14 * len(ids))}px")
        elif result.get("top_pairs"):
            ui.label(f"{len(ids)} alphas are too many for a grid: the 20 most correlated pairs.").classes("ag-muted text-xs")
            ui.aggrid(grid_options([{"field": "a", "headerName": "Alpha", "width": 120},
                                    {"field": "b", "headerName": "With", "width": 120},
                                    num_col("rho", "Correlation", 4, 120), num_col("deg", "Angle (°)", 2, 110)],
                                   [{"a": a, "b": b, "rho": v, "deg": _angle(v)} for a, b, v in result["top_pairs"]],
                                   pagination=False), theme="quartz").classes("w-full h-72")
    with card("w-full"):
        ui.label("What each alpha adds").classes("ag-section-title")
        ui.label(f"Net figures are after {bps:g} bps. Incremental Sharpe: the book's Sharpe with the alpha minus "
                 "without it. Portfolio Corr: against the equal-weight book of the others.").classes("ag-muted text-xs")
        rows = _member_rows(result.get("members") or [])
        if angles:
            for r in rows:
                for k in ("max_pairwise", "portfolio_corr"):
                    if r.get(k) is not None:
                        r[k] = _angle(r[k])
        ui.aggrid(grid_options(MEMBER_COLUMNS, rows), theme="quartz").classes("w-full h-[420px]")


async def page() -> None:
    page_header("Portfolio", "Your alphas as one equal-weight book: combined figures, after-cost, correlations and "
                             "per-alpha contribution.")
    state: dict[str, Any] = {"pool": [], "facets": {f: set() for f in pf.FACETS}, "pyramids": set(),
                             "members": []}
    with card("w-full"):
        with ui.row().classes("items-end gap-3 w-full flex-wrap"):
            source = ui.select(SOURCES, value="submitted", label="Alphas").classes("w-52")
            group_pick = ui.select({}, label="Tag group").classes("w-56")
            ids_box = ui.textarea("Alpha ids (comma, space or new line)").classes("w-96").props("autogrow")
            bps = ui.number("Trading cost (bps)", value=5.0, min=0, max=100, step=0.5).classes("w-40")
            angles = ui.checkbox("Show correlation as angles")
            count = ui.label("").classes("ag-muted text-sm ml-auto")
        with ui.row().classes("gap-2 items-center"):
            load_btn = ui.button("Load alphas", icon="refresh").props("flat")
            compute_btn = ui.button("Compute portfolio", icon="calculate")
            sync_btn = ui.button("Sync submitted alphas", icon="cloud_sync").props("flat").tooltip(
                "Refresh your submitted alphas from BRAIN, then download the daily PnL and turnover any still lack")
            missing_btn = ui.button("Download missing PnL", icon="download").props("flat")
        job_line(["prepare_pnl", "fetch_series", "sync_alphas"])

    with card("w-full"):
        ui.label("Selection Expression").classes("ag-section-title")
        ui.label("Fast Expression over each alpha's metrics. Each statement ends at ';' or a new line; name = … names "
                 "a value; every other statement is a condition and an alpha is selected when it meets them all. "
                 "25% is 0.25, 5bps is 0.0005; a metric an alpha lacks fails any condition on it.").classes(
            "ag-muted text-xs")
        expr = ui.textarea(placeholder=pf.EXAMPLE).classes("w-full ag-mono").props("autogrow")
        error = ui.label("").classes("ag-red text-sm")
        ui.label("Metrics: " + ", ".join(pf.METRICS) + ".  Functions: " + ", ".join(FUNCTIONS) + ".").classes(
            "ag-muted text-xs")
        facet_row = ui.row().classes("w-full gap-3 flex-wrap mt-2")
        ui.label("Pyramids (a cell picks one, a heading every pyramid in its row or column)").classes(
            "ag-section-title mt-3")
        grid_box = ui.column().classes("w-full")
        selected = ui.label("").classes("ag-muted text-sm")

    results = ui.column().classes("w-full gap-3")

    async def fill_groups() -> None:
        found = await tag_groups.all_groups()
        group_pick.options = {g.id: f"{g.name} ({'all' if g.every else 'any'} of {', '.join(g.tags)})" for g in found}
        group_pick.update()

    def chosen_members() -> list[Any]:
        error.set_text("")
        chips = {f: set(v) for f, v in state["facets"].items()}
        if state["pyramids"]:
            chips["pyramids"] = chips.get("pyramids", set()) | state["pyramids"]
        try:
            out = pf.select_members(state["pool"], expr.value or "", chips)
        except SelectionError as exc:
            error.set_text(str(exc))
            out = []
        state["members"] = out
        selected.set_text(f"{len(out):,} of {len(state['pool']):,} alphas selected")
        return out

    def draw_filters() -> None:
        facet_row.clear()
        counts = pf.facet_counts(state["pool"])
        labels = {"classifications": "Classifications", "investability": "Investability", "region": "Region",
                  "delay": "Delay"}
        with facet_row:
            for facet, title in labels.items():
                options = {v: f"{v} ({n})" for v, n in counts[facet]}
                if not options:
                    continue
                sel = ui.select(options, multiple=True, label=title,
                                value=[v for v in state["facets"][facet] if v in options]).props(
                    "use-chips dense").classes("min-w-[180px]")

                def changed(e, facet=facet) -> None:  # noqa: ANN001
                    state["facets"][facet] = set(e.value or [])
                    chosen_members()
                sel.on_value_change(changed)
        draw_pyramids()

    def draw_pyramids() -> None:
        grid_box.clear()
        categories, markets, cells = pf.pyramid_grid(state["pool"])
        with grid_box:
            if not cells:
                ui.label("None of these alphas carries a pyramid yet (a sync reads them).").classes("ag-muted text-sm")
                return
            chosen = state["pyramids"]

            def toggle(pyramids: list[str]) -> None:
                on = not (pyramids and all(p in chosen for p in pyramids))
                for p in pyramids:
                    (chosen.add if on else chosen.discard)(p)
                draw_pyramids()
                chosen_members()

            def style(pyramids: list[str]) -> str:
                return "unelevated color=primary" if pyramids and all(p in chosen for p in pyramids) else "outline"
            with ui.element("table").classes("text-sm"):
                with ui.element("tr"):
                    with ui.element("th"):
                        ui.label("Category").classes("ag-muted")
                    for m in markets:
                        col = [cells[(c, m)][0] for c in categories if (c, m) in cells]
                        with ui.element("th"):
                            ui.button(m, on_click=lambda col=col: toggle(col)).props(f"dense no-caps {style(col)}")
                for c in categories:
                    row = [cells[(c, m)][0] for m in markets if (c, m) in cells]
                    with ui.element("tr"):
                        with ui.element("th"):
                            ui.button(c, on_click=lambda row=row: toggle(row)).props(f"dense no-caps {style(row)}")
                        for m in markets:
                            with ui.element("td"):
                                if (c, m) in cells:
                                    p, n = cells[(c, m)]
                                    ui.button(str(n), on_click=lambda p=p: toggle([p])).props(
                                        f"dense {style([p])}").tooltip(f"{p.replace('/', ' / ')}: {n} alphas")

    async def load() -> None:
        kind = source.value
        if kind == "submitted":
            pool = await store.submitted_members()
        elif kind == "group":
            found = {g.id: g for g in await tag_groups.all_groups()}
            g = found.get(group_pick.value)
            if g is None:
                ui.notify("Pick a tag group first (save one on the Alpha Pool page).", type="warning")
                return
            f = store.StoredFilter(tags=list(g.tags), tags_all=bool(g.every))
            ids = await store.matching_ids(f)
            f.submitted = True
            ids += await store.matching_ids(f)
            pool = list((await store.by_ids(ids)).values())
        else:
            import re
            ids = [t for t in re.split(r"[\s,;]+", ids_box.value or "") if t]
            found_alphas = await store.by_ids(ids)
            if unknown := [a for a in ids if a not in found_alphas]:
                ui.notify(f"{len(unknown)} id(s) are not stored here yet (sync them first): {', '.join(unknown[:5])}",
                          type="warning")
            pool = list(found_alphas.values())
        state["pool"] = pool
        count.set_text(f"{len(pool):,} alphas loaded")
        draw_filters()
        chosen_members()

    async def compute() -> None:
        members = chosen_members()
        if not members:
            ui.notify("No alphas are selected: load some, or loosen the Selection Expression.", type="warning")
            return
        if len(members) > store.GROUP_LIMIT:
            ui.notify(f"Taking the first {store.GROUP_LIMIT:,} of {len(members):,}.", type="warning")
            members = members[:store.GROUP_LIMIT]
        compute_btn.props("loading")
        try:
            result = await pf.compute([a.id for a in members], float(bps.value or 0))
        finally:
            compute_btn.props(remove="loading")
        state["result"] = result
        draw_results()

    def draw_results() -> None:
        results.clear()
        if "result" in state:
            with results:
                render_results(state["result"], angles=bool(angles.value))

    async def sync() -> None:
        await jobs.submit("prepare_pnl", "Sync submitted alphas and their daily PnL", {"sync": True})
        await jobs.submit("fetch_series", "Daily PnL and turnover of submitted alphas",
                          {"submitted": True, "turnover": True})
        ui.notify("Syncing your submitted alphas from BRAIN, then downloading what they lack (see Tasks).")

    async def download_missing() -> None:
        ids = [a.id for a in (state["members"] or state["pool"])]
        from ...engine import series
        todo = await series.missing(ids, turnover=True)
        if not todo:
            ui.notify("Every selected alpha already has its daily PnL and turnover stored.")
            return
        await jobs.submit("fetch_series", f"Daily PnL for {len(todo)} alphas", {"alpha_ids": todo, "turnover": True})
        ui.notify(f"Downloading the daily PnL and turnover of {len(todo)} alpha(s) (see Tasks).")

    def show_source() -> None:
        group_pick.set_visibility(source.value == "group")
        ids_box.set_visibility(source.value == "ids")

    source.on_value_change(lambda _: show_source())
    expr.on("blur", lambda _: chosen_members())
    expr.on("keydown.ctrl.enter", lambda _: chosen_members())
    angles.on_value_change(lambda _: draw_results())
    load_btn.on_click(load)
    compute_btn.on_click(compute)
    sync_btn.on_click(sync)
    missing_btn.on_click(download_missing)
    show_source()
    await fill_groups()
    await load()

