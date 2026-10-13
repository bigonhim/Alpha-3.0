"""Analytics — throughput, success rate, Sharpe distribution, failing checks, yield by generator / dataset."""

from __future__ import annotations

from nicegui import ui

from ...engine.stats import check_failure_counts, daily_series, sharpe_histogram, yield_by
from ..components import card, fmt_num, page_header

AXIS = {"axisLabel": {"color": "#8b98ad"}, "splitLine": {"lineStyle": {"color": "#1e2b40"}}}


def _chart(options: dict, height: str = "h-72") -> None:
    ui.echart({"backgroundColor": "transparent", "tooltip": {"trigger": "axis"},
               "grid": {"left": 60, "right": 30, "top": 30, "bottom": 40}, **options}).classes(f"w-full {height}")


async def page() -> None:
    page_header("Analytics", "Is the research process improving? Throughput, hit rate, where ideas fail, and which "
                             "generators and datasets pay off.")
    series = await daily_series(30)
    with ui.row().classes("w-full gap-5 no-wrap"):
        with card("w-1/2"):
            ui.label("Simulations per day (30 d)").classes("ag-section-title")
            _chart({"xAxis": {"type": "category", "data": series["labels"], **AXIS}, "yAxis": {"type": "value", **AXIS},
                    "series": [{"type": "bar", "data": series["simulations"], "itemStyle": {"color": "#8b5cf6"}}]})
        with card("grow"):
            ui.label("AI success rate (% promising or better)").classes("ag-section-title")
            _chart({"xAxis": {"type": "category", "data": series["labels"], **AXIS},
                    "yAxis": {"type": "value", "max": 100, **AXIS},
                    "series": [{"type": "line", "data": series["success_rate"], "connectNulls": True,
                                "itemStyle": {"color": "#5eead4"}}]})
    labels, counts = await sharpe_histogram()
    fails = await check_failure_counts()
    with ui.row().classes("w-full gap-5 no-wrap"):
        with card("w-1/2"):
            ui.label("Sharpe distribution (app alphas)").classes("ag-section-title")
            if not counts:
                ui.label("No simulated alphas yet.").classes("ag-muted text-sm")
            else:
                _chart({"xAxis": {"type": "category", "data": labels, **AXIS}, "yAxis": {"type": "value", **AXIS},
                        "series": [{"type": "bar", "data": counts, "itemStyle": {"color": "#7dd3fc"}}]})
        with card("grow"):
            ui.label("Most frequent failing checks").classes("ag-section-title")
            names = list(fails)[:12]
            if not names:
                ui.label("No failed checks recorded yet.").classes("ag-muted text-sm")
            else:
                _chart({"xAxis": {"type": "value", **AXIS},
                        "yAxis": {"type": "category", "data": names[::-1], "axisLabel": {"color": "#8b98ad"}},
                        "grid": {"left": 210, "right": 30, "top": 20, "bottom": 30},
                        "series": [{"type": "bar", "data": [fails[n] for n in names][::-1],
                                    "itemStyle": {"color": "#f472b6"}}]})
    with ui.row().classes("w-full gap-5 no-wrap"):
        for column, title in (("origin", "Yield by generator"), ("dataset", "Yield by dataset (top 20)")):
            rows = await yield_by(column)
            with card("w-1/2"):
                ui.label(title).classes("ag-section-title")
                ui.table(columns=[
                    {"name": "key", "label": column.title(), "field": "key", "align": "left"},
                    {"name": "n", "label": "Simulated", "field": "n"},
                    {"name": "good", "label": "Promising+", "field": "good"},
                    {"name": "rate", "label": "Rate", "field": "rate"},
                    {"name": "avg", "label": "Avg Sharpe", "field": "avg"}],
                    rows=[{**r, "rate": f"{r['rate'] * 100:.1f}%", "avg": fmt_num(r["avg_sharpe"])} for r in rows]
                ).props("dense flat").classes("w-full")
    with card("w-full"):
        ui.label("Gemini calls per day").classes("ag-section-title")
        _chart({"xAxis": {"type": "category", "data": series["labels"], **AXIS}, "yAxis": {"type": "value", **AXIS},
                "series": [{"type": "bar", "data": series["llm_calls"], "itemStyle": {"color": "#c084fc"}}]}, "h-56")
