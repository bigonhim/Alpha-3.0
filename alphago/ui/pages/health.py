"""Health — how the AI providers and BRAIN behaved over time, which repairs work, how good the estimates are."""

from __future__ import annotations

from nicegui import ui

from ...engine import health
from ...engine.similarity import calibration
from ...llm import llm
from ..components import card, grid_options, info_box, num_col, page_header

SURFACE = "#0d1422"
# validated (dataviz validator, dark, surface #0d1422): categorical slots 1-5 in fixed order
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
LLM_KINDS = [("rate_limited", "Rate-limited"), ("timeout", "Timed out"), ("model_unavailable", "Model unavailable"),
             ("server_error", "Server error"), ("other", "Other")]
BRAIN_KINDS = [("throttled", "Throttled (429)"), ("server_errors", "Server errors (5xx)"),
               ("network_errors", "Network errors")]


def stacked_bars(labels: list[str], series: list[tuple[str, list[int]]]) -> dict:
    return {
        "backgroundColor": "transparent",
        "color": SERIES,
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"top": 0, "textStyle": {"color": "#8b98ad"}},
        "grid": {"left": 44, "right": 16, "top": 36, "bottom": 28},
        "xAxis": {"type": "category", "data": [d[5:] for d in labels], "axisLabel": {"color": "#8b98ad"},
                  "axisLine": {"lineStyle": {"color": "#2a3b57"}}},
        "yAxis": {"type": "value", "minInterval": 1, "axisLabel": {"color": "#8b98ad"},
                  "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
        "series": [{"name": name, "type": "bar", "stack": "all", "data": data, "barMaxWidth": 22,
                    "itemStyle": {"borderColor": SURFACE, "borderWidth": 1, "borderRadius": 2}}
                   for name, data in series],
    }


async def page() -> None:
    page_header("Health", "Rate limits, timeouts, unavailable models and quota use over time — tune the settings "
                          "with evidence. Also: which repairs actually work, and how good the self-correlation "
                          "estimate has been.")
    days = 14
    llm_days = await health.llm_by_day(days)
    labels = llm_days["labels"]

    fo = llm.last_failover
    if fo:
        import datetime as _dt
        ui.label(f"Last failover: {fo['from']} → {fo['to']} at {_dt.datetime.fromtimestamp(fo['at']):%Y-%m-%d %H:%M} "
                 f"({fo['error'][:160]})").classes("ag-warn-box text-sm w-full")

    with card("w-full"):
        ui.label(f"AI provider errors per day (last {days} days)").classes("ag-section-title")
        totals = {k: [0] * days for k, _ in LLM_KINDS}
        idx = {d: i for i, d in enumerate(labels)}
        for r in llm_days["rows"]:
            i = idx.get(r["day"])
            if i is None:
                continue
            for k, _ in LLM_KINDS:
                totals[k][i] += r.get(k, 0)
            totals["other"][i] += r.get("auth", 0) + r.get("budget", 0) + r.get("bad_json", 0)
        if any(any(v) for v in totals.values()):
            ui.echart(stacked_bars(labels, [(name, totals[k]) for k, name in LLM_KINDS])).classes("w-full h-64")
        else:
            ui.label("No AI errors recorded in this period.").classes("ag-muted text-sm")
        ui.aggrid(grid_options([
            {"field": "day", "headerName": "Day", "width": 110},
            {"field": "provider", "headerName": "Provider", "width": 110},
            {"field": "calls", "headerName": "Calls", "width": 85, "type": "numericColumn"},
            {"field": "ok", "headerName": "OK", "width": 80, "type": "numericColumn"},
            {"field": "rate_limited", "headerName": "Rate-limited", "width": 115, "type": "numericColumn"},
            {"field": "timeout", "headerName": "Timeouts", "width": 100, "type": "numericColumn"},
            {"field": "model_unavailable", "headerName": "Model unavail.", "width": 125, "type": "numericColumn"},
            {"field": "server_error", "headerName": "Server err.", "width": 105, "type": "numericColumn"},
            {"field": "budget", "headerName": "Budget", "width": 90, "type": "numericColumn"},
            num_col("avg_latency_s", "Avg latency (s)", 1, 130),
            {"field": "tokens", "headerName": "Tokens", "width": 110, "type": "numericColumn"},
        ], sorted(llm_days["rows"], key=lambda r: r["day"], reverse=True)), theme="quartz").classes("w-full h-64")

    with card("w-full"):
        ui.label("Models (last 14 days)").classes("ag-section-title")
        ui.aggrid(grid_options([
            {"field": "provider", "headerName": "Provider", "width": 110},
            {"field": "model", "headerName": "Model", "flex": 1, "minWidth": 220},
            {"field": "calls", "headerName": "Calls", "width": 90, "type": "numericColumn"},
            {"field": "errors", "headerName": "Errors", "width": 90, "type": "numericColumn"},
            num_col("error_rate", "Error rate", 1, 110, pct=True),
            num_col("avg_latency_s", "Avg latency (s)", 1, 130),
        ], await health.llm_by_model(days)), theme="quartz").classes("w-full h-56")

    brain_rows = await health.brain_by_day(days)
    with card("w-full"):
        ui.label(f"BRAIN per day (last {days} days)").classes("ag-section-title")
        ui.label("Counts start when this version first runs. Throttled = BRAIN answered 429 (the app waits and "
                 "retries); concurrency rejections are 429s on starting a simulation.").classes("ag-muted text-xs")
        series = [(name, [r[k] for r in brain_rows]) for k, name in BRAIN_KINDS]
        if any(any(v) for _, v in series):
            ui.echart(stacked_bars([r["day"] for r in brain_rows], series)).classes("w-full h-56")
        ui.aggrid(grid_options([
            {"field": "day", "headerName": "Day", "width": 110},
            {"field": "simulated", "headerName": "Simulated", "width": 105, "type": "numericColumn"},
            {"field": "sim_errors", "headerName": "Sim errors", "width": 105, "type": "numericColumn"},
            {"field": "requests", "headerName": "Requests", "width": 100, "type": "numericColumn"},
            {"field": "throttled", "headerName": "429s", "width": 80, "type": "numericColumn"},
            {"field": "concurrency_rejections", "headerName": "Concurrency", "width": 115, "type": "numericColumn"},
            {"field": "server_errors", "headerName": "5xx", "width": 75, "type": "numericColumn"},
            {"field": "network_errors", "headerName": "Network", "width": 95, "type": "numericColumn"},
            {"field": "quota_limit", "headerName": "Quota", "width": 90, "type": "numericColumn"},
            {"field": "quota_remaining", "headerName": "Left", "width": 85, "type": "numericColumn"},
            {"field": "sent_by_app", "headerName": "Sent by app", "width": 110, "type": "numericColumn"},
        ], list(reversed(brain_rows))), theme="quartz").classes("w-full h-64")

    with card("w-full"):
        ui.label("Which repairs work").classes("ag-section-title")
        ui.label("Every repair the optimiser tried, per failing check: how often it passed the check or closed at "
                 "least half the gap without breaking anything else. The repair loop orders its fixes by these "
                 "numbers (prior = the starting guess before there is data).").classes("ag-muted text-xs")
        rows = await health.repair_table()
        if rows:
            ui.aggrid(grid_options([
                {"field": "target", "headerName": "Failing check", "width": 200},
                {"field": "fix", "headerName": "Fix", "width": 170},
                {"field": "tried", "headerName": "Tried", "width": 85, "type": "numericColumn"},
                {"field": "worked", "headerName": "Worked", "width": 90, "type": "numericColumn"},
                num_col("rate", "Success", 0, 100, pct=True),
                num_col("prior", "Prior", 0, 90, pct=True),
            ], rows), theme="quartz").classes("w-full h-64")
        else:
            ui.label("No repair outcomes yet.").classes("ag-muted text-sm")

    cal = await calibration()
    with card("w-full"):
        ui.label("Self-correlation estimate vs BRAIN").classes("ag-section-title")
        if cal["n"]:
            ui.label(f"{cal['n']} BRAIN checks compared · mean error {cal['mae']:.3f} · bias {cal['bias']:+.3f} "
                     "(negative = the estimate reads low)").classes("text-sm")
            ui.aggrid(grid_options([
                {"field": "alpha_id", "headerName": "Alpha", "width": 110},
                num_col("estimate", "Estimate", 3, 100), num_col("real", "BRAIN", 3, 90),
                {"field": "closest", "headerName": "Closest submitted", "width": 150},
                num_col("structural", "Structural", 2, 110),
                {"field": "ts", "headerName": "When", "width": 170},
            ], list(reversed(cal["rows"]))), theme="quartz").classes("w-full h-56")
        else:
            info_box("Not measured yet", "The estimate is compared with BRAIN's SELF_CORRELATION each time an alpha "
                                         "that had an estimate is checked. It needs your submitted alphas' daily PnL "
                                         "(Osmosis → Prepare data).")
