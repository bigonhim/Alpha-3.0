"""Reusable UI building blocks (cards, KPI tiles, badges, formatting helpers)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from nicegui import ui


def fmt_num(value: Any, digits: int = 2, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value: Any, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: Any, digits: int = 1, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def fmt_compact(value: Any) -> str:
    if value is None:
        return "—"
    v = float(value)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v / div:.1f}{unit}"
    return f"{int(v):,}" if v == int(v) else f"{v:.2f}"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "0m"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m = rem // 60
    return f"{h}h {m}m" if h else f"{m}m"


def ago(ts: datetime | None) -> str:
    if ts is None:
        return "never"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    secs = (datetime.now(timezone.utc) - ts).total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return f"{int(secs // 86400)} d ago"


def page_header(title: str, subtitle: str = "") -> None:
    with ui.column().classes("gap-0 mb-2"):
        ui.label(title).classes("text-2xl font-semibold")
        if subtitle:
            ui.label(subtitle).classes("ag-muted text-sm")


def card(classes: str = "") -> ui.card:
    return ui.card().classes(f"ag-card p-5 {classes}")


def kpi_tile(icon: str, label: str, color: str = "ag-teal", link: tuple[str, Callable] | None = None):
    """Returns (value_label, sub_label) to update later."""
    with ui.card().classes("ag-card p-5 grow min-w-[220px]"):
        with ui.row().classes("items-center w-full justify-between no-wrap"):
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.icon(icon).classes(f"{color} text-xl")
                ui.label(label).classes("text-[15px]")
            if link:
                ui.label(link[0]).classes("ag-link").on("click", link[1])
        value = ui.label("—").classes("ag-kpi mt-2")
        sub = ui.label("").classes("ag-muted text-sm")
    return value, sub


def mini_tile(icon: str, label: str, color: str = "ag-cyan"):
    with ui.card().classes("ag-tile grow min-w-[180px]"):
        with ui.row().classes("items-start gap-2 no-wrap min-h-[40px]"):
            ui.icon(icon).classes(f"{color} text-lg mt-0.5")
            ui.label(label).classes(f"{color} text-[15px] leading-tight")
        value = ui.label("—").classes("ag-kpi-sm mt-2 text-[#e6edf7]")
        sub = ui.label("").classes("ag-muted text-sm")
    return value, sub


def badge(text: str, kind: str = "") -> ui.label:
    return ui.label(text).classes(f"ag-badge {kind}")


def check_badge(check: dict[str, Any]) -> None:
    result = str(check.get("result", "")).upper()
    kind = {"PASS": "pass", "FAIL": "fail", "PENDING": "pending", "WARNING": "warning", "ERROR": "fail"}.get(result, "")
    value, limit = check.get("value"), check.get("limit")
    detail = f" {fmt_num(value, 3)} / {fmt_num(limit, 3)}" if value is not None and limit is not None else ""
    badge(f"{check.get('name')} · {result}{detail}", kind).tooltip(str(check.get("message") or ""))


def verdict_badge(verdict: str | None) -> None:
    kind = {"ready": "pass", "submitted": "pass", "strong": "pass", "promising": "pending", "weak": "warning",
            "rejected": "fail"}.get(verdict or "", "")
    badge((verdict or "new").upper(), kind)


def empty_state(icon: str, title: str, hint: str = "") -> None:
    with ui.column().classes("items-center w-full py-10 gap-1"):
        ui.icon(icon).classes("text-5xl ag-muted")
        ui.label(title).classes("text-lg")
        if hint:
            ui.label(hint).classes("ag-muted text-sm text-center max-w-xl")


def notify_error(exc: BaseException, prefix: str = "") -> None:
    message = getattr(exc, "message", None) or str(exc) or exc.__class__.__name__
    ui.notify(f"{prefix}{message}", type="negative", multi_line=True, close_button=True, timeout=8000)


def grid_options(columns: list[dict[str, Any]], rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    opts = {
        "columnDefs": columns,
        "rowData": rows,
        "defaultColDef": {"sortable": True, "filter": True, "resizable": True, "minWidth": 80},
        "rowSelection": {"mode": "multiRow", "checkboxes": True, "headerCheckbox": True} if extra.pop("multi", False)
        else {"mode": "singleRow", "checkboxes": False, "enableClickSelection": True},
        "animateRows": False,
        "pagination": True,
        "paginationPageSize": 50,
    }
    opts.update(extra)
    return opts


NUM_COL = {"type": "numericColumn", "cellClass": "ag-num"}


def num_col(field: str, header: str, digits: int = 2, width: int | None = None, pct: bool = False) -> dict[str, Any]:
    fmt = (f"value == null ? '' : (value*100).toFixed({digits}) + '%'" if pct
           else f"value == null ? '' : Number(value).toFixed({digits})")
    col = {"field": field, "headerName": header, ":valueFormatter": f"params => {{ const value = params.value; return {fmt}; }}",
           **NUM_COL}
    if width:
        col["width"] = width
    return col


def money_col(field: str, header: str, width: int | None = None) -> dict[str, Any]:
    """Dollar amounts shown compactly ($697.74K)."""
    js = ("params => { const v = params.value; if (v == null) return ''; const a = Math.abs(v); "
          "const s = a >= 1e6 ? (a/1e6).toFixed(2) + 'M' : a >= 1e3 ? (a/1e3).toFixed(2) + 'K' : a.toFixed(0); "
          "return (v < 0 ? '-$' : '$') + s; }")
    col = {"field": field, "headerName": header, ":valueFormatter": js, **NUM_COL}
    if width:
        col["width"] = width
    return col


# Inline-SVG cell renderers (AG Grid inserts a returned string as HTML).
_SPARK_JS = """params => {
  const v = params.value; if (!v || v.length < 2) return '';
  const w = 84, h = 22, lo = Math.min(...v), hi = Math.max(...v), span = (hi - lo) || 1;
  const pts = v.map((y, i) => (i * w / (v.length - 1)).toFixed(1) + ',' + (h - 2 - (y - lo) * (h - 4) / span).toFixed(1));
  const color = v[v.length - 1] >= v[0] ? '#4ade80' : '#f87171';
  return '<svg width="' + w + '" height="' + h + '" style="vertical-align:middle"><polyline fill="none" stroke="'
    + color + '" stroke-width="1.5" points="' + pts.join(' ') + '"/></svg>';
}"""

_BAR_JS = """params => {
  const v = params.value; if (v == null) return '';
  const pct = Math.max(0, Math.min(100, v));
  return '<div style="display:flex;align-items:center;gap:8px"><div style="flex:1;height:5px;border-radius:3px;'
    + 'background:#1e2b40"><div style="width:' + pct + '%;height:5px;border-radius:3px;background:#8b5cf6"></div></div>'
    + '<span style="font-family:var(--ag-mono);min-width:22px;text-align:right">' + Math.round(v) + '</span></div>';
}"""


def sparkline_col(field: str, header: str = "PnL", width: int = 110) -> dict[str, Any]:
    return {"field": field, "headerName": header, "width": width, ":cellRenderer": _SPARK_JS, "sortable": False,
            "filter": False, "cellDataType": False}


def bar_col(field: str, header: str, width: int = 140) -> dict[str, Any]:
    return {"field": field, "headerName": header, "width": width, ":cellRenderer": _BAR_JS, "type": "numericColumn"}


def stat_tile(label: str, color: str = "", icon: str | None = None, tooltip: str | None = None):
    """Compact tile (label above a mono value), as on the newer reference pages. Returns (value, sub)."""
    with ui.card().classes("ag-card p-4 grow min-w-[170px]") as c:
        with ui.row().classes("items-center gap-2 no-wrap"):
            if icon:
                ui.icon(icon).classes(f"{color or 'ag-cyan'} text-base")
            ui.label(label).classes(f"text-sm {color or 'ag-cyan'}")
        value = ui.label("—").classes(f"ag-kpi-sm mt-1 {color}")
        sub = ui.label("").classes("ag-muted text-xs")
    if tooltip:
        c.tooltip(tooltip)
    return value, sub


def segmented(options: dict[Any, str], value: Any, on_change: Callable[[Any], Any] | None = None) -> ui.toggle:
    """A compact segmented control ("Max 15 | Max 20 | Max 30")."""
    t = ui.toggle(options, value=value).props("no-caps unelevated dense toggle-color=primary").classes("ag-seg")
    if on_change:
        t.on_value_change(lambda e: on_change(e.value))
    return t


def info_box(title: str, text: str, icon: str = "info") -> None:
    with ui.card().classes("ag-card p-4 w-full"):
        with ui.row().classes("items-start gap-3 no-wrap"):
            ui.icon(icon).classes("ag-muted text-lg mt-0.5")
            with ui.column().classes("gap-1"):
                ui.label(title).classes("font-semibold text-sm")
                ui.label(text).classes("text-sm ag-muted")


def stage_bar(stages: list[str], current: str | None, done: bool = False) -> None:
    """Analyzing → Generating → Simulating → … with the reached stages highlighted."""
    idx = stages.index(current) if current in stages else (len(stages) - 1 if done else -1)
    with ui.row().classes("items-center gap-2 flex-wrap"):
        for i, st in enumerate(stages):
            kind = "pass" if i < idx or (done and i <= idx) else ("running" if i == idx else "")
            badge(st.title(), kind)
            if i < len(stages) - 1:
                ui.label("→").classes("ag-muted text-xs")


def lang_toggle(value: str, on_change: Callable[[str], Any]) -> ui.toggle:
    return segmented({"English": "English", "Hinglish": "Hinglish"}, value, on_change)
