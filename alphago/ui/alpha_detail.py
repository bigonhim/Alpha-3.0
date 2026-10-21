"""Alpha detail dialog (shared by Alphas, Simulation Matrix, Autopilot, Power Pool ...) and the submit confirmation."""

from __future__ import annotations

from typing import Any, Callable

from nicegui import ui

from ..brain.errors import BrainError
from ..db import db
from ..db.models import Alpha, Candidate
from ..engine import actions
from ..engine.evaluator import evaluate
from ..jobs.queue import jobs
from .components import (badge, check_badge, fmt_int, fmt_num, fmt_pct, notify_error, verdict_badge)

BRAIN_ALPHA_URL = "https://platform.worldquantbrain.com/alpha/{}"


def _metric(label: str, value: str, color: str = "") -> None:
    with ui.column().classes("gap-0 min-w-[92px]"):
        ui.label(label).classes("ag-muted text-xs")
        ui.label(value).classes(f"ag-mono text-lg {color}")


def metrics_row(a: Alpha) -> None:
    ev = evaluate(sharpe=a.sharpe, fitness=a.fitness, turnover=a.turnover, checks=a.checks, delay=a.delay,
                  region=a.region, status=a.status)
    sc = "ag-green" if (a.sharpe or 0) >= ev.sharpe_limit else ("ag-amber" if (a.sharpe or 0) > 0 else "ag-red")
    fc = "ag-green" if (a.fitness or 0) >= ev.fitness_limit else "ag-amber"
    with ui.row().classes("gap-6 flex-wrap"):
        _metric("Sharpe", fmt_num(a.sharpe), sc)
        _metric("Fitness", fmt_num(a.fitness), fc)
        _metric("Turnover", fmt_pct(a.turnover))
        _metric("Returns", fmt_pct(a.returns))
        _metric("Drawdown", fmt_pct(a.drawdown))
        _metric("Margin", f"{fmt_num((a.margin or 0) * 10000, 1)} bps" if a.margin is not None else "—")
        _metric("Long / Short", f"{fmt_int(a.long_count)} / {fmt_int(a.short_count)}")
        _metric("Self corr", fmt_num(a.self_corr))
        _metric("Ops / fields", f"{fmt_int(a.op_count)} / {len(a.fields or []) or '—'}")


CHECK_LABEL = {"pass": ("BRAIN CHECK PASSED", "pass"), "fail": ("BRAIN CHECK FAILED", "fail"),
               "pending": ("BRAIN CHECK PENDING", "pending"), "stale": ("RE-CHECK NEEDED", "warning")}


def robust_label(a: Alpha) -> str:
    if a.robust_state == "robust":
        return f"robust {fmt_pct(a.robust_ratio, 0)}"
    if a.robust_state == "fragile":
        return f"fragile {fmt_pct(a.robust_ratio, 0)}"
    return "untested" if a.robust_state == "unknown" else ""


def quality_badges(a: Alpha) -> None:
    """BRAIN check state, robustness, estimated self-correlation, honesty figures and idea cluster as badges."""
    with ui.row().classes("gap-1 flex-wrap items-center"):
        label = CHECK_LABEL.get(a.check_state or "")
        if label:
            badge(*label).tooltip("Stale: you submitted another alpha in this region/delay since this check; "
                                  "self-correlation may have changed." if a.check_state == "stale" else
                                  "Result of BRAIN's full submission check")
        if a.check_state == "pass" and not a.robust_state:
            badge("ROBUSTNESS NOT TESTED").tooltip("The robustness test did not run for this alpha (switched off, or "
                                                   "the task's repair budget was used up).")
        if a.robust_state:
            kind = {"robust": "pass", "fragile": "fail"}.get(a.robust_state, "")
            badge(robust_label(a).upper(), kind).tooltip(
                "Robustness: the worst of three neighbours (decay +2, windows ×0.8, ×1.25) keeps this share of the "
                "Sharpe. Below 80% = fragile. AlphaGo's own test; BRAIN does not require it.")
        if a.est_self_corr is not None:
            kind = "fail" if a.est_self_corr >= 0.7 else ("warning" if a.est_self_corr >= 0.5 else "pass")
            badge(f"EST. SELF-CORR {a.est_self_corr:.2f}", kind).tooltip(
                f"Estimated from daily PnL against your submitted alphas (closest: {a.est_corr_with or '—'}). "
                "An estimate and a lower bound; BRAIN's check decides.")
        if a.wf_ratio is not None:
            kind = "pass" if a.wf_ratio >= 0.7 else ("warning" if a.wf_ratio >= 0.4 else "fail")
            badge(f"RECENT 20% KEEPS {fmt_pct(a.wf_ratio, 0)}", kind).tooltip(
                "Walk-forward: Sharpe of the last 20% of the in-sample period ÷ the first 80%. The search saw these "
                "years, so this measures recent stability, not true out-of-sample.")
        if a.dsr is not None:
            badge(f"SEARCH-ADJUSTED {a.dsr:.2f}", "pass" if a.dsr >= 0.95 else "warning").tooltip(
                f"Deflated-Sharpe confidence after {a.n_trials or '?'} simulations ({a.n_eff or '?'} distinct "
                "lineages) in the search that found it. Below 0.95 the Sharpe may be partly luck. Advisory only.")
        if a.os_retention_pred is not None:
            badge(f"PREDICTED OS/IS {a.os_retention_pred:.2f}").tooltip(
                "From your own submitted alphas: the out-of-sample share of Sharpe alphas like this kept.")
        if a.cluster:
            badge(a.cluster.upper()).tooltip("Economic idea (used for the daily cap per idea)")


def settings_line(settings: dict[str, Any] | None) -> str:
    s = settings or {}
    keys = ["region", "universe", "delay", "neutralization", "decay", "truncation", "pasteurization", "nanHandling",
            "unitHandling", "maxTrade", "testPeriod"]
    return " · ".join(f"{k}={s[k]}" for k in keys if k in s)


async def load_alpha(alpha_id: str) -> Alpha | None:
    async with db.session() as s:
        return await s.get(Alpha, alpha_id)


async def confirm_submit(alpha_id: str, on_done: Callable[[], Any] | None = None) -> None:
    """Explicit confirmation dialog — the only path that submits an alpha."""
    a = await load_alpha(alpha_id)
    if a is None:
        ui.notify("Alpha not found locally.", type="warning")
        return
    kind = a.type or "REGULAR"
    used = await actions.submissions_today(kind)
    from ..config import settings
    limit = settings.programs.daily_super_submissions if kind == "SUPER" else settings.programs.daily_regular_submissions
    with ui.dialog() as dialog, ui.card().classes("ag-card p-6 w-[720px] max-w-[90vw]"):
        ui.label(f"Submit {alpha_id} to WorldQuant BRAIN?").classes("text-lg font-semibold")
        ui.label(a.code or a.selection or "").classes("ag-mono text-sm whitespace-pre-wrap break-all")
        ui.label(settings_line(a.settings)).classes("ag-muted text-xs")
        metrics_row(a)
        quality_badges(a)
        if a.check_state == "stale":
            ui.label("⚠ You submitted another alpha in this region/delay since this alpha was checked, so its "
                     "self-correlation may have changed. Run 'Check' again before submitting.").classes(
                "ag-amber text-sm")
        elif a.check_state != "pass":
            ui.label("⚠ BRAIN's full submission check has not passed yet for this alpha. Run 'Check' first — BRAIN will "
                     "reject the submission if any test fails.").classes("ag-amber text-sm")
        if a.robust_state == "fragile":
            ui.label(f"⚠ Fragile: neighbouring settings keep only {fmt_pct(a.robust_ratio, 0)} of its Sharpe (AlphaGo's "
                     "own test). BRAIN accepts it, but fragile alphas tend to hold up worse out of sample.").classes(
                "ag-amber text-sm")
        with ui.row().classes("gap-1 flex-wrap"):
            for c in a.checks or []:
                check_badge(c)
        ui.label(f"Today's {kind.lower()} submissions: {used}/{limit}. Submission is final and visible to "
                 "WorldQuant.").classes("ag-muted text-sm")

        async def go() -> None:
            dialog.close()
            fresh = await load_alpha(alpha_id)
            if fresh is not None and fresh.check_state == "stale":
                ui.notify(f"Re-running BRAIN's check on {alpha_id} first (its last pass is stale)…", type="info")
                try:
                    state, _ = await actions.run_check(alpha_id)
                except BrainError as exc:
                    notify_error(exc, "Check failed, not submitted: ")
                    return
                if state != "pass":
                    ui.notify(f"{alpha_id} no longer passes BRAIN's check ({state}) — not submitted.", type="warning")
                    return
            await jobs.submit("submit_alpha", f"Submit {alpha_id}", {"alpha_id": alpha_id, "confirmed": True},
                              priority=1)
            ui.notify(f"Submitting {alpha_id} — follow progress in Tasks.", type="info")
            if on_done:
                on_done()

        with ui.row().classes("justify-end w-full gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Submit to BRAIN", icon="send", on_click=go).props(
                f"color=positive {'disable' if used >= limit else ''}")
    dialog.open()


async def properties_dialog(alpha_id: str) -> None:
    a = await load_alpha(alpha_id)
    if a is None:
        return
    with ui.dialog() as dialog, ui.card().classes("ag-card p-6 w-[640px]"):
        ui.label(f"Properties · {alpha_id}").classes("text-lg font-semibold")
        name = ui.input("Name", value=a.name or "").classes("w-full")
        tags = ui.input("Tags (comma separated)", value=", ".join(a.tags or [])).classes("w-full")
        color = ui.select(["", "RED", "YELLOW", "GREEN", "BLUE", "PURPLE"], value=a.color or "", label="Color").classes("w-40")
        desc = ui.textarea("Description (Idea / Rationale for data / Rationale for operators)",
                           value=a.description or "").classes("w-full").props("autogrow")
        counter = ui.label().classes("ag-muted text-xs")
        desc.on_value_change(lambda e: counter.set_text(f"{len(e.value or '')} characters (Power Pool needs ≥ 100)"))
        counter.set_text(f"{len(a.description or '')} characters (Power Pool needs ≥ 100)")

        async def write_desc() -> None:
            from ..engine.describe import describe_alpha
            try:
                desc.value = await describe_alpha(alpha_id)
            except Exception as exc:
                notify_error(exc, "Description failed: ")

        async def save() -> None:
            try:
                await actions.set_properties(
                    alpha_id, name=name.value or None, color=color.value or None,
                    tags=[t.strip() for t in tags.value.split(",") if t.strip()],
                    regular_description=desc.value if (a.type or "REGULAR") == "REGULAR" else None)
                ui.notify("Saved to BRAIN.", type="positive")
                dialog.close()
            except BrainError as exc:
                notify_error(exc)

        with ui.row().classes("justify-between w-full"):
            ui.button("Write with AI", icon="auto_awesome", on_click=write_desc).props("flat")
            with ui.row():
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Save to BRAIN", icon="save", on_click=save)
    dialog.open()


async def open_alpha(alpha_id: str, on_change: Callable[[], Any] | None = None) -> None:
    a = await load_alpha(alpha_id)
    if a is None:
        ui.notify("Alpha not in the local database yet.", type="warning")
        return
    cand = None
    if a.candidate_id:
        async with db.session() as s:
            cand = await s.get(Candidate, a.candidate_id)

    with ui.dialog().props("maximized=false") as dialog, ui.card().classes("ag-card p-6 w-[1100px] max-w-[95vw]"):
        with ui.row().classes("items-center justify-between w-full"):
            with ui.row().classes("items-center gap-3"):
                ui.label(a.id).classes("text-xl font-semibold ag-mono")
                verdict_badge(a.verdict)
                badge(a.status or "—")
                if a.type == "SUPER":
                    badge("SUPER", "pending")
                for cl in a.classifications or []:
                    badge(str(cl.get("name") if isinstance(cl, dict) else cl))
            with ui.row().classes("gap-1"):
                ui.button(icon="open_in_new", on_click=lambda: ui.navigate.to(BRAIN_ALPHA_URL.format(a.id),
                                                                             new_tab=True)).props("flat round").tooltip(
                    "Open on BRAIN")
                ui.button(icon="close", on_click=dialog.close).props("flat round")
        if a.type == "SUPER":
            ui.label(f"Selection: {a.selection}").classes("ag-mono text-sm")
            ui.label(f"Combo: {a.combo}").classes("ag-mono text-sm")
        else:
            ui.code(a.code or "", language="javascript").classes("w-full text-sm")
        ui.label(settings_line(a.settings)).classes("ag-muted text-xs")
        if cand and cand.hypothesis:
            ui.label(f"Hypothesis: {cand.hypothesis}").classes("text-sm ag-cyan")
        metrics_row(a)
        quality_badges(a)
        with ui.row().classes("gap-1 flex-wrap"):
            for c in a.checks or []:
                check_badge(c)

        chart_box = ui.column().classes("w-full")
        info = ui.label().classes("ag-muted text-xs")

        async def run_check() -> None:
            check_btn.props("loading")
            try:
                state, _ = await actions.run_check(a.id)
                ui.notify(f"Check: {state.upper()}", type="positive" if state == "pass" else "warning")
                dialog.close()
                await open_alpha(a.id, on_change)
                if on_change:
                    on_change()
            except BrainError as exc:
                notify_error(exc)
            finally:
                check_btn.props(remove="loading")

        async def load_pnl() -> None:
            pnl_btn.props("loading")
            try:
                rows = await actions.pnl_series(a.id)
                chart_box.clear()
                if not rows:
                    info.set_text("No PnL record set returned.")
                    return
                keys = [k for k in rows[0].keys() if k.lower() != "date"]
                with chart_box:
                    ui.echart({
                        "backgroundColor": "transparent",
                        "tooltip": {"trigger": "axis"},
                        "legend": {"textStyle": {"color": "#8b98ad"}},
                        "grid": {"left": 60, "right": 20, "top": 30, "bottom": 30},
                        "xAxis": {"type": "category", "data": [r.get("date") or r.get("Date") for r in rows],
                                  "axisLabel": {"color": "#8b98ad"}},
                        "yAxis": {"type": "value", "axisLabel": {"color": "#8b98ad"},
                                  "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
                        "series": [{"name": k, "type": "line", "showSymbol": False, "data": [r.get(k) for r in rows],
                                    "lineStyle": {"width": 1.5}} for k in keys],
                    }).classes("w-full h-72")
                yearly = await actions.yearly_stats(a.id)
                if yearly:
                    cols = list(yearly[0].keys())
                    with chart_box:
                        ui.table(columns=[{"name": c, "label": c, "field": c} for c in cols],
                                 rows=yearly).props("dense flat").classes("w-full")
            except BrainError as exc:
                notify_error(exc)
            finally:
                pnl_btn.props(remove="loading")

        async def load_corr() -> None:
            corr_btn.props("loading")
            try:
                res = await actions.correlations(a.id, "self")
                info.set_text(f"Self-correlation max {fmt_num(res.get('max'), 3)} across "
                              f"{len(res.get('records') or [])} of your alphas.")
            except BrainError as exc:
                notify_error(exc)
            finally:
                corr_btn.props(remove="loading")

        async def optimize() -> None:
            from ..engine.optimizer import optimize_alpha
            try:
                ids = await optimize_alpha(a.id, rounds_hint=1)
                ui.notify(f"Queued {len(ids)} optimisation variants — they appear in Alphas when done.", type="info")
            except Exception as exc:
                notify_error(exc, "Optimise failed: ")

        with ui.row().classes("gap-2 pt-2 flex-wrap"):
            check_btn = ui.button("Check", icon="rule", on_click=run_check)
            ui.button("Submit…", icon="send", on_click=lambda: confirm_submit(a.id, on_change)).props(
                "color=positive" + ("" if a.status != "ACTIVE" else " disable"))
            ui.button("Optimise", icon="tune", on_click=optimize).props("flat")
            pnl_btn = ui.button("PnL & yearly stats", icon="show_chart", on_click=load_pnl).props("flat")
            corr_btn = ui.button("Self-correlation", icon="hub", on_click=load_corr).props("flat")
            ui.button("Properties", icon="edit", on_click=lambda: properties_dialog(a.id)).props("flat")
    dialog.open()
