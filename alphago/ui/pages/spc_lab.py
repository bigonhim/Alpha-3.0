"""SPC Lab — test a Systematic Predictions Challenge prompt before submitting it on BRAIN's website.

Paste the prompt and the JSON array it returned, check the output passes the form rules, then see what it makes from
the run date using Yahoo Finance prices. Runs are kept so they can be re-tested later.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

from nicegui import ui
from sqlalchemy import delete, func, select

from ...config import settings
from ...db import db
from ...db.models import SPCPrompt, SPCRun
from ...engine import spc
from ..components import card, fmt_num, fmt_pct, grid_options, notify_error, num_col, page_header

OUTPUT_PLACEHOLDER = '[{"company_name": "Apple Inc.", "ticker": "AAPL", ...}]'


def _date(v: str | None) -> date | None:
    try:
        return date.fromisoformat(v) if v else None
    except ValueError:
        return None


async def page() -> None:
    rules = settings.spc
    state: dict = {"prompt_id": None, "run_id": None, "check": None}
    page_header("SPC Lab", "Test a Systematic Predictions Challenge prompt before you submit it. Paste the prompt and its "
                           "output, check the output passes the form, then see what it makes from the run date using "
                           "Yahoo Finance prices. Come back later and test again to watch it.")

    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        # ------------------------------------------------------------ 1. prompt
        with card("w-1/2"):
            with ui.row().classes("items-center w-full gap-2"):
                ui.icon("description").classes("ag-muted")
                with ui.column().classes("gap-0"):
                    ui.label("1. Prompt").classes("ag-section-title")
                    ui.label("This text goes in the form's Prompt box.").classes("ag-muted text-xs")
                ui.space()
                ui.button("Reset", icon="restart_alt", on_click=lambda: reset()).props("flat dense")
                ui.button("Copy prompt", icon="content_copy", on_click=lambda: copy()).props("outline dense")
            with ui.row().classes("items-end gap-2 w-full no-wrap"):
                library = ui.select({}, label="Prompt library").classes("grow")
                ui.button("Load", icon="download", on_click=lambda: load_selected()).props("outline")
            ui.label("Pick a ready-made or saved prompt, then Load fills the name and the prompt.").classes(
                "ag-muted text-xs")
            name = ui.input("Prompt name", value="").classes("w-full")
            with ui.row().classes("items-center gap-2"):
                ui.label("Form settings:").classes("ag-muted text-sm")
                f_model = ui.input("Model", value=rules.model).props("dense").classes("w-24")
                f_version = ui.input("Version", value=rules.version).props("dense").classes("w-24")
                f_freq = ui.select(["Daily", "Weekly", "Monthly"], value=rules.frequency, label="Frequency").props(
                    "dense").classes("w-32")
                f_weight = ui.input("Weight", value=rules.weight).props("dense").classes("w-24")
            text = ui.textarea("Prompt").classes("w-full ag-mono").props(
                "outlined autogrow input-style='min-height:300px'")
            counter = ui.label().classes("ag-muted text-xs")
            with ui.row().classes("gap-2"):
                ui.button("Save as new version", icon="save", on_click=lambda: save_prompt()).props("outline")

        # ------------------------------------------------------------ 2. output
        with card("grow"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("output").classes("ag-muted")
                with ui.column().classes("gap-0"):
                    ui.label("2. Output").classes("ag-section-title")
                    ui.label("Paste the JSON array the prompt returned.").classes("ag-muted text-xs")
            with ui.row().classes("w-full gap-3 no-wrap"):
                with ui.column().classes("gap-0 grow"):
                    run_date = ui.input("Run date", value=date.today().isoformat()).props("type=date").classes("w-full")
                    ui.label("The day the portfolio was built. The PnL test starts here.").classes("ag-muted text-xs")
                with ui.column().classes("gap-0 grow"):
                    until = ui.input("Test until (optional)").props("type=date").classes("w-full")
                    ui.label("Empty means up to today.").classes("ag-muted text-xs")
            output = ui.textarea("Output (JSON array)", placeholder=OUTPUT_PLACEHOLDER).classes("w-full ag-mono").props(
                "outlined input-style='min-height:220px'")
            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("Check", icon="check", on_click=lambda: do_check()).props("outline")
                test_btn = ui.button("Test PnL", icon="play_arrow", on_click=lambda: do_test())
            check_box = ui.column().classes("w-full gap-1")
            result_box = ui.column().classes("w-full gap-2")
            with ui.expansion("Stability check: paste a 2nd and 3rd run of the same prompt", icon="compare").classes(
                    "w-full"):
                ui.label("The same prompt can return different stocks each time. Paste two more runs to see how much "
                         "they overlap and whether their returns move together.").classes("ag-muted text-sm")
                run2 = ui.textarea("2nd run (JSON array)").classes("w-full ag-mono").props("outlined autogrow")
                run3 = ui.textarea("3rd run (JSON array)").classes("w-full ag-mono").props("outlined autogrow")
                stab_btn = ui.button("Compare runs", icon="compare_arrows", on_click=lambda: do_stability())
                stab_box = ui.column().classes("w-full gap-1")

    # ------------------------------------------------------------ rules + saved runs
    with ui.expansion("Form rules (editable — BRAIN does not publish the SPC rules)", icon="rule").classes("w-full"):
        with ui.row().classes("gap-3 items-end flex-wrap"):
            r_req = ui.input("Required fields (comma separated)", value=", ".join(rules.required_fields)).classes("w-80")
            r_min = ui.number("Min rows", value=rules.min_rows, min=0).classes("w-28")
            r_max = ui.number("Max rows (0 = no limit)", value=rules.max_rows, min=0).classes("w-44")
            r_wf = ui.input("Weight field (empty = equal weight)", value=rules.weight_field).classes("w-64")
            r_sf = ui.input("Long/short field (empty = all long)", value=rules.side_field).classes("w-64")
            r_bm = ui.input("Benchmark ticker", value=rules.benchmark).classes("w-36")
            r_chars = ui.number("Prompt character limit", value=rules.max_chars, min=100).classes("w-48")

        def save_rules() -> None:
            rules.required_fields = [x.strip() for x in (r_req.value or "").split(",") if x.strip()]
            rules.min_rows, rules.max_rows = int(r_min.value or 0), int(r_max.value or 0)
            rules.weight_field, rules.side_field = (r_wf.value or "").strip(), (r_sf.value or "").strip()
            rules.benchmark, rules.max_chars = (r_bm.value or "").strip(), int(r_chars.value or 10000)
            settings.save()
            update_counter()

        for el in (r_req, r_min, r_max, r_wf, r_sf, r_bm, r_chars):
            el.on_value_change(lambda _: save_rules())
        ui.label("Paste BRAIN's official SPC rules into these fields when you have them. The 'Earnings Season Drift "
                 "LS' starter prompt returns a 'side' field: set the long/short field to 'side' to test it as "
                 "long/short.").classes("ag-muted text-xs")

    with card("w-full"):
        with ui.row().classes("items-center w-full gap-2"):
            ui.label("Saved runs").classes("ag-section-title")
            ui.label("Every Test PnL is saved here. Re-test a run later to watch it.").classes("ag-muted text-sm")
            ui.space()
            ui.button("Load selected", icon="open_in_browser", on_click=lambda: load_run()).props("flat")
            ui.button("Re-test selected", icon="replay", on_click=lambda: retest()).props("flat")
            ui.button("Delete selected", icon="delete", on_click=lambda: delete_runs()).props("flat color=negative")
        runs_grid = ui.aggrid(grid_options([
            {"field": "id", "headerName": "#", "width": 70},
            {"field": "label", "headerName": "Prompt", "flex": 1, "minWidth": 220},
            {"field": "run_date", "headerName": "Run date", "width": 115},
            {"field": "rows", "headerName": "Stocks", "width": 90},
            num_col("total", "Return", 2, 100, pct=True),
            num_col("excess", "vs benchmark", 2, 120, pct=True),
            num_col("sharpe", "Sharpe", 2, 90),
            {"field": "days", "headerName": "Days", "width": 80},
            {"field": "tested", "headerName": "Last tested", "width": 150},
        ], [], multi=True), theme="quartz").classes("w-full h-[300px]")

    # ------------------------------------------------------------ behaviour
    def update_counter() -> None:
        n = len(text.value or "")
        counter.set_text(f"{n:,} / {rules.max_chars:,} characters")
        counter.classes(replace="text-xs " + ("ag-red" if n > rules.max_chars else "ag-muted"))

    text.on_value_change(lambda _: update_counter())

    async def load_library() -> None:
        async with db.session() as s:
            rows = list(await s.scalars(select(SPCPrompt).order_by(SPCPrompt.id.desc())))
        opts = {f"starter:{k}": f"★ {k}" for k in spc.STARTER_PROMPTS}
        opts.update({f"saved:{p.id}": f"{p.name} · v{p.version}" for p in rows})
        library.options = opts
        library.update()

    async def load_selected() -> None:
        key = library.value
        if not key:
            ui.notify("Pick a prompt first.", type="warning")
            return
        kind, ident = key.split(":", 1)
        if kind == "starter":
            name.value, text.value, state["prompt_id"] = ident, spc.STARTER_PROMPTS[ident], None
            return
        async with db.session() as s:
            p = await s.get(SPCPrompt, int(ident))
        if p:
            name.value, text.value, state["prompt_id"] = p.name, p.prompt, p.id
            fs = p.form_settings or {}
            f_model.value = fs.get("model", rules.model)
            f_version.value = fs.get("version", rules.version)
            f_freq.value = fs.get("frequency", rules.frequency)
            f_weight.value = fs.get("weight", rules.weight)

    def reset() -> None:
        name.value, text.value, state["prompt_id"] = "", "", None
        f_model.value, f_version.value, f_freq.value, f_weight.value = (rules.model, rules.version, rules.frequency,
                                                                        rules.weight)

    async def copy() -> None:
        ui.clipboard.write(text.value or "")
        ui.notify("Prompt copied — paste it into the form's Prompt box on BRAIN.")

    async def save_prompt() -> int | None:
        if not (text.value or "").strip():
            ui.notify("Write or load a prompt first.", type="warning")
            return None
        nm = (name.value or "").strip() or "Untitled prompt"
        async with db.write() as s:
            version = (await s.scalar(select(func.max(SPCPrompt.version)).where(SPCPrompt.name == nm)) or 0) + 1
            p = SPCPrompt(name=nm, version=version, prompt=text.value, form_settings={
                "model": f_model.value, "version": f_version.value, "frequency": f_freq.value, "weight": f_weight.value})
            s.add(p)
            await s.flush()
            pid = p.id
        state["prompt_id"] = pid
        await load_library()
        library.value = f"saved:{pid}"
        ui.notify(f"Saved {nm} v{version}.", type="positive")
        return pid

    def show_check(res: spc.CheckResult) -> None:
        check_box.clear()
        with check_box:
            if res.ok:
                ui.label(f"✓ Passes the form rules: {len(res.rows)} stocks.").classes("ag-green text-sm")
            for e in res.errors[:20]:
                ui.label(f"✗ {e}").classes("ag-red text-sm")
            for w in res.warnings[:20]:
                ui.label(f"⚠ {w}").classes("ag-amber text-sm")

    def do_check() -> spc.CheckResult:
        res = spc.check_output(output.value or "", rules)
        state["check"] = res
        show_check(res)
        return res

    def show_result(res: dict) -> None:
        result_box.clear()
        with result_box:
            if not res.get("days"):
                ui.label("No trading days after the run date yet — test again later.").classes("ag-amber text-sm")
                return
            with ui.row().classes("gap-8"):
                for label, value in (("Return", fmt_pct(res.get("total_return"), 2)),
                                     (f"vs {rules.benchmark or 'benchmark'}", fmt_pct(res.get("excess_return"), 2)),
                                     ("Sharpe (annualised)", fmt_num(res.get("sharpe"))),
                                     ("Max drawdown", fmt_pct(res.get("max_drawdown"), 2)),
                                     ("Trading days", str(res.get("days")))):
                    with ui.column().classes("gap-0"):
                        ui.label(label).classes("ag-muted text-xs")
                        ui.label(value).classes("ag-kpi-sm ag-teal")
            curve = res.get("curve") or {}
            if curve.get("dates"):
                series = [{"name": "Portfolio", "type": "line", "showSymbol": False, "data": curve["portfolio"]}]
                if curve.get("benchmark"):
                    series.append({"name": rules.benchmark, "type": "line", "showSymbol": False,
                                   "data": curve["benchmark"]})
                ui.echart({"backgroundColor": "transparent", "tooltip": {"trigger": "axis"},
                           "legend": {"top": 0, "textStyle": {"color": "#8b98ad"}},
                           "grid": {"left": 60, "right": 20, "top": 36, "bottom": 30},
                           "xAxis": {"type": "category", "data": curve["dates"], "axisLabel": {"color": "#8b98ad"}},
                           "yAxis": {"type": "value", "axisLabel": {"color": "#8b98ad",
                                                                     ":formatter": "v => (v*100).toFixed(1) + '%'"},
                                     "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
                           "series": series}).classes("w-full h-60")
            best = ", ".join(f"{t} {v:+.1%}" for t, v in res.get("best") or [])
            worst = ", ".join(f"{t} {v:+.1%}" for t, v in res.get("worst") or [])
            if best:
                ui.label(f"Best: {best}").classes("text-sm ag-green")
            if worst:
                ui.label(f"Worst: {worst}").classes("text-sm ag-red")
            if res.get("missing"):
                ui.label(f"No Yahoo prices for: {', '.join(res['missing'])} (left out of the test).").classes(
                    "text-sm ag-amber")
            ui.label("Equal weight unless a weight field is set; short rows count negative; rebalanced daily. Prices: "
                     "Yahoo Finance adjusted closes.").classes("ag-muted text-xs")

    async def do_test() -> None:
        res = do_check()
        if not res.ok:
            ui.notify("Fix the output first (see the check).", type="warning")
            return
        rd = _date(run_date.value)
        if rd is None:
            ui.notify("Pick a run date.", type="warning")
            return
        test_btn.props("loading")
        try:
            out = await spc.test_pnl(res.rows, rules, rd, _date(until.value))
            show_result(out)
            await save_run(res, out)
        except Exception as exc:
            notify_error(exc, "PnL test failed: ")
        finally:
            test_btn.props(remove="loading")

    async def save_run(res: spc.CheckResult, out: dict) -> None:
        label = f"{(name.value or 'Untitled').strip()}"
        slim = {k: v for k, v in out.items() if k != "curve"}
        async with db.write() as s:
            row = await s.get(SPCRun, state["run_id"]) if state["run_id"] else None
            if row is None or row.output != res.rows or row.run_date != run_date.value:
                row = SPCRun(prompt_id=state["prompt_id"], label=label, run_date=run_date.value, output=res.rows)
                s.add(row)
            row.test_until = until.value or None
            row.check = {"errors": res.errors, "warnings": res.warnings}
            row.result = slim
            row.tested_at = datetime.now(timezone.utc)
            await s.flush()
            state["run_id"] = row.id
        await load_runs()

    async def do_stability() -> None:
        outs = []
        for i, box in enumerate((output, run2, run3), 1):
            if not (box.value or "").strip():
                continue
            r = spc.check_output(box.value, rules)
            if not r.ok:
                ui.notify(f"Run {i} does not pass the check: {r.errors[0]}", type="warning")
                return
            outs.append(r.rows)
        if len(outs) < 2:
            ui.notify("Paste at least one more run (plus the output above).", type="warning")
            return
        stab_btn.props("loading")
        try:
            cmp_ = spc.compare_runs(outs)
            rd = _date(run_date.value)
            pnl_ = await spc.stability_pnl(outs, rules, rd, _date(until.value)) if rd else None
            stab_box.clear()
            with stab_box:
                ui.label("Stocks per run: " + " · ".join(str(n) for n in cmp_["sizes"])).classes("text-sm")
                for pr in cmp_["pairs"]:
                    ui.label(f"Runs {pr['runs']}: {fmt_pct(pr['overlap'], 0)} of stocks in common").classes("text-sm")
                ui.label(f"In every run ({len(cmp_['common'])}): {', '.join(cmp_['common']) or 'none'}").classes(
                    "text-sm ag-muted")
                if pnl_:
                    ui.label("Return per run: " + " · ".join(fmt_pct(t, 2) for t in pnl_["totals"])).classes("text-sm")
                    corr = pnl_["corr"]
                    keys = list(corr)
                    for i in range(len(keys)):
                        for j in range(i + 1, len(keys)):
                            ui.label(f"Daily return correlation {keys[i]}–{keys[j]}: "
                                     f"{fmt_num(corr[keys[i]].get(keys[j]))}").classes("text-sm")
                ui.label("High overlap and correlation mean the prompt gives a stable answer.").classes(
                    "ag-muted text-xs")
        except Exception as exc:
            notify_error(exc, "Stability check failed: ")
        finally:
            stab_btn.props(remove="loading")

    async def load_runs() -> None:
        async with db.session() as s:
            rows = list(await s.scalars(select(SPCRun).order_by(SPCRun.id.desc()).limit(300)))
        runs_grid.options["rowData"] = [{
            "id": r.id, "label": r.label, "run_date": r.run_date, "rows": len(r.output or []),
            "total": (r.result or {}).get("total_return"), "excess": (r.result or {}).get("excess_return"),
            "sharpe": (r.result or {}).get("sharpe"), "days": (r.result or {}).get("days"),
            "tested": r.tested_at.astimezone().strftime("%Y-%m-%d %H:%M") if r.tested_at else ""} for r in rows]
        runs_grid.update()

    async def selected_runs() -> list[int]:
        return [r["id"] for r in await runs_grid.get_selected_rows()]

    async def load_run() -> None:
        ids = await selected_runs()
        if not ids:
            ui.notify("Tick a run first.", type="warning")
            return
        async with db.session() as s:
            r = await s.get(SPCRun, ids[0])
        if r:
            state["run_id"], state["prompt_id"] = r.id, r.prompt_id
            run_date.value, until.value = r.run_date, r.test_until or ""
            output.value = json.dumps(r.output, indent=1)
            name.value = r.label or ""
            if r.result:
                show_result({**r.result, "curve": {}})

    async def retest() -> None:
        ids = await selected_runs()
        if not ids:
            ui.notify("Tick one or more runs first.", type="warning")
            return
        done = 0
        for rid in ids:
            async with db.session() as s:
                r = await s.get(SPCRun, rid)
            if not r or not _date(r.run_date):
                continue
            try:
                out = await spc.test_pnl(r.output, rules, _date(r.run_date), _date(r.test_until))
                async with db.write() as s:
                    row = await s.get(SPCRun, rid)
                    row.result = {k: v for k, v in out.items() if k != "curve"}
                    row.tested_at = datetime.now(timezone.utc)
                done += 1
                if len(ids) == 1:
                    show_result(out)
            except Exception as exc:
                notify_error(exc, f"Run #{rid}: ")
        ui.notify(f"Re-tested {done} run(s).", type="positive")
        await load_runs()

    async def delete_runs() -> None:
        ids = await selected_runs()
        if ids:
            async with db.write() as s:
                await s.execute(delete(SPCRun).where(SPCRun.id.in_(ids)))
            await load_runs()

    update_counter()
    await load_library()
    await load_runs()
