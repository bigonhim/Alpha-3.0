"""Python Alphas: write (or have the AI write) an alpha in Python, check it locally, simulate it on BRAIN."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ...brain import brain
from ...config import settings
from ...db import db
from ...db.models import Job
from ...engine import python_alpha as pa
from ...jobs.queue import jobs
from ..components import badge, card, grid_options, notify_error, num_col, page_header


async def page() -> None:
    cfg = settings.python_alpha
    r = settings.research
    state: dict[str, Any] = {"job": None, "hypothesis": "", "name": ""}
    page_header("Python Alphas", "Alphas written in Python instead of FASTEXPR. The code is checked locally first (free), "
                "then simulated on BRAIN like any other alpha; results land on Alphas, where you check, repair and "
                "submit them.")

    sup_box = ui.row().classes("items-center gap-3 w-full")

    def paint_support() -> None:
        sup = pa.support()
        sup_box.clear()
        with sup_box:
            badge("Python alphas available" if sup["supported"] else "not available for your account",
                  "pass" if sup["supported"] else "warning")
            ui.label(sup["why"]).classes("text-sm")
            ui.space()
            ui.button("Refresh from BRAIN", icon="refresh", on_click=refresh_options).props("flat dense")

    async def refresh_options() -> None:
        try:
            from ...engine.sync import sync_options
            await sync_options()
        except Exception as exc:
            notify_error(exc, "Could not read the allowed settings: ")
        paint_support()

    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        with card("w-[58%]"):
            ui.label("Code").classes("ag-section-title")
            name = ui.input("Name").classes("w-full")
            code = ui.textarea("Python code", value=cfg.starter or "").classes("w-full ag-mono").props(
                "outlined autogrow input-style='min-height:320px'")
            check_box = ui.column().classes("w-full gap-0")
            opts = brain.options
            regions = opts.regions or ["USA"]
            with ui.row().classes("gap-2 items-end flex-wrap"):
                region = ui.select(regions, value=r.region if r.region in regions else regions[0],
                                   label="Region").classes("w-24")
                delay = ui.select(opts.delays(region.value), value=opts.delays(region.value)[0], label="Delay").classes(
                    "w-20")
                universe = ui.select(opts.universes(region.value), value=opts.universes(region.value)[0],
                                     label="Universe").classes("w-36")
                neut = ui.select(opts.neutralizations(region.value), value=opts.neutralizations(region.value)[0],
                                 label="Neutralization").classes("w-44")
                decay = ui.number("Decay", value=4, min=0, max=512).classes("w-24")
                trunc = ui.number("Truncation", value=0.08, min=0, max=1, step=0.01, format="%.2f").classes("w-28")

            def on_region(_: Any = None) -> None:
                for el, vals in ((delay, opts.delays(region.value)), (universe, opts.universes(region.value)),
                                 (neut, opts.neutralizations(region.value))):
                    el.options, el.value = vals, vals[0]
                    el.update()

            region.on_value_change(on_region)
            with ui.row().classes("gap-2"):
                ui.button("Check code", icon="fact_check", on_click=lambda: do_check()).props("outline")
                ui.button("Simulate (1 simulation)", icon="play_arrow", on_click=lambda: do_simulate())

        with card("grow"):
            ui.label("Write with AI").classes("ag-section-title")
            idea = ui.textarea("Idea", placeholder="e.g. 12-1 month momentum, skipping the most recent month, "
                               "within industry").classes("w-full").props("autogrow")
            base_expr = ui.input("Or translate this FASTEXPR expression (optional)").classes("w-full ag-mono")
            with ui.row().classes("gap-2"):
                ui.button("Write code", icon="auto_awesome", on_click=lambda: do_write())
            ui.separator()
            ui.label("Automation").classes("ag-section-title")
            ui.label("One idea per line: the AI writes each alpha, the local check gates it, and the ones that pass are "
                     "simulated (one simulation each, never past the daily reserve).").classes("ag-muted text-xs")
            ideas = ui.textarea("Ideas (one per line)").classes("w-full").props("autogrow")
            n = ui.number("Alphas to write", value=5, min=1, max=20).classes("w-36")
            ui.button("Write & simulate in the background", icon="smart_toy", on_click=lambda: do_batch()).props(
                "outline")
            hyp = ui.label().classes("ag-muted text-sm")

    progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
    progress.set_visibility(False)
    status = ui.label().classes("ag-muted text-sm")

    with ui.expansion("Reference and template (paste from BRAIN's Python Alpha docs)", icon="menu_book").classes(
            "w-full"):
        ui.label("BRAIN publishes no public spec for Python alphas. Paste the official documentation or a working "
                 "example from the platform here: the AI writes code against it, not from guesses. The template is "
                 "loaded into the editor when the page opens.").classes("ag-muted text-xs")
        ref = ui.textarea("Reference", value=cfg.reference).classes("w-full ag-mono").props("outlined autogrow")
        starter = ui.textarea("Template", value=cfg.starter).classes("w-full ag-mono").props("outlined autogrow")
        with ui.row().classes("gap-3 items-end"):
            lang = ui.input("Language value (empty = detect)", value=cfg.language).classes("w-56")
            rounds = ui.number("AI fixes", value=cfg.ai_rounds, min=0, max=5).classes("w-28")
            maxc = ui.number("Max characters", value=cfg.max_chars, min=500, step=500).classes("w-40")

        def save() -> None:
            cfg.reference, cfg.starter = ref.value or "", starter.value or ""
            cfg.language = (lang.value or "").strip()
            cfg.ai_rounds, cfg.max_chars = int(rounds.value or 0), int(maxc.value or 20_000)
            settings.save()
            paint_support()

        for el in (ref, starter, lang, rounds, maxc):
            el.on_value_change(lambda _: save())

    with card("w-full"):
        ui.label("Python alphas simulated from this page").classes("ag-section-title")
        hist = ui.aggrid(grid_options([
            {"field": "alpha_id", "headerName": "Alpha", "width": 110},
            {"field": "name", "headerName": "Name", "flex": 1, "minWidth": 180},
            {"field": "status", "headerName": "Status", "width": 100},
            {"field": "verdict", "headerName": "Verdict", "width": 100},
            num_col("sharpe", "Sharpe", 2, 90),
            num_col("fitness", "Fitness", 2, 90),
            num_col("turnover", "Turnover", 2, 95, pct=True),
            {"field": "message", "headerName": "Message", "flex": 1, "minWidth": 200},
        ], []), theme="quartz").classes("w-full h-[260px]")

    # ------------------------------------------------------------ behaviour
    def show_check(chk: dict) -> None:
        check_box.clear()
        with check_box:
            if chk["ok"]:
                ui.label("✓ Passes the local check (syntax, size, no file/network/process access).").classes(
                    "ag-green text-sm")
            for e in chk["errors"]:
                ui.label("✗ " + e).classes("ag-red text-sm")
            for w in chk["warnings"]:
                ui.label("⚠ " + w).classes("ag-amber text-sm")

    def do_check() -> bool:
        c = pa.check_code(code.value or "")
        show_check({"ok": c.ok, "errors": c.errors, "warnings": c.warnings})
        return c.ok

    def base_settings() -> dict[str, Any]:
        return {"region": region.value, "delay": int(delay.value), "universe": universe.value,
                "neutralization": neut.value, "decay": int(decay.value or 0), "truncation": float(trunc.value or 0.08)}

    async def run_job(kind: str, title: str, params: dict) -> None:
        if state["job"]:
            ui.notify("A Python-alpha job is already running.", type="warning")
            return
        state["job"] = await jobs.submit(kind, title, params)
        progress.set_visibility(True)
        status.set_text("Queued…")

    async def do_simulate() -> None:
        if not do_check():
            ui.notify("Fix the code first (see the check).", type="warning")
            return
        sup = pa.support()
        if not sup["supported"]:
            ui.notify(sup["why"], type="warning", multi_line=True)
            return
        await run_job("python_alpha_simulate", f"Python alpha: {name.value or 'untitled'}",
                      {"code": code.value, "settings": base_settings(), "name": name.value or "",
                       "hypothesis": state["hypothesis"]})

    async def do_write() -> None:
        from ...llm import llm
        if not llm.configure():
            ui.notify("Set up an AI provider on the LLM Integration page first.", type="warning")
            return
        status.set_text("The AI is writing the code…")
        try:
            d = await pa.write_code(idea.value or "", fastexpr=base_expr.value or "", region=region.value,
                                    delay=int(delay.value), universe=universe.value)
        except Exception as exc:
            notify_error(exc, "")
            status.set_text("")
            return
        name.value, code.value, state["hypothesis"] = d["name"], d["code"], d["hypothesis"]
        hyp.set_text(f"Hypothesis: {d['hypothesis']}" + ("" if d["verified"] else
                     " · Unverified: no reference pasted, so check the data access against BRAIN's docs."))
        show_check(d["check"])
        status.set_text(f"Written in {d['rounds']} round(s).")

    async def do_batch() -> None:
        from ...llm import llm
        if not llm.configure():
            ui.notify("Set up an AI provider on the LLM Integration page first.", type="warning")
            return
        sup = pa.support()
        if not sup["supported"]:
            ui.notify(sup["why"], type="warning", multi_line=True)
            return
        lines = [x.strip() for x in (ideas.value or "").splitlines() if x.strip()]
        await run_job("python_alpha_batch", f"Python alphas ×{int(n.value or 1)}",
                      {"ideas": lines, "n": int(n.value or 1), "settings": base_settings()})

    async def load_history() -> None:
        rows = list(reversed(await db.kv_get(pa.HISTORY_KEY, []) or []))
        hist.options["rowData"] = rows[:200]
        hist.update()

    async def tick() -> None:
        jid = state["job"]
        if not jid:
            return
        async with db.session() as s:
            job = await s.get(Job, jid)
        if job is None:
            state["job"] = None
            return
        progress.set_value(job.progress or 0)
        status.set_text(job.message or job.status)
        if job.status in ("queued", "running"):
            return
        state["job"] = None
        progress.set_visibility(False)
        status.set_text(f"Failed: {job.error}" if job.status == "failed" else (job.message or "Done."))
        await load_history()

    ui.timer(1.5, tick)
    paint_support()
    await load_history()
