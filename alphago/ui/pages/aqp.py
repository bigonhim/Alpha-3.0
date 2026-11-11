"""Alpha Quality Project: choose up to 3 USA alphas, write their descriptions, tag them for the monthly cycle.

Nothing is written to BRAIN until the confirmation box: tagging is permanent for an alpha (it can never be used in another
cycle) and the description replaces the one on BRAIN.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ...config import settings
from ...db import db
from ...db.models import Job
from ...engine import aqp
from ...jobs.queue import jobs
from ..components import badge, card, grid_options, num_col, page_header

STATE_KIND = {"selected": "pass", "eligible": "", "used": "fail", "excluded": "warning"}


async def page() -> None:
    cfg = settings.aqp
    state: dict[str, Any] = {"job": None, "rows": []}
    page_header("Alpha Quality Project", "Each month, tag up to 3 of your best USA alphas and describe them. 80% of the "
                "score is the description (match to the expression, uniqueness, quality of the hypothesis), 20% is "
                "out-of-sample performance. An alpha tagged in one cycle can never be tagged again.")

    with card("w-full"):
        with ui.row().classes("items-end gap-3 w-full"):
            cycle = ui.input("Cycle tag", value=aqp.current_cycle()).classes("w-48")
            ui.button("Refresh list", icon="refresh", on_click=lambda: load()).props("outline")
            ui.button("Suggest the best 3", icon="auto_fix_high", on_click=lambda: suggest()).props("outline")
            ui.space()
            ui.label("Ranking is the app's heuristic, not BRAIN's score: 80% how well the alpha can be explained "
                     "(simple, few fields, clear idea, distinct), 20% out-of-sample Sharpe.").classes("ag-muted text-xs")
        summary = ui.label().classes("text-sm")
        grid = ui.aggrid(grid_options([
            {"field": "id", "headerName": "Alpha", "width": 110},
            {"field": "state", "headerName": "State", "width": 100},
            num_col("score", "AQP score", 2, 100),
            num_col("describability", "Explainable", 2, 105),
            num_col("os", "OS Sharpe", 2, 95),
            {"field": "os_basis", "headerName": "OS basis", "width": 85},
            num_col("is_sharpe", "IS Sharpe", 2, 90),
            num_col("turnover", "Turnover", 2, 90, pct=True),
            {"field": "ops", "headerName": "Ops", "width": 65},
            {"field": "cluster", "headerName": "Idea", "width": 110},
            {"field": "draft", "headerName": "Description", "width": 130},
            {"field": "code", "headerName": "Expression", "flex": 1, "minWidth": 260},
            {"field": "note", "headerName": "Note", "flex": 1, "minWidth": 220},
        ], [], multi=True), theme="quartz").classes("w-full h-[380px]")
        with ui.row().classes("gap-2 w-full items-center"):
            ui.button("Write descriptions (AI)", icon="edit_note", on_click=lambda: describe())
            ui.button("Review & apply…", icon="send", on_click=lambda: review()).props("color=positive")
            ui.button("Remove this cycle's tag…", icon="label_off", on_click=lambda: untag()).props("flat color=negative")
            ui.space()
        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
        progress.set_visibility(False)
        status = ui.label().classes("ag-muted text-sm")

    with ui.expansion("Pre-check settings", icon="tune").classes("w-full"):
        with ui.row().classes("gap-3 items-end flex-wrap"):
            s_region = ui.input("Region", value=cfg.region).classes("w-28")
            s_max = ui.number("Alphas per cycle", value=cfg.max_picks, min=1, max=3).classes("w-40")
            s_chars = ui.number("Min description characters", value=cfg.min_description_chars, min=200).classes("w-56")
            s_grade = ui.number("Min AI-judge score (0–10)", value=cfg.min_grade, min=1, max=10).classes("w-52")
            s_rounds = ui.number("AI rewrites", value=cfg.max_rounds, min=0, max=4).classes("w-32")
            s_sim = ui.number("Max overlap between descriptions (0–1)", value=cfg.max_similarity, min=0.1, max=1,
                              step=0.05).classes("w-64")

        def save() -> None:
            cfg.region, cfg.max_picks = (s_region.value or "USA").strip().upper(), max(1, min(3, int(s_max.value or 3)))
            cfg.min_description_chars, cfg.min_grade = int(s_chars.value or 700), int(s_grade.value or 7)
            cfg.max_rounds, cfg.max_similarity = int(s_rounds.value or 0), float(s_sim.value or 0.55)
            settings.save()

        for el in (s_region, s_max, s_chars, s_grade, s_rounds, s_sim):
            el.on_value_change(lambda _: save())
        ui.label("The pre-checks are the app's, not BRAIN's: every field and operator of the expression must be named, an "
                 "AI judge scores match, uniqueness and hypothesis, and the descriptions must differ from each other. "
                 "Passing them does not guarantee a score.").classes("ag-muted text-xs")

    # ------------------------------------------------------------ list
    async def load() -> None:
        rows = await aqp.candidates((cycle.value or "").strip() or None)
        state["rows"] = rows
        grid.options["rowData"] = [{**r, "draft": ("✓ passes" if r["draft_ok"] else "draft: fix" if r["has_draft"] else ""),
                                    "note": r["reason"] or r["notes"]} for r in rows]
        grid.update()
        free = sum(r["state"] == "eligible" for r in rows)
        tagged = sum(r["state"] == "selected" for r in rows)
        used = sum(r["state"] == "used" for r in rows)
        if rows:
            summary.set_text(f"{free} eligible · {tagged} already tagged {cycle.value} (of {cfg.max_picks}) · {used} used "
                             "in an earlier cycle (can never be tagged again)")
        else:
            summary.set_text(f"No submitted {cfg.region} alphas yet: run Sync my alphas on Sync with BRAIN first. An alpha "
                             "tagged in an earlier cycle can never be tagged again.")

    async def selected() -> list[str]:
        return [r["id"] for r in await grid.get_selected_rows()]

    async def suggest() -> None:
        ids = aqp.pick_best(state["rows"])
        if not ids:
            ui.notify("No eligible alphas.", type="warning")
            return
        grid.run_grid_method("deselectAll")
        for i, r in enumerate(state["rows"]):          # AG Grid's default row id is the index in rowData
            if r["id"] in ids:
                grid.run_row_method(str(i), "setSelected", True)
        ui.notify("Suggested: " + ", ".join(ids) + " (different idea families where possible).", type="info")

    # ------------------------------------------------------------ jobs
    async def run_job(kind: str, title: str, params: dict) -> None:
        if state["job"]:
            ui.notify("A job is already running here.", type="warning")
            return
        state["job"] = await jobs.submit(kind, title, params, priority=1 if params.get("confirmed") else 5)
        progress.set_visibility(True)
        status.set_text("Queued…")

    async def describe() -> None:
        ids = await selected()
        if not ids or len(ids) > cfg.max_picks:
            ui.notify(f"Tick 1 to {cfg.max_picks} alphas first.", type="warning")
            return
        from ...llm import llm
        if not llm.configure():
            ui.notify("Set up an AI provider on the LLM Integration page first.", type="warning")
            return
        await run_job("aqp_describe", f"AQP descriptions ({len(ids)})", {"alpha_ids": ids})

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
        if job.status == "failed":
            status.set_text(f"Failed: {job.error}")
            ui.notify(job.error or "Failed", type="negative", multi_line=True, close_button=True, timeout=10000)
        await load()

    ui.timer(1.5, tick)

    # ------------------------------------------------------------ review and apply
    async def review() -> None:
        ids = await selected()
        if not ids:
            ui.notify("Tick the alphas to tag first.", type="warning")
            return
        drafts = {i: await db.kv_get(aqp.DRAFT_KEY.format(i)) for i in ids}
        missing = [i for i, d in drafts.items() if not d]
        if missing:
            ui.notify("Write the descriptions first (no draft for " + ", ".join(missing) + ").", type="warning")
            return
        tag = (cycle.value or "").strip()
        boxes: dict[str, ui.textarea] = {}
        with ui.dialog() as dlg, ui.card().classes("ag-card p-6 w-[960px] max-w-[95vw] max-h-[90vh] overflow-auto"):
            ui.label(f"Describe and tag {len(ids)} alpha(s) with {tag}?").classes("text-lg font-semibold")
            ui.label("Tagging is permanent for each alpha: it can never be tagged in another cycle. Each description "
                     "replaces the one on BRAIN (the old one is kept locally). Edit the text if you like, then press "
                     "Re-check.").classes("ag-amber text-sm")
            for aid in ids:
                d = drafts[aid]
                row = next((r for r in state["rows"] if r["id"] == aid), {})
                with ui.column().classes("w-full gap-1 border-t pt-2"):
                    ui.label(f"{aid} · {row.get('code', '')}").classes("ag-mono text-sm break-all")
                    result = ui.row().classes("gap-1 flex-wrap")
                    box = ui.textarea(value=d["text"]).classes("w-full").props("outlined autogrow")
                    boxes[aid] = box

                    def paint(res: dict, holder: ui.row = result) -> None:
                        holder.clear()
                        with holder:
                            ck, g = res["checks"], res.get("grade") or {}
                            badge("names every field/operator: " + ("yes" if not ck["missing"]
                                                                    else "missing " + ", ".join(ck["missing"])),
                                  "pass" if not ck["missing"] else "fail")
                            badge(f"{ck['chars']} chars", "pass" if ck["chars"] >= cfg.min_description_chars else "fail")
                            if g:
                                for k in ("match", "uniqueness", "hypothesis"):
                                    badge(f"{k} {g[k]}/10" + (" (before edits)" if res.get("edited") else ""),
                                          "pass" if g[k] >= cfg.min_grade else "fail")
                            for gap in (g.get("gaps") or [])[:3]:
                                ui.label("• " + gap).classes("text-xs ag-muted w-full")

                    paint(d)

                    async def recheck(a: str = aid, b: ui.textarea = box, p: Any = paint) -> None:
                        p(await aqp.recheck_text(a, b.value or ""))

                    ui.button("Re-check", icon="fact_check", on_click=recheck).props("flat dense")
            err = ui.column().classes("w-full gap-0")

            async def go() -> None:
                items = [{"alpha_id": a, "description": b.value or ""} for a, b in boxes.items()]
                problems = await aqp.validate_cycle(tag, items)
                err.clear()
                if problems:
                    with err:
                        for p in problems[:8]:
                            ui.label("✗ " + p).classes("ag-red text-sm")
                    return
                dlg.close()
                await run_job("aqp_apply", f"AQP {tag}: describe and tag {len(items)}",
                              {"cycle": tag, "items": items, "confirmed": True})
                ui.notify("Writing to BRAIN: follow progress here or in Tasks.", type="info")

            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                ui.button(f"Describe and tag with {tag}", icon="send", on_click=go).props("color=positive")
        dlg.open()

    async def untag() -> None:
        ids = [r["id"] for r in await grid.get_selected_rows() if r["state"] == "selected"]
        if not ids:
            ui.notify("Tick alphas that are tagged this cycle.", type="warning")
            return
        tag = (cycle.value or "").strip()
        with ui.dialog() as dlg, ui.card().classes("ag-card p-6"):
            ui.label(f"Remove {tag} from {', '.join(ids)}?").classes("text-lg font-semibold")
            ui.label("Only this cycle's tag is removed; the description stays.").classes("ag-muted text-sm")

            async def go() -> None:
                dlg.close()
                await run_job("aqp_untag", f"AQP untag {len(ids)}", {"cycle": tag, "alpha_ids": ids, "confirmed": True})

            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                ui.button("Remove tag", on_click=go).props("color=negative")
        dlg.open()

    await load()
