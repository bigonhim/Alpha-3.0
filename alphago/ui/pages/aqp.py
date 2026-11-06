"""Alpha Quality Project — pick up to 3 USA alphas for this month's cycle, write scored hypothesis descriptions, then
tag and describe them on BRAIN (after you confirm)."""

from __future__ import annotations

from nicegui import ui

from ...engine import aqp
from ..components import card, fmt_num, grid_options, notify_error, num_col, page_header


async def page() -> None:
    tag = aqp.cycle_tag()
    state: dict = {"rows": {}, "selected": None}
    page_header("Alpha Quality Project", f"This cycle's tag is {tag}. Tag up to {aqp.MAX_PER_CYCLE} of your best "
                                         "submitted USA alphas and describe each hypothesis. Score: 80% description "
                                         "(match to the expression, uniqueness, hypothesis quality) + 20% "
                                         "out-of-sample performance. An alpha tagged in an earlier cycle cannot be "
                                         "tagged again.")

    with card("w-full"):
        with ui.row().classes("items-center w-full gap-2"):
            ui.label("Eligible alphas").classes("ag-section-title")
            ui.label("Submitted USA regular alphas never tagged in an earlier cycle, ranked by an estimate of the "
                     "score (distinctive idea + out-of-sample evidence). Run 'Sync my alphas' first so tags and OS "
                     "results are fresh.").classes("ag-muted text-sm")
            ui.space()
            ui.button("Suggest best 3", icon="auto_awesome", on_click=lambda: suggest()).props("outline")
            ui.button("Refresh", icon="refresh", on_click=lambda: load()).props("flat")
        summary = ui.label().classes("text-sm")
        grid = ui.aggrid(grid_options([
            {"field": "id", "headerName": "Alpha", "width": 110},
            {"field": "in_cycle", "headerName": tag, "width": 110,
             ":valueFormatter": "p => p.value ? 'tagged' : ''"},
            {"field": "code", "headerName": "Expression", "flex": 1, "minWidth": 300},
            {"field": "cluster", "headerName": "Idea", "width": 110},
            num_col("estimate", "Pick score", 2, 105),
            num_col("unique", "Distinct", 2, 95),
            num_col("performance", "Perf.", 2, 85),
            num_col("os_sharpe", "OS Sharpe", 2, 105),
            num_col("sharpe", "IS Sharpe", 2, 100),
            num_col("self_corr", "Self-corr", 2, 100),
            {"field": "described", "headerName": "Has text", "width": 95,
             ":valueFormatter": "p => p.value ? 'yes' : ''"},
            {"field": "submitted", "headerName": "Submitted", "width": 115},
        ], []), theme="quartz").classes("w-full h-[340px]")

    with card("w-full"):
        with ui.row().classes("items-center w-full gap-2"):
            ui.label("Description").classes("ag-section-title")
            current = ui.label("Select an alpha above.").classes("ag-muted text-sm ag-mono")
            ui.space()
            write_btn = ui.button("Write & grade with AI", icon="edit_note", on_click=lambda: write()).props("outline")
            apply_btn = ui.button(f"Tag {tag} & save on BRAIN", icon="sell", on_click=lambda: confirm_apply()).props(
                "color=positive")
        grade_box = ui.column().classes("w-full gap-0")
        editor = ui.textarea("Description (editable before you apply)").classes("w-full").props(
            "outlined autogrow input-style='min-height:260px'")
        counter = ui.label().classes("ag-muted text-xs")
        editor.on_value_change(lambda _: counter.set_text(f"{len(editor.value or ''):,} characters"))

    def show_grade(d: dict | None) -> None:
        grade_box.clear()
        if not d:
            return
        with grade_box:
            g = d.get("grade") or {}
            row = state["rows"].get(state["selected"]) or {}
            est = aqp.estimated_score(d.get("score"), row.get("performance") or 0.0)
            if g:
                ui.label(f"AI grade — match {g.get('match')}/10 · uniqueness {g.get('uniqueness')}/10 · hypothesis "
                         f"{g.get('hypothesis_quality')}/10 · description score {fmt_num(d.get('score'))}/10 · "
                         f"estimated AQP score {fmt_num(est, 1)}/100 · {d.get('rounds')} round(s)").classes("text-sm")
            for p in d.get("problems") or []:
                ui.label(f"✗ {p}").classes("ag-red text-sm")
            if not d.get("problems"):
                ui.label("✓ Every field, operator, window and grouping in the expression is described.").classes(
                    "ag-green text-sm")
            for m in (g.get("mismatches") or [])[:5]:
                ui.label(f"⚠ {m}").classes("ag-amber text-sm")
            if d.get("note"):
                ui.label(d["note"]).classes("ag-amber text-sm")

    async def load() -> None:
        try:
            cands = await aqp.eligible(tag)
        except Exception as exc:
            notify_error(exc, "Could not load alphas: ")
            return
        rows = [c.row() for c in cands]
        state["rows"] = {r["id"]: r for r in rows}
        state["cands"] = cands
        grid.options["rowData"] = rows
        grid.update()
        n_in = sum(1 for r in rows if r["in_cycle"])
        summary.set_text(f"{len(rows)} eligible · {n_in}/{aqp.MAX_PER_CYCLE} tagged {tag}"
                         + ("" if rows else " — sync your alphas, or submit USA alphas first."))

    async def suggest() -> None:
        cands = state.get("cands") or []
        picks = aqp.pick(cands)
        if not picks:
            ui.notify("No eligible alphas.", type="warning")
            return
        ids = [p.alpha.id for p in picks]
        ui.notify("Suggested: " + ", ".join(ids) + " (different ideas, best estimates). Write each description next.",
                  type="positive", multi_line=True)
        grid.options["rowData"] = sorted(grid.options["rowData"], key=lambda r: r["id"] not in ids)
        grid.update()

    async def on_select() -> None:
        rows = await grid.get_selected_rows()
        if not rows:
            return
        aid = rows[0]["id"]
        state["selected"] = aid
        current.set_text(f"{aid}: {rows[0]['code']}")
        d = await aqp.draft(aid)
        editor.value = (d or {}).get("text") or ""
        show_grade(d)

    grid.on("selectionChanged", lambda _: on_select())

    async def write() -> None:
        aid = state["selected"]
        if not aid:
            ui.notify("Select an alpha first.", type="warning")
            return
        write_btn.props("loading")
        try:
            d = await aqp.write_description(aid)
            editor.value = d["text"]
            show_grade(d)
        except Exception as exc:
            notify_error(exc, "Could not write the description: ")
        finally:
            write_btn.props(remove="loading")

    def confirm_apply() -> None:
        aid = state["selected"]
        if not aid:
            ui.notify("Select an alpha first.", type="warning")
            return
        with ui.dialog() as dlg, ui.card().classes("ag-card p-5 w-[600px] max-w-[95vw]"):
            ui.label(f"Tag {aid} with {tag} and save this description on BRAIN?").classes("font-semibold")
            ui.label("Once the cycle closes the alpha counts as used: it can never be entered in another AQP "
                     "cycle.").classes("ag-amber text-sm")
            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat")

                async def go() -> None:
                    dlg.close()
                    apply_btn.props("loading")
                    try:
                        await aqp.apply(aid, editor.value or "", tag)
                        ui.notify(f"{aid} tagged {tag} and described on BRAIN.", type="positive")
                        await load()
                    except Exception as exc:
                        notify_error(exc, "Not applied: ")
                    finally:
                        apply_btn.props(remove="loading")

                ui.button("Apply", icon="sell", on_click=go).props("color=positive")
        dlg.open()

    await load()
