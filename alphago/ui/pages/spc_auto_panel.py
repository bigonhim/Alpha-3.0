"""SPC Lab panel: generate an SPC prompt with the AI, check it against the gates, and submit it to BRAIN.

Self-contained: it is added to the SPC Lab page with one call and uses the page only to load a prompt into the editor.
Submitting goes through the confirmation dialog here, unless the user switched on the opt-in `auto_submit` setting.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from nicegui import ui
from sqlalchemy import select

from ...brain import brain
from ...brain import spc as brain_spc
from ...config import settings
from ...db import db
from ...db.models import Job, SPCPrompt
from ...engine import spc_auto
from ...jobs.queue import jobs
from ..components import badge, card, notify_error

GATE_KIND = {"pass": "pass", "warn": "warning", "off": "", "fail": "fail", "error": "fail"}
GATE_MARK = {"pass": "✓", "warn": "⚠", "off": "–", "fail": "✗", "error": "?"}
STYLES = {"long": "Long only", "long_short": "Long and short"}


def _label_of(p: SPCPrompt) -> str:
    stats = p.stats or {}
    state = "submitted" if p.submitted else "ready" if stats.get("ready") else "not ready"
    return f"{p.name} · v{p.version} · {state}"


async def panel(get_editor: Callable[[], tuple[str, str]], on_load: Callable[[int], Awaitable[None]]) -> None:
    rules = settings.spc
    state: dict[str, Any] = {"job": None, "prompt_id": None}

    with card("w-full"):
        with ui.row().classes("items-center w-full gap-2"):
            ui.icon("auto_awesome").classes("ag-muted")
            with ui.column().classes("gap-0"):
                ui.label("Generate with AI, check, and submit").classes("ag-section-title")
                ui.label("The AI writes the prompt, runs it several times, and a failed gate sends its evidence back for "
                         "a rewrite. A prompt is only submitted after every gate passes and you confirm.").classes(
                    "ag-muted text-xs")
        with ui.row().classes("w-full gap-3 items-end no-wrap"):
            theme = ui.input("Theme / idea", value="quality momentum, large caps").classes("grow")
            n_stocks = ui.number("Stocks", value=20, min=1, max=200).classes("w-24")
            style = ui.select(STYLES, value="long", label="Portfolio").classes("w-44")
        notes = ui.input("Extra instructions (optional)", placeholder="e.g. avoid energy, max 3 per sector").classes(
            "w-full")
        with ui.row().classes("w-full gap-2 items-center"):
            gen_btn = ui.button("Generate & check", icon="auto_awesome", on_click=lambda: start(False))
            chk_btn = ui.button("Check the prompt in the editor", icon="fact_check", on_click=lambda: start(True)).props(
                "outline")
            rewrite = ui.checkbox("let the AI rewrite the editor prompt if it fails", value=False)
            ui.space()
            ui.label("Each run is one AI call. BRAIN runs the prompt on its own model, so these runs are a proxy "
                     "(form, tickers, stability), not a forecast.").classes("ag-muted text-xs")
        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
        progress.set_visibility(False)
        status = ui.label().classes("ag-muted text-sm")

        with ui.row().classes("items-end gap-2 w-full no-wrap"):
            picker = ui.select({}, label="Checked prompts (reopen a report)").classes("grow")
            ui.button("Open", icon="open_in_browser", on_click=lambda: open_selected()).props("outline")
        report_box = ui.column().classes("w-full gap-2")

        with ui.expansion("Gate settings", icon="tune").classes("w-full"):
            with ui.row().classes("gap-3 items-end flex-wrap"):
                s_runs = ui.number("Runs per round", value=rules.auto_runs, min=2, max=10).classes("w-36")
                s_rounds = ui.number("AI rewrites (0 = none)", value=rules.auto_max_rounds, min=0, max=5).classes("w-44")
                s_overlap = ui.number("Min overlap between runs (0–1)", value=rules.auto_min_overlap, min=0, max=1,
                                      step=0.05).classes("w-56")
                s_prices = ui.switch("Check tickers on Yahoo", value=rules.auto_check_prices)
                s_priced = ui.number("Min share priced (0–1)", value=rules.auto_min_priced, min=0, max=1,
                                     step=0.05).classes("w-48")
                s_back = ui.number("Backtest days (0 = off)", value=rules.auto_backtest_days, min=0).classes("w-44")
                s_excess = ui.number("Backtest min excess return", value=rules.auto_min_excess, step=0.005).classes("w-52")
            ui.label("A gate that cannot run (for example Yahoo is down) counts as not passed. The backtest is off by "
                     "default: the AI may remember how that period went.").classes("ag-muted text-xs")

        with ui.expansion("Submit settings (BRAIN publishes no SPC endpoint)", icon="send").classes("w-full"):
            ui.label("Submit the SPC form on BRAIN by hand once with the browser's developer tools open (F12 → Network), "
                     "then either paste the request into Import (right-click it → Copy → Copy as cURL or Copy as "
                     "fetch), or copy its path and JSON body below by hand. Use {prompt}, {name}, {model}, {version}, "
                     "{frequency}, {weight} and {competition} where the values go. Nothing is sent until you confirm a "
                     "submission. See SPC_SUBMISSION_GUIDE.md for every step.").classes("ag-muted text-xs")
            with ui.row().classes("w-full gap-2 items-end no-wrap"):
                captured = ui.textarea("Import a captured request (Copy as cURL / Copy as fetch)",
                                       placeholder="curl 'https://api.worldquantbrain.com/…' -X POST --data-raw '{…}'"
                                       ).classes("grow ag-mono").props("outlined autogrow")
                ui.button("Import", icon="download", on_click=lambda: do_import()).props("outline")
            ui.label("Cookies and headers in the pasted request are ignored: AlphaGo signs in itself. Only the path, "
                     "method and body are kept.").classes("ag-muted text-xs")
            with ui.row().classes("gap-3 items-end flex-wrap"):
                e_path = ui.input("Path (relative, starts with /)", value=rules.submit_path).classes("w-96")
                e_method = ui.select(list(brain_spc.METHODS), value=rules.submit_method.upper(), label="Method").classes(
                    "w-28")
                e_comp = ui.input("Competition id (fills {competition})", value=rules.competition_id).classes("w-64")
                e_max = ui.number("Max submissions per day", value=rules.max_submits_per_day, min=1).classes("w-52")
            e_body = ui.textarea("JSON body", value=rules.submit_body).classes("w-full ag-mono").props("outlined autogrow")
            e_auto = ui.switch("Submit automatically once every gate passes (no confirmation click)",
                               value=rules.auto_submit)
            ui.label("Off by default. When on, 'Generate & check' sends the prompt to BRAIN by itself as soon as every "
                     "gate has passed, within the daily limit above. Use it only after a hand-checked submission "
                     "worked and Preview request matched.").classes("ag-amber text-xs")
            e_state = ui.label().classes("text-sm")
            with ui.row().classes("gap-2"):
                ui.button("Find my competitions", icon="search", on_click=lambda: find_competitions()).props("outline")
                ui.button("Look for the SPC (read-only)", icon="travel_explore", on_click=lambda: discover()).props(
                    "flat")

    # ------------------------------------------------------------ settings
    def show_endpoint_state() -> None:
        problems = brain_spc.validate_endpoint(rules)
        e_state.set_text("✓ Endpoint settings are valid." if not problems else "✗ " + problems[0])
        e_state.classes(replace="text-sm " + ("ag-green" if not problems else "ag-amber"))

    def save_settings() -> None:
        rules.auto_runs, rules.auto_max_rounds = max(2, int(s_runs.value or 2)), int(s_rounds.value or 0)
        rules.auto_min_overlap, rules.auto_check_prices = float(s_overlap.value or 0), bool(s_prices.value)
        rules.auto_min_priced, rules.auto_backtest_days = float(s_priced.value or 0), int(s_back.value or 0)
        rules.auto_min_excess = float(s_excess.value or 0)
        rules.auto_submit = bool(e_auto.value)
        rules.submit_path, rules.submit_method = (e_path.value or "").strip(), (e_method.value or "POST")
        rules.competition_id = (e_comp.value or "").strip()
        rules.submit_body, rules.max_submits_per_day = e_body.value or "{}", max(1, int(e_max.value or 1))
        settings.save()
        show_endpoint_state()

    for el in (s_runs, s_rounds, s_overlap, s_prices, s_priced, s_back, s_excess, e_path, e_method, e_comp, e_max,
               e_body, e_auto):
        el.on_value_change(lambda _: save_settings())
    show_endpoint_state()

    def do_import() -> None:
        try:
            got = brain_spc.parse_captured_request(captured.value or "")
        except ValueError as exc:
            ui.notify(str(exc), type="warning", multi_line=True)
            return
        if got["method"] not in brain_spc.METHODS:
            ui.notify(f"That request is a {got['method']}, not a submission. Copy the request sent when you pressed "
                      "Submit.", type="warning", multi_line=True)
            return
        if got["host"] and "worldquantbrain.com" not in got["host"]:
            ui.notify(f"That request went to {got['host']}, not BRAIN's API. Only BRAIN API paths are used.",
                      type="warning", multi_line=True)
        e_path.value, e_method.value = got["path"], got["method"]
        if got["competition_id"]:
            e_comp.value = got["competition_id"]
        e_body.value = got["submit_body"]
        save_settings()
        mapped = ", ".join(f"{k}→{{{v}}}" for k, v in got["mapped"].items())
        kept = ", ".join(got["unmapped"][:8])
        ui.notify(f"Imported {got['method']} {got['path']}. Filled: {mapped}." + (f" Kept as captured: {kept}." if kept
                                                                                 else "") + " Check Preview request "
                  "before your first submission.", type="positive", multi_line=True, timeout=12000)

    async def find_competitions() -> None:
        try:
            rows = await brain_spc.SPCAPI(brain.client).competitions()
        except Exception as exc:
            notify_error(exc, "Could not read your competitions: ")
            return
        with ui.dialog() as dlg, ui.card().classes("ag-card p-6 w-[760px] max-w-[90vw]"):
            ui.label("Competitions your account is in").classes("text-lg font-semibold")
            if not rows:
                ui.label("BRAIN lists no competitions for your account. Join the SPC on the website first, or the SPC "
                         "may not be a competition: then leave the id empty.").classes("ag-muted text-sm")
            for r in rows:
                with ui.row().classes("items-center gap-3 w-full"):
                    ui.label(f"{r['id']} · {r['name'] or ''} · {r['status'] or ''}").classes("text-sm grow")

                    def use(cid: Any = r["id"]) -> None:
                        e_comp.value = str(cid)
                        save_settings()
                        dlg.close()

                    ui.button("Use", on_click=use).props("flat dense")
            ui.button("Close", on_click=dlg.close).props("flat")
        dlg.open()

    async def discover() -> None:
        try:
            found = await brain_spc.SPCAPI(brain.client).discover()
        except Exception as exc:
            notify_error(exc, "Could not read from BRAIN: ")
            return
        with ui.dialog() as dlg, ui.card().classes("ag-card p-6 w-[760px] max-w-[90vw]"):
            ui.label("What BRAIN lists for your account").classes("text-lg font-semibold")
            ui.label("Read-only. Look for the SPC or its id; the request itself is easiest to copy from the browser's "
                     "Network tab.").classes("ag-muted text-xs")
            ui.code(json.dumps(found, indent=1)[:6000], language="json").classes("w-full")
            ui.button("Close", on_click=dlg.close).props("flat")
        dlg.open()

    # ------------------------------------------------------------ running
    async def start(from_editor: bool) -> None:
        if state["job"] and await _job_status(state["job"]) in ("queued", "running"):
            ui.notify("A generation is already running.", type="warning")
            return
        params: dict[str, Any]
        brief = {"theme": theme.value, "n_stocks": n_stocks.value, "style": style.value, "notes": notes.value}
        if from_editor:
            nm, prompt = get_editor()
            if not (prompt or "").strip():
                ui.notify("Write or load a prompt in the editor first.", type="warning")
                return
            params = {"prompt": prompt, "name": nm or None, "revise": bool(rewrite.value), "brief": brief}
            title = f"SPC check: {nm or 'editor prompt'}"
        else:
            if not (theme.value or "").strip():
                ui.notify("Describe the idea first.", type="warning")
                return
            params = {"brief": brief}
            title = f"SPC generate: {theme.value[:60]}"
        from ...llm import llm
        if not llm.configure():
            ui.notify("Set up an AI provider on the LLM Integration page first.", type="warning")
            return
        state["job"] = await jobs.submit("spc_generate", title, params)
        state["started"] = datetime.now(timezone.utc)
        gen_btn.props("disable")
        chk_btn.props("disable")
        progress.set_visibility(True)
        status.set_text("Queued…")
        ui.notify("Generating and checking: follow progress here or in Tasks.", type="info")

    async def _job_status(job_id: int) -> str:
        async with db.session() as s:
            job = await s.get(Job, job_id)
        return job.status if job else "gone"

    async def tick() -> None:
        job_id = state["job"]
        if not job_id:
            return
        async with db.session() as s:
            job = await s.get(Job, job_id)
        if job is None:
            state["job"] = None
            return
        progress.set_value(job.progress or 0)
        status.set_text(job.message or job.status)
        if job.status in ("queued", "running"):
            return
        state["job"] = None
        progress.set_visibility(False)
        gen_btn.props(remove="disable")
        chk_btn.props(remove="disable")
        shown = state.pop("show_after", None)            # set for a submission: reopen that prompt's report
        if job.status == "failed":
            status.set_text(f"Failed: {job.error}")
            ui.notify(f"SPC job failed: {job.error}", type="negative", multi_line=True, close_button=True)
        elif job.status == "cancelled":
            status.set_text("Cancelled.")
        else:
            status.set_text(job.message or "Done.")
        await refresh_picker()
        pid = shown or (await _latest_checked(state.get("started")) if job.status == "done" else None)
        if pid:
            await show_report(pid)

    async def _latest_checked(since: datetime | None) -> int | None:
        async with db.session() as s:
            rows = list(await s.scalars(select(SPCPrompt).order_by(SPCPrompt.id.desc()).limit(20)))
        for p in rows:
            gen = (p.stats or {}).get("generated_at")
            if gen and (since is None or datetime.fromisoformat(gen) >= since):
                return p.id
        return None

    ui.timer(1.5, tick)

    # ------------------------------------------------------------ report
    async def refresh_picker() -> None:
        async with db.session() as s:
            rows = list(await s.scalars(select(SPCPrompt).order_by(SPCPrompt.id.desc()).limit(100)))
        picker.options = {p.id: _label_of(p) for p in rows if (p.stats or {}).get("gates")}
        picker.update()

    async def open_selected() -> None:
        if picker.value:
            await show_report(int(picker.value))
        else:
            ui.notify("Pick a checked prompt first.", type="warning")

    async def show_report(prompt_id: int) -> None:
        async with db.session() as s:
            p = await s.get(SPCPrompt, prompt_id)
        state["prompt_id"] = prompt_id
        report_box.clear()
        if p is None or not (p.stats or {}).get("gates"):
            return
        st = p.stats
        used = await spc_auto.submissions_today()
        endpoint_problems = brain_spc.validate_endpoint(rules)
        can_submit = bool(st.get("ready")) and not p.submitted
        with report_box:
            with ui.row().classes("items-center gap-3"):
                ui.label(f"{p.name} · v{p.version}").classes("text-base font-semibold")
                if p.submitted:
                    badge("submitted", "pass")
                else:
                    badge("ready to submit" if st.get("ready") else "not ready", "pass" if st.get("ready") else "fail")
                ui.label(f"{st.get('rounds', 1)} round(s) · {st.get('runs', '?')} runs · source {st.get('source', '?')}"
                         ).classes("ag-muted text-xs")
            for g in st["gates"]:
                with ui.row().classes("items-center gap-2 no-wrap"):
                    badge(f"{GATE_MARK.get(g['status'], '?')} {g['name']}", GATE_KIND.get(g["status"], "")).classes("w-44")
                    ui.label(g.get("detail", "")).classes("text-sm")
            if st.get("rationale"):
                ui.label(f"Approach: {st['rationale']}").classes("ag-muted text-sm")
            if st.get("submission"):
                sub = st["submission"]
                ui.label(f"Last submission: {sub.get('status')} — {sub.get('message', '')}").classes(
                    "text-sm " + ("ag-green" if sub.get("status") == "accepted" else "ag-amber"))
            if not st.get("ready"):
                ui.label("Not submittable: fix the failed gates above (edit the prompt in the editor, then use "
                         "'Check the prompt in the editor', or generate again).").classes("ag-amber text-sm")
            elif endpoint_problems and not p.submitted:
                ui.label("Passed every gate, but submission is not set up: " + endpoint_problems[0]).classes(
                    "ag-amber text-sm")
            with ui.row().classes("gap-2"):
                ui.button("Load into editor", icon="edit", on_click=lambda: on_load(prompt_id)).props("outline")
                ui.button("Copy prompt", icon="content_copy",
                          on_click=lambda: (ui.clipboard.write(p.prompt),
                                            ui.notify("Prompt copied: paste it into the Prompt box on BRAIN's SPC "
                                                      "form."))).props("outline")
                ui.button("Preview request", icon="visibility", on_click=lambda: preview(prompt_id)).props(
                    "outline" + ("" if can_submit and not endpoint_problems else " disable"))
                ui.button("Submit to BRAIN…", icon="send", on_click=lambda: confirm_submit(prompt_id)).props(
                    "color=positive" + ("" if can_submit and not endpoint_problems
                                        and used < rules.max_submits_per_day else " disable"))
            if can_submit and used >= rules.max_submits_per_day:
                ui.label(f"Daily guard reached: {used}/{rules.max_submits_per_day} submissions today.").classes(
                    "ag-amber text-xs")

    async def preview(prompt_id: int) -> None:
        try:
            req = await spc_auto.preview_request(prompt_id)
        except Exception as exc:
            notify_error(exc, "")
            return
        with ui.dialog() as dlg, ui.card().classes("ag-card p-6 w-[760px] max-w-[90vw]"):
            ui.label("This is the request that would be sent").classes("text-lg font-semibold")
            ui.label("Nothing has been sent.").classes("ag-muted text-xs")
            ui.code(f"{req['method']} {req['path']}\n" + json.dumps(req["body"], indent=1)[:6000], language="json").classes(
                "w-full")
            ui.button("Close", on_click=dlg.close).props("flat")
        dlg.open()

    async def confirm_submit(prompt_id: int) -> None:
        try:
            req = await spc_auto.preview_request(prompt_id)
        except Exception as exc:
            notify_error(exc, "")
            return
        async with db.session() as s:
            p = await s.get(SPCPrompt, prompt_id)
        used = await spc_auto.submissions_today()
        with ui.dialog() as dlg, ui.card().classes("ag-card p-6 w-[760px] max-w-[90vw]"):
            ui.label(f"Submit {p.name} · v{p.version} to WorldQuant BRAIN?").classes("text-lg font-semibold")
            ui.label(p.prompt[:1500] + ("…" if len(p.prompt) > 1500 else "")).classes(
                "ag-mono text-sm whitespace-pre-wrap break-words")
            with ui.row().classes("gap-1 flex-wrap"):
                for g in (p.stats or {}).get("gates", []):
                    badge(f"{GATE_MARK.get(g['status'], '?')} {g['name']}", GATE_KIND.get(g["status"], ""))
            ui.label(f"Request: {req['method']} {req['path']}").classes("ag-mono text-xs ag-muted")
            ui.label(f"Today's SPC submissions: {used}/{rules.max_submits_per_day}. Submission is final and visible "
                     "to WorldQuant. The endpoint comes from your settings, so check that the preview matches what the "
                     "browser form sends.").classes("ag-muted text-sm")

            async def go() -> None:
                dlg.close()
                state["job"] = await jobs.submit("spc_submit", f"Submit SPC prompt {p.name} v{p.version}",
                                                 {"prompt_id": prompt_id, "confirmed": True}, priority=1)
                state["show_after"] = prompt_id
                gen_btn.props("disable")
                chk_btn.props("disable")
                progress.set_visibility(True)
                status.set_text("Submitting to BRAIN…")
                ui.notify("Submitting: follow progress here or in Tasks.", type="info")

            with ui.row().classes("justify-end w-full gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat")
                ui.button("Submit to BRAIN", icon="send", on_click=go).props("color=positive")
        dlg.open()

    await refresh_picker()
