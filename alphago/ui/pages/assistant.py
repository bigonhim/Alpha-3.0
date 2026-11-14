"""AI Assistant (ported from alpha-harness `screens/ai/assistant.tsx`).

Describe a hunch in your own words; the assistant answers with data fields that really exist in the chosen market
(checked against the local catalogue — anything it invented is shown as dropped). "Use in Search Lab" hands the
picks to Search Lab. One request per message, no streaming.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui
from sqlalchemy import func, select

from ...ai import chat, service
from ...ai.keys import keystore
from ...ai.registry import registry
from ...brain import brain
from ...config import settings
from ...db import db
from ...db.models import DataField, Dataset
from ...llm import LLMError, llm
from ..components import ago, badge, card, empty_state, fmt_int, notify_error, page_header

PREF = "pref:assistant"
EXAMPLES = ["Companies whose analysts keep raising their earnings estimates",
            "Stocks with high operating cash flow relative to enterprise value",
            "Short-term reversal after a spike in trading volume"]


async def _datasets(scope: chat.Scope) -> dict[str, str]:
    async with db.session() as s:
        rows = (await s.execute(select(Dataset.id, Dataset.name).where(
            Dataset.region == scope.region, Dataset.delay == scope.delay, Dataset.universe == scope.universe)
            .order_by(Dataset.id))).all()
    return {r[0]: f"{r[0]} — {r[1]}" if r[1] else r[0] for r in rows}


async def _field_count(scope: chat.Scope) -> int:
    async with db.session() as s:
        return await s.scalar(select(func.count(DataField.pk)).where(
            DataField.region == scope.region, DataField.delay == scope.delay,
            DataField.universe == scope.universe)) or 0


async def page() -> None:
    page_header("AI Assistant", "Talk through an idea; the assistant picks data fields from your catalogue to try it "
                                "with, and hands them to Search Lab.")
    await registry.load()
    await keystore.load()
    if not llm.configure():
        with card("w-full"):
            empty_state("key", "The assistant needs a key",
                        "Add a free key from any provider under LLM Integration › Keys (it takes a minute and needs "
                        "no card), or set up a local Ollama model. Then describe an idea in your own words and get "
                        "back real data fields.")
            ui.button("Open LLM Integration", icon="auto_awesome", on_click=lambda: ui.navigate.to("/llm"))
        return

    pref = await db.kv_get(PREF) or {}
    res = settings.research
    state: dict[str, Any] = {
        "thread": None, "region": pref.get("region") or res.region, "delay": int(pref.get("delay", res.delay)),
        "universe": pref.get("universe") or res.universe, "model": pref.get("model") or "",
        "reasoning": pref.get("reasoning") or chat.DEFAULT_REASONING, "datasets": [], "busy": False, "error": None}

    def scope() -> chat.Scope:
        return chat.Scope(state["region"], int(state["delay"]), state["universe"])

    async def remember() -> None:
        await db.kv_set(PREF, {k: state[k] for k in ("region", "delay", "universe", "model", "reasoning")})

    opts = brain.options
    runnable = await service.runnable()
    model_options = {"": f"Auto — {llm.label} ({llm.pick('ideation') or 'its own pick'})"} | {
        m["ref"]: m["label"] + (f" · {fmt_int(m['remaining_today'])} left today"
                                if m["remaining_today"] is not None else "") for m in runnable}
    if state["model"] not in model_options:
        state["model"] = ""

    with ui.row().classes("w-full gap-4 no-wrap items-start"):
        with card("w-[300px] shrink-0"):
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("Conversations").classes("ag-section-title")
                ui.button("New", icon="add", on_click=lambda: open_thread(None)).props("flat dense")
            thread_list = ui.column().classes("w-full gap-1")

        with card("grow"):
            title = ui.label("New conversation").classes("text-lg font-semibold")
            ui.label("Describe a hunch in your own words. The assistant answers with fields that really exist in this "
                     "market.").classes("ag-muted text-sm")
            with ui.row().classes("items-end gap-3 w-full"):
                region = ui.select(opts.regions or [state["region"]], value=state["region"], label="Region").classes(
                    "w-28")
                delay = ui.select(opts.delays(state["region"]) or [1, 0], value=state["delay"], label="Delay").classes(
                    "w-24")
                universe = ui.select(opts.universes(state["region"]) or [state["universe"]], value=state["universe"],
                                     label="Universe").classes("w-36")
                datasets = ui.select({}, multiple=True, value=[], with_input=True,
                                     label="Only these datasets (optional)").classes("grow").props("use-chips")
            scope_note = ui.label().classes("ag-muted text-xs")
            messages = ui.column().classes("w-full gap-3 rounded-md p-3 min-h-[280px] max-h-[60vh] overflow-auto "
                                           "border border-[#1f2a3c]")
            error_box = ui.column().classes("w-full")
            text = ui.textarea(placeholder="Describe an idea in your own words  (Ctrl+Enter sends)").classes(
                "w-full").props("outlined autogrow")
            with ui.row().classes("items-center gap-3 w-full"):
                model = ui.select(model_options, value=state["model"], label="Model").classes("w-96")
                reasoning = ui.select({k: v["label"] for k, v in chat.REASONING.items()}, value=state["reasoning"],
                                      label="Reasoning").classes("w-36")
                ui.space()
                send_btn = ui.button("Send", icon="send")
            reasoning_help = ui.label().classes("ag-muted text-xs")

    def show_reasoning_help() -> None:
        reasoning_help.set_text(f"{chat.REASONING[reasoning.value]['description']} Each message spends one assistant "
                                f"request. {chat.REASONING_NOTE}")

    async def refresh_scope_widgets() -> None:
        sc = scope()
        datasets.set_options(await _datasets(sc), value=[d for d in state["datasets"]])
        count = await _field_count(sc)
        locked = state["thread"] is not None
        for w in (region, delay, universe):
            w.set_enabled(not locked and not state["busy"])
        scope_note.set_text((f"{sc.label}: {fmt_int(count)} data fields in the local catalogue."
                             if count else f"{sc.label} is not in the local catalogue yet, so the assistant has no "
                                           "fields to choose from — sync its datasets on Data Explorer or Datasets.")
                            + (" The market is fixed once a conversation starts." if locked else ""))
        send_btn.set_enabled(bool(count) and not state["busy"])

    async def on_region(e) -> None:  # noqa: ANN001
        state["region"] = e.value
        universe.set_options(opts.universes(e.value) or [state["universe"]])
        if universe.value not in (opts.universes(e.value) or []):
            universe.value = (opts.universes(e.value) or [state["universe"]])[0]
        delay.set_options(opts.delays(e.value) or [1, 0])
        state["datasets"] = []
        await remember()
        await refresh_scope_widgets()

    async def on_scope_part(key: str, value: Any) -> None:
        state[key] = value
        state["datasets"] = []
        await remember()
        await refresh_scope_widgets()

    region.on_value_change(on_region)
    delay.on_value_change(lambda e: on_scope_part("delay", int(e.value)))
    universe.on_value_change(lambda e: on_scope_part("universe", e.value))
    datasets.on_value_change(lambda e: state.__setitem__("datasets", list(e.value or [])))

    async def on_model(e) -> None:  # noqa: ANN001
        state["model"] = e.value or ""
        await remember()

    async def on_reasoning(e) -> None:  # noqa: ANN001
        state["reasoning"] = e.value
        show_reasoning_help()
        await remember()

    model.on_value_change(on_model)
    reasoning.on_value_change(on_reasoning)
    show_reasoning_help()

    async def use_in_search_lab(meta: dict[str, Any]) -> None:
        sc = scope()
        picks = [p["field"] for p in meta.get("picks") or []]
        await chat.hand_off(sc, list(meta.get("datasets") or []), picks)
        ui.notify(f"Handed {len(picks)} fields of {sc.label} to Search Lab.", type="positive")
        ui.navigate.to("/labs")

    def render_turn(role: str, body: str, meta: dict[str, Any]) -> None:
        if role == "user":
            with ui.row().classes("w-full justify-end"):
                ui.label(body).classes("max-w-[85%] rounded-lg px-3 py-2 bg-[#1b2436] whitespace-pre-wrap")
            return
        with ui.column().classes("max-w-[92%] gap-2"):
            ui.label(body).classes("whitespace-pre-wrap")
            picks = meta.get("picks") or []
            if picks:
                with ui.column().classes("gap-1 rounded-md p-2 border border-[#1f2a3c] w-full"):
                    ui.label(f"Fields picked ({len(picks)})").classes("ag-muted text-xs")
                    with ui.row().classes("gap-1 flex-wrap"):
                        for p in picks:
                            ui.chip(p["field"], icon="dataset").props("dense outline color=cyan").tooltip(
                                p.get("why") or "")
                    for p in picks:
                        with ui.row().classes("gap-2 no-wrap items-baseline"):
                            ui.label(p["field"]).classes("ag-mono text-xs")
                            ui.label(p.get("why") or "").classes("ag-muted text-xs")
            dsets = meta.get("datasets") or []
            if dsets or picks:
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    if dsets:
                        ui.label("Datasets").classes("ag-muted text-xs")
                        for d in dsets:
                            badge(d)
                    ui.button("Use in Search Lab", icon="science",
                              on_click=lambda m=meta: use_in_search_lab(m)).props("flat dense")
            dropped = meta.get("dropped") or []
            if dropped:
                ui.label("Named by the model but not in the catalogue, so dropped: " + ", ".join(dropped)).classes(
                    "ag-amber text-xs")
            if meta.get("catalog_note"):
                ui.label(meta["catalog_note"]).classes("ag-muted text-xs")
            usage = " · ".join(str(x) for x in (meta.get("model"), meta.get("reasoning"),
                                                f"{fmt_int(meta['tokens'])} tokens" if meta.get("tokens") else None)
                               if x)
            if usage:
                ui.label(usage).classes("ag-mono ag-muted text-xs")

    async def render_messages() -> None:
        messages.clear()
        rows: list[Any] = []
        if state["thread"] is not None:
            found = await chat.thread(state["thread"])
            if found is not None:
                rows = found[1]
        with messages:
            if not rows and not state["busy"]:
                with ui.column().classes("items-center w-full gap-2 py-6"):
                    ui.label("Start with an idea").classes("text-lg")
                    ui.label(f"About {scope().label}: the market is set when the conversation starts, so choose it "
                             "above first. Each message uses one request from your assistant budget.").classes(
                        "ag-muted text-sm text-center")
                    with ui.row().classes("gap-2 flex-wrap justify-center"):
                        for ex in EXAMPLES:
                            ui.button(ex, on_click=lambda e=ex: text.set_value(e)).props("outline dense no-caps")
            for m in rows:
                render_turn(m.role, m.text, m.meta or {})
            if state["busy"]:
                with ui.row().classes("items-center gap-2"):
                    ui.spinner(size="sm")
                    ui.label("Reading the catalogue and thinking…").classes("ag-muted text-sm")
        messages.run_method("scrollTo", 0, 1_000_000)

    async def render_threads() -> None:
        thread_list.clear()
        rows = await chat.threads(50)
        with thread_list:
            if not rows:
                ui.label("No conversations yet").classes("ag-muted text-sm")
            for t in rows:
                active = t.id == state["thread"]
                with ui.row().classes("w-full items-center no-wrap gap-1 rounded-md px-2 py-1 cursor-pointer "
                                      + ("bg-[#1b2436]" if active else "")):
                    with ui.column().classes("gap-0 grow min-w-0"):
                        ui.label(t.title).classes("truncate text-sm").on("click", lambda tid=t.id: open_thread(tid))
                        ui.label(f"{t.region}/D{t.delay}/{t.universe} · {ago(t.updated_at)}").classes(
                            "ag-muted text-xs").on("click", lambda tid=t.id: open_thread(tid))
                    ui.button(icon="delete", on_click=lambda tid=t.id, tt=t.title: confirm_delete(tid, tt)).props(
                        "flat dense round size=sm")

    async def open_thread(thread_id: int | None) -> None:
        state["thread"] = thread_id
        state["error"] = None
        error_box.clear()
        if thread_id is not None:
            found = await chat.thread(thread_id)
            if found is not None:
                row = found[0]
                title.set_text(row.title)
                state.update(region=row.region, delay=int(row.delay), universe=row.universe)
                region.value, delay.value, universe.value = row.region, int(row.delay), row.universe
        else:
            title.set_text("New conversation")
        await refresh_scope_widgets()
        await render_messages()
        await render_threads()

    async def confirm_delete(thread_id: int, name: str) -> None:
        with ui.dialog() as d, ui.card().classes("ag-card p-5"):
            ui.label(f"Delete “{name}”?").classes("font-semibold")
            ui.label("The conversation and its messages are removed from this PC.").classes("ag-muted text-sm")

            async def go() -> None:
                await chat.remove(thread_id)
                d.close()
                ui.notify("Conversation deleted.")
                await open_thread(None if state["thread"] == thread_id else state["thread"])

            with ui.row():
                ui.button("Delete", color="negative", on_click=go)
                ui.button("Cancel", on_click=d.close).props("flat")
        d.open()

    async def send() -> None:
        message = (text.value or "").strip()
        if not message or state["busy"]:
            return
        state["busy"] = True
        send_btn.props("loading")
        error_box.clear()
        await render_messages()
        try:
            answer = await chat.say(state["thread"], message, scope=scope(), model=state["model"] or None,
                                    reasoning=state["reasoning"], dataset_ids=state["datasets"])
        except (chat.ChatError, LLMError) as exc:
            with error_box:
                with ui.card().classes("ag-card p-3 w-full"):
                    ui.label("The assistant could not answer").classes("ag-red font-semibold text-sm")
                    ui.label(getattr(exc, "message", None) or str(exc)).classes("text-sm")
            state["busy"] = False
            send_btn.props(remove="loading")
            await render_messages()
            return
        except Exception as exc:
            notify_error(exc, "The assistant failed: ")
            state["busy"] = False
            send_btn.props(remove="loading")
            await render_messages()
            return
        state["busy"] = False
        send_btn.props(remove="loading")
        text.value = ""
        await open_thread(answer["thread_id"])

    send_btn.on_click(send)
    text.on("keydown.ctrl.enter", send)
    await open_thread(None)
