"""LLM Integration — Google AI Studio (Gemini) key, models per role, pacing, usage and prompt templates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nicegui import ui
from sqlalchemy import func, select

from ... import credentials
from ...config import settings
from ...db import db
from ...db.models import LLMCall
from ...knowledge.pack import static_prefix
from ...llm import LLMError, gemini
from ...llm.prompts import DEFAULTS, get_template, set_template
from ..components import card, fmt_int, notify_error, page_header


async def page() -> None:
    page_header("LLM Integration", "Google AI Studio (Gemini). The key is stored in Windows Credential Manager; "
                                   "rate limits are learned automatically from 429 responses.")
    cfg = settings.llm

    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        with card("w-[48%]"):
            ui.label("API key & models").classes("ag-section-title")
            has_key = bool(credentials.gemini_key())
            key_in = ui.input("Gemini API key (from aistudio.google.com)", password=True, password_toggle_button=True,
                              placeholder="●●●●●●●● saved" if has_key else "").classes("w-full")
            status = ui.label().classes("text-sm")

            async def save_key() -> None:
                if key_in.value:
                    credentials.set_gemini_key(key_in.value.strip())
                    key_in.value = ""
                gemini.configure(credentials.gemini_key())
                await test()

            async def test() -> None:
                test_btn.props("loading")
                try:
                    msg = await gemini.test()
                    ui.notify(msg, type="positive", multi_line=True)
                    render_models()
                except LLMError as exc:
                    notify_error(exc)
                except Exception as exc:
                    notify_error(exc, "Gemini test failed: ")
                finally:
                    test_btn.props(remove="loading")

            def forget_key() -> None:
                credentials.set_gemini_key(None)
                gemini.configure(None)
                ui.notify("Gemini key removed.")

            with ui.row().classes("gap-2"):
                ui.button("Save & test", icon="key", on_click=save_key)
                test_btn = ui.button("Test connection", icon="bolt", on_click=test).props("flat")
                ui.button("Forget key", icon="key_off", on_click=forget_key).props("flat color=grey")

            models_box = ui.column().classes("w-full gap-2 mt-2")

            def render_models() -> None:
                models_box.clear()
                names = [m["name"] for m in gemini.models]
                options = {"": "Auto (best available)"} | {n: n for n in names}
                with models_box:
                    if not names:
                        ui.label("Models appear here after a successful test.").classes("ag-muted text-sm")
                    for role, attr, help_ in (("Ideation (bulk ideas)", "ideation_model", "fast 'flash' model"),
                                              ("Refinement (near misses)", "refine_model", "strongest model"),
                                              ("Descriptions", "describe_model", "cheapest model")):
                        sel = ui.select(options, value=getattr(cfg, attr) if getattr(cfg, attr) in options else "",
                                        label=f"{role} — auto: {gemini.pick(attr.split('_')[0])}").classes("w-full")
                        sel.on_value_change(lambda e, a=attr: (setattr(cfg, a, e.value or ""), settings.save()))
                        sel.tooltip(help_)
                    think = ui.select({"minimal": "minimal", "low": "low", "medium": "medium", "high": "high",
                                       "off": "off (older models)"}, value=cfg.thinking_level,
                                      label="Thinking level (reasoning depth)").classes("w-64")
                    think.on_value_change(lambda e: (setattr(cfg, "thinking_level", e.value), settings.save()))

            render_models()

        with card("grow"):
            ui.label("Pacing & budget").classes("ag-section-title")
            pace = ui.label().classes("text-sm")
            ideas = ui.number("Ideas per Gemini call", value=cfg.ideas_per_call, min=1, max=30).classes("w-56")
            ideas.on_value_change(lambda e: (setattr(cfg, "ideas_per_call", int(e.value or 8)), settings.save()))
            budget = ui.number("Daily call budget", value=cfg.daily_call_budget, min=10).classes("w-56")
            budget.on_value_change(lambda e: (setattr(cfg, "daily_call_budget", int(e.value or 400)), settings.save()))
            start_rpm = ui.number("Starting requests / minute", value=cfg.start_rpm, min=0.5, step=0.5).classes("w-56")
            start_rpm.on_value_change(lambda e: (setattr(cfg, "start_rpm", float(e.value or 6)), settings.save()))
            lang = ui.select(["English", "Hinglish"], value=cfg.ai_language,
                             label="Language of AI explanations (Osmosis Review)").classes("w-72")
            lang.on_value_change(lambda e: (setattr(cfg, "ai_language", e.value), settings.save()))
            usage = ui.label().classes("ag-muted text-sm")
            chart_box = ui.column().classes("w-full")

            async def refresh() -> None:
                status.set_text(f"Status: {gemini.status}" + (f" — {gemini.last_error}" if gemini.last_error else ""))
                status.classes(replace="text-sm " + {"ok": "ag-green", "rate_limited": "ag-amber",
                                                      "error": "ag-red"}.get(gemini.status, "ag-muted"))
                pace.set_text(f"Current pacing: {gemini.rpm:.1f} requests/min (adaptive)")
                since = datetime.now(timezone.utc) - timedelta(days=14)
                async with db.session() as s:
                    today = await gemini.calls_today()
                    rows = (await s.execute(select(func.date(LLMCall.ts), func.count(LLMCall.id),
                                                   func.sum(LLMCall.prompt_tokens), func.sum(LLMCall.output_tokens))
                                            .where(LLMCall.ts >= since).group_by(func.date(LLMCall.ts)))).all()
                    errors = await s.scalar(select(func.count(LLMCall.id)).where(LLMCall.ts >= since,
                                                                                 LLMCall.status != "ok"))
                usage.set_text(f"Calls today: {fmt_int(today)} / {fmt_int(cfg.daily_call_budget)} · "
                               f"errors (14 d): {fmt_int(errors)}")
                chart_box.clear()
                if rows:
                    with chart_box:
                        ui.echart({"backgroundColor": "transparent", "tooltip": {"trigger": "axis"},
                                   "legend": {"textStyle": {"color": "#8b98ad"}},
                                   "grid": {"left": 60, "right": 50, "top": 30, "bottom": 30},
                                   "xAxis": {"type": "category", "data": [str(r[0]) for r in rows],
                                             "axisLabel": {"color": "#8b98ad"}},
                                   "yAxis": [{"type": "value", "name": "calls", "axisLabel": {"color": "#8b98ad"}},
                                             {"type": "value", "name": "tokens", "axisLabel": {"color": "#8b98ad"}}],
                                   "series": [{"name": "calls", "type": "bar", "data": [r[1] for r in rows]},
                                              {"name": "tokens", "type": "line", "yAxisIndex": 1,
                                               "data": [(r[2] or 0) + (r[3] or 0) for r in rows]}]}).classes("w-full h-56")

            await refresh()
            ui.timer(10.0, refresh)

    with card("w-full"):
        with ui.row().classes("items-center w-full"):
            ui.label("Prompt templates").classes("ag-section-title")
            ui.label("Jinja2 templates. The knowledge pack (rules, your operators, thresholds, templates) is sent as "
                     "the system instruction so Gemini can cache it.").classes("ag-muted text-sm")
        with ui.row().classes("w-full gap-4 no-wrap"):
            which = ui.select(list(DEFAULTS), value="ideation", label="Template").classes("w-56")
            editor = ui.textarea().classes("grow ag-mono").props("outlined autogrow input-style='min-height:260px'")
        with ui.row().classes("gap-2"):
            async def load() -> None:
                editor.value = await get_template(which.value)

            async def save() -> None:
                await set_template(which.value, editor.value if editor.value != DEFAULTS[which.value] else None)
                ui.notify("Template saved.", type="positive")

            async def reset() -> None:
                await set_template(which.value, None)
                editor.value = DEFAULTS[which.value]
                ui.notify("Reset to default.")

            async def show_pack() -> None:
                text = await static_prefix()
                with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[1000px] max-w-[95vw]"):
                    ui.label(f"Knowledge pack — {len(text):,} characters (~{len(text) // 4:,} tokens)").classes(
                        "font-semibold")
                    ui.textarea(value=text).classes("w-full ag-mono").props("readonly autogrow")
                d.open()

            ui.button("Save template", icon="save", on_click=save)
            ui.button("Reset to default", icon="restart_alt", on_click=reset).props("flat")
            ui.button("View knowledge pack", icon="menu_book", on_click=show_pack).props("flat")
        which.on_value_change(lambda _: load())
        await load()
