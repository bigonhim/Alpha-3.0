"""LLM Integration — the active AI provider (Gemini, Claude, OpenAI / compatible, local Ollama), its key or server,
models per role, pacing, usage and prompt templates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nicegui import ui
from sqlalchemy import func, select

from ... import credentials
from ...config import settings
from ...db import db
from ...db.models import LLMCall
from ...knowledge.pack import static_prefix
from ...llm import PROVIDER_LABELS, PROVIDERS, LLMError, llm
from ...llm.prompts import DEFAULTS, get_template, set_template
from ..components import card, fmt_int, notify_error, page_header

_KEY_LABEL = {"gemini": "Gemini API key (from aistudio.google.com)",
              "claude": "Anthropic API key (from console.anthropic.com)",
              "openai": "API key (leave empty for a local server that needs none)"}
_URL_LABEL = {"openai": "Server URL (OpenAI, or OpenRouter, Groq, DeepSeek, LM Studio, vLLM …)",
              "ollama": "Ollama server URL"}
_ROLE_HELP = {"gemini": ("fast 'flash' model", "strongest model", "cheapest model"),
              "claude": ("auto: Claude Opus 5.5 — Sonnet 5.5 / Haiku 5.5 cost less",) * 3,
              "openai": ("any chat model the server lists",) * 3,
              "ollama": ("auto: first installed model",) * 3}


def _role_cfg(provider: str):  # noqa: ANN202
    """Where a provider's per-role model choices live (Gemini's stay at the top level of settings.llm)."""
    return settings.llm if provider == "gemini" else getattr(settings.llm, provider)


async def page() -> None:
    page_header("LLM Integration", "Choose the AI provider — Google Gemini, Anthropic Claude, OpenAI (or any "
                                   "OpenAI-compatible server) or a local Ollama model. Keys are stored in Windows "
                                   "Credential Manager; cloud rate limits are learned automatically from 429 responses.")
    cfg = settings.llm

    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        with card("w-[48%]"):
            ui.label("Provider, key & models").classes("ag-section-title")
            choice = ui.select(PROVIDER_LABELS, value=cfg.provider if cfg.provider in PROVIDERS else "gemini",
                               label="Active provider (used by every AI feature)").classes("w-80")
            status = ui.label().classes("text-sm")
            body = ui.column().classes("w-full gap-2")

            def render() -> None:
                body.clear()
                name = choice.value
                provider = PROVIDERS[name]
                with body:
                    url_in = key_in = ctx_in = None
                    if name in _URL_LABEL:
                        url_in = ui.input(_URL_LABEL[name], value=getattr(cfg, name).base_url).classes("w-full")
                    if name in _KEY_LABEL:
                        key_in = ui.input(_KEY_LABEL[name], password=True, password_toggle_button=True,
                                          placeholder="●●●●●●●● saved" if credentials.llm_key(name) else "").classes(
                            "w-full")
                    if name == "ollama":
                        ctx_in = ui.number("Context window (tokens)", value=cfg.ollama.num_ctx, min=2048,
                                           step=1024).classes("w-56")
                        ui.label("Install Ollama from ollama.com and download a model with \"ollama pull <model>\". "
                                 "Calls run on this PC, one at a time, and don't count toward the daily budget. "
                                 "The knowledge pack plus a dataset's fields can need 15–20k tokens of context.").classes(
                            "ag-muted text-sm")

                    async def save_and_test() -> None:
                        if url_in is not None:
                            getattr(cfg, name).base_url = (url_in.value or "").strip()
                        if ctx_in is not None:
                            cfg.ollama.num_ctx = int(ctx_in.value or 32_768)
                        settings.save()
                        if key_in is not None and key_in.value:
                            credentials.set_llm_key(name, key_in.value.strip())
                            key_in.value = ""
                        provider.configure(credentials.llm_key(name))
                        await test()

                    async def test() -> None:
                        test_btn.props("loading")
                        try:
                            msg = await provider.test()
                            ui.notify(msg, type="positive", multi_line=True)
                            render_models()
                        except LLMError as exc:
                            notify_error(exc)
                        except Exception as exc:
                            notify_error(exc, f"{provider.label} test failed: ")
                        finally:
                            test_btn.props(remove="loading")

                    def forget_key() -> None:
                        credentials.set_llm_key(name, None)
                        provider.configure(None)
                        ui.notify(f"{provider.label} key removed.")

                    with ui.row().classes("gap-2"):
                        ui.button("Save & test", icon="key", on_click=save_and_test)
                        test_btn = ui.button("Test connection", icon="bolt", on_click=test).props("flat")
                        if name in _KEY_LABEL:
                            ui.button("Forget key", icon="key_off", on_click=forget_key).props("flat color=grey")

                    models_box = ui.column().classes("w-full gap-2 mt-2")

                    def render_models() -> None:
                        models_box.clear()
                        rc = _role_cfg(name)
                        names = [m["name"] for m in provider.models]
                        chosen = [getattr(rc, a) for a in ("ideation_model", "refine_model", "describe_model")]
                        options = {"": "Auto"} | {n: n for n in names} | {c: c for c in chosen if c}
                        with models_box:
                            if not names:
                                ui.label("Models appear here after a successful test." +
                                         (" You can also type a model name." if name in ("openai", "ollama") else "")
                                         ).classes("ag-muted text-sm")
                            for (role, attr), help_ in zip((("Ideation (bulk ideas)", "ideation_model"),
                                                            ("Refinement (near misses)", "refine_model"),
                                                            ("Descriptions", "describe_model")), _ROLE_HELP[name]):
                                auto = provider.pick(attr.split("_")[0]) or "choose a model"
                                sel = ui.select(options, value=getattr(rc, attr) if getattr(rc, attr) in options else "",
                                                label=f"{role} — auto: {auto}", with_input=True,
                                                new_value_mode="add-unique" if name in ("openai", "ollama") else None
                                                ).classes("w-full")
                                sel.on_value_change(lambda e, a=attr: (setattr(rc, a, e.value or ""), settings.save()))
                                sel.tooltip(help_)
                            think = ui.select({"minimal": "minimal", "low": "low", "medium": "medium", "high": "high",
                                               "off": "off (older models)"}, value=cfg.thinking_level,
                                              label="Thinking level (reasoning depth)").classes("w-64")
                            think.on_value_change(lambda e: (setattr(cfg, "thinking_level", e.value), settings.save()))

                    render_models()

            def switch(e) -> None:  # noqa: ANN001
                cfg.provider = e.value
                settings.save()
                PROVIDERS[e.value].configure()
                ui.notify(f"AI features now use {PROVIDERS[e.value].label}.")
                render()

            choice.on_value_change(switch)
            render()

        with card("grow"):
            ui.label("Pacing & budget").classes("ag-section-title")
            pace = ui.label().classes("text-sm")
            ideas = ui.number("Ideas per AI call", value=cfg.ideas_per_call, min=1, max=30).classes("w-56")
            ideas.on_value_change(lambda e: (setattr(cfg, "ideas_per_call", int(e.value or 8)), settings.save()))
            budget = ui.number("Daily call budget (cloud providers)", value=cfg.daily_call_budget, min=10).classes(
                "w-64")
            budget.on_value_change(lambda e: (setattr(cfg, "daily_call_budget", int(e.value or 400)), settings.save()))
            start_rpm = ui.number("Starting requests / minute", value=cfg.start_rpm, min=0.5, step=0.5).classes("w-56")
            start_rpm.on_value_change(lambda e: (setattr(cfg, "start_rpm", float(e.value or 6)), settings.save()))
            lang = ui.select(["English", "Hinglish"], value=cfg.ai_language,
                             label="Language of AI explanations (Osmosis Review)").classes("w-72")
            lang.on_value_change(lambda e: (setattr(cfg, "ai_language", e.value), settings.save()))
            usage = ui.label().classes("ag-muted text-sm")
            chart_box = ui.column().classes("w-full")

            async def refresh() -> None:
                p = llm.active
                status.set_text(f"{p.label} · status: {p.status}" + (f" — {p.last_error}" if p.last_error else ""))
                status.classes(replace="text-sm " + {"ok": "ag-green", "rate_limited": "ag-amber",
                                                      "error": "ag-red"}.get(p.status, "ag-muted"))
                pace.set_text("Local model: no rate limit, one request at a time" if p.local
                              else f"Current pacing: {p.rpm:.1f} requests/min (adaptive)")
                since = datetime.now(timezone.utc) - timedelta(days=14)
                async with db.session() as s:
                    today = await llm.calls_today()
                    rows = (await s.execute(select(func.date(LLMCall.ts), func.count(LLMCall.id),
                                                   func.sum(LLMCall.prompt_tokens), func.sum(LLMCall.output_tokens))
                                            .where(LLMCall.ts >= since).group_by(func.date(LLMCall.ts)))).all()
                    errors = await s.scalar(select(func.count(LLMCall.id)).where(LLMCall.ts >= since,
                                                                                 LLMCall.status != "ok"))
                usage.set_text(f"Cloud calls today: {fmt_int(today)} / {fmt_int(cfg.daily_call_budget)} · "
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
                     "the system instruction so the provider can cache it.").classes("ag-muted text-sm")
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
