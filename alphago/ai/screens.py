"""The AI screens on the LLM Integration page (alpha-harness `frontend/src/screens/ai`): Keys, Models, Budget,
Providers and the prompt library.

Keys are the only secret here, and leave the Credential Manager only as a masked hint. Every limit is the user's to
set, copied from each provider's own limits page, because no table built into an app stays right for long.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ..llm.base import LLMError
from ..ui.components import ago, badge, card, fmt_duration, fmt_int, grid_options, notify_error, stat_tile
from . import prompts, providers, service
from .keys import keystore
from .registry import DEFAULT_PROMPT_TOKENS, NOTE, registry

TIME_ZONES = ["America/Los_Angeles", "UTC", "America/New_York", "Europe/London", "Europe/Paris", "Asia/Kolkata",
              "Asia/Shanghai", "Asia/Tokyo"]


def _family_label(fam: str) -> str:
    spec = providers.get(fam)
    users = [providers.get(u).label for u in providers.users_of(fam) if u != fam]
    return spec.label + (f" — also used by {', '.join(users)}" if users else "") + ("  · paid" if spec.paid else "")


def _countdown(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f} s"
    return fmt_duration(seconds)


async def render() -> None:
    """The tabbed AI card. Loads the key mirror and the registry first, so a page opened after a restore is current."""
    await registry.load()
    await keystore.load()
    with card("w-full"):
        with ui.row().classes("items-center w-full gap-3"):
            ui.label("AI keys, models & budget").classes("ag-section-title")
            ui.label("Several keys per provider add their free budgets together: every AI call goes to the key with "
                     "the most requests left for its model, and a key the provider turns away (429, 401/403) is "
                     "skipped at once. When every key of a provider is spent, the call fails over to your fallback "
                     "providers.").classes("ag-muted text-sm")
        with ui.tabs().classes("w-full") as tabs:
            t_keys = ui.tab("Keys", icon="key")
            t_models = ui.tab("Models", icon="tune")
            t_budget = ui.tab("Budget", icon="speed")
            t_providers = ui.tab("Providers", icon="hub")
            t_prompts = ui.tab("Prompts", icon="notes")
        with ui.tab_panels(tabs, value=t_keys).classes("w-full bg-transparent"):
            with ui.tab_panel(t_keys):
                await _keys_panel()
            with ui.tab_panel(t_models):
                await _models_panel()
            with ui.tab_panel(t_budget):
                await _budget_panel()
            with ui.tab_panel(t_providers):
                await _providers_panel()
            with ui.tab_panel(t_prompts):
                await _prompts_panel()


# ---------------------------------------------------------------- Keys

async def _keys_panel() -> None:
    families = {p.id: _family_label(p.id) for p in providers.key_providers()}
    with ui.row().classes("items-end gap-3 w-full"):
        fam = ui.select(families, value="gemini", label="Provider").classes("w-96")
        secret = ui.input("API key", password=True, password_toggle_button=True).classes("w-80")
        label = ui.input("Label (which account)", placeholder="optional").classes("w-56")
        cap = ui.number("Daily request cap", min=1, step=10, format="%d").classes("w-44")
        add_btn = ui.button("Add key", icon="add")
    help_line = ui.label().classes("ag-muted text-sm")

    def describe() -> None:
        spec = providers.get(fam.value)
        text = f"Get a key at {spec.onboarding_url} · keys look like {spec.key_hint or 'any string'} · limits: " \
               f"{spec.limits_url}"
        if spec.paid:
            text = f"{spec.tier_note} A daily request cap is required, and can be moved but not removed. " + text
            if not cap.value:
                cap.value = 200
        else:
            text += " · a cap is optional (the free tier stops by itself)."
        help_line.set_text(text)

    fam.on_value_change(lambda _: describe())
    describe()

    columns = [
        {"field": "label", "headerName": "Key", "minWidth": 150, "flex": 1},
        {"field": "provider_label", "headerName": "Provider", "minWidth": 170},
        {"field": "hint", "headerName": "Hint", "width": 140, "cellClass": "ag-mono"},
        {"field": "state", "headerName": "State", "width": 110},
        {"field": "cap", "headerName": "Daily cap", "width": 105},
        {"field": "requests_today", "headerName": "Requests today", "width": 135, "type": "numericColumn"},
        {"field": "tokens_today", "headerName": "Tokens today", "width": 125, "type": "numericColumn"},
        {"field": "health", "headerName": "Last check / call", "minWidth": 240, "flex": 1},
    ]
    grid = ui.aggrid(grid_options(columns, [], pagination=False), theme="quartz").classes("w-full h-64")
    legacy = ui.label().classes("ag-muted text-xs")

    async def refresh() -> None:
        status = await keystore.status()
        rows = []
        for k in status["keys"]:
            ok = f"Working · {ago(k['last_ok_at'])}" if k["last_ok_at"] else "Not checked yet"
            rows.append({"id": k["id"], "label": k["label"], "provider": k["provider"],
                         "provider_label": providers.get(k["provider"]).label, "hint": k["hint"],
                         "state": "enabled" if k["enabled"] else "disabled",
                         "cap": fmt_int(k["daily_limit"]) if k["daily_limit"] else "model limits",
                         "requests_today": k["requests_today"], "tokens_today": k["tokens_today"],
                         "health": f"✗ {k['last_error']}" if k["last_error"] else f"✓ {ok}" if k["last_ok_at"] else ok})
        grid.options["rowData"] = rows
        grid.update()
        from .. import credentials
        saved = credentials.saved_llm_providers()
        legacy.set_text(("Single keys saved in the provider form (" + ", ".join(providers.get(p).label for p in saved)
                         + ") join the pool as “Saved key” automatically.") if saved else "")

    async def add() -> None:
        try:
            row = await keystore.add(secret.value or "", label.value or None, fam.value,
                                     daily_limit=int(cap.value) if cap.value else None)
        except LLMError as exc:
            notify_error(exc)
            return
        secret.value = ""
        label.value = ""
        result = await service.check_key(row.id)
        ui.notify(f"{row.label} added ({row.hint}). " + (f"It works: {result['models']} models reachable."
                                                         if result["ok"] else f"Check failed: {result['error']}"),
                  type="positive" if result["ok"] else "warning", multi_line=True)
        await refresh()

    async def selected() -> dict[str, Any] | None:
        picked = await grid.get_selected_rows()
        if not picked:
            ui.notify("Select a key first.", type="warning")
            return None
        return picked[0]

    async def check_one() -> None:
        row = await selected()
        if row:
            r = await service.check_key(row["id"])
            ui.notify(f"{row['label']}: works — {r['models']} models reachable." if r["ok"]
                      else f"{row['label']}: {r['error']}", type="positive" if r["ok"] else "negative", multi_line=True)
            await refresh()

    async def check_every() -> None:
        results = await service.check_all()
        good = sum(1 for r in results if r["ok"])
        ui.notify(f"{good} of {len(results)} keys work. The checks cost no generation requests.",
                  type="positive" if good == len(results) else "warning")
        await refresh()

    async def toggle() -> None:
        row = await selected()
        if row:
            try:
                await keystore.set_enabled(row["id"], row["state"] != "enabled")
            except LLMError as exc:
                notify_error(exc)
            await refresh()

    async def edit_cap(clear: bool = False) -> None:
        row = await selected()
        if not row:
            return
        if clear:
            try:
                await keystore.set_enabled(row["id"], row["state"] == "enabled", clear=True)
                ui.notify(f"{row['label']} now follows each model's own limits.")
            except LLMError as exc:
                notify_error(exc)
            await refresh()
            return
        with ui.dialog() as d, ui.card().classes("ag-card p-5"):
            ui.label(f"Daily request cap of {row['label']}").classes("font-semibold")
            ui.label("Counts every model this key serves. Lower it the moment it looks too high.").classes(
                "ag-muted text-sm")
            value = ui.number("Requests per day", min=1, step=10, format="%d",
                              value=int(row["cap"].replace(",", "")) if row["cap"][:1].isdigit() else 200)

            async def save() -> None:
                try:
                    await keystore.set_enabled(row["id"], row["state"] == "enabled", cap=int(value.value or 0))
                except LLMError as exc:
                    notify_error(exc)
                    return
                d.close()
                await refresh()

            with ui.row():
                ui.button("Save", on_click=save)
                ui.button("Cancel", on_click=d.close).props("flat")
        d.open()

    async def remove() -> None:
        row = await selected()
        if not row:
            return
        with ui.dialog() as d, ui.card().classes("ag-card p-5"):
            ui.label(f"Remove {row['label']} ({row['hint']})?").classes("font-semibold")
            ui.label("The key is deleted from Windows Credential Manager and its usage history from this PC.").classes(
                "ag-muted text-sm")

            async def go() -> None:
                await keystore.remove(row["id"])
                d.close()
                ui.notify(f"{row['label']} removed.")
                await refresh()

            with ui.row():
                ui.button("Remove", color="negative", on_click=go)
                ui.button("Cancel", on_click=d.close).props("flat")
        d.open()

    add_btn.on_click(add)
    with ui.row().classes("gap-2"):
        ui.button("Check selected", icon="verified", on_click=check_one).props("flat")
        ui.button("Check all keys", icon="fact_check", on_click=check_every).props("flat")
        ui.button("Enable / disable", icon="toggle_on", on_click=toggle).props("flat")
        ui.button("Set daily cap…", icon="speed", on_click=lambda: edit_cap(False)).props("flat")
        ui.button("Remove cap", icon="block", on_click=lambda: edit_cap(True)).props("flat")
        ui.button("Remove key", icon="delete", on_click=remove).props("flat color=negative")
    ui.label("A key check asks the provider's model list, which no provider bills against your daily generation "
             "requests. The same key cannot be added twice: quota is per account, so a copy adds nothing.").classes(
        "ag-muted text-xs")
    await refresh()


# ---------------------------------------------------------------- Models

async def _models_panel() -> None:
    choices = {p.id: p.label for p in providers.PROVIDERS.values() if p.kind != "local"}
    ui.label(f"{NOTE} Copy each limit from the provider's rate-limit page (Providers tab). A model that is not set up "
             "here still rotates keys and learns from 429s, but has no local ceiling.").classes("ag-muted text-sm")
    with ui.row().classes("items-end gap-3 w-full"):
        prov = ui.select(choices, value="gemini", label="Provider").classes("w-72")
        model = ui.select([], label="Model ID", with_input=True, new_value_mode="add-unique").classes("w-80")
        rpm = ui.number("Requests / minute", min=1, value=10, format="%d").classes("w-36")
        rpd = ui.number("Requests / day", min=1, value=250, format="%d").classes("w-36")
        tz = ui.select(TIME_ZONES, value="America/Los_Angeles", label="Day resets at midnight in",
                       with_input=True, new_value_mode="add-unique").classes("w-60")
        tokens = ui.number(f"Max prompt tokens (empty = {DEFAULT_PROMPT_TOKENS:,})", min=1, format="%d").classes(
            "w-60")
    offered_note = ui.label().classes("ag-muted text-xs")

    def on_provider() -> None:
        tz.value = providers.get(prov.value).reset_timezone
        model.set_options([])
        offered_note.set_text("")

    prov.on_value_change(lambda _: on_provider())

    async def list_offered() -> None:
        got = await service.offered(prov.value)
        model.set_options(got["models"])
        offered_note.set_text(f"{len(got['models'])} models listed." if not got["error"] else
                              f"{providers.get(prov.value).label} did not list its models: {got['error']} "
                              "You can still type the model ID.")

    columns = [
        {"field": "ref", "headerName": "Model", "minWidth": 260, "flex": 1, "cellClass": "ag-mono"},
        {"field": "provider_label", "headerName": "Provider", "minWidth": 170},
        {"field": "rpm", "headerName": "Per minute", "width": 110, "type": "numericColumn"},
        {"field": "rpd", "headerName": "Per day", "width": 110, "type": "numericColumn"},
        {"field": "reset_timezone", "headerName": "Day resets", "width": 180},
        {"field": "prompt_tokens", "headerName": "Max prompt tokens", "width": 150, "type": "numericColumn"},
        {"field": "keys", "headerName": "Enabled keys", "width": 120, "type": "numericColumn"},
    ]
    grid = ui.aggrid(grid_options(columns, [], pagination=False), theme="quartz").classes("w-full h-56")

    def refresh() -> None:
        grid.options["rowData"] = [
            {"ref": m.ref, "provider": m.provider, "id": m.id, "provider_label": providers.get(m.provider).label,
             "rpm": m.rpm, "rpd": m.rpd, "reset_timezone": m.reset_timezone,
             "prompt_tokens": m.max_prompt_tokens or DEFAULT_PROMPT_TOKENS, "max_prompt_tokens": m.max_prompt_tokens,
             "keys": len(keystore.usable(m.provider))} for m in registry.all()]
        grid.update()

    async def save() -> None:
        try:
            info = await registry.set(prov.value, str(model.value or ""), int(rpm.value or 0), int(rpd.value or 0),
                                      str(tz.value or ""), int(tokens.value) if tokens.value else None)
        except LLMError as exc:
            notify_error(exc)
            return
        ui.notify(f"{info.ref} set up: {info.rpm}/min, {info.rpd:,}/day, resets at midnight {info.reset_timezone}.",
                  type="positive")
        refresh()

    async def load_selected() -> None:
        rows = await grid.get_selected_rows()
        if not rows:
            ui.notify("Select a model first.", type="warning")
            return
        r = rows[0]
        prov.value = r["provider"]
        model.set_options([r["id"]])
        model.value = r["id"]
        rpm.value, rpd.value, tz.value, tokens.value = r["rpm"], r["rpd"], r["reset_timezone"], r["max_prompt_tokens"]

    async def remove() -> None:
        rows = await grid.get_selected_rows()
        if not rows:
            ui.notify("Select a model first.", type="warning")
            return
        try:
            await registry.remove(rows[0]["provider"], rows[0]["id"])
        except LLMError as exc:
            notify_error(exc)
            return
        ui.notify(f"{rows[0]['ref']} removed.")
        refresh()

    with ui.row().classes("gap-2"):
        ui.button("Save model", icon="save", on_click=save)
        ui.button("List this provider's models", icon="list", on_click=list_offered).props("flat")
        ui.button("Edit selected", icon="edit", on_click=load_selected).props("flat")
        ui.button("Remove selected", icon="delete", on_click=remove).props("flat color=negative")
    refresh()


# ---------------------------------------------------------------- Budget

async def _budget_panel() -> None:
    with ui.row().classes("w-full gap-3"):
        left_v, left_s = stat_tile("Requests left today", icon="speed",
                                   tooltip="Every set-up model, across every enabled key.")
        reset_v, reset_s = stat_tile("Next reset in", icon="schedule",
                                     tooltip="The soonest a model's day turns over; each resets at midnight in its "
                                             "own time zone.")
        keys_v, keys_s = stat_tile("Enabled keys", icon="key", tooltip="Each key adds its own daily allowance.")
    per_model = ui.aggrid(grid_options([
        {"field": "model", "headerName": "Model", "minWidth": 240, "flex": 1, "cellClass": "ag-mono"},
        {"field": "provider", "headerName": "Provider", "minWidth": 150},
        {"field": "keys", "headerName": "Keys", "width": 80, "type": "numericColumn"},
        {"field": "today", "headerName": "Requests today / limit", "width": 190},
        {"field": "remaining", "headerName": "Left today", "width": 110, "type": "numericColumn"},
        {"field": "minute", "headerName": "Last 60 s / limit", "width": 150},
        {"field": "resets", "headerName": "Resets in", "minWidth": 210},
        {"field": "state", "headerName": "State", "minWidth": 200, "flex": 1},
    ], [], pagination=False), theme="quartz").classes("w-full h-56")
    ui.label("Per key and model, today").classes("ag-section-title mt-2")
    per_key = ui.aggrid(grid_options([
        {"field": "key", "headerName": "Key", "minWidth": 150},
        {"field": "model", "headerName": "Model", "minWidth": 240, "flex": 1, "cellClass": "ag-mono"},
        {"field": "requests", "headerName": "Requests", "width": 110, "type": "numericColumn"},
        {"field": "remaining", "headerName": "Left today", "width": 110, "type": "numericColumn"},
        {"field": "minute", "headerName": "Last 60 s", "width": 100, "type": "numericColumn"},
        {"field": "tokens", "headerName": "Tokens", "width": 110, "type": "numericColumn"},
        {"field": "last", "headerName": "Last request", "width": 130},
    ], [], pagination=False), theme="quartz").classes("w-full h-56")
    empty = ui.label().classes("ag-muted text-sm")

    async def refresh() -> None:
        status = await keystore.status()
        budget = status["budget"]
        left_v.set_text(fmt_int(sum(b["remaining_today"] for b in budget)) if budget else "—")
        left_s.set_text(f"of {fmt_int(sum(b['allowed_today'] for b in budget))} across set-up models" if budget else
                        "set up a model to see its budget")
        reset_v.set_text(_countdown(status["reset_in"]))
        reset_s.set_text("BRAIN simulations reset separately, at 00:00 US Eastern")
        keys_v.set_text(f"{status['enabled']} / {len(status['keys'])}")
        keys_s.set_text("requests rotate across enabled keys")
        per_model.options["rowData"] = [{
            "model": b["model"], "provider": providers.get(b["provider"]).label, "keys": b["keys"],
            "today": f"{fmt_int(b['used_today'])} / {fmt_int(b['allowed_today'])}",
            "remaining": b["remaining_today"], "minute": f"{b['minute_used']} / {b['minute_limit']}",
            "resets": f"{_countdown(b['reset_in'])} · midnight {b['reset_timezone']}",
            "state": (f"blocked — free again in {_countdown(b['retry_after'])}" if b["blocked"] else "available")}
            for b in budget]
        per_model.update()
        names = {k["id"]: k["label"] for k in status["keys"]}
        rows = []
        for k in status["keys"]:
            for u in k["usage"]:
                if u["model"] == "*":
                    continue
                rows.append({"key": names.get(u["key_id"], u["key_id"]), "model": u["model"],
                             "requests": u["requests"], "remaining": k["remaining_today"].get(u["model"], "—"),
                             "minute": keystore.ledger.minute(k["id"], u["model"]), "tokens": u["tokens"],
                             "last": ago(u["last_request_at"])})
        per_key.options["rowData"] = rows
        per_key.update()
        empty.set_text("" if budget else "No budget without a model: set one up under Models, with the limits your "
                                         "provider shows you, and its daily allowance appears here.")

    await refresh()
    ui.timer(15.0, refresh)


# ---------------------------------------------------------------- Providers

async def _providers_panel() -> None:
    from ..llm.router import PROVIDERS
    ui.label("Every provider below except the last two has a free tier that needs no card. The OpenAI-compatible ones "
             "from alpha-harness (Google's compatible endpoint, Groq, Cerebras, NVIDIA NIM, Mistral, Hugging Face) "
             "take keys only here, under Keys; choose one as the active or a fallback provider above.").classes(
        "ag-muted text-sm")
    specs = sorted((p for p in providers.PROVIDERS.values() if p.kind != "local"), key=lambda p: p.paid)
    paid_shown = False
    with ui.row().classes("w-full gap-3 flex-wrap"):
        for spec in specs:
            if spec.paid and not paid_shown:
                paid_shown = True
                ui.label(providers.PAID_NOTE).classes("ag-muted text-sm w-full mt-2")
            with ui.card().classes("ag-card p-4 w-[360px]"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.label(spec.label).classes("font-semibold")
                    badge("paid" if spec.paid else "free tier", "warning" if spec.paid else "pass")
                provider = PROVIDERS.get(spec.id)
                keys = len(keystore.usable(spec.id))
                models = len([m for m in registry.all() if m.provider == spec.id])
                ui.label(f"{keys} enabled key(s) · {models} model(s) set up · "
                         + ("set up" if provider is not None and provider.configure() else "not set up")).classes(
                    "ag-muted text-xs")
                if spec.base_url:
                    ui.label(spec.base_url).classes("ag-mono text-xs ag-muted")
                if spec.tier_note:
                    ui.label(spec.tier_note).classes("text-xs ag-amber")
                if spec.key_family:
                    ui.label(f"Uses your {providers.get(spec.key_family).label} keys.").classes("text-xs ag-muted")
                with ui.row().classes("gap-3"):
                    if spec.onboarding_url:
                        ui.link("Get a key", spec.onboarding_url, new_tab=True).classes("text-sm")
                    if spec.limits_url:
                        ui.link("Rate limits", spec.limits_url, new_tab=True).classes("text-sm")
                if spec.key_hint:
                    ui.label(f"Keys look like {spec.key_hint}").classes("text-xs ag-muted")


# ---------------------------------------------------------------- Prompts

async def _prompts_panel() -> None:
    library = await prompts.library()
    ui.label("Every prompt AlphaGo sends, in the version in use. The token estimate (about four characters per token) "
             "matters because prompt tokens come out of the same per-minute budget as the answer.").classes(
        "ag-muted text-sm")
    grid = ui.aggrid(grid_options([
        {"field": "label", "headerName": "Prompt", "minWidth": 170},
        {"field": "purpose", "headerName": "What it does", "minWidth": 320, "flex": 1},
        {"field": "characters", "headerName": "Characters", "width": 120, "type": "numericColumn"},
        {"field": "tokens", "headerName": "≈ Tokens", "width": 110, "type": "numericColumn"},
    ], [{"slug": p.slug, "label": p.label, "purpose": p.purpose, "characters": p.characters,
         "tokens": p.estimated_tokens} for p in library], pagination=False), theme="quartz").classes("w-full h-72")
    bodies = {p.slug: p for p in library}

    async def show() -> None:
        rows = await grid.get_selected_rows()
        if not rows:
            ui.notify("Select a prompt first.", type="warning")
            return
        p = bodies[rows[0]["slug"]]
        with ui.dialog() as d, ui.card().classes("ag-card p-5 w-[1000px] max-w-[95vw]"):
            ui.label(f"{p.label} — {p.characters:,} characters (≈ {p.estimated_tokens:,} tokens)").classes(
                "font-semibold")
            ui.textarea(value=p.body).classes("w-full ag-mono").props("readonly autogrow")
        d.open()

    ui.button("Show the full prompt", icon="visibility", on_click=show).props("flat")
