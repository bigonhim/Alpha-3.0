"""Sync with BRAIN — account login (incl. biometric verification), catalogue and account synchronisation."""

from __future__ import annotations

import webbrowser

from nicegui import ui
from sqlalchemy import select

from ... import credentials
from ...brain import brain
from ...brain.errors import AuthError, BrainError, PersonaRequired
from ...config import settings
from ...db import db
from ...db.models import Job
from ...engine import sync as sync_engine
from ...events import bus
from ...jobs.queue import jobs
from ...runtime import load_options_quietly
from ..components import ago, card, fmt_duration, fmt_int, notify_error, page_header


def _scope_selectors():
    opts = brain.options
    region = ui.select(opts.regions, value=settings.research.region if settings.research.region in opts.regions
                       else (opts.regions[0] if opts.regions else "USA"), label="Region").classes("w-36")
    delay = ui.select([], label="Delay").classes("w-28")
    universe = ui.select([], label="Universe").classes("w-44")

    def refresh_choices() -> None:
        r = region.value
        delay.options = opts.delays(r)
        delay.value = settings.research.delay if settings.research.delay in delay.options else delay.options[0]
        universe.options = opts.universes(r)
        universe.value = (settings.research.universe if settings.research.universe in universe.options
                          else universe.options[0])
        delay.update()
        universe.update()

    region.on_value_change(lambda _: refresh_choices())
    refresh_choices()
    return region, delay, universe


async def page() -> None:
    page_header("Sync with BRAIN", "Sign in, then pull the data catalogue and your alphas into the local research database.")

    with ui.row().classes("w-full gap-5 items-stretch"):
        # ---------------------------------------------------------- account
        with card("w-[520px]"):
            ui.label("BRAIN account").classes("ag-section-title")
            status = ui.label().classes("text-sm")
            session = ui.label().classes("ag-muted text-sm")
            email = ui.input("Email", value=credentials.brain_email() or "").classes("w-full")
            password = ui.input("Password", password=True, password_toggle_button=True).classes("w-full")
            remember = ui.checkbox("Save credentials in Windows Credential Manager", value=True)
            persona_box = ui.column().classes("w-full gap-2 p-3 rounded-lg").style(
                "border:1px solid #5a4a1a;background:rgba(80,60,10,.15)")
            with persona_box:
                ui.label("Biometric verification required").classes("ag-amber font-semibold")
                ui.label("BRAIN asks you to confirm your identity in the browser (Persona). Open the link, complete "
                         "the check, then click Continue.").classes("text-sm")
                with ui.row():
                    ui.button("Open verification page", icon="open_in_new",
                              on_click=lambda: webbrowser.open(brain.client.persona_url or "")).props("color=warning")
                    persona_btn = ui.button("Continue", icon="verified")

            async def do_login() -> None:
                if not email.value or (not password.value and not credentials.brain_credentials()):
                    ui.notify("Enter your BRAIN email and password.", type="warning")
                    return
                if remember.value and password.value:
                    credentials.set_brain_credentials(email.value.strip(), password.value)
                login_btn.props("loading")
                try:
                    if password.value:
                        await brain.client.login(email.value.strip(), password.value)
                    else:
                        await brain.client.login()
                    ui.notify("Signed in to BRAIN.", type="positive")
                    bus.publish("sync", "Signed in to WorldQuant BRAIN.", "success")
                    await load_options_quietly()
                    from ...engine.simulator import simulator
                    if simulator.paused and "verification" in (simulator.pause_reason or ""):
                        simulator.resume()
                except PersonaRequired:
                    ui.notify("Biometric verification required — use the yellow box.", type="warning")
                except (AuthError, BrainError) as exc:
                    notify_error(exc)
                except Exception as exc:
                    notify_error(exc, "Login failed: ")
                finally:
                    login_btn.props(remove="loading")
                    password.value = ""

            async def do_persona() -> None:
                persona_btn.props("loading")
                try:
                    ok = await brain.client.complete_persona(timeout_s=180)
                    if ok:
                        ui.notify("Verified — signed in.", type="positive")
                        await load_options_quietly()
                        from ...engine.simulator import simulator
                        simulator.resume()
                    else:
                        ui.notify("Still waiting for verification. Finish it in the browser, then try again.",
                                  type="warning")
                except BrainError as exc:
                    notify_error(exc)
                finally:
                    persona_btn.props(remove="loading")

            persona_btn.on_click(do_persona)

            async def do_logout() -> None:
                await brain.client.logout()
                ui.notify("Signed out.")

            def forget() -> None:
                credentials.set_brain_credentials(None, None)
                email.value = ""
                ui.notify("Saved credentials removed from Windows Credential Manager.")

            with ui.row().classes("gap-2"):
                login_btn = ui.button("Sign in", icon="login", on_click=do_login)
                ui.button("Sign out", icon="logout", on_click=do_logout).props("flat")
                ui.button("Forget saved credentials", icon="key_off", on_click=forget).props("flat color=grey")

            def refresh_status() -> None:
                c = brain.client
                persona_box.set_visibility(c.state == "persona_required")
                if c.logged_in:
                    status.set_text(f"● Signed in as {c.user_id or '—'}")
                    status.classes(replace="text-sm ag-green")
                    session.set_text(f"Session expires in {fmt_duration(c.seconds_left)} · permissions: "
                                     f"{', '.join(c.permissions) or '—'}")
                elif c.state == "persona_required":
                    status.set_text("● Waiting for biometric verification")
                    status.classes(replace="text-sm ag-amber")
                    session.set_text("")
                else:
                    status.set_text("● Signed out" + (f" — {c.last_error}" if c.last_error else ""))
                    status.classes(replace="text-sm ag-red")
                    session.set_text("Credentials are stored only in Windows Credential Manager.")

            refresh_status()
            ui.timer(1.0, refresh_status)

        # ---------------------------------------------------------- settings available
        with card("grow min-w-[420px]"):
            ui.label("Simulation settings available to you").classes("ag-section-title")
            opts_box = ui.column().classes("gap-1 w-full")

            def render_opts() -> None:
                opts_box.clear()
                o = brain.options
                with opts_box:
                    ui.label("Live from OPTIONS /simulations" if o.live else
                             "Defaults (sign in and refresh to load your account's real choices)").classes(
                        "ag-muted text-sm")
                    for r in o.regions:
                        with ui.row().classes("items-center gap-2 no-wrap"):
                            ui.label(r).classes("ag-mono ag-teal w-12")
                            ui.label(f"D{'/'.join(str(d) for d in o.delays(r))} · "
                                     f"{', '.join(o.universes(r))}").classes("text-sm")

            render_opts()

            async def refresh_opts() -> None:
                try:
                    await brain.load_options()
                    await load_options_quietly()
                    render_opts()
                    ui.notify("Settings refreshed from BRAIN.", type="positive")
                except BrainError as exc:
                    notify_error(exc)

            ui.button("Refresh from BRAIN", icon="refresh", on_click=refresh_opts).props("flat")

    with ui.row().classes("w-full gap-5 items-stretch"):
        # ---------------------------------------------------------- catalogue
        with card("grow min-w-[520px]"):
            ui.label("Data catalogue").classes("ag-section-title")
            ui.label("Operators, datasets and fields for a region / delay / universe. Fields enable the local "
                     "validator, so expressions are checked before they spend simulation quota.").classes(
                "ag-muted text-sm")
            with ui.row().classes("items-end gap-3"):
                region, delay, universe = _scope_selectors()

            def scope() -> dict:
                return {"region": region.value, "delay": int(delay.value), "universe": universe.value}

            async def submit(kind: str, title: str, params: dict | None = None) -> None:
                if not brain.client.logged_in:
                    ui.notify("Sign in to BRAIN first.", type="warning")
                    return
                await jobs.submit(kind, title, params or {})
                ui.notify(f"Queued: {title}")

            with ui.row().classes("gap-2 flex-wrap"):
                ui.button("Sync everything for scope", icon="cloud_download", on_click=lambda: submit(
                    "sync_scope", f"Catalogue {scope()['region']} D{scope()['delay']} {scope()['universe']}", scope()))
                ui.button("Operators", on_click=lambda: submit("sync_operators", "Sync operators")).props("flat")
                ui.button("Data categories", on_click=lambda: submit("sync_categories", "Sync data categories")).props("flat")
                ui.button("Datasets", on_click=lambda: submit(
                    "sync_datasets", f"Datasets {scope()['region']} D{scope()['delay']} {scope()['universe']}",
                    scope())).props("flat")
                ui.button("Fields", on_click=lambda: submit(
                    "sync_fields", f"Fields {scope()['region']} D{scope()['delay']} {scope()['universe']}",
                    scope())).props("flat")
            counts_label = ui.label().classes("ag-muted text-sm")

        # ---------------------------------------------------------- account data
        with card("w-[520px]"):
            ui.label("Your alphas & programs").classes("ag-section-title")
            ui.label("Submitted (ACTIVE) alphas in full plus recent unsubmitted ones; Genius summary and pyramids "
                     "for this quarter.").classes("ag-muted text-sm")
            with ui.row().classes("gap-2 flex-wrap"):
                ui.button("Sync my alphas", icon="layers", on_click=lambda: submit("sync_alphas", "Sync my alphas"))
                ui.button("Full alpha sync", on_click=lambda: submit("sync_alphas", "Full alpha sync",
                                                                     {"full": True})).props("flat")
                ui.button("Pyramids & Genius", icon="workspace_premium",
                          on_click=lambda: submit("sync_programs", "Sync pyramids & Genius summary")).props("flat")
            last_label = ui.label().classes("ag-muted text-sm")

    # ---------------------------------------------------------- recent jobs
    with card("w-full"):
        ui.label("Recent sync tasks").classes("ag-section-title")
        jobs_box = ui.column().classes("w-full gap-1")

    state = {"version": -1}

    async def refresh_lists() -> None:
        counts = await sync_engine.catalog_counts()
        counts_label.set_text(f"Local catalogue: {fmt_int(counts['operators'])} operators · "
                              f"{fmt_int(counts['datasets'])} dataset scopes · {fmt_int(counts['fields'])} fields")
        la, lp = await sync_engine.last_synced("alphas"), await sync_engine.last_synced("programs")
        last_label.set_text(f"{fmt_int(counts['alphas'])} alphas stored ({fmt_int(counts['active'])} submitted) · "
                            f"alphas synced {ago(la)} · programs synced {ago(lp)}")
        if state["version"] == jobs.version:
            return
        state["version"] = jobs.version
        async with db.session() as s:
            rows = list(await s.scalars(select(Job).where(Job.kind.like("sync_%")).order_by(Job.id.desc()).limit(8)))
        jobs_box.clear()
        with jobs_box:
            if not rows:
                ui.label("No sync tasks yet.").classes("ag-muted text-sm")
            for j in rows:
                with ui.row().classes("items-center gap-3 w-full no-wrap"):
                    color = {"done": "ag-green", "failed": "ag-red", "running": "ag-cyan"}.get(j.status, "ag-muted")
                    ui.label(j.status.upper()).classes(f"{color} ag-mono text-xs w-20")
                    ui.label(j.title).classes("text-sm grow")
                    if j.status == "running":
                        ui.linear_progress(value=j.progress, show_value=False).classes("w-40")
                    ui.label(j.error or j.message or "").classes("ag-muted text-xs max-w-[480px] truncate")

    await refresh_lists()
    ui.timer(2.0, refresh_lists)
