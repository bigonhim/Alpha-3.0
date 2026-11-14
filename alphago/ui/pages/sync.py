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
from ..components import ago, card, empty_state, fmt_duration, fmt_int, notify_error, page_header


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

    await catalogue_matrix()

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
                ui.button("Data categories",
                          on_click=lambda: submit("sync_categories", "Sync data categories")).props("flat")
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
    await alerts_and_background()


async def alerts_and_background() -> None:
    """Email alerts and running in the background (both off until you switch them on)."""
    import asyncio

    from ... import background
    from ...engine import alerts

    a = settings.alerts
    with ui.row().classes("w-full gap-5 items-stretch no-wrap"):
        with card("w-[55%]"):
            ui.label("Email alerts").classes("ag-section-title")
            ui.label("A message when an alpha passes BRAIN's check, a task finishes, the queue is stuck, the quota "
                     "pauses, BRAIN asks for verification, or the AI provider keeps failing. Messages carry alpha IDs "
                     "and counts only — never expressions or credentials.").classes("ag-muted text-xs")
            on = ui.switch("Alerts on", value=a.enabled)
            with ui.row().classes("gap-3 w-full"):
                host = ui.input("SMTP server", value=a.smtp_host, placeholder="smtp.gmail.com").classes("w-56")
                port = ui.number("Port", value=a.smtp_port, min=1, max=65535).classes("w-24")
                sec = ui.select({"starttls": "STARTTLS", "ssl": "SSL", "none": "None"}, value=a.security,
                                label="Security").classes("w-32")
            with ui.row().classes("gap-3 w-full"):
                user = ui.input("SMTP user", value=a.smtp_user).classes("w-56")
                pw = ui.input("SMTP password (app password)", password=True, password_toggle_button=True,
                              placeholder="●●●●●●●● saved" if alerts.smtp_password() else "").classes("w-56")
            with ui.row().classes("gap-3 w-full"):
                sender = ui.input("From (empty = SMTP user)", value=a.sender).classes("w-56")
                to = ui.input("Send to", value=a.recipient).classes("w-56")
            with ui.row().classes("gap-3 w-full flex-wrap"):
                boxes = {k: ui.checkbox(label, value=a.triggers.get(k, True)) for k, label in alerts.TRIGGERS.items()}
            with ui.row().classes("gap-3"):
                quiet = ui.number("Same alert at most every (min)", value=a.quiet_minutes, min=1).classes("w-56")
                stuck = ui.number("Stuck after (min)", value=a.stuck_minutes, min=5).classes("w-40")
                hour = ui.number("Daily summary hour", value=a.summary_hour, min=0, max=23).classes("w-40")

            def save() -> None:
                a.enabled, a.smtp_host, a.smtp_port = bool(on.value), (host.value or "").strip(), int(port.value or 587)
                a.security, a.smtp_user = sec.value, (user.value or "").strip()
                a.sender, a.recipient = (sender.value or "").strip(), (to.value or "").strip()
                a.triggers = {k: bool(b.value) for k, b in boxes.items()}
                a.quiet_minutes, a.stuck_minutes = int(quiet.value or 30), int(stuck.value or 30)
                a.summary_hour = int(hour.value or 18)
                if pw.value:
                    alerts.set_smtp_password(pw.value)
                    pw.value = ""
                settings.save()
                ui.notify("Alert settings saved.", type="positive")

            async def test() -> None:
                save()
                try:
                    await alerts.send("Test alert", "If you can read this, AlphaGo's email alerts work.")
                    ui.notify(f"Test email sent to {a.recipient}.", type="positive")
                except Exception as exc:
                    notify_error(exc, "Test email failed: ")

            with ui.row().classes("gap-2"):
                ui.button("Save", icon="save", on_click=save)
                ui.button("Send test", icon="mail", on_click=test).props("flat")

        with card("grow"):
            ui.label("Run in the background").classes("ag-section-title")
            ui.label("'Start with Windows' runs the engine without a window at logon (as you, so it can read your "
                     "keys); open the UI at http://127.0.0.1:8090 whenever you want. Only one AlphaGo engine runs at "
                     "a time. Your PC still has to be on.").classes("ag-muted text-xs")
            status = ui.label().classes("text-sm")

            async def refresh_status() -> None:
                enabled = await asyncio.to_thread(background.autostart_enabled)
                status.set_text("Start with Windows: ON" if enabled else "Start with Windows: off")

            async def toggle(on_: bool) -> None:
                try:
                    msg = await asyncio.to_thread(background.set_autostart, on_)
                    ui.notify(msg, type="positive")
                except Exception as exc:
                    notify_error(exc, "Task Scheduler: ")
                await refresh_status()

            with ui.row().classes("gap-2"):
                ui.button("Start with Windows", icon="schedule", on_click=lambda: toggle(True)).props("flat")
                ui.button("Don't start with Windows", icon="block", on_click=lambda: toggle(False)).props("flat")
            awake = ui.checkbox("Keep the PC awake while simulations are queued or running",
                                value=settings.service.keep_awake)
            awake.on_value_change(lambda e: (setattr(settings.service, "keep_awake", bool(e.value)),
                                             settings.save()))
            ui.label(f"Command: {background.headless_command()}").classes("ag-mono text-xs ag-muted break-all")
            await refresh_status()


# ---------------------------------------------------------------- the all-markets sync matrix

_HALF = {"waiting": "background:#1b2536", "fetching": "background:#3b4a66", "fields": "background:#56657f",
         "details": "background:#56657f;box-shadow:inset 0 0 0 1px #e6edf7", "done": "background:#d7dee9",
         "failed": "background:#6b2633"}
_PULSE = {"fetching", "details"}
_NO_DELAY = "background:repeating-linear-gradient(135deg,#2a3850 0 1px,transparent 1px 5px)"


def _market_states(synced: list[dict], run: dict | None) -> dict[tuple[str, int, str], dict]:
    """While a sync runs its own per-market states are the truth; otherwise what the catalogue holds, plus any
    market the last sync could not finish."""
    states = {(r["region"], r["delay"], r["universe"]): {"state": "done", "fields": r["fields"]} for r in synced}
    running = bool(run and run["status"] == "RUNNING")
    for m in (run or {}).get("markets") or []:
        key = (m["region"], int(m["delay"]), m["universe"])
        if running or m["state"] in ("failed", "fields"):
            states[key] = {"state": m["state"], "fields": m.get("fields") or states.get(key, {}).get("fields")}
    return states


def _running_text(run: dict, elapsed: float) -> str:
    stage = "Filling in Dataset Details" if run["stage"] == "details" else "Syncing Data Fields"
    m, s = divmod(int(max(0, elapsed)), 60)
    return (f"{stage} · {run['scopes_done']}/{run['scopes_total']} markets · {run['fraction'] * 100:.0f}% · "
            f"{m}m {s:02d}s elapsed")


async def catalogue_matrix() -> None:
    """Every market BRAIN offers as a Region × Universe matrix, each cell split into Delay 0 | Delay 1, filling live
    as the sync works through it. Click a half to open that market in the Data Explorer."""
    from ...catalog import queries as cq
    from ...catalog import sync as catalog_sync
    from .data_explorer import open_market

    with card("w-full gap-3"):
        with ui.row().classes("items-center w-full gap-3"):
            ui.label("BRAIN Datasets").classes("ag-section-title")
            headline = ui.label().classes("ag-muted text-sm")
            ui.space()
            action = ui.button().props("no-caps")
        ui.label("Downloads every market's data fields and datasets (every region, delay and universe BRAIN offers "
                 "you), USA D1 TOP3000 first so research can start while the rest arrive. Click a cell to open that "
                 "market in the Data Explorer.").classes("ag-muted text-xs")
        notice = ui.column().classes("w-full")
        matrix = ui.column().classes("w-full")
        bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
        with ui.row().classes("gap-3 flex-wrap text-xs ag-muted items-center"):
            for st, text in catalog_sync.STATE_LABELS.items():
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.element("span").style(f"display:inline-block;width:20px;height:11px;border-radius:2px;"
                                             f"{_HALF[st]}")
                    ui.label(text)
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.element("span").style(f"display:inline-block;width:20px;height:11px;border-radius:2px;{_NO_DELAY}")
                ui.label("Not Offered")
    agnostic_box = ui.column().classes("w-full")
    state: dict = {"sig": None, "agnostic_sig": None}

    async def start(region_agnostic: bool) -> None:
        if not brain.client.logged_in:
            ui.notify("Sign in to BRAIN first.", type="warning")
            return
        try:
            await catalog_sync.start(region_agnostic=region_agnostic)
            ui.notify("Catalogue sync queued: the markets download in the background.", type="positive")
        except RuntimeError as exc:
            ui.notify(str(exc), type="warning")

    def cancel() -> None:
        with ui.dialog() as dlg, ui.card().classes("ag-card p-5"):
            ui.label("Cancel this sync?").classes("text-lg")
            ui.label("Stops the sync. Markets that already arrived stay in the catalogue; press Sync again to finish "
                     "the rest.").classes("text-sm")
            with ui.row().classes("self-end"):
                ui.button("Keep syncing", on_click=dlg.close).props("flat")

                async def go() -> None:
                    dlg.close()
                    ok = await catalog_sync.cancel()
                    ui.notify("Sync cancelled." if ok else "The sync had already finished.")
                ui.button("Cancel sync", on_click=go).props("color=negative")
        dlg.open()

    def half(region: str, delay: int, universe: str, tile: dict | None, offered: bool) -> None:
        if not offered:
            ui.element("div").style(f"height:20px;border-radius:3px;flex:1;{_NO_DELAY}").tooltip(
                f"{region} · {universe} has no Delay {delay}")
            return
        tile = tile or {"state": "waiting", "fields": None}
        label = (f"{region} · Delay {delay} · {universe}: {catalog_sync.STATE_LABELS[tile['state']]}"
                 + (f" · {tile['fields']:,} fields" if tile.get("fields") else "")
                 + " · click to open in the Data Explorer")
        el = ui.element("div").style(f"height:20px;border-radius:3px;flex:1;cursor:pointer;{_HALF[tile['state']]}")
        if tile["state"] in _PULSE:
            el.classes("animate-pulse")
        el.tooltip(label)
        el.on("click", lambda _e, r=region, d=delay, u=universe: open_market(r, d, u))

    async def refresh() -> None:
        synced = await cq.synced_tuples()
        run = await catalog_sync.latest_run("all")
        markets = [m for m in catalog_sync.offered_markets() if m.region != catalog_sync.REGION_AGNOSTIC]
        running = bool(run and run["status"] == "RUNNING")
        if running and run:
            headline.set_text(_running_text(run, (catalog_sync.now() - run["started_at"]).total_seconds()))
        states = _market_states(synced, run)
        sig = (running, run and run.get("status"), run and run.get("error"),
               tuple(sorted((k, v["state"], v.get("fields")) for k, v in states.items())), len(markets))
        await render_agnostic(synced)
        if sig == state["sig"]:
            return
        state["sig"] = sig
        exists = {(m.region, m.delay, m.universe) for m in markets}
        held = [r for r in synced if (r["region"], r["delay"], r["universe"]) in exists]
        if running:
            action.set_text("Cancel Sync")
            action.props(remove="color=primary").props("color=negative icon=close")
        else:
            headline.set_text(f"{len(held):,} of {len(markets):,} markets synced · "
                              f"{sum(r['fields'] for r in held):,} fields")
            action.set_text("Sync")
            action.props(remove="color=negative").props("color=primary icon=cloud_download")
        bar.set_visibility(running)
        if run:
            bar.set_value(run.get("fraction") or 0)
        notice.clear()
        if run and not running and run.get("error"):
            with notice, ui.row().classes("items-start gap-2 no-wrap"):
                title = ("The sync failed" if run["status"] == "FAILED" else "The sync was cancelled"
                         if run["status"] == "CANCELLED" else "Some markets did not sync, even after 3 tries")
                ui.icon("warning").classes("ag-amber")
                ui.label(f"{title}: {run['error']}").classes("text-sm ag-amber")
        regions: dict[str, list[str]] = {}
        delays = sorted({m.delay for m in markets})
        for m in markets:
            regions.setdefault(m.region, [])
            if m.universe not in regions[m.region]:
                regions[m.region].append(m.universe)
        order = sorted(regions, key=lambda r: (-len(regions[r]), r))     # widest first: the grid fills from the left
        matrix.clear()
        with matrix:
            if not markets:
                empty_state("public_off", "BRAIN offered no markets", "Sign in, then press Refresh from BRAIN.")
            with ui.row().classes("w-full gap-2 no-wrap overflow-x-auto items-start"):
                for region in order:
                    with ui.column().classes("gap-1 min-w-[118px] grow"):
                        with ui.row().classes("w-full items-baseline justify-between px-1 no-wrap"):
                            ui.label(region).classes("ag-mono text-sm")
                            ui.label(" · ".join(f"D{d}" for d in delays)).classes("ag-muted text-xs")
                        for universe in regions[region]:
                            with ui.column().classes("gap-1 w-full p-1 rounded").style(
                                    "border:1px solid var(--ag-border);background:rgba(255,255,255,.02)"):
                                ui.label(universe).classes("ag-mono ag-muted text-[11px] truncate px-0.5")
                                with ui.row().classes("gap-1 w-full no-wrap"):
                                    for d in delays:
                                        half(region, d, universe, states.get((region, d, universe)),
                                             (region, d, universe) in exists)

    async def render_agnostic(synced: list[dict]) -> None:
        universes = [m.universe for m in catalog_sync.offered_markets() if m.region == catalog_sync.REGION_AGNOSTIC]
        run = await catalog_sync.latest_run("agnostic") if universes else None
        states = _market_states([r for r in synced if r["region"] == catalog_sync.REGION_AGNOSTIC], run)
        running = bool(run and run["status"] == "RUNNING")
        sig = (tuple(universes), running, tuple(sorted((k, v["state"], v.get("fields")) for k, v in states.items())))
        if sig == state["agnostic_sig"]:
            return
        state["agnostic_sig"] = sig
        agnostic_box.clear()
        if not universes:
            return          # BRAIN lists region ALL only for an account that may simulate region-agnostically
        with agnostic_box, card("w-full gap-2"):
            with ui.row().classes("items-center w-full gap-3"):
                ui.label("Sync Region Agnostic Data").classes("ag-section-title")
                held = sum(1 for u in universes if (catalog_sync.REGION_AGNOSTIC, 1, u) in states)
                ui.label(f"{held} of {len(universes)} universes synced").classes("ag-muted text-sm")
                ui.space()
                if running:
                    ui.button("Cancel Sync", icon="close", on_click=cancel).props("color=negative no-caps")
                else:
                    ui.button("Sync Region Agnostic Data", icon="cloud_download",
                              on_click=lambda: start(True)).props("outline no-caps")
            ui.label("One expression run in USA, Europe, Asia and Global at once. BRAIN hands this market over fifty "
                     "fields at a time, dataset by dataset (about ten minutes per universe), so it is its own "
                     "download.").classes("ag-muted text-xs")
            with ui.row().classes("gap-2 w-full"):
                for u in universes:
                    tile = states.get((catalog_sync.REGION_AGNOSTIC, 1, u)) or {"state": "waiting", "fields": None}
                    with ui.column().classes("gap-1 p-2 rounded min-w-[160px] cursor-pointer").style(
                            "border:1px solid var(--ag-border)").on(
                            "click", lambda _e, uu=u: open_market(catalog_sync.REGION_AGNOSTIC, 1, uu)):
                        ui.label(u).classes("ag-mono text-sm")
                        ui.element("div").style(f"height:10px;border-radius:2px;{_HALF[tile['state']]}")
                        ui.label(f"{catalog_sync.STATE_LABELS[tile['state']]} · "
                                 f"{fmt_int(tile.get('fields'))} fields").classes("ag-muted text-xs")

    async def on_action() -> None:
        if catalog_sync.running():
            cancel()
        else:
            await start(False)

    action.on_click(on_action)
    await refresh()
    ui.timer(1.0, refresh)
