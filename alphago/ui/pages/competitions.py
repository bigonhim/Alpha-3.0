"""Competitions (ported from alpha-harness screens/competitions): what BRAIN is running now and where you stand.
Joining and submitting both happen on BRAIN; AlphaGo never signs up for you."""

from __future__ import annotations

import webbrowser

from nicegui import ui

from ...brain import brain
from ...brain.errors import BrainError
from ...catalog import competitions as comp
from ..components import badge, card, empty_state, fmt_int, notify_error, page_header


def _date(value: str | None) -> str:
    at = comp.when(value)
    return at.strftime("%d %b %Y") if at else "—"


def _card(c: comp.Competition) -> None:
    left = c.days_left()
    with card("w-full gap-3"):
        with ui.row().classes("items-center w-full gap-3"):
            ui.icon("emoji_events").classes("text-2xl ag-amber")
            ui.label(c.name).classes("text-lg")
            badge("Enrolled" if c.enrolled else "Not enrolled", "pass" if c.enrolled else "warning")
            ui.space()
            if c.faq:
                ui.button("Rules", icon="menu_book", on_click=lambda u=c.faq: webbrowser.open(u)).props("flat no-caps")
            ui.button("Open in BRAIN", icon="open_in_new", on_click=lambda u=c.url: webbrowser.open(u)).props(
                "outline no-caps")
        with ui.row().classes("gap-3 w-full"):
            for label, value in (("Ends", _date(c.end_date)), ("Days Left", "—" if left is None else fmt_int(left)),
                                 ("Your Rank", f"#{fmt_int(c.rank)}" if c.rank else "—"),
                                 ("Your Alphas", fmt_int(c.alphas))):
                with ui.card().classes("ag-tile p-3 min-w-[150px] grow"):
                    ui.label(label).classes("ag-muted text-xs")
                    ui.label(value).classes("ag-kpi-sm")
        if c.sign_up_open():
            ui.label(f"Sign up on BRAIN by {_date(c.sign_up_end_date)} for your alphas to count. AlphaGo never signs "
                     "up for you.").classes("ag-amber text-sm")
        elif c.ongoing and not c.standing:
            ui.label("BRAIN did not show your standing in this competition right now; it may be restricted.") \
                .classes("ag-muted text-xs")


async def page() -> None:
    page_header("Competitions", "Ongoing and past BRAIN competitions, your rank and alphas.")
    with ui.row().classes("w-full"):
        ui.label("Your rank comes with each ongoing competition; joining and submitting happen on BRAIN.").classes(
            "ag-muted text-sm")
        ui.space()
        refresh = ui.button("Refresh", icon="refresh").props("flat")
    body = ui.column().classes("w-full gap-4")

    async def load() -> None:
        body.clear()
        if not brain.client.logged_in:
            with body, card("w-full"):
                empty_state("login", "Sign in to BRAIN to see its competitions",
                            "Competitions and your rank in each come from your BRAIN account.")
                ui.button("Open Sync with BRAIN", on_click=lambda: ui.navigate.to("/sync")).props("flat")
            return
        try:
            rows = await comp.competitions()
        except BrainError as exc:
            with body, card("w-full"):
                empty_state("cloud_off", "Could not read competitions", exc.message)
            return
        except Exception as exc:
            notify_error(exc, "Competitions: ")
            return
        live = [c for c in rows if c.ongoing]
        past = [c for c in rows if not c.ongoing]
        with body:
            if not live:
                with card("w-full"):
                    empty_state("emoji_events", "No competition is running right now.")
            for c in live:
                _card(c)
            if past:
                with ui.expansion(f"Past Competitions ({len(past):,})", icon="history").classes("w-full ag-card"):
                    for c in past:
                        with ui.row().classes("items-center gap-3 w-full py-1"):
                            ui.label(c.name).classes("text-sm")
                            ui.label(f"{_date(c.start_date)} – {_date(c.end_date)}"
                                     + (" · enrolled" if c.enrolled else "")).classes("ag-muted text-xs")

    refresh.on_click(load)
    await load()
