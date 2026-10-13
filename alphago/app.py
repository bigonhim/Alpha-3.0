"""NiceGUI application entry: wires lifecycle hooks and starts the native window."""

from __future__ import annotations

from nicegui import app, ui

from . import runtime
from .ui.layout import root


def run(native: bool = True, port: int | None = None) -> None:
    app.on_startup(runtime.startup)
    app.on_shutdown(runtime.shutdown)
    if native:
        app.native.window_args["min_size"] = (1200, 760)
        app.native.settings["ALLOW_DOWNLOADS"] = True
    ui.run(
        root,
        title="AlphaGo · BRAIN Alpha Research",
        native=native,
        window_size=(1600, 960) if native else None,
        dark=True,
        reload=False,
        port=port,
        show=not native,
        favicon="🔺",
        show_welcome_message=False,
    )
