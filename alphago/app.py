"""NiceGUI application entry: wires lifecycle hooks and starts the native window (or the browser / headless UI)."""

from __future__ import annotations

import hmac

from nicegui import app, ui

from . import runtime
from .ui.layout import root

TOKEN_COOKIE = "alphago_token"


def _require_token(token: str) -> None:
    """Without a valid token (?token=... once, then a cookie) every request is refused."""
    from fastapi import Request
    from fastapi.responses import PlainTextResponse, RedirectResponse

    @app.middleware("http")
    async def check(request: Request, call_next):  # noqa: ANN202
        given = request.query_params.get("token")
        if given is not None and hmac.compare_digest(given, token):
            response = RedirectResponse(str(request.url.remove_query_params("token")))
            response.set_cookie(TOKEN_COOKIE, token, httponly=True, samesite="strict")
            return response
        if hmac.compare_digest(request.cookies.get(TOKEN_COOKIE, ""), token):
            return await call_next(request)
        return PlainTextResponse("AlphaGo: open this address with ?token=<your token> once.", status_code=401)


def run(native: bool = True, port: int | None = None, host: str = "127.0.0.1", token: str | None = None,
        show: bool | None = None) -> None:
    app.on_startup(runtime.startup)
    app.on_shutdown(runtime.shutdown)
    if token:
        _require_token(token)
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
        host=host,
        port=port,
        show=(not native) if show is None else show,
        favicon="🔺",
        show_welcome_message=False,
    )
