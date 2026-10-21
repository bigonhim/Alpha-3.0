"""`python -m alphago` — launch the desktop app.

--web             the same app in your browser (http://127.0.0.1:8090) instead of a native window
--headless        run the engine without a window (for "Start with Windows"); the UI stays at http://127.0.0.1:8090
--port N          serve on another port
--host ADDR       listen on another address (e.g. 0.0.0.0 for another PC on your network) — requires --token
--token SECRET    required to open the UI when --host is not 127.0.0.1 (open http://host:port/?token=SECRET once)
"""

from __future__ import annotations

import multiprocessing
import sys


def _arg(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv and sys.argv.index(name) + 1 < len(sys.argv) \
        else None


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, message, "AlphaGo", 0x10)
        except Exception:
            pass
    sys.exit(2)


def main() -> None:
    multiprocessing.freeze_support()  # NiceGUI native mode spawns the pywebview process
    from .app import run

    web = "--web" in sys.argv
    headless = "--headless" in sys.argv
    port = int(_arg("--port")) if _arg("--port") else None
    host = _arg("--host") or "127.0.0.1"
    token = _arg("--token")
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        _fail("--host other than 127.0.0.1 needs --token SECRET (the UI has no other login).")
    if multiprocessing.parent_process() is None:          # not the native window's helper process
        from .background import acquire_instance_lock
        if not acquire_instance_lock():
            _fail("AlphaGo is already running on this PC (window or background). Close it first — two engines "
                  "would spend the same simulation quota twice.")
    run(native=not (web or headless), port=port or (8090 if (web or headless) else None), host=host, token=token,
        show=web)


if __name__ in {"__main__", "__mp_main__"}:
    main()
