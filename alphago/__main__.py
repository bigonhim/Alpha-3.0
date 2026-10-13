"""`python -m alphago` — launch the desktop app. `--web` opens it in the browser instead of a native window."""

from __future__ import annotations

import multiprocessing
import sys


def main() -> None:
    multiprocessing.freeze_support()  # NiceGUI native mode spawns the pywebview process
    from .app import run

    web = "--web" in sys.argv
    port = None
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    run(native=not web, port=port or (8090 if web else None))


if __name__ in {"__main__", "__mp_main__"}:
    main()
