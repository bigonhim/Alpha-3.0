"""PyInstaller entry point (an entry script cannot use the package's relative imports)."""

import os
import sys

# A --windowed build has no console: stdout/stderr are None and uvicorn's logging would crash on them.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")  # noqa: SIM115

from alphago.__main__ import main  # noqa: E402

if __name__ in {"__main__", "__mp_main__"}:
    main()
