"""Build the Windows app with PyInstaller (one-folder: faster start-up than one-file).

    .venv\\Scripts\\python.exe scripts\\build_exe.py            # -> dist\\AlphaGo\\AlphaGo.exe
    .venv\\Scripts\\python.exe scripts\\build_exe.py --console  # keep a console window for debugging

Zip `dist\\AlphaGo` to distribute. The target PC needs the Microsoft Edge WebView2 runtime (preinstalled on
Windows 10/11); without it, run `AlphaGo.exe --web` to use the browser instead of a native window.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import nicegui
import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent.parent
SEP = os.pathsep


def main() -> None:
    console = "--console" in sys.argv
    args = [
        str(ROOT / "scripts" / "launch.py"),
        "--name", "AlphaGo",
        "--onedir",
        "--noconfirm",
        "--clean",
        "--console" if console else "--windowed",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT / "build"),
        "--paths", str(ROOT),
        # NiceGUI serves its Vue/Quasar bundle and element JS from the package directory.
        "--add-data", f"{Path(nicegui.__file__).parent}{SEP}nicegui",
        "--add-data", f"{ROOT / 'alphago' / 'fastexpr' / 'grammar.lark'}{SEP}alphago/fastexpr",
        "--collect-submodules", "alphago",
        # Loaded dynamically / via entry points, so static analysis misses them.
        "--hidden-import", "sqlalchemy.dialects.sqlite.aiosqlite",
        "--hidden-import", "aiosqlite",
        "--collect-submodules", "keyring.backends",
        "--copy-metadata", "keyring",
        "--hidden-import", "win32ctypes.core",
        "--hidden-import", "clr",                 # pythonnet, used by pywebview's EdgeChromium backend
        "--collect-data", "tzdata",               # zoneinfo on Windows (US/Eastern quota day, Pacific Gemini day)
        "--collect-data", "lark",
        "--copy-metadata", "google-genai",
        # SPC Lab PnL test: yfinance talks to Yahoo through curl_cffi (compiled libcurl + certificates)
        "--collect-all", "curl_cffi",
        "--collect-submodules", "yfinance",
        "--copy-metadata", "yfinance",
        "--exclude-module", "tkinter",
        "--exclude-module", "matplotlib",
        "--exclude-module", "pytest",
    ]
    icon = ROOT / "scripts" / "alphago.ico"
    if icon.exists():
        args += ["--icon", str(icon)]
    PyInstaller.__main__.run(args)
    # Ship the user guide and its screenshots next to the exe so a zipped copy of dist\AlphaGo is self-explanatory.
    out = ROOT / "dist" / "AlphaGo"
    shutil.copy2(ROOT / "USER_GUIDE.md", out / "USER_GUIDE.md")
    shutil.copytree(ROOT / "docs" / "user-guide", out / "docs" / "user-guide", dirs_exist_ok=True)
    print(f"\nBuilt {ROOT / 'dist' / 'AlphaGo' / 'AlphaGo.exe'}")


if __name__ == "__main__":
    main()
