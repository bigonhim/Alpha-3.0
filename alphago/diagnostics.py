"""A file a consultant can send when they ask for help: what runs, and the end of the log, scrubbed.

Ported from alpha-harness ``diagnostics.py``. The log names the Windows username in every path, and carries Persona
inquiry ids and Alpha ids. The export keeps where a path was without whose it was, and swaps each id for a short
hash, so one id can still be followed through the file without being readable.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import platform
import re
import sys
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePath

LOG_FILE = "alphago.log"
#: The end of the log is where a fault is; whole, a long-running log grows to tens of MB.
TAIL_BYTES = 4 * 1024 * 1024
_INQUIRY = re.compile(r"\binq_[A-Za-z0-9]+")
_ALPHA = re.compile(
    r"(?P<lead>\balpha_?id['\"]?\s*[=:]\s*['\"]?|/alphas/)(?P<id>[A-Za-z0-9]{5,12})\b",
    re.IGNORECASE,
)


def _hashed(prefix: str, value: str) -> str:
    return prefix + hashlib.sha256(value.encode()).hexdigest()[:10]


def scrubber(home: PurePath, user: str) -> Callable[[str], str]:
    """What removes ``home``, ``user``, inquiry ids and Alpha ids from a text."""
    parts = [p for p in (part.rstrip("\\/") for part in home.parts) if p]
    # Either slash and any case: one log has C:\Users\Name, c:/users/name and C:\\Users\\Name.
    folder = re.compile(
        (r"[\\/]" if home.anchor in ("/", "\\") else "") + r"[\\/]+".join(re.escape(p) for p in parts),
        re.IGNORECASE,
    )
    # Too short a name would match inside ordinary words.
    name = (re.compile(rf"(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])", re.IGNORECASE)
            if len(user) >= 3 else None)

    def scrub(text: str) -> str:
        text = folder.sub("~", text) if parts else text
        if name is not None:
            text = name.sub("<user>", text)
        text = _INQUIRY.sub(lambda m: _hashed("inq#", m.group()), text)
        return _ALPHA.sub(lambda m: m.group("lead") + _hashed("alpha#", m.group("id")), text)

    return scrub


def _user() -> str:
    try:
        return getpass.getuser()
    except OSError:
        return Path.home().name


def _tail(folder: Path) -> str:
    """The last :data:`TAIL_BYTES` of the log, reaching into the rotated one when it is short."""
    chunks: list[bytes] = []
    left = TAIL_BYTES
    for name in (LOG_FILE, f"{LOG_FILE}.1"):
        if left <= 0:
            break
        try:
            with (folder / name).open("rb") as log:
                size = log.seek(0, os.SEEK_END)
                log.seek(max(0, size - left))
                chunk = log.read()
        except OSError:
            continue
        chunks.insert(0, chunk)
        left -= len(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _about() -> str:
    from . import __version__
    return (
        f"AlphaGo {__version__}\n"
        f"System: {platform.platform()} {platform.machine()}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"Written: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
    )


def export(downloads: Path | None = None) -> Path:
    """Write the zip into Downloads, where a person looks for what they just saved (the app folder without one)."""
    from .config import log_dir
    logs = log_dir()
    downloads = downloads or Path.home() / "Downloads"
    folder = downloads if downloads.is_dir() else logs
    target = folder / f"alphago-diagnostics-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.zip"
    scrub = scrubber(Path.home(), _user())
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("about.txt", scrub(_about()))
        archive.writestr(LOG_FILE, scrub(_tail(logs)))
    return target


def open_folder(folder: Path) -> None:
    """Show ``folder`` in the system's file manager."""
    if sys.platform == "win32":
        os.startfile(folder)  # noqa: S606 - a folder of ours, opened by the shell
    else:
        import subprocess
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])  # noqa: S603
