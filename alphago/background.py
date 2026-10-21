"""Running AlphaGo in the background: one engine at a time, keep the PC awake while work is queued, start at logon.

`AlphaGo.exe --headless` runs the engine without a window; the UI stays available at http://127.0.0.1:<port>.
"Start with Windows" registers a Task Scheduler task that runs it at logon as you, so it can read your keys in
Windows Credential Manager (a Windows service could not, without storing your Windows password).
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

from .config import app_dir, settings

log = logging.getLogger(__name__)

TASK_NAME = "AlphaGo"
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
_lock_handle = None


# ---------------------------------------------------------------- one engine at a time

def acquire_instance_lock(path: Path | None = None) -> bool:
    """True when this process is the only AlphaGo engine; two engines would spend the same quota twice."""
    global _lock_handle
    path = path or app_dir() / "engine.lock"
    handle = open(path, "a+")  # noqa: SIM115  (held open for the process lifetime)
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False
    _lock_handle = handle
    return True


def release_instance_lock() -> None:
    global _lock_handle
    if _lock_handle is not None:
        try:
            if os.name == "nt":
                import msvcrt
                _lock_handle.seek(0)
                msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        _lock_handle.close()
        _lock_handle = None


# ---------------------------------------------------------------- keep awake while busy

def set_awake(on: bool) -> bool:
    """Ask Windows not to sleep (on) or allow sleep again (off). No-op elsewhere."""
    if os.name != "nt":
        return False
    import ctypes
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))


def busy() -> bool:
    from .engine.research import research
    from .engine.simulator import simulator
    return bool(simulator.queued_count or simulator.running_count or any(
        research.is_running(t) for t in list(research._runners)))


async def keep_awake_loop() -> None:
    """Runs on the event loop's thread (the execution state belongs to the thread that set it)."""
    awake = False
    while True:
        try:
            want = settings.service.keep_awake and busy()
            if want != awake:
                set_awake(want)
                awake = want
                log.info("keep-awake %s", "on" if want else "off")
        except Exception:
            log.exception("keep-awake failed")
        await asyncio.sleep(60)


# ---------------------------------------------------------------- start at logon (Task Scheduler)

def headless_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --headless'
    exe = Path(sys.executable)
    windowless = exe.with_name("pythonw.exe")
    return f'"{windowless if windowless.exists() else exe}" -m alphago --headless'


def _schtasks(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["schtasks", *args], capture_output=True, text=True, timeout=30)


def autostart_enabled() -> bool:
    if os.name != "nt":
        return False
    try:
        return _schtasks("/Query", "/TN", TASK_NAME).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def set_autostart(on: bool) -> str:
    """Create or delete the logon task. Returns a message for the user."""
    if os.name != "nt":
        return "Start with Windows is only available on Windows."
    if on:
        r = _schtasks("/Create", "/F", "/SC", "ONLOGON", "/RL", "LIMITED", "/TN", TASK_NAME, "/TR",
                      headless_command())
    else:
        r = _schtasks("/Delete", "/F", "/TN", TASK_NAME)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "schtasks failed").strip())
    return (r.stdout or "").strip() or ("Scheduled." if on else "Removed.")
