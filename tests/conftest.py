"""Run every test against a throwaway app folder, never the user's real %APPDATA%\\AlphaGo.

`alphago.config` loads settings.json at import time, so ALPHAGO_HOME must be set before anything imports alphago.

Tests also never see the real Windows Credential Manager: an in-memory keyring replaces it, and the BRAIN client
starts every test without credentials, so no code path under test can sign in to a real BRAIN account. Tests that
talk to a mocked BRAIN set their own fake credentials.
"""

import os
import tempfile

import keyring
import pytest
from keyring.backend import KeyringBackend

os.environ["ALPHAGO_HOME"] = tempfile.mkdtemp(prefix="alphago-tests-")


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self) -> None:
        super().__init__()
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self._store.pop((service, username), None)


keyring.set_keyring(MemoryKeyring())


class RealNetworkBlocked(RuntimeError):
    """Raised by any HTTP request a test makes outside a respx mock (not a transport error, so nothing retries)."""


@pytest.fixture(autouse=True)
def _no_real_brain_or_network(monkeypatch):
    """Every test starts signed out and without credentials, and cannot reach the real network: respx mocks
    (which patch the same httpcore methods) still work; anything unmocked fails at once."""
    from httpcore._backends.auto import AutoBackend
    from httpcore._backends.sync import SyncBackend

    from alphago.brain import brain

    def blocked(self, host=None, port=None, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        raise RealNetworkBlocked(f"real network disabled in tests: connection to {host}:{port}")

    async def blocked_async(self, host=None, port=None, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        blocked(self, host, port)

    # the socket layer: respx answers mocked requests before reaching it, so only real traffic is stopped
    for name in ("connect_tcp", "connect_unix_socket"):
        monkeypatch.setattr(SyncBackend, name, blocked)
        monkeypatch.setattr(AutoBackend, name, blocked_async)
    brain.client._credentials = lambda: None
    brain.client.state, brain.client.expiry_at = "logged_out", None
    from alphago.engine import catalogs
    catalogs.invalidate()                     # per-scope catalogues are cached; each test has its own database
    # the app's singletons hold asyncio primitives that bind to the first event loop that waits on them, and every
    # test runs in a new loop: give each test fresh ones (and a fresh HTTP client)
    import asyncio

    from alphago.brain.slots import SlotManager
    from alphago.db import db
    brain.client._login_lock = asyncio.Lock()
    brain.client.slots = SlotManager(brain.client.cfg.max_slots)
    brain.client._http = None
    brain.client.observers.clear()
    db._write_lock = asyncio.Lock()
    yield
