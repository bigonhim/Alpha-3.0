"""The BRAIN session cookie kept across restarts (in Windows Credential Manager, never on disk in the clear).

Signing in costs a captcha and, for some accounts, a biometric check, and every sign-in counts toward BRAIN's
lockout. A session that is still valid is simply resumed: the cookie is restored and `GET /authentication` says
how long it has left. A dead one is cleared and the app signs in as before.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import keyring
from keyring.errors import KeyringError

SERVICE = "AlphaGo"
KEY = "brain_session_cookies"
MAX_BYTES = 2400                 # Windows Credential Manager blobs hold ~2.5 KB

log = logging.getLogger(__name__)


def export(jar: httpx.Cookies) -> list[dict[str, Any]]:
    return [{"name": c.name, "value": c.value, "domain": c.domain, "path": c.path, "expires": c.expires,
             "secure": c.secure} for c in jar.jar]


def save(jar: httpx.Cookies) -> None:
    data = json.dumps(export(jar), separators=(",", ":"))
    if len(data) > MAX_BYTES:
        log.info("BRAIN session cookies too large to keep (%d bytes); will sign in on next start", len(data))
        return
    try:
        keyring.set_password(SERVICE, KEY, data)
    except KeyringError:
        log.warning("keyring unavailable: BRAIN session not kept across restarts")


def load(jar: httpx.Cookies) -> bool:
    """Restore saved cookies into `jar`. True when some were restored."""
    try:
        data = keyring.get_password(SERVICE, KEY)
    except KeyringError:
        return False
    if not data:
        return False
    try:
        items = json.loads(data)
    except ValueError:
        return False
    for c in items:
        if c.get("name") and c.get("value") is not None:
            jar.set(c["name"], c["value"], domain=c.get("domain") or "", path=c.get("path") or "/")
    return bool(items)


def clear() -> None:
    try:
        keyring.delete_password(SERVICE, KEY)
    except (KeyringError, keyring.errors.PasswordDeleteError):
        pass
