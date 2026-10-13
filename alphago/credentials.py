"""Credentials in Windows Credential Manager (via keyring). Never logged, never written to disk by the app."""

from __future__ import annotations

import logging

import keyring
from keyring.errors import KeyringError

SERVICE = "AlphaGo"
log = logging.getLogger(__name__)


def _get(key: str) -> str | None:
    try:
        return keyring.get_password(SERVICE, key)
    except KeyringError:
        log.warning("keyring unavailable while reading %s", key)
        return None


def _set(key: str, value: str | None) -> None:
    try:
        if value:
            keyring.set_password(SERVICE, key, value)
        else:
            try:
                keyring.delete_password(SERVICE, key)
            except keyring.errors.PasswordDeleteError:
                pass
    except KeyringError:
        log.warning("keyring unavailable while writing %s", key)


def brain_credentials() -> tuple[str, str] | None:
    email, password = _get("brain_email"), _get("brain_password")
    return (email, password) if email and password else None


def set_brain_credentials(email: str | None, password: str | None) -> None:
    _set("brain_email", email)
    _set("brain_password", password)


def brain_email() -> str | None:
    return _get("brain_email")


def gemini_key() -> str | None:
    return _get("gemini_api_key")


def set_gemini_key(key: str | None) -> None:
    _set("gemini_api_key", key)
