"""Exceptions raised by the BRAIN API layer."""

from __future__ import annotations

from typing import Any


class BrainError(Exception):
    def __init__(self, message: str, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.body = body


class AuthError(BrainError):
    """Bad or missing credentials."""


class PersonaRequired(BrainError):
    """BRAIN requires a biometric (Persona) verification in the browser before login can complete."""

    def __init__(self, url: str) -> None:
        super().__init__("Biometric verification required — open the link to continue.", 401)
        self.url = url


class ConcurrencyLimit(BrainError):
    """Too many simulations in flight for this account."""


class QuotaExhausted(BrainError):
    """Daily simulation quota (or the configured reserve) reached."""


class SimulationRejected(BrainError):
    """BRAIN rejected the simulation request (400, invalid settings or expression)."""


def describe_error_body(body: Any) -> str:
    """Flatten DRF-style error bodies ({"field": ["msg"]}, {"detail": "..."}, lists) into one line."""
    if body is None:
        return ""
    if isinstance(body, str):
        return body[:500]
    if isinstance(body, dict):
        if "detail" in body and isinstance(body["detail"], str):
            return body["detail"]
        if "message" in body and isinstance(body["message"], str):
            return body["message"]
        parts = []
        for key, value in body.items():
            if isinstance(value, list):
                value = "; ".join(str(v) for v in value)
            elif isinstance(value, dict):
                value = describe_error_body(value)
            parts.append(f"{key}: {value}")
        return " | ".join(parts)[:500]
    if isinstance(body, list):
        return " | ".join(describe_error_body(b) for b in body)[:500]
    return str(body)[:500]
