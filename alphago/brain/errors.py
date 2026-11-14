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


class StartThrottled(ConcurrencyLimit):
    """The API rate limit (not the concurrency cap) refused a simulation start: wait, do not shrink the slots."""


class QuotaExhausted(BrainError):
    """Daily simulation quota (or the configured reserve) reached."""


class SimulationRejected(BrainError):
    """BRAIN rejected the simulation request (400, invalid settings or expression).

    `per_child` holds one message per alpha (None = that alpha was fine) when BRAIN answered a multi-simulation's
    400 with a list as long as the batch; then only the alphas with a message were at fault."""

    def __init__(self, message: str, status: int | None = None, body: Any = None,
                 per_child: list[str | None] | None = None) -> None:
        super().__init__(message, status, body)
        self.per_child = per_child


class BatchRefused(BrainError):
    """403 on a multi-simulation: the account may not batch. Send the alphas one at a time instead."""


class MaybeDelivered(BrainError):
    """A request that spends quota may have reached BRAIN although no answer came back (read timeout, 500/502/504).

    It must never be resent blindly: the simulation could run (and be paid for) twice. The alphas are marked
    orphaned and matched against the alpha listing later (engine/reconcile.py)."""


# `detail` codes BRAIN puts in simulation-start refusals
DAILY_LIMIT_DETAIL = "DAILY_SIMULATION_LIMIT_EXCEEDED"
CONCURRENT_LIMIT_DETAIL = "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"

# Fixed wording for the `detail` codes of a refused sign-in (400/401/403/429), so a person reads what to do.
AUTH_DETAIL_MESSAGES: dict[str, str] = {
    "NO_CREDENTIALS": "No BRAIN email and password were sent. Enter them on Sync with BRAIN.",
    "INVALID_CREDENTIALS": "BRAIN refused this email and password. Check them on Sync with BRAIN; AlphaGo will not "
                           "retry on its own, so the account is not locked by repeated attempts.",
    "NOT_VERIFIED": "This BRAIN account has not verified its email address yet. Verify it on the BRAIN website.",
    "NOT_APPROVED": "This BRAIN account is not approved yet.",
    "INACTIVE": "This BRAIN account is inactive. Sign in on the BRAIN website to reactivate it.",
    "DISABLED": "This BRAIN account is disabled. Contact BRAIN support.",
    "LOCKED": "This BRAIN account is locked after too many failed sign-ins. Wait, or reset the password on BRAIN.",
    "DEACTIVATED": "This BRAIN account was deactivated.",
    "INQUIRY_INCOMPLETE": "The identity check is not finished yet. Complete it in the browser, then continue.",
    "RECERTIFICATION_FAILED": "BRAIN's recertification failed. Sign in on the BRAIN website for details.",
    "CONSULTANT_COUNTRY_CHANGE": "BRAIN needs to confirm a change of country. Sign in on the BRAIN website.",
    "ALPHA_SELLER": "This account type cannot use the BRAIN API.",
    "THROTTLED": "Too many sign-in attempts. BRAIN asks to wait before trying again.",
    "AGREEMENT_EXPIRED": "A BRAIN agreement has expired. Accept the new one on the BRAIN website.",
    "BIOMETRICS_THROTTLED": "Too many identity checks. BRAIN asks to wait before starting another.",
}


def detail_code(body: Any) -> str:
    """The upper-case `detail` code of an error body ('' when there is none)."""
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail.strip().upper()
    return ""


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
