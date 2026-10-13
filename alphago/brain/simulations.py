"""Simulation endpoints: single, multi (2–10 alphas sharing region/delay/type/language) and SUPER."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from .client import BrainClient
from .errors import BrainError, ConcurrencyLimit, QuotaExhausted, SimulationRejected, describe_error_body

log = logging.getLogger(__name__)

TERMINAL_OK = {"COMPLETE", "WARNING"}
TERMINAL_BAD = {"ERROR", "FAIL", "CANCELLED", "TIMEOUT"}

DEFAULT_SETTINGS: dict[str, Any] = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "INDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "testPeriod": "P0Y0M0D",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}

# settings that change the simulation result (used for de-duplication)
RESULT_KEYS = (
    "instrumentType", "region", "universe", "delay", "decay", "neutralization", "truncation",
    "pasteurization", "unitHandling", "nanHandling", "language", "maxTrade", "testPeriod",
    "selectionHandling", "selectionLimit", "componentActivation",
)


def normalize_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**DEFAULT_SETTINGS, **(settings or {})}
    merged["delay"] = int(merged["delay"])
    merged["decay"] = int(merged["decay"])
    merged["truncation"] = float(merged["truncation"])
    return merged


def settings_hash(settings: dict[str, Any]) -> str:
    norm = normalize_settings(settings)
    key = {k: norm.get(k) for k in RESULT_KEYS if norm.get(k) is not None}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def group_key(settings: dict[str, Any]) -> tuple:
    """Alphas in one multi-simulation must share these."""
    s = normalize_settings(settings)
    return (s["instrumentType"], s["region"], s["delay"], s["language"])


def regular_payload(expression: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "REGULAR", "settings": normalize_settings(settings), "regular": expression}


def super_payload(
    selection: str,
    combo: str,
    settings: dict[str, Any] | None = None,
    *,
    selection_limit: int = 100,
    selection_handling: str = "POSITIVE",
    nested: bool = False,
) -> dict[str, Any]:
    s = normalize_settings(settings)
    s.update({"selectionLimit": int(selection_limit), "selectionHandling": selection_handling})
    if nested:  # alternative shape reported by one community tool
        return {"type": "SUPER", "settings": s, "regular": {"selection": selection, "combo": combo}}
    return {"type": "SUPER", "settings": s, "selection": selection, "combo": combo}


@dataclass
class SimOutcome:
    index: int
    sim_id: str | None
    status: str
    alpha_id: str | None = None
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.alpha_id is not None


ProgressFn = Callable[[float | None], None]


class SimulationAPI:
    def __init__(self, client: BrainClient) -> None:
        self.client = client

    async def start(self, payloads: list[dict[str, Any]]) -> str:
        """POST the simulation; returns the Location URL to poll."""
        body: Any = payloads[0] if len(payloads) == 1 else payloads
        response = await self.client.request("POST", "/simulations", json=body, retry_statuses=(500, 502, 504))
        location = response.headers.get("Location")
        if response.status_code in (200, 201, 202) and location:
            self.client.quota.record_sent(len(payloads))
            return location
        content = self.client._json(response)
        text = describe_error_body(content)
        lowered = text.lower()
        if response.status_code == 429 or "concurrent" in lowered or "previous to finish" in lowered:
            self.client.slots.on_throttle()
            raise ConcurrencyLimit(text or "Concurrent simulation limit reached", response.status_code, content)
        if "limit" in lowered and ("daily" in lowered or "exceed" in lowered):
            raise QuotaExhausted(text, response.status_code, content)
        if response.status_code == 400:
            raise SimulationRejected(text or "Simulation rejected", 400, content)
        raise BrainError(text or f"Simulation start failed (HTTP {response.status_code})", response.status_code, content)

    async def _finish_one(self, url: str, index: int, on_progress: ProgressFn | None) -> SimOutcome:
        response = await self.client.poll(url, on_progress=on_progress)
        body = self.client._json(response) or {}
        if response.status_code >= 400 or not isinstance(body, dict):
            return SimOutcome(index, None, "ERROR", None, describe_error_body(body) or f"HTTP {response.status_code}")
        status = str(body.get("status") or "UNKNOWN").upper()
        alpha = body.get("alpha")
        message = body.get("message")
        if isinstance(message, (dict, list)):
            message = describe_error_body(message)
        return SimOutcome(index, str(body.get("id") or url.rsplit("/", 1)[-1]), status, alpha, message)

    async def wait(self, location: str, count: int, on_progress: ProgressFn | None = None) -> list[SimOutcome]:
        response = await self.client.poll(location, on_progress=on_progress)
        body = self.client._json(response) or {}
        if not isinstance(body, dict):
            return [SimOutcome(i, None, "ERROR", None, "Unexpected simulation response") for i in range(count)]
        children = body.get("children")
        if isinstance(children, list) and children:
            outcomes = await asyncio.gather(
                *(self._finish_one(f"/simulations/{child}", i, None) for i, child in enumerate(children)),
                return_exceptions=True,
            )
            result: list[SimOutcome] = []
            for i, out in enumerate(outcomes):
                if isinstance(out, BaseException):
                    result.append(SimOutcome(i, None, "ERROR", None, str(out)))
                else:
                    result.append(out)
            while len(result) < count:
                result.append(SimOutcome(len(result), None, "ERROR", None, "missing child simulation"))
            return result
        status = str(body.get("status") or "UNKNOWN").upper()
        if count > 1:
            message = describe_error_body(body.get("message")) or f"Multi-simulation {status}"
            return [SimOutcome(i, None, status, None, message) for i in range(count)]
        message = body.get("message")
        return [SimOutcome(0, str(body.get("id") or location.rsplit("/", 1)[-1]), status, body.get("alpha"),
                           describe_error_body(message) if message else None)]

    async def run(
        self,
        payloads: list[dict[str, Any]],
        *,
        on_started: Callable[[str], None] | None = None,
        on_progress: ProgressFn | None = None,
        max_start_attempts: int = 20,
    ) -> list[SimOutcome]:
        """Acquire a slot, start, wait. Retries start on concurrency rejections."""
        if not 1 <= len(payloads) <= 10:
            raise ValueError("a (multi-)simulation takes 1..10 alphas")
        async with self.client.slots.slot():
            attempt = 0
            while True:
                try:
                    location = await self.start(payloads)
                    break
                except ConcurrencyLimit:
                    attempt += 1
                    if attempt >= max_start_attempts:
                        raise
                    await asyncio.sleep(min(60.0, 5.0 * attempt))
            if on_started:
                on_started(location)
            started = time.time()
            outcomes = await self.wait(location, len(payloads), on_progress)
            log.info("simulation %s finished in %.0fs", location.rsplit("/", 1)[-1], time.time() - started)
            return outcomes

    async def super_selection_preview(
        self, selection: str, settings: dict[str, Any], selection_limit: int = 100, selection_handling: str = "POSITIVE"
    ) -> dict[str, Any]:
        s = normalize_settings(settings)
        params = {
            "selection": selection,
            "settings.instrumentType": s["instrumentType"],
            "settings.region": s["region"],
            "settings.delay": s["delay"],
            "selectionLimit": selection_limit,
            "selectionHandling": selection_handling,
            "limit": 10,
        }
        return await self.client.get_json("/simulations/super-selection", params=params)
