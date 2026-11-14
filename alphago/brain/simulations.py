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
from .errors import (
    CONCURRENT_LIMIT_DETAIL,
    DAILY_LIMIT_DETAIL,
    BatchRefused,
    BrainError,
    ConcurrencyLimit,
    MaybeDelivered,
    QuotaExhausted,
    SimulationRejected,
    StartThrottled,
    describe_error_body,
    detail_code,
)
from .ratelimit import bucket_for

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
    "pasteurization", "unitHandling", "nanHandling", "language", "maxTrade", "maxPosition", "testPeriod",
    "selectionHandling", "selectionLimit", "componentActivation", "simulationMode",
)


def normalize_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**DEFAULT_SETTINGS, **(settings or {})}
    merged["delay"] = int(merged["delay"])
    merged["decay"] = int(merged["decay"])
    merged["truncation"] = float(merged["truncation"])
    return merged


def settings_hash(settings: dict[str, Any]) -> str:
    norm = normalize_settings(settings)
    # simulationMode counts only when QUICK: a Quick alpha must never answer a Full request (BRAIN needs the Full run
    # to submit), while Full and unspecified stay one identity so every existing hash is unchanged
    key = {k: norm.get(k) for k in RESULT_KEYS if norm.get(k) is not None
           and not (k == "simulationMode" and str(norm.get(k)).upper() != "QUICK")}
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def group_key(settings: dict[str, Any]) -> tuple:
    """Alphas in one multi-simulation must share these."""
    s = normalize_settings(settings)
    return (s["instrumentType"], s["region"], s["delay"], s["language"])


def slot_cost(settings: dict[str, Any] | None) -> int:
    """Concurrent-simulation slots one run takes: BRAIN lets only 4 GLB runs go at once (2 slots of 8)."""
    region = str((settings or {}).get("region") or "").upper()
    return 2 if region == "GLB" else 1


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


def _retry_after_s(response: Any) -> float | None:
    try:
        return float(response.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


class SimulationAPI:
    def __init__(self, client: BrainClient) -> None:
        self.client = client

    async def start(self, payloads: list[dict[str, Any]]) -> str:
        """POST the simulation; returns the Location URL to poll.

        The POST is never resent after an answer that may mean BRAIN took it (500/502/504, a read timeout): that
        raises MaybeDelivered, and the alphas are reconciled against the alpha listing instead of paid for twice.
        Only a 503 (BRAIN says it did not take the request) and a failed connect are retried."""
        body: Any = payloads[0] if len(payloads) == 1 else payloads
        response = await self.client.request("POST", "/simulations", json=body, retry_statuses=(503,),
                                             idempotent=False)
        location = response.headers.get("Location")
        status = response.status_code
        if status in (200, 201, 202) and location:
            self.client.quota.record_sent(len(payloads))
            return location
        content = self.client._json(response)
        text = describe_error_body(content)
        lowered = text.lower()
        code = detail_code(content)
        # the daily cap first: it also answers 429, and must stop the queue rather than be retried
        if code == DAILY_LIMIT_DETAIL or ("limit" in lowered and "daily" in lowered):
            self.client.quota.mark_exhausted()
            raise QuotaExhausted(text or "Daily simulation limit reached", status, content)
        if code == CONCURRENT_LIMIT_DETAIL or "concurrent" in lowered or "previous to finish" in lowered:
            self.client.slots.on_throttle()
            raise ConcurrencyLimit(text or "Concurrent simulation limit reached", status, content)
        if status == 429:            # the API rate limit: wait it out, the account's slots are fine
            self.client.gate.throttled(bucket_for("/simulations"), _retry_after_s(response))
            raise StartThrottled(text or "BRAIN rate limit", status, content)
        if status in (500, 502, 504):
            raise MaybeDelivered(f"BRAIN answered HTTP {status} to a simulation start; it may have been taken, "
                                 "so it is not resent.", status, content)
        if status == 403 and len(payloads) > 1:
            raise BatchRefused(text or "BRAIN refused a multi-simulation", status, content)
        if status == 400:
            per_child = None
            if isinstance(content, list) and len(content) == len(payloads) and len(payloads) > 1:
                per_child = [describe_error_body(item) or None for item in content]
            raise SimulationRejected(text or "Simulation rejected", 400, content, per_child=per_child)
        raise BrainError(text or f"Simulation start failed (HTTP {status})", status, content)

    async def cancel(self, sim_id: str) -> bool:
        """DELETE a running (multi-)simulation. True when BRAIN accepted; a non-empty list body means refused."""
        response = await self.client.request("DELETE", f"/simulations/{sim_id}", retry_statuses=(502, 503, 504))
        body = self.client._json(response)
        if response.status_code >= 400:
            return response.status_code == 404
        return not (isinstance(body, list) and body)

    LOST_AFTER_S = 30 * 60        # 5xx answers this long in a row = the run is lost
    STALL_AFTER_S = 3 * 3600      # progress unchanged this long = stuck on BRAIN: cancel it there
    UNAUTHORIZED_WAIT_S = 10.0

    async def poll_sim(self, url: str, on_progress: ProgressFn | None = None) -> Any:
        """Read a simulation until it finishes; returns the final httpx.Response.

        There is no overall deadline (a long multi-simulation is normal). A run ends as lost only after 30 minutes
        of server errors, and is cancelled on BRAIN after 3 hours without progress."""
        failing_since: float | None = None
        errors = 0
        last_progress: float | None = None
        progress_since = time.time()
        while True:
            try:
                response = await self.client.request("GET", url, retries=0, retry_statuses=(429,))
            except BrainError as exc:                   # network trouble: keep reading, it is still running
                failing_since = failing_since or time.time()
                if time.time() - failing_since > self.LOST_AFTER_S:
                    raise BrainError(f"Lost the simulation {url}: no answer for 30 minutes ({exc.message})") from exc
                errors += 1
                await asyncio.sleep(min(240.0, 15.0 * 2 ** (errors - 1)))
                continue
            status = response.status_code
            if status >= 500:
                failing_since = failing_since or time.time()
                if time.time() - failing_since > self.LOST_AFTER_S:
                    return response
                errors += 1
                await asyncio.sleep(min(240.0, 15.0 * 2 ** (errors - 1)))
                continue
            failing_since, errors = None, 0
            if status == 401:
                await asyncio.sleep(self.UNAUTHORIZED_WAIT_S)
                continue
            wait = _retry_after_s(response)
            if wait is None or status >= 400:
                return response
            body = self.client._json(response)
            progress = body.get("progress") if isinstance(body, dict) else None
            if isinstance(progress, (int, float)):
                if progress != last_progress:
                    last_progress, progress_since = float(progress), time.time()
                if on_progress is not None:
                    try:
                        on_progress(float(progress))
                    except Exception:
                        log.exception("progress callback failed")
            if time.time() - progress_since > self.STALL_AFTER_S:
                sim_id = url.rstrip("/").rsplit("/", 1)[-1]
                await self.cancel(sim_id)
                raise BrainError(f"TIMEOUT: simulation {sim_id} made no progress for 3 hours; cancelled on BRAIN.")
            await asyncio.sleep(min(30.0, max(0.5, wait)))

    async def _finish_one(self, url: str, index: int, on_progress: ProgressFn | None) -> SimOutcome:
        response = await self.poll_sim(url, on_progress=on_progress)
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
        response = await self.poll_sim(location, on_progress=on_progress)
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
        on_started: Callable[[str], Any] | None = None,
        on_progress: ProgressFn | None = None,
        max_start_attempts: int = 8,
    ) -> list[SimOutcome]:
        """Acquire a slot, start, wait.

        A concurrency or rate-limit refusal spent nothing, so the start is retried, backing off 30 s to 5 min; after
        `max_start_attempts` it is raised for the caller to requeue. `on_started` may be async and is awaited before
        polling, so the simulation is on record first."""
        if not 1 <= len(payloads) <= 10:
            raise ValueError("a (multi-)simulation takes 1..10 alphas")
        async with self.client.slots.slot(slot_cost(payloads[0].get("settings"))):
            attempt = 0
            while True:
                try:
                    location = await self.start(payloads)
                    break
                except ConcurrencyLimit:
                    attempt += 1
                    if attempt >= max_start_attempts:
                        raise
                    await asyncio.sleep(min(300.0, 30.0 * 2 ** (attempt - 1)))
            self.client.slots.on_accepted()
            if on_started:
                res = on_started(location)
                if asyncio.iscoroutine(res):
                    await res
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
