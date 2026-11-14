"""BRAIN calls the catalogue, pyramid, quarter and competition features need (ported from alpha-harness endpoints.py).

Most ride on `brain.client.get_json`. Two need what that helper cannot send: `GET /data-fields` at `Accept:
version=3.0` (with all four scope parameters it returns a whole market in one response) and a longer read timeout for
those ~40 MB answers. `versioned_get` adds exactly that on top of the shared client: the same session, the same rate
gate and 429 cooldown, the same observers and quota header tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import date, datetime, time, timedelta
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

from .client import BrainClient
from .errors import BrainError, describe_error_body
from .ratelimit import bucket_for

log = logging.getLogger(__name__)

#: BRAIN's day and quarter run on New York time (the daily quota resets at midnight US Eastern).
PLATFORM_TZ = ZoneInfo("America/New_York")

#: GET /data-fields with all four scope parameters: one unpaginated answer at this version.
V_FIELDS_ALL = "3.0"
#: A market's fields are ~40 MB; measured transfers take about forty seconds, so the ordinary timeout cuts them off.
ALL_FIELDS_TIMEOUT = 120.0
#: A region's datasets take eight to sixteen seconds; BRAIN itself gives up at thirty with a 504.
ALL_SETS_TIMEOUT = 45.0
#: Region ALL answers fifty fields at a time and stops at 10,000 per query.
ALL_FIELDS_PAGE = 50
ALL_FIELDS_CEILING = 10_000
#: The alpha listing's page size and the deepest offset BRAIN serves.
ALPHA_LIST_PAGE = 100
ALPHA_LIST_MAX_OFFSET = 1_000
#: BRAIN's daily submitted-alpha counts ("Submitted Alphas" on its own site).
SUBMISSIONS_PATH = "/users/self/activities/submissions"
TRANSIENT = (500, 502, 503, 504)


def _decode(content: bytes) -> Any:
    return json.loads(content) if content else None


async def versioned_get(client: BrainClient, path: str, *, params: dict[str, Any] | None = None,
                        version: str | None = None, timeout: float | None = None, retries: int = 5) -> Any:
    """GET with an `Accept: application/json;version=…` header and an optional read timeout, through the shared
    client's session, rate gate and observers. Large bodies are decoded off the event loop."""
    await client.start()
    await client.ensure_session()
    headers = {"Accept": f"application/json;version={version}"} if version else None
    request_timeout = httpx.Timeout(timeout, connect=20.0) if timeout else httpx.USE_CLIENT_DEFAULT
    bucket = bucket_for(path)
    relogged = False
    attempt = throttles = 0
    while True:
        await client.gate.acquire(bucket)
        try:
            response = await client.http.request("GET", path, params=params, headers=headers, timeout=request_timeout)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            attempt += 1
            client._observe("GET", path, 0, None, attempt > retries, None)
            if attempt > retries:
                raise BrainError(f"Network error talking to BRAIN ({path}): {exc.__class__.__name__}") from exc
            await asyncio.sleep(min(30.0, 2 ** attempt + random.random()))
            continue
        client.quota.update_from_headers(response.headers)
        status = response.status_code
        throttled = status == 429 and throttles < client.RATE_LIMIT_RETRIES
        transient = status in TRANSIENT and attempt < retries
        reauth = status == 401 and not relogged
        client._observe("GET", path, status, None, not (throttled or transient or reauth), response)
        if reauth:
            relogged = True
            await client.login()
            continue
        if throttled:
            throttles += 1
            retry_after = response.headers.get("Retry-After")
            client.gate.throttled(bucket, float(retry_after) if retry_after and retry_after.isdigit() else None)
            continue
        if transient:
            attempt += 1
            await asyncio.sleep(min(60.0, 2 ** attempt + random.random()))
            continue
        body = await asyncio.to_thread(_decode, response.content) if len(response.content) > 1_000_000 \
            else _decode(response.content)
        if status >= 400:
            raise BrainError(f"GET {path} -> {status}: {describe_error_body(body)}", status, body)
        client.gate.succeeded(bucket)
        return body


def _rows(body: Any) -> list[dict[str, Any]]:
    rows = body if isinstance(body, list) else (body or {}).get("results") if isinstance(body, dict) else None
    return [r for r in rows or [] if isinstance(r, dict)]


class CatalogAPI:
    """The BRAIN surface of the Data Explorer, Pyramids, Competitions and the quarter standing."""

    def __init__(self, client: BrainClient) -> None:
        self.client = client

    # ---------------------------------------------------------------- data catalogue

    @staticmethod
    def scope(region: str, delay: int, universe: str, instrument_type: str = "EQUITY") -> dict[str, Any]:
        return {"instrumentType": instrument_type, "region": region, "delay": int(delay), "universe": universe}

    async def data_fields_all(self, region: str, delay: int, universe: str,
                              instrument_type: str = "EQUITY") -> list[dict[str, Any]]:
        """Every field of one market in one request (version 3.0, all four scope parameters). BRAIN answers with a
        bare list or a `{results}` envelope; both are read."""
        body = await versioned_get(self.client, "/data-fields", params=self.scope(region, delay, universe,
                                                                                    instrument_type),
                                   version=V_FIELDS_ALL, timeout=ALL_FIELDS_TIMEOUT)
        return _rows(body)

    async def data_fields_paged(self, dataset_id: str, region: str, delay: int, universe: str,
                                instrument_type: str = "EQUITY") -> list[dict[str, Any]]:
        """Every field of one dataset, fifty at a time. Region ALL answers an unlimited read with nothing and caps
        its flat listing at 10,000, so this is its only complete route."""
        out: list[dict[str, Any]] = []
        offset = 0
        base = {**self.scope(region, delay, universe, instrument_type), "dataset.id": dataset_id,
                "limit": ALL_FIELDS_PAGE}
        while offset < ALL_FIELDS_CEILING:
            page = _rows(await versioned_get(self.client, "/data-fields", params={**base, "offset": offset},
                                             version=V_FIELDS_ALL))
            out.extend(page)
            if len(page) < ALL_FIELDS_PAGE:
                return out
            offset += ALL_FIELDS_PAGE
        raise BrainError(f"{dataset_id} has more than {ALL_FIELDS_CEILING:,} fields, which is as far as BRAIN will "
                         "page this market.")

    async def data_sets_all(self, **params: Any) -> list[dict[str, Any]]:
        """Every dataset matching a scope, however partial (naming only the region answers for all its markets).
        Without `limit` BRAIN returns the whole list; a short answer against its `count` fails rather than quietly
        shrinking the catalogue."""
        body = await versioned_get(self.client, "/data-sets", params=params, timeout=ALL_SETS_TIMEOUT)
        rows = _rows(body)
        total = body.get("count") if isinstance(body, dict) else None
        if isinstance(total, int) and len(rows) < total:
            raise BrainError(f"/data-sets returned {len(rows)} of {total} datasets")
        return rows

    # ---------------------------------------------------------------- pyramids and quarters

    async def pyramid_multipliers(self) -> list[dict[str, Any]]:
        """`{category, region, delay, multiplier}` for every pyramid on this account (shape `{pyramids: [...]}`)."""
        body = await self.client.get_json("/users/self/activities/pyramid-multipliers")
        items = body.get("pyramids") if isinstance(body, dict) else None
        return [i for i in items or [] if isinstance(i, dict)]

    async def pyramid_alphas(self, start_date: str | None = None, end_date: str | None = None) -> list[dict[str, Any]]:
        """`{category, region, delay, alphaCount}` submitted between two ISO dates, or ever without them."""
        dates = {"startDate": start_date, "endDate": end_date}
        body = await self.client.get_json("/users/self/activities/pyramid-alphas",
                                          params={k: v for k, v in dates.items() if v is not None} or None)
        items = body.get("pyramids") if isinstance(body, dict) else None
        return [i for i in items or [] if isinstance(i, dict)]

    async def quarters(self) -> list[dict[str, str]]:
        """`{name, startDate, endDate}` for each quarter BRAIN's Genius page offers, newest first, each once.
        `name` reads `2026-Q4`; both dates are inclusive."""
        body = await self.client.get_json("/users/self/consultant/summary")
        performance = (body or {}).get("performance") or {} if isinstance(body, dict) else {}
        listed = [performance.get("currentQuarter")] + [h.get("quarter") for h in performance.get("history") or []
                                                        if isinstance(h, dict)]
        unique = {q["name"]: q for q in listed if isinstance(q, dict) and q.get("name")}
        return sorted(unique.values(), key=lambda q: q.get("startDate") or "", reverse=True)

    async def submission_activity(self) -> list[tuple[str, int]]:
        """Alphas submitted per day as `(date, count)`. The envelope's `current` window is two months, so the dated
        rows are what a quarterly figure is built from."""
        body = await self.client.get_json(SUBMISSIONS_PATH)
        rows = (((body or {}).get("records") or {}).get("records") or []) if isinstance(body, dict) else []
        return [(str(r[0]), int(r[1])) for r in rows
                if isinstance(r, list) and len(r) >= 2 and str(r[1]).lstrip("-").isdigit()]

    async def submitted_alphas(self, start: date, after: date) -> tuple[list[dict[str, Any]], int]:
        """The alphas submitted in `[start, after)` (New York days), hidden ones included, and how many BRAIN says
        there are. Stops at BRAIN's offset cap; the difference is reported as unlisted."""
        def moment(d: date) -> str:
            return quote(datetime.combine(d, time(0), PLATFORM_TZ).isoformat(), safe="")
        window = f"status%21=UNSUBMITTED&dateSubmitted%3E={moment(start)}&dateSubmitted%3C{moment(after)}"
        alphas: list[dict[str, Any]] = []
        total = 0
        for offset in range(0, ALPHA_LIST_MAX_OFFSET + 1, ALPHA_LIST_PAGE):
            page = await self.client.get_json(f"/users/self/alphas?limit={ALPHA_LIST_PAGE}&offset={offset}"
                                              f"&order=-dateSubmitted&{window}&hidden=true%1Ffalse")
            results = _rows(page)
            total = int((page or {}).get("count") or 0) if isinstance(page, dict) else len(results)
            alphas.extend(results)
            if len(results) < ALPHA_LIST_PAGE:
                break
        return alphas, max(total, len(alphas))

    # ---------------------------------------------------------------- competitions

    async def competitions(self) -> list[dict[str, Any]]:
        """Every competition, past and ongoing, with this account's `status` in each."""
        return _rows(await self.client.get_json("/competitions"))

    async def competition(self, competition_id: str) -> dict[str, Any]:
        """One competition; only this carries the account's `leaderboard` standing."""
        body = await self.client.get_json(f"/competitions/{quote(str(competition_id), safe='')}")
        return body if isinstance(body, dict) else {}


def ny_today() -> date:
    return datetime.now(PLATFORM_TZ).date()


def quarter_start(day: date) -> date:
    return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


def next_quarter_start(day: date) -> date:
    start = quarter_start(day)
    return date(start.year + 1, 1, 1) if start.month == 10 else date(start.year, start.month + 3, 1)


def quarter_label(start: date) -> str:
    return f"{start.year}-Q{(start.month - 1) // 3 + 1}"


def quarter_last_day(day: date) -> date:
    return next_quarter_start(day) - timedelta(days=1)


_api: CatalogAPI | None = None


def catalog_api() -> CatalogAPI:
    """The shared instance, bound to the app's one BRAIN client."""
    global _api
    from . import brain
    if _api is None or _api.client is not brain.client:
        _api = CatalogAPI(brain.client)
    return _api
