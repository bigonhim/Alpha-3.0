"""Alpha endpoints: details, properties, submission check, submit, correlations, record sets, listing."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator
from urllib.parse import quote, urlencode

from .client import BrainClient
from .errors import BrainError, describe_error_body

log = logging.getLogger(__name__)

LIST_CAP = 1000  # GET /users/self/alphas answers 400 "Cannot display more than the first 1,000 alphas" past this


def _next_window(edge: str, *, descending: bool) -> str:
    """Raw filter for the listing window after the alpha created at `edge` (an ISO time as BRAIN returned it).

    Descending uses `dateCreated<` the next whole second, so alphas in the edge's own second are fetched again (and
    skipped as seen) rather than lost; ascending uses `dateCreated>=` the edge itself."""
    if descending:
        bound = (datetime.fromisoformat(edge).replace(microsecond=0)
                 + timedelta(seconds=1)).isoformat(timespec="seconds")
        return f"dateCreated%3C{quote(bound, safe=':')}"
    return f"dateCreated%3E={quote(edge, safe=':')}"


@dataclass
class SubmitResult:
    status: str                       # active | accepted | rejected | throttled | error
    message: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)


def _records(table: dict[str, Any] | None) -> list[dict[str, Any]]:
    """BRAIN record sets come as {schema: {properties: [{name}]}, records: [[...]]}."""
    if not table:
        return []
    names = [p.get("name") for p in (table.get("schema") or {}).get("properties", [])]
    return [dict(zip(names, row)) for row in table.get("records", [])]


class AlphasAPI:
    def __init__(self, client: BrainClient) -> None:
        self.client = client

    async def get(self, alpha_id: str) -> dict[str, Any]:
        return await self.client.get_json(f"/alphas/{alpha_id}")

    async def patch(
        self,
        alpha_id: str,
        *,
        name: str | None = None,
        color: str | None = None,
        tags: list[str] | None = None,
        category: str | None = None,
        favorite: bool | None = None,
        hidden: bool | None = None,
        regular_description: str | None = None,
        selection_description: str | None = None,
        combo_description: str | None = None,
        osmosis_points: int | None = None,
        clear_osmosis: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        for key, value in (("name", name), ("color", color), ("tags", tags), ("category", category),
                           ("favorite", favorite), ("hidden", hidden)):
            if value is not None:
                body[key] = value
        if regular_description is not None:
            body["regular"] = {"description": regular_description}
        if selection_description is not None:
            body["selection"] = {"description": selection_description}
        if combo_description is not None:
            body["combo"] = {"description": combo_description}
        if osmosis_points is not None:
            body["osmosisPoints"] = int(osmosis_points)
        elif clear_osmosis:
            body["osmosisPoints"] = None
        response = await self.client.request("PATCH", f"/alphas/{alpha_id}", json=body)
        return self.client.raise_for(response)

    async def check(self, alpha_id: str, max_wait_s: float = 900) -> list[dict[str, Any]]:
        """Run BRAIN's submission check (adds SELF/PROD correlation). Returns the checks list."""
        response = await self.client.poll(f"/alphas/{alpha_id}/check", max_wait_s=max_wait_s)
        body = self.client.raise_for(response) or {}
        checks = ((body.get("is") or {}).get("checks")) if isinstance(body, dict) else None
        return list(checks or [])

    async def submit(self, alpha_id: str, max_wait_s: float = 1200) -> SubmitResult:
        url = f"/alphas/{alpha_id}/submit"
        deadline = time.time() + max_wait_s
        response = await self.client.request("POST", url, retry_statuses=(500, 502, 504))
        while True:
            status = response.status_code
            body = self.client._json(response)
            if status == 403:
                checks = (((body or {}).get("is") or {}).get("checks") or []) if isinstance(body, dict) else []
                failing = [c for c in checks if str(c.get("result")).upper() == "FAIL"]
                msg = ", ".join(f"{c.get('name')} ({c.get('value')} vs {c.get('limit')})" for c in failing)
                return SubmitResult("rejected", msg or describe_error_body(body), checks)
            if status == 429:
                return SubmitResult("throttled", describe_error_body(body) or "THROTTLED — try again in a few minutes")
            if status >= 400 and status != 503:
                return SubmitResult("error", describe_error_body(body) or f"HTTP {status}")
            retry = response.headers.get("Retry-After")
            if status in (303, 503) or retry is not None:
                if time.time() > deadline:
                    return SubmitResult("accepted", "Submission still processing on BRAIN; will verify later.")
                await asyncio.sleep(min(30.0, max(1.0, float(retry) if retry else 5.0)))
                response = await self.client.request("GET", url, retry_statuses=(500, 502, 504))
                continue
            break
        # accepted: wait until the alpha turns ACTIVE
        while time.time() < deadline:
            alpha = await self.get(alpha_id)
            if str(alpha.get("status")).upper() == "ACTIVE":
                return SubmitResult("active", "Alpha is ACTIVE")
            await asyncio.sleep(10)
        return SubmitResult("accepted", "Accepted; waiting for ACTIVE status")

    async def correlations(self, alpha_id: str, kind: str = "self", max_wait_s: float = 600) -> dict[str, Any]:
        response = await self.client.poll(f"/alphas/{alpha_id}/correlations/{kind}", max_wait_s=max_wait_s)
        body = self.client.raise_for(response) or {}
        return {"max": body.get("max"), "min": body.get("min"), "records": _records(body), "raw": body}

    async def recordset(self, alpha_id: str, name: str = "pnl", max_wait_s: float = 300) -> list[dict[str, Any]]:
        response = await self.client.poll(f"/alphas/{alpha_id}/recordsets/{name}", max_wait_s=max_wait_s)
        return _records(self.client.raise_for(response))

    async def before_after(self, alpha_id: str) -> dict[str, Any]:
        response = await self.client.poll(f"/users/self/alphas/{alpha_id}/before-and-after-performance")
        return self.client.raise_for(response) or {}

    async def list_page(self, filters: dict[str, Any] | str | None = None, limit: int = 100, offset: int = 0,
                        order: str = "-dateCreated") -> dict[str, Any]:
        """GET /users/self/alphas. String filters are appended raw (e.g. 'is.sharpe>=1.25')."""
        if isinstance(filters, str):
            query = f"limit={limit}&offset={offset}&order={order}"
            if filters:
                query += "&" + filters.replace("+", "%2B")
            return await self.client.get_json(f"/users/self/alphas?{query}")
        params = {"limit": limit, "offset": offset, "order": order, **(filters or {})}
        return await self.client.get_json("/users/self/alphas", params=params)

    async def iterate(self, filters: dict[str, Any] | str | None = None, *, page_size: int = 100,
                      max_items: int | None = None, order: str = "-dateCreated") -> AsyncIterator[dict[str, Any]]:
        """Every alpha matching `filters`, in `order`.

        BRAIN serves only the first 1,000 results of a listing (HTTP 400 beyond). Ordered by dateCreated, the walk
        carries on in windows: each next window filters dateCreated past the last alpha seen, and alphas sharing that
        second are de-duplicated by id. Any other order cannot be windowed and raises at the cap."""
        base = filters.replace("+", "%2B") if isinstance(filters, str) else urlencode(filters or {}, doseq=True)
        windowed = order.lstrip("-") == "dateCreated"
        window = ""
        seen: set[str] = set()
        yielded = 0
        while True:
            offset = 0
            fresh = 0
            edge: str | None = None
            while offset + page_size <= LIST_CAP:
                query = "&".join(p for p in (f"limit={page_size}&offset={offset}&order={order}", base, window) if p)
                page = await self.client.get_json(f"/users/self/alphas?{query}")
                results = page.get("results") or []
                for item in results:
                    edge = item.get("dateCreated") or edge
                    key = str(item.get("id"))
                    if key in seen:
                        continue
                    seen.add(key)
                    fresh += 1
                    yield item
                    yielded += 1
                    if max_items is not None and yielded >= max_items:
                        return
                if not page.get("next") or not results:
                    return
                offset += page_size
            if not windowed:
                raise BrainError(f"BRAIN lists only the first {LIST_CAP:,} alphas when ordered by {order!r}; "
                                 "order by dateCreated to read further.")
            if not fresh or not edge:      # a whole window inside one second, or no dates: cannot move on safely
                log.warning("alpha listing stopped after %d alphas: window did not advance", yielded)
                return
            window = _next_window(edge, descending=order.startswith("-"))

    async def set_tag_list(self, name: str, alpha_ids: list[str]) -> dict[str, Any]:
        response = await self.client.request("POST", "/tags", json={"type": "LIST", "name": name, "alphas": alpha_ids})
        return self.client.raise_for(response)


def parse_checks(alpha: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((alpha.get("is") or {}).get("checks")) or [])


def check_failures(checks: list[dict[str, Any]], include_pending: bool = False) -> list[str]:
    bad = {"FAIL", "ERROR"} | ({"PENDING"} if include_pending else set())
    return [str(c.get("name")) for c in checks if str(c.get("result", "")).upper() in bad]


__all__ = ["AlphasAPI", "SubmitResult", "parse_checks", "check_failures", "BrainError"]
