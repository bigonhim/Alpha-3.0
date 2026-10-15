"""Data discovery endpoints: operators, data categories, datasets, data fields, tutorial pages."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Iterable

from .client import BrainClient

PAGE = 50  # BRAIN caps data-sets / data-fields pages at 50
MAX_RESULTS = 10_000  # deepest offset a data-fields query will page to


async def gather_all(coros: Iterable[Awaitable[Any]]) -> list[Any]:
    """Run coroutines concurrently; on the first failure cancel the rest and re-raise that error unchanged."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class DataAPI:
    def __init__(self, client: BrainClient) -> None:
        self.client = client

    async def operators(self) -> list[dict[str, Any]]:
        body = await self.client.get_json("/operators")
        return body if isinstance(body, list) else (body or {}).get("results", [])

    async def categories(self) -> list[dict[str, Any]]:
        body = await self.client.get_json("/data-categories")
        return body if isinstance(body, list) else (body or {}).get("results", [])

    @staticmethod
    def _scope(region: str, delay: int, universe: str, instrument_type: str) -> dict[str, Any]:
        return {"instrumentType": instrument_type, "region": region, "delay": delay, "universe": universe}

    async def datasets(self, region: str, delay: int, universe: str, *, instrument_type: str = "EQUITY",
                       **filters: Any) -> AsyncIterator[dict[str, Any]]:
        params = {**self._scope(region, delay, universe, instrument_type), **filters, "limit": PAGE}
        offset = 0
        while True:
            body = await self.client.get_json("/data-sets", params={**params, "offset": offset})
            results = (body or {}).get("results") or []
            for item in results:
                yield item
            count = (body or {}).get("count") or 0
            offset += PAGE
            if not results or offset >= count:
                return

    async def datafields(self, region: str, delay: int, universe: str, *, dataset_id: str | None = None,
                         field_type: str | None = None, search: str | None = None,
                         instrument_type: str = "EQUITY", max_items: int | None = None) -> AsyncIterator[dict[str, Any]]:
        params: dict[str, Any] = {**self._scope(region, delay, universe, instrument_type), "limit": PAGE}
        if dataset_id:
            params["dataset.id"] = dataset_id
        if field_type:
            params["type"] = field_type
        if search:
            params["search"] = search
        offset = 0
        seen = 0
        while True:
            body = await self.client.get_json("/data-fields", params={**params, "offset": offset})
            results = (body or {}).get("results") or []
            for item in results:
                yield item
                seen += 1
                if max_items is not None and seen >= max_items:
                    return
            count = min((body or {}).get("count") or 0, MAX_RESULTS)
            offset += PAGE
            if not results or offset >= count:
                return

    async def datafields_all(self, region: str, delay: int, universe: str, *, dataset_id: str,
                             instrument_type: str = "EQUITY",
                             limiter: asyncio.Semaphore | None = None) -> list[dict[str, Any]]:
        """Every field of one dataset. The first page gives the count; the remaining pages are then fetched
        concurrently, with `limiter` bounding the requests in flight (shared across datasets by the caller)."""
        params = {**self._scope(region, delay, universe, instrument_type), "limit": PAGE, "dataset.id": dataset_id}
        limiter = limiter or asyncio.Semaphore(1)

        async def page(offset: int) -> tuple[list[dict[str, Any]], int]:
            async with limiter:
                body = await self.client.get_json("/data-fields", params={**params, "offset": offset})
            return (body or {}).get("results") or [], (body or {}).get("count") or 0

        results, count = await page(0)
        out = list(results)
        if results and count > PAGE:
            for rest, _ in await gather_all(page(o) for o in range(PAGE, min(count, MAX_RESULTS), PAGE)):
                out.extend(rest)
        return out

    async def tutorial_page(self, page_id: str) -> dict[str, Any]:
        return await self.client.get_json(f"/tutorial-pages/{page_id}")

    async def tutorials(self) -> Any:
        return await self.client.get_json("/tutorials")
