"""Data discovery endpoints: operators, data categories, datasets, data fields, tutorial pages."""

from __future__ import annotations

from typing import Any, AsyncIterator

from .client import BrainClient

PAGE = 50  # BRAIN caps data-sets / data-fields pages at 50


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
            count = min((body or {}).get("count") or 0, 10_000)
            offset += PAGE
            if not results or offset >= count:
                return

    async def tutorial_page(self, page_id: str) -> dict[str, Any]:
        return await self.client.get_json(f"/tutorial-pages/{page_id}")

    async def tutorials(self) -> Any:
        return await self.client.get_json("/tutorials")
