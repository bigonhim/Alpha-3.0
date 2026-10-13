"""User / consultant / program endpoints (Genius, pyramids, diversity, Osmosis, competitions, messages)."""

from __future__ import annotations

from typing import Any

from .client import BrainClient


class UsersAPI:
    def __init__(self, client: BrainClient) -> None:
        self.client = client

    async def me(self) -> dict[str, Any]:
        return await self.client.get_json("/users/self")

    async def consultant_summary(self) -> dict[str, Any]:
        return await self.client.get_json("/users/self/consultant/summary")

    async def pyramid_multipliers(self) -> dict[str, Any]:
        return await self.client.get_json("/users/self/activities/pyramid-multipliers")

    async def pyramid_alphas(self, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
        params = {k: v for k, v in (("startDate", start_date), ("endDate", end_date)) if v}
        return await self.client.get_json("/users/self/activities/pyramid-alphas", params=params or None)

    async def diversity(self, grouping: str | None = None) -> dict[str, Any]:
        params = {"grouping": grouping} if grouping else None
        return await self.client.get_json("/users/self/activities/diversity", params=params)

    async def activity(self, kind: str) -> dict[str, Any]:
        return await self.client.get_json(f"/users/self/activities/{kind}")

    async def messages(self, kind: str = "NOTIFICATION", limit: int = 20) -> dict[str, Any]:
        return await self.client.get_json(
            "/users/self/messages", params={"type": kind, "order": "-dateCreated", "limit": limit, "offset": 0})

    async def competitions(self) -> Any:
        return await self.client.get_json("/users/self/competitions")

    async def competition(self, competition_id: str) -> dict[str, Any]:
        return await self.client.get_json(f"/competitions/{competition_id}")

    async def leaderboard(self, board: str = "leader", params: dict[str, Any] | None = None) -> Any:
        return await self.client.get_json(f"/consultant/boards/{board}", params=params)

    # --- Osmosis (endpoint shapes partly unverified; callers must handle BrainError gracefully)
    async def osmosis_summary(self, user_id: str = "self") -> dict[str, Any]:
        return await self.client.get_json(f"/users/{user_id}/osmosis/summary")

    async def osmosis_scale_points(self, user_id: str = "self") -> Any:
        return await self.client.get_json(f"/users/{user_id}/osmosis/scale-points/ALL")
