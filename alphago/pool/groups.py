"""Tag groups: a set of tags kept under a name, to filter the pool by and analyse as one portfolio."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import delete, select

from ..db import db
from ..db.models_pool import TagGroup
from .brainops import parse_tags


async def all_groups() -> list[TagGroup]:
    async with db.session() as s:
        return list(await s.scalars(select(TagGroup).order_by(TagGroup.name)))


async def save_group(name: str, tags: Sequence[str], every: bool) -> TagGroup:
    """Keep these tags under `name`, replacing a group of the same name."""
    label = name.strip()
    if not label:
        raise ValueError("Give the group a name.")
    if len(label) > 60:
        raise ValueError("Keep the name to 60 characters.")
    kept = parse_tags(list(tags))
    async with db.write() as s:
        row = await s.scalar(select(TagGroup).where(TagGroup.name == label))
        if row is None:
            row = TagGroup(name=label, tags=kept, every=bool(every))
            s.add(row)
        else:
            row.tags, row.every = kept, bool(every)
        await s.flush()
        return row


async def forget_group(group_id: int) -> None:
    async with db.write() as s:
        await s.execute(delete(TagGroup).where(TagGroup.id == group_id))
