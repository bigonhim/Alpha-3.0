"""What the tools share: reading the one Alpha a tool works on, the held settings, and the local data catalogue.

A tool that works on one Alpha takes its ID. It is read from the local Alpha table when that holds its expression and
settings, and from BRAIN (`GET /alphas/{id}`, then stored) otherwise, so a tool works on an Alpha this computer has
never seen. An expression pasted where an ID belongs is refused in plain words: sent to BRAIN it would come back as
"not found", which tells the consultant nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select

from ..brain.platform import GROUPING_FIELDS
from ..db import db
from ..db.models import Alpha, DataField

ALPHA_ID = re.compile(r"^[A-Za-z0-9]{1,64}$")
_LINK = re.compile(r"/alphas?/([A-Za-z0-9]{1,64})(?:[/?#]|$)")
NOT_AN_ID = ("That looks like an expression, not an Alpha ID. Paste the Alpha's ID or its BRAIN link here; to try an "
             "expression at other settings, use the Settings Sampler.")


class ToolError(ValueError):
    """A plain refusal a tool shows as it is (nothing was simulated)."""


def alpha_id_of(text: str | None) -> str:
    """The Alpha ID typed (or the ID inside a pasted BRAIN link), or ToolError saying what went wrong."""
    raw = (text or "").strip()
    if not raw:
        raise ToolError("Enter an Alpha ID.")
    link = _LINK.search(raw)
    if link and raw.lower().startswith(("http", "platform.", "www.")):
        raw = link.group(1)
    if not ALPHA_ID.fullmatch(raw):
        raise ToolError(NOT_AN_ID)
    return raw


def looks_like_id(text: str | None) -> bool:
    try:
        alpha_id_of(text)
        return True
    except ToolError:
        return False


def body_of(a: Alpha) -> dict[str, Any]:
    """A stored Alpha in the shape BRAIN's `GET /alphas/{id}` answers (the parts the tools read)."""
    raw = dict(a.raw or {})
    return {
        **raw,
        "id": a.id,
        "type": a.type,
        "status": a.status,
        "regular": {"code": a.code, "description": a.description},
        "settings": dict(a.settings or {}),
        "classifications": a.classifications or raw.get("classifications") or [],
        "is": {"checks": list(a.checks or a.is_checks or []), "sharpe": a.sharpe, "fitness": a.fitness},
    }


async def alpha_body(alpha_id: str, *, refresh: bool = False) -> dict[str, Any]:
    """The Alpha as BRAIN describes it: from the local table when it is complete there, else read from BRAIN."""
    alpha_id = alpha_id_of(alpha_id)
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
    if a is not None and a.code and a.settings and not refresh:
        return body_of(a)
    from ..brain import brain
    from ..brain.errors import BrainError
    try:
        body = await brain.alphas.get(alpha_id)
    except BrainError as exc:
        if a is not None and a.code:
            return body_of(a)
        if getattr(exc, "status", None) == 404 or "not found" in str(exc).lower():
            raise ToolError(f"BRAIN has no Alpha {alpha_id} on this account.") from exc
        raise ToolError(f"Could not read {alpha_id} from BRAIN: {getattr(exc, 'message', None) or exc}") from exc
    except RuntimeError as exc:                     # not signed in
        if a is not None and a.code:
            return body_of(a)
        raise ToolError(f"{alpha_id} is not stored on this computer, and BRAIN could not be asked: {exc}") from exc
    if not isinstance(body, dict) or not body.get("id"):
        raise ToolError(f"BRAIN has no Alpha {alpha_id} on this account.")
    from ..engine.ingest import upsert_alpha
    try:
        await upsert_alpha(body, origin="synced")
    except Exception:          # storing is a convenience; the tool can go on with the body it read
        pass
    return body


def code_of(body: dict[str, Any]) -> str | None:
    part = body.get("regular") or body.get("combo") or body.get("selection")
    code = part.get("code") if isinstance(part, dict) else part
    return code if isinstance(code, str) and code.strip() else None


def held_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """The settings a variant of the Alpha runs at: its own, shown and never edited."""
    return {k: settings.get(k) for k in ("region", "delay", "universe", "neutralization", "decay", "truncation")}


def settings_line(settings: dict[str, Any]) -> str:
    s = held_settings(settings)
    return (f"{s['region'] or '—'} D{s['delay'] if s['delay'] is not None else '—'} {s['universe'] or '—'} · "
            f"{s['neutralization'] or '—'} · decay {s['decay'] if s['decay'] is not None else '—'} · "
            f"truncation {s['truncation'] if s['truncation'] is not None else '—'}")


def fields_in_order(expression: str, *, grouping: bool = False) -> list[str]:
    """Data fields the expression reads, in the order it first reads them (grouping fields only if asked)."""
    from ..fastexpr import Ident, parse, walk
    from ..labs.fastexpr import data_fields
    program = parse(expression)
    wanted = data_fields(program, grouping=grouping)
    out: list[str] = []
    for node in walk(program):
        if isinstance(node, Ident) and node.name in wanted and node.name not in out:
            out.append(node.name)
    return out + sorted(wanted - set(out))


def grouping_in(fields: Iterable[str]) -> list[str]:
    return [f for f in fields if f in GROUPING_FIELDS]


# ---------------------------------------------------------------- the local data catalogue

async def field_availability(field_ids: Iterable[str]) -> dict[str, dict[tuple[str, int, str], float]]:
    """Each field's markets in the local catalogue, (region, delay, universe) -> coverage (0 when not reported)."""
    ids = sorted(set(field_ids))
    out: dict[str, dict[tuple[str, int, str], float]] = {f: {} for f in ids}
    if not ids:
        return out
    async with db.session() as s:
        rows = (await s.execute(select(DataField.id, DataField.region, DataField.delay, DataField.universe,
                                       DataField.coverage).where(DataField.id.in_(ids)))).all()
    for fid, region, delay, universe, coverage in rows:
        out[fid][(str(region), int(delay), str(universe))] = float(coverage or 0.0)
    return out


async def synced_markets() -> set[tuple[str, int, str]]:
    async with db.session() as s:
        rows = (await s.execute(select(DataField.region, DataField.delay, DataField.universe).distinct())).all()
    return {(str(r), int(d), str(u)) for r, d, u in rows}


async def fields_at(region: str, delay: int, universe: str, field_ids: Iterable[str] | None = None) -> list[DataField]:
    stmt = select(DataField).where(DataField.region == region, DataField.delay == int(delay),
                                   DataField.universe == universe)
    if field_ids is not None:
        ids = sorted(set(field_ids))
        if not ids:
            return []
        stmt = stmt.where(DataField.id.in_(ids))
    async with db.session() as s:
        return list(await s.scalars(stmt))


async def described(field_ids: Iterable[str], region: str | None) -> dict[str, DataField]:
    """Each field's catalogue row, read where `region` holds it, else wherever the catalogue does."""
    ids = sorted(set(field_ids))
    if not ids:
        return {}
    async with db.session() as s:
        rows = list(await s.scalars(select(DataField).where(DataField.id.in_(ids))))
    rows.sort(key=lambda r: (r.region != region, r.region, r.delay, r.universe))
    out: dict[str, DataField] = {}
    for r in rows:
        out.setdefault(r.id, r)
    return out
