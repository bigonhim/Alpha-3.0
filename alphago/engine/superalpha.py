"""SuperAlpha workflow (per the Super Alpha guide): group ACTIVE alphas under a code name, select them with a selection
expression (e.g. `name == '321'`), weight them with a combo expression (`1` = equal weight), simulate, describe."""

from __future__ import annotations

import logging
from typing import Any

from ..brain import brain
from ..brain.errors import SimulationRejected
from ..brain.simulations import super_payload
from ..db import db
from ..events import bus
from ..llm import gemini
from ..llm.prompts import render
from ..llm.schemas import SuperAlphaDescriptions
from . import actions
from .ingest import upsert_alpha

log = logging.getLogger(__name__)


async def rename_alphas(alpha_ids: list[str], code: str) -> int:
    ok = 0
    for aid in alpha_ids:
        await actions.set_properties(aid, name=code)
        ok += 1
    bus.publish("sync", f"Renamed {ok} alphas to '{code}' for SuperAlpha selection.")
    return ok


async def tag_alphas(alpha_ids: list[str], tag: str) -> int:
    from ..db.models import Alpha
    ok = 0
    for aid in alpha_ids:
        async with db.session() as s:
            a = await s.get(Alpha, aid)
        tags = list(dict.fromkeys([*(a.tags or []), tag])) if a else [tag]
        await actions.set_properties(aid, tags=tags)
        ok += 1
    return ok


async def preview(selection: str, settings: dict[str, Any], limit: int, handling: str) -> dict[str, Any]:
    return await brain.sims.super_selection_preview(selection, settings, limit, handling)


async def simulate_super(selection: str, combo: str, settings: dict[str, Any], limit: int = 100,
                         handling: str = "POSITIVE") -> str:
    """Simulate a SUPER alpha; tries the documented payload shape first, then the alternative one."""
    variant = await db.kv_get("super_payload_variant", "top")
    order = [variant, "nested" if variant == "top" else "top"]
    last_exc: Exception | None = None
    for shape in order:
        payload = super_payload(selection, combo, settings, selection_limit=limit, selection_handling=handling,
                                nested=shape == "nested")
        try:
            outcomes = await brain.sims.run([payload])
        except SimulationRejected as exc:
            last_exc = exc
            log.info("SUPER payload shape %s rejected: %s", shape, exc.message)
            continue
        out = outcomes[0]
        if not out.alpha_id:
            raise RuntimeError(out.message or f"SuperAlpha simulation {out.status}")
        await db.kv_set("super_payload_variant", shape)
        data = await brain.alphas.get(out.alpha_id)
        await upsert_alpha(data, origin="app")
        bus.publish("simulate", f"SuperAlpha simulated: {out.alpha_id} (selection {selection!r}, combo {combo!r}).",
                    "success")
        return out.alpha_id
    raise last_exc or RuntimeError("SuperAlpha simulation failed")


async def write_descriptions(alpha_id: str, selection: str, combo: str, components: str) -> SuperAlphaDescriptions:
    prompt = await render("super_describe", selection=selection, combo=combo, components=components)
    desc = await gemini.generate_json(prompt, SuperAlphaDescriptions, role="describe", purpose="super_describe",
                                      thinking="low")
    await brain.alphas.patch(alpha_id, selection_description=desc.selection_description,
                             combo_description=desc.combo_description)
    return desc
