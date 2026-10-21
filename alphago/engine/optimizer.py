"""Near-miss optimiser entry points: one-round repairs from the alpha page, the AI refine call and the decay rule.

The fix library, the learned fix order and the multi-round repair loop live in engine/refine.py."""

from __future__ import annotations

import logging
from typing import Any

from ..brain.simulations import normalize_settings
from ..db import db
from ..db.models import Alpha, Candidate, OptimizationAttempt
from ..events import bus
from ..fastexpr import validate
from ..knowledge import pack
from ..llm import llm
from ..llm.prompts import render
from ..llm.schemas import IdeaBatch
from .catalogs import get_catalog
from .generators import settings_from_idea
from .simulator import CandidateSpec

log = logging.getLogger(__name__)

def retune_decay(decay: int, turnover: float | None) -> int:
    """machine_lib rule of thumb for turnover → decay."""
    t = turnover or 0
    if t > 0.7:
        return max(4, decay * 4)
    if t > 0.6:
        return decay * 3 + 3
    if t > 0.5:
        return max(3, decay * 3)
    if t > 0.4:
        return max(2, decay * 2)
    if t > 0.35:
        return decay + 4
    if t > 0.3:
        return decay + 2
    return decay


def plan_variants(a: Alpha, max_variants: int = 10) -> list[tuple[str, dict[str, Any], str]]:
    """Targeted variants (expression, settings, kind) for a near miss, in the fix library's default order.

    Kept for callers that want a plan without the database or the AI; the research engine uses the learned order in
    engine/refine.py."""
    from .refine import plan_static
    return [(v.expr, v.settings, v.detail) for v in plan_static(a, max_variants)]


async def llm_refine(a: Alpha, k: int = 4, *, tried: list[str] | None = None,
                     collide: str | None = None) -> list[CandidateSpec]:
    """k AI variants of `a` aimed at its failing checks. `tried` lists fixes already tried (and their results) so the
    model does not repeat them; `collide` is the submitted alpha it is too correlated with."""
    failing = [f"- {c.get('name')}: {c.get('result')} (value {c.get('value')}, limit {c.get('limit')})"
               for c in (a.checks or []) if str(c.get("result", "")).upper() in ("FAIL", "WARNING", "ERROR")]
    yearly = await db.kv_get(f"yearly:{a.id}") or []
    yearly_txt = "\n".join(str(r) for r in yearly[-6:])
    s = normalize_settings(a.settings or {})
    ds_ctx = ""
    if a.dataset_ids:
        ds_id = str(a.dataset_ids[0]).split(",")[0]
        ds_ctx = await pack.dataset_context(ds_id, s["region"], s["delay"], s["universe"], max_fields=60)
    prompt = await render("refine", k=k, expression=a.code, settings=pack.settings_text(s), sharpe=a.sharpe,
                          fitness=a.fitness, turnover=a.turnover, returns=a.returns, drawdown=a.drawdown,
                          margin=a.margin, failing="\n".join(failing) or "- none (improve robustness)",
                          tried="\n".join(tried or []), collide=collide or "", yearly=yearly_txt,
                          dataset_context=ds_ctx)
    batch = await llm.generate_json(prompt, IdeaBatch, system=await pack.static_prefix(s["region"], s["delay"]),
                                    role="refine", purpose="refine")
    catalog = await get_catalog(s["region"], s["delay"], s["universe"])
    specs = []
    for idea in batch.ideas[:k * 2]:
        if validate(idea.expression, catalog).ok:
            specs.append(CandidateSpec(expression=idea.expression, settings=settings_from_idea(idea, s, None),
                                       origin="optimize", parent_alpha_id=a.id, hypothesis=idea.idea,
                                       rationale={"refine": idea.operator_rationale}))
    return specs[:k]


async def optimize_alpha(alpha_id: str, *, rounds_hint: int = 1, batch_id: int | None = None, use_llm: bool = True,
                         max_variants: int = 8, task_id: int | None = None) -> list[int]:
    """Queue one round of targeted repairs of an alpha (no waiting); returns the candidate ids (created + reused).
    The automatic research uses the multi-round loop in engine/refine.py instead."""
    from .refine import RefineContext, expand_once
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
    if a is None or not a.code:
        raise ValueError("Alpha not found or has no expression")
    ctx = RefineContext(task_id=task_id, batch_id=batch_id, use_llm=use_llm and llm.configure(),
                        root_fanout=max_variants + (3 if use_llm else 0))
    ids = await expand_once(a, ctx)
    bus.publish("optimize", f"Optimising {alpha_id}: {len(ids)} targeted variants queued.")
    return ids


async def count_attempts() -> int:
    from sqlalchemy import func, select
    async with db.session() as s:
        return await s.scalar(select(func.count(OptimizationAttempt.id))) or 0


__all__ = ["plan_variants", "optimize_alpha", "llm_refine", "retune_decay", "count_attempts", "Candidate"]
