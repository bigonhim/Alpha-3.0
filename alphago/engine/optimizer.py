"""Near-miss optimiser: map each failing BRAIN check to targeted variants (settings sweeps + AST transforms),
optionally plus an LLM "refine" round that sees only the failing checks (AlphaGen-style)."""

from __future__ import annotations

import logging
from typing import Any

from ..brain import brain
from ..brain.simulations import normalize_settings
from ..db import db
from ..db.models import Alpha, Candidate, OptimizationAttempt
from ..events import bus
from ..fastexpr import negate, scale_windows, validate, wrap
from ..knowledge import pack
from ..llm import llm
from ..llm.prompts import render
from ..llm.schemas import IdeaBatch
from .catalogs import get_catalog
from .evaluator import evaluate
from .generators import settings_from_idea
from .simulator import CandidateSpec, simulator

log = logging.getLogger(__name__)

RISK_NEUTRALIZATIONS = ["STATISTICAL", "CROWDING", "FAST", "SLOW", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM"]


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


def _src(p: Any) -> str:
    return p.to_source() if hasattr(p, "to_source") else str(p)


def plan_variants(a: Alpha, max_variants: int = 10) -> list[tuple[str, dict[str, Any], str]]:
    ev = evaluate(sharpe=a.sharpe, fitness=a.fitness, turnover=a.turnover, checks=a.checks, delay=a.delay,
                  region=a.region)
    s = normalize_settings(a.settings or {})
    region = s["region"]
    expr = a.code or ""
    out: list[tuple[str, dict[str, Any], str]] = []

    def add(e: Any, settings: dict[str, Any], kind: str) -> None:
        out.append((_src(e), normalize_settings(settings), kind))

    if ev.flip_sign:
        expr = _src(negate(expr))
        add(expr, s, "sign flip")
    failing = set(ev.failing)
    turnover = a.turnover or 0.0
    decay = int(s.get("decay") or 0)

    if "HIGH_TURNOVER" in failing or turnover > 0.3:
        nd = retune_decay(decay, turnover)
        if nd != decay:
            add(expr, {**s, "decay": nd}, f"decay {decay}→{nd}")
        add(wrap(expr, "ts_decay_linear", 5), s, "ts_decay_linear 5")
        add(wrap(expr, "hump", hump=0.01), s, "hump 0.01")
    if "LOW_TURNOVER" in failing and decay > 0:
        add(expr, {**s, "decay": max(0, decay // 2)}, f"decay {decay}→{decay // 2}")
    if "CONCENTRATED_WEIGHT" in failing:
        add(expr, {**s, "truncation": 0.05}, "truncation 0.05")
        add(wrap(expr, "rank"), s, "rank wrap")
    if "LOW_SUB_UNIVERSE_SHARPE" in failing:
        add(wrap(expr, "group_neutralize", 'bucket(rank(cap), range="0.1,1,0.1")'), s, "size-bucket neutral")
    if failing & {"LOW_SHARPE", "LOW_FITNESS", "IS_LADDER_SHARPE", "LOW_2Y_SHARPE"} or ev.verdict == "promising":
        allowed = brain.options.neutralizations(region)
        current = s.get("neutralization")
        sweep = [n for n in ["SUBINDUSTRY", "INDUSTRY", "SECTOR", "MARKET"] + RISK_NEUTRALIZATIONS
                 if n in allowed and n != current][:3]
        for n in sweep:
            add(expr, {**s, "neutralization": n}, f"neutralization {current}→{n}")
        add(scale_windows(expr, 0.5), s, "windows ×0.5")
        add(scale_windows(expr, 2.0), s, "windows ×2")
        add(wrap(expr, "group_rank", "subindustry"), s, "group_rank subindustry")
    if failing & {"SELF_CORRELATION", "PROD_CORRELATION"}:
        add(wrap(expr, "group_neutralize", "sector"), s, "sector neutral (decorrelate)")
    # de-duplicate
    seen, uniq = set(), []
    for e, st, kind in out:
        key = (e, tuple(sorted(st.items())))
        if key not in seen and e:
            seen.add(key)
            uniq.append((e, st, kind))
    return uniq[:max_variants]


async def llm_refine(a: Alpha, k: int = 4) -> list[CandidateSpec]:
    failing =[f"- {c.get('name')}: {c.get('result')} (value {c.get('value')}, limit {c.get('limit')})"
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
                          yearly=yearly_txt, dataset_context=ds_ctx)
    batch = await llm.generate_json(prompt, IdeaBatch, system=await pack.static_prefix(), role="refine",
                                       purpose="refine")
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
    """Queue optimisation variants of an alpha; returns the candidate ids (created + reused)."""
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
    if a is None or not a.code:
        raise ValueError("Alpha not found or has no expression")
    specs = [CandidateSpec(expression=e, settings=st, origin="optimize", parent_alpha_id=a.id,
                           dataset_id=(a.dataset_ids or [None])[0], hypothesis=f"Optimisation of {a.id}: {kind}",
                           rationale={"kind": kind})
             for e, st, kind in plan_variants(a, max_variants)]
    if use_llm and llm.configure():
        try:
            specs += await llm_refine(a, k=3)
        except Exception as exc:
            log.warning("LLM refine failed for %s: %s", alpha_id, exc)
    for spec in specs:
        spec.task_id = task_id
    result = await simulator.submit(specs, batch_id=batch_id)
    async with db.write() as s:
        for c in result.created + result.reused:
            rationale = c.rationale or {}
            kind = rationale.get("kind") or ("llm refine" if rationale.get("refine") else "variant")
            s.add(OptimizationAttempt(parent_alpha_id=a.id, candidate_id=c.id, kind=str(kind)[:48],
                                      detail=c.expression[:500]))
    bus.publish("optimize", f"Optimising {alpha_id}: {len(result.created)} variants queued "
                            f"({len(result.reused)} reused).")
    return [c.id for c in result.created + result.reused]


async def count_attempts() -> int:
    from sqlalchemy import func, select
    async with db.session() as s:
        return await s.scalar(select(func.count(OptimizationAttempt.id))) or 0


__all__ = ["plan_variants", "optimize_alpha", "llm_refine", "retune_decay", "count_attempts", "Candidate"]
