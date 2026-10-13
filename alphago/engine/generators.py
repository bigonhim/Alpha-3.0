"""Candidate generators: Gemini ideation (with a validator repair round), template factory, AST mutation/crossover."""

from __future__ import annotations

import logging
import random
from typing import Any

from sqlalchemy import select

from ..brain import brain
from ..brain.simulations import normalize_settings
from ..config import settings
from ..db import db
from ..db.models import Alpha, DataField, Dataset
from ..fastexpr import inline, parse, transforms, validate
from ..knowledge import pack
from ..knowledge.templates import factory, settings_for_category, third_order
from ..llm import LLMError, gemini
from ..llm.prompts import render
from ..llm.schemas import AlphaIdea, IdeaBatch
from .catalogs import get_catalog
from .simulator import CandidateSpec

log = logging.getLogger(__name__)

THEMES = ["momentum", "short-term reversal", "value", "quality / profitability", "growth", "analyst revisions",
          "earnings surprise", "sentiment", "volatility / risk", "liquidity", "seasonality", "event-driven"]
MOVES = {
    "EXPLORE": "new mechanism or under-used fields; maximise novelty",
    "EXPLOIT": "refine around the best working ideas for this dataset",
    "RECOMBINE": "blend two weakly-related signals (e.g. rank(a) + rank(b)) or condition one on another",
    "SIMPLIFY": "the simplest possible alphas: ≤ 4 operators, 1–2 fields (Power Pool / Genius friendly)",
}


async def dataset_category(dataset_id: str | None, scope: dict[str, Any]) -> str | None:
    if not dataset_id:
        return None
    async with db.session() as s:
        ds = await s.scalar(select(Dataset).where(Dataset.id == dataset_id, Dataset.region == scope["region"],
                                                  Dataset.delay == scope["delay"],
                                                  Dataset.universe == scope["universe"]))
    return ds.category_name if ds else None


def settings_from_idea(idea: AlphaIdea, base: dict[str, Any], category: str | None) -> dict[str, Any]:
    decay, neut, trunc, nan = settings_for_category(category)
    s = dict(base)
    region = s.get("region", "USA")
    allowed = brain.options.neutralizations(region)
    neut_choice = (idea.neutralization or "").upper().strip()
    s["neutralization"] = neut_choice if neut_choice in allowed else (neut if neut in allowed else allowed[0])
    s["decay"] = int(idea.decay) if 0 <= int(idea.decay or 0) <= 40 else decay
    s["truncation"] = float(idea.truncation) if 0.005 <= float(idea.truncation or 0) <= 0.2 else trunc
    s.setdefault("nanHandling", nan)
    return normalize_settings(s)


async def llm_ideas(scope: dict[str, Any], *, dataset_id: str | None, n: int, goal: str, move: str = "EXPLORE",
                    themes: list[str] | None = None, base_settings: dict[str, Any] | None = None,
                    origin: str = "llm") -> list[CandidateSpec]:
    region, delay, universe = scope["region"], int(scope["delay"]), scope["universe"]
    base = normalize_settings({**(base_settings or {}), "region": region, "delay": delay, "universe": universe})
    system = await pack.static_prefix()
    ds_ctx = await pack.dataset_context(dataset_id, region, delay, universe) if dataset_id else \
        "No specific dataset: use price/volume (pv1) fields and any fields you are confident exist."
    examples = await pack.best_examples(region)
    avoid = await pack.avoid_list(region, dataset_id)
    # Several small calls with fresh themes beat one big call: less repetition, no truncated JSON.
    per_call = max(1, settings.llm.ideas_per_call)
    ideas: list[AlphaIdea] = []
    seen: set[str] = set()
    remaining = max(n, 1)
    while remaining > 0:
        k = min(per_call, remaining)
        call_themes = themes if themes and not ideas else random.sample(THEMES, 4)
        prompt = await render("ideation", n=k, goal=goal, move=move, move_hint=MOVES.get(move, ""),
                              themes=", ".join(call_themes), settings=pack.settings_text(base),
                              dataset_context=ds_ctx, examples="\n".join(examples), avoid="\n".join(avoid))
        try:
            batch = await gemini.generate_json(prompt, IdeaBatch, system=system, role="ideation", purpose="ideation")
        except LLMError:
            if not ideas:
                raise
            break  # keep what earlier chunks produced (budget hit / persistent throttling)
        for idea in batch.ideas[: k * 2]:
            key = "".join(idea.expression.split())
            if key not in seen:
                seen.add(key)
                ideas.append(idea)
        remaining -= k
    category = await dataset_category(dataset_id, scope)
    catalog = await get_catalog(region, delay, universe)

    good: list[AlphaIdea] = []
    bad: list[tuple[AlphaIdea, str]] = []
    for idea in ideas:
        r = validate(idea.expression, catalog)
        (good.append(idea) if r.ok else bad.append((idea, "; ".join(i.message for i in r.errors))))
    if bad:  # one repair round, fed with the validator's messages
        errors = "\n".join(f"- {i.expression}\n  errors: {msg}" for i, msg in bad[:12])
        try:
            fixed = await gemini.generate_json(await render("repair", errors=errors), IdeaBatch, system=system,
                                               role="ideation", purpose="repair", thinking="low")
            for idea in fixed.ideas:
                if validate(idea.expression, catalog).ok:
                    good.append(idea)
        except Exception as exc:
            log.warning("repair round failed: %s", exc)
    specs = []
    for idea in good[:n]:
        specs.append(CandidateSpec(
            expression=idea.expression, settings=settings_from_idea(idea, base, category), origin=origin,
            hypothesis=idea.idea, dataset_id=dataset_id,
            rationale={"mechanism": idea.mechanism, "data": idea.data_rationale, "operators": idea.operator_rationale,
                       "move": move}))
    return specs


async def dataset_fields(scope: dict[str, Any], dataset_id: str, min_coverage: float = 0.3,
                         limit: int = 60) -> list[tuple[str, str]]:
    async with db.session() as s:
        rows = list(await s.scalars(select(DataField).where(
            DataField.dataset_id == dataset_id, DataField.region == scope["region"],
            DataField.delay == int(scope["delay"]), DataField.universe == scope["universe"],
            DataField.type.in_(["MATRIX", "VECTOR"]))))
    rows = [r for r in rows if (r.coverage or 0) >= min_coverage] or rows
    rows.sort(key=lambda r: ((r.alpha_count or 0), -(r.coverage or 0)))
    return [(r.id, r.type) for r in rows[:limit]]


async def template_ideas(scope: dict[str, Any], *, dataset_id: str | None, n: int,
                         base_settings: dict[str, Any] | None = None, rng: random.Random | None = None,
                         families: list[str] | None = None) -> list[CandidateSpec]:
    rng = rng or random.Random()
    fields = await dataset_fields(scope, dataset_id) if dataset_id else [
        ("close", "MATRIX"), ("volume", "MATRIX"), ("vwap", "MATRIX"), ("returns", "MATRIX"), ("cap", "MATRIX")]
    category = await dataset_category(dataset_id, scope)
    decay, neut, trunc, nan = settings_for_category(category)
    allowed = brain.options.neutralizations(scope["region"])
    base = normalize_settings({**(base_settings or {}), "region": scope["region"], "delay": int(scope["delay"]),
                               "universe": scope["universe"], "decay": decay,
                               "neutralization": neut if neut in allowed else allowed[0], "truncation": trunc,
                               "nanHandling": nan})
    catalog = await get_catalog(scope["region"], int(scope["delay"]), scope["universe"])
    specs = []
    for expr, family in factory(fields, n=n * 2, rng=rng, families=families):
        if validate(expr, catalog).ok:
            specs.append(CandidateSpec(expression=expr, settings=base, origin="template", dataset_id=dataset_id,
                                       hypothesis=f"Template family '{family}'", rationale={"family": family}))
        if len(specs) >= n:
            break
    return specs


def _mutations(expr: str, rng: random.Random) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    try:
        out.append((transforms.scale_windows(expr, 0.5).to_source(), "windows ×0.5"))
        out.append((transforms.scale_windows(expr, 2.0).to_source(), "windows ×2"))
        group = rng.choice(["subindustry", "industry", "sector"])
        out.append((transforms.wrap(expr, "group_neutralize", group).to_source(), f"group_neutralize {group}"))
        out.append((transforms.wrap(expr, "rank").to_source(), "rank"))
        out.append((transforms.wrap(expr, "ts_decay_linear", rng.choice([3, 5, 10])).to_source(), "ts_decay_linear"))
        for alt in third_order(expr, rng, k=1):
            out.append((alt, "trade_when gate"))
    except Exception as exc:
        log.debug("mutation failed for %s: %s", expr, exc)
    return out


async def mutation_ideas(scope: dict[str, Any], *, n: int, dataset_id: str | None = None,
                         rng: random.Random | None = None) -> list[CandidateSpec]:
    """Mutate / recombine the best recent alphas in scope (warm-start GP style)."""
    rng = rng or random.Random()
    async with db.session() as s:
        stmt = select(Alpha).where(Alpha.region == scope["region"], Alpha.delay == int(scope["delay"]),
                                   Alpha.verdict.in_(["promising", "strong", "ready", "submitted"]),
                                   Alpha.code.is_not(None), Alpha.type == "REGULAR")
        parents = list(await s.scalars(stmt.order_by(Alpha.score.desc().nulls_last()).limit(30)))
    if dataset_id:
        scoped = [p for p in parents if dataset_id in (p.dataset_ids or [])]
        parents = scoped or parents
    if not parents:
        return []
    catalog = await get_catalog(scope["region"], int(scope["delay"]), scope["universe"])
    specs: list[CandidateSpec] = []
    rng.shuffle(parents)
    for p in parents:
        for expr, kind in _mutations(p.code, rng):
            if validate(expr, catalog).ok:
                specs.append(CandidateSpec(expression=expr, settings=normalize_settings(p.settings or {}),
                                           origin="mutation", parent_alpha_id=p.id, dataset_id=dataset_id,
                                           hypothesis=f"Mutation of {p.id}: {kind}", rationale={"mutation": kind}))
        if len(specs) >= n * 2:
            break
    # recombination of two parents with different skeletons
    if len(parents) >= 2:
        a, b = rng.sample(parents, 2)
        if a.skeleton_hash != b.skeleton_hash:
            try:
                expr = (f"0.5 * rank({inline(parse(a.code)).to_source()}) + "
                        f"0.5 * rank({inline(parse(b.code)).to_source()})")
            except Exception:
                expr = None
            if expr and validate(expr, catalog).ok:
                specs.append(CandidateSpec(expression=expr, settings=normalize_settings(a.settings or {}),
                                           origin="mutation", parent_alpha_id=a.id,
                                           hypothesis=f"Recombination of {a.id} and {b.id}",
                                           rationale={"mutation": "recombine"}))
    rng.shuffle(specs)
    return specs[:n]
