"""Dataset intelligence: opportunity scoring, Thompson-sampling dataset choice, LLM dataset briefs, field probes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..brain.simulations import normalize_settings
from ..db import db
from ..db.models import (Alpha, Candidate, DataField, Dataset, DatasetInsight, DatasetRating, OptimizationAttempt,
                          Pyramid, utcnow)
from ..events import bus
from ..jobs.queue import JobContext, jobs
from ..knowledge import pack
from ..llm import llm
from ..llm.prompts import render
from ..llm.schemas import DatasetBrief
from .simulator import CandidateSpec, simulator
from .sync import current_quarter


@dataclass
class DatasetScore:
    id: str
    name: str
    category: str | None
    category_id: str | None
    coverage: float | None
    value_score: float | None
    alpha_count: int | None
    user_count: int | None
    field_count: int | None
    pyramid_multiplier: float | None
    tried: int
    good: float           # survival-weighted successes (see survival_weight)
    analyzed: bool
    pyramid_alphas: int
    opportunity: float
    passed: int = 0       # first-pass ideas whose lineage passed BRAIN's check (ready or submitted)

    def as_row(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _norm(values: list[float], v: float | None) -> float:
    vals = [x for x in values if x is not None]
    if v is None or not vals:
        return 0.5
    lo, hi = min(vals), max(vals)
    return 0.5 if hi == lo else (v - lo) / (hi - lo)


def _split_counts(pairs) -> dict[str, int]:  # noqa: ANN001
    """Candidate.dataset_id may hold "A,B" for two-dataset alphas: credit each dataset."""
    out: dict[str, int] = {}
    for key, n in pairs:
        for ds in str(key or "").split(","):
            if ds:
                out[ds] = out.get(ds, 0) + n
    return out


# repair, robustness and probe simulations are not new ideas: they neither count as tries nor as successes of their own
NOT_FIRST_PASS = ("optimize", "robustness", "probe")


def survival_weight(verdict: str | None, robust_state: str | None = None) -> float:
    """How far an idea got: promising 0.1 · strong 0.3 · passed BRAIN's check but fragile 0.6 · passed (robust or
    untested) 1.0 · submitted 1.0. Favours datasets that produce submittable alphas, not just good-looking ones."""
    if verdict == "submitted":
        return 1.0
    if verdict == "ready":
        return 0.6 if robust_state == "fragile" else 1.0
    return {"strong": 0.3, "promising": 0.1}.get(verdict or "", 0.0)


async def survival_by_dataset(region: str, delay: int) -> tuple[dict[str, int], dict[str, float], dict[str, int]]:
    """(tries, weighted successes, ideas that passed BRAIN's check) per dataset, from first-pass ideas only. An idea
    is credited with the best result of its repair lineage (a near miss the repair loop made submittable counts)."""
    async with db.session() as s:
        rows = (await s.execute(select(Candidate.dataset_id, Candidate.alpha_id, Alpha.verdict, Alpha.robust_state)
                                .outerjoin(Alpha, Alpha.id == Candidate.alpha_id)
                                .where(Candidate.region == region, Candidate.delay == delay,
                                       Candidate.status == "done", Candidate.origin.notin_(NOT_FIRST_PASS)))).all()
        roots = [aid for _, aid, _, _ in rows if aid]
        lineage: dict[str, float] = {}
        for i in range(0, len(roots), 400):
            for root, verdict, robust in (await s.execute(
                    select(OptimizationAttempt.root_alpha_id, Alpha.verdict, Alpha.robust_state)
                    .join(Alpha, Alpha.id == OptimizationAttempt.child_alpha_id)
                    .where(OptimizationAttempt.root_alpha_id.in_(roots[i:i + 400])))).all():
                lineage[root] = max(lineage.get(root, 0.0), survival_weight(verdict, robust))
    tried: dict[str, int] = {}
    good: dict[str, float] = {}
    passed: dict[str, int] = {}
    for ds_key, aid, verdict, robust in rows:
        w = max(survival_weight(verdict, robust), lineage.get(aid or "", 0.0))
        for ds in str(ds_key or "").split(","):
            if ds:
                tried[ds] = tried.get(ds, 0) + 1
                good[ds] = good.get(ds, 0.0) + w
                passed[ds] = passed.get(ds, 0) + (w >= 0.6)
    return tried, good, passed


async def scored_datasets(region: str, delay: int, universe: str) -> list[DatasetScore]:
    tried, good, passed = await survival_by_dataset(region, delay)
    async with db.session() as s:
        rows = list(await s.scalars(select(Dataset).where(Dataset.region == region, Dataset.delay == delay,
                                                          Dataset.universe == universe)))
        analyzed = set(await s.scalars(select(DatasetInsight.dataset_id).where(
            DatasetInsight.region == region, DatasetInsight.delay == delay, DatasetInsight.universe == universe)))
        pyr = {(p.category_id, p.region, p.delay): p.alpha_count for p in await s.scalars(
            select(Pyramid).where(Pyramid.quarter == current_quarter()))}
    mult = [r.pyramid_multiplier for r in rows]
    vs = [r.value_score for r in rows]
    ac = [math.log1p(r.alpha_count or 0) for r in rows]
    uc = [math.log1p(r.user_count or 0) for r in rows]
    out = []
    for r in rows:
        n, g = tried.get(r.id, 0), round(good.get(r.id, 0.0), 2)
        yield_rate = (g + 1) / (n + 4)  # Bayesian-smoothed survival-weighted success
        opp = (0.30 * _norm(mult, r.pyramid_multiplier) + 0.20 * _norm(vs, r.value_score)
               + 0.20 * (1 - _norm(ac, math.log1p(r.alpha_count or 0)))
               + 0.10 * (1 - _norm(uc, math.log1p(r.user_count or 0)))
               + 0.10 * min(1.0, r.coverage or 0.5) + 0.10 * min(1.0, yield_rate * 4))
        if (r.field_count or 0) == 0:
            opp *= 0.5
        out.append(DatasetScore(r.id, r.name or r.id, r.category_name, r.category_id, r.coverage, r.value_score,
                                r.alpha_count, r.user_count, r.field_count, r.pyramid_multiplier, n, g,
                                r.id in analyzed, pyr.get((r.category_id, region, delay), 0), round(opp * 100, 1),
                                passed.get(r.id, 0)))
    out.sort(key=lambda d: d.opportunity, reverse=True)
    return out


async def choose_dataset(region: str, delay: int, universe: str, strategy: str = "opportunity",
                         pinned: list[str] | None = None, rng: random.Random | None = None) -> DatasetScore | None:
    """Thompson sampling over datasets: prior = opportunity score, likelihood = observed yield."""
    rng = rng or random.Random()
    scored = await scored_datasets(region, delay, universe)
    scored = [d for d in scored if (d.field_count or 0) > 0] or scored
    if not scored:
        return None
    if strategy == "pinned" and pinned:
        pool = [d for d in scored if d.id in pinned]
        return rng.choice(pool) if pool else None
    if strategy == "pyramid_gap":
        gap = [d for d in scored if d.pyramid_alphas < 3]
        scored = gap or scored
    if strategy == "under_used":
        scored = sorted(scored, key=lambda d: (d.alpha_count or 0))[:max(5, len(scored) // 3)]

    def draw(d: DatasetScore) -> float:
        beta = rng.betavariate(d.good + 1, max(1.0, d.tried - d.good) + 1)
        return d.opportunity / 100 * 0.6 + beta * 0.4

    return max(scored[:60], key=draw)


async def analyze_dataset(dataset_id: str, region: str, delay: int, universe: str) -> DatasetBrief:
    ctx = await pack.dataset_context(dataset_id, region, delay, universe, max_fields=150)
    prompt = await render("dataset_brief", dataset_context=ctx)
    brief = await llm.generate_json(prompt, DatasetBrief, system=await pack.static_prefix(region, delay),
                                    role="ideation", purpose="dataset_brief")
    async with db.write() as s:
        stmt = sqlite_insert(DatasetInsight).values(
            dataset_id=dataset_id, region=region, delay=delay, universe=universe, summary=brief.summary,
            ideas=[f.model_dump() for f in brief.idea_families],
            field_notes={"key_fields": [f.model_dump() for f in brief.key_fields], "themes": brief.themes,
                         "pitfalls": brief.pitfalls, "neutralization": brief.suggested_neutralization,
                         "decay": brief.suggested_decay})
        await s.execute(stmt.on_conflict_do_update(
            index_elements=["dataset_id", "region", "delay", "universe"],
            set_={"summary": stmt.excluded.summary, "ideas": stmt.excluded.ideas,
                  "field_notes": stmt.excluded.field_notes}))
    bus.publish("analyse", f"Analysed dataset {dataset_id}: {len(brief.idea_families)} idea families, "
                           f"{len(brief.key_fields)} key fields.", "success")
    return brief


async def get_insight(dataset_id: str, region: str, delay: int, universe: str) -> DatasetInsight | None:
    async with db.session() as s:
        return await s.scalar(select(DatasetInsight).where(
            DatasetInsight.dataset_id == dataset_id, DatasetInsight.region == region,
            DatasetInsight.delay == delay, DatasetInsight.universe == universe))


PROBES = [
    ("coverage", "{f}"),
    ("non-zero", "{f} != 0 ? 1 : 0"),
    ("updates (5d)", "ts_std_dev({f}, 5) != 0 ? 1 : 0"),
    ("updates (66d)", "ts_std_dev({f}, 66) != 0 ? 1 : 0"),
]


async def probe_field(field_id: str, field_type: str, region: str, delay: int, universe: str) -> int:
    """Simulate exploration probes (neutralization NONE, decay 0): long/short counts reveal coverage & update rate."""
    f = f"vec_avg({field_id})" if field_type == "VECTOR" else field_id
    base = normalize_settings({"region": region, "delay": delay, "universe": universe, "neutralization": "NONE",
                               "decay": 0, "truncation": 0.08})
    specs = [CandidateSpec(expression=t.format(f=f), settings=base, origin="probe",
                           hypothesis=f"Probe '{name}' of {field_id}") for name, t in PROBES]
    result = await simulator.submit(specs)
    bus.publish("analyse", f"Probing {field_id}: {len(result.created)} probe simulations queued.")
    return len(result.created)


@jobs.register("analyze_dataset")
async def _job_analyze(ctx: JobContext, params: dict) -> str:
    brief = await analyze_dataset(params["dataset_id"], params["region"], int(params["delay"]), params["universe"])
    return f"{len(brief.idea_families)} idea families"


# ---------------------------------------------------------------- AI ratings (Single / Double / Interaction)

USAGES = {"single": "Single Field", "double": "Double", "interaction": "Interaction"}


def _usage(value: str) -> str:
    v = (value or "").lower()
    return "double" if "double" in v or "pair" in v else "interaction" if "inter" in v else "single"


async def _rating_context(d: DatasetScore, region: str, delay: int, universe: str) -> str:
    async with db.session() as s:
        ds = await s.scalar(select(Dataset).where(Dataset.id == d.id, Dataset.region == region,
                                                  Dataset.delay == delay, Dataset.universe == universe))
        fields = list(await s.scalars(select(DataField).where(
            DataField.dataset_id == d.id, DataField.region == region, DataField.delay == delay,
            DataField.universe == universe).order_by(DataField.coverage.desc().nulls_last()).limit(8)))
    desc = ((ds.description if ds else "") or "").replace("\n", " ")[:220]
    flds = "; ".join(f"{f.id} ({f.type}{'' if f.coverage is None else f', cov {f.coverage:.0%}'}): "
                     f"{(f.description or '')[:60]}" for f in fields)
    return (f"- dataset_id={d.id} · {d.name} · BRAIN category {d.category or '?'} · coverage {d.coverage} · "
            f"value score {d.value_score} · {d.alpha_count} alphas by {d.user_count} users · {d.field_count} fields · "
            f"pyramid ×{d.pyramid_multiplier}\n  about: {desc}\n  fields: {flds or 'not synced'}")


async def rate_datasets(region: str, delay: int, universe: str, *, only_missing: bool = True, batch: int = 10,
                        ctx: JobContext | None = None) -> int:
    from ..llm.schemas import THEMES, DatasetRatingBatch
    scored = [d for d in await scored_datasets(region, delay, universe) if (d.field_count or 0) > 0]
    if only_missing:
        async with db.session() as s:
            have = set(await s.scalars(select(DatasetRating.dataset_id).where(
                DatasetRating.region == region, DatasetRating.delay == delay, DatasetRating.universe == universe)))
        scored = [d for d in scored if d.id not in have]
    done = 0
    for i in range(0, len(scored), batch):
        chunk = scored[i:i + batch]
        text = "\n".join([await _rating_context(d, region, delay, universe) for d in chunk])
        prompt = await render("dataset_rating", scope=f"{region} / D{delay} / {universe}", themes=", ".join(THEMES),
                              datasets=text)
        res = await llm.generate_json(prompt, DatasetRatingBatch, role="ideation", purpose="ds_rating",
                                         thinking="low")
        valid = {d.id for d in chunk}
        async with db.write() as s:
            for r in res.ratings:
                if r.dataset_id not in valid:
                    continue
                theme = next((t for t in THEMES if t.lower() == (r.theme or "").strip().lower()), "Other")
                vals = dict(ai_score=max(0, min(100, r.ai_score)), single=max(0, min(100, r.single)),
                            double=max(0, min(100, r.double)), interaction=max(0, min(100, r.interaction)),
                            recommended=_usage(r.recommended), theme=theme, reason=r.reason[:500], rated_at=utcnow())
                stmt = sqlite_insert(DatasetRating).values(dataset_id=r.dataset_id, region=region, delay=delay,
                                                           universe=universe, **vals)
                await s.execute(stmt.on_conflict_do_update(index_elements=["dataset_id", "region", "delay", "universe"],
                                                           set_=vals))
                done += 1
        if ctx:
            await ctx.update(min(1.0, (i + len(chunk)) / max(1, len(scored))), f"rated {done}/{len(scored)} datasets")
    bus.publish("analyse", f"AI rated {done} datasets in {region}/D{delay}/{universe}.", "success" if done else "info")
    return done


async def rated_datasets(region: str, delay: int, universe: str) -> list[dict[str, Any]]:
    """Dataset rows for the Dataset Intelligence table: opportunity data + AI rating, blended with your own yield."""
    scored = await scored_datasets(region, delay, universe)
    async with db.session() as s:
        ratings = {r.dataset_id: r for r in await s.scalars(select(DatasetRating).where(
            DatasetRating.region == region, DatasetRating.delay == delay, DatasetRating.universe == universe))}
    out = []
    for d in scored:
        r = ratings.get(d.id)
        row = d.as_row()
        ai = r.ai_score if r else None
        if ai is not None and d.tried >= 20:
            own = 100 * min(1.0, 4 * (d.good + 1) / (d.tried + 4))
            ai = round(0.75 * ai + 0.25 * own, 1)
        row.update(ai_score=ai, single=r.single if r else None, double=r.double if r else None,
                   interaction=r.interaction if r else None, recommended=r.recommended if r else None,
                   recommended_label=USAGES.get(r.recommended, "") if r else "", theme=(r.theme if r else None)
                   or d.category or "Other", reason=r.reason if r else "", rated=r is not None)
        out.append(row)
    out.sort(key=lambda x: (x["ai_score"] is None, -(x["ai_score"] or 0), -x["opportunity"]))
    return out


@jobs.register("rate_datasets")
async def _job_rate(ctx: JobContext, params: dict) -> str:
    n = await rate_datasets(params["region"], int(params["delay"]), params["universe"],
                            only_missing=not params.get("all"), ctx=ctx)
    return f"{n} datasets rated"


@jobs.register("analyze_top_datasets")
async def _job_analyze_top(ctx: JobContext, params: dict) -> str:
    region, delay, universe = params["region"], int(params["delay"]), params["universe"]
    top = [d for d in await scored_datasets(region, delay, universe) if not d.analyzed][: int(params.get("n", 5))]
    for i, d in enumerate(top):
        await ctx.update(i / max(1, len(top)), f"analysing {d.id}")
        try:
            await analyze_dataset(d.id, region, delay, universe)
        except Exception as exc:
            bus.publish("analyse", f"Dataset {d.id} analysis failed: {getattr(exc, 'message', exc)}", "warning")
    return f"{len(top)} datasets analysed"
