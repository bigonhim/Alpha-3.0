"""LLM knowledge pack: a large, stable prompt prefix (benefits from provider prompt caching) + dynamic context."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from ..db import db
from ..db.models import Alpha, DataField, Dataset, JournalEntry, Operator
from .operators import CURATED_OPERATORS
from .templates import FUNDAMENTAL_SEEDS, PRICE_VOLUME_SEEDS, template_catalog_text

FASTEXPR_RULES = """\
FASTEXPR (WorldQuant BRAIN alpha language) — rules you MUST follow:
1. An alpha is an expression evaluated every day for every stock; the result is the stock's weight (long if
   positive, short if negative; BRAIN normalises the book, so only the cross-sectional shape matters).
2. Program = statements separated by ';'. Assignments `name = expr`; the LAST statement must be a bare expression.
3. Operators: only use operators from the provided list, with the exact argument order. Named args go last,
   e.g. winsorize(x, std=4), ts_decay_exp_window(x, 20, factor=0.5), bucket(rank(cap), range="0.1,1,0.1").
4. There is NO '^' operator: use power(x, y) or signed_power(x, y). Comparisons/logic: < <= > >= == != && || !,
   ternary c ? a : b, or if_else(c, a, b).
5. Lookback windows are positive integer literals (typical: 5, 22, 66, 126, 252; max ~500).
6. Field types: MATRIX fields are used directly. VECTOR fields MUST be reduced first with vec_avg(x) or vec_sum(x).
   GROUP fields (and market/sector/industry/subindustry) may ONLY be used as the group argument of group_* ops.
7. Sparse / quarterly data (fundamentals, analyst estimates, news) should be wrapped in ts_backfill(x, 60..252).
   Normalise raw levels with rank(), zscore(), ts_rank(), ts_zscore() or divide by cap / assets.
8. Never use future information; BRAIN already aligns data by delay.
9. Keep expressions simple: ≤ 6 operators and 1–3 data fields is ideal (simpler alphas are more robust,
   less correlated, and score better in Genius tie-breakers and the Power Pool).
"""

SUBMISSION_RULES = """\
Submission tests (BRAIN computes; limits come from the platform — the numbers below are typical values, NOT verified
for this account; when "your account's limits" are listed further down, those are authoritative):
- Sharpe ≥ 1.58 (D0: 2.69), Fitness ≥ 1.0 (D0: 1.5). Fitness = Sharpe * sqrt(|Returns| / max(Turnover, 0.125)).
- Turnover between 1% and 70%; very high turnover destroys fitness, below 12.5% costs nothing extra.
- No stock above 10% weight (CONCENTRATED_WEIGHT) – rank()/truncation fix this.
- Sub-universe Sharpe ≥ 0.75*sqrt(sub/universe)*Sharpe – avoid pure size tilts.
- Self/production correlation < 0.7 against existing alphas — novelty (different data or mechanism) matters more
  than tweaking windows or neutralization.
- IS ladder: recent years must also be strong (Sharpe ≥ 2.38 over the last 2 years for single-dataset alphas).
Settings heuristics: fundamentals → decay 0, SUBINDUSTRY, nanHandling ON; analyst → decay 0–4, INDUSTRY;
technical reversal → decay 10–30, INDUSTRY; sentiment/news → decay 4–10, INDUSTRY. Increase decay or use
trade_when/hump to cut turnover; flip the sign of strongly negative ideas.
"""


async def operator_reference(limit: int = 220) -> str:
    async with db.session() as s:
        rows = list(await s.scalars(select(Operator).order_by(Operator.category, Operator.name)))
    lines = []
    if rows:
        for r in rows[:limit]:
            scope = r.scope if isinstance(r.scope, list) else [r.scope] if r.scope else []
            if scope and "REGULAR" not in [str(x).upper() for x in scope]:
                continue
            definition = (r.definition or r.name).replace("\n", " ")
            desc = (r.description or "").replace("\n", " ")[:140]
            lines.append(f"- {definition} — {desc}")
    else:
        for spec in list(CURATED_OPERATORS.values())[:limit]:
            if spec.tier != "B":
                continue
            lines.append(f"- {spec.definition or spec.name} — {spec.description[:140]}")
    return "Available operators (signature — meaning):\n" + "\n".join(lines)


# check name -> how its limit reads ("≥" for minimums, "≤" for maximums)
_LIMIT_SIDE = {"LOW_SHARPE": "Sharpe ≥", "LOW_FITNESS": "Fitness ≥", "LOW_TURNOVER": "Turnover ≥",
               "HIGH_TURNOVER": "Turnover ≤", "CONCENTRATED_WEIGHT": "max stock weight ≤",
               "LOW_SUB_UNIVERSE_SHARPE": "sub-universe Sharpe ≥", "IS_LADDER_SHARPE": "IS-ladder Sharpe ≥",
               "LOW_2Y_SHARPE": "last-2-year Sharpe ≥", "SELF_CORRELATION": "self-correlation ≤",
               "PROD_CORRELATION": "production correlation ≤"}


async def observed_limits(region: str | None, delay: int | None, sample: int = 300) -> dict[str, float]:
    """The latest `limit` BRAIN reported for each check in this region/delay (from stored check results)."""
    if region is None:
        return {}
    async with db.session() as s:
        stmt = select(Alpha.checks, Alpha.is_checks).where(Alpha.region == region)
        if delay is not None:
            stmt = stmt.where(Alpha.delay == delay)
        rows = (await s.execute(stmt.order_by(Alpha.updated_at.desc()).limit(sample))).all()
    out: dict[str, float] = {}
    for checks, is_checks in rows:
        for c in (checks or []) + (is_checks or []):
            name, limit = c.get("name"), c.get("limit")
            if name and name not in out and isinstance(limit, (int, float)):
                out[str(name)] = float(limit)
    return out


def limits_text(limits: dict[str, float], region: str | None = None, delay: int | None = None) -> str:
    if not limits:
        return ""
    parts = [f"{_LIMIT_SIDE[k]} {v:g}" for k, v in sorted(limits.items()) if k in _LIMIT_SIDE] + \
            [f"{k}: limit {v:g}" for k, v in sorted(limits.items()) if k not in _LIMIT_SIDE]
    scope = f" for {region} D{delay}" if region else ""
    return f"Your account's limits as last reported by BRAIN{scope} (authoritative): " + "; ".join(parts) + "."


async def static_prefix(region: str | None = None, delay: int | None = None) -> str:
    """Large, rarely-changing prefix: identical across calls so the provider can cache it. With a scope, BRAIN's
    observed limits for that region/delay are added (they rarely change, so caching still works)."""
    ops = await operator_reference()
    observed = limits_text(await observed_limits(region, delay), region, delay) if region else ""
    return "\n\n".join(part for part in [
        "You are a senior quantitative researcher writing WorldQuant BRAIN alphas.",
        FASTEXPR_RULES,
        SUBMISSION_RULES,
        observed,
        ops,
        "Proven templates ({f},{f2} fields, {d} window, {g} group):\n" + template_catalog_text(),
        "Classic examples:\n" + "\n".join(f"- {e}" for e in PRICE_VOLUME_SEEDS + FUNDAMENTAL_SEEDS),
    ] if part)


async def dataset_context(dataset_id: str, region: str, delay: int, universe: str, max_fields: int = 120) -> str:
    async with db.session() as s:
        ds = await s.scalar(select(Dataset).where(Dataset.id == dataset_id, Dataset.region == region,
                                                  Dataset.delay == delay, Dataset.universe == universe))
        fields = list(await s.scalars(select(DataField).where(
            DataField.dataset_id == dataset_id, DataField.region == region, DataField.delay == delay,
            DataField.universe == universe)))
    # prefer usable, well-covered, less-crowded MATRIX/VECTOR fields
    def rank_key(f: DataField) -> tuple:
        cov = f.coverage or 0
        return (f.type == "GROUP", -min(cov, 1.0) + (f.alpha_count or 0) / 50000.0)

    fields.sort(key=rank_key)
    lines = [f"- {f.id} [{f.type}] cov={(f.coverage or 0):.2f} alphas={f.alpha_count or 0}: "
             f"{(f.description or '').replace(chr(10), ' ')[:150]}" for f in fields[:max_fields]]
    head = f"Dataset {dataset_id}"
    if ds:
        head += (f" — {ds.name} (category: {ds.category_name}; coverage {ds.coverage}; value score "
                 f"{ds.value_score}; {ds.alpha_count} alphas by {ds.user_count} users; pyramid multiplier "
                 f"{ds.pyramid_multiplier}).\n{(ds.description or '')[:800]}")
    return f"{head}\nScope: region {region}, delay {delay}, universe {universe}.\nFields:\n" + "\n".join(lines)


async def best_examples(region: str, limit: int = 8) -> list[str]:
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(
            Alpha.region == region, Alpha.verdict.in_(["submitted", "ready", "strong"]), Alpha.code.is_not(None))
            .order_by(Alpha.score.desc().nulls_last()).limit(limit * 3)))
    seen, out = set(), []
    for a in rows:
        key = a.skeleton_hash or a.code
        if key in seen:
            continue
        seen.add(key)
        out.append(f"- {a.code}   (Sharpe {a.sharpe:.2f}, fitness {a.fitness or 0:.2f}, turnover "
                   f"{(a.turnover or 0) * 100:.0f}%, {a.neutralization}, decay {a.decay})")
        if len(out) >= limit:
            break
    return out


EDIT_MARK = "replaced by an edited expression"


async def review_examples(region: str, per_kind: int = 6, max_chars: int = 180) -> str:
    """Your own review of earlier AI batches, as examples for the next one: ideas you approved (best results
    first), ideas you rejected before any simulation, and expressions you rewrote (the model's → yours)."""
    from sqlalchemy.orm import aliased

    from ..db.models import AutopilotRun, Batch, Candidate
    parent = aliased(Candidate)
    async with db.session() as s:
        approved = (await s.execute(
            select(Candidate.expression, Alpha.verdict, Alpha.sharpe)
            .join(Batch, Batch.id == Candidate.batch_id).join(AutopilotRun, AutopilotRun.id == Batch.run_id)
            .outerjoin(Alpha, Alpha.id == Candidate.alpha_id)
            .where(AutopilotRun.mode == "assisted", Candidate.region == region, Candidate.parent_id.is_(None),
                   Candidate.status.in_(["done", "queued", "simulating", "reused"]))
            .order_by(Alpha.score.desc().nulls_last(), Candidate.id.desc()).limit(per_kind))).all()
        rejected = list(await s.scalars(
            select(Candidate.expression).where(Candidate.region == region, Candidate.status == "rejected_by_user",
                                               Candidate.error.is_(None))
            .order_by(Candidate.id.desc()).limit(per_kind)))
        edits = (await s.execute(
            select(parent.expression, Candidate.expression).join(parent, parent.id == Candidate.parent_id)
            .where(Candidate.region == region, parent.status == "rejected_by_user", parent.error == EDIT_MARK)
            .order_by(Candidate.id.desc()).limit(per_kind))).all()
    if not (approved or rejected or edits):
        return ""
    cut = lambda e: (e or "")[:max_chars]  # noqa: E731
    lines = ["How the researcher reviewed earlier AI ideas — learn their preferences, do not copy these:"]
    if approved:
        lines.append("Approved (with how they did once simulated):")
        lines += [f"- {cut(e)}" + (f"  → {v}, Sharpe {sh:.2f}" if v and sh is not None else "  → not simulated yet")
                  for e, v, sh in approved]
    if rejected:
        lines.append("Rejected before any simulation (avoid ideas like these):")
        lines += [f"- {cut(e)}" for e in rejected]
    if edits:
        lines.append("Rewritten by the researcher (yours → theirs; write the way they rewrite):")
        lines += [f"- {cut(a)}\n  → {cut(b)}" for a, b in edits]
    return "\n".join(lines)


async def avoid_list(region: str, dataset_id: str | None = None, limit: int = 25) -> list[str]:
    async with db.session() as s:
        stmt = select(JournalEntry).where(JournalEntry.region == region,
                                          JournalEntry.verdict.in_(["rejected", "weak"]))
        if dataset_id:
            stmt = stmt.where(JournalEntry.dataset_id == dataset_id)
        rows = list(await s.scalars(stmt.order_by(JournalEntry.id.desc()).limit(limit * 3)))
    seen, out = set(), []
    for r in rows:
        if r.skeleton_hash in seen:
            continue
        seen.add(r.skeleton_hash)
        out.append(f"- {r.expression[:160]} (Sharpe {r.sharpe if r.sharpe is not None else 'n/a'})")
        if len(out) >= limit:
            break
    return out


def settings_text(settings: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in settings.items() if k in (
        "region", "universe", "delay", "neutralization", "decay", "truncation", "nanHandling"))
