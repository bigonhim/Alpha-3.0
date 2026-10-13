"""LLM knowledge pack: a large, stable prompt prefix (benefits from Gemini implicit caching) + dynamic context."""

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
Submission tests (BRAIN computes; limits come from the platform, typical consultant Delay-1 values):
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


async def static_prefix() -> str:
    """Large, rarely-changing prefix: identical across calls so Gemini can cache it implicitly."""
    ops = await operator_reference()
    return "\n\n".join([
        "You are a senior quantitative researcher writing WorldQuant BRAIN alphas.",
        FASTEXPR_RULES,
        SUBMISSION_RULES,
        ops,
        "Proven templates ({f},{f2} fields, {d} window, {g} group):\n" + template_catalog_text(),
        "Classic examples:\n" + "\n".join(f"- {e}" for e in PRICE_VOLUME_SEEDS + FUNDAMENTAL_SEEDS),
    ])


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
