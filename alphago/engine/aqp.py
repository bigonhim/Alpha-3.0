"""Alpha Quality Project (AQP): the monthly cycle where you tag up to 3 of your best USA alphas and prove why.

Rules (BRAIN's announcement): tag up to 3 USA alphas with the cycle tag (AQP + YYYYMM, e.g. AQP202611); each needs a
detailed description of its hypothesis. Score = 80% description (how well it matches the expression, how unique the
idea is, the quality of the hypothesis) + 20% the alpha's out-of-sample performance. An alpha tagged in an earlier
cycle can never be tagged again, so every cycle needs 3 different alphas.

The app:
1. lists eligible alphas (submitted, USA, regular, never in an earlier AQP cycle) ranked by an estimate of the score:
   out-of-sample evidence first (OS Sharpe, else IS Sharpe × your predicted retention), plus how distinctive the idea
   is (low self-correlation, uncommon fields/datasets, a cluster not already picked);
2. picks a diverse set of 3 (different economic ideas);
3. writes each description with the AI in a fixed structure, then grades it twice: a deterministic match check (every
   field, operator, window and group in the expression must be described; nothing described may be missing from it)
   and an AI judge for match / uniqueness / hypothesis quality, rewriting until the grade stops improving;
4. tags and describes the alphas on BRAIN only when you press Apply and confirm (PATCH /alphas/{id}).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import select

from ..brain import brain
from ..db import db
from ..db.models import Alpha, Candidate, DataField, Dataset
from ..fastexpr.metrics import compute_metrics
from ..knowledge import pack
from ..llm import llm
from ..llm.schemas import AQPDescription, DescriptionGrade

log = logging.getLogger(__name__)

MAX_PER_CYCLE = 3
REGION = "USA"
TAG_RE = re.compile(r"^AQP\d{6}$", re.I)
GROUPS = ("subindustry", "industry", "sector", "market", "country", "exchange")
DESCRIPTION_WEIGHT, PERFORMANCE_WEIGHT = 0.8, 0.2
MAX_REWRITES = 2


def cycle_tag(d: date | None = None) -> str:
    d = d or date.today()
    return f"AQP{d.year}{d.month:02d}"


def aqp_tags(tags: Any) -> list[str]:
    return [str(t).upper() for t in (tags or []) if TAG_RE.match(str(t))]


def tagged_before(a: Alpha, tag: str) -> bool:
    """True when the alpha carries an AQP tag of another cycle (it can never be tagged again)."""
    return any(t != tag.upper() for t in aqp_tags(a.tags))


def in_cycle(a: Alpha, tag: str) -> bool:
    return tag.upper() in aqp_tags(a.tags)


# ---------------------------------------------------------------- performance & uniqueness estimate

def performance_score(a: Alpha) -> float:
    """0–1 estimate of the 20% performance part: out-of-sample Sharpe when BRAIN has it, else IS Sharpe discounted by
    your predicted OS/IS retention (0.6 when unknown, a typical haircut)."""
    if a.os_sharpe is not None:
        s = float(a.os_sharpe)
    else:
        s = float(a.sharpe or 0.0) * float(a.os_retention_pred if a.os_retention_pred is not None else 0.6)
    return max(0.0, min(1.0, s / 2.0))


def distinctiveness(a: Alpha, field_popularity: dict[str, int]) -> float:
    """0–1 proxy for idea uniqueness: low self-correlation and uncommon fields score higher."""
    corr = a.self_corr if a.self_corr is not None else a.est_self_corr
    corr_part = 1.0 - max(0.0, min(1.0, float(corr))) if corr is not None else 0.5
    pops = [field_popularity.get(f, 0) for f in (a.fields or [])]
    if pops:
        avg = sum(pops) / len(pops)
        rare = 1.0 / (1.0 + avg / 500.0)          # a field used in ~500 alphas on BRAIN halves this
    else:
        rare = 0.5
    shape = min(1.0, (a.field_count or len(a.fields or []) or 1) / 3.0)   # 2–3 fields read as a richer idea
    return round(0.5 * corr_part + 0.35 * rare + 0.15 * shape, 3)


@dataclass
class Candidate_:
    alpha: Alpha
    performance: float
    unique: float
    estimate: float
    in_cycle: bool
    reasons: list[str] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        a = self.alpha
        return {"id": a.id, "code": a.code, "cluster": a.cluster or "", "sharpe": a.sharpe, "fitness": a.fitness,
                "os_sharpe": a.os_sharpe, "self_corr": a.self_corr if a.self_corr is not None else a.est_self_corr,
                "performance": round(self.performance, 3), "unique": round(self.unique, 3),
                "estimate": round(self.estimate, 3), "in_cycle": self.in_cycle,
                "described": bool((a.description or "").strip()), "reasons": "; ".join(self.reasons),
                "submitted": a.date_submitted.date().isoformat() if a.date_submitted else ""}


async def _field_popularity(field_ids: set[str]) -> dict[str, int]:
    if not field_ids:
        return {}
    async with db.session() as s:
        rows = (await s.execute(select(DataField.id, DataField.alpha_count).where(
            DataField.id.in_(list(field_ids)), DataField.region == REGION))).all()
    out: dict[str, int] = {}
    for fid, n in rows:
        out[fid] = max(out.get(fid, 0), int(n or 0))
    return out


async def eligible(tag: str | None = None) -> list[Candidate_]:
    """Submitted USA regular alphas not tagged in an earlier cycle, best estimate first (the cycle's own picks first)."""
    tag = (tag or cycle_tag()).upper()
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE", Alpha.region == REGION,
                                                        Alpha.type == "REGULAR", Alpha.code.is_not(None))))
    rows = [a for a in rows if not tagged_before(a, tag)]
    pop = await _field_popularity({f for a in rows for f in (a.fields or [])})
    out = []
    for a in rows:
        perf, uniq = performance_score(a), distinctiveness(a, pop)
        # the description is 80% and is written by the app, so pick on what the description can draw on (a
        # distinctive idea) and on the 20% it cannot change (out-of-sample performance)
        est = 0.55 * uniq + 0.45 * perf
        reasons = []
        if a.os_sharpe is not None:
            reasons.append(f"OS Sharpe {a.os_sharpe:.2f}")
        elif a.sharpe is not None:
            reasons.append(f"IS Sharpe {a.sharpe:.2f} (no OS yet)")
        corr = a.self_corr if a.self_corr is not None else a.est_self_corr
        if corr is not None:
            reasons.append(f"self-corr {corr:.2f}")
        out.append(Candidate_(a, perf, uniq, est, in_cycle(a, tag), reasons))
    out.sort(key=lambda c: (not c.in_cycle, -c.estimate))
    return out


def pick(cands: list[Candidate_], n: int = MAX_PER_CYCLE) -> list[Candidate_]:
    """Greedy diverse pick: the alphas already in this cycle stay; then the best estimate whose economic idea
    (cluster) and field set differ from those already picked."""
    chosen = [c for c in cands if c.in_cycle][:n]
    for c in cands:
        if len(chosen) >= n:
            break
        if c in chosen:
            continue
        same_idea = any(c.alpha.cluster and c.alpha.cluster == p.alpha.cluster for p in chosen)
        same_fields = any(set(c.alpha.fields or []) and set(c.alpha.fields or []) == set(p.alpha.fields or [])
                          for p in chosen)
        if same_idea or same_fields:
            continue
        chosen.append(c)
    for c in cands:                        # not enough distinct ideas: fill with the best remaining
        if len(chosen) >= n:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


# ---------------------------------------------------------------- descriptions

@dataclass
class Facts:
    fields: list[str]
    operators: list[str]
    windows: list[int]
    groups: list[str]
    field_notes: dict[str, str]
    datasets: list[str]


async def expression_facts(a: Alpha) -> Facts:
    try:
        m = compute_metrics(a.code or "")
        ops, wins, mfields = sorted(m.unique_operators), sorted(set(m.windows)), set(m.fields)
    except Exception:                         # unparsable code: fall back to what BRAIN reported
        ops, wins, mfields = [], [], set()
    fields = sorted(mfields | set(a.fields or []))
    groups = sorted({g for g in GROUPS if re.search(rf"\b{g}\b", a.code or "")})
    fields = [f for f in fields if f not in GROUPS]
    notes: dict[str, str] = {}
    datasets: set[str] = set()
    if fields:
        async with db.session() as s:
            rows = list(await s.scalars(select(DataField).where(DataField.id.in_(fields), DataField.region == REGION)))
            ds_ids = {r.dataset_id for r in rows if r.dataset_id}
            ds_rows = list(await s.scalars(select(Dataset).where(Dataset.id.in_(list(ds_ids)),
                                                                 Dataset.region == REGION))) if ds_ids else []
        for r in rows:
            if r.description and r.id not in notes:
                notes[r.id] = r.description
        datasets = {f"{d.id} ({d.name})" if d.name else d.id for d in ds_rows}
    return Facts(fields=fields, operators=ops, windows=wins, groups=groups,
                 field_notes=notes, datasets=sorted(datasets))


def match_check(text: str, facts: Facts, settings: dict[str, Any] | None = None) -> list[str]:
    """Deterministic match problems: parts of the expression the description never mentions."""
    low = (text or "").lower()
    problems = []
    for f in facts.fields:
        if f.lower() not in low:
            problems.append(f"field {f} is not described")
    for op in facts.operators:
        if op.startswith(("add", "subtract", "multiply", "divide", "reverse", "less", "greater", "equal", "not",
                          "and", "or", "if_else")):
            continue                        # arithmetic needs no prose of its own
        if op.lower() not in low:
            problems.append(f"operator {op} is not explained")
    for w in facts.windows:
        if w >= 2 and not re.search(rf"\b{w}\b", low):
            problems.append(f"window {w} is not justified")
    for g in facts.groups:
        if g not in low:
            problems.append(f"grouping by {g} is not explained")
    neut = str((settings or {}).get("neutralization") or "").lower()
    if neut and neut not in ("none",) and neut not in low:
        problems.append(f"{neut} neutralization is not mentioned")
    return problems


def _prompt(a: Alpha, facts: Facts, *, feedback: str = "", previous: str = "") -> str:
    notes = "\n".join(f"- {f}: {facts.field_notes.get(f, '(no catalogue description)')}" for f in facts.fields)
    return f"""TASK: Write the description for WorldQuant BRAIN's Alpha Quality Project (AQP). It is scored on (1) how
exactly it matches the expression, (2) how unique the idea is, (3) the quality of the hypothesis. Be concrete and
technical; every claim must be verifiable from the expression. Do not invent fields, operators, windows or data.

Expression: {a.code}
Settings: {pack.settings_text(a.settings or {})}
Fields and what they measure:
{notes or '- (none)'}
Datasets: {', '.join(facts.datasets) or 'unknown'}
Operators used: {', '.join(facts.operators)}
Windows used: {', '.join(map(str, facts.windows)) or 'none'}
Groupings used: {', '.join(facts.groups) or 'none'}
In-sample: Sharpe {a.sharpe}, Fitness {a.fitness}, Turnover {a.turnover}, Returns {a.returns}
Economic idea cluster (app's guess): {a.cluster or 'unknown'}

Requirements:
- Name every field, explain every non-arithmetic operator and justify every window and grouping, in the order the
  expression applies them, including the sign (which stocks are bought, which are sold).
- State a falsifiable hypothesis: who misprices what, why, and why it corrects over the holding horizon implied by the
  windows and turnover.
- Say what is distinctive versus textbook factors (momentum, value, size, low-vol) without overclaiming.
- No marketing language, no performance promises.
{('Feedback on your previous version (fix all of it):' + chr(10) + feedback) if feedback else ''}
{('Previous version:' + chr(10) + previous) if previous else ''}
Return JSON matching the schema."""


def _grade_prompt(a: Alpha, text: str) -> str:
    return f"""You grade descriptions for WorldQuant BRAIN's Alpha Quality Project. Be strict.
Expression: {a.code}
Settings: {pack.settings_text(a.settings or {})}

Description:
{text}

Score match (every claim corresponds to the expression; nothing in the expression is left unexplained), uniqueness
(originality versus standard factors) and hypothesis quality (clear, economically grounded, falsifiable). List every
mismatch and the concrete improvements. Return JSON matching the schema."""


def total(grade: DescriptionGrade, problems: list[str]) -> float:
    """0–10 description score: the AI's three grades with the deterministic match problems as a penalty."""
    match = max(0.0, grade.match - 1.0 * len(problems))
    return round((match * 0.45 + grade.uniqueness * 0.25 + grade.hypothesis_quality * 0.30), 2)


def fallback(a: Alpha, facts: Facts, hypothesis: str | None) -> str:
    fields = ", ".join(facts.fields) or "the selected data fields"
    notes = "; ".join(f"{f} measures {facts.field_notes[f]}" for f in facts.fields if f in facts.field_notes)
    neut = (a.settings or {}).get("neutralization", "")
    return "\n\n".join([
        f"Idea: {hypothesis or 'A cross-sectional signal built from ' + fields + '.'}",
        f"Rationale for data used: {notes or fields}.",
        f"Rationale for operators used (construction): the expression {a.code} applies "
        f"{', '.join(facts.operators)}" + (f" over windows of {', '.join(map(str, facts.windows))} days" if
                                           facts.windows else "")
        + (f", compares stocks within {', '.join(facts.groups)}" if facts.groups else "")
        + (f", and is neutralized by {neut.lower()}" if neut else "") + ".",
        "Risks and limitations: the signal can weaken when the information becomes widely used or in regimes where "
        "the underlying behaviour changes.",
    ])


async def write_description(alpha_id: str) -> dict[str, Any]:
    """Write, grade and rewrite an AQP description. Returns {text, grade, problems, score, rounds}."""
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
        cand = await s.get(Candidate, a.candidate_id) if a and a.candidate_id else None
    if a is None:
        raise ValueError(f"Unknown alpha {alpha_id}")
    facts = await expression_facts(a)
    if not llm.configure():
        text = fallback(a, facts, cand.hypothesis if cand else None)
        problems = match_check(text, facts, a.settings)
        return {"text": text, "grade": None, "problems": problems, "score": None, "rounds": 0,
                "note": "No AI provider set up: a template description was written. Set one up for a scored one."}
    best: dict[str, Any] | None = None
    feedback, previous = "", ""
    for rounds in range(1, MAX_REWRITES + 2):
        desc = await llm.generate_json(_prompt(a, facts, feedback=feedback, previous=previous), AQPDescription,
                                       role="describe", purpose="aqp_describe", thinking="medium")
        text = desc.render()
        problems = match_check(text, facts, a.settings)
        try:
            grade = await llm.generate_json(_grade_prompt(a, text), DescriptionGrade, role="refine",
                                            purpose="aqp_grade", thinking="low")
        except Exception as exc:              # the judge is advisory: keep the description
            log.warning("AQP grading failed: %s", exc)
            grade = DescriptionGrade(match=7, uniqueness=6, hypothesis_quality=6)
        score = total(grade, problems)
        current = {"text": text, "grade": grade.model_dump(), "problems": problems, "score": score,
                   "rounds": rounds}
        if best is None or score > best["score"]:
            best = current
        elif score <= best["score"]:
            break                             # no longer improving
        if score >= 9.0 and not problems:
            break
        feedback = "\n".join([f"- {p}" for p in problems] + [f"- mismatch: {m}" for m in grade.mismatches]
                             + [f"- improve: {i}" for i in grade.improvements])
        previous = text
    await db.kv_set(f"aqp:draft:{alpha_id}", {**best, "at": datetime.now(timezone.utc).isoformat()})
    return best


async def draft(alpha_id: str) -> dict[str, Any] | None:
    return await db.kv_get(f"aqp:draft:{alpha_id}")


def estimated_score(description_score: float | None, performance: float) -> float | None:
    """Expected AQP score (0–100) from the 0–10 description grade and the 0–1 performance estimate."""
    if description_score is None:
        return None
    return round(100 * (DESCRIPTION_WEIGHT * description_score / 10 + PERFORMANCE_WEIGHT * performance), 1)


# ---------------------------------------------------------------- apply on BRAIN

async def apply(alpha_id: str, description: str, tag: str | None = None) -> dict[str, Any]:
    """Tag the alpha with the cycle tag and set its description on BRAIN. Refuses past 3 alphas a cycle and alphas
    tagged in an earlier cycle."""
    tag = (tag or cycle_tag()).upper()
    if not TAG_RE.match(tag):
        raise ValueError(f"{tag} is not an AQP cycle tag (AQP + YYYYMM).")
    if len((description or "").strip()) < 200:
        raise ValueError("Write a detailed description first (at least 200 characters).")
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
        tagged = list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE", Alpha.region == REGION)))
    if a is None:
        raise ValueError(f"Unknown alpha {alpha_id}")
    if a.region != REGION or str(a.status).upper() != "ACTIVE":
        raise ValueError("AQP takes submitted USA alphas only.")
    if tagged_before(a, tag):
        raise ValueError(f"{alpha_id} was tagged in an earlier AQP cycle and cannot be tagged again.")
    in_now = [x.id for x in tagged if in_cycle(x, tag)]
    if alpha_id not in in_now and len(in_now) >= MAX_PER_CYCLE:
        raise ValueError(f"{tag} already has {MAX_PER_CYCLE} alphas ({', '.join(in_now)}). Remove one on BRAIN first.")
    tags = list(dict.fromkeys([*(a.tags or []), tag]))
    body = await brain.alphas.patch(alpha_id, tags=tags, regular_description=description.strip())
    async with db.write() as s:
        row = await s.get(Alpha, alpha_id)
        if row:
            row.tags = (body or {}).get("tags", tags) if isinstance(body, dict) else tags
            row.description = description.strip()
    return {"alpha_id": alpha_id, "tag": tag, "tags": tags}
