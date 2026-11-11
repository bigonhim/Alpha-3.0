"""Alpha Quality Project (AQP): pick up to 3 USA alphas per monthly cycle, write their descriptions, tag them.

BRAIN's rules for a cycle: tag up to 3 of your best USA alphas with AQP<YYYYMM>; each description is scored on how well it
matches the expression, how unique the idea is and how good the hypothesis is (80% together), plus the alpha's
out-of-sample performance (20%); an alpha tagged in an earlier cycle cannot be tagged again.

What this module does with that:
  - candidates   ACTIVE, regular, USA alphas, with the ones already used in an earlier cycle marked and excluded;
  - ranking      a heuristic, NOT BRAIN's score: 80% "describability" (simple expression, few fields, a known idea family,
                 low overlap with your other alphas) and 20% out-of-sample Sharpe (BRAIN's, else an estimate from your
                 predicted OS/IS retention, else a 50% haircut of the in-sample Sharpe);
  - pick         three alphas from different idea families;
  - description  written by the AI to the three scored criteria, then pre-checked: every field and operator of the
                 expression must be named (a literal check), an AI judge scores match / uniqueness / hypothesis, and the
                 three descriptions must differ from each other. A failed check sends its evidence back for a rewrite;
  - apply        one confirmed job writes the description and the cycle tag together, all-or-nothing validation first.

Tagging is permanent for that alpha (it can never be used in another cycle), so nothing is written without a confirmed job.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select

from ..config import settings
from ..db import db
from ..db.models import Alpha, DataField
from ..events import bus
from ..jobs.queue import JobContext, jobs
from ..llm import llm
from . import actions

log = logging.getLogger(__name__)

LEDGER_KEY = "aqp_ledger"
DRAFT_KEY = "aqp:draft:{}"
BACKUP_KEY = "aqp:backup:{}"
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"(?<![A-Za-z_\.])\d+(?:\.\d+)?")
_WORDS = re.compile(r"[a-z]{4,}")
_RESERVED = {"true", "false", "nan", "inf", "if", "else", "and", "or", "not", "in"}


# ---------------------------------------------------------------- cycle and eligibility

def current_cycle(today: date | None = None) -> str:
    return f"{settings.aqp.tag_prefix}{(today or date.today()):%Y%m}"


def cycle_pattern() -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(settings.aqp.tag_prefix)}\d{{6}}$")


def is_cycle_tag(tag: str) -> bool:
    return bool(cycle_pattern().match(tag or ""))


def eligibility(a: Alpha, cycle: str, ledger: list[dict[str, Any]] | None = None) -> tuple[str, str]:
    """(state, reason): selected | eligible | used | excluded."""
    tags = list(a.tags or [])
    if (a.type or "REGULAR") != "REGULAR":
        return "excluded", "Only regular alphas can be described this way."
    if a.status != "ACTIVE":
        return "excluded", "Not submitted yet, so it has no out-of-sample performance."
    if (a.region or "") != settings.aqp.region:
        return "excluded", f"Not a {settings.aqp.region} alpha."
    if cycle in tags:
        return "selected", f"Tagged {cycle}."
    earlier = [t for t in tags if is_cycle_tag(t)] + [e["cycle"] for e in (ledger or []) if e.get("alpha_id") == a.id
                                                    and e.get("cycle") != cycle]
    if earlier:
        return "used", f"Already used in {sorted(set(earlier))[0]}: it cannot be tagged again."
    return "eligible", ""


# ---------------------------------------------------------------- ranking (a heuristic, not BRAIN's score)

def describability(a: Alpha) -> tuple[float, list[str]]:
    """0 to 1: how well this alpha can be explained accurately and distinctively."""
    score, why = 1.0, []
    ops, fields = a.op_count or 0, a.field_count or len(a.fields or [])
    if ops > 12:
        score -= 0.35
        why.append(f"{ops} operators: hard to explain step by step")
    elif ops > 8:
        score -= 0.15
        why.append(f"{ops} operators")
    if fields > 3:
        score -= 0.15
        why.append(f"{fields} fields")
    if not a.cluster:
        score -= 0.10
        why.append("no clear idea family")
    sc = a.est_self_corr if a.est_self_corr is not None else a.self_corr
    if sc is not None and sc > 0.6:
        score -= 0.20
        why.append(f"close to another of your alphas ({sc:.2f})")
    return max(0.0, score), why


def performance(a: Alpha) -> tuple[float, float | None, str]:
    """0 to 1 from BRAIN's out-of-sample Sharpe when there is one, else an estimate from the in-sample Sharpe: your
    predicted OS/IS retention when the app has one, else a 50% haircut."""
    if a.os_sharpe is not None:
        return max(0.0, min(1.0, a.os_sharpe / 2.0)), a.os_sharpe, "OS"
    if a.sharpe is not None:
        keep = a.os_retention_pred if a.os_retention_pred is not None else 0.5
        est = a.sharpe * max(0.0, min(1.5, float(keep)))
        return max(0.0, min(1.0, est / 2.0)), est, "est."
    return 0.0, None, "none"


def score_alpha(a: Alpha) -> dict[str, Any]:
    d, why = describability(a)
    perf, sharpe, basis = performance(a)
    return {"describability": round(d, 3), "perf": round(perf, 3), "os": sharpe, "os_basis": basis,
            "score": round(0.8 * d + 0.2 * perf, 3), "notes": "; ".join(why)}


async def candidates(cycle: str | None = None) -> list[dict[str, Any]]:
    cycle = cycle or current_cycle()
    ledger = await db.kv_get(LEDGER_KEY, []) or []
    async with db.session() as s:
        alphas = list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE")))
    rows = []
    for a in alphas:
        state, reason = eligibility(a, cycle, ledger)
        if state == "excluded" and (a.region or "") != settings.aqp.region:
            continue                                  # other regions are not part of this list at all
        draft = await db.kv_get(DRAFT_KEY.format(a.id))
        rows.append({"id": a.id, "code": a.code or "", "state": state, "reason": reason, "is_sharpe": a.sharpe,
                     "turnover": a.turnover, "ops": a.op_count, "fields": len(a.fields or []), "cluster": a.cluster,
                     "draft_ok": bool(draft and draft.get("ok")), "has_draft": bool(draft),
                     "description_chars": len(a.description or ""), **score_alpha(a)})
    rows.sort(key=lambda r: ({"selected": 0, "eligible": 1, "used": 2, "excluded": 3}[r["state"]], -r["score"]))
    return rows


def pick_best(rows: list[dict[str, Any]], n: int | None = None) -> list[str]:
    """The best `n` eligible alphas, from different idea families where there are enough of them."""
    n = n or settings.aqp.max_picks
    pool = [r for r in rows if r["state"] in ("eligible", "selected")]
    pool.sort(key=lambda r: (r["state"] != "selected", -r["score"]))     # this cycle's picks stay picked
    picked: list[dict[str, Any]] = []
    for r in pool:
        if len(picked) < n and (not r["cluster"] or r["cluster"] not in {p["cluster"] for p in picked}):
            picked.append(r)
    for r in pool:                                    # not enough distinct families: fill with the next best
        if len(picked) < n and r not in picked:
            picked.append(r)
    return [r["id"] for r in picked]


# ---------------------------------------------------------------- the literal "does it match the expression" check

def expression_terms(code: str) -> tuple[list[str], list[str], list[str]]:
    """(operators, fields/groups, numbers) the description has to cover. Assigned names and keyword arguments are not."""
    ops, names, assigned = [], [], set()
    text = code or ""
    for m in _TOKEN.finditer(text):
        word, rest = m.group(0), text[m.end():].lstrip()
        if rest.startswith("("):
            ops.append(word)
        elif rest.startswith("=") and not rest.startswith("=="):
            assigned.add(word)
        elif word.lower() not in _RESERVED:
            names.append(word)
    dedupe = lambda xs: list(dict.fromkeys(xs))  # noqa: E731
    numbers = dedupe(n for n in _NUMBER.findall(text) if n not in ("0", "1"))
    return dedupe(ops), dedupe(n for n in names if n not in assigned), numbers


def _mentions(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text, re.I))


def coverage(description: str, code: str) -> dict[str, Any]:
    ops, names, numbers = expression_terms(code)
    terms = ops + names
    missing = [t for t in terms if not _mentions(description, t)]
    missing_numbers = [n for n in numbers if not _mentions(description, n)]
    return {"terms": len(terms), "missing": missing, "missing_numbers": missing_numbers,
            "share": 1.0 if not terms else 1 - len(missing) / len(terms)}


def similarity(a: str, b: str) -> float:
    x, y = set(_WORDS.findall(a.lower())), set(_WORDS.findall(b.lower()))
    return len(x & y) / len(x | y) if x | y else 0.0


def check_description(text: str, code: str) -> dict[str, Any]:
    """The pre-checks that need no AI: length and the literal coverage of the expression."""
    cov = coverage(text, code)
    problems = []
    if len(text) < settings.aqp.min_description_chars:
        problems.append(f"{len(text)} characters; at least {settings.aqp.min_description_chars} needed to explain every "
                        "field and operator.")
    if cov["missing"]:
        problems.append("Does not name: " + ", ".join(cov["missing"]) + ".")
    return {"ok": not problems, "problems": problems, "chars": len(text), **cov}


# ---------------------------------------------------------------- writing the description

class AQPDescription(BaseModel):
    idea: str = Field(description=("The hypothesis: mechanism, who is wrong or paid, "
                           "expected sign and horizon (4 to 7 sentences)"))
    rationale_data: str = Field(description="One short paragraph per field: what it measures and why it is informative here")
    rationale_operators: str = Field(description=("The expression walked through in evaluation "
                                          "order, every operator and window explained"))
    different: str = Field(description=("What differs from the textbook version of "
                                "this idea and from the other alphas listed"))
    risks: str = Field(description="When it should fail, and what to expect in those regimes")

    def render(self) -> str:
        return (f"Idea: {self.idea.strip()}\n\nRationale for data used: {self.rationale_data.strip()}\n\n"
                f"Rationale for operators used: {self.rationale_operators.strip()}\n\n"
                f"What makes it different: {self.different.strip()}\n\nExpected behaviour and risks: {self.risks.strip()}")


class AQPGrade(BaseModel):
    match: int = Field(description="0-10: does the description say exactly what the expression computes")
    uniqueness: int = Field(description=("0-10: does it say what is different from "
                                 "the standard version and from the other alphas"))
    hypothesis: int = Field(description=("0-10: is the mechanism specific, plausible, "
                                 "testable and honest about failure regimes"))
    gaps: list[str] = Field(default_factory=list, description="Concrete things to fix, most important first")


DESCRIBE_TEMPLATE = """\
You write the description of a WorldQuant BRAIN alpha for the Alpha Quality Project. Reviewers score it on:
1. MATCH: every field, operator, window, group and setting in the expression is explained, and the text says what the
   expression really computes, in evaluation order.
2. UNIQUENESS: it says what is different from the textbook version of this idea and from the other alphas listed below.
3. HYPOTHESIS: a specific, testable economic or behavioural mechanism: who is wrong or paid, why the effect persists, in
   which regimes it fails, which sign and horizon to expect.
Be concrete and accurate. Never claim anything the expression does not do. Do not invent facts about the data: use the
field descriptions given. No marketing language and no performance numbers. Use the literal name of every field and every
operator, and mention the window lengths and the neutralization.

Expression: {code}
Settings: {settings}
Fields:
{fields}
Idea family: {cluster}
Other alphas described this cycle (this one must be clearly different from them):
{others}
{revision}
Write at least {min_chars} characters in total. Return JSON matching the schema."""

REVISION_TEMPLATE = """\
YOUR PREVIOUS ATTEMPT:
{previous}

WHAT MUST CHANGE (measured; fix exactly these, keep what already works):
{gaps}
"""

JUDGE_TEMPLATE = """\
You are a strict reviewer of WorldQuant BRAIN alpha descriptions. Score the description of this alpha out of 10 on each of:
match (it says exactly what the expression computes, field by field, operator by operator),
uniqueness (it says what differs from the standard version of the idea and from the other alphas),
hypothesis (specific, plausible, testable mechanism; names failure regimes). Be harsh: 7 means good, 9 is rare.
List concrete gaps, most important first.

Expression: {code}
Settings: {settings}
Other alphas this cycle: {others}

DESCRIPTION:
{description}

Return JSON matching the schema."""


async def _field_notes(a: Alpha) -> str:
    names = list(a.fields or [])
    if not names:
        return "(none recorded)"
    async with db.session() as s:
        rows = list(await s.scalars(select(DataField).where(DataField.id.in_(names), DataField.region == a.region)))
    desc = {r.id: (r.description or "").strip() for r in rows}
    return "\n".join(f"- {n}: {desc.get(n) or 'no description synced'}" for n in names)


def _settings_text(a: Alpha) -> str:
    s = a.settings or {}
    keys = ("region", "universe", "delay", "neutralization", "decay", "truncation")
    return ", ".join(f"{k}={s.get(k)}" for k in keys if s.get(k) is not None) or "defaults"


async def _others_text(alpha_ids: list[str], skip: str) -> str:
    out = []
    for aid in alpha_ids:
        if aid == skip:
            continue
        async with db.session() as s:
            o = await s.get(Alpha, aid)
        if o:
            out.append(f"- {o.code} ({o.cluster or 'unclassified'})")
    return "\n".join(out) or "(none)"


async def write_description(alpha_id: str, *, others: list[str] | None = None,
                            rounds: int | None = None) -> dict[str, Any]:
    """Draft, pre-check, rewrite on failure. Returns and stores {text, ok, checks, grade, rounds}."""
    cfg = settings.aqp
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
    if a is None or not a.code:
        raise ValueError(f"Unknown alpha {alpha_id}")
    fields, others_txt = await _field_notes(a), await _others_text(others or [], alpha_id)
    base = dict(code=a.code, settings=_settings_text(a), fields=fields, cluster=a.cluster or "unclassified",
                others=others_txt, min_chars=cfg.min_description_chars)
    neut = str((a.settings or {}).get("neutralization") or "").strip()
    text, gaps, result = "", [], {}
    for rnd in range(1, (cfg.max_rounds if rounds is None else rounds) + 2):
        revision = REVISION_TEMPLATE.format(previous=text, gaps="\n".join(f"- {g}" for g in gaps)) if gaps else ""
        draft = await llm.generate_json(DESCRIBE_TEMPLATE.format(revision=revision, **base), AQPDescription,
                                        role="describe", purpose="aqp_describe", thinking="medium")
        text = draft.render()
        check = check_description(text, a.code)
        grade = await llm.generate_json(JUDGE_TEMPLATE.format(code=a.code, settings=base["settings"], others=others_txt,
                                                              description=text), AQPGrade, role="refine",
                                        purpose="aqp_judge", thinking="low")
        low = [n for n, v in (("match", grade.match), ("uniqueness", grade.uniqueness), ("hypothesis", grade.hypothesis))
               if v < cfg.min_grade]
        gaps = list(check["problems"]) + ([f"Judge scores below {cfg.min_grade}/10: {', '.join(low)}."] if low else []) \
            + list(grade.gaps[:4] if (low or check["problems"]) else [])
        if check["missing_numbers"]:
            gaps.append("Mention the window lengths / constants: " + ", ".join(check["missing_numbers"]) + ".")
        if neut and neut.upper() != "NONE" and not _mentions(text, neut):
            gaps.append(f"Explain why the signal is {neut.lower()}-neutralized.")
        ok = check["ok"] and not low
        result = {"alpha_id": alpha_id, "text": text, "ok": ok, "rounds": rnd, "checks": check,
                  "grade": {"match": grade.match, "uniqueness": grade.uniqueness, "hypothesis": grade.hypothesis,
                            "gaps": grade.gaps[:4]}, "at": datetime.now(timezone.utc).isoformat()}
        if ok:
            break
    await db.kv_set(DRAFT_KEY.format(alpha_id), result)
    return result


async def recheck_text(alpha_id: str, text: str) -> dict[str, Any]:
    """The no-AI pre-checks for text the user edited, stored back as the draft (the AI judge scores are kept, marked
    as from before the edit)."""
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
    if a is None:
        raise ValueError(f"Unknown alpha {alpha_id}")
    prev = await db.kv_get(DRAFT_KEY.format(alpha_id)) or {}
    check = check_description(text, a.code or "")
    result = {**prev, "alpha_id": alpha_id, "text": text, "checks": check, "ok": check["ok"] and bool(prev.get("grade")),
              "edited": True, "at": datetime.now(timezone.utc).isoformat()}
    if prev.get("grade"):
        g = prev["grade"]
        result["ok"] = check["ok"] and min(g["match"], g["uniqueness"], g["hypothesis"]) >= settings.aqp.min_grade
    await db.kv_set(DRAFT_KEY.format(alpha_id), result)
    return result


# ---------------------------------------------------------------- applying (description + tag, confirmed only)

async def validate_cycle(cycle: str, items: list[dict[str, str]]) -> list[str]:
    """Every reason the batch cannot be applied. Nothing is written when this is not empty."""
    cfg, errors = settings.aqp, []
    if not cycle_pattern().match(cycle or ""):
        return [f"The cycle tag must look like {cfg.tag_prefix}YYYYMM (got {cycle!r})."]
    if not items:
        return ["Pick at least one alpha."]
    ids = [i["alpha_id"] for i in items]
    if len(set(ids)) != len(ids):
        errors.append("The same alpha was picked twice.")
    ledger = await db.kv_get(LEDGER_KEY, []) or []
    async with db.session() as s:
        tagged = list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE")))
    already = sum(1 for a in tagged if cycle in (a.tags or []) and a.id not in ids)
    if already + len(set(ids)) > cfg.max_picks:
        errors.append(f"{cycle} allows {cfg.max_picks} alphas; {already} already tagged plus {len(set(ids))} new.")
    for it in items:
        async with db.session() as s:
            a = await s.get(Alpha, it["alpha_id"])
        if a is None:
            errors.append(f"{it['alpha_id']}: unknown alpha.")
            continue
        state, reason = eligibility(a, cycle, ledger)
        if state not in ("eligible", "selected"):
            errors.append(f"{a.id}: {reason}")
        chk = check_description(it.get("description") or "", a.code or "")
        errors += [f"{a.id}: {p}" for p in chk["problems"]]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sim = similarity(items[i].get("description", ""), items[j].get("description", ""))
            if sim > cfg.max_similarity:
                errors.append(f"{items[i]['alpha_id']} and {items[j]['alpha_id']}: descriptions overlap {sim:.0%} "
                              f"(at most {cfg.max_similarity:.0%}); each alpha needs its own idea.")
    return errors


async def apply_cycle(cycle: str, items: list[dict[str, str]], *, confirmed: bool) -> str:
    """Write each description and add the cycle tag. `confirmed` must come from the user's click."""
    if not confirmed:
        raise PermissionError("Tagging and describing requires explicit user confirmation.")
    for it in items:                                   # fresh tags and status from BRAIN before judging anything
        await actions.refresh_alpha(it["alpha_id"])
    errors = await validate_cycle(cycle, items)
    if errors:
        raise RuntimeError("Nothing was written. " + " ".join(errors[:6]))
    done: list[str] = []
    for it in items:
        aid = it["alpha_id"]
        async with db.session() as s:
            a = await s.get(Alpha, aid)
        await db.kv_set(BACKUP_KEY.format(aid), {"description": a.description, "tags": list(a.tags or []),
                                                  "at": datetime.now(timezone.utc).isoformat()})
        tags = list(a.tags or [])
        if cycle not in tags:
            tags.append(cycle)
        try:
            await actions.set_properties(aid, tags=tags, regular_description=it["description"])
        except Exception as exc:
            msg = getattr(exc, "message", None) or str(exc)
            written = f" Already written: {', '.join(done)}." if done else " Nothing was written."
            bus.publish("sync", f"AQP {cycle}: writing {aid} failed: {msg}.{written}", "error")
            raise RuntimeError(f"Writing {aid} to BRAIN failed: {msg}.{written} Retry when BRAIN answers again; the "
                               "alphas already written stay tagged.") from exc
        ledger = await db.kv_get(LEDGER_KEY, []) or []
        await db.kv_set(LEDGER_KEY, [*ledger, {"cycle": cycle, "alpha_id": aid,
                                                "ts": datetime.now(timezone.utc).isoformat()}])
        done.append(aid)
    bus.publish("sync", f"AQP {cycle}: described and tagged {', '.join(done)}.", "success")
    return f"{cycle}: described and tagged {len(done)} alpha(s): {', '.join(done)}"


async def untag_cycle(cycle: str, alpha_ids: list[str], *, confirmed: bool) -> str:
    """Remove the current cycle's tag (the description stays). Only that cycle's tag is ever removed."""
    if not confirmed:
        raise PermissionError("Untagging requires explicit user confirmation.")
    if not cycle_pattern().match(cycle or ""):
        raise ValueError(f"The cycle tag must look like {settings.aqp.tag_prefix}YYYYMM.")
    for aid in alpha_ids:
        a = await actions.refresh_alpha(aid)
        await actions.set_properties(aid, tags=[t for t in (a.tags or []) if t != cycle])
    ledger = await db.kv_get(LEDGER_KEY, []) or []
    await db.kv_set(LEDGER_KEY, [e for e in ledger if not (e.get("cycle") == cycle and e.get("alpha_id") in alpha_ids)])
    return f"Removed {cycle} from {len(alpha_ids)} alpha(s)."


# ---------------------------------------------------------------- jobs

@jobs.register("aqp_describe")
async def _job_describe(ctx: JobContext, params: dict) -> str:
    ids = list(params.get("alpha_ids") or [])
    ok = 0
    for i, aid in enumerate(ids, 1):
        await ctx.update((i - 1) / max(1, len(ids)), f"Writing the description of {aid} ({i}/{len(ids)})")
        try:
            ok += bool((await write_description(aid, others=ids))["ok"])
        except Exception as exc:
            log.warning("AQP description for %s failed: %s", aid, exc)
            bus.publish("sync", f"AQP description for {aid} failed: {getattr(exc, 'message', exc)}", "warning")
    await ctx.update(1.0)
    return f"{ok}/{len(ids)} descriptions passed every pre-check"


@jobs.register("aqp_apply")
async def _job_apply(ctx: JobContext, params: dict) -> str:
    return await apply_cycle(str(params["cycle"]), list(params.get("items") or []),
                             confirmed=bool(params.get("confirmed")))


@jobs.register("aqp_untag")
async def _job_untag(ctx: JobContext, params: dict) -> str:
    return await untag_cycle(str(params["cycle"]), list(params.get("alpha_ids") or []),
                             confirmed=bool(params.get("confirmed")))
