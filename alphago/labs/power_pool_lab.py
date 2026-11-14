"""LLM Power Pool Lab: an LLM writes Power Pool Alphas for one dataset at a time (ported from alpha-harness
labs/power_pool.py and api/power_pool_lab.py).

While a task runs, a background call asks the chosen model for 20 expressions. Each is checked offline (operators
and fields against the market's catalogue, at most 8 operators and 3 data fields counted the way BRAIN counts them,
at least one field of the dataset, VECTOR fields straight inside a vec_ operator), given a random universe,
neutralization and decay, and parked as a trial until cores are free. A rejected draft is recorded as a PRUNED
trial with the reason, so the next call can be shown what was thrown away and why.

The prompt is laid out for the providers' prompt caches: what every call of the task repeats comes first, then the
dataset and its slice of fields (rotating through a large dataset at fixed places), and what changes on every call
(the task's memory of the dataset) last.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, model_validator
from pydantic import Field as PydField
from sqlalchemy import func, select

from ..brain.platform import (
    DEFAULT_TRUNCATION,
    GROUPING_FIELDS,
    POWER_POOL_MAX_FIELDS,
    POWER_POOL_MAX_OPERATORS,
    REGION_AGNOSTIC,
)
from ..db import db
from ..db.models import Alpha, DataField, Dataset, Operator
from ..db.models_labs import LabTask, LabTrial
from .fastexpr import data_fields, operator_count, operators_in, render
from .runner import FAILURE, Lab, RefillContext, TrialSpec

log = logging.getLogger(__name__)

PER_CALL = 20
KEEP_PER_CALL = 40                       # a model that writes more than asked: the first 40 are kept
FIELDS_PER_CALL = 200
MAX_DATASETS = 50
MAX_HYPOTHESIS = 2000
CHARS_PER_TOKEN = 4
DEFAULT_PROMPT_TOKENS = 40_000           # alpha-harness' default Max Prompt Tokens
OUTPUT_RESERVE = 8_000                   # a model's context also holds the answer
DECAYS = (0, 3, 5, 7, 10)
TRUNCATION = DEFAULT_TRUNCATION
FAILURE_RETRY_S = 60.0
#: Price and volume basics every market has, offered beside the chosen datasets' own fields.
DATA_FIELDS = ("close", "open", "high", "low", "vwap", "volume", "adv20", "returns", "cap")
#: alpha-harness' order of the grouping fields (the prompt lists them in it).
GROUP_ORDER = ("market", "sector", "industry", "subindustry", "country", "exchange", "currency", "split", "adjfactor")
#: Grouping fields every market carries, offered when the catalogue holds none of them as rows.
STANDARD_GROUPS = ("market", "sector", "industry", "subindustry")
NO_NEUTRALIZATION = "Choose at least one Neutralization."
NO_SIMULATIONS = "Assign the simulations for this task."
OPERATORS_UNREAD = "Your BRAIN operators are not synced yet. Sync them on the Sync page, then try again."
HOLD_FIELD_TYPES = ("MATRIX", "VECTOR", "GROUP")
FENCE = re.compile(r"```(?:ya?ml|json)?\s*\n(.*?)```", re.DOTALL)

# The LLM Power Pool Lab system prompt, word for word as alpha-harness sends it (llm/prompts.py POWER_POOL_LAB).
POWER_POOL_LAB = """\
You write Alphas for WorldQuant BRAIN in Fast Expression, and every one must be a Power Pool Alpha.

HOW BRAIN READS AN ALPHA
- An Alpha is one expression, evaluated each day for every stock in a universe. Its value is a \
weight: BRAIN goes long stocks with high values and short stocks with low values. You write only \
the expression; universe, neutralization, decay and truncation are set for you.
- MATRIX field: one value per stock per day. It can go into any operator.
- VECTOR field: several values per stock per day. Put it straight inside a vec_ operator, \
e.g. vec_avg(field).
- GROUP field: the group each stock belongs to. Use it as the group input of a group_ operator, \
e.g. group_rank(x, industry).
- Coverage is the share of stocks that have a value. Fill gaps of a low-coverage field with \
ts_backfill(field, d).
- Lookbacks are trading days: 5 a week, 20 a month, 60 a quarter, 120 half a year, 252 a year.

FAST EXPRESSION
- Operators are called as name(x, d); options go by name, e.g. winsorize(x, std=4).
- Infix forms: + - * / ^, comparisons < <= > >= == !=, logic && || !, and cond ? a : b.
- Use only the operators and data fields listed in the message, spelled exactly. No comments.

POWER POOL RULES
1. At most 8 operators. Count every operator call, every + - * / ^, every comparison, every \
&& || !, every ?:, and every minus sign in front of something. A negative number such as -1 is \
not an operator. Repeats count each time. ts_backfill and group_backfill are not counted.
     rank(-ts_delta(close, 5))                          counts 3
     group_rank(ts_backfill(x, 60) / cap, industry)     counts 2
     trade_when(volume > adv20, rank(vec_avg(z)), -1)   counts 4
2. At most 3 different data fields, and at least 1 from the dataset in the message. The grouping \
fields country, industry, subindustry, currency, market, sector, exchange, split and adjfactor \
are not counted; \
every other field is, price and volume fields included.
An Alpha that breaks a rule is thrown away before it is simulated.

WHAT TO WRITE
- Explore: use fields not used yet and spread the 20 Alphas across the dataset.
- Vary the idea, not only the numbers: a level or its change; against its own history (ts_rank, \
ts_zscore, ts_delta); against peers (group_rank, group_neutralize); one field against another \
(ratio, difference, ts_corr); a condition that switches a signal (trade_when, ?:); smoothing.
- One, two or three fields are all fine, and so are short Alphas.
- Your earlier Alphas on this dataset are listed with their Sharpe. Lean toward what worked, move \
away from what did not, and never write one of them again.

ANSWER
Only JSON: {"alphas": [{"expression": "..."}]} holding exactly 20 different expressions.
"""

#: What a model cannot infer from the market line when the region is ALL. The warning about cross-sectional
#: comparison is BRAIN's own: one expression is translated into four markets whose currencies, market caps and
#: face values are not on one scale.
REGION_AGNOSTIC_BRIEF = """
This expression runs in USA, Europe, Asia and Global at once, and the alpha is submittable
where two or more of them hold up. Two fields combine only where their regions overlap, so
prefer fields carried by all four, and stay inside one dataset. Comparing raw values across
stocks is unsafe here — currencies, market caps and face values differ by region — so
normalise by scale, or compare a stock against its own history with time-series operators.
""".rstrip()

#: In memory, so a restart loses at most one in-flight LLM request per task.
_calls: dict[int, asyncio.Task] = {}
_retry: dict[int, float] = {}


def runner_hooks() -> bool:
    """True when the lab runner asks a lab whether it is still busy before finishing a task (`Lab.busy`): then the
    LLM call runs in the background. Without it the call is made inside the refill, so the task never ends while
    the model is still writing."""
    return "busy" in Lab.__dict__


class Rejected(ValueError):
    """Why an expression is thrown away before it is simulated."""


# ---------------------------------------------------------------- structured answer

class PowerPoolItem(BaseModel):
    expression: str = ""


class PowerPoolBatch(BaseModel):
    """The answer asked for: {"alphas": [{"expression": ...}]} holding 20 expressions. Parsing is tolerant: a bare
    list, plain strings or extra keys are accepted, so a batch the model shaped a little differently is not lost."""

    alphas: list[PowerPoolItem] = PydField(default_factory=list,
                                           description=f"exactly {PER_CALL} different Power Pool Alpha expressions")

    @model_validator(mode="before")
    @classmethod
    def _tolerant(cls, data: Any) -> Any:
        if isinstance(data, list):
            data = {"alphas": data}
        if isinstance(data, dict) and isinstance(data.get("alphas"), list):
            items = []
            for a in data["alphas"]:
                if isinstance(a, str):
                    items.append({"expression": a})
                elif isinstance(a, dict):
                    items.append({"expression": str(a.get("expression") or "")})
            return {"alphas": items}
        return data


def parse_alphas(text: str) -> list[dict[str, Any]]:
    """Pull the candidate list out of a text answer: the JSON itself, else each fenced block; a dict holding
    "alphas" or a bare list. A batch that cannot be read is an empty list, never an exception."""
    for candidate in (text, *(m.group(1) for m in FENCE.finditer(text or ""))):
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("alphas"), list):
            return [a for a in payload["alphas"] if isinstance(a, dict) and a.get("expression")]
        if isinstance(payload, list):
            return [a for a in payload if isinstance(a, dict) and a.get("expression")]
    return []


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


# ---------------------------------------------------------------- the dataset as the prompt shows it

@dataclass(frozen=True)
class Field:
    id: str
    type: str
    coverage: float | None
    description: str
    regions: int | None          # how many regions carry it, in the region-agnostic market; None elsewhere


@dataclass(frozen=True)
class Context:
    id: str
    name: str
    category: str
    description: str
    fields: tuple[Field, ...]    # the dataset's own, most complete first
    basics: tuple[Field, ...]
    groups: tuple[str, ...]
    types: dict[str, str]
    own: frozenset[str]
    names: frozenset[str]
    held: dict[str, frozenset[str]]


@dataclass
class Run:
    """What every call of a task shares."""

    region: str
    delay: int
    universes: list[str]
    neutralizations: list[str]
    hypothesis: str = ""
    favour_operators: list[str] | None = None


def _regions(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, dict)):
        return len(value)
    return None


def _shown(f: DataField, narrow: dict[str, Any] | None) -> bool:
    """The field passes the form's field filter (search text, minimum coverage, field types)."""
    if not narrow:
        return True
    text = str(narrow.get("search") or "").strip().lower()
    if text and text not in f.id.lower() and text not in (f.description or "").lower():
        return False
    floor = narrow.get("min_coverage")
    if floor not in (None, "") and (f.coverage or 0) < float(floor):
        return False
    types = narrow.get("types") or []
    return not types or str(f.type).upper() in {str(t).upper() for t in types}


async def context_for(region: str, delay: int, universes: list[str], dataset: str,
                      narrow: dict[str, Any] | None = None) -> Context | None:
    """The dataset as one call sees it: its own fields across the task's universes (each held per universe, the
    first universe's row describing it), the price/volume basics and the grouping fields; None when the dataset is
    not in the downloaded catalogue of the market."""
    if not universes:
        return None
    extra = (*DATA_FIELDS, *GROUPING_FIELDS)
    async with db.session() as s:
        rows = list(await s.scalars(select(DataField).where(
            DataField.region == region, DataField.delay == int(delay), DataField.universe.in_(universes),
            DataField.type.in_(HOLD_FIELD_TYPES),
            (DataField.dataset_id == dataset) | DataField.id.in_(extra))))
        meta = (await s.scalars(select(Dataset).where(Dataset.id == dataset, Dataset.region == region,
                                                      Dataset.delay == int(delay)).limit(1))).first()
    if not any(r.dataset_id == dataset for r in rows):
        return None
    held: dict[str, set[str]] = {}
    info: dict[str, DataField] = {}
    rank = {u: i for i, u in enumerate(universes)}
    for r in rows:
        held.setdefault(r.id, set()).add(r.universe)
        best = info.get(r.id)
        if best is None or rank.get(r.universe, 99) < rank.get(best.universe, 99):
            info[r.id] = r
    own = {f for f, r in info.items() if r.dataset_id == dataset}
    if narrow:
        own = {f for f in own if f in GROUPING_FIELDS or _shown(info[f], narrow)}

    def field(f: str) -> Field:
        r = info[f]
        return Field(f, str(r.type or "MATRIX").upper(), r.coverage, str(r.description or "")[:160],
                     _regions(r.region_coverage) if region == REGION_AGNOSTIC else None)

    # ties broken by name: a set's order changes between calls and restarts, which would move fields between the
    # slices `offset` walks and break the prompt's cached prefix
    fields = sorted((field(f) for f in own if f not in GROUPING_FIELDS), key=lambda x: (-(x.coverage or 0), x.id))
    groups = tuple(g for g in GROUP_ORDER if g in info) or STANDARD_GROUPS
    types = {f: str(r.type or "MATRIX").upper() for f, r in info.items()}
    for g in groups:
        types.setdefault(g, "GROUP")
    category = " › ".join(str(c) for c in ((meta.category_name, meta.subcategory) if meta else ()) if c)
    return Context(
        id=dataset,
        name=str((meta.name if meta else None) or dataset),
        category=category,
        description=str((meta.description if meta else None) or "")[:800],
        fields=tuple(fields),
        basics=tuple(field(f) for f in DATA_FIELDS if f in info and f not in own),
        groups=groups,
        types=types,
        own=frozenset(own),
        names=frozenset(info) | frozenset(groups),
        held={f: frozenset(u) for f, u in held.items()},
    )


# ---------------------------------------------------------------- operators

def _scopes(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {s.upper() for s in re.split(r"[\s,;|]+", value) if s}
    return {str(s).upper() for s in value}


async def synced_operators() -> list[dict[str, Any]]:
    """The account's operators as last synced, each with name, category, scope, definition and description."""
    async with db.session() as s:
        rows = list(await s.scalars(select(Operator)))
    return [{"name": r.name, "category": r.category, "scope": r.scope, "definition": r.definition,
             "description": r.description} for r in rows]


def operator_table(operators: list[dict[str, Any]], scope: str = "REGULAR") -> dict[str, dict[str, Any]]:
    """Operators whose published scope includes `scope` (or that name none) and whose definition can be read, so a
    COMBO-only operator is never offered for a regular Alpha."""
    out: dict[str, dict[str, Any]] = {}
    for o in operators:
        name, definition = o.get("name"), o.get("definition")
        if not isinstance(name, str) or not isinstance(definition, str) or f"{name}(" not in definition.replace(
                " (", "("):
            continue
        scopes = _scopes(o.get("scope"))
        if scopes and scope not in scopes:
            continue
        out[name] = o
    return out


def operators_text(operators: list[dict[str, Any]]) -> str:
    table = operator_table(operators)
    by: dict[str, list[str]] = {}
    for o in operators:
        name = o.get("name")
        if name in table:
            line = f"{name}: {o.get('definition')} | {str(o.get('description') or '')[:160]}"
            by.setdefault(str(o.get("category") or "Other"), []).append(line)
    return "\n".join(f"## {c}\n" + "\n".join(lines) for c, lines in sorted(by.items()))


def _quarter_start(day: date) -> date:
    return date(day.year, (day.month - 1) // 3 * 3 + 1, 1)


async def unused_operators(today: date | None = None) -> dict[str, Any]:
    """Operators none of your Alphas submitted this quarter used: Total distinct operators over the quarter is one
    of BRAIN Genius's tiebreakers, so the lab can ask the LLM to reach for them. Read from the local alpha list (sync
    it first) against the synced operators."""
    day = today or datetime.now(timezone.utc).date()
    start = _quarter_start(day)
    listed = sorted(operator_table(await synced_operators()))
    since = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    async with db.session() as s:
        codes = list(await s.scalars(select(Alpha.code).where(Alpha.date_submitted >= since,
                                                             Alpha.code.is_not(None))))
    used: set[str] = set()
    for code in codes:
        try:
            used.update(operators_in(code))
        except Exception:                     # a SuperAlpha or Python alpha: not FASTEXPR
            continue
    unused = [o for o in listed if o not in used]
    return {"label": f"{day.year} Q{(day.month - 1) // 3 + 1}", "operators": listed, "unused": unused,
            "alphas": len(codes)}


# ---------------------------------------------------------------- checking and drawing

def _children(node: Any) -> list[Any]:
    from ..fastexpr import Call
    if isinstance(node, Call):
        return [*node.args, *(v for _, v in node.kwargs)]
    return list(node.children())


def _walk_with_parents(node: Any, parent: Any = None):  # noqa: ANN202
    yield node, parent
    for child in _children(node):
        yield from _walk_with_parents(child, node)


def catalog_for(ctx: Context, operators: list[dict[str, Any]]) -> Any:
    """AlphaGo's FASTEXPR catalogue for one call: the account's operators and every field the call may use."""
    from ..knowledge.operators import build_catalog
    fields = [{"id": f, "type": ctx.types.get(f, "MATRIX"), "dataset": {"id": ctx.id if f in ctx.own else ""}}
              for f in sorted(ctx.names)]
    return build_catalog(operators or None, fields)


def check(text: str, ctx: Context, catalog: Any) -> tuple[str, frozenset[str]]:
    """The canonical expression and the fields it uses, or `Rejected` with the reason."""
    from ..fastexpr import Assign, Call, FastExprSyntaxError, Ident, parse, validate
    try:
        program = parse(text)
    except FastExprSyntaxError as exc:
        raise Rejected(f"Could not be read: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - any parser failure is an unreadable draft
        raise Rejected(f"Could not be read: {exc}") from exc
    result = validate(program, catalog)
    if not result.ok:
        raise Rejected(result.errors[0].message)
    count = operator_count(program)
    if count > POWER_POOL_MAX_OPERATORS:
        raise Rejected(f"{count} operators; Power Pool allows {POWER_POOL_MAX_OPERATORS}.")
    used = frozenset(f for f in data_fields(program, grouping=True) if f in ctx.names)
    data = sorted(used - GROUPING_FIELDS)
    if len(data) > POWER_POOL_MAX_FIELDS:
        raise Rejected(f"{len(data)} data fields ({', '.join(data)}); Power Pool allows {POWER_POOL_MAX_FIELDS}.")
    if not used & (ctx.own - GROUPING_FIELDS):
        raise Rejected(f"Uses no field of {ctx.id}.")
    for stmt in program.statements:
        root = stmt.value if isinstance(stmt, Assign) else stmt
        for node, parent in _walk_with_parents(root):
            if (isinstance(node, Ident) and ctx.types.get(node.name) == "VECTOR"
                    and not (isinstance(parent, Call) and parent.name.startswith("vec_"))):
                raise Rejected(f"{node.name} is a VECTOR field: put it straight inside a vec_ operator.")
    return render(program), used


def draw(rng: random.Random, used: frozenset[str], ctx: Context, run: Run) -> dict[str, Any]:
    """Random settings for one Alpha: a universe that holds every field it uses, a neutralization the task chose,
    a decay from 0, 3, 5, 7, 10 and truncation 0.08."""
    fit = [u for u in run.universes if all(u in ctx.held[f] for f in used if f in ctx.held)]
    if not fit:
        raise Rejected(f"No downloaded universe has all of {', '.join(sorted(used))}.")
    return {"instrumentType": "EQUITY", "region": run.region, "delay": int(run.delay), "universe": rng.choice(fit),
            "neutralization": rng.choice(run.neutralizations), "decay": rng.choice(DECAYS),
            "truncation": TRUNCATION, "language": "FASTEXPR"}


# ---------------------------------------------------------------- the prompt

def _line(f: Field) -> str:
    coverage = "?" if f.coverage is None else f"{f.coverage * 100:.0f}%"
    # the region count only exists in the region-agnostic market, and there it decides whether two fields can
    # appear in one expression at all
    regions = f" · {f.regions}/4 regions" if f.regions is not None else ""
    return f"{f.id} · {f.type} · {coverage}{regions} · {f.description}"


def memory_text(done: list[LabTrial], waiting: list[LabTrial], thrown: list[LabTrial]) -> str:
    parts = []
    if done:
        parts.append("Simulated, best Sharpe first:\n" + "\n".join(
            f"{_sharpe(t):.2f} | {(t.expression or '')[:300]}" for t in done))
    if waiting:
        parts.append("Not simulated yet:\n" + "\n".join((t.expression or "")[:300] for t in waiting))
    if thrown:
        parts.append("Thrown away:\n" + "\n".join(
            f"{(t.expression or '')[:300]} | {(t.message or '')[:120]}" for t in thrown))
    return "\n".join(parts) or "None yet."


def _sharpe(t: LabTrial) -> float:
    sharpe = (t.figures or {}).get("sharpe")
    return float(sharpe if sharpe is not None else (t.value or 0.0))


async def memory_of(task_id: int, dataset: str) -> str:
    """What the task already wrote for `dataset`, as the few lines the prompt shows: the 20 best scored, the 15
    newest still waiting, the 5 newest thrown away. Three slices read by the database, not the whole task."""
    mine = (LabTrial.task_id == task_id, func.json_extract(LabTrial.params, "$.dataset") == dataset)
    async with db.session() as s:
        done = list(await s.scalars(select(LabTrial).where(
            *mine, LabTrial.state == "COMPLETE", LabTrial.value > FAILURE).order_by(
            LabTrial.value.desc(), LabTrial.number).limit(20)))
        waiting = list(await s.scalars(select(LabTrial).where(
            *mine, LabTrial.state.in_(("PARKED", "RUNNING"))).order_by(LabTrial.number.desc()).limit(15)))[::-1]
        thrown = list(await s.scalars(select(LabTrial).where(
            *mine, func.json_extract(LabTrial.params, "$.rejected") == 1).order_by(
            LabTrial.number.desc()).limit(5)))[::-1]
    return memory_text(done, waiting, thrown)


def user_prompt(ctx: Context, operators: list[dict[str, Any]], run: Run, memory: str, offset: int,
                budget: int) -> tuple[str, int]:
    """The user turn, and how many field lines it shows.

    Laid out for the providers' prompt caches, which reuse the longest start a recent request shared: what every
    call of the task repeats comes first, then the dataset and its slice of fields, and what changes on every call
    last."""
    head = "\n\n".join([
        f"MARKET\n{run.region} · Delay {run.delay} · Universes {', '.join(run.universes)}"
        + (REGION_AGNOSTIC_BRIEF if run.region == REGION_AGNOSTIC else ""),
        "OPERATORS\n" + operators_text(operators),
        *(["OPERATORS NOT USED THIS QUARTER · use them where they fit the idea; a sound Alpha comes first\n"
           + ", ".join(run.favour_operators)] if run.favour_operators else []),
        "PRICE AND VOLUME FIELDS · count as data fields\n" + "\n".join(_line(f) for f in ctx.basics),
        "GROUPING_FIELDS FIELDS · not counted\n" + ", ".join(ctx.groups),
        f"DATASET\n{ctx.id} · {ctx.name} · {ctx.category}\n{ctx.description}",
    ])
    tail = "\n\n".join([
        f"YOUR EARLIER ALPHAS ON {ctx.id}\n{memory}",
        *(["HYPOTHESIS TO TEST\n" + run.hypothesis,
           f"Write {PER_CALL} new Power Pool Alphas that use {ctx.id} to test this hypothesis: each one a different "
           "way of measuring it with these fields, or of trading on it. Where the dataset cannot express it, stay as "
           "close to it as the fields allow."] if run.hypothesis
          else [f"Write {PER_CALL} new Power Pool Alphas that use {ctx.id}."]),
    ])
    room = budget * CHARS_PER_TOKEN - len(POWER_POOL_LAB) - len(head) - len(tail) - 200
    total = len(ctx.fields)
    start = offset % total if total else 0
    ordered = ctx.fields[start:] + ctx.fields[:start]
    lines: list[str] = []
    for f in ordered[:FIELDS_PER_CALL]:
        line = _line(f)
        if room - len(line) < 0:
            break
        room -= len(line) + 1
        lines.append(line)
    title = f"FIELDS OF {ctx.id} · {start + 1}-{start + len(lines)} of {total:,}, most complete first"
    return f"{head}\n\n{title}\n" + "\n".join(lines) + f"\n\n{tail}", len(lines)


def next_offset(offset: int, shown: int, total: int) -> int:
    """Where the next call's fields start: after this call's, or back at the first once the whole dataset has been
    shown. Every pass then splits it at the same places, so a slice that comes round again can still be cached."""
    end = (offset % total if total else 0) + shown
    return end if end < total else 0


# ---------------------------------------------------------------- models (AlphaGo's providers)

def _providers() -> dict[str, Any]:
    from ..llm.router import PROVIDERS
    return PROVIDERS


def model_options() -> list[dict[str, Any]]:
    """Every model a set-up provider offers, as `provider:model` refs, the primary provider's pick first."""
    from ..llm.router import PROVIDER_LABELS, llm
    out: list[dict[str, Any]] = []
    chain = [p.name for p in llm.chain()]
    names = list(dict.fromkeys([*chain, *_providers()]))
    for name in names:
        provider = _providers().get(name)
        try:
            if provider is None or not provider.configure():
                continue
            pick = provider.pick("ideation") or ""
        except Exception:
            continue
        models = [str(m.get("name")) for m in (provider.models or []) if m.get("name")]
        for model in dict.fromkeys([m for m in [pick, *models] if m]):
            out.append({"ref": f"{name}:{model}", "provider": name, "model": model,
                        "label": f"{model} · {PROVIDER_LABELS.get(name, name)}",
                        "local": bool(getattr(provider, "local", False))})
    return out


def split_ref(ref: str | None) -> tuple[str, str]:
    provider, _, model = str(ref or "").partition(":")
    return provider, model


def prompt_tokens(provider_name: str, model: str) -> int:
    """How much prompt a model takes: alpha-harness' 40,000-token default, less when the model's own context is
    smaller (keeping room for the answer)."""
    provider = _providers().get(provider_name)
    limit = None
    for m in (getattr(provider, "models", None) or []):
        if m.get("name") == model:
            limit = m.get("input_limit") or m.get("context")
            break
    if provider_name == "ollama":
        try:
            limit = int(provider.cfg.num_ctx or 32_768)
        except Exception:
            limit = 32_768
    if limit:
        return max(4_000, min(DEFAULT_PROMPT_TOKENS, int(limit) - OUTPUT_RESERVE))
    return DEFAULT_PROMPT_TOKENS


def model_order(model_ref: str, only_model: bool) -> list[tuple[str, str]]:
    """The task's model first; unless only that one may answer, then each other set-up provider's own pick, in the
    order of the LLM Integration page's fallbacks."""
    from ..llm.router import llm
    order = [split_ref(model_ref)]
    if not only_model:
        for p in llm.chain():
            try:
                if p.configure() and p.pick("ideation"):
                    order.append((p.name, p.pick("ideation")))
            except Exception:
                continue
    out, seen = [], set()
    for provider, model in order:
        if provider in _providers() and model and (provider, model) not in seen:
            seen.add((provider, model))
            out.append((provider, model))
    return out


def seconds_to_budget_reset(now: datetime | None = None) -> float:
    """AlphaGo's daily LLM call budget resets at midnight Pacific."""
    from ..llm.base import PACIFIC
    now = (now or datetime.now(timezone.utc)).astimezone(PACIFIC)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(5.0, (midnight - now).total_seconds())


async def ask(provider_name: str, model: str, system: str, user: str) -> list[dict[str, Any]]:
    """One structured call on one provider and model, without the router's failover (the lab chooses the next
    model itself, and an "only this model" task must never move)."""
    from ..llm.base import MODEL_OVERRIDE
    provider = _providers()[provider_name]
    token = MODEL_OVERRIDE.set(model)
    try:
        batch = await provider.generate_json(user, PowerPoolBatch, system=system, role="ideation",
                                             purpose="power_pool_lab")
    finally:
        MODEL_OVERRIDE.reset(token)
    return [{"expression": a.expression} for a in batch.alphas if a.expression.strip()]


# ---------------------------------------------------------------- planning (form preview and task plan)

async def synced_universes(region: str, delay: int, first: str | None = None) -> list[str]:
    """The market's universes whose fields are downloaded, in BRAIN's order, the chosen one first."""
    from ..brain import brain
    async with db.session() as s:
        synced = set(await s.scalars(select(DataField.universe).where(
            DataField.region == region, DataField.delay == int(delay)).distinct()))
    legal = list(brain.options.universes(region))
    out = [u for u in legal if u in synced] + sorted(u for u in synced if u not in legal)
    if first in out:
        out.remove(first)
        out.insert(0, first)
    return out


async def datasets_of(region: str, delay: int) -> list[dict[str, Any]]:
    """Downloaded datasets of a market, for the form's picker."""
    async with db.session() as s:
        rows = list(await s.scalars(select(Dataset).where(Dataset.region == region, Dataset.delay == int(delay))
                                    .order_by(Dataset.id)))
    out: dict[str, dict[str, Any]] = {}
    for d in rows:
        out.setdefault(d.id, {"id": d.id, "name": d.name or d.id, "category": d.category_name or "",
                              "fields": d.field_count})
    return list(out.values())


async def budget_left() -> int:
    from ..config import settings
    from ..llm.base import calls_today
    return max(0, int(settings.llm.daily_call_budget) - int(await calls_today()))


async def plan_task(params: dict[str, Any]) -> dict[str, Any]:
    """Everything a task with these inputs would send, without calling the LLM or simulating: the problems that
    refuse it, warnings, the resolved universes and neutralizations, field and call counts and the exact first
    prompt with a token estimate."""
    from ..brain import brain
    problems: list[str] = []
    warnings: list[str] = []
    region = str(params.get("region") or "USA")
    delay = int(params.get("delay") if params.get("delay") is not None else 1)
    universe = params.get("universe")
    dataset_ids = list(dict.fromkeys(str(d) for d in params.get("dataset_ids") or []))
    hypothesis = str(params.get("hypothesis") or "").strip()
    simulations = int(params.get("simulations") or 0)
    operators = await synced_operators()
    listed = operator_table(operators)
    if not listed:
        problems.append(OPERATORS_UNREAD)
    favour = [o for o in dict.fromkeys(params.get("favour_operators") or []) if o in listed]
    if not dataset_ids:
        problems.append("Choose at least one dataset.")
    elif len(dataset_ids) > MAX_DATASETS:
        problems.append(f"Choose at most {MAX_DATASETS} datasets; {len(dataset_ids)} are chosen.")
    if len(hypothesis) > MAX_HYPOTHESIS:
        problems.append(f"The hypothesis is {len(hypothesis):,} characters; keep it to {MAX_HYPOTHESIS:,}.")

    options = {m["ref"]: m for m in model_options()}
    ref = str(params.get("model") or next(iter(options), ""))
    provider_name, model = split_ref(ref)
    if not ref:
        problems.append("No model is set up. Set one up on the LLM Integration page.")
    elif ref not in options and not (provider_name in _providers() and model and _configured(provider_name)):
        problems.append(f"{ref} is not set up. Set it up on the LLM Integration page, or choose another model.")

    universes = await synced_universes(region, delay, universe)
    chosen = [str(u) for u in params.get("universes") or []]
    if chosen and universes:
        universes = [u for u in universes if u in set(chosen)]
        if not universes:
            problems.append(f"None of the chosen universes is downloaded for {region} delay {delay}. Sync it on "
                            "the Sync page, or choose another.")
    elif not universes:
        problems.append(f"No {region} delay {delay} market is downloaded. Sync its datasets on the Sync page.")
    offered = [str(n) for n in brain.options.neutralizations(region) if str(n) != "NONE"]
    wanted = {str(n) for n in params.get("neutralizations") or []}
    neutralizations = [n for n in offered if n in wanted]
    if not wanted:
        problems.append(NO_NEUTRALIZATION)
    elif not offered:
        problems.append("BRAIN's settings list is not loaded. Sign in again.")
    elif not neutralizations:
        problems.append(f"BRAIN offers none of the chosen neutralizations in {region}.")

    run = Run(region, delay, universes, neutralizations, hypothesis, favour)
    fields = 0
    prompt = None
    narrow = params.get("field_filter") or None
    if universes:
        for dataset in dataset_ids[:MAX_DATASETS]:
            ctx = await context_for(region, delay, universes, dataset, narrow)
            if ctx is None:
                problems.append(f"{dataset} is not in the downloaded {region} delay {delay} catalogue.")
                continue
            if narrow and not ctx.fields:
                problems.append(f"No field in {dataset} matches the field filter. Untick it, or loosen the filter.")
            fields += len(ctx.fields)
            if prompt is None and listed and model:
                budget = prompt_tokens(provider_name, model)
                user, shown = user_prompt(ctx, operators, run, "None yet.", 0, budget)
                prompt = {"system": POWER_POOL_LAB, "user": user,
                          "tokens": estimate_tokens(POWER_POOL_LAB + user), "fields": shown, "budget": budget}
                if ctx.fields and shown < min(10, len(ctx.fields)):
                    problems.append(f"Only {shown} of {dataset}'s fields fit in {model}'s {budget:,} prompt tokens, "
                                    "too few to work with. Choose a model with a longer context, or another "
                                    "dataset.")
    calls = math.ceil(simulations / PER_CALL) if simulations > 0 else 0
    local = bool(getattr(_providers().get(provider_name), "local", False))
    if calls and not local:
        left = await budget_left()
        if calls > left:
            warnings.append(f"About {calls:,} LLM calls; today's LLM budget has {left:,} left, so the task waits "
                            "for the reset at midnight Pacific (raise the budget on the LLM Integration page).")
    return {"problems": problems, "warnings": warnings, "sample": [], "fields": fields, "universes": universes,
            "neutralizations": neutralizations, "llmCalls": calls, "prompt": prompt, "model": ref,
            "favourOperators": favour, "datasets": dataset_ids}


def _configured(provider_name: str) -> bool:
    try:
        return bool(_providers()[provider_name].configure())
    except Exception:
        return False


# ---------------------------------------------------------------- the lab

def call_cap(max_trials: int) -> int:
    return max(3, 2 * math.ceil(max(1, int(max_trials)) / PER_CALL))


def stop_reason(task: LabTask) -> str | None:
    """Why the task writes no more: the last 3 calls wrote nothing valid, or it made its share of calls."""
    tally = (task.state or {}).get("llm") or {}
    cap = call_cap(task.max_trials)
    if int(tally.get("empty") or 0) >= 3:
        return "the last 3 LLM calls wrote no new valid Alpha"
    if int(tally.get("calls") or 0) >= cap:
        return f"it reached {cap} LLM calls"
    return None


def calling(task_id: int) -> bool:
    call = _calls.get(task_id)
    return call is not None and not call.done()


class PowerPoolLab(Lab):
    name = "power_pool"
    title = "LLM Power Pool Lab"
    objective = "sharpe"

    async def preview(self, params: dict[str, Any]) -> dict[str, Any]:
        return await plan_task(params)

    async def plan(self, task: LabTask) -> dict[str, Any]:
        params = dict(task.params or {})
        params.setdefault("simulations", task.max_trials)
        if int(params.get("simulations") or 0) < 1:
            raise ValueError(NO_SIMULATIONS)
        plan = await plan_task(params)
        if plan["problems"]:
            raise ValueError(plan["problems"][0])
        task.state = {"llm": {"calls": 0}, "calls": []}
        task.universe = plan["universes"][0]
        return {"universes": plan["universes"], "neutralizations": plan["neutralizations"],
                "dataset_ids": plan["datasets"][:MAX_DATASETS], "model": plan["model"],
                "only_model": bool(params.get("only_model")), "hypothesis": str(params.get("hypothesis") or "").strip(),
                "favour_operators": plan["favourOperators"], "field_filter": params.get("field_filter") or None}

    def done_early(self, task: LabTask, ctx: RefillContext) -> str | None:
        stop = stop_reason(task)
        if stop and not calling(task.id) and not any(v is True for v in ctx.seen.values()):
            return f"Stopped: {stop}."
        return None

    async def busy(self, task: LabTask) -> bool:
        """Still writing: a call is out, or a new one comes once its wait (budget, a busy provider) is over."""
        return calling(task.id) or stop_reason(task) is None

    async def refill(self, task: LabTask, want: int, ctx: RefillContext) -> list[TrialSpec]:
        """Never calls the LLM in the runner's way: a background call writes the next 20 while cores work, and the
        trials it parks are sent on the next tick. (Without the runner's busy hook the call is made here, so the
        task cannot end while the model is still writing.)"""
        if stop_reason(task) or calling(task.id) or time.monotonic() < _retry.get(task.id, 0.0):
            return []
        if not runner_hooks():
            return await write(task.id, store=False)
        _calls[task.id] = asyncio.get_running_loop().create_task(write(task.id), name=f"power-pool-{task.id}")
        return []


# ---------------------------------------------------------------- one LLM call

async def _pause(task_id: int, message: str) -> None:
    from . import runner as runner_mod
    try:
        await runner_mod.runner.pause(task_id)
    finally:
        await _note(task_id, message)


async def _note(task_id: int, message: str | None) -> None:
    async with db.write() as s:
        row = await s.get(LabTask, task_id)
        if row is not None:
            row.message = message


async def write(task_id: int, *, store: bool = True) -> list[TrialSpec]:
    """Ask the model for 20 Alphas on the dataset with the fewest calls, check each, and park the valid ones
    (rejected ones as PRUNED with the reason). Stores them itself in the background; with `store=False` returns
    them for the runner to store. Never raises: a failure is logged in the task's call log."""
    try:
        return await _write(task_id, store)
    except Exception:
        log.exception("power pool call failed for task %s", task_id)
        _retry[task_id] = time.monotonic() + FAILURE_RETRY_S
        return []


async def _write(task_id: int, store: bool) -> list[TrialSpec]:
    from ..llm.base import LLMError
    from ..llm.router import failover_worthy
    from . import runner as runner_mod
    async with db.session() as s:
        task = await s.get(LabTask, task_id)
    if task is None or task.status != "RUNNING":
        return []
    space = task.space or {}
    run = Run(task.region or "USA", int(task.delay or 0), list(space.get("universes") or []),
              list(space.get("neutralizations") or []), str(space.get("hypothesis") or ""),
              list(space.get("favour_operators") or []))
    tally = dict((task.state or {}).get("llm") or {})
    by = dict(tally.get("byDataset") or {})
    ids = list(space.get("dataset_ids") or [])
    if not ids:
        await _pause(task_id, "This task has no dataset to write for. Add it again with a dataset.")
        return []
    dataset = min(ids, key=lambda d: (int((by.get(d) or {}).get("calls", 0)), ids.index(d)))
    models = model_order(str(space.get("model") or ""), bool(space.get("only_model")))
    operators = await synced_operators()
    ctx = await context_for(run.region, run.delay, run.universes, dataset, space.get("field_filter"))
    if not models:
        await _pause(task_id, f"{space.get('model')} is no longer set up. Set it up again on the LLM Integration "
                              "page, then resume.")
        return []
    if not operator_table(operators):
        await _pause(task_id, "Your BRAIN operators could not be read. Sync them on the Sync page, then resume.")
        return []
    if ctx is None:
        await _pause(task_id, f"{dataset} is not in the downloaded catalogue. Sync it, then resume.")
        return []

    memory = await memory_of(task_id, dataset)
    offset = int((by.get(dataset) or {}).get("offset", 0))
    items: list[dict[str, Any]] = []
    answered = False
    failing: str | None = None
    spent = 0
    down: LLMError | None = None
    entry: dict[str, Any] = {}
    used_model = models[0][1]
    shown = 0
    for provider_name, model in models:
        used_model = model
        user, shown = user_prompt(ctx, operators, run, memory, offset, prompt_tokens(provider_name, model))
        entry = {"at": datetime.now(timezone.utc).isoformat(), "dataset": dataset, "model": f"{provider_name}:{model}",
                 "fields": shown, "tokens": estimate_tokens(POWER_POOL_LAB + user)}
        try:
            items = (await ask(provider_name, model, POWER_POOL_LAB, user))[:KEEP_PER_CALL]
        except LLMError as exc:
            text = (exc.message or "").lower()
            if "budget" in text:                      # AlphaGo's shared daily call budget: every cloud model
                spent += 1
                continue
            if "malformed json" in text:              # answered, but not the JSON asked for
                answered = True
                entry["error"] = "The answer was not the JSON asked for: " + exc.message[:200]
                break
            if failover_worthy(exc):                  # rate-limited, overloaded, timed out, model unavailable
                down = exc
                continue
            _retry[task_id] = time.monotonic() + FAILURE_RETRY_S
            entry["error"] = exc.message[:300]
            tally["failed"] = int(tally.get("failed") or 0) + 1
            if tally["failed"] >= 3:
                tally["failed"] = 0                   # saved as none, so a resumed task counts afresh
                failing = f"The LLM failed 3 times in a row: {exc.message[:300]}"
            break
        answered = True
        if not items:
            entry["error"] = "The answer held no expressions."
        break
    else:
        if down is None:
            # every model is out of budget: wait for the reset
            wait = seconds_to_budget_reset()
            _retry[task_id] = time.monotonic() + wait
            await _note(task_id, "Waiting for LLM budget: today's AlphaGo LLM call budget is used up. The task "
                                 "carries on after the reset at midnight Pacific (or raise the budget on the LLM "
                                 "Integration page).")
            return []
        # a busy provider passes: tried again in a minute, never counted toward a pause
        _retry[task_id] = time.monotonic() + FAILURE_RETRY_S
        entry["error"] = down.message[:300]

    catalog = catalog_for(ctx, operators)
    rng = random.Random()
    async with db.session() as s:
        seen = set(await s.scalars(select(LabTrial.expression).where(LabTrial.task_id == task_id)))
    specs: list[TrialSpec] = []
    valid = rejected = 0
    market = {"region": run.region, "delay": run.delay, "universe": (run.universes or ["TOP3000"])[0]}
    for item in items:
        text = str(item.get("expression") or "").strip()
        try:
            expression, used = check(text, ctx, catalog)
            if expression in seen:
                raise Rejected("Already written in this task.")
            settings = draw(rng, used, ctx, run)
            seen.add(expression)
            valid += 1
            specs.append(TrialSpec(expression, settings, params={"dataset": dataset, "model": used_model}))
        except Rejected as exc:
            rejected += 1
            specs.append(TrialSpec(text[:2000] or "(empty)", market, state="PRUNED", message=str(exc)[:500],
                                   params={"dataset": dataset, "model": used_model, "rejected": True}))
    if store and specs:
        await runner_mod.runner.add_trials(task_id, specs)

    async with db.write() as s:
        row = await s.get(LabTask, task_id)
        if row is None:
            return []
        state = dict(row.state or {})
        fresh = dict(state.get("llm") or {})
        fresh["failed"] = 0 if items else tally.get("failed", fresh.get("failed", 0))
        # a call that failed showed the model nothing: the same fields go out again next time, and it counts
        # toward neither the call cap nor the empty-answer stop
        if answered:
            fresh["calls"] = int(fresh.get("calls") or 0) + 1
            fresh["empty"] = 0 if valid else int(fresh.get("empty") or 0) + 1
            per = dict(fresh.get("byDataset") or {})
            mine = dict(per.get(dataset) or {})
            mine["calls"] = int(mine.get("calls") or 0) + 1
            mine["offset"] = next_offset(offset, shown, len(ctx.fields))
            per[dataset] = mine
            fresh["byDataset"] = per
            models_tally = dict(fresh.get("byModel") or {})
            models_tally[used_model] = int(models_tally.get(used_model) or 0) + 1
            fresh["byModel"] = models_tally
            fresh["valid"] = int(fresh.get("valid") or 0) + valid
            fresh["rejected"] = int(fresh.get("rejected") or 0) + rejected
        entry |= {"valid": valid, "rejected": rejected}
        state["llm"] = fresh
        state["calls"] = [*(state.get("calls") or []), entry][-100:]
        row.state = state
        if valid and row.message and row.message.startswith("Waiting for LLM budget"):
            row.message = None
    if failing:
        if store:
            await _pause(task_id, failing)
        else:
            await _note(task_id, failing)
    return [] if store else specs
