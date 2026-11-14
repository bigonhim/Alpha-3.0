"""Fields Sampler: one Alpha re-run with each of its data fields swapped for others (alpha-harness
tools/fields_sampler).

One field changes at a time and everything else holds, so a search that would pair every field with every other is a
line through each field instead. The others are picked by mode:

- **Sweep**: every other field of the same dataset (and type), best covered first.
- **Similar**: the fields whose descriptions read closest to this one's — the closest k from the same dataset, then
  the same subcategory, the same category and the whole market.
- **Buckets**: for an Alpha reading several fields of one dataset, every field of it goes to the bucket of the seed it
  is most like (`field_buckets`), and every field is swapped at once, one from each bucket's top picks (a grid).

In USA D1 a field's Fast D1 twin and the overnight delta (`x_fast_d1 - x`) lead every list: BRAIN takes a Fast D1
Alpha only if it beats its regular one, and suggests the delta for a lower Production Correlation. Dates and symbol
identifiers are never offered. In Sweep and Similar a variant nothing refused becomes a seed itself and is swept the
same way (a tree), until the task's simulations run out.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import product, zip_longest
from typing import Any

from sqlalchemy import func, select

from ..brain.platform import FAST_D1_MARKET, FAST_D1_SUFFIX, MAX_SIMULATIONS, MULTI_SIMULATION_SIZE, brain_verdict
from ..db import db
from ..db.models import Alpha, DataField
from ..db.models_labs import LabTask, LabTrial
from ..labs.runner import Lab, RefillContext, TrialSpec, trial_key
from .common import ToolError, alpha_body, alpha_id_of, code_of, fields_at, fields_in_order, held_settings
from .field_buckets import FieldText, split
from .settings_sampler import all_run, written_up_front

NAME = "fields_sampler"
SWEEP, SIMILAR, BUCKETS = "sweep", "similar", "buckets"
MODES = (SWEEP, SIMILAR, BUCKETS)
TIERS = ("dataset", "subcategory", "category", "market")
FAST_D1_TIER = "fast_d1"
DEFAULT_K = 10

_TECHNICAL = frozenset({"date", "dates", "timestamp", "isin", "cusip", "sedol", "ticker"})
_WORDS = re.compile(r"[^0-9a-z]+")
_TERMS = re.compile(r"[a-z]+|\d+")
_STOP = frozenset({"the", "of", "a", "an", "and", "or", "for", "in", "on", "to", "by", "is", "as", "at", "with", "from",
                   "this", "that", "per", "its", "be", "are", "was", "it", "which", "value", "data", "field"})


@dataclass(frozen=True, slots=True)
class Candidate:
    field: str
    dataset: str
    tier: str              # Similar's tier, "dataset" for Sweep, "bucket" for Buckets, "fast_d1" for the twins
    coverage: float | None


@dataclass(frozen=True, slots=True)
class Variant:
    expression: str
    settings: dict[str, Any]
    field: str
    swap: str
    tier: str

    def request(self, parent: int | None = None, generation: int = 1) -> dict[str, Any]:
        return {"expression": self.expression, "settings": self.settings, "generation": generation,
                "params": {"parent": parent, "field": self.field, "swap": self.swap, "tier": self.tier}}


def technical(field_id: str) -> bool:
    """A date or a symbol identifier, which BRAIN advises against: not a measurement."""
    return bool(_TECHNICAL & set(_WORDS.split(field_id.lower())))


def slots(expression: str) -> list[str]:
    """The fields that can be swapped: every data field read, grouping fields and keyword values excepted."""
    from ..labs.fastexpr import keyword_values
    guarded = keyword_values(expression)
    return [f for f in fields_in_order(expression) if f not in guarded]


def swapped(expression: str, field: str, swap: str) -> str:
    """`field` read as `swap` everywhere it appears (a field used twice stays one field); keyword values guarded."""
    from ..labs.fastexpr import swap_field
    return swap_field(expression, field, swap)


def scope_of(settings: dict[str, Any]) -> tuple[str, int, str]:
    return str(settings.get("region") or ""), int(settings.get("delay") or 0), str(settings.get("universe") or "")


def label_of(scope: tuple[str, int, str]) -> str:
    return f"{scope[0]} D{scope[1]} {scope[2]}"


def held(settings: dict[str, Any]) -> dict[str, Any]:
    """The settings a variant runs at: the source's own, the mode left to the task."""
    return {k: v for k, v in settings.items() if k != "simulationMode"}


async def _field(scope: tuple[str, int, str], field: str) -> DataField | None:
    rows = await fields_at(*scope, [field])
    return rows[0] if rows else None


async def bucketable(scope: tuple[str, int, str], fields: list[str]) -> bool:
    """Whether the Alpha reads several fields, all of one dataset: one Buckets can split."""
    if len(fields) < 2:
        return False
    found = {r.id: r for r in await fields_at(*scope, fields)}
    return all(f in found for f in fields) and len({found[f].dataset_id for f in fields}) == 1


async def buckets(scope: tuple[str, int, str], fields: list[str]) -> dict[str, tuple[list[Candidate], str | None]]:
    """Each field's bucket of its dataset, in pick order, with why one is empty."""
    seeds = {r.id: r for r in await fields_at(*scope, fields)}
    missing = [f for f in fields if f not in seeds]
    if missing:
        why = f"{missing[0]} is not in the catalogue for {label_of(scope)}. Sync its dataset first."
        return {f: ([], why) for f in fields}
    datasets = {seeds[f].dataset_id for f in fields}
    if len(datasets) > 1:
        return {f: ([], "Buckets needs every field from one dataset.") for f in fields}
    async with db.session() as s:
        rows = sorted(await s.scalars(select(DataField).where(
            DataField.region == scope[0], DataField.delay == scope[1], DataField.universe == scope[2],
            DataField.dataset_id == next(iter(datasets)))), key=lambda r: r.id)
    own = set(fields)
    usable = [r for r in rows if r.id in own or not technical(r.id)]
    found: dict[str, tuple[list[Candidate], str | None]] = {}
    # a vec_ operator wraps a VECTOR field and nothing a MATRIX one: each type is split among its own seeds
    for kind in dict.fromkeys(seeds[f].type for f in fields):
        of_kind = [f for f in fields if seeds[f].type == kind]
        members = [r for r in usable if r.type == kind]
        texts = [FieldText(r.id, r.description or "", r.coverage) for r in members]
        by_id = {r.id: r for r in members}
        split_up = await asyncio.to_thread(split, texts, of_kind)
        for seed, bucket in zip(of_kind, split_up, strict=True):
            listed = [Candidate(f, by_id[f].dataset_id, "bucket", by_id[f].coverage) for f in bucket if f not in own]
            found[seed] = (listed, None if listed else f"No other {kind} field is like {seed}.")
    return found


async def fast_twins(scope: tuple[str, int, str], field: str) -> list[Candidate]:
    """In the Fast D1 market, a field's twin and the overnight change between the two."""
    if (scope[0], scope[1]) != FAST_D1_MARKET:
        return []
    regular = field.removesuffix(FAST_D1_SUFFIX)
    fast = regular + FAST_D1_SUFFIX
    twin = regular if field == fast else fast
    row = await _field(scope, twin)
    if row is None:
        return []
    return [Candidate(twin, row.dataset_id, FAST_D1_TIER, row.coverage),
            Candidate(f"{fast} - {regular}", row.dataset_id, FAST_D1_TIER, None)]


def _terms(text: str) -> list[str]:
    return [t for t in _TERMS.findall(text.lower()) if t not in _STOP and (len(t) > 1 or t.isdigit())]


def similar_ranked(seed: DataField, rows: list[DataField]) -> list[tuple[float, DataField]]:
    """Fields ranked by how closely their id and description read like the seed's (Okapi BM25 over the market)."""
    query = Counter(_terms(f"{seed.id} {seed.description or ''}"))
    if not query or not rows:
        return []
    docs = [Counter(_terms(f"{r.id} {r.description or ''}")) for r in rows]
    n = len(docs)
    avg = sum(sum(d.values()) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in d if t in query)
    k1, b = 1.2, 0.75
    scored = []
    for r, d in zip(rows, docs, strict=True):
        length = sum(d.values())
        score = 0.0
        for t in query:
            tf = d.get(t, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg))
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: (-x[0], x[1].id))
    return scored


async def candidates(scope: tuple[str, int, str], field: str, mode: str, k: int) -> tuple[list[Candidate], str | None]:
    """What `field` could be swapped for in Sweep or Similar, with why there is nothing when there is nothing."""
    seed = await _field(scope, field)
    if seed is None:
        return [], f"{field} is not in the catalogue for {label_of(scope)}. Sync its dataset first."
    twins = await fast_twins(scope, field)
    twinned = {c.field for c in twins}
    async with db.session() as s:
        stmt = select(DataField).where(DataField.region == scope[0], DataField.delay == scope[1],
                                       DataField.universe == scope[2], DataField.type == seed.type,
                                       DataField.id != field)
        if mode == SWEEP:
            stmt = stmt.where(DataField.dataset_id == seed.dataset_id)
        rows = [r for r in await s.scalars(stmt) if not technical(r.id) and r.id not in twinned]
    if mode == SWEEP:
        rows.sort(key=lambda r: (r.coverage is None, -(r.coverage or 0.0), r.id))
        found = twins + [Candidate(r.id, r.dataset_id, "dataset", r.coverage) for r in rows]
        return found, None if found else f"{seed.dataset_id} has no other {seed.type} field."
    if not (seed.description or "").strip():
        return twins, None if twins else f"{field} has no description to compare."
    nearest: dict[str, list[Candidate]] = {tier: [] for tier in TIERS}
    for _score, r in similar_ranked(seed, rows):
        tier = ("dataset" if r.dataset_id == seed.dataset_id
                else "subcategory" if seed.subcategory_id and r.subcategory_id == seed.subcategory_id
                else "category" if seed.category_id and r.category_id == seed.category_id else "market")
        if len(nearest[tier]) < k:
            nearest[tier].append(Candidate(r.id, r.dataset_id, tier, r.coverage))
    found = twins + [c for tier in TIERS for c in nearest[tier]]
    return found, None if found else f"No field reads like {field}."


async def found_at(settings: dict[str, Any], fields: list[str], mode: str,
                   k: int) -> dict[str, tuple[list[Candidate], str | None]]:
    scope = scope_of(settings)
    if mode == BUCKETS:
        return await buckets(scope, fields)
    return {f: await candidates(scope, f, mode, k) for f in fields}


async def swaps_at(settings: dict[str, Any], fields: list[str], mode: str, k: int) -> dict[str, list[Candidate]]:
    return {f: listed for f, (listed, _) in (await found_at(settings, fields, mode, k)).items()}


def identity(expression: str, settings: dict[str, Any]) -> str:
    return trial_key(expression, held(settings))


def variants(expression: str, settings: dict[str, Any], swaps: dict[str, list[Candidate]],
             seen: set[str]) -> list[Variant]:
    """One simulation per field and candidate, none already in `seen` (which is added to). Fields take turns, each
    in its own candidates' order, so a task that cannot run them all gives every field its best swaps first. A
    candidate the expression already reads is skipped: `liabilities / assets` to `assets / assets` says nothing."""
    own = set(slots(expression))
    out: list[Variant] = []
    for turn in zip_longest(*swaps.values()):
        for field, c in zip(swaps, turn, strict=True):
            if c is None or c.field in own:
                continue
            code = swapped(expression, field, c.field)
            key = identity(code, settings)
            if key in seen:
                continue
            seen.add(key)
            out.append(Variant(code, held(settings), field, c.field, c.tier))
    return out


def grid(expression: str, settings: dict[str, Any], swaps: dict[str, list[Candidate]], top: int,
         seen: set[str]) -> list[Variant]:
    """Every field swapped at once, one from each bucket's `top` picks; nearest the top of every bucket first (by
    the sum of the picks' places), so a short task still pairs each bucket's best with the other's."""
    fields = list(swaps)
    picks = [swaps[f][:top] for f in fields]
    places = sorted(product(*(range(len(p)) for p in picks)), key=lambda at: (sum(at), max(at, default=0), at))
    out: list[Variant] = []
    for at in places:
        chosen = [p[i] for p, i in zip(picks, at, strict=True)]
        code = expression
        for field, c in zip(fields, chosen, strict=True):
            code = swapped(code, field, c.field)
        key = identity(code, settings)
        if key in seen:
            continue
        seen.add(key)
        out.append(Variant(code, held(settings), ",".join(fields), ",".join(c.field for c in chosen), "bucket"))
    return out


def taking_turns(groups: list[list[Variant]], size: int = MULTI_SIMULATION_SIZE) -> list[Variant]:
    """Each set of settings in turn, a multi-simulation's worth at a time: every one is reached, and each batch keeps
    to one market."""
    chunks = [[g[i:i + size] for i in range(0, len(g), size)] for g in groups]
    return [v for turn in zip_longest(*chunks, fillvalue=[]) for chunk in turn for v in chunk]


async def elsewhere(alpha_id: str, expression: str, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """The same expression's other submitted Alphas, one per set of settings: where it is already known to work.
    A region-agnostic Alpha's own family is left out (its children are its own settings, region by region)."""
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(Alpha.code == expression, Alpha.id != alpha_id).order_by(
            Alpha.date_created)))
        source = await s.get(Alpha, alpha_id)
    family = {alpha_id}
    parent = ((source.raw or {}).get("parent") or (source.raw or {}).get("parentId")) if source else None
    if parent:
        family.add(str(parent))
    known = {identity(expression, settings)}
    found = []
    for a in rows:
        if (a.status or "UNSUBMITTED") == "UNSUBMITTED" or not a.settings:
            continue
        own_parent = (a.raw or {}).get("parent") or (a.raw or {}).get("parentId")
        if own_parent and str(own_parent) in family:
            continue
        key = identity(expression, a.settings)
        if key in known:
            continue
        known.add(key)
        found.append({"alpha_id": a.id, "status": a.status, "raw_settings": dict(a.settings),
                      "settings": held_settings(a.settings)})
    return found


async def plan(alpha_id: str, mode: str = SWEEP, k: int = DEFAULT_K) -> dict[str, Any]:
    """The Alpha, its held settings, and each field with what it could be swapped for. Simulates nothing."""
    from ..fastexpr import FastExprSyntaxError
    alpha_id = alpha_id_of(alpha_id)
    if mode not in MODES:
        raise ToolError(f"Unknown mode {mode!r}: choose Sweep, Similar or Buckets.")
    k = max(1, min(int(k or DEFAULT_K), 50))
    problems: list[str] = []
    body = await alpha_body(alpha_id)
    expression = code_of(body)
    settings = dict(body.get("settings") or {})
    found: dict[str, Any] = {"alpha_id": alpha_id, "expression": expression or "", "settings": held_settings(settings),
                             "raw_settings": settings, "slots": [], "others": [], "bucketable": False, "mode": mode,
                             "k": k, "problems": problems}
    if not expression:
        problems.append(f"{alpha_id} has no expression to sample.")
        return found
    try:
        fields = slots(expression)
    except (FastExprSyntaxError, ValueError) as exc:
        problems.append(f"Its expression could not be parsed: {exc}")
        return found
    if not fields:
        problems.append(f"{alpha_id} reads no data field that can be swapped.")
    can_bucket = await bucketable(scope_of(settings), fields)
    found["bucketable"] = can_bucket
    if mode == BUCKETS and not can_bucket:
        problems.append("Buckets is for an Alpha reading several fields of one dataset.")
        return found
    swaps = await found_at(settings, fields, mode, k)
    rows = [{"field": f, "candidates": [{"field": c.field, "dataset": c.dataset, "tier": c.tier,
                                         "coverage": c.coverage} for c in swaps[f][0]], "problem": swaps[f][1] or ""}
            for f in fields]
    if fields and not any(r["candidates"] for r in rows):
        problems.append("None of its fields has anything to be swapped for.")
    others = []
    for o in await elsewhere(alpha_id, expression, settings):
        counts = await swaps_at(o["raw_settings"], fields, mode, k)
        others.append({**o, "swaps": [{"field": f, "count": len(v)} for f, v in counts.items()]})
    found.update(slots=rows, others=others)
    return found


async def chosen_variants(found: dict[str, Any], params: dict[str, Any]) -> tuple[list[Variant], int, list[str]]:
    """What a choice queues first, the task's budget, and the fields swapped. Budget rules (alpha-harness): none set
    runs until no passing swap leads anywhere new; a Sweep of one field leads nowhere (a winner's sweep is the same
    dataset again) and Buckets grows no tree, so there the first sweep is the task; a budget under the first sweep
    runs its best-covered swaps, every field taking turns."""
    mode, k = found["mode"], found["k"]
    wanted = set(params.get("fields") or [])
    fields = [s["field"] for s in found["slots"] if not wanted or s["field"] in wanted]
    include = set(params.get("alphas") or [])
    sources = [found["raw_settings"]] + [o["raw_settings"] for o in found["others"] if o["alpha_id"] in include]
    bucketed = mode == BUCKETS
    if bucketed:
        fields = [s["field"] for s in found["slots"]]          # every field changes at once
    seen: set[str] = {identity(found["expression"], st) for st in sources}
    groups = []
    for st in sources:
        swaps = await swaps_at(st, fields, mode, k)
        groups.append(grid(found["expression"], st, swaps, k, seen) if bucketed
                      else variants(found["expression"], st, swaps, seen))
    chosen = taking_turns(groups, MULTI_SIMULATION_SIZE)
    if not chosen:
        raise ToolError("Nothing to run: no chosen field has a swap.")
    one_sweep = mode == SWEEP and len(found["slots"]) == 1
    budget = params.get("budget")
    budget = int(budget) if budget else (len(chosen) if one_sweep or bucketed else MAX_SIMULATIONS)
    return chosen[:budget], budget, fields


def winner(alpha: Alpha | None) -> bool:
    """A finished alpha nothing refused (BRAIN's verdict submittable, or pending its checks)."""
    if alpha is None:
        return False
    return brain_verdict(alpha.checks or alpha.is_checks, alpha.simulation_mode) in ("submittable", "pending")


class FieldsSamplerLab(Lab):
    """The first sweep is written up front; every winner is swept the same way as it is scored (Sweep, Similar)."""

    name = NAME
    title = "Fields Sampler"

    async def plan(self, task: LabTask) -> dict[str, Any]:
        params = task.params or {}
        found = await plan(params.get("alpha_id") or "", params.get("mode") or SWEEP,
                           int(params.get("k") or DEFAULT_K))
        if found["problems"]:
            raise ToolError(found["problems"][0])
        chosen, budget, fields = await chosen_variants(found, params)
        return {"alpha_id": found["alpha_id"], "expression": found["expression"], "mode": found["mode"],
                "k": found["k"], "fields": fields, "budget": budget, "settings": found["settings"],
                "sources": [found["raw_settings"]] + [o["raw_settings"] for o in found["others"]
                                                      if o["alpha_id"] in set(params.get("alphas") or [])],
                "requests": [v.request() for v in chosen]}

    async def initial(self, task: LabTask) -> list[TrialSpec]:
        return await written_up_front(task, budget=int((task.space or {}).get("budget") or 0) or None)

    def done_early(self, task: LabTask, ctx: RefillContext) -> str | None:
        return all_run(ctx)

    async def on_scored(self, task: LabTask, trials: list[LabTrial]) -> None:
        await expand(task, trials)

    async def preview(self, params: dict[str, Any]) -> dict[str, Any]:
        found = await plan(params.get("alpha_id") or "", params.get("mode") or SWEEP, int(params.get("k") or DEFAULT_K))
        out = {**found, "simulations": 0, "budget": None, "sample": []}
        if found["problems"]:
            return out
        try:
            chosen, budget, _ = await chosen_variants(found, params)
        except ToolError as exc:
            out["problems"] = [str(exc)]
            return out
        out.update(simulations=len(chosen), budget=budget, sample=[v.request() for v in chosen[:8]],
                   batches=math.ceil(len(chosen) / MULTI_SIMULATION_SIZE))
        return out


async def expand(task: LabTask, harvested: list[LabTrial]) -> int:
    """Sweep each new winner the way the seed was swept, within the task's simulations; returns trials added."""
    from ..labs import runner as runner_mod
    space = task.space or {}
    mode, k = space.get("mode") or SWEEP, int(space.get("k") or DEFAULT_K)
    if mode == BUCKETS:
        return 0              # the grid is the task: a winner's own grid would be the same buckets again
    done = [t for t in harvested if t.state == "COMPLETE" and t.alpha_id]
    if not done:
        return 0
    async with db.session() as s:
        alphas = {a.id: a for a in await s.scalars(select(Alpha).where(Alpha.id.in_([t.alpha_id for t in done])))}
        count = int(await s.scalar(select(func.count(LabTrial.id)).where(LabTrial.task_id == task.id)) or 0)
        tried = (await s.execute(select(LabTrial.expression, LabTrial.settings).where(
            LabTrial.task_id == task.id))).all()
    winners = [t for t in done if winner(alphas.get(t.alpha_id))]
    room = int(task.max_trials) - count
    if not winners or room <= 0:
        return 0
    seen = {identity(e or "", dict(st or {})) for e, st in tried}
    source = space.get("expression") or ""
    seen |= {identity(source, dict(st or {})) for _, st in tried}
    specs: list[TrialSpec] = []
    for parent in winners:
        if room <= 0:
            break
        settings = dict(parent.settings or {})
        try:
            fields = slots(parent.expression)
        except ValueError:
            continue
        swaps = await swaps_at(settings, fields, mode, k)
        for v in variants(parent.expression, settings, swaps, seen)[:room]:
            room -= 1
            r = v.request(parent=parent.number, generation=(parent.generation or 1) + 1)
            specs.append(TrialSpec(r["expression"], r["settings"], params=r["params"], generation=r["generation"],
                                   parent=[parent.number]))
    if specs:
        await runner_mod.runner.add_trials(task.id, specs)
    return len(specs)


def tree(trials: list[LabTrial]) -> list[dict[str, Any]]:
    """The task's trials as a tree for display: each swap under the winner it grew from."""
    by_number = {t.number: t for t in trials}
    nodes: dict[int, dict[str, Any]] = {}
    roots: list[dict[str, Any]] = []
    for t in sorted(trials, key=lambda x: x.number):
        p = t.params or {}
        sharpe = (t.figures or {}).get("sharpe")
        label = (f"#{t.number} {p.get('field', '?')} → {p.get('swap', '?')} · {t.state.lower()}"
                 + (f" · Sharpe {sharpe:.2f}" if isinstance(sharpe, (int, float)) else ""))
        node = {"id": str(t.number), "label": label, "children": []}
        nodes[t.number] = node
        parent = p.get("parent")
        if parent is not None and parent in by_number and parent in nodes:
            nodes[parent]["children"].append(node)
        else:
            roots.append(node)
    return roots
