"""Correlation Breaker: re-shape a submittable Alpha that BRAIN says is already in the pool (alpha-harness
tools/correlation_breaker).

Failing PROD_CORRELATION means the daily PnL is correlated at 0.70 or more with an Alpha already in production and
this one's Sharpe is not 10% above it. The signal is sound; it is the exposure that is shared. So the settings are
never touched — region, delay, universe, neutralization, decay and truncation are the source Alpha's, exactly, held
whole, in Full mode (the only one that runs the production-correlation check) — and only the expression changes.

Every recipe binds the original expression to one name, `alpha`, and wraps that. The binding is appended after the
reader's own statements, so it cannot disturb them. The recipes are what the platform's guidance says works
(residualize, orthogonalize, trade the innovation, gate on catalysts, regroup) and never added noise.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from ..brain.platform import (
    FULL_MODE,
    GROUPING_FIELDS,
    MULTI_COUNTRY_REGIONS,
    POWER_POOL_MAX_FIELDS,
    POWER_POOL_MAX_OPERATORS,
    PROD_CORRELATION_CHECK,
    SINGLE_DATASET_CLASS,
    brain_verdict,
    is_power_pool,
)
from ..db import db
from ..db.models import Alpha, DataField, Operator
from ..db.models_labs import LabTask, LabTrial
from ..labs.runner import Lab, RefillContext, TrialSpec
from .common import ToolError, alpha_body, alpha_id_of, code_of, held_settings
from .settings_sampler import all_run, written_up_front

NAME = "correlation_breaker"
PREF = "pref:correlation_breaker"
ALPHA = "alpha"
GROUP_TYPE = "GROUP"
CUSTOM = "custom:"
NAME_LIMIT = 120
CODE_LIMIT = 4000
GIVE_UP_AFTER = 3          # runs without one success after which a recipe starts unticked
OPERATORS_UNREAD = ("The operator list for this account is not stored yet, so no recipe can be checked. Sign in "
                    "and sync operators on the Sync page, then analyse again.")
_LAST_ASSIGN = re.compile(r"(?:^|;)\s*([A-Za-z_]\w*)\s*=(?!=)[^;]*;?\s*$", re.S)


@dataclass(frozen=True, slots=True)
class Compressed:
    """One Alpha's expression, reduced to its statements plus the line binding `alpha`."""

    prelude: tuple[str, ...]
    binding: str

    def program(self, tail: str, extra: Sequence[str] = ()) -> str:
        """The whole expression: the reader's work, the binding, any recipe setup, then `tail`."""
        return ";\n".join([*self.prelude, *([self.binding] if self.binding else []), *extra, tail])

    @property
    def bound(self) -> str:
        """What `alpha` means, shown once above the re-shapes."""
        return "\n".join(f"{line};" for line in [*self.prelude, *([self.binding] if self.binding else [])])

    @staticmethod
    def transform(tail: str, extra: Sequence[str] = ()) -> str:
        """Just the re-shape, written against `alpha`."""
        return ";\n".join([*extra, tail])


def compress(expression: str) -> Compressed:
    """`rank(close)` becomes `alpha = rank(close)`; a multi-statement program keeps every statement and gains one
    more; a program whose last statement already assigns a name is bound from that name (nothing computed twice)."""
    from ..fastexpr import Assign, parse
    text = (expression or "").strip().rstrip(";").strip()
    if not text:
        raise ValueError("The expression is empty.")
    last = _LAST_ASSIGN.search(text)
    if last:
        # AlphaGo's parser wants an expression last; read the program as ending in the name it assigned
        program = parse(f"{text}; {last.group(1)}")
        statements = program.statements[:-1]
        prelude = tuple(s.to_source() for s in statements)
        name = statements[-1].name if isinstance(statements[-1], Assign) else last.group(1)
        return Compressed(prelude, "" if name == ALPHA else f"{ALPHA} = {name}")
    program = parse(text)
    statements = program.statements
    return Compressed(tuple(s.to_source() for s in statements[:-1]), f"{ALPHA} = {statements[-1].to_source()}")


@dataclass(frozen=True, slots=True)
class Recipe:
    id: str
    name: str
    why: str
    setup: tuple[str, ...]
    tail: str
    operators: tuple[str, ...]
    fields: tuple[str, ...] = ()
    caution: str = ""
    regions: frozenset[str] = frozenset()
    missing: str = ""


RECIPES: tuple[Recipe, ...] = (
    Recipe(
        id="resid_returns", name="Keep what returns do not explain",
        why=("Regresses the signal on each instrument's own daily returns over a quarter and keeps only the "
             "residual. What the pool shares is often the part of a signal that is just price reaction; this "
             "takes it out one instrument at a time."),
        setup=(), tail=f"ts_regression({ALPHA}, returns, 63, rettype = 0)", operators=("ts_regression",),
        fields=("returns",), caution="A residual moves more day to day than the signal it came from: watch turnover."),
    Recipe(
        id="resid_size", name="Keep what size does not explain",
        why=("Regresses the signal on the instrument's log market cap over a year and keeps the residual: how the "
             "signal drifts with each company's own size, taken out over time rather than across the market on "
             "one day."),
        setup=(), tail=f"ts_regression({ALPHA}, log(cap), 252, rettype = 0)", operators=("ts_regression", "log"),
        fields=("cap",)),
    Recipe(
        id="neut_momentum", name="Orthogonalize against short-term momentum",
        why=("Forces the daily exposure to five-day price momentum to zero. Fundamental and analyst signals often "
             "ride momentum without meaning to."),
        setup=(), tail=f"vector_neut({ALPHA}, ts_delta(close, 5))", operators=("vector_neut", "ts_delta"),
        fields=("close",)),
    Recipe(
        id="neut_size", name="Orthogonalize against size",
        why=("Projects out market capitalisation, the single most shared exposure in the pool. What is left is the "
             "part of the signal that is not a size bet."),
        setup=(), tail=f"vector_neut({ALPHA}, log(cap))", operators=("vector_neut", "log"), fields=("cap",)),
    Recipe(
        id="neut_both", name="Orthogonalize against size and momentum, then decay",
        why=("Both crowded vectors projected out in turn, then smoothed. The decay is there to settle the turnover "
             "that orthogonalizing stirs up."),
        setup=(f"no_size = vector_neut({ALPHA}, log(cap))", "orthogonal = vector_neut(no_size, ts_delta(close, 5))"),
        tail="ts_decay_linear(orthogonal, 3)", operators=("vector_neut", "log", "ts_delta", "ts_decay_linear"),
        fields=("cap", "close"),
        caution="Orthogonalizing lifts turnover; the decay is what keeps it inside the limit."),
    Recipe(
        id="fast_innovation", name="Trade the innovation, not the level",
        why=("Subtracts the signal's own twenty-day mean, leaving the part that is new. The pool is full of models "
             "trading the level of the same quantity."),
        setup=(), tail=f"ts_av_diff({ALPHA}, 20)", operators=("ts_av_diff",)),
    Recipe(
        id="event_gate", name="Trade only on volume catalysts",
        why=("Holds its weights between catalysts instead of re-ranking every day. A PnL that moves on event days "
             "cannot track a continuously rebalanced model."),
        setup=("catalyst = volume > 1.25 * adv20",), tail=f"trade_when(catalyst, {ALPHA}, -1)",
        operators=("trade_when",), fields=("volume", "adv20"),
        caution=("Gating cuts turnover, which must stay above the 1% floor, and a quiet day leaves few instruments "
                 "carrying the book: watch the weight test.")),
    Recipe(
        id="cluster_country", name="Neutralize inside country and industry",
        why=("Across many countries, industry neutralization alone leaves each country's own market move in the "
             "book. BRAIN's documented double neutralization removes both."),
        setup=("cluster = densify(group_cartesian_product(industry, country))",),
        tail=f"group_neutralize({ALPHA}, cluster)",
        operators=("densify", "group_cartesian_product", "group_neutralize"), fields=("industry", "country"),
        regions=MULTI_COUNTRY_REGIONS),
    Recipe(
        id="cluster_zscore", name="Normalize inside sector and size clusters",
        why=("Scores each instrument against peers of the same sector and size band rather than against the whole "
             "market. The pool almost always neutralizes on sector alone."),
        setup=('cap_bucket = bucket(rank(cap), buckets = "0.33, 0.66")',
               "cluster = densify(group_cartesian_product(sector, cap_bucket))"),
        tail=f"group_zscore({ALPHA}, cluster)",
        operators=("bucket", "rank", "densify", "group_cartesian_product", "group_zscore"), fields=("cap", "sector"),
        caution=("Three coarse bands on sector, not subindustry: finer clusters leave too few instruments in a "
                 "bucket and fail the weight test.")),
    Recipe(
        id="cluster_volatility", name="Neutralize inside sector and volatility clusters",
        why=("Compares each instrument only with peers of the same sector and volatility band. The pool competes in "
             "the same static sector buckets; a volatility band cuts the cross-section along a line they do not."),
        setup=('vol_bucket = bucket(rank(ts_std_dev(returns, 63)), buckets = "0.33, 0.66")',
               "cluster = densify(group_cartesian_product(sector, vol_bucket))"),
        tail=f"group_neutralize({ALPHA}, cluster)",
        operators=("bucket", "rank", "ts_std_dev", "densify", "group_cartesian_product", "group_neutralize"),
        fields=("returns", "sector"),
        caution="Three bands for the same reason as the size clusters: finer ones starve buckets."),
)


# ---------------------------------------------------------------- this market's own groupings

def _usage(row: DataField) -> tuple[int, int]:
    """Users first (one consultant's sweep multiplies an Alpha count, not a user count); Alphas settle a tie."""
    return int(row.user_count or 0), int(row.alpha_count or 0)


async def most_used_group(region: str, delay: int, universe: str, *, exclusive: bool) -> DataField | None:
    """The GROUP field most consultants use in this market, the standard groupings aside. `exclusive`: one found in
    no other synced region, which no Alpha from elsewhere can have been neutralized on."""
    async with db.session() as s:
        rows = [r for r in await s.scalars(select(DataField).where(
            DataField.region == region, DataField.delay == int(delay), DataField.universe == universe,
            DataField.type == GROUP_TYPE)) if r.id not in GROUPING_FIELDS]
        if exclusive and rows:
            elsewhere = set(await s.scalars(select(DataField.id).where(
                DataField.id.in_([r.id for r in rows]), DataField.region != region).distinct()))
            rows = [r for r in rows if r.id not in elsewhere]
    return max(rows, key=_usage) if rows else None


def _users(row: DataField) -> str:
    return f"{int(row.user_count or 0):,} consultants"


def _unbuilt(id_: str, name: str, missing: str) -> Recipe:
    return Recipe(id=id_, name=name, why="", setup=(), tail="", operators=(), missing=missing)


def _grouped(id_: str, field: str, about: str) -> list[Recipe]:
    return [
        Recipe(id=id_, name=f"Neutralize inside {field}",
               why=(f"Compares each instrument only with peers in {field}: {about}. The pool mostly neutralizes on "
                    "sector and industry, so peers drawn by another rule move the book along a line it does not "
                    "share."),
               setup=(), tail=f"group_neutralize({ALPHA}, densify({field}))",
               operators=("group_neutralize", "densify"), fields=(field,)),
        Recipe(id=f"{id_}_sector", name=f"Neutralize inside sector and {field}",
               why=(f"Neutralizes inside every pair of sector and {field} at once, the double neutralization BRAIN "
                    "suggests where country and industry are one market."),
               setup=(f"cluster = densify(group_cartesian_product(sector, {field}))",),
               tail=f"group_neutralize({ALPHA}, cluster)",
               operators=("densify", "group_cartesian_product", "group_neutralize"), fields=("sector", field),
               caution=f"If {field} already sits inside sectors the pair is {field} again, and this repeats the one "
                       "above."),
    ]


def group_recipes(top: DataField | None, exclusive: DataField | None, region: str) -> list[Recipe]:
    """Neutralization inside this market's own groupings, alone and paired with sector (through
    group_cartesian_product, never two group_neutralize in turn: the second undoes part of the first)."""
    exclusive_name = "Neutralize inside a region-exclusive grouping"
    out: list[Recipe] = []
    if top is None:
        out.append(_unbuilt("group_top", "Neutralize inside the most used grouping",
                            "no grouping beyond the standard ones covers this market"))
    else:
        out.extend(_grouped("group_top", top.id,
                            f"the grouping most used here after the standard ones ({_users(top)})"))
    if exclusive is None:
        out.append(_unbuilt("group_exclusive", exclusive_name,
                            f"no grouping is exclusive to {region} among the synced regions"))
    elif top is not None and exclusive.id == top.id:
        out.append(_unbuilt("group_exclusive", exclusive_name,
                            f"the most used grouping above is already exclusive to {region}"))
    else:
        out.extend(_grouped("group_exclusive", exclusive.id,
                            f"found in {region} and no other synced region, so no Alpha built elsewhere can have been "
                            f"neutralized on it ({_users(exclusive)})"))
    return out


# ---------------------------------------------------------------- re-shapes written by hand

def custom_recipe(key: str, name: str, code: str) -> Recipe:
    """A re-shape written by hand around `alpha`: any statements it needs, then the line that runs. One that cannot
    run says why, as a built-in one does."""
    from ..fastexpr import Call, FastExprSyntaxError, Ident, parse, walk
    from ..labs.fastexpr import data_fields
    id_ = CUSTOM + key
    name = (name or "").strip()
    if not name:
        return _unbuilt(id_, name, "has no name: give it one")
    if len(name) > NAME_LIMIT:
        return _unbuilt(id_, name, f"name is {len(name)} characters: keep it to {NAME_LIMIT}")
    if len(code) > CODE_LIMIT:
        return _unbuilt(id_, name, f"is {len(code)} characters: keep it to {CODE_LIMIT}")
    if not code.strip():
        return _unbuilt(id_, name, f"empty: write it around {ALPHA}")
    try:
        program = parse(code)
    except FastExprSyntaxError as exc:
        if "last statement must be an expression" in str(exc):
            return _unbuilt(id_, name, "its last line assigns a name: end with the expression to run")
        return _unbuilt(id_, name, f"could not be read: {exc}")
    statements = program.statements
    if ALPHA not in {n.name for n in walk(program) if isinstance(n, Ident)}:
        return _unbuilt(id_, name, f"never reads {ALPHA}: write it around {ALPHA}")
    return Recipe(id=id_, name=name, why="", setup=tuple(s.to_source() for s in statements[:-1]),
                  tail=statements[-1].to_source(),
                  operators=tuple(sorted({n.name for n in walk(program) if isinstance(n, Call)})),
                  fields=tuple(sorted(f for f in data_fields(program, grouping=True) if f != ALPHA)))


async def saved_custom() -> list[dict[str, str]]:
    pref = await db.kv_get(PREF, {}) or {}
    return [c for c in pref.get("custom") or [] if isinstance(c, dict) and c.get("id")]


async def save_custom(custom: list[dict[str, str]]) -> None:
    """Keep the consultant's re-shapes on this computer (ids are card ids: letters, digits, _ and -)."""
    clean = []
    for c in custom[:50]:
        cid = str(c.get("id") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid):
            raise ToolError(f"A re-shape id may hold only letters, digits, _ and -: {cid!r}.")
        clean.append({"id": cid, "name": str(c.get("name") or "")[:10_000], "code": str(c.get("code") or "")[:100_000]})
    pref = await db.kv_get(PREF, {}) or {}
    await db.kv_set(PREF, {**pref, "custom": clean})


# ---------------------------------------------------------------- what has worked on this account

async def history() -> dict[str, tuple[int, int]]:
    """Each recipe's runs on this account so far, as (judged, worked). Worked: BRAIN's Production Correlation check
    passed and nothing else refused the Alpha. A run whose correlation check is not settled counts for neither."""
    async with db.session() as s:
        task_ids = list(await s.scalars(select(LabTask.id).where(LabTask.lab == NAME)))
        if not task_ids:
            return {}
        trials = (await s.execute(select(LabTrial.params, LabTrial.alpha_id).where(
            LabTrial.task_id.in_(task_ids), LabTrial.alpha_id.is_not(None)))).all()
        ids = list({a for _, a in trials if a})
        alphas = {a.id: a for a in await s.scalars(select(Alpha).where(Alpha.id.in_(ids)))} if ids else {}
    found: dict[str, tuple[int, int]] = {}
    for params, alpha_id in trials:
        recipe = (params or {}).get("recipe")
        a = alphas.get(alpha_id)
        if not recipe or a is None:
            continue
        checks = a.checks or a.is_checks or []
        prod = next((c for c in checks if str(c.get("name")) == PROD_CORRELATION_CHECK), None)
        result = str((prod or {}).get("result") or "").upper()
        if result not in ("PASS", "FAIL"):
            continue
        judged, worked = found.get(recipe, (0, 0))
        broke = result == "PASS" and brain_verdict(checks, a.simulation_mode) != "refused"
        found[recipe] = (judged + 1, worked + int(broke))
    return found


def ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What has worked here first, best rate first; then the untried, in the research's order; then what has only
    ever failed here."""
    def key(item: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
        place, row = item
        judged, worked = row["judged"], row["worked"]
        if worked:
            return 0, -worked / judged, place
        return (2 if judged else 1), 0.0, place
    return [row for _, row in sorted(enumerate(rows), key=key)]


def offered(recipes: Sequence[Recipe], operators: set[str], fields: set[str], region: str) -> list[tuple[Recipe, str]]:
    """Each recipe with the reason it cannot run here, empty when it can: reported rather than filtered away, since
    a silently shorter list reads as if there were fewer ideas."""
    out: list[tuple[Recipe, str]] = []
    for recipe in recipes:
        missing_ops = [o for o in recipe.operators if o not in operators]
        missing_fields = [f for f in recipe.fields if f not in fields]
        if recipe.missing:
            out.append((recipe, recipe.missing))
        elif recipe.regions and region not in recipe.regions:
            out.append((recipe, f"only for regions spanning countries: {', '.join(sorted(recipe.regions))}"))
        elif missing_ops:
            out.append((recipe, f"needs {', '.join(missing_ops)}, not on this account"))
        elif missing_fields:
            out.append((recipe, f"needs {', '.join(missing_fields)}, not in this market"))
        else:
            out.append((recipe, ""))
    return out


def counted(program: str) -> tuple[int, int]:
    """Operators and data fields, the way Power Pool counts them."""
    from ..labs.fastexpr import data_fields, operator_count
    return operator_count(program), len(data_fields(program))


def over_power_pool(operators: int, fields: int) -> str:
    over = ([f"{operators} operators (limit {POWER_POOL_MAX_OPERATORS})"] if operators > POWER_POOL_MAX_OPERATORS
            else []) + ([f"{fields} data fields (limit {POWER_POOL_MAX_FIELDS})"] if fields > POWER_POOL_MAX_FIELDS
                        else [])
    return " and ".join(over)


def over_single_dataset(recipe: Recipe, dataset_of: dict[str, str], own: set[str]) -> str:
    if not own:
        return ""
    return ", ".join(f"{f} ({dataset_of[f]})" for f in recipe.fields
                     if f not in GROUPING_FIELDS and f in dataset_of and dataset_of[f] not in own)


def requests(compressed: Compressed, recipes: Sequence[Recipe], settings: dict[str, Any]) -> list[dict[str, Any]]:
    """One simulation per recipe, every one at the source Alpha's own settings copied whole, in Full mode."""
    held = {**settings, "simulationMode": FULL_MODE}
    return [{"expression": compressed.program(r.tail, r.setup), "settings": dict(held),
             "params": {"recipe": r.id, "name": r.name}} for r in recipes]


def _correlation(checks: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    return next((c for c in checks if str(c.get("name", "")).upper() == PROD_CORRELATION_CHECK), None)


def _empty(alpha_id: str, settings: dict[str, Any], checks: list[dict[str, Any]], problems: list[str]) -> dict:
    return {"alpha_id": alpha_id, "expression": "", "bound": "", "settings": held_settings(settings),
            "raw_settings": settings, "correlation": _correlation(checks), "power_pool": is_power_pool(checks),
            "single_dataset": False, "operators": None, "data_fields": None, "recipes": [], "recipe_set": [],
            "problems": problems}


async def plan(alpha_id: str, custom: Sequence[tuple[str, str, str]] | None = None) -> dict[str, Any]:
    """Everything the screen shows: the Alpha, its settings and each recipe's expression. Simulates nothing."""
    from ..fastexpr import FastExprSyntaxError
    alpha_id = alpha_id_of(alpha_id)
    if custom is None:
        custom = [(c["id"], c.get("name", ""), c.get("code", "")) for c in await saved_custom()]
    problems: list[str] = []
    body = await alpha_body(alpha_id)
    expression = code_of(body)
    settings = dict(body.get("settings") or {})
    checks = [c for c in (body.get("is") or {}).get("checks") or [] if isinstance(c, dict)]
    if not expression:
        return _empty(alpha_id, settings, checks, [f"{alpha_id} has no expression to re-shape."])
    try:
        compressed = compress(expression)
    except (FastExprSyntaxError, ValueError) as exc:
        return _empty(alpha_id, settings, checks,
                      [f"Its expression could not be parsed, so it cannot be re-shaped: {exc}"])
    region, delay = str(settings.get("region") or ""), int(settings.get("delay") or 0)
    universe = str(settings.get("universe") or "")
    async with db.session() as s:
        operators = set(await s.scalars(select(Operator.name)))
    if not operators:
        problems.append(OPERATORS_UNREAD)
    top = await most_used_group(region, delay, universe, exclusive=False)
    only_here = await most_used_group(region, delay, universe, exclusive=True)
    recipes = (*RECIPES, *group_recipes(top, only_here, region), *(custom_recipe(*c) for c in custom))
    wanted = sorted({f for recipe in recipes for f in recipe.fields})
    from .common import fields_at
    here = {r.id: r for r in await fields_at(region, delay, universe, wanted)}
    # every market carries the standard groupings (alpha-harness platform.GROUPING_FIELDS); the rest must be here
    fields = set(here) | (set(wanted) & GROUPING_FIELDS)
    power_pool = is_power_pool(checks)
    past = await history()
    single = any(isinstance(c, dict) and c.get("id") == SINGLE_DATASET_CLASS for c in body.get("classifications") or [])
    from ..labs.fastexpr import data_fields
    own = sorted(f for f in data_fields(expression, grouping=True) if f not in GROUPING_FIELDS)
    extra = sorted({f for r in recipes for f in r.fields if f not in GROUPING_FIELDS})
    dataset_of = ({r.id: r.dataset_id for r in await fields_at(region, delay, universe, [*own, *extra])}
                  if single else {})
    own_datasets = {dataset_of[f] for f in own if f in dataset_of}
    rows = []
    for recipe, blocked in offered(recipes, operators, fields, region):
        used, read = (0, 0) if recipe.missing else counted(compressed.program(recipe.tail, recipe.setup))
        judged, worked = past.get(recipe.id, (0, 0))
        rows.append({"id": recipe.id, "name": recipe.name, "custom": recipe.id.startswith(CUSTOM), "why": recipe.why,
                     "caution": recipe.caution,
                     "transform": "" if recipe.missing else compressed.transform(recipe.tail, recipe.setup),
                     "blocked": blocked, "operators": used, "data_fields": read,
                     "over_power_pool": over_power_pool(used, read) if power_pool else "",
                     "over_single_dataset": over_single_dataset(recipe, dataset_of, own_datasets),
                     "judged": judged, "worked": worked, "gave_up": judged >= GIVE_UP_AFTER and not worked})
    used, read = counted(expression)
    return {"alpha_id": alpha_id, "expression": expression, "bound": compressed.bound,
            "settings": held_settings(settings), "raw_settings": settings, "correlation": _correlation(checks),
            "power_pool": power_pool, "single_dataset": single, "operators": used, "data_fields": read,
            "recipes": ranked(rows), "recipe_set": list(recipes), "problems": problems}


def chosen_requests(found: dict[str, Any], wanted: Sequence[str] = ()) -> list[dict[str, Any]]:
    """The simulations a choice queues: a recipe the plan marked blocked is never queued, asked for or not."""
    runnable = {r["id"] for r in found["recipes"] if not r["blocked"]}
    wanted_set = set(wanted)
    chosen = [r for r in found["recipe_set"] if r.id in runnable and (not wanted_set or r.id in wanted_set)]
    if not chosen:
        raise ToolError("Nothing to run: every chosen re-shape needs an operator or a field this market does not "
                        "have.")
    return requests(compress(found["expression"]), chosen, found["raw_settings"])


def default_ticks(found: dict[str, Any]) -> list[str]:
    """Ticked to begin with: every runnable recipe except those this account gave up on."""
    return [r["id"] for r in found["recipes"] if not r["blocked"] and not r["gave_up"]]


class CorrelationBreakerLab(Lab):
    name = NAME
    title = "Correlation Breaker"

    async def plan(self, task: LabTask) -> dict[str, Any]:
        params = task.params or {}
        custom = params.get("custom")
        found = await plan(params.get("alpha_id") or "", None if custom is None else [
            (c["id"], c.get("name", ""), c.get("code", "")) for c in custom])
        if found["problems"]:
            raise ToolError(found["problems"][0])
        reqs = chosen_requests(found, params.get("recipes") or [])
        return {"alpha_id": found["alpha_id"], "expression": found["expression"], "settings": found["settings"],
                "recipes": [r["params"]["recipe"] for r in reqs], "requests": reqs}

    async def initial(self, task: LabTask) -> list[TrialSpec]:
        return await written_up_front(task)

    def done_early(self, task: LabTask, ctx: RefillContext) -> str | None:
        return all_run(ctx)

    async def preview(self, params: dict[str, Any]) -> dict[str, Any]:
        found = await plan(params.get("alpha_id") or "")
        return {**found, "sample": [] if found["problems"] else chosen_requests(found, params.get("recipes") or [])[:3]}
