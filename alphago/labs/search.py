"""Search Lab: first- and second-degree Alphas over chosen datasets, searched for train Sharpe (alpha-harness
labs/search.py and api/search_lab.py).

Every field is cleaned with ``ts_backfill`` first, carried as many days as its category needs. That is data
cleaning, not a degree, and neither is the vector operator a VECTOR field needs before any other operator can read
it. On that base ``B`` the lab writes six shapes:

    cs(B)    ts(B, d)    group(B, g)    cs(ts(B, d))    ts(ts(B, d), d)    group(ts(B, d), g)

A run is a single-objective TPE study asked define-by-run, because the space is conditional: a lookback exists only
where a time-series operator does, a vector operator only for a vector field, and a field only in a universe that
has it. The first pass gives every field one trial, budget permitting; after that the sampler concentrates on what
scores. Every simulation holds the last two years out (testPeriod P2Y0M0D) and is scored on the train years.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Sequence

from ..brain.platform import DEFAULT_TRUNCATION, MAX_DECAY, MAX_SIMULATIONS, SIMULATION_SLOTS, TEST_PERIOD
from ..db.models_labs import LabTask, LabTrial
from . import launch
from .launch import BACKFILL, NO_NEUTRALIZATION, OPERATORS_UNREAD
from .runner import Lab, RefillContext, TrialSpec
from .study import Driver, cover_for, nothing_new
from .tree import OperatorInfo, operator_table

LOOKBACKS = (5, 21, 63, 126, 252)
#: Grouping fields for group operators; they do not count as data fields.
GROUPS = ("market", "sector", "industry", "subindustry")
DECAYS = (0, 3, 5, 7, 10)
TRUNCATION = DEFAULT_TRUNCATION
MAX_CORES = SIMULATION_SLOTS
FAMILIES = ("cs", "ts", "group", "cs_ts", "ts_ts", "group_ts")
FAMILY_LABELS = {"cs": "cs(B)", "ts": "ts(B, d)", "group": "group(B, g)", "cs_ts": "cs(ts(B, d))",
                 "ts_ts": "ts(ts(B, d), d)", "group_ts": "group(ts(B, d), g)"}
#: The second input a two-input shape writes, by slot: a lookback, or a grouping.
SECOND_INPUT = {"ts": "d", "group": "group"}
#: Cleaning operators, and operators whose inputs do not fit a one-field shape.
EXCLUDED = frozenset({
    "ts_backfill", "group_backfill", "ts_count_nans", "ts_step", "hump", "days_from_last_change", "kth_element",
    "ts_corr", "ts_covariance", "ts_regression", "trade_when", "vector_neut", "regression_neut", "vector_proj",
    "regression_proj", "vec_choose", "group_cartesian_product",
})


@dataclass(frozen=True)
class Catalogue:
    """The account's operators that fit each slot of a shape."""

    cs: tuple[str, ...]
    ts: tuple[str, ...]
    group: tuple[str, ...]
    vector: tuple[str, ...]
    #: Options to write after an operator's inputs: {"ts_moment": "k=0"}.
    keywords: dict[str, str]


def _kind(info: OperatorInfo) -> str | None:
    category = info.category.lower().replace("-", " ").replace("_", " ")
    if "vector" in category or info.name.startswith("vec_"):
        return "vector"
    if "group" in category or info.name.startswith("group_"):
        return "group"
    if "time series" in category or info.name.startswith("ts_"):
        return "ts"
    if "cross sectional" in category:
        return "cs"
    return None


def catalogue(operators: list[dict[str, Any]]) -> Catalogue:
    """Sort the account's operators into the slots a shape can use.

    Arity comes from each operator's published definition: one input for cross-sectional and vector operators,
    ``(x, d)`` for time-series, ``(x, group)`` for group operators. The shape writes a lookback or a grouping second:
    an operator wanting another series there would be handed a number, so it is left out."""
    slots: dict[str, list[str]] = {"cs": [], "ts": [], "group": [], "vector": []}
    keywords: dict[str, str] = {}
    for name, info in sorted(operator_table(operators, "REGULAR").items()):
        kind = _kind(info)
        if kind is None or name in EXCLUDED:
            continue
        inputs = 2 if kind in ("ts", "group") else 1
        second = SECOND_INPUT.get(kind)
        if info.required == inputs and (info.maximum is None or info.maximum >= inputs) and \
                (second is None or info.inputs[1:2] == (second,)):
            slots[kind].append(name)
            if info.explicit:
                keywords[name] = ", ".join(f"{key}={value}" for key, value in info.explicit)
    return Catalogue(**{kind: tuple(names) for kind, names in slots.items()}, keywords=keywords)


def families(cat: Catalogue) -> tuple[str, ...]:
    """The shapes this account's operators can fill."""
    has = {"cs": bool(cat.cs), "ts": bool(cat.ts), "group": bool(cat.group)}
    needs = {"cs": ("cs",), "ts": ("ts",), "group": ("group",), "cs_ts": ("cs", "ts"), "ts_ts": ("ts",),
             "group_ts": ("group", "ts")}
    return tuple(f for f in FAMILIES if all(has[n] for n in needs[f]))


# --- expressions ------------------------------------------------------------

def base(field_id: str, vector_op: str | None = None, backfill: int = BACKFILL) -> str:
    """A field as every shape reads it: reduced to a matrix if it is a vector, then backfilled."""
    inner = f"{vector_op}({field_id})" if vector_op else field_id
    return f"ts_backfill({inner}, {backfill})"


def render(params: dict[str, Any], backfill: int = BACKFILL, keywords: dict[str, str] | None = None) -> str:
    """The Fast Expression for one point of the search."""
    b = base(params["field"], params.get("vector_op"), backfill)
    written = keywords or {}

    def call(op: str, *inputs: object) -> str:
        options = written.get(op)
        return f"{op}({', '.join(map(str, inputs))}{', ' + options if options else ''})"

    family = params["family"]
    if family == "cs":
        return call(params["cs"], b)
    if family == "ts":
        return call(params["ts"], b, params["d"])
    if family == "group":
        return call(params["group_op"], b, params["group"])
    if family == "cs_ts":
        return call(params["cs"], call(params["ts"], b, params["d"]))
    if family == "ts_ts":
        return call(params["ts_outer"], call(params["ts"], b, params["d"]), params["d_outer"])
    if family == "group_ts":
        return call(params["group_op"], call(params["ts"], b, params["d"]), params["group"])
    raise ValueError(f"Unknown shape {family!r}")


def field_choices(space: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Per searched universe, the fields it has, in the pool's order (frozen with the task)."""
    absent = space.get("absent") or {}
    choices: dict[str, tuple[str, ...]] = {}
    for universe in space["universes"]:
        missing = set(absent.get(universe) or ())
        choices[universe] = tuple(f for f in space["fields"] if f not in missing)
    return choices


def slots(space: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Every field slot, by the name :func:`suggest` asks it under, with its choices."""
    return {f"field@{universe}": choices for universe, choices in field_choices(space).items()}


def choices_of(space: dict[str, Any], name: str) -> tuple[Any, ...] | None:
    """The choices every name :func:`suggest` can ask is drawn from, so a study rebuilds from the space alone."""
    if name.startswith("field@"):
        return field_choices(space).get(name.split("@", 1)[1])
    lists = {"universe": "universes", "vector_op": "vector", "family": "families", "cs": "cs", "ts": "ts",
             "d": "lookbacks", "ts_outer": "ts", "d_outer": "lookbacks", "group_op": "group", "group": "groups",
             "neutralization": "neutralizations", "decay": "decays"}
    key = lists.get(name)
    return tuple(space.get(key) or ()) if key else None


def coverage(space: dict[str, Any]) -> tuple[str, list[str]]:
    """The parameter the first pass covers, and the fields it gives a trial each."""
    return "field", list(space["fields"])


def first_pass(space: dict[str, Any], field_id: str) -> dict[str, Any]:
    """Fixed choices that try a field once, in the first universe that has it."""
    absent = space.get("absent") or {}
    universe = next((u for u in space["universes"] if field_id not in set(absent.get(u) or ())),
                    space["universes"][0])
    fixed = {f"field@{universe}": field_id}
    if len(space["universes"]) > 1:
        fixed["universe"] = universe
    return fixed


def suggest(trial: Any, space: dict[str, Any], choices: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """One point, asked define-by-run: each choice exists only where its shape needs it.

    Names are shared across shapes (``ts`` and ``d`` are the same slot in every shape that has one), so each name
    keeps a single distribution for the whole run. Fields are the exception, because BRAIN scopes them by universe:
    the universe is chosen first, then a field from that universe's own slot."""
    fields: dict[str, str] = space["fields"]
    universes = list(space["universes"])
    universe = universes[0] if len(universes) == 1 else trial.suggest_categorical("universe", universes)
    field_id = trial.suggest_categorical(f"field@{universe}", choices[universe])
    params: dict[str, Any] = {"universe": universe, "field": field_id}
    if fields[field_id] == "VECTOR":
        params["vector_op"] = trial.suggest_categorical("vector_op", list(space["vector"]))
    family = trial.suggest_categorical("family", list(space["families"]))
    params["family"] = family
    if family in ("cs", "cs_ts"):
        params["cs"] = trial.suggest_categorical("cs", list(space["cs"]))
    if family in ("ts", "cs_ts", "ts_ts", "group_ts"):
        params["ts"] = trial.suggest_categorical("ts", list(space["ts"]))
        params["d"] = trial.suggest_categorical("d", list(space["lookbacks"]))
    if family == "ts_ts":
        params["ts_outer"] = trial.suggest_categorical("ts_outer", list(space["ts"]))
        params["d_outer"] = trial.suggest_categorical("d_outer", list(space["lookbacks"]))
    if family in ("group", "group_ts"):
        params["group_op"] = trial.suggest_categorical("group_op", list(space["group"]))
        params["group"] = trial.suggest_categorical("group", list(space["groups"]))
    neutralizations = list(space["neutralizations"])
    params["neutralization"] = neutralizations[0] if len(neutralizations) == 1 else \
        trial.suggest_categorical("neutralization", neutralizations)
    decays = list(space.get("decays") or ())
    if decays:
        params["decay"] = decays[0] if len(decays) == 1 else trial.suggest_categorical("decay", decays)
    return params


class RandomTrial:
    """Stands in for an Optuna trial, to draw example points without a study."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def suggest_categorical(self, _name: str, choices: Sequence[Any]) -> Any:
        return self.rng.choice(list(choices))


def settings_for(params: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """The simulation settings of one point: the market, its universe and neutralization, the decay drawn or held,
    the truncation (the Truncation Agent's, when asked for) and the two held-out test years."""
    from ..brain.platform import truncation_for
    region, delay = run["region"], int(run["delay"])
    truncation = truncation_for(region, delay, params["universe"]) if run.get("truncation_agent") else \
        float(run.get("truncation") or TRUNCATION)
    settings = {"instrumentType": "EQUITY", "region": region, "delay": delay, "universe": params["universe"],
                "neutralization": params["neutralization"], "decay": int(params.get("decay", run.get("decay", 0))),
                "truncation": truncation, "testPeriod": run.get("test_period") or TEST_PERIOD}
    if run.get("quick"):
        settings["simulationMode"] = "QUICK"
    return settings


def request_for(params: dict[str, Any], run: dict[str, Any], space: dict[str, Any]) -> TrialSpec:
    expression = render(params, (space.get("backfill") or {}).get(params["field"], BACKFILL), space.get("keywords"))
    return TrialSpec(expression, settings_for(params, run), params=dict(params))


# --- planning ----------------------------------------------------------------

def _int_list(values: Any) -> list[int]:
    out: list[int] = []
    for v in values if isinstance(values, (list, tuple)) else str(values or "").split(","):
        if str(v).strip() == "":
            continue
        out.append(int(float(str(v).strip())))
    return out


async def plan(params: dict[str, Any]) -> dict[str, Any]:
    """Everything a task would search, checked, without queueing anything.

    params: region, delay, universe, only_universe, dataset_ids, vector_operators, neutralizations, decays, cores,
    simulations, field_types, min_coverage, field_ids (only these fields), truncation_agent, quick."""
    problems: list[str] = []
    warnings: list[str] = []
    region, delay = str(params.get("region") or "USA"), int(params.get("delay") or 1)
    cores = max(1, min(int(params.get("cores") or MAX_CORES), MAX_CORES))
    operators = await launch.account_operators()
    if not operators:
        problems.append(OPERATORS_UNREAD)
    elif "ts_backfill" not in {o.get("name") for o in operators}:
        problems.append("ts_backfill is not available on this account, so fields can't be cleaned.")
    try:
        decays = list(dict.fromkeys(_int_list(params.get("decays", [0]))))
    except ValueError:
        decays = []
    if not decays or any(not 0 <= d <= MAX_DECAY for d in decays):
        problems.append(f"Decay must be one or more whole numbers from 0 to {MAX_DECAY}.")
        decays = [d for d in decays if 0 <= d <= MAX_DECAY] or [0]

    universe = params.get("universe")
    universes = await launch.synced_universes(region, delay, universe)
    if params.get("only_universe"):
        universes = [u for u in universes if u == universe]
    downloaded = bool(universes)
    chosen_neut = list(params.get("neutralizations") or [])
    neutralizations = launch.neutralizations_for(region, chosen_neut)
    if not chosen_neut:
        problems.append(NO_NEUTRALIZATION)
    elif not neutralizations:
        problems.append(f"BRAIN offers none of the chosen neutralizations in {region}.")

    cat = catalogue(operators)
    vector_ops = [v for v in dict.fromkeys(params.get("vector_operators") or []) if v in cat.vector]
    dataset_ids = list(params.get("dataset_ids") or [])
    pool = launch.Pool({}, (), {}, 0)
    if not dataset_ids:
        problems.append("Choose at least one dataset.")
    elif not downloaded:
        where = f"{region} delay {delay}"
        problems.append((f"{where} {universe} is not downloaded." if params.get("only_universe")
                         else f"No {where} market is downloaded.") + " Sync it in the Data Explorer.")
    else:
        pool = await launch.field_pool(region, delay, universes, dataset_ids, allow_vector=bool(vector_ops),
                                       field_types=params.get("field_types") or None,
                                       min_coverage=params.get("min_coverage"),
                                       only_fields=list(params.get("field_ids") or []) or None)
        if not pool.fields:
            problems.append(("No field in the chosen datasets matches the filter." if params.get("field_types")
                             or params.get("min_coverage") or params.get("field_ids") else
                             "The chosen datasets have no usable fields in this market.")
                            + (" Allow a vector operator to use their vector fields." if pool.vector_skipped else ""))
    if pool.vector_skipped and pool.fields:
        warnings.append(f"{pool.vector_skipped:,} vector fields are left out; allow a vector operator to use them.")
    shapes = families(cat)
    if operators and not shapes:
        problems.append("None of your operators fit a search shape.")

    space = {"fields": pool.fields, "families": list(shapes), "cs": list(cat.cs), "ts": list(cat.ts),
             "group": list(cat.group), "keywords": cat.keywords, "vector": vector_ops, "lookbacks": list(LOOKBACKS),
             "groups": list(GROUPS), "universes": list(pool.universes), "absent": pool.absent,
             "backfill": pool.backfill, "neutralizations": neutralizations, "decays": decays}
    run = {"region": region, "delay": delay, "decay": decays[0], "truncation_agent": bool(params.get("truncation_agent")),
           "quick": bool(params.get("quick"))}
    sample: list[dict[str, Any]] = []
    if not problems:
        choices = field_choices(space)

        def draw(rng: random.Random) -> dict[str, Any]:
            spec = request_for(suggest(RandomTrial(rng), space, choices), run, space)
            return {"expression": spec.expression, "settings": spec.settings}

        sample = launch.preview_samples(draw, tries=5)
    return {"round": cores * 10, "fields": launch.field_counts(pool.fields), "left_out": {"vector": pool.vector_skipped},
            "universes": list(pool.universes), "sample": sample, "problems": problems, "warnings": warnings,
            "space": space, "run": run}


# --- the lab -------------------------------------------------------------------

class SearchLab(Lab):
    name = "search"
    title = "Search Lab"
    objective = "train_sharpe"
    max_cores = MAX_CORES

    choices_of = staticmethod(choices_of)
    coverage = staticmethod(coverage)
    first_pass = staticmethod(first_pass)

    def __init__(self) -> None:
        self.driver = Driver(self)
        self._stopped: dict[int, str] = {}

    def draw(self, trial: Any, task: LabTask) -> tuple[dict[str, Any], TrialSpec | None]:
        space = task.space
        params = suggest(trial, space, self._choices(task))
        return params, request_for(params, (task.params or {}).get("run") or {}, space)

    def _choices(self, task: LabTask) -> dict[str, tuple[str, ...]]:
        cache = getattr(self, "_choice_cache", {})
        self._choice_cache = cache
        if task.id not in cache:
            cache[task.id] = field_choices(task.space)
        return cache[task.id]

    async def plan(self, task: LabTask) -> dict[str, Any]:
        params = dict(task.params or {})
        planned = await plan(params)
        if planned["problems"]:
            raise ValueError(planned["problems"][0])
        space = planned["space"]
        params["run"] = planned["run"]
        params["n_startup_trials"] = launch.startup_trials(len(space["fields"]), task.max_trials, task.cores * 10)
        task.params = params
        task.universe = task.universe or (space["universes"][0] if space["universes"] else None)
        return space

    async def preview(self, params: dict[str, Any]) -> dict[str, Any]:
        planned = await plan(params)
        planned.pop("space", None)
        return planned

    async def refill(self, task: LabTask, want: int, ctx: RefillContext) -> list[TrialSpec]:
        await self.driver.sync(task)
        run = await self.driver.run_for(task)
        from . import runner as _rm
        all_trials = await _rm.runner.trials(task.id)
        cover = cover_for(self, task, want, ctx, all_trials)
        picked, withdrawn = await asyncio.to_thread(self.driver.ask, run, task, want, ctx, cover)
        if not picked:
            self._stopped[task.id] = nothing_new(withdrawn)
        return picked

    async def on_scored(self, task: LabTask, trials: list[LabTrial]) -> None:
        await self.driver.tell_finished(task, trials)

    def nothing_left(self, task: LabTask) -> str:
        return self._stopped.pop(task.id, nothing_new(None))

    def forget(self, task_id: int) -> None:
        self.driver.forget(task_id)
        getattr(self, "_choice_cache", {}).pop(task_id, None)


__all__ = ["MAX_SIMULATIONS", "SearchLab", "catalogue", "families", "plan", "render", "suggest"]
