"""Template Lab: a Fast Expression with ``$variables``, searched for the best train Sharpe (alpha-harness
labs/template.py and api/template_lab.py).

A template is written the way BRAIN reads an Alpha, ``x = ...;`` lines naming its steps and the last line the
Alpha, with a ``$name`` wherever the search chooses. Each variable is defined beside the template: **fields** from
chosen datasets, or **values** typed as a list of numbers, groups, operator names or whole expressions. Written as
``$name(...)`` it is an operator. A name takes one value per Alpha, however often it is written. ``...`` marks where
a signal still has to be written: such a template can be read and counted, not run.

A task freezes its template, each variable's choices and the settings it searches when it is added. The search then
asks, define-by-run: the universe, every variable in the order it is first written, the neutralization and the
investability constraint.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import math
import operator
import random
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator

from ..brain.platform import (
    INVESTABILITY,
    MAX_DECAY,
    MAX_POSITION_REGIONS,
    MAX_TRADE_REGIONS,
    POWER_POOL_MAX_FIELDS,
    POWER_POOL_MAX_OPERATORS,
    TEST_PERIOD,
    TEST_PERIOD_PATTERN,
    UNCOUNTED_OPERATORS,
    truncation_for,
)
from ..db.models_labs import LabTask, LabTrial
from . import launch
from .launch import NO_NEUTRALIZATION, OPERATORS_UNREAD
from .runner import Lab, RefillContext, TrialSpec
from .search import MAX_CORES, TRUNCATION, RandomTrial, catalogue
from .study import Driver, cover_for, nothing_new
from .tree import (
    Node,
    OperatorInfo,
    ParseError,
    arity_problem,
    error_position,
    operator_count,
    operator_table,
    parse,
    strip_comments,
    walk,
    walk_with_parents,
)
from .tree import data_fields as names_read
from .tree import render as write

#: What a variable named one of these starts as. Only a start: it changes like any other.
PRESETS: dict[str, str] = {
    "lookback": "5, 10, 21, 63, 126, 252",
    "fast_lookback": "5, 10, 21",
    "slow_lookback": "63, 126, 252",
    "group": "market, sector, industry, subindustry",
    "weight": "0.5, 1, 2",
    "power": "0.5, 1, 2",
    "ts_op": "ts_rank, ts_zscore, ts_delta, ts_av_diff",
    "group_op": "group_rank, group_zscore, group_neutralize",
}
#: Operators that read price-volume data themselves: a template calling one reads pv1.
PV1_OPERATORS = frozenset({"inst_pnl", "convert"})
MAX_TEXT = 8000
MAX_VARIABLES = 16
MAX_VALUES = 64
HOLE = "..."
_LIST = "values"
LIMITS = [POWER_POOL_MAX_OPERATORS, POWER_POOL_MAX_FIELDS]
INVESTABILITY_LABELS = {"none": "None", "max_trade": "Max Trade", "max_position": "Max Position"}


def takes_max_position(region: str) -> bool:
    return str(region).upper() in MAX_POSITION_REGIONS


@functools.lru_cache(maxsize=4096)
def program(text: str) -> Node:
    """The template as a tree. Raises :class:`ParseError`."""
    return parse(text)


@functools.lru_cache(maxsize=4096)
def _value(text: str) -> Node:
    return parse(text)


def values(text: str) -> list[str]:
    """A typed list, each value as BRAIN will read it. Raises ValueError saying why not.

    Read as the inputs of one call, so a comma inside brackets or quotes stays in its value:
    ``sector, bucket(rank(cap), range="0, 1, 0.1")`` is two values."""
    return list(_listed(text))


@functools.lru_cache(maxsize=4096)
def _listed(text: str) -> tuple[str, ...]:
    text = text.strip().rstrip(",").rstrip()       # a comma left at the end is the next value not typed yet
    if not text:
        raise ValueError("Type its values, separated by commas.")
    try:
        listed = parse(f"{_LIST}({text})")
    except ParseError as exc:
        raise ValueError(re.sub(r" at \d+", "", str(exc))) from exc
    if listed.kind != "call" or listed.value != _LIST or listed.kwargs or not listed.args:
        raise ValueError("Type its values, separated by commas.")
    if any(n.value.startswith("$") for arg in listed.args for _, n in walk(arg)):
        raise ValueError("A value can't hold another $variable.")
    if any(n.value == HOLE for arg in listed.args for _, n in walk(arg)):
        raise ValueError(f"A value can't be {HOLE}: type the signal itself.")
    found = tuple(dict.fromkeys(write(arg) for arg in listed.args))
    if len(found) > MAX_VALUES:
        raise ValueError(f"A variable holds at most {MAX_VALUES} values.")
    return found


# --- how a template writes its variables ------------------------------------

@dataclass
class Use:
    """How a template writes one variable."""

    name: str
    calls: list[int] = field(default_factory=list)   # written as an operator: each call's input count
    inputs: int = 0                                   # written as an input
    in_vector: int = 0                                # ... directly inside a vector operator
    options: int = 0                                  # written as an option's value, driver = $driver


def uses(tree: Node, vector: Callable[[str], bool]) -> dict[str, Use]:
    """Each variable the template writes, in the order it is first written. `vector` says whether a call's operator
    reduces a vector field: one of the account's vector operators, or a variable whose every value is one."""
    found: dict[str, Use] = {}
    for path, node, parent in walk_with_parents(tree):
        if node.kind not in ("name", "call") or not node.value.startswith("$"):
            continue
        use = found.setdefault(node.value[1:], Use(node.value[1:]))
        if node.kind == "call":
            use.calls.append(len(node.args))
            continue
        if parent is not None and path[-1] >= len(parent.args):
            use.options += 1
            continue
        use.inputs += 1
        if parent is not None and parent.kind == "call" and vector(parent.value):
            use.in_vector += 1
    return found


@functools.lru_cache(maxsize=4096)
def assigned(tree: Node) -> frozenset[str]:
    return frozenset(node.value for _, node in walk(tree) if node.kind == "assign")


@functools.lru_cache(maxsize=4096)
def called(tree: Node) -> frozenset[str]:
    return frozenset(node.value for _, node in walk(tree) if node.kind == "call")


@functools.lru_cache(maxsize=4096)
def _names(tree: Node) -> tuple[str, ...]:
    return tuple(names_read(tree))


@functools.lru_cache(maxsize=4096)
def _written(tree: Node) -> dict[str, Use]:
    return uses(tree, lambda _: False)


def template_problems(tree: Node, table: dict[str, OperatorInfo]) -> list[str]:
    """What is wrong with the template itself, before any variable is looked at."""
    found = [f"{node.value} is chosen by the search, so it can't be defined with =. Name the step without $."
             for _, node in walk(tree) if node.kind == "assign" and node.value.startswith("$")]
    if holes(tree):
        found.append(f"Replace each {HOLE} with a signal.")
    found.extend(_calls(tree, table))
    found.extend(_statements(tree))
    return list(dict.fromkeys(found))


def _statements(tree: Node) -> Iterator[str]:
    """BRAIN refuses an Alpha with a statement nothing reads ("Found unused operators")."""
    if tree.kind != "seq":
        return
    steps = tree.args
    for number, step in enumerate(steps[:-1], start=1):
        if step.kind != "assign":
            yield (f"Statement {number} ({_short(step)}) has no name, so BRAIN finds its operators unused. "
                   "Write it as name = …; and use name in a later statement.")
        elif not any(n.kind in ("name", "call") and n.value == step.value
                     for later in steps[number:] for _, n in walk(later)):
            yield (f"{step.value} is never used after it is set, so BRAIN finds its operators unused. "
                   f"Use {step.value} in a later statement, or remove it.")
    if steps and steps[-1].kind == "assign":
        yield f"The last statement sets {steps[-1].value}: end with the expression to simulate."


def _short(step: Node) -> str:
    text = write(step)
    return text if len(text) <= 40 else f"{text[:39]}…"


@functools.lru_cache(maxsize=4096)
def holes(tree: Node) -> int:
    """How many ``...`` are still to be filled with a signal."""
    return sum(1 for _, node in walk(tree) if node.value == HOLE and node.kind in ("name", "call"))


def _calls(tree: Node, table: dict[str, OperatorInfo]) -> Iterator[str]:
    for _, node in walk(tree):
        if node.kind != "call" or node.value.startswith("$") or node.value == HOLE:
            continue
        info = table.get(node.value)
        if info is None:
            yield f"{node.value} is not one of your operators."
        elif (problem := arity_problem(node, info)) is not None:
            yield problem


def _inputs(count: int) -> str:
    return f"{count} input{'' if count == 1 else 's'}"


def _takes(info: OperatorInfo, count: int) -> bool:
    return info.required <= count and (info.maximum is None or count <= info.maximum)


def values_problems(use: Use, listed: list[str], table: dict[str, OperatorInfo]) -> list[str]:
    """Why a variable typed as values can't be written where the template writes it."""
    if use.calls and (use.inputs or use.options):
        return ["It is written both as an operator and as an input. Give each its own name."]
    found: list[str] = []
    if use.calls:
        for value in listed:
            info = table.get(value)
            if info is None:
                found.append(f"{value} is not one of your operators.")
                continue
            found.extend(f"{value} does not take {_inputs(count)}." for count in dict.fromkeys(use.calls)
                         if not _takes(info, count))
        return list(dict.fromkeys(found))
    for value in listed:
        found.extend(_calls(_value(value), table))
    return list(dict.fromkeys(found))


def field_types(use: Use, chosen: list[str] | None) -> list[str]:
    """The field types searched: as chosen, else what the template's writing implies."""
    picked = [t for t in ("MATRIX", "VECTOR") if t in (chosen or ())]
    if picked:
        return picked
    inside = use.inputs > 0 and use.in_vector == use.inputs
    return ["VECTOR"] if inside else ["MATRIX"]


def fields_problems(use: Use, types: list[str], vector: list[str]) -> list[str]:
    """Why a variable of fields can't be written where the template writes it.

    Matrix and vector fields together are read as matrix fields, the vector ones reduced by an operator the search
    chooses. Vector fields alone are the template's to reduce, so each place must be inside a vector operator."""
    if use.calls:
        return ["Fields can't be an operator. Type operator names as its values instead."]
    if use.options:
        return ["Fields can't be an option's value. Type the values instead."]
    found: list[str] = []
    if types == ["VECTOR"]:
        if use.in_vector < use.inputs:
            found.append(f"It holds vector fields only, so write it inside a vector operator, e.g. vec_avg(${use.name}).")
    elif use.in_vector:
        found.append("It is inside a vector operator, so choose Vector fields only.")
    if types == ["MATRIX", "VECTOR"] and not vector:
        found.append("Choose the vector operators its vector fields are reduced with.")
    return found


def data_names(tree: Node, defined: frozenset[str] | set[str]) -> list[str]:
    """Fixed data fields a tree reads: not variables, groups, options or the template's steps."""
    return _fixed(_names(tree), defined)


def _fixed(names: Iterable[str], defined: frozenset[str] | set[str]) -> list[str]:
    skipped = set(defined) | {HOLE}
    return [n for n in names if not n.startswith("$") and n not in skipped]


def value_names(listed: list[str], defined: frozenset[str] | set[str]) -> list[str]:
    return sorted({n for value in listed for n in _fixed(_names(_value(value)), defined)})


# --- arithmetic on numbers, and how large a template's Alphas are ------------------

_ARITHMETIC: dict[str, Callable[[float, float], Any]] = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "^": operator.pow}


def _number(node: Node) -> float | None:
    if node.kind == "num":
        return float(node.value)
    if node.kind == "unary" and node.value == "-" and node.args[0].kind == "num":
        return -float(node.args[0].value)
    return None


def _mapped(node: Node, step: Callable[[Node], Node], value: str | None = None) -> Node:
    """`node` with `step` applied to every input and, if given, a new value: the same node when nothing changed."""
    if value is None and not (node.args or node.kwargs):
        return node
    args = tuple(step(arg) for arg in node.args)
    kwargs = tuple((key, step(arg)) for key, arg in node.kwargs)
    if (value is None or value == node.value) and all(n is o for n, o in zip(args, node.args)) and \
            all(n[1] is o[1] for n, o in zip(kwargs, node.kwargs)):
        return node
    return Node(node.kind, node.value if value is None else value, args, kwargs)


def fold(node: Node) -> Node:
    """Arithmetic between numbers worked out, so ``$lookback / 2`` reaches BRAIN as ``10.5`` rather than as an
    operator Power Pool counts."""
    return _arithmetic(_mapped(node, fold))


def _arithmetic(node: Node) -> Node:
    if node.kind != "binary" or node.value not in _ARITHMETIC:
        return node
    left, right = (_number(arg) for arg in node.args)
    if left is None or right is None:
        return node
    try:
        result = _ARITHMETIC[node.value](left, right)
    except (ZeroDivisionError, OverflowError):
        return node
    if not isinstance(result, float) or not math.isfinite(result):   # a negative to a fractional power is complex
        return node
    number = Node("num", f"{abs(result):.10g}")
    return Node("unary", "-", (number,)) if result < 0 else number


@functools.lru_cache(maxsize=4096)
def _folded(text: str) -> tuple[Node, int, tuple[str, ...]]:
    node = fold(_value(text))
    return node, operator_count(node), tuple(names_read(node))


@dataclass(frozen=True)
class Size:
    """How large a template's Alphas are, as Power Pool counts them: fewest and most."""

    operators: tuple[int, int]
    fields: tuple[int, int]
    holes: int
    read: tuple[str, ...]            # data fields read by name, in the template or in a variable's values
    slots: tuple[str, ...]           # variables standing for a field: defined as fields, or not defined at all

    @property
    def power_pool(self) -> tuple[int, int] | None:
        """Operators and fields every Alpha still has free under Power Pool's limits; None once one is over."""
        free = (POWER_POOL_MAX_OPERATORS - self.operators[1], POWER_POOL_MAX_FIELDS - self.fields[1])
        return free if min(free) >= 0 else None


def size(tree: Node, typed: dict[str, list[str]]) -> Size:
    """Operators and unique data fields, fewest and most over the values variables can take. A variable not typed
    as values stands for one field; written as an operator it is the operator it draws (ts_backfill costs 0)."""
    return _size(tree, tuple((name, tuple(listed)) for name, listed in typed.items()))


@functools.lru_cache(maxsize=4096)
def _size(tree: Node, choices: tuple[tuple[str, tuple[str, ...]], ...]) -> Size:
    typed = {name: list(listed) for name, listed in choices}
    written = _written(tree)
    defined = assigned(tree)
    calls = {name for name, use in written.items() if use.calls}

    def fields_of(names: Iterable[str]) -> set[str]:
        return {n for n in names if n not in defined and n != HOLE}

    def weight(name: str, text: str) -> tuple[int, int]:
        if name in calls:
            return int(text not in UNCOUNTED_OPERATORS), 0
        _, operators, names = _folded(text)
        return operators, len(fields_of(names))

    weights = {name: {text: weight(name, text) for text in listed} for name, listed in typed.items() if listed}

    def counted(chosen: dict[str, str]) -> tuple[int, int]:
        def filled(node: Node) -> Node:
            drawn = chosen.get(node.value[1:]) if node.value.startswith("$") else None
            if drawn is None:
                return _arithmetic(_mapped(node, filled))
            if node.kind == "name":
                return _folded(drawn)[0]
            return _arithmetic(_mapped(node, filled, drawn if node.kind == "call" else None))

        whole = filled(tree)
        return operator_count(whole), len(fields_of(names_read(whole)))

    fewest = {name: min(each, key=each.__getitem__) for name, each in weights.items()}
    most = {name: max(each, key=each.__getitem__) for name, each in weights.items()}
    fewest_ops, fewest_fields = counted(fewest)
    most_ops, most_fields = (fewest_ops, fewest_fields) if most == fewest else counted(most)
    read = set(data_names(tree, defined))
    for name, listed in typed.items():
        if name in written and written[name].inputs:
            read.update(value_names(listed, defined))
    return Size(operators=(fewest_ops, most_ops),
                fields=(min(fewest_fields, most_fields), max(fewest_fields, most_fields)),
                holes=holes(tree), read=tuple(sorted(read)),
                slots=tuple(n for n, use in written.items() if use.inputs and n not in typed))


def single_dataset(tree: Node, sized: Size, datasets: dict[str, str],
                   slot_datasets: dict[str, frozenset[str] | None]) -> bool:
    """Whether every Alpha the template makes reads one dataset: BRAIN's single dataset Alpha."""
    sources: list[frozenset[str] | None] = [frozenset({datasets[n]}) if n in datasets else None for n in sized.read]
    sources.extend(slot_datasets.get(slot) for slot in sized.slots)
    if called(tree) & PV1_OPERATORS:
        sources.append(frozenset({"pv1"}))
    if sized.holes or not sources:
        return False
    if len(sources) == 1:
        return True
    known = [s for s in sources if s is not None]
    return len(known) == len(sources) and len(frozenset().union(*known)) == 1


def most_pyramids(tree: Node, sized: Size, typed: dict[str, list[str]], categories: dict[str, str],
                  slot_categories: dict[str, frozenset[str] | None]) -> int | None:
    """The most dataset categories one Alpha the template makes reads (BRAIN's pyramids, grouping fields aside).
    None until every field's category is known."""
    defined = assigned(tree)
    fixed = data_names(tree, defined)
    if any(n not in categories for n in fixed):
        return None
    base = {categories[n] for n in fixed}
    written = _written(tree)
    choices: list[list[set[str]]] = []
    for name, listed in typed.items():
        if name in written and written[name].inputs:
            names = [_fixed(_names(_value(v)), defined) for v in listed]
            if any(n not in categories for each in names for n in each):
                return None
            choices.append([{categories[n] for n in each} for each in names])
    for slot in sized.slots:
        found = slot_categories.get(slot)
        if found is None:
            return None
        choices.append([{c} for c in found])
    reachable = base.union(*(c for options in choices for c in options))
    added = sum(max((len(c - base) for c in options), default=0) for options in choices)
    return min(len(reachable), len(base) + added)


# --- the search ---------------------------------------------------------------

def field_choices(space: dict[str, Any]) -> dict[str, dict[str, tuple[str, ...]]]:
    """Per variable of fields, per universe, the fields it can take there (frozen with the task)."""
    found: dict[str, dict[str, tuple[str, ...]]] = {}
    for variable in space["variables"]:
        if variable["kind"] != "fields":
            continue
        absent = variable.get("absent") or {}
        found[variable["name"]] = {u: tuple(f for f in variable["fields"] if f not in set(absent.get(u) or ()))
                                   for u in space["universes"]}
    return found


def slots(space: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    return {f"${name}@{u}": ch for name, by_u in field_choices(space).items() for u, ch in by_u.items()}


def choices_of(space: dict[str, Any], name: str) -> tuple[Any, ...] | None:
    """The choices every name :func:`suggest` can ask is drawn from, so a study rebuilds from the space alone."""
    if name in ("universe", "neutralization", "investability"):
        return tuple(space.get({"universe": "universes", "neutralization": "neutralizations",
                                "investability": "investability"}[name]) or ())
    by_name = {v["name"]: v for v in space.get("variables") or []}
    if name.startswith("$") and "@" in name:
        var, universe = name[1:].split("@", 1)
        return field_choices(space).get(var, {}).get(universe)
    if name.startswith("$") and name.endswith("#vector"):
        v = by_name.get(name[1:-len("#vector")])
        return tuple(v.get("vector") or ()) if v else None
    if name.startswith("$"):
        v = by_name.get(name[1:])
        return tuple(v.get("values") or ()) if v and v["kind"] == "values" else None
    return None


def _first_fields(space: dict[str, Any]) -> dict[str, Any] | None:
    return next((v for v in space["variables"] if v["kind"] == "fields"), None)


def coverage(space: dict[str, Any]) -> tuple[str, list[str]]:
    """The first variable of fields, whose every field (held by a searched universe) the first pass tries once."""
    first = _first_fields(space)
    if not first:
        return "", []
    absent = first.get("absent") or {}
    missing = [set(absent.get(u) or ()) for u in space["universes"]]
    return f"${first['name']}", [f for f in first["fields"] if any(f not in m for m in missing)]


def first_pass(space: dict[str, Any], field_id: str) -> dict[str, Any]:
    first = _first_fields(space) or {}
    absent = first.get("absent") or {}
    universe = next((u for u in space["universes"] if field_id not in set(absent.get(u) or ())),
                    space["universes"][0])
    fixed = {f"${first.get('name')}@{universe}": field_id}
    if len(space["universes"]) > 1:
        fixed["universe"] = universe
    return fixed


def _ask(trial: Any, name: str, choices: list[Any]) -> Any:
    """Only a real choice is asked, so a single value never becomes a dimension."""
    return choices[0] if len(choices) == 1 else trial.suggest_categorical(name, choices)


def suggest(trial: Any, space: dict[str, Any], choices: dict[str, dict[str, tuple[str, ...]]]) -> dict[str, Any]:
    """One point, asked define-by-run in a fixed order so every name keeps one distribution."""
    universe = _ask(trial, "universe", list(space["universes"]))
    params: dict[str, Any] = {"universe": universe}
    for variable in space["variables"]:
        key = f"${variable['name']}"
        if variable["kind"] == "fields":
            field_id = trial.suggest_categorical(f"{key}@{universe}", choices[variable["name"]][universe])
            params[key] = field_id
            if variable["fields"][field_id] == "VECTOR" and variable.get("vector"):
                params[f"{key}#vector"] = _ask(trial, f"{key}#vector", list(variable["vector"]))
        else:
            params[key] = _ask(trial, key, list(variable["values"]))
    params["neutralization"] = _ask(trial, "neutralization", list(space["neutralizations"]))
    params["investability"] = _ask(trial, "investability", list(space["investability"]))
    return params


def render(text: str, space: dict[str, Any], params: dict[str, Any]) -> str:
    """The Fast Expression for one point: variables filled in, numbers folded, the last line as an expression."""
    kinds = {v["name"]: v["kind"] for v in space["variables"]}

    def chosen(name: str) -> Node:
        value = params[f"${name}"]
        if kinds.get(name) != "fields":
            return _value(value)
        vector = params.get(f"${name}#vector")
        return Node("call", vector, (Node("name", value),)) if vector else Node("name", value)

    def filled(node: Node) -> Node:
        if node.kind == "name" and node.value.startswith("$"):
            return chosen(node.value[1:])
        name = params[node.value] if node.kind == "call" and node.value.startswith("$") else node.value
        return replace(node, value=name, args=tuple(filled(a) for a in node.args),
                       kwargs=tuple((k, filled(a)) for k, a in node.kwargs))

    tree = fold(filled(program(text)))
    steps = list(tree.args) if tree.kind == "seq" else [tree]
    if steps[-1].kind == "assign":
        steps[-1] = steps[-1].args[0]
    return write(Node("seq", args=tuple(steps)) if len(steps) > 1 else steps[0])


def settings_for(params: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    region, delay = run["region"], int(run["delay"])
    trade, position = INVESTABILITY[params["investability"]]
    truncation = truncation_for(region, delay, params["universe"]) if run.get("truncation_agent") else \
        float(run.get("truncation") if run.get("truncation") is not None else TRUNCATION)
    settings = {"instrumentType": "EQUITY", "region": region, "delay": delay, "universe": params["universe"],
                "neutralization": params["neutralization"], "decay": int(run.get("decay") or 0),
                "truncation": truncation, "pasteurization": run.get("pasteurization") or "ON",
                "nanHandling": run.get("nan_handling") or "ON", "testPeriod": run.get("test_period") or TEST_PERIOD,
                "maxTrade": trade}
    if takes_max_position(region):
        settings["maxPosition"] = position
    if run.get("quick"):
        settings["simulationMode"] = "QUICK"
    return settings


def draw(trial: Any, run: dict[str, Any], space: dict[str, Any],
         choices: dict[str, dict[str, tuple[str, ...]]]) -> tuple[dict[str, Any], TrialSpec | None]:
    """One point and its simulation; none when two variables of fields landed on one field.

    Two names for fields mean two different fields: a template uses one field twice by writing the same name twice.
    A variable with a single field is no choice: refusing it would refuse every draw."""
    params = suggest(trial, space, choices)
    drawn = [params[f"${v['name']}"] for v in space["variables"] if v["kind"] == "fields" and len(v["fields"]) > 1]
    if len(set(drawn)) < len(drawn):
        return params, None
    return params, TrialSpec(render(run["template"], space, params), settings_for(params, run), params=dict(params))


def combinations(space: dict[str, Any]) -> int:
    """How many different simulations the search space holds."""
    sizes = [len(space["universes"]), len(space["neutralizations"]), len(space["investability"])]
    for variable in space["variables"]:
        if variable["kind"] == "fields":
            kinds = list(variable["fields"].values())
            wraps = max(1, len(variable.get("vector") or []))
            sizes.append(kinds.count("MATRIX") + kinds.count("VECTOR") * wraps)
        else:
            sizes.append(len(variable["values"]))
    return math.prod(sizes)


# --- templates saved while they were built from blocks --------------------------------

_V1_CHOICE = re.compile(r"\{\s*([a-z][a-z0-9_]*(?:\s+OR\s+[a-z][a-z0-9_]*)+)\s*\}")
_V1_VARIABLE = re.compile(r"\b(FAST_LOOKBACK|SLOW_LOOKBACK|LOOKBACK|FIELD|GROUP|WEIGHT|POWER)(?:[ \t]+([A-D]))?\b")


def from_blocks(skeleton: str) -> tuple[str, dict[str, dict[str, str]]]:
    """A template saved as blocks, as text, with the variables its choice blocks become.

    ``FIELD A`` is ``$field`` and ``LOOKBACK B`` is ``$lookback_b``; each choice block, ``{ts_rank OR ts_zscore}``,
    becomes ``$op1`` listing its operators."""
    variables: dict[str, dict[str, str]] = {}

    def choice(match: re.Match[str]) -> str:
        name = f"op{len(variables) + 1}"
        variables[name] = {"kind": "values", "values": ", ".join(re.split(r"\s+OR\s+", match.group(1)))}
        return f"${name}"

    def variable(match: re.Match[str]) -> str:
        tag = (match.group(2) or "A").lower()
        return f"${match.group(1).lower()}" + ("" if tag == "a" else f"_{tag}")

    return _V1_VARIABLE.sub(variable, _V1_CHOICE.sub(choice, skeleton)), variables


# --- the editor's live lint ---------------------------------------------------------

_DOLLAR = re.compile(r"\$([A-Za-z_]\w*)")


def names_in(text: str) -> list[str]:
    """Every ``$name`` the template writes, in the order first written (comments ignored)."""
    return list(dict.fromkeys(_DOLLAR.findall(strip_comments(text))))


def _brackets(text: str) -> list[dict[str, Any]]:
    clean = strip_comments(text)
    found, stack, quote = [], [], None
    for i, c in enumerate(clean):
        if quote:
            if c == quote:
                quote = None
            continue
        if c in "\"'":
            quote = c
        elif c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                stack.pop()
            else:
                found.append({"message": "This ) closes nothing.", "position": i})
    found.extend({"message": "This ( is never closed.", "position": i} for i in stack)
    return found


def line_col(text: str, position: int) -> tuple[int, int]:
    return text.count("\n", 0, position) + 1, position - (text.rfind("\n", 0, position) + 1) + 1


def lint(text: str, table: dict[str, OperatorInfo]) -> list[dict[str, Any]]:
    """What the editor marks while typing: brackets, where the parser stopped, unknown operators, holes and the
    template's own problems. Each {message, position?, line?, column?, level: error|warning}."""
    out: list[dict[str, Any]] = []
    if not text.strip():
        return out
    if len(text) > MAX_TEXT:
        out.append({"message": f"A template holds at most {MAX_TEXT:,} characters.", "level": "error"})
    brackets = _brackets(text)
    for b in brackets:
        out.append({**b, "level": "error"})
    try:
        tree = program(text)
    except ParseError as exc:
        position = error_position(str(exc))
        item = {"message": f"The template can't be read: {exc}", "level": "error"}
        if position is not None:
            item["position"] = position
        if not brackets:
            out.append(item)
    else:
        for problem in template_problems(tree, table) if table else template_problems(tree, {}):
            if not table and "is not one of your operators" in problem:
                continue
            level = "warning" if problem.startswith("Replace each") else "error"
            pos = _position_of(text, problem)
            out.append({"message": problem, "level": level, **({"position": pos} if pos is not None else {})})
        if len(_written(tree)) > MAX_VARIABLES:
            out.append({"message": f"A template holds at most {MAX_VARIABLES} variables.", "level": "error"})
    for item in out:
        if item.get("position") is not None:
            item["line"], item["column"] = line_col(text, int(item["position"]))
    return out


def _position_of(text: str, problem: str) -> int | None:
    """Where a problem's operator or name is first written, so the editor can point at it."""
    m = re.match(r"([$\w.]+) (?:is not one of your operators|does not take|takes|is never used|is chosen)", problem)
    if not m:
        if problem.startswith("Replace each"):
            i = text.find(HOLE)
            return i if i >= 0 else None
        return None
    i = strip_comments(text).find(m.group(1))
    return i if i >= 0 else None


# --- variables, defaults by name ---------------------------------------------------

def default_of(name: str, presets: dict[str, str], datasets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """What a name means before anyone defines it: a preset's values, or the fields of the dataset, subcategory or
    category it is named after. ``$earnings4_b`` reads like ``$earnings4``, so a second field from one dataset is a
    suffix away."""
    for stem in dict.fromkeys([name, re.sub(r"_[a-z0-9]+$", "", name, flags=re.IGNORECASE)]):
        preset = presets.get(stem)
        if preset:
            return {"kind": "values", "values": preset}
        if any(d.get("id") == stem for d in datasets):
            return {"kind": "fields", "dataset_ids": [stem]}
        for level in ("subcategory_id", "category_id"):
            ids = [d["id"] for d in datasets if str(d.get(level) or "").replace("-", "_") == stem]
            if ids:
                return {"kind": "fields", "dataset_ids": ids}
    return None


def resolve(text: str, given: dict[str, dict[str, Any]], datasets: list[dict[str, Any]],
            presets: dict[str, str] = PRESETS) -> dict[str, dict[str, Any]]:
    """What each ``$name`` in `text` stands for: given here, else its default; undefined ones left out."""
    out: dict[str, dict[str, Any]] = {}
    for name in names_in(text):
        d = given.get(name) or default_of(name, presets, datasets)
        if d:
            out[name] = d
    return out


# --- stats and planning -------------------------------------------------------------

def _typed(variables: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    typed: dict[str, list[str]] = {}
    for name, v in variables.items():
        if (v or {}).get("kind") == "values":
            with contextlib.suppress(ValueError):
                typed[name] = values(str(v.get("values") or ""))
    return typed


def _missing(tree: Node, known: set[str]) -> list[str]:
    named = {n for n in called(tree) if not n.startswith("$")}
    return sorted(named - known - {HOLE})


async def stats(items: list[dict[str, Any]], region: str, delay: int,
                operators: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Each template's operators and fields against Power Pool's limits, and whether its Alphas are Power Pool,
    single dataset ones, how many pyramids at most, and the operators it calls that the account lacks.

    items: [{text, variables}]. Free; reads only the local catalogue."""
    operators = await launch.account_operators() if operators is None else operators
    known = {str(o.get("name")) for o in operators}
    read: list[tuple[Node, Size] | str] = []
    for i, item in enumerate(items):
        try:
            tree = program(str(item.get("text") or ""))
        except ParseError as exc:
            read.append(str(exc))
            continue
        read.append((tree, size(tree, _typed(item.get("variables") or {}))))
        if i % 10 == 9:
            await asyncio.sleep(0)
    names = sorted({n for entry in read if not isinstance(entry, str) for n in entry[1].read})
    datasets = await launch.datasets_of_fields(region, delay, names) if names else {}
    chosen = {d for item in items for v in (item.get("variables") or {}).values()
              if (v or {}).get("kind") == "fields" for d in v.get("dataset_ids") or []}
    categories = await launch.dataset_categories({*datasets.values(), *chosen})
    by_field = {f: categories[d] for f, d in datasets.items() if d in categories}
    out: list[dict[str, Any]] = []
    for item, entry in zip(items, read):
        if isinstance(entry, str):
            out.append({"operators": [0, 0], "fields": [0, 0], "holes": 0, "power_pool": None, "limits": LIMITS,
                        "single_dataset": False, "pyramids": None, "missing": [], "problem": entry})
            continue
        tree, sized = entry
        variables = item.get("variables") or {}
        slot_ds = {name: frozenset(v.get("dataset_ids") or []) if (v or {}).get("kind") == "fields"
                   and v.get("dataset_ids") else None for name, v in variables.items()}
        slot_cats = {name: frozenset(categories[d] for d in ids) if (ids := slot_ds.get(name)) and
                     all(d in categories for d in ids) else None for name in sized.slots}
        free = sized.power_pool
        out.append({"operators": list(sized.operators), "fields": list(sized.fields), "holes": sized.holes,
                    "power_pool": list(free) if free is not None else None, "limits": LIMITS,
                    "single_dataset": single_dataset(tree, sized, datasets, slot_ds),
                    "pyramids": most_pyramids(tree, sized, _typed(variables), by_field, slot_cats),
                    "missing": _missing(tree, known) if known else [], "problem": None})
    return out


_EXAMPLE_INPUTS = {"x": "rank(x)", "d": "20", "lookback": "20", "group": "industry"}


def _example(call: str, info: OperatorInfo | None) -> str:
    inputs = info.inputs if info is not None else ("x", "d")
    return f"{call}({', '.join(_EXAMPLE_INPUTS.get(name, name) for name in inputs)})"


async def plan(body: dict[str, Any]) -> dict[str, Any]:
    """Everything a task would search, checked, without queueing anything.

    body: region, delay, template, variables{name: def}, universes, neutralizations, investability, decay,
    truncation, truncation_agent, pasteurization, nan_handling, test_period, cores, simulations, template_name."""
    template_problems_: list[str] = []
    settings_problems: list[str] = []
    warnings: list[str] = []
    region, delay = str(body.get("region") or "USA"), int(body.get("delay") or 1)
    market = f"{region} delay {delay}"
    text = str(body.get("template") or "")

    operators = await launch.account_operators()
    if not operators:
        settings_problems.append(OPERATORS_UNREAD)
    table = operator_table(operators)
    vector = set(catalogue(operators).vector)

    tree: Node | None = None
    if not text.strip():
        template_problems_.append("Write a template.")
    elif len(text) > MAX_TEXT:
        template_problems_.append(f"A template holds at most {MAX_TEXT:,} characters.")
    else:
        try:
            tree = program(text)
        except ParseError as exc:
            template_problems_.append(f"The template can't be read: {exc}")

    decay = int(body.get("decay") or 0)
    if not 0 <= decay <= MAX_DECAY:
        settings_problems.append(f"Decay must be a whole number from 0 to {MAX_DECAY}.")
    truncation = float(body.get("truncation") if body.get("truncation") is not None else TRUNCATION)
    if not 0 <= truncation <= 1:
        settings_problems.append("Truncation must be from 0 to 1.")
    test_period = str(body.get("test_period") or TEST_PERIOD)
    if not TEST_PERIOD_PATTERN.match(test_period):
        settings_problems.append("Test Period must be P{years}Y{months}M0D, at most P6Y0M0D.")
    offered = await launch.synced_universes(region, delay)
    if not offered:
        settings_problems.append(f"No {market} market is downloaded. Sync it in BRAIN › Sync.")
    universes = [u for u in offered if u in set(body.get("universes") or [])]
    if not universes:
        settings_problems.append("Choose at least one Universe.")
    neutralizations = launch.neutralizations_for(region, body.get("neutralizations") or []) \
        if body.get("neutralizations") else []
    if not neutralizations:
        settings_problems.append(NO_NEUTRALIZATION)
    investable = [n for n in INVESTABILITY if n != "max_position" or takes_max_position(region)]
    investability = [n for n in investable if n in set(body.get("investability") or [])]
    if not investability:
        settings_problems.append("Choose at least one Investability.")
    elif region in MAX_TRADE_REGIONS and investability != ["max_trade"]:
        warnings.append(f"In {region}, an Alpha without Max Trade is only submittable if it keeps 70% of its Sharpe "
                        "under investability constraints.")

    given: dict[str, dict[str, Any]] = dict(body.get("variables") or {})

    def vector_call(op: str) -> bool:
        if not op.startswith("$"):
            return op in vector
        d = given.get(op[1:])
        if not d or d.get("kind") != "values":
            return False
        try:
            return all(v in vector for v in values(str(d.get("values") or "")))
        except ValueError:
            return False

    used = uses(tree, vector_call) if tree is not None else {}
    if tree is not None:
        template_problems_.extend(template_problems(tree, table))
    if len(used) > MAX_VARIABLES:
        template_problems_.append(f"A template holds at most {MAX_VARIABLES} variables.")
    defined = assigned(tree) if tree is not None else frozenset()
    searched = universes or offered

    infos: list[dict[str, Any]] = []
    space_variables: list[dict[str, Any]] = []
    reads: dict[str, set[str | None]] = {}
    for name in data_names(tree, defined) if tree is not None else []:
        reads.setdefault(name, set()).add(None)
    pools: dict[str, launch.Pool] = {}
    for name, use in list(used.items())[:MAX_VARIABLES]:
        definition = given.get(name)
        info: dict[str, Any] = {"name": name, "kind": (definition or {}).get("kind", ""), "operator": bool(use.calls),
                                "fields": None, "field_types": [], "values": 0, "problems": []}
        infos.append(info)
        if not definition:
            info["problems"].append("Define it: choose its fields, or type its values.")
        elif definition.get("kind") == "values":
            try:
                listed = values(str(definition.get("values") or ""))
            except ValueError as exc:
                info["problems"].append(str(exc))
                continue
            info["values"] = len(listed)
            info["problems"].extend(values_problems(use, listed, table))
            if use.inputs:
                for r in value_names(listed, defined):
                    reads.setdefault(r, set()).add(name)
            space_variables.append({"name": name, "kind": "values", "values": listed})
        else:
            types = field_types(use, definition.get("field_types") or None)
            info["field_types"] = types
            wraps = [v for v in dict.fromkeys(definition.get("vector_operators") or []) if v in vector]
            info["problems"].extend(fields_problems(use, types, wraps))
            ids = list(definition.get("dataset_ids") or [])
            if not ids:
                info["problems"].append("Choose its datasets.")
                continue
            if not searched:
                continue
            pool = await launch.field_pool(region, delay, searched, ids, allow_vector="VECTOR" in types,
                                           field_types=types, min_coverage=definition.get("min_coverage"))
            pools[name] = pool
            info["fields"] = launch.field_counts(pool.fields)
            if not pool.fields:
                info["problems"].append(f"No field in its datasets matches in {market}.")
            space_variables.append({"name": name, "kind": "fields", "fields": pool.fields, "absent": pool.absent,
                                    "vector": wraps if types == ["MATRIX", "VECTOR"] else []})

    held = await launch.universes_holding(region, delay, set(reads)) if reads and searched else {}

    def problems_of(reader: str | None) -> list[str]:
        if reader is None:
            return template_problems_
        return next(i for i in infos if i["name"] == reader)["problems"]

    operator_names = {str(o.get("name")) for o in operators}
    misplaced: dict[str | None, list[str]] = {}
    for name, readers in reads.items() if searched else ():
        if held.get(name):
            continue
        for reader in readers:
            if name in operator_names:
                misplaced.setdefault(reader, []).append(name)
            elif any(i["name"] == reader for i in infos) or reader is None:
                problems_of(reader).append(f"{name} is not a data field in {market}.")
    for reader, names in misplaced.items():
        one = len(names) == 1
        what = f"{', '.join(names)} {'is an operator' if one else 'are operators'}"
        it = "it" if one else "they"
        example = _example(f"${reader}" if reader else names[0], table.get(names[0]))
        problems_of(reader).append(f"{what}: write ${reader}(…) around what {it} should wrap, e.g. {example}."
                                   if reader else f"{what}: call {it} on what {it} should wrap, e.g. {example}.")
    kept: list[str] = []
    for universe in universes:
        short = [n for n, there in held.items() if there and universe not in there]
        empty = [n for n, pool in pools.items() if pool.fields and universe not in pool.universes]
        if short or empty:
            warnings.append(f"{universe} is left out: it has no {', '.join([*short, *(f'${n}' for n in empty)])}.")
        else:
            kept.append(universe)
    if universes and not kept:
        settings_problems.append("No ticked Universe has everything the template reads.")

    variable_problems = [f"${i['name']}: {p}" for i in infos for p in i["problems"]]
    template_problems_ = list(dict.fromkeys(template_problems_))
    settings_problems = list(dict.fromkeys(settings_problems))
    problems = [*template_problems_, *variable_problems, *settings_problems]
    space = {"universes": kept, "neutralizations": neutralizations, "investability": investability,
             "variables": space_variables}
    run = {"region": region, "delay": delay, "template": text, "decay": decay, "truncation": truncation,
           "truncation_agent": bool(body.get("truncation_agent")), "pasteurization": body.get("pasteurization") or "ON",
           "nan_handling": body.get("nan_handling") or "ON", "test_period": test_period,
           "quick": bool(body.get("quick"))}
    sample: list[dict[str, Any]] = []
    if not problems:
        picks = field_choices(space)

        def draw_one(rng: random.Random) -> dict[str, Any] | None:
            _, spec = draw(RandomTrial(rng), run, space, picks)
            return {"expression": spec.expression, "settings": spec.settings} if spec else None

        sample = launch.preview_samples(draw_one, tries=25)   # a draw can land two variables on one field
    own = {"text": text, "variables": given}
    return {"stats": (await stats([own], region, delay, operators))[0] if tree is not None else None,
            "variables": infos, "universes": offered, "investability": investable,
            "combinations": combinations(space) if not problems else 0, "sample": sample, "problems": problems,
            "template_problems": template_problems_, "settings_problems": settings_problems, "warnings": warnings,
            "space": space, "run": run,
            "dataset_ids": sorted({d for v in given.values() if (v or {}).get("kind") == "fields"
                                   for d in v.get("dataset_ids") or []})}


class TemplateLab(Lab):
    name = "template"
    title = "Template Lab"
    objective = "train_sharpe"
    max_cores = MAX_CORES

    choices_of = staticmethod(choices_of)
    coverage = staticmethod(coverage)
    first_pass = staticmethod(first_pass)

    def __init__(self) -> None:
        self.driver = Driver(self)
        self._stopped: dict[int, str] = {}
        self._choice_cache: dict[int, dict[str, dict[str, tuple[str, ...]]]] = {}

    def draw(self, trial: Any, task: LabTask) -> tuple[dict[str, Any], TrialSpec | None]:
        if task.id not in self._choice_cache:
            self._choice_cache[task.id] = field_choices(task.space)
        return draw(trial, (task.params or {})["run"], task.space, self._choice_cache[task.id])

    async def plan(self, task: LabTask) -> dict[str, Any]:
        params = dict(task.params or {})
        planned = await plan(params)
        if planned["problems"]:
            raise ValueError(planned["problems"][0])
        params["run"] = planned["run"]
        params["dataset_ids"] = planned["dataset_ids"]
        _, covered = coverage(planned["space"])
        params["n_startup_trials"] = launch.startup_trials(len(covered), task.max_trials, task.cores * 10)
        task.params = params
        from .objectives import objective_for
        task.objective = params.get("objective") or objective_for("template", planned["run"]["test_period"])
        task.universe = planned["space"]["universes"][0] if planned["space"]["universes"] else None
        return planned["space"]

    async def preview(self, params: dict[str, Any]) -> dict[str, Any]:
        planned = await plan(params)
        planned.pop("space", None)
        return planned

    async def refill(self, task: LabTask, want: int, ctx: RefillContext) -> list[TrialSpec]:
        await self.driver.sync(task)
        run = await self.driver.run_for(task)
        from . import runner as _rm
        cover = cover_for(self, task, want, ctx, await _rm.runner.trials(task.id))
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
        self._choice_cache.pop(task_id, None)
