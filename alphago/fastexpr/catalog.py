"""Operator and data-field catalog used to validate and analyse FASTEXPR programs.

The default operator table lives in :mod:`alphago.knowledge.operators` (imported
lazily to avoid an import cycle); :func:`alphago.knowledge.operators.build_catalog`
merges it with the operator / data-field lists returned by the BRAIN API.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from .ast import Node

__all__ = [
    "ARG_KINDS",
    "BUILTIN_GROUPS",
    "BUILTIN_MATRIX_FIELDS",
    "DEFAULT_ENUM_WORDS",
    "INFIX_TO_OPERATOR",
    "ArgSpec",
    "BoundArg",
    "Binding",
    "Catalog",
    "FieldInfo",
    "OperatorSpec",
    "bind_call",
    "default_catalog",
]

#: Argument kinds understood by the validator.  ``"any"`` is used for permissive specs.
ARG_KINDS = frozenset({"expr", "group", "vector", "int", "number", "string", "bool", "enum", "any"})

#: Function-form name of every infix / prefix / ternary operator (``"u-"`` is unary minus).
INFIX_TO_OPERATOR: dict[str, str] = {
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide",
    "<": "less",
    "<=": "less_equal",
    ">": "greater",
    ">=": "greater_equal",
    "==": "equal",
    "!=": "not_equal",
    "&&": "and",
    "||": "or",
    "!": "not",
    "u-": "reverse",
    "?:": "if_else",
}

BUILTIN_GROUPS = frozenset({"market", "sector", "industry", "subindustry", "country", "exchange", "currency"})

#: Price/volume fields known as MATRIX even without a loaded field list (non-strict mode only).
BUILTIN_MATRIX_FIELDS = frozenset(
    {"open", "high", "low", "close", "vwap", "volume", "returns", "cap", "adv20", "sharesout"}
)

#: Bare words accepted as keyword-argument values (stored lower-case).
DEFAULT_ENUM_WORDS = frozenset(
    {"gaussian", "uniform", "cauchy", "long", "short", "both", "nan", "zero", "dollar2share", "share2dollar"}
)


@dataclass(frozen=True)
class ArgSpec:
    """One operator parameter.

    ``kind`` is one of ``expr`` (matrix/boolean expression), ``group``, ``vector``,
    ``int`` (positive integer literal such as a lookback window), ``number``, ``string``,
    ``bool``, ``enum`` (bare word from ``choices``) or ``any``.
    """

    name: str
    kind: str = "expr"
    required: bool = True
    default: object = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperatorSpec:
    """Signature and documentation of a FASTEXPR operator.

    ``variadic`` allows extra positional arguments of the last positional kind.  For
    non-variadic operators, surplus positional arguments overflow into ``keywords`` in
    declared order (BRAIN accepts e.g. ``ts_rank(x, 20, 0)``).  ``permissive`` specs
    (API operators without a curated signature) accept any arguments.
    """

    name: str
    category: str
    positional: tuple[ArgSpec, ...]
    keywords: tuple[ArgSpec, ...] = ()
    variadic: bool = False
    min_args: int | None = None
    returns: str = "matrix"
    definition: str = ""
    description: str = ""
    tier: str = "B"
    available: bool = True
    permissive: bool = False

    @property
    def min_positional(self) -> int:
        if self.min_args is not None:
            return self.min_args
        return sum(1 for a in self.positional if a.required)

    @property
    def max_positional(self) -> int | None:
        """Maximum number of positional arguments, or ``None`` when unbounded."""
        if self.variadic or self.permissive:
            return None
        return len(self.positional) + len(self.keywords)

    @property
    def all_args(self) -> tuple[ArgSpec, ...]:
        return self.positional + self.keywords

    @property
    def takes_group(self) -> bool:
        return any(a.kind == "group" for a in self.all_args)

    def find_arg(self, name: str) -> ArgSpec | None:
        """Look up a parameter by name (exact match first, then case-insensitive)."""
        for a in self.all_args:
            if a.name == name:
                return a
        lowered = name.lower()
        for a in self.all_args:
            if a.name.lower() == lowered:
                return a
        return None

    def signature(self) -> str:
        """Human-readable signature, e.g. ``ts_rank(x, d, constant=0)``."""
        parts = [a.name for a in self.positional]
        if self.variadic:
            parts.append("...")
        parts.extend(f"{a.name}={_format_default(a)}" for a in self.keywords)
        return f"{self.name}({', '.join(parts)})"


def _format_default(arg: ArgSpec) -> str:
    value = arg.default
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "..."
    if arg.kind == "string" or (isinstance(value, str) and arg.kind != "enum"):
        return f'"{value}"'
    return str(value)


@dataclass(frozen=True)
class FieldInfo:
    """A BRAIN data field.  ``type`` is MATRIX, VECTOR, GROUP, UNIVERSE or SYMBOL."""

    id: str
    type: str = "MATRIX"
    dataset_id: str = ""
    category: str = ""
    coverage: float | None = None
    alpha_count: int | None = None
    description: str = ""


class Catalog:
    """Operators and data fields available to the validator.

    ``strict_fields`` (default: a non-empty field list was given) turns unknown
    identifiers into errors instead of warnings.  ``strict_operators`` (set when the
    operator list came from the API) makes ``available=False`` operators errors.
    Treat instances as read-only once built; :func:`default_catalog` is shared.
    """

    def __init__(
        self,
        operators: Mapping[str, OperatorSpec] | Iterable[OperatorSpec] | None = None,
        fields: Mapping[str, FieldInfo] | Iterable[FieldInfo] | None = None,
        *,
        strict_fields: bool | None = None,
        strict_operators: bool | None = None,
    ) -> None:
        if operators is None:
            from alphago.knowledge.operators import CURATED_OPERATORS  # lazy: avoids import cycle

            operators = CURATED_OPERATORS
        if isinstance(operators, Mapping):
            self.operators: dict[str, OperatorSpec] = dict(operators)
        else:
            self.operators = {spec.name: spec for spec in operators}

        if fields is None:
            self.fields: dict[str, FieldInfo] = {}
        elif isinstance(fields, Mapping):
            self.fields = dict(fields)
        else:
            self.fields = {f.id: f for f in fields}

        self.strict_fields: bool = bool(self.fields) if strict_fields is None else bool(strict_fields)
        self.strict_operators: bool = bool(strict_operators)
        self.builtin_groups: frozenset[str] = BUILTIN_GROUPS
        choices = {c.lower() for spec in self.operators.values() for a in spec.all_args for c in a.choices}
        self.enum_words: frozenset[str] = DEFAULT_ENUM_WORDS | frozenset(choices)
        self._fields_lower = {k.lower(): v for k, v in self.fields.items()}
        self._ops_lower = {k.lower(): v for k, v in self.operators.items()}

    def __repr__(self) -> str:
        return (
            f"Catalog({len(self.operators)} operators, {len(self.fields)} fields, "
            f"strict_fields={self.strict_fields}, strict_operators={self.strict_operators})"
        )

    def get_operator(self, name: str) -> OperatorSpec | None:
        spec = self.operators.get(name)
        if spec is None:
            spec = self._ops_lower.get(name.lower())
        return spec

    def get_field(self, name: str) -> FieldInfo | None:
        """Field info for ``name``; builtin price fields are known in non-strict mode."""
        info = self.fields.get(name) or self._fields_lower.get(name.lower())
        if info is None and not self.strict_fields and name in BUILTIN_MATRIX_FIELDS:
            info = FieldInfo(name, "MATRIX", dataset_id="pv1", category="pv")
        return info

    def field_type(self, name: str) -> str | None:
        info = self.get_field(name)
        return info.type.upper() if info else None

    def is_builtin_group(self, name: str) -> bool:
        return name in self.builtin_groups

    def is_enum_word(self, name: str) -> bool:
        return name.lower() in self.enum_words

    def is_group_name(self, name: str) -> bool:
        """True for builtin groups and GROUP-typed fields."""
        return name in self.builtin_groups or self.field_type(name) == "GROUP"


@lru_cache(maxsize=1)
def default_catalog() -> Catalog:
    """Shared non-strict catalog built from the curated operator table."""
    return Catalog()


# --------------------------------------------------------------------------- argument binding


@dataclass(frozen=True)
class BoundArg:
    """An argument matched to its parameter; ``slot`` is the positional index or keyword as written."""

    spec: ArgSpec
    node: Node
    slot: int | str


@dataclass
class Binding:
    """Result of :func:`bind_call`: matched arguments plus ``(code, message)`` issues."""

    bound: list[BoundArg] = field(default_factory=list)
    issues: list[tuple[str, str]] = field(default_factory=list)
    unbound: list[Node] = field(default_factory=list)

    def get(self, name: str) -> BoundArg | None:
        for b in self.bound:
            if b.spec.name == name:
                return b
        return None


def bind_call(spec: OperatorSpec, args: Sequence[Node], kwargs: Sequence[tuple[str, Node]]) -> Binding:
    """Match call arguments to ``spec`` the way BRAIN does (Python-like rules)."""
    binding = Binding()
    filled: set[str] = set()
    npos = len(spec.positional)

    for i, node in enumerate(args):
        if i < npos:
            arg = spec.positional[i]
        elif spec.variadic or spec.permissive:
            base = spec.positional[-1] if spec.positional else ArgSpec("x", "any")
            kind = "any" if spec.permissive else base.kind
            arg = ArgSpec(f"{base.name}{i + 1}", kind, required=False)
        elif i - npos < len(spec.keywords):
            arg = spec.keywords[i - npos]
        else:
            binding.unbound.append(node)
            continue
        filled.add(arg.name)
        binding.bound.append(BoundArg(arg, node, i))

    if binding.unbound:
        binding.issues.append(
            (
                "arity",
                f"{spec.name}() takes at most {spec.max_positional} positional argument(s) "
                f"[{spec.signature()}], got {len(args)}",
            )
        )

    for key, node in kwargs:
        arg = spec.find_arg(key)
        if arg is None:
            if spec.permissive:
                arg = ArgSpec(key, "any", required=False)
            else:
                names = ", ".join(a.name for a in spec.all_args) or "none"
                binding.issues.append(
                    ("unknown_kwarg", f"{spec.name}() has no parameter named '{key}' (parameters: {names})")
                )
                binding.unbound.append(node)
                continue
        if arg.name in filled:
            binding.issues.append(
                ("duplicate_kwarg", f"parameter '{arg.name}' of {spec.name}() is given more than once")
            )
            binding.unbound.append(node)
            continue
        filled.add(arg.name)
        binding.bound.append(BoundArg(arg, node, key))

    if not spec.permissive:
        missing = [a.name for a in spec.all_args if a.required and a.name not in filled]
        if missing or (spec.variadic and len(args) < spec.min_positional):
            need = max(spec.min_positional, sum(1 for a in spec.positional if a.required))
            binding.issues.append(
                (
                    "arity",
                    f"{spec.name}() requires at least {need} argument(s) [{spec.signature()}], "
                    f"got {len(args) + len(kwargs)}"
                    + (f"; missing {', '.join(missing)}" if missing else ""),
                )
            )
    return binding
