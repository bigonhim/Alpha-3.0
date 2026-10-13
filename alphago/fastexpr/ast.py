"""Immutable abstract syntax tree for FASTEXPR programs.

Every node is a frozen dataclass with structural equality, ``children()`` and
``to_source()``.  ``to_source()`` produces canonical spacing (``f(a, b, k=v)``,
``a + b``) and the minimal parentheses needed to re-parse to an equal tree.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

__all__ = [
    "ATOM_PRECEDENCE",
    "BINARY_PRECEDENCE",
    "TERNARY_PRECEDENCE",
    "UNARY_PRECEDENCE",
    "Assign",
    "BinOp",
    "Bool",
    "Call",
    "Ident",
    "Node",
    "Num",
    "Program",
    "Str",
    "Ternary",
    "Unary",
    "format_number",
    "num",
    "transform",
    "walk",
]

# Binding strength of each infix operator (higher binds tighter).
BINARY_PRECEDENCE: dict[str, int] = {
    "||": 2,
    "&&": 3,
    "==": 4,
    "!=": 4,
    "<": 5,
    "<=": 5,
    ">": 5,
    ">=": 5,
    "+": 6,
    "-": 6,
    "*": 7,
    "/": 7,
}
TERNARY_PRECEDENCE = 1
UNARY_PRECEDENCE = 8
ATOM_PRECEDENCE = 9
UNARY_OPS = frozenset({"-", "+", "!"})


def format_number(value: float) -> str:
    """Render a number as FASTEXPR source: integers without ``.0``, floats via ``repr``."""
    if isinstance(value, bool):
        raise TypeError("booleans are not numbers in FASTEXPR; use Bool")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"FASTEXPR has no literal for {value!r}")
    if v.is_integer() and abs(v) < 1e16:
        return str(int(v))
    return repr(v)


class Node:
    """Base class of all AST nodes."""

    __slots__ = ()

    @property
    def precedence(self) -> int:
        return ATOM_PRECEDENCE

    def to_source(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def children(self) -> tuple[Node, ...]:
        return ()

    def with_children(self, children: tuple[Node, ...]) -> Node:
        """Return a copy of this node with its children replaced (same order as ``children()``)."""
        return self

    def __str__(self) -> str:
        return self.to_source()


@dataclass(frozen=True, slots=True)
class Num(Node):
    """Numeric literal.  ``text`` keeps the original spelling and is ignored by equality."""

    value: float
    text: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(self.value))
        if not self.text:
            object.__setattr__(self, "text", format_number(self.value))

    @property
    def precedence(self) -> int:
        return UNARY_PRECEDENCE if self.text.startswith("-") else ATOM_PRECEDENCE

    @property
    def is_integer(self) -> bool:
        return self.value.is_integer()

    def to_source(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class Str(Node):
    """String literal (FASTEXPR strings have no escape sequences)."""

    value: str

    def to_source(self) -> str:
        if '"' in self.value:
            return f"'{self.value}'"
        return f'"{self.value}"'


@dataclass(frozen=True, slots=True)
class Bool(Node):
    """Boolean literal ``true`` / ``false``."""

    value: bool

    def to_source(self) -> str:
        return "true" if self.value else "false"


@dataclass(frozen=True, slots=True)
class Ident(Node):
    """Identifier: data field, variable, group name or bare enum word."""

    name: str

    def to_source(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Call(Node):
    """Operator call ``name(arg, ..., key=value, ...)``."""

    name: str
    args: tuple[Node, ...] = ()
    kwargs: tuple[tuple[str, Node], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple):
            object.__setattr__(self, "args", tuple(self.args))
        if not isinstance(self.kwargs, tuple) or any(not isinstance(kv, tuple) for kv in self.kwargs):
            object.__setattr__(self, "kwargs", tuple((str(k), v) for k, v in self.kwargs))

    def kwarg(self, key: str) -> Node | None:
        """Value of keyword argument ``key`` (last one wins), or ``None``."""
        found = None
        for k, v in self.kwargs:
            if k == key:
                found = v
        return found

    def to_source(self) -> str:
        parts = [a.to_source() for a in self.args]
        parts.extend(f"{k}={v.to_source()}" for k, v in self.kwargs)
        return f"{self.name}({', '.join(parts)})"

    def children(self) -> tuple[Node, ...]:
        return self.args + tuple(v for _, v in self.kwargs)

    def with_children(self, children: tuple[Node, ...]) -> Node:
        n = len(self.args)
        kwargs = tuple((k, v) for (k, _), v in zip(self.kwargs, children[n:], strict=True))
        return Call(self.name, tuple(children[:n]), kwargs)


@dataclass(frozen=True, slots=True)
class Unary(Node):
    """Prefix operator: ``-x``, ``+x`` or ``!x``."""

    op: str
    operand: Node

    @property
    def precedence(self) -> int:
        return UNARY_PRECEDENCE

    def to_source(self) -> str:
        inner = self.operand.to_source()
        if self.operand.precedence < UNARY_PRECEDENCE:
            inner = f"({inner})"
        return f"{self.op}{inner}"

    def children(self) -> tuple[Node, ...]:
        return (self.operand,)

    def with_children(self, children: tuple[Node, ...]) -> Node:
        (operand,) = children
        return Unary(self.op, operand)


@dataclass(frozen=True, slots=True)
class BinOp(Node):
    """Left-associative infix operator."""

    op: str
    left: Node
    right: Node

    @property
    def precedence(self) -> int:
        return BINARY_PRECEDENCE[self.op]

    def to_source(self) -> str:
        p = self.precedence
        left = self.left.to_source()
        if self.left.precedence < p:
            left = f"({left})"
        right = self.right.to_source()
        if self.right.precedence <= p:
            right = f"({right})"
        return f"{left} {self.op} {right}"

    def children(self) -> tuple[Node, ...]:
        return (self.left, self.right)

    def with_children(self, children: tuple[Node, ...]) -> Node:
        left, right = children
        return BinOp(self.op, left, right)


@dataclass(frozen=True, slots=True)
class Ternary(Node):
    """Right-associative conditional ``cond ? then : other``."""

    cond: Node
    then: Node
    other: Node

    @property
    def precedence(self) -> int:
        return TERNARY_PRECEDENCE

    def to_source(self) -> str:
        cond = self.cond.to_source()
        if self.cond.precedence <= TERNARY_PRECEDENCE:
            cond = f"({cond})"
        then = self.then.to_source()
        if self.then.precedence <= TERNARY_PRECEDENCE:  # not required, but far more readable
            then = f"({then})"
        return f"{cond} ? {then} : {self.other.to_source()}"

    def children(self) -> tuple[Node, ...]:
        return (self.cond, self.then, self.other)

    def with_children(self, children: tuple[Node, ...]) -> Node:
        cond, then, other = children
        return Ternary(cond, then, other)


@dataclass(frozen=True, slots=True)
class Assign(Node):
    """Statement ``name = value``."""

    name: str
    value: Node

    @property
    def precedence(self) -> int:
        return 0

    def to_source(self) -> str:
        return f"{self.name} = {self.value.to_source()}"

    def children(self) -> tuple[Node, ...]:
        return (self.value,)

    def with_children(self, children: tuple[Node, ...]) -> Node:
        (value,) = children
        return Assign(self.name, value)


@dataclass(frozen=True, slots=True)
class Program(Node):
    """A sequence of statements; the last one is the alpha expression."""

    statements: tuple[Node, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.statements, tuple):
            object.__setattr__(self, "statements", tuple(self.statements))

    @property
    def precedence(self) -> int:
        return 0

    @property
    def expression(self) -> Node:
        """The final statement (the alpha expression)."""
        return self.statements[-1]

    @property
    def assignments(self) -> dict[str, Node]:
        """Mapping of variable name to assigned value (a later assignment wins)."""
        return {s.name: s.value for s in self.statements if isinstance(s, Assign)}

    def to_source(self) -> str:
        return "; ".join(s.to_source() for s in self.statements)

    def children(self) -> tuple[Node, ...]:
        return self.statements

    def with_children(self, children: tuple[Node, ...]) -> Node:
        return Program(tuple(children))


def num(x: float) -> Num:
    """Create a :class:`Num` with canonical text (``5`` rather than ``5.0``)."""
    return Num(float(x), format_number(x))


def walk(node: Node) -> Iterator[Node]:
    """Yield ``node`` and all its descendants in pre-order."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children()))


def transform(node: Node, fn: Callable[[Node], Node]) -> Node:
    """Rebuild ``node`` bottom-up, applying ``fn`` to every (already rebuilt) node."""
    children = node.children()
    if children:
        new_children = tuple(transform(c, fn) for c in children)
        if any(a is not b for a, b in zip(children, new_children, strict=True)):
            node = node.with_children(new_children)
    return fn(node)
