"""Pure AST rewrites used by the optimizer.

Every function accepts source text, a :class:`Program` or an expression node and
returns a new :class:`Program`; inputs are never mutated (nodes are immutable).
Assignments are preserved and only the relevant nodes are rebuilt.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable

from .ast import Assign, Bool, Call, Ident, Node, Num, Program, Str, Unary, num, transform
from .canonical import as_program
from .catalog import Catalog, OperatorSpec, bind_call, default_catalog
from .parser import FastExprSyntaxError, parse

__all__ = [
    "add_backfill",
    "negate",
    "program_from_expr",
    "replace_identifier",
    "scale_windows",
    "set_kwarg",
    "set_windows",
    "to_node",
    "trade_when_wrap",
    "wrap",
]

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


# --------------------------------------------------------------------------- helpers


def program_from_expr(node: Node | str) -> Program:
    """Wrap a bare expression node (or parse source text) into a :class:`Program`."""
    return as_program(node)


def to_node(value: object, kind: str | None = None) -> Node:
    """Convert a Python value to an AST node.

    Nodes pass through; ``bool`` -> :class:`Bool`; numbers -> :class:`Num` (negative
    numbers become ``Unary('-', Num)`` exactly as the parser produces them); strings
    become :class:`Str` for ``kind="string"``, an :class:`Ident` when they look like an
    identifier, and are otherwise parsed as an expression (e.g. ``"-1"``); unparsable text
    falls back to a string literal when the kind is unknown.
    """
    if isinstance(value, Node):
        return value
    if isinstance(value, bool):
        return Bool(value)
    if isinstance(value, (int, float)):
        if value < 0:
            return Unary("-", num(-value))
        return num(value)
    if isinstance(value, str):
        if kind == "string":
            return Str(value)
        text = value.strip()
        if _IDENT_RE.match(text):
            if text.lower() in ("true", "false"):
                return Bool(text.lower() == "true")
            return Ident(text)
        try:
            return parse(text).expression
        except FastExprSyntaxError:
            if kind in (None, "any"):
                return Str(value)
            raise
    raise TypeError(f"cannot convert {type(value).__name__} to a FASTEXPR node")


def _with_final(program: Program, final: Node, extra: Iterable[Node] = ()) -> Program:
    return Program(program.statements[:-1] + tuple(extra) + (final,))


def _replace_slots(call: Call, replacements: dict[int | str, Node]) -> Call:
    args = tuple(replacements.get(i, a) for i, a in enumerate(call.args))
    kwargs = tuple((k, replacements.get(k, v)) for k, v in call.kwargs)
    return Call(call.name, args, kwargs)


def _spec(name: str, catalog: Catalog | None) -> OperatorSpec | None:
    spec = (catalog or default_catalog()).get_operator(name)
    return None if spec is None or spec.permissive else spec


def _slot_kind(spec: OperatorSpec | None, index: int) -> str | None:
    """Kind of positional slot ``index`` (overflowing into keywords) or ``None``."""
    if spec is None:
        return None
    if index < len(spec.positional):
        return spec.positional[index].kind
    if spec.variadic and spec.positional:
        return spec.positional[-1].kind
    overflow = index - len(spec.positional)
    if overflow < len(spec.keywords):
        return spec.keywords[overflow].kind
    return None


def _keyword_kind(spec: OperatorSpec | None, key: str) -> str | None:
    if spec is None:
        return None
    arg = spec.find_arg(key)
    return arg.kind if arg else None


def _call_slot_kinds(call: Call, catalog: Catalog | None) -> dict[int | str, str]:
    spec = _spec(call.name, catalog)
    if spec is None:
        return {}
    return {b.slot: b.spec.kind for b in bind_call(spec, call.args, call.kwargs).bound}


# --------------------------------------------------------------------------- windows


def _rewrite_windows(program: str | Node, new_value: Callable[[int], int], catalog: Catalog | None) -> Program:
    program = as_program(program)
    literals = {
        name: value
        for name, value in program.assignments.items()
        if isinstance(value, Num) and value.value.is_integer() and value.value > 0
    }
    window_vars: set[str] = set()

    def rewrite(node: Node) -> Node:
        if not isinstance(node, Call):
            return node
        spec = _spec(node.name, catalog)
        if spec is None:
            return node
        replacements: dict[int | str, Node] = {}
        for bound in bind_call(spec, node.args, node.kwargs).bound:
            if bound.spec.kind != "int":
                continue
            arg = bound.node
            if isinstance(arg, Num) and arg.value.is_integer() and arg.value > 0:
                replacements[bound.slot] = num(new_value(int(arg.value)))
            elif isinstance(arg, Ident) and arg.name in literals:
                window_vars.add(arg.name)
        return _replace_slots(node, replacements) if replacements else node

    statements = [transform(stmt, rewrite) for stmt in program.statements]
    statements = [
        Assign(s.name, num(new_value(int(literals[s.name].value))))
        if isinstance(s, Assign) and s.name in window_vars and s.value == literals[s.name]
        else s
        for s in statements
    ]
    return Program(tuple(statements))


def scale_windows(
    program: str | Node,
    factor: float,
    *,
    min_window: int = 2,
    max_window: int = 512,
    catalog: Catalog | None = None,
) -> Program:
    """Multiply every integer window argument (kind ``"int"``) by ``factor``, rounding half up.

    Results are clamped to ``[min_window, max_window]`` but never move against the
    direction of ``factor`` (e.g. ``ts_decay_linear(x, 1)`` stays 1 when shrinking).
    Windows given through a variable bound to a literal (``lookback = 10``) are scaled
    at the assignment.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")

    def new_value(v: int) -> int:
        scaled = int(math.floor(v * factor + 0.5))
        scaled = min(max(scaled, min_window), max_window)
        if (factor >= 1 and scaled < v) or (factor <= 1 and scaled > v):
            scaled = v
        return max(scaled, 1)

    return _rewrite_windows(program, new_value, catalog)


def set_windows(program: str | Node, window: int, *, catalog: Catalog | None = None) -> Program:
    """Set every integer window argument to ``window``."""
    if window < 1:
        raise ValueError("window must be a positive integer")
    return _rewrite_windows(program, lambda _v: int(window), catalog)


# --------------------------------------------------------------------------- wrapping


def wrap(program: str | Node, op_name: str, *extra_args: Node | str | float, **kwargs: object) -> Program:
    """Wrap the final expression: ``wrap(p, "group_neutralize", "subindustry")``.

    Extra positional strings become identifiers (or are parsed as expressions); keyword
    values are converted according to the operator's signature (``range="0.1,1,0.1"``
    becomes a string literal for ``bucket``).
    """
    program = as_program(program)
    spec = _spec(op_name, None)
    args = [program.expression]
    for i, value in enumerate(extra_args, start=1):
        kind = _slot_kind(spec, i)
        args.append(to_node(value, "string" if kind == "string" else None))
    kw = [(key, to_node(value, _keyword_kind(spec, key))) for key, value in kwargs.items()]
    return _with_final(program, Call(op_name, tuple(args), tuple(kw)))


def negate(program: str | Node) -> Program:
    """Negate the final expression, unwrapping ``-x`` / ``reverse(x)`` instead of doubling."""
    program = as_program(program)
    final = program.expression
    if isinstance(final, Unary) and final.op == "-":
        return _with_final(program, final.operand)
    if isinstance(final, Call) and final.name == "reverse" and len(final.args) == 1 and not final.kwargs:
        return _with_final(program, final.args[0])
    return _with_final(program, Unary("-", final))


def trade_when_wrap(program: str | Node, open_event: str | Node, exit_event: str | Node | float = "-1") -> Program:
    """Return ``trade_when(open_event, alpha, exit_event)`` around the final expression.

    Event strings may be full programs; their assignments are inserted before the final
    statement.
    """
    program = as_program(program)
    extra: list[Node] = []

    def event(value: str | Node | float) -> Node:
        if isinstance(value, str):
            value = parse(value)
        if isinstance(value, Program):
            extra.extend(value.statements[:-1])
            return value.expression
        return to_node(value)

    open_node = event(open_event)
    exit_node = event(exit_event)
    return _with_final(program, Call("trade_when", (open_node, program.expression, exit_node)), extra)


# --------------------------------------------------------------------------- substitutions


def replace_identifier(program: str | Node, old: str, new: str) -> Program:
    """Rename identifier ``old`` to ``new`` everywhere (including assignment targets)."""
    program = as_program(program)

    def rewrite(node: Node) -> Node:
        if isinstance(node, Ident) and node.name == old:
            return Ident(new)
        if isinstance(node, Assign) and node.name == old:
            return Assign(new, node.value)
        return node

    result = transform(program, rewrite)
    assert isinstance(result, Program)
    return result


def set_kwarg(program: str | Node, op_name: str, key: str, value: object) -> Program:
    """Set parameter ``key`` on every call of ``op_name`` (replacing it if given, even positionally)."""
    program = as_program(program)
    spec = _spec(op_name, None)
    arg = spec.find_arg(key) if spec else None
    name = arg.name if arg else key
    new_value = to_node(value, arg.kind if arg else None)

    def rewrite(node: Node) -> Node:
        if not isinstance(node, Call) or node.name != op_name:
            return node
        if spec is not None and arg is not None:
            bound = bind_call(spec, node.args, node.kwargs).get(arg.name)
            if bound is not None:
                return _replace_slots(node, {bound.slot: new_value})
        for k, _ in node.kwargs:
            if k == key or k.lower() == name.lower():
                return _replace_slots(node, {k: new_value})
        return Call(node.name, node.args, node.kwargs + ((name, new_value),))

    result = transform(program, rewrite)
    assert isinstance(result, Program)
    return result


def add_backfill(
    program: str | Node,
    fields: Iterable[str],
    lookback: int = 120,
    *,
    catalog: Catalog | None = None,
) -> Program:
    """Wrap occurrences of ``fields`` in ``ts_backfill(field, lookback)``.

    Occurrences that are already the first argument of ``ts_backfill`` are skipped, as
    are identifiers in non-matrix positions (VECTOR / GROUP / constant arguments) and
    names that are program variables.
    """
    program = as_program(program)
    targets = set(fields) - set(program.assignments)

    def rewrite(node: Node, wrap_ok: bool) -> Node:
        if isinstance(node, Ident):
            if wrap_ok and node.name in targets:
                return Call("ts_backfill", (node, num(lookback)))
            return node
        if isinstance(node, Call):
            kinds = _call_slot_kinds(node, catalog)
            if node.name == "ts_backfill" and node.args:
                kinds[0] = "skip"
            args = tuple(rewrite(a, kinds.get(i, "any") in ("expr", "any")) for i, a in enumerate(node.args))
            kwargs = tuple((k, rewrite(v, kinds.get(k, "any") in ("expr", "any"))) for k, v in node.kwargs)
            return Call(node.name, args, kwargs)
        children = node.children()
        if not children:
            return node
        return node.with_children(tuple(rewrite(c, True) for c in children))

    result = rewrite(program, True)
    assert isinstance(result, Program)
    return result
