"""Structural metrics of FASTEXPR programs (operator counts, fields, windows, size).

Counting conventions:

* ``operator_count`` counts every call and every infix / prefix / ternary operator,
  repeats included (BRAIN Power Pool rule "at most 8 operators").  Operators are counted
  as written: an assignment's operators count once, however often the variable is
  used.  A negative literal such as ``-1`` is a number, not a ``reverse`` call, and
  unary ``+`` is ignored.
* ``node_count`` counts the nodes as written (statement wrappers excluded) while
  ``depth`` is the nesting depth of the final expression with variables inlined
  (a leaf has depth 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from .ast import Assign, BinOp, Call, Ident, Node, Num, Ternary, Unary, walk
from .canonical import as_program, inline
from .catalog import INFIX_TO_OPERATOR, Catalog, bind_call, default_catalog

__all__ = ["ExprMetrics", "compute_metrics", "data_fields", "int_literal", "is_signed_literal", "tree_depth"]


@dataclass(frozen=True)
class ExprMetrics:
    operator_count: int
    unique_operators: frozenset[str]
    fields: frozenset[str]
    group_fields: frozenset[str]
    non_group_fields: frozenset[str]
    node_count: int
    depth: int
    windows: tuple[int, ...]
    has_trade_when: bool


def is_signed_literal(node: Node) -> bool:
    """True for ``5``, ``-5`` or ``+5`` (a number, possibly with a sign)."""
    if isinstance(node, Unary) and node.op in "+-":
        return isinstance(node.operand, Num)
    return isinstance(node, Num)


def int_literal(node: Node, literals: dict[str, Node] | None = None) -> int | None:
    """Integer value of an integral Num (or a variable bound to one), else ``None``."""
    if isinstance(node, Ident) and literals and node.name in literals:
        node = literals[node.name]
    if isinstance(node, Num) and node.value.is_integer():
        return int(node.value)
    return None


def tree_depth(node: Node) -> int:
    """Nesting depth of ``node`` (a leaf has depth 1)."""
    children = node.children()
    return 1 + max((tree_depth(c) for c in children), default=0)


def _operator_name(node: Node) -> str | None:
    if isinstance(node, Call):
        return node.name
    if isinstance(node, BinOp):
        return INFIX_TO_OPERATOR[node.op]
    if isinstance(node, Unary):
        if node.op == "+" or is_signed_literal(node):
            return None
        return INFIX_TO_OPERATOR["!"] if node.op == "!" else INFIX_TO_OPERATOR["u-"]
    if isinstance(node, Ternary):
        return INFIX_TO_OPERATOR["?:"]
    return None


def _classify(name: str, catalog: Catalog, variables: set[str] | frozenset[str]) -> tuple[bool, bool]:
    """Return ``(is_data_field, is_group)`` for an identifier."""
    if name in variables:
        return False, False
    info = catalog.get_field(name)
    if info is not None:
        return True, info.type.upper() == "GROUP"
    if catalog.is_builtin_group(name):
        return False, True
    if catalog.strict_fields or catalog.is_enum_word(name):
        return False, False
    return True, False  # unknown identifier: presumably a field missing from the catalog


def data_fields(node: Node, catalog: Catalog | None = None, variables: frozenset[str] = frozenset()) -> set[str]:
    """Data fields (GROUP fields included, builtin groups excluded) referenced under ``node``."""
    catalog = catalog or default_catalog()
    found = set()
    for n in walk(node):
        if isinstance(n, Ident) and _classify(n.name, catalog, variables)[0]:
            found.add(n.name)
    return found


def compute_metrics(program: str | Node, catalog: Catalog | None = None) -> ExprMetrics:
    """Compute :class:`ExprMetrics` for a program (source text, Program or expression node)."""
    program = as_program(program)
    catalog = catalog or default_catalog()
    variables = {s.name for s in program.statements if isinstance(s, Assign)}
    literals = {k: v for k, v in program.assignments.items() if isinstance(v, Num)}

    operators: list[str] = []
    fields: set[str] = set()
    groups: set[str] = set()
    windows: list[int] = []
    in_group_slot: set[str] = set()
    node_count = 0

    for stmt in program.statements:
        root = stmt.value if isinstance(stmt, Assign) else stmt
        for node in walk(root):
            node_count += 1
            op = _operator_name(node)
            if op is not None:
                operators.append(op)
            if isinstance(node, Call):
                spec = catalog.get_operator(node.name)
                if spec is not None and not spec.permissive:
                    for bound in bind_call(spec, node.args, node.kwargs).bound:
                        if bound.spec.kind == "int":
                            value = int_literal(bound.node, literals)
                            if value is not None:
                                windows.append(value)
                        elif bound.spec.kind == "group" and isinstance(bound.node, Ident):
                            in_group_slot.add(bound.node.name)
            elif isinstance(node, Ident):
                is_field, is_group = _classify(node.name, catalog, variables)
                if is_field:
                    fields.add(node.name)
                if is_group:
                    groups.add(node.name)

    # Unknown identifiers passed as group arguments are presumably GROUP fields.
    groups |= {name for name in fields & in_group_slot if catalog.get_field(name) is None}

    final = inline(program)
    return ExprMetrics(
        operator_count=len(operators),
        unique_operators=frozenset(operators),
        fields=frozenset(fields),
        group_fields=frozenset(groups),
        non_group_fields=frozenset(fields - groups),
        node_count=node_count,
        depth=tree_depth(final),
        windows=tuple(windows),
        has_trade_when="trade_when" in operators,
    )
