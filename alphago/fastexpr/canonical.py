"""Canonical forms, hashes and subtree comparison for FASTEXPR expressions.

The canonical form inlines variables, normalises numbers and booleans, orders keyword
arguments by name and sorts the operands of commutative operators, so that trivially
different spellings of the same alpha hash identically.  The *skeleton* additionally
replaces every numeric literal with ``#`` (same structure, different parameters).
"""

from __future__ import annotations

import hashlib

from .ast import Assign, BinOp, Call, Ident, Node, Num, Program, format_number, transform
from .catalog import bind_call, default_catalog
from .parser import parse

__all__ = [
    "as_program",
    "canonical_hash",
    "canonical_node",
    "canonical_source",
    "inline",
    "largest_common_subtree",
    "skeleton",
    "skeleton_hash",
    "subtree_sizes",
]

#: Associative + commutative infix operators: chains are flattened and sorted.
_ASSOCIATIVE_INFIX = frozenset({"+", "*", "&&", "||"})
#: Commutative but not associative infix operators: the two operands are sorted.
_COMMUTATIVE_INFIX = frozenset({"==", "!="})
#: Function-form operators whose positional arguments may be reordered.
_COMMUTATIVE_CALLS = frozenset({"add", "multiply", "max", "min", "and", "or", "equal", "not_equal"})

_PLACEHOLDER = Ident("#")


def as_program(expr: str | Node) -> Program:
    """Coerce source text, a Program or a bare expression node into a Program."""
    if isinstance(expr, str):
        return parse(expr)
    if isinstance(expr, Program):
        return expr
    if isinstance(expr, Node):
        return Program((expr,))
    raise TypeError(f"expected FASTEXPR source or AST node, got {type(expr).__name__}")


def inline(program: str | Node) -> Node:
    """Substitute every assignment into the final expression (handles chains and re-assignment)."""
    program = as_program(program)
    env: dict[str, Node] = {}

    def substitute(node: Node) -> Node:
        if not env:
            return node
        return transform(node, lambda n: env.get(n.name, n) if isinstance(n, Ident) else n)

    for stmt in program.statements[:-1]:
        if isinstance(stmt, Assign):
            env[stmt.name] = substitute(stmt.value)
    last = program.statements[-1]
    if isinstance(last, Assign):  # only possible for programmatically built Programs
        last = last.value
    return substitute(last)


def _flatten(node: Node, op: str) -> list[Node]:
    if isinstance(node, BinOp) and node.op == op:
        return _flatten(node.left, op) + _flatten(node.right, op)
    return [node]


def _canon_call(node: Call) -> Call:
    """Normalise argument placement using the curated signature, then sort kwargs."""
    args = list(node.args)
    kwargs = list(node.kwargs)
    spec = default_catalog().get_operator(node.name)
    if spec is not None and not spec.permissive and not spec.variadic:
        binding = bind_call(spec, node.args, node.kwargs)
        if not binding.issues:
            by_name = {b.spec.name: b.node for b in binding.bound}
            args = []
            for a in spec.positional:
                if a.name not in by_name:
                    break
                args.append(by_name.pop(a.name))
            kwargs = list(by_name.items())
    if node.name in _COMMUTATIVE_CALLS:
        args.sort(key=lambda n: n.to_source())
    kwargs.sort(key=lambda kv: kv[0])
    return Call(node.name, tuple(args), tuple(kwargs))


def _canon(node: Node, skeleton: bool) -> Node:
    def rewrite(n: Node) -> Node:
        if isinstance(n, Num):
            return _PLACEHOLDER if skeleton else Num(n.value, format_number(n.value))
        if isinstance(n, Call):
            return _canon_call(n)
        if isinstance(n, BinOp):
            if n.op in _ASSOCIATIVE_INFIX:
                operands = sorted(_flatten(n, n.op), key=lambda x: x.to_source())
                result = operands[0]
                for operand in operands[1:]:
                    result = BinOp(n.op, result, operand)
                return result
            if n.op in _COMMUTATIVE_INFIX and n.right.to_source() < n.left.to_source():
                return BinOp(n.op, n.right, n.left)
        return n

    return transform(node, rewrite)


def canonical_node(expr: str | Node, *, skeleton: bool = False) -> Node:
    """Canonical expression tree (variables inlined) for ``expr``."""
    return _canon(inline(expr), skeleton)


def canonical_source(expr: str | Node) -> str:
    """Canonical source text of ``expr`` (see module docstring)."""
    return canonical_node(expr).to_source()


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def canonical_hash(expr: str | Node) -> str:
    """First 16 hex chars of the SHA-1 of :func:`canonical_source`."""
    return _hash(canonical_source(expr))


def skeleton(expr: str | Node) -> str:
    """Canonical source with every numeric literal replaced by ``#``."""
    return canonical_node(expr, skeleton=True).to_source()


def skeleton_hash(expr: str | Node) -> str:
    """First 16 hex chars of the SHA-1 of :func:`skeleton`."""
    return _hash(skeleton(expr))


def subtree_sizes(node: str | Node) -> dict[str, int]:
    """Map the canonical source of every subtree of ``node`` to its size in nodes."""
    root = canonical_node(node)
    sizes: dict[str, int] = {}

    def visit(n: Node) -> int:
        size = 1 + sum(visit(c) for c in n.children())
        sizes[n.to_source()] = size
        return size

    visit(root)
    return sizes


def largest_common_subtree(a: str | Node, b: str | Node) -> int:
    """Size (in nodes) of the largest identical canonical subtree shared by ``a`` and ``b``."""
    sa, sb = subtree_sizes(a), subtree_sizes(b)
    if len(sb) < len(sa):
        sa, sb = sb, sa
    return max((size for src, size in sa.items() if src in sb), default=0)
