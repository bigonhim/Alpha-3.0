"""Expression helpers the labs and tools share, with BRAIN's counting rules (from alpha-harness labs/fastexpr.py).

The differences from `alphago.fastexpr.compute_metrics` are deliberate and measured on BRAIN:

- Power Pool's "at most 8 operators" does not count `ts_backfill` or `group_backfill`;
- data fields exclude the grouping fields (market, sector, ... split, adjfactor), which never count as data;
- `negated` flips only the alpha (the last statement), never the steps before it.
"""

from __future__ import annotations

from ..brain.platform import GROUPING_FIELDS, POWER_POOL_MAX_FIELDS, POWER_POOL_MAX_OPERATORS, UNCOUNTED_OPERATORS
from ..fastexpr import Assign, Call, Ident, Node, Program, Unary, compute_metrics, parse, walk
from ..fastexpr.canonical import as_program


def program_of(expr: str | Node) -> Program:
    return parse(expr) if isinstance(expr, str) else as_program(expr)


def operators_in(expr: str | Node) -> list[str]:
    """Every operator as BRAIN names it (symbols too: + is add, unary - is reverse), repeats included."""
    from ..fastexpr.metrics import _operator_name
    out: list[str] = []
    for stmt in program_of(expr).statements:
        root = stmt.value if isinstance(stmt, Assign) else stmt
        for node in walk(root):
            name = _operator_name(node)
            if name is not None:
                out.append(name)
    return out


def operator_count(expr: str | Node) -> int:
    """Operators BRAIN counts against Power Pool's limit: every call and symbol except the backfills."""
    return sum(1 for name in operators_in(expr) if name not in UNCOUNTED_OPERATORS)


def data_fields(expr: str | Node, *, grouping: bool = False) -> set[str]:
    """Data fields the alpha reads. Grouping fields are left out unless `grouping` (they must still exist where
    the alpha runs, so placement checks ask for them)."""
    program = program_of(expr)
    metrics = compute_metrics(program)
    fields = set(metrics.fields)
    if grouping:
        assigned = {s.name for s in program.statements if isinstance(s, Assign)}
        fields |= {n.name for n in walk(program) if isinstance(n, Ident) and n.name in GROUPING_FIELDS
                   and n.name not in assigned}
    else:
        fields -= GROUPING_FIELDS
    return fields


def keyword_values(expr: str | Node) -> set[str]:
    """Identifiers written as a keyword value (`rettype = x`): guarded, never swapped for another field."""
    out: set[str] = set()
    for node in walk(program_of(expr)):
        if isinstance(node, Call):
            for _key, value in node.kwargs:
                if isinstance(value, Ident):
                    out.add(value.name)
    return out


def power_pool_problems(expr: str | Node) -> list[str]:
    """Why an expression cannot be a Power Pool alpha ([] when it can)."""
    out = []
    n_ops = operator_count(expr)
    n_fields = len(data_fields(expr))
    if n_ops > POWER_POOL_MAX_OPERATORS:
        out.append(f"{n_ops} operators (limit {POWER_POOL_MAX_OPERATORS})")
    if n_fields > POWER_POOL_MAX_FIELDS:
        out.append(f"{n_fields} data fields (limit {POWER_POOL_MAX_FIELDS})")
    return out


def render(expr: str | Node) -> str:
    """One line of BRAIN source: statements joined with '; '."""
    return "; ".join(s.to_source() for s in program_of(expr).statements)


def negated(expr: str | Node) -> str:
    """The same alpha with its sign flipped: only the last statement changes; an existing minus is removed."""
    program = program_of(expr)
    *steps, last = program.statements
    if isinstance(last, Assign):
        value = last.value
        flipped = value.operand if isinstance(value, Unary) and value.op == "-" else Unary("-", value)
        last = Assign(last.name, flipped)
    else:
        last = last.operand if isinstance(last, Unary) and last.op == "-" else Unary("-", last)
    return render(Program((*steps, last)))


def swap_field(expr: str | Node, field: str, replacement: str) -> str:
    """Every read of `field` replaced by `replacement` (an id, or an expression such as a Fast D1 delta).
    A field written as a keyword value (`rettype = x`) is guarded and stays as written."""
    new = parse(replacement)
    swap: Node = new.statements[-1] if len(new.statements) == 1 else new

    def go(n: Node) -> Node:
        if isinstance(n, Ident):
            return swap if n.name == field else n
        if isinstance(n, Call):
            return Call(n.name, tuple(go(a) for a in n.args), n.kwargs)    # keyword values are guarded
        children = n.children()
        return n.with_children(tuple(go(c) for c in children)) if children else n

    program = program_of(expr)
    return render(Program(tuple(go(st) for st in program.statements)))
