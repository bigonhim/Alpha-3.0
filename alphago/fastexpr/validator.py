"""Static validation of FASTEXPR programs against a :class:`Catalog`.

:func:`validate` never raises for bad input: syntax errors, unknown operators, arity
and window problems, type misuse (VECTOR / GROUP values in the wrong place) and
statement-order problems are all reported as :class:`Issue` objects.

Inferred value types: MATRIX, VECTOR, GROUP, BOOLEAN, NUMBER, STRING and UNKNOWN
(UNKNOWN never triggers type errors, so unknown identifiers or permissive operators
cannot cascade into spurious errors).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from .ast import Assign, BinOp, Bool, Call, Ident, Node, Num, Program, Str, Ternary, Unary
from .canonical import inline
from .catalog import ArgSpec, Catalog, OperatorSpec, bind_call, default_catalog
from .metrics import ExprMetrics, compute_metrics, data_fields, is_signed_literal
from .parser import FastExprSyntaxError, parse

__all__ = ["Issue", "ValidationResult", "validate"]

MATRIX = "MATRIX"
VECTOR = "VECTOR"
GROUP = "GROUP"
BOOLEAN = "BOOLEAN"
NUMBER = "NUMBER"
STRING = "STRING"
UNKNOWN = "UNKNOWN"

_FIELD_TYPES = {"MATRIX": MATRIX, "VECTOR": VECTOR, "GROUP": GROUP, "UNIVERSE": MATRIX}
_RETURN_TYPES = {"matrix": MATRIX, "group": GROUP, "boolean": BOOLEAN}
_BOOLEAN_INFIX = frozenset({"<", "<=", ">", ">=", "==", "!=", "&&", "||"})

MAX_NODES = 60
MAX_DEPTH = 8


@dataclass
class Issue:
    code: str
    message: str
    snippet: str = ""


@dataclass
class ValidationResult:
    ok: bool
    errors: list[Issue]
    warnings: list[Issue]
    program: Program | None
    metrics: ExprMetrics | None

    @property
    def issues(self) -> list[Issue]:
        return self.errors + self.warnings

    def codes(self) -> set[str]:
        """Codes of all errors and warnings."""
        return {i.code for i in self.issues}


def _snippet(node: Node | None, limit: int = 120) -> str:
    if node is None:
        return ""
    text = node.to_source()
    return text if len(text) <= limit else text[: limit - 3] + "..."


class _Checker:
    def __init__(self, catalog: Catalog, max_window: int) -> None:
        self.catalog = catalog
        self.max_window = max_window
        self.errors: list[Issue] = []
        self.warnings: list[Issue] = []
        self._seen: set[tuple[str, str]] = set()
        self.var_types: dict[str, str] = {}
        self.literals: dict[str, Node] = {}
        self.pending: set[str] = set()
        self.used: set[str] = set()

    # ------------------------------------------------------------------ reporting

    def _add(self, bucket: list[Issue], code: str, message: str, node: Node | None) -> None:
        key = (code, message)
        if key not in self._seen:
            self._seen.add(key)
            bucket.append(Issue(code, message, _snippet(node)))

    def error(self, code: str, message: str, node: Node | None = None) -> None:
        self._add(self.errors, code, message, node)

    def warn(self, code: str, message: str, node: Node | None = None) -> None:
        self._add(self.warnings, code, message, node)

    # ------------------------------------------------------------------ program

    def run(self, program: Program) -> None:
        statements = program.statements
        if not statements:
            self.error("syntax", "empty program")
            return
        assigned_at = [s.name if isinstance(s, Assign) else None for s in statements]
        assigned: list[str] = []
        for i, stmt in enumerate(statements):
            last = i == len(statements) - 1
            self.pending = {n for n in assigned_at[i + 1 :] if n} - set(self.var_types)
            if isinstance(stmt, Assign):
                if last:
                    self.error("assignment_last", "the last statement must be an expression, not an assignment", stmt)
                self._check_assignment_name(stmt)
                value_type = self.infer(stmt.value)
                self.var_types[stmt.name] = value_type
                if is_signed_literal(stmt.value) or isinstance(stmt.value, (Str, Bool)):
                    self.literals[stmt.name] = stmt.value
                else:
                    self.literals.pop(stmt.name, None)
                assigned.append(stmt.name)
            else:
                if not last:
                    self.error(
                        "expression_not_last",
                        "only the last statement may be an expression; earlier statements must be assignments",
                        stmt,
                    )
                self.expect(stmt, self.infer(stmt), "expr", "the final expression" if last else "a statement")
        for name in dict.fromkeys(assigned):
            if name not in self.used:
                self.warn("unused_variable", f"variable '{name}' is assigned but never used")

    def _check_assignment_name(self, stmt: Assign) -> None:
        name = stmt.name
        if self.catalog.get_operator(name) is not None:
            self.warn("shadow_operator", f"variable '{name}' has the same name as an operator", stmt)
        elif self.catalog.get_field(name) is not None or self.catalog.is_builtin_group(name):
            self.warn("shadow_field", f"variable '{name}' shadows the data field / group '{name}'", stmt)

    # ------------------------------------------------------------------ inference

    def infer(self, node: Node) -> str:
        if isinstance(node, Num):
            return NUMBER
        if isinstance(node, Str):
            return STRING
        if isinstance(node, Bool):
            return BOOLEAN
        if isinstance(node, Ident):
            return self.resolve(node)
        if isinstance(node, Unary):
            t = self.infer(node.operand)
            self.expect(node.operand, t, "expr", f"the operand of unary '{node.op}'")
            if node.op == "!":
                return BOOLEAN
            return NUMBER if t == NUMBER else (UNKNOWN if t == UNKNOWN else MATRIX)
        if isinstance(node, BinOp):
            tl, tr = self.infer(node.left), self.infer(node.right)
            where = f"an operand of '{node.op}'"
            self.expect(node.left, tl, "expr", where)
            self.expect(node.right, tr, "expr", where)
            if node.op in _BOOLEAN_INFIX:
                return BOOLEAN
            return NUMBER if tl == tr == NUMBER else MATRIX
        if isinstance(node, Ternary):
            tc, ta, tb = self.infer(node.cond), self.infer(node.then), self.infer(node.other)
            self.expect(node.cond, tc, "expr", "the condition of '?:'")
            self.expect(node.then, ta, "expr", "a branch of '?:'")
            self.expect(node.other, tb, "expr", "a branch of '?:'")
            return NUMBER if ta == tb == NUMBER else MATRIX
        if isinstance(node, Call):
            return self.infer_call(node)
        if isinstance(node, (Assign, Program)):
            self.error("syntax", "assignments may only appear as top-level statements", node)
        return UNKNOWN

    def resolve(self, node: Ident) -> str:
        name = node.name
        if name in self.var_types:
            self.used.add(name)
            return self.var_types[name]
        if name in self.pending:
            self.error("use_before_assignment", f"variable '{name}' is used before it is assigned", node)
            self.used.add(name)
            return UNKNOWN
        info = self.catalog.get_field(name)
        if info is not None:
            return _FIELD_TYPES.get(info.type.upper(), UNKNOWN)
        if self.catalog.is_builtin_group(name):
            return GROUP
        if self.catalog.strict_fields:
            self.error("unknown_identifier", f"unknown identifier '{name}' (not a data field in the catalog)", node)
        else:
            self.warn("unknown_identifier", f"unknown identifier '{name}' (assumed to be a data field)", node)
        return UNKNOWN

    def infer_loose(self, node: Node) -> str:
        """Infer without expectations; bare enum words are accepted."""
        if isinstance(node, Ident) and node.name not in self.var_types and self.catalog.is_enum_word(node.name):
            return STRING
        return self.infer(node)

    def infer_call(self, node: Call) -> str:
        spec = self.catalog.get_operator(node.name)
        if spec is None:
            close = difflib.get_close_matches(node.name, list(self.catalog.operators), n=3, cutoff=0.75)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            self.error("unknown_operator", f"unknown operator '{node.name}'{hint}", node)
            for child in node.children():
                self.infer_loose(child)
            return UNKNOWN
        if not spec.available and self.catalog.strict_operators:
            self.error(
                "unavailable_operator",
                f"operator '{spec.name}' is not enabled for this account (missing from the BRAIN operator list)",
                node,
            )
        elif spec.tier == "C" and not self.catalog.strict_operators:
            self.warn(
                "consultant_operator",
                f"operator '{spec.name}' is usually consultant-only; load the account's operator list to confirm",
                node,
            )
        if spec.permissive:
            for child in node.children():
                self.infer_loose(child)
            return UNKNOWN

        binding = bind_call(spec, node.args, node.kwargs)
        for code, message in binding.issues:
            self.error(code, message, node)
        for bound in binding.bound:
            self.check_arg(spec, bound.spec, bound.node)
        for extra in binding.unbound:
            self.infer_loose(extra)
        return _RETURN_TYPES.get(spec.returns, MATRIX)

    # ------------------------------------------------------------------ expectations

    def expect(self, node: Node, t: str, kind: str, where: str) -> None:
        if t == UNKNOWN:
            return
        src = _snippet(node, 60)
        if kind == "expr":
            if t == VECTOR:
                self.error(
                    "vector_misuse",
                    f"VECTOR value '{src}' used as {where}, where a matrix is expected; "
                    f"wrap it in vec_avg()/vec_sum() first",
                    node,
                )
            elif t == GROUP:
                self.error(
                    "group_misuse",
                    f"GROUP value '{src}' used as {where}, where a matrix is expected; group fields may only "
                    f"be passed as group arguments (e.g. group_neutralize(x, {src}))",
                    node,
                )
            elif t == STRING:
                self.error("type_mismatch", f"string '{src}' used as {where}, where a matrix is expected", node)
        elif kind == "vector" and t != VECTOR:
            self.error("vector_required", f"{where} must be a VECTOR field, got {t.lower()} '{src}'", node)
        elif kind == "group" and t != GROUP:
            self.error(
                "group_required",
                f"{where} must be a group (sector, industry, subindustry, market, a GROUP field, "
                f"bucket(...), densify(...), ...), got {t.lower()} '{src}'",
                node,
            )

    def _literal(self, node: Node) -> Node:
        """Follow a variable bound to a literal (``lookback = 10``)."""
        if isinstance(node, Ident) and node.name in self.literals:
            self.used.add(node.name)
            return self.literals[node.name]
        return node

    def check_arg(self, op: OperatorSpec, arg: ArgSpec, node: Node) -> None:
        kind = arg.kind
        where = f"argument '{arg.name}' of {op.name}()"
        if kind in ("expr", "vector", "group"):
            self.expect(node, self.infer(node), kind, where)
            return
        if kind == "any":
            self.infer_loose(node)
            return
        value = self._literal(node)
        if isinstance(value, Ident) and value is node:
            # A non-literal variable or field where a constant is required: still resolve it
            # (use-before-assignment, unknown identifiers) unless it is an enum word.
            if kind != "enum":
                self.infer_loose(value)
        src = _snippet(node, 60)
        if kind == "int":
            if isinstance(value, Num) and value.value.is_integer() and value.value > 0:
                if value.value > self.max_window:
                    self.error(
                        "window", f"{where} is {int(value.value)}, above the maximum window of {self.max_window}", node
                    )
            else:
                self.error("window", f"{where} must be a positive integer literal, got '{src}'", node)
        elif kind == "number":
            if not is_signed_literal(value):
                self.error("literal_required", f"{where} must be a numeric constant, got '{src}'", node)
        elif kind == "string":
            if not isinstance(value, Str):
                self.error("literal_required", f'{where} must be a quoted string (e.g. "..."), got \'{src}\'', node)
        elif kind == "bool":
            if not (isinstance(value, Bool) or (isinstance(value, Num) and value.value in (0.0, 1.0))):
                self.error("literal_required", f"{where} must be true or false, got '{src}'", node)
        elif kind == "enum":
            word = value.name if isinstance(value, Ident) else value.value if isinstance(value, Str) else None
            if word is None:
                self.error("literal_required", f"{where} must be one of {', '.join(arg.choices)}, got '{src}'", node)
            elif arg.choices and word.lower() not in {c.lower() for c in arg.choices}:
                self.error(
                    "invalid_choice", f"{where} must be one of {', '.join(arg.choices)}, got '{word}'", node
                )


def validate(expr: str | Program, catalog: Catalog | None = None, *, max_window: int = 1000) -> ValidationResult:
    """Validate a FASTEXPR program; never raises for bad input."""
    catalog = catalog or default_catalog()
    if not isinstance(expr, (str, Node)):
        message = f"expected FASTEXPR source text or a Program, got {type(expr).__name__}"
        return ValidationResult(False, [Issue("syntax", message)], [], None, None)
    try:
        program = parse(expr) if isinstance(expr, str) else expr
        if not isinstance(program, Program):
            program = Program((program,))
    except FastExprSyntaxError as exc:
        return ValidationResult(False, [Issue("syntax", str(exc), exc.snippet)], [], None, None)
    except Exception as exc:  # noqa: BLE001 - validate() must never raise
        return ValidationResult(False, [Issue("syntax", f"could not parse expression: {exc}")], [], None, None)

    checker = _Checker(catalog, max_window)
    metrics: ExprMetrics | None = None
    try:
        checker.run(program)
        metrics = compute_metrics(program, catalog)
        if not data_fields(inline(program), catalog):
            checker.warn("constant_alpha", "the alpha does not reference any data field")
        if metrics.node_count > MAX_NODES or metrics.depth > MAX_DEPTH:
            checker.warn(
                "complexity",
                f"expression is complex ({metrics.node_count} nodes, depth {metrics.depth}); "
                f"consider simplifying (limits {MAX_NODES} nodes / depth {MAX_DEPTH})",
            )
    except Exception as exc:  # noqa: BLE001 - defensive: report instead of raising
        checker.error("internal", f"validator error: {type(exc).__name__}: {exc}")

    return ValidationResult(
        ok=not checker.errors,
        errors=checker.errors,
        warnings=checker.warnings,
        program=program,
        metrics=metrics,
    )
