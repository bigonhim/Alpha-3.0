"""FASTEXPR: parser, validator and AST toolkit for WorldQuant BRAIN alpha expressions.

Typical use::

    from alphago.fastexpr import parse, validate, canonical_hash

    result = validate("rank(-ts_delta(close, 5))")
    if result.ok:
        key = canonical_hash(result.program)
"""

from .ast import (
    Assign,
    BinOp,
    Bool,
    Call,
    Ident,
    Node,
    Num,
    Program,
    Str,
    Ternary,
    Unary,
    format_number,
    num,
    transform,
    walk,
)
from .canonical import (
    as_program,
    canonical_hash,
    canonical_node,
    canonical_source,
    inline,
    largest_common_subtree,
    skeleton,
    skeleton_hash,
    subtree_sizes,
)
from .catalog import (
    BUILTIN_GROUPS,
    BUILTIN_MATRIX_FIELDS,
    INFIX_TO_OPERATOR,
    ArgSpec,
    Binding,
    BoundArg,
    Catalog,
    FieldInfo,
    OperatorSpec,
    bind_call,
    default_catalog,
)
from .metrics import ExprMetrics, compute_metrics
from .parser import FastExprSyntaxError, parse, strip_comments
from .transforms import (
    add_backfill,
    negate,
    program_from_expr,
    replace_identifier,
    scale_windows,
    set_kwarg,
    set_windows,
    to_node,
    trade_when_wrap,
    wrap,
)
from .validator import Issue, ValidationResult, validate

__all__ = [
    "BUILTIN_GROUPS",
    "BUILTIN_MATRIX_FIELDS",
    "INFIX_TO_OPERATOR",
    "ArgSpec",
    "Assign",
    "BinOp",
    "Binding",
    "Bool",
    "BoundArg",
    "Call",
    "Catalog",
    "ExprMetrics",
    "FastExprSyntaxError",
    "FieldInfo",
    "Ident",
    "Issue",
    "Node",
    "Num",
    "OperatorSpec",
    "Program",
    "Str",
    "Ternary",
    "Unary",
    "ValidationResult",
    "add_backfill",
    "as_program",
    "bind_call",
    "canonical_hash",
    "canonical_node",
    "canonical_source",
    "compute_metrics",
    "default_catalog",
    "format_number",
    "inline",
    "largest_common_subtree",
    "negate",
    "num",
    "parse",
    "program_from_expr",
    "replace_identifier",
    "scale_windows",
    "set_kwarg",
    "set_windows",
    "skeleton",
    "skeleton_hash",
    "strip_comments",
    "subtree_sizes",
    "to_node",
    "trade_when_wrap",
    "transform",
    "validate",
    "walk",
    "wrap",
]
