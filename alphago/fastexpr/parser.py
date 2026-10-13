"""FASTEXPR parser: text -> :class:`~alphago.fastexpr.ast.Program`.

Comments (``/* */``, ``//`` and ``#``) and trailing semicolons are stripped before
parsing.  The grammar lives in ``grammar.lark`` and is compiled once (LALR) and cached;
parse results are cached as well since AST nodes are immutable.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import NamedTuple

from lark import Lark, Token, Transformer, Tree, v_args
from lark.exceptions import UnexpectedCharacters, UnexpectedInput, UnexpectedToken, VisitError

from .ast import Assign, BinOp, Bool, Call, Ident, Node, Num, Program, Str, Ternary, Unary

__all__ = ["FastExprSyntaxError", "parse", "strip_comments"]


class FastExprSyntaxError(ValueError):
    """Raised for text that is not a valid FASTEXPR program."""

    def __init__(self, message: str, line: int = 1, column: int = 1, source: str = "") -> None:
        self.message = message
        self.line = line
        self.column = column
        self.source = source
        super().__init__(f"{message} (line {line}, column {column})")

    @property
    def snippet(self) -> str:
        """The offending source line with a caret under the error column."""
        lines = self.source.splitlines()
        if not 1 <= self.line <= len(lines):
            return ""
        text = lines[self.line - 1]
        return f"{text}\n{' ' * max(self.column - 1, 0)}^"


# --------------------------------------------------------------------------- comments


def _blank(text: str) -> str:
    """Replace every non-newline character with a space (keeps line/column positions)."""
    return re.sub(r"[^\n]", " ", text)


def strip_comments(text: str, *, keep_positions: bool = False) -> str:
    """Remove ``/* ... */``, ``// ...`` and ``# ...`` comments outside string literals.

    With ``keep_positions=True`` comments are replaced by spaces so that line and column
    numbers of the remaining text are unchanged (used for error reporting).  Otherwise
    block comments become a single space and the result is stripped.  An unterminated
    block comment runs to the end of the text (:func:`parse` reports it as an error).
    """
    return _strip(text, keep_positions)[0]


def _strip(text: str, keep_positions: bool) -> tuple[str, int | None]:
    """Implementation of :func:`strip_comments`; also returns the offset of an unterminated ``/*``."""
    unterminated: int | None = None
    out: list[str] = []
    i, n = 0, len(text)
    quote: str | None = None
    while i < n:
        c = text[i]
        if quote:
            out.append(c)
            if c == quote or c == "\n":  # strings cannot span lines
                quote = None
            i += 1
        elif c in "\"'":
            quote = c
            out.append(c)
            i += 1
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            end = n if j < 0 else j + 2
            if j < 0:
                unterminated = i
            out.append(_blank(text[i:end]) if keep_positions else " ")
            i = end
        elif text.startswith("//", i) or c == "#":
            j = text.find("\n", i)
            end = n if j < 0 else j
            if keep_positions:
                out.append(" " * (end - i))
            i = end
        else:
            out.append(c)
            i += 1
    result = "".join(out)
    if not keep_positions:
        result = "\n".join(line.rstrip() for line in result.splitlines()).strip()
    return result, unterminated


def _outside_strings(text: str):
    """Yield ``(index, char)`` for characters that are not inside string literals."""
    quote: str | None = None
    for i, c in enumerate(text):
        if quote:
            if c == quote or c == "\n":
                quote = None
        elif c in "\"'":
            quote = c
        else:
            yield i, c


def _line_col(text: str, pos: int) -> tuple[int, int]:
    line = text.count("\n", 0, pos) + 1
    col = pos - (text.rfind("\n", 0, pos) + 1) + 1
    return line, col


# --------------------------------------------------------------------------- lark


def _grammar_text() -> str:
    try:
        return resources.files(__package__).joinpath("grammar.lark").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, TypeError):  # pragma: no cover - frozen apps
        return (Path(__file__).parent / "grammar.lark").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _lark() -> Lark:
    return Lark(
        _grammar_text(),
        start="program",
        parser="lalr",
        lexer="contextual",
        propagate_positions=True,
        maybe_placeholders=True,
    )


class _Kwarg(NamedTuple):
    name: str
    value: Node


class _ToAst(Transformer):
    """Turns the lark parse tree into AST nodes."""

    def program(self, items: list[Node]) -> Program:
        return Program(tuple(items))

    def assign(self, items: list) -> Assign:
        name, value = items
        return Assign(str(name), value)

    def ternary(self, items: list[Node]) -> Ternary:
        cond, then, other = items
        return Ternary(cond, then, other)

    def binop(self, items: list) -> BinOp:
        left, op, right = items
        return BinOp(str(op), left, right)

    def unary(self, items: list) -> Unary:
        op, operand = items
        return Unary(str(op), operand)

    def number(self, items: list[Token]) -> Num:
        text = str(items[0])
        return Num(float(text), text)

    def string(self, items: list[Token]) -> Str:
        return Str(str(items[0])[1:-1])

    def var(self, items: list[Token]) -> Node:
        name = str(items[0])
        lowered = name.lower()
        if lowered in ("true", "false"):
            return Bool(lowered == "true")
        return Ident(name)

    def args(self, items: list) -> list:
        return list(items)

    def kwarg(self, items: list) -> _Kwarg:
        name, value = items
        return _Kwarg(str(name), value)

    @v_args(meta=True)
    def call(self, meta, items: list) -> Call:
        name, arglist = items
        positional: list[Node] = []
        keywords: list[tuple[str, Node]] = []
        for item in arglist or ():
            if isinstance(item, _Kwarg):
                keywords.append((item.name, item.value))
            elif keywords:
                raise FastExprSyntaxError(
                    f"positional argument follows keyword argument in call to {name}()",
                    getattr(meta, "line", 1),
                    getattr(meta, "column", 1),
                )
            else:
                positional.append(item)
        return Call(str(name), tuple(positional), tuple(keywords))


_TERMINAL_NAMES = {
    "RPAR": "')'",
    "LPAR": "'('",
    "COMMA": "','",
    "NAME": "an identifier",
    "NUMBER": "a number",
    "STRING": "a string",
    "_SEMI": "';'",
    "SEMICOLON": "';'",
    "EQUAL": "'='",
    "COLON": "':'",
    "QMARK": "'?'",
    "OR": "'||'",
    "AND": "'&&'",
    "EQ": "'=='",
    "NE": "'!='",
    "LE": "'<='",
    "GE": "'>='",
    "LT": "'<'",
    "GT": "'>'",
    "PLUS": "'+'",
    "MINUS": "'-'",
    "STAR": "'*'",
    "SLASH": "'/'",
    "BANG": "'!'",
    "$END": "end of input",
}


def _describe_expected(expected: set[str] | frozenset[str]) -> str:
    names = sorted({_TERMINAL_NAMES.get(t, t) for t in expected})
    if not names:
        return ""
    if len(names) > 8:
        names = names[:8] + ["..."]
    return "; expected " + ", ".join(names)


def _paren_balance(text: str) -> int:
    depth = 0
    for _, c in _outside_strings(text):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
    return depth


def _convert_lark_error(err: UnexpectedInput, text: str, source: str) -> FastExprSyntaxError:
    """Translate a lark error on ``text`` (comment-blanked ``source``) into a friendly message."""
    if isinstance(err, UnexpectedToken) and err.token.type == "$END":
        line, col = _line_col(text, len(text.rstrip()))
        message = "unexpected end of expression"
        balance = _paren_balance(text)
        if balance > 0:
            message += f" (missing {balance} closing ')')"
        else:
            message += _describe_expected(err.expected)
        return FastExprSyntaxError(message, line, col, source)
    if isinstance(err, UnexpectedToken):
        tok = err.token
        message = f"unexpected {str(tok)!r}" + _describe_expected(err.expected)
        if tok.type == "RPAR" and _paren_balance(text) < 0:
            message = "unbalanced ')'"
        return FastExprSyntaxError(message, err.line or 1, err.column or 1, source)
    if isinstance(err, UnexpectedCharacters):
        ch = text[err.pos_in_stream] if 0 <= err.pos_in_stream < len(text) else "?"
        message = f"unexpected character {ch!r}"
        if ch in "\"'":
            message = "unterminated string literal"
        return FastExprSyntaxError(message, err.line or 1, err.column or 1, source)
    return FastExprSyntaxError(str(err), getattr(err, "line", 1) or 1, getattr(err, "column", 1) or 1, source)


def _check_statements(tree: Tree, text: str) -> None:
    statements = tree.children
    for i, stmt in enumerate(statements):
        is_assign = isinstance(stmt, Tree) and stmt.data == "assign"
        meta = stmt.meta if isinstance(stmt, Tree) else None
        line = getattr(meta, "line", 1) if meta is not None and not meta.empty else 1
        col = getattr(meta, "column", 1) if meta is not None and not meta.empty else 1
        last = i == len(statements) - 1
        if last and is_assign:
            name = str(stmt.children[0])
            raise FastExprSyntaxError(
                f"the last statement must be an expression (the alpha), not an assignment to '{name}'",
                line,
                col,
                text,
            )
        if not last and not is_assign:
            raise FastExprSyntaxError(
                "only the last statement may be an expression; earlier statements must be "
                "assignments of the form 'name = expression'",
                line,
                col,
                text,
            )


# --------------------------------------------------------------------------- public API


def parse(text: str) -> Program:
    """Parse FASTEXPR source into a :class:`Program`.

    Raises :class:`FastExprSyntaxError` for invalid input.
    """
    if not isinstance(text, str):
        raise TypeError(f"parse() expects a string, got {type(text).__name__}")
    return _parse_cached(text)


@lru_cache(maxsize=4096)
def _parse_cached(text: str) -> Program:
    cleaned, unterminated = _strip(text, keep_positions=True)
    if unterminated is not None:
        line, col = _line_col(text, unterminated)
        raise FastExprSyntaxError("unterminated /* comment", line, col, text)

    for i, c in _outside_strings(cleaned):
        if c == "^" or (c == "*" and cleaned.startswith("**", i)):
            line, col = _line_col(cleaned, i)
            op = "^" if c == "^" else "**"
            raise FastExprSyntaxError(
                f"'{op}' is not a FASTEXPR operator; use power(x, y) or signed_power(x, y) instead",
                line,
                col,
                text,
            )

    body = cleaned.rstrip()
    while body.endswith(";"):
        body = body[:-1].rstrip()
    if not body.strip():
        raise FastExprSyntaxError("empty expression", 1, 1, text)

    try:
        tree = _lark().parse(body)
        _check_statements(tree, text)
        return _ToAst().transform(tree)
    except UnexpectedInput as err:
        raise _convert_lark_error(err, body, text) from None
    except VisitError as err:
        if isinstance(err.orig_exc, FastExprSyntaxError):
            exc = err.orig_exc
            raise FastExprSyntaxError(exc.message, exc.line, exc.column, text) from None
        if isinstance(err.orig_exc, RecursionError):
            raise FastExprSyntaxError("expression is nested too deeply", 1, 1, text) from None
        raise
    except RecursionError:
        raise FastExprSyntaxError("expression is nested too deeply", 1, 1, text) from None
