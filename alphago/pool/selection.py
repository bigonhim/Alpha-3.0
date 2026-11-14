"""Selection Expression: Fast Expression over an alpha's metrics rather than its data (alpha-harness selection.ts).

Statements end at `;` or a new line. `name = …` names a value for the statements after it; every other statement is
a condition, and an alpha is selected when it meets them all. As in Fast Expression a comparison is 1 or 0 and NaN is
neither, and a metric an alpha lacks is NaN, so any condition on it leaves the alpha out. `&&`, `||` and `!` are
`and`, `or` and `not`; `25%` is 0.25 and `5bps` is 0.0005. `#` and `//` start a comment. A ratio over nought is NaN
(Infinity would pass). `^` is right-associative and binds below a sign: `-2 ^ 2` is 4.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

NAN = math.nan
Metrics = dict[str, float | None]


class SelectionError(ValueError):
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"Line {line}: {message}")
        self.line = line


def _truth(x: float) -> bool:
    return not math.isnan(x) and x != 0


def _bool(x: bool) -> float:
    return 1.0 if x else 0.0


def _compare(test: Callable[[float, float], bool]) -> Callable[[float, float], float]:
    return lambda a, b: NAN if math.isnan(a) or math.isnan(b) else _bool(test(a, b))


def _and(a: float, b: float) -> float:
    # three-valued: a false side settles it, otherwise a missing side stays missing
    return 0.0 if a == 0 or b == 0 else NAN if math.isnan(a) or math.isnan(b) else 1.0


def _or(a: float, b: float) -> float:
    return 1.0 if _truth(a) or _truth(b) else NAN if math.isnan(a) or math.isnan(b) else 0.0


def _not(a: float) -> float:
    return NAN if math.isnan(a) else _bool(a == 0)


def _if_else(c: float, a: float, b: float) -> float:
    return NAN if math.isnan(c) else a if c != 0 else b


def _safe(fn: Callable[..., float]) -> Callable[..., float]:
    """Math domain errors are NaN, as in JavaScript (sqrt(-1), log(0))."""
    def run(*args: float) -> float:
        try:
            value = fn(*args)
        except (ValueError, OverflowError, ZeroDivisionError):
            return NAN
        if isinstance(value, complex):
            return NAN
        return float(value)
    return run


def _sign(a: float) -> float:
    return NAN if math.isnan(a) else float((a > 0) - (a < 0))


def _log(a: float) -> float:
    if math.isnan(a) or a < 0:
        return NAN
    return -math.inf if a == 0 else math.log(a)


def _max(*a: float) -> float:
    return NAN if any(math.isnan(x) for x in a) else max(a)


def _min(*a: float) -> float:
    return NAN if any(math.isnan(x) for x in a) else min(a)


INF = 10**9
#: Fast Expression's own names: [least arguments, most arguments, function].
FUNCTIONS: dict[str, tuple[int, int, Callable[..., float]]] = {
    "abs": (1, 1, _safe(abs)), "sign": (1, 1, _sign), "sqrt": (1, 1, _safe(math.sqrt)), "log": (1, 1, _log),
    "power": (2, 2, _safe(lambda a, b: a ** b)), "max": (2, INF, _max), "min": (2, INF, _min),
    "and": (2, 2, _and), "or": (2, 2, _or), "not": (1, 1, _not), "if_else": (3, 3, _if_else),
    "is_nan": (1, 1, lambda a: _bool(math.isnan(a))),
}

BINARY: dict[str, Callable[[float, float], float]] = {
    ">=": _compare(lambda a, b: a >= b), "<=": _compare(lambda a, b: a <= b), ">": _compare(lambda a, b: a > b),
    "<": _compare(lambda a, b: a < b), "==": _compare(lambda a, b: a == b), "!=": _compare(lambda a, b: a != b),
    "&&": _and, "||": _or, "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
    "/": lambda a, b: NAN if b == 0 else a / b, "^": _safe(lambda a, b: a ** b),
}
UNARY: dict[str, Callable[[float], float]] = {"-": lambda a: -a, "+": lambda a: a, "!": _not}
COMPARISONS = (">=", "<=", ">", "<", "==", "!=")
LEVELS: tuple[tuple[str, ...], ...] = (("||",), ("&&",), COMPARISONS, ("+", "-"), ("*", "/"))
MAX_DEPTH = 64

_TOKEN = re.compile(r"""[ \t\r]*(?:
    (?P<num>(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?(?:%|bps)?)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>&&|\|\||==|!=|<=|>=|[-+*/^<>!?:(),=;\n])
)""", re.VERBOSE)
_COMMENT = re.compile(r"/\*.*?\*/|(?://|#)[^\n]*", re.DOTALL)


@dataclass
class Node:
    kind: str                     # num | name | call | unary | binary | ternary | assign
    value: str = ""
    args: list[Node] = field(default_factory=list)
    at: int = 0


def _number(text: str) -> float:
    if text.endswith("%"):
        return float(text[:-1]) / 100
    if text.endswith("bps"):
        return float(text[:-3]) / 10_000
    return float(text)


def tokenize(text: str) -> list[tuple[str, str, int]]:
    blanked = _COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    tokens: list[tuple[str, str, int]] = []
    pos = depth = 0
    while pos < len(blanked):
        if blanked[pos:].strip(" \t\r") == "":
            break
        m = _TOKEN.match(blanked, pos)
        if m is None or m.lastgroup is None:
            start = pos + len(blanked[pos:]) - len(blanked[pos:].lstrip(" \t\r"))
            raise _ParseError(f"cannot read {blanked[start:start + 12]!r}", start)
        kind = m.lastgroup
        value = m.group(kind)
        pos = m.end()
        if value == "(":
            depth += 1
        elif value == ")":
            depth = max(0, depth - 1)
        elif value == "\n" and depth:
            continue                      # a line break inside brackets only wraps the statement
        tokens.append(("sep" if value in (";", "\n") else kind, value, m.start(kind)))
    return tokens


class _ParseError(Exception):
    def __init__(self, message: str, at: int) -> None:
        super().__init__(message)
        self.message, self.at = message, at


class _Parser:
    def __init__(self, tokens: list[tuple[str, str, int]], end: int) -> None:
        self.tokens, self.i, self.depth, self.end = tokens, 0, 0, end

    def peek(self, k: int = 0) -> tuple[str, str, int] | None:
        return self.tokens[self.i + k] if self.i + k < len(self.tokens) else None

    def at_op(self, symbol: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t is not None and t[0] == "op" and t[1] == symbol

    def take(self) -> tuple[str, str, int]:
        t = self.peek()
        if t is None or t[0] == "sep":
            raise _ParseError("the statement ends too early", t[2] if t else self.end)
        self.i += 1
        return t

    def expect(self, symbol: str) -> None:
        t = self.take()
        if t[0] != "op" or t[1] != symbol:
            raise _ParseError(f"expected {symbol!r}, found {t[1]!r}", t[2])

    def deeper(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise _ParseError("the expression nests too deeply", self.peek()[2] if self.peek() else self.end)

    def program(self) -> list[Node]:
        statements: list[Node] = []
        while self.peek() is not None:
            if self.peek()[0] == "sep":
                self.i += 1
                continue
            statements.append(self.statement())
            t = self.peek()
            if t is not None and t[0] != "sep":
                raise _ParseError(f"unexpected {t[1]!r}; end the statement with ';' or a new line", t[2])
        return statements

    def statement(self) -> Node:
        t = self.peek()
        if t is not None and t[0] == "name" and self.at_op("=", 1):
            self.i += 2
            return Node("assign", t[1], [self.expr()], t[2])
        return self.expr()

    def expr(self) -> Node:
        self.deeper()
        node = self.binary(0)
        if self.at_op("?"):
            at = self.take()[2]
            yes = self.expr()
            self.expect(":")
            node = Node("ternary", "?", [node, yes, self.expr()], at)
        self.depth -= 1
        return node

    def binary(self, level: int) -> Node:
        if level == len(LEVELS):
            return self.power()
        left = self.binary(level + 1)
        while (t := self.peek()) is not None and t[0] == "op" and t[1] in LEVELS[level]:
            self.i += 1
            left = Node("binary", t[1], [left, self.binary(level + 1)], t[2])
        return left

    def power(self) -> Node:
        base = self.unary()
        if not self.at_op("^"):
            return base
        at = self.take()[2]
        self.deeper()
        node = Node("binary", "^", [base, self.power()], at)
        self.depth -= 1
        return node

    def unary(self) -> Node:
        t = self.peek()
        if t is not None and t[0] == "op" and t[1] in UNARY:
            self.i += 1
            self.deeper()
            node = Node("unary", t[1], [self.unary()], t[2])
            self.depth -= 1
            return node
        return self.primary()

    def primary(self) -> Node:
        kind, text, at = self.take()
        if kind == "num":
            return Node("num", text, at=at)
        if kind == "name":
            if self.at_op("("):
                return self.call(text, at)
            return Node("name", text, at=at)
        if text == "(":
            inner = self.expr()
            self.expect(")")
            return inner
        raise _ParseError(f"expected a number, metric or '(', found {text!r}", at)

    def call(self, name: str, at: int) -> Node:
        self.expect("(")
        args: list[Node] = []
        if self.at_op(")"):
            self.i += 1
            return Node("call", name, args, at)
        while True:
            t = self.peek()
            if t is not None and t[0] == "name" and self.at_op("=", 1):
                raise _ParseError(f"{name} takes no named inputs", t[2])
            args.append(self.expr())
            if self.at_op(","):
                self.i += 1
                continue
            self.expect(")")
            return Node("call", name, args, at)


def _meant(name: str, names: Iterable[str]) -> str:
    """The name meant when `name` differs from one only in case or underscores."""
    def squash(n: str) -> str:
        return n.lower().replace("_", "")
    for n in names:
        if squash(n) == squash(name):
            return f'; did you mean "{n}"?'
    return ""


Value = Callable[[dict[str, float]], float]


@dataclass
class Selection:
    conditions: int
    _named: list[tuple[str, Value]]
    _tests: list[Value]

    def test(self, metrics: Metrics) -> bool:
        scope = {k: (NAN if v is None else float(v)) for k, v in metrics.items()}
        for name, value in self._named:
            scope[name] = value(scope)
        return all(_truth(c(scope)) for c in self._tests)


def compile_selection(text: str, metrics: Iterable[str]) -> Selection:
    """Compile a Selection Expression; raises SelectionError naming the line at fault. `metrics` are the names
    allowed. An empty expression (only comments and blank lines) selects everything."""
    known = set(metrics)

    def line(at: int) -> int:
        return text[:at].count("\n") + 1

    try:
        statements = _Parser(tokenize(text), len(text)).program()
    except _ParseError as exc:
        raise SelectionError(exc.message, line(exc.at)) from None

    def fail(message: str, node: Node) -> None:
        raise SelectionError(message, line(node.at))

    def value(node: Node) -> Value:
        if node.kind == "num":
            constant = _number(node.value)
            return lambda s: constant
        if node.kind == "name":
            name = node.value
            if name not in known:
                fail(f'no metric named "{name}"{_meant(name, known)}', node)
            return lambda s: s.get(name, NAN)
        if node.kind == "call":
            if node.value not in FUNCTIONS:
                fail(f'no function named "{node.value}"{_meant(node.value, FUNCTIONS)}', node)
            least, most, fn = FUNCTIONS[node.value]
            args = [value(a) for a in node.args]
            if not least <= len(args) <= most:
                count = least if least == most else f"at least {least}"
                fail(f"{node.value} takes {count} argument{'' if least == 1 else 's'}", node)
            return lambda s: fn(*(a(s) for a in args))
        if node.kind == "unary":
            op, inner = UNARY[node.value], value(node.args[0])
            return lambda s: op(inner(s))
        if node.kind == "binary":
            op = BINARY[node.value]
            left, right = value(node.args[0]), value(node.args[1])
            return lambda s: op(left(s), right(s))
        if node.kind == "ternary":
            test, yes, no = (value(a) for a in node.args)
            return lambda s: _if_else(test(s), yes(s), no(s))
        fail(f"cannot read {node.value!r}", node)
        raise AssertionError  # unreachable

    named: list[tuple[str, Value]] = []
    tests: list[Value] = []
    for st in statements:
        if st.kind == "assign":
            if st.value in FUNCTIONS:
                fail(f'"{st.value}" is a function', st)
            named.append((st.value, value(st.args[0])))
            known.add(st.value)
        else:
            tests.append(value(st))
    return Selection(len(tests), named, tests)
