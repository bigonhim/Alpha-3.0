"""Fast Expression as the labs read and write it (alpha-harness labs/fastexpr.py, ported).

AlphaGo's own `alphago.fastexpr` parser is the validator of record for simulations. The Template and Evolution labs
need a little more of the language than it reads, and a tree they can edit by path:

- Template Lab writes two names BRAIN never sees: a ``$variable`` and ``...`` where a template's signal goes;
- ``^`` is read (right-associative, a sign binding tighter than it), so a template can fold ``$d ^ 2``;
- every node is ``Node(kind, value, args, kwargs)``, so a mutation or a graft is a relabel or a replace at a path.

``parse(render(tree)) == tree`` for every tree :func:`parse` returns, which is what makes changing a tree and writing
it back safe. Operator arity comes from each operator's published definition (:func:`operator_table`).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

from ..brain.platform import GROUPING_FIELDS, UNCOUNTED_OPERATORS

COMPARISONS = ("==", "!=", "<", "<=", ">", ">=")


class ParseError(ValueError):
    """Not an expression this grammar can read."""


@dataclass(frozen=True, slots=True)
class Node:
    kind: str  # num | str | name | call | unary | binary | ternary | assign | seq
    value: str = ""
    args: tuple[Node, ...] = ()
    kwargs: tuple[tuple[str, Node], ...] = ()


@dataclass(frozen=True, slots=True)
class OperatorInfo:
    name: str
    category: str
    #: Positional inputs without a default.
    required: int
    #: Most positional inputs accepted; ``None`` when it takes any number.
    maximum: int | None
    #: Each parameter as the definition writes it, e.g. ``("x", "d", "constant = 0")``.
    params: tuple[str, ...] = ()
    #: The names of the inputs written by position, e.g. ``("x", "d")``.
    inputs: tuple[str, ...] = ()
    #: Options BRAIN wants written out although the definition gives a default: ``(("k", "1"),)``.
    explicit: tuple[tuple[str, str], ...] = ()


_SIGNATURE = re.compile(r"([A-Za-z_]\w*)\s*\(")
_VARIADIC = re.compile(r"two or more inputs|all inputs|at least 2 inputs", re.IGNORECASE)
#: Options BRAIN refuses to default ("Required attribute "k" must have a value") although the definitions give one.
EXPLICIT = ("k", "target_tvr")
#: Accepted without those options thousands of times: writing them out would only turn identical Alphas into new text.
DEFAULTS_ACCEPTED = frozenset({"ts_backfill"})
_QUOTES = "\"'“”"


def operator_table(operators: Iterable[dict[str, Any]], scope: str = "REGULAR") -> dict[str, OperatorInfo]:
    """Arity read from each operator's published definition, e.g. ``ts_rank(x, d, constant = 0)``.

    Only operators whose ``scope`` includes ``scope`` are kept, so a COMBO-only operator is never swapped into a
    regular Alpha. An operator whose definition cannot be read is left out, so it never passes validation.
    """
    read: list[tuple[str, dict[str, Any], list[str]]] = []
    for operator in operators:
        name, definition = operator.get("name"), operator.get("definition")
        if not isinstance(name, str) or not isinstance(definition, str):
            continue
        match = _SIGNATURE.search(definition)
        if match is None or match.group(1) != name:
            continue
        params = _signature_params(definition, match.end())
        if params is not None:
            read.append((name, operator, params))
    # A keyword whose "default" is one of these names is an input in disguise: ts_backfill(x, lookback = d, k = 1).
    bare = {p for _, _, params in read for p in params if _bare(p) and p.isidentifier()}
    placeholders = bare - {"true", "false", "nan"}
    table: dict[str, OperatorInfo] = {}
    for name, operator, params in read:
        scopes = operator.get("scope")
        if isinstance(scopes, list) and scopes and scope not in scopes:
            continue
        inputs = tuple(_keyword(p)[0] for p in params if _bare(p) or ("=" in p and _keyword(p)[1] in placeholders))
        variadic = any(".." in p for p in params) or bool(_VARIADIC.search(str(operator.get("description") or "")))
        table[name] = OperatorInfo(
            name=name, category=str(operator.get("category") or ""), required=len(inputs),
            maximum=None if variadic else len(inputs), params=tuple(params), inputs=inputs,
            explicit=() if name in DEFAULTS_ACCEPTED else tuple(
                kv for p in params if "=" in p and (kv := _keyword(p))[0] in EXPLICIT))
    return table


def arity_problem(call: Node, info: OperatorInfo) -> str | None:
    """Why BRAIN would refuse this call's inputs, or nothing."""
    count = len(call.args)
    given = {*info.inputs[:count], *(key for key, _ in call.kwargs)}
    if any(name not in given for name in info.inputs):
        return f"{call.value} does not take {count} input{'' if count == 1 else 's'}."
    if info.maximum is not None and count > info.maximum:
        options = [_keyword(p)[0] for p in info.params if _keyword(p)[0] not in info.inputs]
        named = f": write {options[0]}=… by name" if options else ""
        plural = "" if info.maximum == 1 else "s"
        return f"{call.value} takes {info.maximum} input{plural}{named}."
    return None


def _bare(param: str) -> bool:
    return "=" not in param and ".." not in param


def _keyword(param: str) -> tuple[str, str]:
    name, _, default = param.partition("=")
    return name.strip(), default.strip()


def written_out(call: Node, info: OperatorInfo) -> Node:
    """One call with the options BRAIN will not default written out, as their defaults."""
    given = {k for k, _ in call.kwargs}
    missing = tuple((key, parse(value)) for key, value in info.explicit if key not in given)
    return Node(call.kind, call.value, call.args, call.kwargs + missing) if missing else call


def _signature_params(definition: str, start: int) -> list[str] | None:
    params: list[str] = []
    depth, quoted, current = 0, False, []
    for char in definition[start:]:
        if char in _QUOTES:
            quoted = not quoted
        elif not quoted and char == "(":
            depth += 1
        elif not quoted and char == ")":
            if depth == 0:
                tail = "".join(current).strip()
                return [*params, tail] if tail or params else params
            depth -= 1
        elif not quoted and depth == 0 and char == ",":
            params.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    return None


# -- reading ---------------------------------------------------------------

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<num>(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)
      | (?P<str>"[^"]*"|'[^']*')
      | (?P<name>\$?[A-Za-z_][A-Za-z0-9_]*|\.\.\.)
      | (?P<op>&&|\|\||==|!=|<=|>=|[-+*/^<>!?:(),=;])
    )""",
    re.VERBOSE,
)
#: ``//`` and ``#`` line comments and ``/* */`` blocks; a string is matched first and kept.
_COMMENT = re.compile(r"""("[^"]*"|'[^']*')|/\*.*?\*/|(?://|#)[^\n]*""", re.DOTALL)
_LEVELS: tuple[tuple[str, ...], ...] = (("||",), ("&&",), COMPARISONS, ("+", "-"), ("*", "/"))
#: How deep an expression may go; a real Alpha goes a dozen deep.
MAX_DEPTH = 64


def strip_comments(text: str) -> str:
    """Comments blanked, not removed, so positions hold and an error points at the character the reader sees."""
    return _COMMENT.sub(lambda m: m.group(1) or " " * len(m.group()), text)


def tokenize(text: str) -> list[tuple[str, str, int]]:
    tokens: list[tuple[str, str, int]] = []
    text = strip_comments(text).rstrip()
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None or match.lastgroup is None:
            raise ParseError(f"Cannot read {text[position: position + 12]!r} at {position}.")
        kind = match.lastgroup
        tokens.append((kind, match.group(kind), match.start(kind)))
        position = match.end()
    return tokens


def parse(text: str) -> Node:
    """A tree for the expression, or :class:`ParseError` saying where it stopped making sense."""
    return _Parser(tokenize(text)).program()


def error_position(message: str) -> int | None:
    """The character a ParseError message points at ("... at 17."), if it names one."""
    m = re.search(r" at (\d+)", message)
    return int(m.group(1)) if m else None


class _Parser:
    def __init__(self, tokens: list[tuple[str, str, int]]) -> None:
        self.tokens = tokens
        self.index = 0
        self.depth = 0

    def deeper(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ParseError("The expression nests too deeply.")

    def peek(self, offset: int = 0) -> tuple[str, str, int] | None:
        at = self.index + offset
        return self.tokens[at] if at < len(self.tokens) else None

    def at(self, symbol: str, offset: int = 0) -> bool:
        token = self.peek(offset)
        return token is not None and token[0] == "op" and token[1] == symbol

    def take(self) -> tuple[str, str, int]:
        token = self.peek()
        if token is None:
            raise ParseError("The expression ends too early.")
        self.index += 1
        return token

    def expect(self, symbol: str) -> None:
        token = self.take()
        if token[0] != "op" or token[1] != symbol:
            raise ParseError(f"Expected {symbol!r} at {token[2]}, found {token[1]!r}.")

    def program(self) -> Node:
        statements: list[Node] = []
        while self.peek() is not None:
            if self.at(";"):
                self.take()
                continue
            statements.append(self.statement())
            if self.peek() is not None:
                self.expect(";")
        if not statements:
            raise ParseError("The expression is empty.")
        return statements[0] if len(statements) == 1 else Node("seq", args=tuple(statements))

    def statement(self) -> Node:
        token = self.peek()
        if token is not None and token[0] == "name" and self.at("=", 1):
            self.index += 2
            return Node("assign", token[1], (self.expr(),))
        return self.expr()

    def expr(self) -> Node:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ParseError("The expression nests too deeply.")
        node = self.binary(0)
        if self.at("?"):
            self.take()
            yes = self.expr()
            self.expect(":")
            node = Node("ternary", args=(node, yes, self.expr()))
        self.depth -= 1
        return node

    def binary(self, level: int) -> Node:
        if level == len(_LEVELS):
            return self.power()
        left = self.binary(level + 1)
        chained = 0
        while (token := self.peek()) is not None and token[0] == "op" and token[1] in _LEVELS[level]:
            self.take()
            chained += 1
            self.deeper()
            left = Node("binary", token[1], (left, self.binary(level + 1)))
        self.depth -= chained
        return left

    def power(self) -> Node:
        base = self.unary()
        if not self.at("^"):
            return base
        self.take()
        self.deeper()
        node = Node("binary", "^", (base, self.power()))
        self.depth -= 1
        return node

    def unary(self) -> Node:
        if not (self.at("-") or self.at("!")):
            return self.primary()
        symbol = self.take()[1]
        self.deeper()
        node = Node("unary", symbol, (self.unary(),))
        self.depth -= 1
        return node

    def primary(self) -> Node:
        kind, text, position = self.take()
        if kind in ("num", "str"):
            return Node(kind, text)
        if kind == "name":
            return self.call(text) if self.at("(") else Node("name", text)
        if text == "(":
            inner = self.expr()
            self.expect(")")
            return inner
        raise ParseError(f"Unexpected {text!r} at {position}.")

    def call(self, name: str) -> Node:
        self.expect("(")
        args: list[Node] = []
        kwargs: list[tuple[str, Node]] = []
        if self.at(")"):
            self.take()
            return Node("call", name)
        while True:
            token = self.peek()
            if token is not None and token[0] == "name" and self.at("=", 1):
                self.index += 2
                kwargs.append((token[1], self.expr()))
            else:
                args.append(self.expr())
            if self.at(","):
                self.take()
                continue
            self.expect(")")
            return Node("call", name, tuple(args), tuple(kwargs))


# -- writing ---------------------------------------------------------------

_PRECEDENCE = {"||": 2, "&&": 3, **dict.fromkeys(COMPARISONS, 4), "+": 5, "-": 5, "*": 6, "/": 6, "^": 7}
_TERNARY, _UNARY, _ATOM = 1, 8, 9


def _precedence(node: Node) -> int:
    if node.kind == "binary":
        return _PRECEDENCE[node.value]
    if node.kind == "unary":
        return _UNARY
    if node.kind == "ternary":
        return _TERNARY
    if node.kind in ("assign", "seq"):
        return 0
    return _ATOM


def _wrapped(node: Node, parenthesise: bool) -> str:
    text = render(node)
    return f"({text})" if parenthesise else text


def render(node: Node) -> str:
    kind = node.kind
    if kind in ("num", "str", "name"):
        return node.value
    if kind == "call":
        parts = [render(a) for a in node.args] + [f"{k}={render(v)}" for k, v in node.kwargs]
        return f"{node.value}({', '.join(parts)})"
    if kind == "unary":
        child = node.args[0]
        return node.value + _wrapped(child, _precedence(child) < _UNARY)
    if kind == "binary":
        own = _PRECEDENCE[node.value]
        left, right = node.args
        if node.value == "^":
            left_parens, right_parens = _precedence(left) <= own, _precedence(right) < own
        else:
            left_parens, right_parens = _precedence(left) < own, _precedence(right) <= own
        return f"{_wrapped(left, left_parens)} {node.value} {_wrapped(right, right_parens)}"
    if kind == "ternary":
        condition, yes, no = node.args
        return f"{_wrapped(condition, _precedence(condition) <= _TERNARY)} ? {render(yes)} : {render(no)}"
    if kind == "assign":
        return f"{node.value} = {render(node.args[0])}"
    if kind == "seq":
        return "; ".join(render(s) for s in node.args)
    raise ValueError(f"Unknown node kind {node.kind!r}")


# -- walking ---------------------------------------------------------------

Path = tuple[int, ...]


def children(node: Node) -> tuple[Node, ...]:
    if not node.kwargs:
        return node.args
    return (*node.args, *(value for _, value in node.kwargs))


def walk(node: Node, path: Path = ()) -> list[tuple[Path, Node]]:
    """Every node with the route to it. Keyword values follow positional arguments."""
    found: list[tuple[Path, Node]] = []

    def go(n: Node, p: Path) -> None:
        found.append((p, n))
        for i, c in enumerate(children(n)):
            go(c, (*p, i))

    go(node, path)
    return found


def walk_with_parents(node: Node, path: Path = (), parent: Node | None = None) -> list[tuple[Path, Node, Node | None]]:
    found: list[tuple[Path, Node, Node | None]] = []

    def go(n: Node, p: Path, up: Node | None) -> None:
        found.append((p, n, up))
        for i, c in enumerate(children(n)):
            go(c, (*p, i), n)

    go(node, path, parent)
    return found


def node_at(node: Node, path: Path) -> Node:
    for index in path:
        node = children(node)[index]
    return node


def relabel(node: Node, values: dict[Path, str]) -> Node:
    """``node`` with the value at each of ``values``' paths replaced, in one pass (the shape never changes)."""
    if not values:
        return node
    own = values.get(())
    below: dict[int, dict[Path, str]] = {}
    for path, value in values.items():
        if path:
            below.setdefault(path[0], {})[path[1:]] = value
    if not below:
        return replace(node, value=own) if own is not None else node
    count = len(node.args)
    return replace(
        node,
        value=node.value if own is None else own,
        args=tuple(relabel(a, below[i]) if i in below else a for i, a in enumerate(node.args)),
        kwargs=tuple((key, relabel(v, below[count + i])) if count + i in below else (key, v)
                     for i, (key, v) in enumerate(node.kwargs)),
    )


def replace_at(node: Node, path: Path, new: Node) -> Node:
    if not path:
        return new
    index, rest = path[0], path[1:]
    if index < len(node.args):
        args = list(node.args)
        args[index] = replace_at(args[index], rest, new)
        return replace(node, args=tuple(args))
    kwargs = list(node.kwargs)
    key, value = kwargs[index - len(node.args)]
    kwargs[index - len(node.args)] = (key, replace_at(value, rest, new))
    return replace(node, kwargs=tuple(kwargs))


def data_fields(tree: Node, *, grouping: bool = False) -> list[str]:
    """The data fields an expression reads, the way BRAIN counts them, in order of first appearance.

    Not data: anything assigned earlier, a keyword value (``driver=gaussian`` names an option), a grouping field
    (unless ``grouping``) and the literals true/false/nan.
    """
    nodes = walk_with_parents(tree)
    assigned = {n.value for _, n, _ in nodes if n.kind == "assign"}
    return list(dict.fromkeys(
        n.value for path, n, parent in nodes
        if n.kind == "name"
        and not (parent is not None and path[-1] >= len(parent.args))
        and n.value not in assigned
        and (grouping or n.value not in GROUPING_FIELDS)
        and n.value.lower() not in ("true", "false", "nan")))


def operator_count(node: Node) -> int:
    """Power Pool's count: every call and arithmetic or logical operator, backfills excepted; -5 is a constant."""
    negative = node.kind == "unary" and node.value == "-" and node.args[0].kind == "num"
    own = int((node.kind == "call" and node.value not in UNCOUNTED_OPERATORS)
              or (node.kind in ("unary", "binary", "ternary") and not negative))
    return own + sum(operator_count(child) for child in children(node))


def guarded(parent: Node | None, path: Path) -> bool:
    """A grouping input or a keyword value: slots that must not receive a signal."""
    if parent is None:
        return False
    if path[-1] >= len(parent.args):
        return True
    return parent.kind == "call" and parent.value.startswith("group_") and path[-1] >= 1


# -- changing --------------------------------------------------------------

def graft_sites(tree: Node) -> list[Path]:
    """Where a graft may go into ``tree``: any call but the root, outside the grouping and keyword slots."""
    return [p for p, n, up in walk_with_parents(tree) if p and n.kind == "call" and not guarded(up, p)]


def grafts(tree: Node) -> list[Node]:
    """The calls a graft may take from ``tree``."""
    return [n for p, n, up in walk_with_parents(tree) if n.kind == "call" and not guarded(up, p)]


def graft(rng: random.Random, left: Node, sites: list[Path], taken: list[Node]) -> Node | None:
    """A call from one parent grafted in place of a call inside the other; None when that changes nothing."""
    if not sites or not taken:
        return None
    child = replace_at(left, rng.choice(sites), rng.choice(taken))
    return None if child == left else child


def validate(tree: Node, table: dict[str, OperatorInfo], known_names: set[str] | frozenset[str]) -> list[str]:
    """Why BRAIN would reject this tree, or nothing."""
    problems: list[str] = []
    nodes = walk_with_parents(tree)
    assigned = {node.value for _, node, _ in nodes if node.kind == "assign"}
    for path, node, parent in nodes:
        if node.kind == "call":
            info = table.get(node.value)
            if info is None:
                problems.append(f"{node.value} is not an operator on BRAIN.")
                continue
            if (problem := arity_problem(node, info)) is not None:
                problems.append(problem)
        elif (node.kind == "name" and not (parent is not None and path[-1] >= len(parent.args))
              and node.value not in known_names and node.value not in assigned
              and node.value.lower() not in ("true", "false", "nan")):
            problems.append(f"{node.value} is not a data field here.")
    return problems
