"""The description BRAIN requires before it takes a Power Pool Alpha, written by a model (alpha-harness
tools/description and api/alphas.write_description).

BRAIN's own client accepts one only in its template: each heading starting a line, text after every one, the whole at
least 100 characters. The model is asked for the three parts alone ({idea, data, operators}) and the template is put
together here, so its shape never rests on the model. The model sees the expression and only the fields and operators
it uses, each with BRAIN's own description; a part under 100 characters is asked for once more, naming it (free models
often stop early). Nothing is saved: the caller shows it for the consultant to edit and save.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select

from ..db import db
from ..db.models import Operator
from .common import ToolError, alpha_body, code_of, described, fields_in_order

PARTS = ("idea", "data", "operators")
HEADINGS = ("Idea", "Rationale for data used", "Rationale for operators used")
MIN_PART = 100
MIN_LENGTH = 100
ARITHMETIC = "Arithmetic"           # add, divide, abs and the like: a model knows them without being told

SYSTEM = """\
You write the description BRAIN requires before it takes a Power Pool Alpha: the idea behind \
the Alpha, why its data suits that idea, and what its operators do.

WHAT YOU ARE GIVEN
The Alpha's Fast Expression, then each data field it reads and each operator it applies, with \
BRAIN's own description of every one. Plain arithmetic (add, divide, abs, log and the like) is \
not listed: you know it already. A field's type is its shape: MATRIX is one value per stock \
per day, VECTOR several, GROUP the group a stock belongs to.

HOW TO WRITE IT
- Idea: the market behaviour the Alpha bets on, and which stocks it ranks high and which low. \
Two to four sentences.
- Rationale for data used: each data field by name, what it measures and why it carries that \
behaviour. Name each grouping field too, and say what it compares stocks within.
- Rationale for operators used: each operator by name, in the order the expression applies them \
from the inside out: what it does to the signal here, and why that is needed.
- Each of the three parts at least 100 characters.
- Plain sentences: no Markdown, no lists, no headings inside the text.
- Use only the fields and operators in the expression, spelled exactly. Claim nothing about \
results: no Sharpe, no returns, no promise that it works.

ANSWER
Only JSON: {"idea": "...", "data": "...", "operators": "..."}, each value the text of that part \
without its heading.
"""

_TEMPLATE = re.compile(
    r"^\s*idea:\s+(?P<idea>.+?)\n\s*rationale for data used:\s+(?P<data>.+?)\n"
    r"\s*rationale for operators used:\s+(?P<operators>.+)",
    re.IGNORECASE | re.DOTALL,
)
# BRAIN's client check (frontend screens/alpha/brain.ts descriptionProblems)
_CLIENT = re.compile(r"^\s*idea:(\s+(.+?)\n)\s*rationale for data used:\s+((.+?)\n)\s*rationale for operators "
                     r"used:(\s+(.+?))", re.IGNORECASE | re.DOTALL)


class DescriptionParts(BaseModel):
    idea: str
    data: str
    operators: str


def operators_in(expression: str) -> list[str]:
    """Every operator the expression applies, by BRAIN's name, once each in order of appearance. Symbols included
    (`x / y` is divide); a minus sign on a number is the number."""
    from ..fastexpr import Assign, Num, Unary, parse, walk
    from ..fastexpr.metrics import _operator_name
    found: list[str] = []
    for stmt in parse(expression).statements:
        root = stmt.value if isinstance(stmt, Assign) else stmt
        for node in walk(root):
            if isinstance(node, Unary) and node.op == "-" and isinstance(node.operand, Num):
                continue
            name = _operator_name(node)
            if name is not None:
                found.append(name)
    return list(dict.fromkeys(found))


def message(expression: str, fields: list[dict[str, Any]], operators: list[dict[str, Any]]) -> str:
    """What the model is shown: the expression, then only its own fields and operators with BRAIN's descriptions,
    then the template to fill. Arithmetic operators are left out."""
    operators = [o for o in operators if o.get("category") != ARITHMETIC]
    lines = [f"{f['name']} ({f.get('type') or 'UNKNOWN'}): {f.get('description') or 'No description'}" for f in fields]
    lines += [f"{o.get('definition') or o['name']}: {o.get('description') or 'No description'}" for o in operators]
    template = "\n".join(f"{heading}: " for heading in HEADINGS)
    return (f"{expression.strip()}\n\n" + "\n".join(lines)
            + "\n\nAlpha Description following the given Template:\n" + f"```\n{template}\n```")


def _plain(heading: str, text: str) -> str:
    """One line, as the template is laid out, without its heading if the model wrote that too."""
    text = " ".join(str(text).split())
    if text.lower().startswith(heading.lower()):
        text = text[len(heading):].lstrip(" :")
    return text


def assemble(parts: dict[str, Any] | DescriptionParts) -> str | None:
    """The description in BRAIN's template from the three parts; None when one is missing."""
    values = parts.model_dump() if isinstance(parts, DescriptionParts) else parts
    texts = [_plain(h, values.get(p) or "") for h, p in zip(HEADINGS, PARTS, strict=True)]
    if not all(texts):
        return None
    return "\n".join(f"{h}: {t}" for h, t in zip(HEADINGS, texts, strict=True))


def read(answer: str) -> str | None:
    """A description written out as the template by the model itself (when it ignored the JSON asked for)."""
    found = _TEMPLATE.search(answer or "")
    return assemble({p: found.group(p) for p in PARTS}) if found else None


def short(written: str) -> list[str]:
    """The headings whose text has fewer than MIN_PART characters."""
    found = _TEMPLATE.search(written)
    return [h for h, p in zip(HEADINGS, PARTS, strict=True) if found is None or len(found.group(p).strip()) < MIN_PART]


def _template_gap(text: str) -> str | None:
    """The first heading missing, out of order or with nothing after it."""
    lower = text.lower()
    starts: list[int] = []
    for heading in HEADINGS:
        at = lower.find(f"{heading.lower()}:", starts[-1] if starts else 0)
        if at == -1:
            return heading
        starts.append(at)
    for i, heading in enumerate(HEADINGS):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        if not text[starts[i] + len(heading) + 1:end].strip():
            return heading
    return None


def description_problems(text: str) -> list[str]:
    """What BRAIN says is wrong with a Power Pool description, in its words, and which heading."""
    out = []
    if len((text or "").strip()) < MIN_LENGTH:
        out.append(f"Ensure this field has at least {MIN_LENGTH} characters.")
    found = _CLIENT.search(text or "")
    if not (found and all((found.group(i) or "").strip() for i in (1, 3, 5))):
        gap = _template_gap(text or "")
        out.append("Ensure the description is following the provided template."
                   + (f' "{gap}:" is missing, out of order or empty.' if gap else ""))
    return out


async def write_description(alpha_id: str) -> str:
    """A Power Pool description in BRAIN's template, by the model set up on the LLM Integration page."""
    from ..fastexpr import FastExprSyntaxError
    from ..llm import LLMError, llm
    if not llm.configure():
        raise ToolError("Write with AI needs a model: set one up on the LLM Integration page.")
    body = await alpha_body(alpha_id)
    code = code_of(body)
    try:
        fields = fields_in_order(code, grouping=True) if code else None
        ops = operators_in(code) if code else None
    except (FastExprSyntaxError, ValueError):
        fields = ops = None
    if fields is None or ops is None:
        raise ToolError(f"{alpha_id} has no expression this app can read, so there is nothing to describe.")
    known = await described(fields, str((body.get("settings") or {}).get("region") or ""))
    async with db.session() as s:
        by_name = {o.name: o for o in await s.scalars(select(Operator).where(Operator.name.in_(ops)))} if ops else {}
    user = message(code,
                   [{"name": f, "type": getattr(known.get(f), "type", None),
                     "description": getattr(known.get(f), "description", None)} for f in fields],
                   [{"name": n, "category": getattr(by_name.get(n), "category", None),
                     "definition": getattr(by_name.get(n), "definition", None),
                     "description": getattr(by_name.get(n), "description", None)} for n in ops])
    retried = False
    while True:
        try:
            parts = await llm.generate_json(user, DescriptionParts, system=SYSTEM, role="describe",
                                            purpose="pp_description")
        except LLMError as exc:
            raise ToolError(f"The model did not answer in the template ({exc.message[:200]}). Try again, or choose "
                            "another model on the LLM Integration page.") from exc
        written = assemble(parts)
        if written is None:
            raise ToolError("The model left a part of the template empty. Try again, or choose another model on the "
                            "LLM Integration page.")
        missing = short(written)
        if not missing:
            return written
        if retried:
            raise ToolError(f"The model wrote fewer than {MIN_PART} characters for {', '.join(missing)}, twice. Try "
                            "again, or choose another model on the LLM Integration page.")
        retried = True
        user += (f"\n\nYour last answer was too short for {', '.join(missing)}. Each part needs at least {MIN_PART} "
                 "characters.")
