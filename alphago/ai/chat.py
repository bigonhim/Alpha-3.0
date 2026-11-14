"""A conversation with the assistant, and the field picks that come out of it (ported from alpha-harness
`llm/chat.py`).

Anything a consultant types steers the search, so the assistant's job is narrow: read the hunch and come back with
**actual data fields** to run on. The conversation is the input; a field list is the output; Search Lab takes it
from there ("Use in Search Lab" stores a hand-off in `db.kv("handoff:search")`).

Two controls: the **model** (each with its own daily budget, see `registry`) and the **reasoning** level — thinking
tokens are billed like the answer's, so "think harder" is a real cost rather than a free upgrade.

History lives in the database so a conversation survives a restart. Nothing is stored until the model has answered,
so a failed exchange leaves no conversation behind that holds only the question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select

from ..db import db
from ..db.models import DataField, LLMCall, utcnow
from ..db.models_ai import ChatMessage, ChatThread
from ..llm import llm
from ..llm.base import loads_or
from .prompts import ASSISTANT, CHARS_PER_TOKEN
from .registry import ModelInfo, registry, split_ref

log = logging.getLogger(__name__)

#: How much of a conversation is replayed to the model: long enough to hold a thread, short enough that prompt
#: tokens do not eat the per-minute budget.
HISTORY_TURNS = 12
CLIP_CHARS = 600
FIELD_LIMIT = 400
COVERAGE_FLOOR = 0.5
TEMPERATURE = 0.6

#: Reasoning effort in words; the levels are AlphaGo's thinking levels (Google's MINIMAL / LOW / MEDIUM / HIGH).
REASONING: dict[str, dict[str, str]] = {
    "quick": {"level": "minimal", "label": "Quick", "description": "Answers straight away. Best for looking "
                                                                   "something up."},
    "normal": {"level": "low", "label": "Normal", "description": "Thinks a little first. The right choice almost "
                                                                 "always."},
    "careful": {"level": "medium", "label": "Careful", "description": "Works through it properly. Slower, and uses "
                                                                      "more of the daily budget."},
    "deep": {"level": "high", "label": "Deep", "description": "Thinks hard before answering. Save it for a genuinely "
                                                              "difficult question — it is the most expensive setting."},
}
DEFAULT_REASONING = "normal"
REASONING_NOTE = ("Thinking harder is not free — those tokens come out of the same daily budget as the answer. "
                  "Normal is the right choice almost always.")


class ChatError(ValueError):
    """Something the consultant can act on (no fields downloaded, an unknown conversation)."""


class FieldPick(BaseModel):
    field: str
    why: str


class FieldPickReply(BaseModel):
    """What the assistant returns: kept small on purpose — a handful of things to run, not a survey."""
    reply: str
    picks: list[FieldPick] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Scope:
    region: str
    delay: int
    universe: str
    instrument_type: str = "EQUITY"

    @property
    def label(self) -> str:
        return f"{self.region}/D{self.delay}/{self.universe}"


def clip(text: Any, limit: int) -> str:
    """One line, at most `limit` characters, broken on a word."""
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[:limit].rsplit(" ", 1)[0] + "…"


# ---------------------------------------------------------------- threads

async def threads(limit: int = 30) -> list[ChatThread]:
    async with db.session() as s:
        return list((await s.scalars(select(ChatThread).order_by(ChatThread.updated_at.desc(), ChatThread.id.desc())
                                     .limit(limit))).all())


async def thread(thread_id: int) -> tuple[ChatThread, list[ChatMessage]] | None:
    async with db.session() as s:
        row = await s.get(ChatThread, thread_id)
        if row is None:
            return None
        messages = list((await s.scalars(select(ChatMessage).where(ChatMessage.thread_id == thread_id)
                                         .order_by(ChatMessage.id))).all())
        return row, messages


async def start(title: str, scope: Scope) -> ChatThread:
    async with db.write() as s:
        row = ChatThread(title=" ".join(title.split())[:120] or "Conversation", instrument_type=scope.instrument_type,
                         region=scope.region, delay=scope.delay, universe=scope.universe)
        s.add(row)
        await s.flush()
        return row


async def remove(thread_id: int) -> None:
    async with db.write() as s:
        await s.execute(delete(ChatMessage).where(ChatMessage.thread_id == thread_id))
        await s.execute(delete(ChatThread).where(ChatThread.id == thread_id))


def scope_of(row: ChatThread) -> Scope:
    return Scope(row.region, int(row.delay), row.universe, row.instrument_type or "EQUITY")


# ---------------------------------------------------------------- the turn

def model_for(ref: str | None) -> tuple[str, str, ModelInfo]:
    """(provider, model id, limits) of the model asked for; with none named, the active provider's own pick."""
    if ref:
        provider, model_id = split_ref(ref)
        if not model_id:
            raise ChatError(f"{ref!r} does not name a model: use provider:model.")
        return provider, model_id, registry.info(provider, model_id)
    provider = llm.name
    model_id = llm.pick("ideation")
    return provider, model_id, registry.info(provider, model_id)


async def say(thread_id: int | None, text: str, *, scope: Scope, model: str | None = None,
              reasoning: str = DEFAULT_REASONING, dataset_ids: list[str] | None = None) -> dict[str, Any]:
    """One exchange: their words in, a reply and a set of field picks out."""
    text = (text or "").strip()
    if not text:
        raise ChatError("Type a message first.")
    if reasoning not in REASONING:
        raise ChatError(f"Unknown reasoning level {reasoning!r}.")
    history: list[ChatMessage] = []
    if thread_id is not None:
        found = await thread(thread_id)
        if found is None:
            raise ChatError(f"No conversation {thread_id}.")
        row, history = found
        scope = scope_of(row)                # a conversation stays on the market it started in

    provider, model_id, info = model_for(model)
    transcript = _transcript(history)
    message = f"THEIR MESSAGE:\n{text}"
    # the fields get what the model's prompt allows once everything else is in
    room = info.prompt_tokens * CHARS_PER_TOKEN - len(ASSISTANT) - len(transcript) - len(message) - 200
    fields, available, catalog_note = await field_menu(scope, dataset_ids or [], room)
    if not fields:
        raise ChatError(f"No data fields of {scope.label} are in the local catalogue yet, so there is nothing for the "
                        "assistant to choose from. Sync this market's datasets (Data Explorer or Datasets) first.")
    prompt = "\n\n---\n\n".join([f"DATA FIELDS AVAILABLE IN {scope.label}:\n{fields}", transcript, message])

    since = utcnow()
    raw, answered_by, answered_model = await llm.generate_raw_with(
        provider, model_id, prompt, FieldPickReply, system=ASSISTANT, role="ideation", purpose="assistant",
        thinking=REASONING[reasoning]["level"], temperature=TEMPERATURE)
    parsed = parse(raw)
    # drop anything not really in the catalogue: a hallucinated field costs a simulation to discover
    kept = [p for p in parsed["picks"] if p.get("field") in available]
    dropped = [p["field"] for p in parsed["picks"] if p.get("field") not in available]
    if dropped:
        log.warning("assistant named fields not in the catalogue: %s (%s)", dropped[:10], scope.label)
    tokens = await _tokens_since(since)

    if thread_id is None:
        thread_id = (await start(text, scope)).id
    meta = {"picks": kept, "datasets": parsed["datasets"], "dropped": dropped, "catalog_note": catalog_note,
            "model": answered_model or model_id, "provider": answered_by, "reasoning": reasoning, "tokens": tokens}
    async with db.write() as s:
        s.add(ChatMessage(thread_id=thread_id, role="user", text=text, meta={}))
        s.add(ChatMessage(thread_id=thread_id, role="assistant", text=parsed["reply"],
                          meta={k: v for k, v in meta.items() if v is not None}))
        row = await s.get(ChatThread, thread_id)
        if row is not None:
            row.updated_at = utcnow()
    return {"thread_id": thread_id, "reply": parsed["reply"], **meta}


async def field_menu(scope: Scope, dataset_ids: list[str], room: int) -> tuple[str, set[str], str | None]:
    """The fields the assistant may choose from, as many as fit in `room` characters: the text, their ids, a note.

    Ordered by coverage and capped: a field present on 4% of the market cannot carry a signal about it however good
    the idea, and listing thousands would spend the context window on names. The coverage floor is lifted once a
    dataset is named, because event-driven data is thin by nature and whole datasets sit below it."""
    where = [DataField.region == scope.region, DataField.delay == scope.delay, DataField.universe == scope.universe]
    if dataset_ids:
        where.append(DataField.dataset_id.in_(list(dataset_ids)))
    else:
        where.append(DataField.coverage >= COVERAGE_FLOOR)
    async with db.session() as s:
        total = await s.scalar(select(func.count(DataField.pk)).where(*where)) or 0
        rows = (await s.execute(select(DataField.id, DataField.dataset_id, DataField.description).where(*where)
                                .order_by(DataField.coverage.desc().nulls_last(), DataField.id)
                                .limit(FIELD_LIMIT))).all()
    if not rows:
        return "", set(), None
    lines: list[str] = []
    for field_id, dataset_id, description in rows:
        line = f"  {field_id} ({dataset_id}): {clip(description, 110)}"
        if room - len(line) < 0:
            break
        room -= len(line) + 1
        lines.append(line)
    if not lines:
        raise ChatError("The model's Max prompt tokens leave no room for a single field. Raise it under LLM "
                        "Integration › Models.")
    note = None
    if int(total) > len(lines):
        note = f"The assistant was shown the {len(lines)} widest-coverage fields of {total} in this market."
    return "\n".join(lines), {str(r[0]) for r in rows[:len(lines)]}, note


def _transcript(history: list[ChatMessage]) -> str:
    recent = history[-HISTORY_TURNS:]
    if not recent:
        return "THE CONVERSATION SO FAR:\n  (this is the first message)"
    lines: list[str] = []
    for m in recent:
        if m.role == "user":
            lines.append(f"  They: {m.text.strip()[:CLIP_CHARS]}")
            continue
        # replay its own picks, so "why the second one?" can be answered
        picks = ", ".join(str(p.get("field")) for p in (m.meta or {}).get("picks") or [])
        said = f"  You: {m.text.strip()[:CLIP_CHARS]}"
        lines.append(f"{said} [picked: {picks}]" if picks else said)
    return "THE CONVERSATION SO FAR:\n" + "\n".join(lines)


def parse(text: str) -> dict[str, Any]:
    """Read the model's JSON, degrading to plain text rather than failing: a reply the consultant can read is worth
    more than a parse error, so an unparseable response becomes the reply itself with no picks."""
    payload = loads_or(text or "")
    if not payload:
        return {"reply": (text or "").strip(), "picks": [], "datasets": []}
    picks, datasets = payload.get("picks"), payload.get("datasets")
    return {
        "reply": str(payload.get("reply") or "").strip() or (text or "").strip(),
        "picks": [{"field": p["field"], "why": str(p.get("why") or "")} for p in picks
                  if isinstance(p, dict) and isinstance(p.get("field"), str) and p["field"]]
        if isinstance(picks, list) else [],
        "datasets": [str(d) for d in datasets if d] if isinstance(datasets, list) else [],
    }


async def _tokens_since(since: Any) -> int | None:
    """Tokens the assistant call just spent, from the call log (providers report them there)."""
    try:
        async with db.session() as s:
            row = (await s.execute(select(func.sum(LLMCall.prompt_tokens), func.sum(LLMCall.output_tokens))
                                   .where(LLMCall.purpose == "assistant", LLMCall.ts >= since,
                                          LLMCall.status == "ok"))).one()
        total = int(row[0] or 0) + int(row[1] or 0)
        return total or None
    except Exception:
        return None


async def hand_off(scope: Scope, dataset_ids: list[str], field_ids: list[str]) -> dict[str, Any]:
    """Store the picks for Search Lab, which reads `handoff:search` when it opens."""
    payload = {"region": scope.region, "delay": scope.delay, "universe": scope.universe,
               "dataset_ids": list(dict.fromkeys(dataset_ids)), "field_ids": list(dict.fromkeys(field_ids))}
    await db.kv_set("handoff:search", payload)
    return payload
