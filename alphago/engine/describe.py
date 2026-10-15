"""Alpha descriptions in BRAIN's "Idea and Rationale" format (needed for Power Pool and recommended generally)."""

from __future__ import annotations

from ..db import db
from ..db.models import Alpha, Candidate
from ..knowledge import pack
from ..llm import llm
from ..llm.prompts import render
from ..llm.schemas import AlphaDescription


def fallback_description(a: Alpha, hypothesis: str | None) -> str:
    fields = ", ".join(a.fields or []) or "the selected data fields"
    return (f"Idea: {hypothesis or 'Cross-sectional signal built from ' + fields + '.'}\n"
            f"Rationale for data used: {fields} capture information that is not yet fully priced, and their relative "
            f"level versus peers is informative about future returns.\n"
            f"Rationale for operators used: time-series normalisation removes scale and persistent levels, and "
            f"group-relative ranking compares each stock with its peers to limit industry and size exposure.")


async def describe_alpha(alpha_id: str) -> str:
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
        cand = await s.get(Candidate, a.candidate_id) if a and a.candidate_id else None
    if a is None:
        raise ValueError("unknown alpha")
    hypothesis = cand.hypothesis if cand else None
    if not llm.configure():
        return fallback_description(a, hypothesis)
    prompt = await render("describe", expression=a.code, settings=pack.settings_text(a.settings or {}),
                          fields=", ".join(a.fields or []), hypothesis=hypothesis or "")
    desc = await llm.generate_json(prompt, AlphaDescription, role="describe", purpose="describe", thinking="low")
    text = desc.render()
    return text if len(text) >= 100 else fallback_description(a, hypothesis)
