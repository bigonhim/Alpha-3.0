"""The assistant's system prompt (alpha-harness `llm/prompts.py` ASSISTANT, word for word; only the application's name
is AlphaGo's), and the prompt library the LLM Integration page lists with a token estimate for each prompt.

Three principles run through it: write for a reader with no quantitative training, never invent a field name (a
hallucinated one costs a simulation from a daily quota), and say plainly what is uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Roughly four characters per token. Only used to size prompts and warn, never to bill.
CHARS_PER_TOKEN = 4

ASSISTANT = """\
You are the research assistant inside AlphaGo. The person talking to you has no \
background in finance and will give this ten minutes. Write for them: short sentences, \
no jargon without a plain-language gloss, never "simply" or "just".

Your job is to turn a hunch into something runnable. They describe an idea in ordinary \
words; you find the data that could measure it.

You are given a list of the data fields available to them. Use ONLY those names. Never \
invent one — a made-up field becomes a simulation that fails, and they have a limited \
number each day.

Return JSON with:
  "reply"    — two or three sentences. What you think they mean, and what you picked.
  "picks"    — the fields worth running, each with one plain sentence on why.
  "datasets" — the dataset ids those fields came from.

Pick between three and ten fields. Fewer is better than padding the list. If nothing in \
the data fits the idea, say so plainly in "reply", return an empty "picks", and suggest \
the nearest thing that does exist.
"""


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class Prompt:
    slug: str
    label: str
    purpose: str
    body: str

    @property
    def characters(self) -> int:
        return len(self.body)

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.body)


_PURPOSES = {
    "ideation": "Proposes new alphas for a dataset (the AI generator in research and autopilot).",
    "refine": "Improves a near-miss alpha against its failing BRAIN checks.",
    "repair": "Targets the exact checks an alpha fails, from the measured diagnosis.",
    "dataset_brief": "Summarises a dataset and its most useful fields.",
    "describe": "Writes an alpha's description in BRAIN's Idea / Rationale template.",
    "genius": "Plans a Genius-programme research session.",
    "super_describe": "Describes a Super Alpha's combination and selection.",
    "dataset_rating": "Rates datasets by theme for the Dataset Intelligence page.",
    "osmosis_predict": "Predicts how an alpha will fare in Osmosis review.",
    "osmosis_explain": "Explains an Osmosis result in plain words.",
    "self_check": "Critiques a batch of ideas before they are simulated (AI self-check).",
    "research_plan": "Plans a research run: datasets, themes and settings.",
}


async def library() -> list[Prompt]:
    """Every prompt AlphaGo sends, in the version in use: the assistant's, each Jinja template (as edited on this
    page) and the knowledge pack sent as the system instruction of research calls."""
    from ..llm.prompts import DEFAULTS, get_template
    out = [Prompt("assistant", "Assistant", "Turns an idea typed in AI Assistant into the data fields that could "
                                            "measure it.", ASSISTANT)]
    for name in DEFAULTS:
        out.append(Prompt(name, name.replace("_", " ").capitalize(), _PURPOSES.get(name, "Template."),
                          await get_template(name)))
    try:
        from ..knowledge.pack import static_prefix
        out.append(Prompt("knowledge_pack", "Knowledge pack", "Sent as the system instruction of research calls: "
                                                             "BRAIN rules, your operators, thresholds, templates.",
                          await static_prefix()))
    except Exception:
        pass
    return out
