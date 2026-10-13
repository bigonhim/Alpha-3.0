"""Prompt templates (Jinja2). Defaults live here; edits from the LLM Integration page are stored in the database."""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined

from ..db import db

_env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)

DEFAULTS: dict[str, str] = {
    "ideation": """\
TASK: Propose {{ n }} NEW, diverse WorldQuant BRAIN alphas.
Goal: {{ goal }}
Research move: {{ move }} — {{ move_hint }}
Strategy themes to rotate through: {{ themes }}
Target settings: {{ settings }}

{{ dataset_context }}

{% if examples %}Your best existing alphas (do NOT copy; new ideas must be economically different to stay uncorrelated):
{{ examples }}
{% endif %}
{% if avoid %}Ideas that already FAILED (avoid these shapes):
{{ avoid }}
{% endif %}
Requirements:
- Use ONLY fields listed above (plus pv1 basics: close, open, high, low, vwap, volume, returns, cap, adv20) and
  ONLY operators from the operator list.
- Each idea must be a complete FASTEXPR expression with a clear economic hypothesis, ≤ 6 operators, 1–3 fields.
- Wrap sparse data with ts_backfill; normalise levels (rank/zscore/ts_rank/group_rank or divide by cap).
- Vary windows, groups and operators across ideas; no two ideas may share the same skeleton.
- Suggest neutralization / decay / truncation per idea following the settings heuristics.
Return JSON matching the schema.""",

    "refine": """\
TASK: Improve this WorldQuant BRAIN alpha. Produce {{ k }} variants, each fixing the failing tests below while keeping
the core economic idea. Prefer small, principled changes (normalisation, grouping, windows, turnover control,
conditioning with trade_when, combining with a complementary field). Avoid overfitting: no exotic constants.

Alpha: {{ expression }}
Settings: {{ settings }}
In-sample: Sharpe {{ sharpe }}, Fitness {{ fitness }}, Turnover {{ turnover }}, Returns {{ returns }},
Drawdown {{ drawdown }}, Margin {{ margin }}
Failing / warning checks:
{{ failing }}
{% if yearly %}Yearly stats:
{{ yearly }}
{% endif %}
{% if dataset_context %}{{ dataset_context }}
{% endif %}
Return JSON matching the schema (each idea = one variant with its suggested settings).""",

    "repair": """\
These FASTEXPR expressions were rejected by the validator. Fix each one (keep its idea) so it uses only the listed
operators/fields with correct arguments. Return the corrected ideas as JSON matching the schema.
{{ errors }}""",

    "dataset_brief": """\
TASK: Analyse this BRAIN dataset for alpha research. Explain what the data measures, which fields matter and how to
use them, and propose 5–8 distinct idea families as FASTEXPR templates (use {window} and {group} placeholders).
Mention pitfalls (sparsity, units, update frequency, look-ahead illusions) and suggested settings.

{{ dataset_context }}
Return JSON matching the schema.""",

    "describe": """\
Write a concise BRAIN alpha description in the official "Idea and Rationale" format (each part 1–3 sentences,
total at least 150 characters, plain English, no marketing language).
Alpha expression: {{ expression }}
Settings: {{ settings }}
Fields used: {{ fields }}
{% if hypothesis %}Original hypothesis: {{ hypothesis }}{% endif %}
Return JSON matching the schema.""",

    "genius": """\
You coach a WorldQuant BRAIN consultant toward the next Genius level this quarter.
Current level: {{ level }}. Target level: {{ target }}.
Gaps: {{ gaps }}
Tie-breaker metrics (lower operator/field averages are better; more distinct operators/fields is better): {{ tiebreak }}
Pyramids this quarter (region/delay/category: alphas, multiplier): {{ pyramids }}
Days left in the quarter: {{ days_left }}.
Produce 5–10 concrete, prioritised "Genius signals" (actionable research goals), e.g. lighting specific pyramids with
3 alphas, raising submissions, keeping alphas simple. Return JSON matching the schema.""",

    "super_describe": """\
Write the two descriptions BRAIN requires for a SuperAlpha (each at least 120 characters, plain English).
Selection expression: {{ selection }}
Combo expression: {{ combo }}
Components: {{ components }}
Explain why this selection picks good, diverse alphas and how the combo weights them. Return JSON.""",

    "dataset_rating": """\
TASK: Rate each BRAIN dataset below for alpha research in {{ scope }}. For each dataset give:
- ai_score: overall promise for NEW, uncorrelated, submittable alphas (crowded datasets with many alphas score lower;
  useful coverage, clear economics and pyramid multipliers above 1 score higher);
- single / double / interaction: how well it suits single-field alphas, pairing with another dataset, and
  field × field interactions inside the dataset;
- recommended: the best of single | double | interaction;
- theme: one of {{ themes }};
- reason: one short sentence.
Rate every dataset listed, using its dataset_id exactly.

{{ datasets }}
Return JSON matching the schema.""",

    "osmosis_predict": """\
You are a quant finance expert reviewing a WorldQuant BRAIN consultant's Osmosis allocation (100,000 points per
region/delay scope spread over submitted alphas). The real Osmosis result is out of sample; the numbers below come from
in-sample daily PnL, so be honest about uncertainty.
Osmosis score the consultant sees now: {{ score_now }}
Per-scope numbers:
{{ numbers }}
Predict the next Osmosis score as a range, say whether it moves up, flat or down, and give the main drivers.
Answer in {{ language }}{% if language == "Hinglish" %} (Hindi written in Latin script mixed with English finance
terms){% endif %}. Return JSON matching the schema.""",

    "osmosis_explain": """\
You are a quant finance expert. Explain the main mistakes in this WorldQuant BRAIN consultant's Osmosis allocation
and what to do before the next weekly lock. Use only these numbers (in-sample estimates, not a certainty):
{{ numbers }}
Rules of thumb: 100,000 points per scope; at least {{ min_alphas }} alphas; leave out alphas whose last-2-year Sharpe is
below {{ min_sharpe2y }} or turnover above {{ max_turnover }}; avoid concentrating points on correlated alphas.
Answer in {{ language }}{% if language == "Hinglish" %} (Hindi written in Latin script mixed with English finance
terms){% endif %}. Return JSON matching the schema.""",

    "research_plan": """\
You plan today's WorldQuant BRAIN research for {{ scope }}.
Research focus requested: {{ focus }}.
Strong alphas so far by dataset theme: {{ strong_by_theme }}
Candidate datasets (id · theme · AI score · single/double/interaction · alphas by others · my simulations · pyramid ×):
{{ datasets }}
Choose ONE research task: kind single (one dataset, single fields), pair (two datasets combined) or interaction (field
interactions inside one dataset). Prefer under-explored datasets and themes where uncorrelated alphas are likely.
Return JSON matching the schema.""",
}

NAMES = list(DEFAULTS)


async def get_template(name: str) -> str:
    override = await db.kv_get(f"prompt:{name}")
    return override or DEFAULTS[name]


async def set_template(name: str, text: str | None) -> None:
    await db.kv_set(f"prompt:{name}", text or None)


async def render(name: str, **variables: object) -> str:
    template = _env.from_string(await get_template(name))
    return template.render(**variables)
