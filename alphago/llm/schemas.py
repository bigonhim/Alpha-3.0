"""Structured-output schemas (kept flat and simple: each provider supports a different JSON-schema subset)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AlphaIdea(BaseModel):
    expression: str = Field(description="A complete, valid FASTEXPR expression")
    idea: str = Field(description="One-sentence economic hypothesis")
    mechanism: str = Field(description="momentum | reversal | value | quality | growth | revisions | sentiment | "
                                       "volatility | liquidity | event | other")
    data_rationale: str = Field(description="Why these data fields carry the signal")
    operator_rationale: str = Field(description="Why these operators / windows / groups")
    neutralization: str = Field(default="", description="Suggested neutralization, e.g. SUBINDUSTRY")
    decay: int = Field(default=0, description="Suggested decay (0–20)")
    truncation: float = Field(default=0.08, description="Suggested truncation (0.01–0.1)")


class IdeaBatch(BaseModel):
    ideas: list[AlphaIdea]


class IdeaReview(BaseModel):
    index: int = Field(description="Index of the idea in the list")
    keep: bool = Field(default=True, description="false only for a serious problem")
    issues: list[str] = Field(default_factory=list, description="overfit | duplicate | field_type | lookahead")
    reason: str = Field(default="", description="One short sentence")


class BatchCritique(BaseModel):
    reviews: list[IdeaReview]


class FieldNote(BaseModel):
    field: str
    meaning: str
    usage: str = Field(description="How to use it in an alpha (normalisation, windows, pitfalls)")


class IdeaFamily(BaseModel):
    name: str
    hypothesis: str
    template: str = Field(description="FASTEXPR template; may use {window} and {group} placeholders")


class DatasetBrief(BaseModel):
    summary: str
    themes: list[str]
    key_fields: list[FieldNote]
    idea_families: list[IdeaFamily]
    suggested_neutralization: str = ""
    suggested_decay: int = 0
    pitfalls: list[str] = Field(default_factory=list)


class AlphaDescription(BaseModel):
    idea: str = Field(description="Idea: the economic hypothesis in 1–2 sentences")
    rationale_data: str = Field(description="Rationale for data used")
    rationale_operators: str = Field(description="Rationale for operators used")

    def render(self) -> str:
        return (f"Idea: {self.idea.strip()}\nRationale for data used: {self.rationale_data.strip()}\n"
                f"Rationale for operators used: {self.rationale_operators.strip()}")


class GeniusSignal(BaseModel):
    title: str
    why: str
    action: str
    region: str = ""
    delay: int = 1
    category: str = ""
    target_count: int = 1
    constraint: str = Field(default="", description="e.g. 'max 4 operators', 'single dataset'")


class GeniusPlan(BaseModel):
    summary: str
    signals: list[GeniusSignal]


class SuperAlphaDescriptions(BaseModel):
    selection_description: str
    combo_description: str


THEMES = ["Sentiment", "Analyst Data", "Momentum", "Risk", "Alternative Data", "Fundamental", "Liquidity",
          "Valuation", "Leverage", "Growth", "Earnings", "Macro", "Price Volume", "Other"]


class DatasetRating(BaseModel):
    dataset_id: str
    ai_score: int = Field(description="0–100: overall promise for new, uncorrelated, submittable alphas")
    single: int = Field(description="0–100: suitability for single-field alphas (one field, simple operators)")
    double: int = Field(description="0–100: suitability for pairing with ANOTHER dataset")
    interaction: int = Field(description="0–100: suitability for field × field interactions inside the dataset")
    recommended: str = Field(description="single | double | interaction")
    theme: str = Field(description="One of: " + ", ".join(THEMES))
    reason: str = Field(description="One short sentence")


class DatasetRatingBatch(BaseModel):
    ratings: list[DatasetRating]


class OsmosisPrediction(BaseModel):
    predicted_low: float = Field(description="Low end of the predicted next Osmosis score")
    predicted_high: float = Field(description="High end of the predicted next Osmosis score")
    direction: str = Field(description="up | flat | down (versus the current score)")
    summary: str
    drivers: list[str] = Field(description="2–5 short reasons, most important first")


class OsmosisMistake(BaseModel):
    scope: str
    issue: str
    fix: str


class OsmosisExplanation(BaseModel):
    summary: str
    mistakes: list[OsmosisMistake]
    next_steps: list[str]


class ResearchPlan(BaseModel):
    focus: str = Field(description="Dataset theme to prioritise")
    reason: str = Field(description="Why, in one or two sentences")
    kind: str = Field(description="single | pair | interaction")
    dataset_ids: list[str] = Field(description="1 dataset for single/interaction, 2 for pair, from the list given")
