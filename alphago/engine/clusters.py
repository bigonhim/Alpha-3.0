"""Economic-idea clusters: group alphas by mechanism (momentum, reversal, value, ...) rather than formula shape.

The per-shape cap (skeleton) still lets many differently-written alphas share one idea; a daily cap per cluster
spreads the quota over ideas. Classification is a rule of thumb: the AI's own stated mechanism when it gave one,
else the dataset category, else the expression's form (price/volume vs fundamentals, sign, horizon).
"""

from __future__ import annotations

import re
from typing import Iterable

CLUSTERS = ["momentum", "reversal", "value", "quality", "growth", "analyst", "sentiment", "volatility", "liquidity",
            "event", "other"]
LABELS = {"momentum": "Momentum", "reversal": "Reversal", "value": "Value", "quality": "Quality / profitability",
          "growth": "Growth", "analyst": "Analyst revisions", "sentiment": "Sentiment / news",
          "volatility": "Volatility / risk", "liquidity": "Liquidity", "event": "Event-driven", "other": "Other"}

# checked in order: the first keyword found wins
_MECHANISM_WORDS: list[tuple[str, tuple[str, ...]]] = [
    ("analyst", ("analyst", "revision", "consensus", "recommendation", "estimate dispersion")),
    ("sentiment", ("sentiment", "news", "social", "tone", "buzz")),
    ("event", ("earnings announcement", "surprise", "event")),
    ("reversal", ("reversal", "mean reversion", "mean-reversion", "overreact", "contrarian")),
    ("momentum", ("momentum", "trend", "continuation")),
    ("value", ("value", "cheap", "valuation", "earnings yield", "book-to", "book to")),
    ("quality", ("quality", "profitab", "margin", "return on", "roe", "roa", "accrual")),
    ("growth", ("growth", "expansion", "acceleration")),
    ("volatility", ("volatility", "risk", "beta", "idiosyncratic", "dispersion")),
    ("liquidity", ("liquidity", "illiquid", "volume", "amihud", "turnover")),
]
_CATEGORY_WORDS: list[tuple[str, tuple[str, ...]]] = [
    ("analyst", ("analyst", "estimate")),
    ("sentiment", ("sentiment", "news", "social")),
    ("volatility", ("option", "volatility", "risk")),
    ("event", ("event", "earnings")),
]
PV_FIELDS = {"close", "open", "high", "low", "vwap", "volume", "returns", "adv20", "cap", "sharesout"}
_QUALITY_FIELD = re.compile(r"income|margin|roe|roa|profit|ebit|assets|equity|cashflow|accrual", re.I)
_VALUE_FIELD = re.compile(r"enterprise_value|book|price|cap\b|market_value|yield", re.I)
_VOLATILITY_OPS = {"ts_std_dev", "ts_corr", "ts_covariance", "ts_kurtosis", "ts_skewness"}


def _keyword(text: str | None, table: list[tuple[str, tuple[str, ...]]]) -> str | None:
    t = (text or "").lower()
    if not t:
        return None
    for cluster, words in table:
        if any(w in t for w in words):
            return cluster
    return None


def _negative(source: str) -> bool:
    s = source.replace(" ", "")
    last = s.rsplit(";", 1)[-1]
    return last.startswith("-") or last.startswith("reverse(") or "-ts_delta(" in last or "-returns" in last \
        or "-rank(ts_delta" in last


def classify(source: str, *, fields: Iterable[str] = (), operators: Iterable[str] = (),
             windows: Iterable[int] = (), category: str | None = None, mechanism: str | None = None) -> str:
    """Economic idea of an alpha (one of CLUSTERS)."""
    by_mechanism = _keyword(mechanism, _MECHANISM_WORDS)
    if by_mechanism:
        return by_mechanism
    by_category = _keyword(category, _CATEGORY_WORDS)
    if by_category:
        return by_category
    flds = {f.lower() for f in fields}
    ops = set(operators)
    wins = [int(w) for w in windows if w]
    longest = max(wins) if wins else 0
    data = flds - PV_FIELDS
    if not data:                                   # price / volume only
        if ops & _VOLATILITY_OPS and "ts_delta" not in ops:
            return "volatility"
        if flds and flds <= {"volume", "adv20"} | {"cap"}:
            return "liquidity"
        if _negative(source) or (longest and longest <= 10):
            return "reversal"
        return "momentum" if longest >= 60 else "reversal"
    if "ts_delta" in ops or "ts_av_diff" in ops or "ts_regression" in ops:
        return "growth"
    if any(_QUALITY_FIELD.search(f) for f in data):
        return "quality"
    if any(_VALUE_FIELD.search(f) for f in data) or "cap" in flds:
        return "value"
    if "fundamental" in (category or "").lower():
        return "value"
    return "other"


def classify_program(source: str, program, *, category: str | None = None,  # noqa: ANN001
                     mechanism: str | None = None) -> str:
    from ..fastexpr import compute_metrics
    m = compute_metrics(program)
    return classify(source, fields=m.non_group_fields, operators=m.unique_operators, windows=m.windows,
                    category=category, mechanism=mechanism)
