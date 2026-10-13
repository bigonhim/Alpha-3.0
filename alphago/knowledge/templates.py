"""Curated FASTEXPR template library (from WorldQuant examples, machine_lib factories and community playbooks).

Placeholders: {f} primary field, {f2} second field, {d} window, {g} group. Fields are wrapped by `wrap_field` so
sparse fundamentals are back-filled and outliers winsorised before use.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

WINDOWS = [5, 22, 66, 126, 252]
GROUPS = ["subindustry", "industry", "sector", "market"]
BUCKET_GROUPS = {
    "cap_bucket": 'bucket(rank(cap), range="0.1,1,0.1")',
    "vol_bucket": 'bucket(rank(ts_std_dev(returns, 20)), range="0.1,1,0.1")',
    "liq_bucket": 'bucket(rank(close*volume), range="0.1,1,0.1")',
}


@dataclass(frozen=True)
class Template:
    family: str
    text: str
    needs_f2: bool = False
    note: str = ""


TEMPLATES: list[Template] = [
    # --- first order (time-series transforms of one field)
    Template("ts_rank", "ts_rank({f}, {d})"),
    Template("ts_zscore", "ts_zscore({f}, {d})"),
    Template("ts_delta", "-ts_delta({f}, {d})", note="reversal of changes"),
    Template("ts_av_diff", "ts_av_diff({f}, {d})"),
    Template("ts_scale", "ts_scale({f}, {d})"),
    Template("quantile_rank", "quantile(ts_rank({f}, {d}))"),
    Template("decay_exp", "ts_decay_exp_window({f}, {d}, factor=0.5)"),
    # --- second order (group-relative)
    Template("group_rank_ts", "group_rank(ts_rank({f}, {d}), {g})"),
    Template("group_neut_zscore", "group_neutralize(ts_zscore({f}, {d}), {g})"),
    Template("group_zscore_level", "group_zscore({f}, {g})"),
    Template("group_relative", "data = divide({f}, group_median({f}, {g})); group_neutralize(ts_rank(data, {d}), {g})"),
    Template("group_std_scaled", "data = ts_decay_exp_window({f} / group_std_dev({f}, {g}), {d}, factor=0.5); "
                                 "group_neutralize(data, {g})"),
    # --- value / quality style ratios
    Template("ratio_to_cap", "group_rank(ts_backfill({f}, 120) / cap, {g})", note="value-style scaling by size"),
    Template("ratio_rank", "group_rank(ts_rank({f} / {f2}, {d}), {g})", needs_f2=True),
    Template("ratio_delta", "rank(ts_delta({f} / {f2}, {d}))", needs_f2=True),
    Template("spread_zscore", "ts_zscore({f} - {f2}, {d})", needs_f2=True),
    Template("corr_pair", "-ts_corr(rank({f}), rank({f2}), {d})", needs_f2=True),
    # --- momentum divergence / trend
    Template("momentum_div", "ts_delta(ts_zscore({f}, 22), 22) - ts_delta(ts_zscore({f}, 126), 126)"),
    Template("trend_slope", "ts_regression({f}, ts_step(1), {d}, lag=0, rettype=2)"),
    # --- event-gated (turnover control)
    Template("vol_event", "trade_when(ts_mean(volume, 10) > ts_mean(volume, 60), group_rank(ts_rank({f}, {d}), {g}), -1)"),
    Template("volatility_event", "trade_when(ts_std_dev(returns, 5) > ts_std_dev(returns, 20), "
                                 "group_neutralize(ts_zscore({f}, {d}), {g}), -1)"),
]

# trade_when open events (machine_lib "third order" factory)
OPEN_EVENTS = [
    "ts_arg_max(volume, 5) == 0",
    "ts_corr(close, volume, 20) < 0",
    "ts_mean(volume, 10) > ts_mean(volume, 60)",
    "group_rank(ts_std_dev(returns, 60), sector) > 0.7",
    "ts_zscore(returns, 60) > 2",
    "ts_std_dev(returns, 5) > ts_std_dev(returns, 20)",
    "ts_arg_max(close, 20) == 0",
    "ts_corr(close, volume, 5) > 0.3",
]
EXIT_EVENTS = ["-1", "abs(returns) > 0.1"]

PRICE_VOLUME_SEEDS = [
    "rank(-ts_delta(close, 5))",
    "(vwap - close) / vwap",
    "-ts_corr(rank(open), rank(volume), 10)",
    "sign(ts_delta(volume, 1)) * -ts_delta(close, 1)",
    "rank(vwap - close) / rank(vwap + close)",
    "(close - open) / ((high - low) + 0.001)",
    "event = volume > adv20; alpha = -ts_delta(close, 5); trade_when(event, alpha, -1)",
    "-rank(ts_std_dev(returns, 22))",
    "rank(ts_delay(ts_sum(returns, 231), 21))",
]

FUNDAMENTAL_SEEDS = [
    "group_rank(ts_rank(operating_income / equity, 126), subindustry)",
    "-ts_zscore(enterprise_value / ebitda, 63)",
    "rank(ts_delta(retained_earnings / sharesout, 90))",
    "zscore(cash_st / debt_st)",
    "ts_rank(operating_income, 252)",
    "-rank(ebit / capex)",
    "alpha = ts_rank(cashflow_op / cap, 60); group_rank(alpha, subindustry)",
]

# starting settings by data category (decay, neutralization, truncation, nanHandling)
CATEGORY_SETTINGS = {
    "fundamental": (0, "SUBINDUSTRY", 0.08, "ON"),
    "analyst": (2, "INDUSTRY", 0.08, "ON"),
    "model": (2, "SUBINDUSTRY", 0.08, "ON"),
    "news": (6, "INDUSTRY", 0.06, "ON"),
    "sentiment": (6, "INDUSTRY", 0.06, "ON"),
    "socialmedia": (6, "INDUSTRY", 0.06, "ON"),
    "option": (4, "INDUSTRY", 0.08, "ON"),
    "pv": (10, "INDUSTRY", 0.08, "OFF"),
    "price volume": (10, "INDUSTRY", 0.08, "OFF"),
}


def settings_for_category(category: str | None) -> tuple[int, str, float, str]:
    c = (category or "").lower()
    for key, value in CATEGORY_SETTINGS.items():
        if key in c:
            return value
    return (4, "SUBINDUSTRY", 0.08, "ON")


def wrap_field(field_id: str, field_type: str = "MATRIX", backfill: int = 120) -> str:
    """Pre-wrap a raw field: vector fields are reduced; sparse data back-filled; outliers winsorised."""
    base = f"vec_avg({field_id})" if field_type == "VECTOR" else field_id
    return f"winsorize(ts_backfill({base}, {backfill}), std=4)"


def instantiate(t: Template, f: str, f2: str | None = None, d: int = 66, g: str = "subindustry") -> str:
    return t.text.format(f=f, f2=f2 or "cap", d=d, g=g)


def factory(fields: list[tuple[str, str]], *, n: int, rng: random.Random | None = None,
            families: list[str] | None = None) -> list[tuple[str, str]]:
    """Random sample of (expression, family) from templates × fields × windows × groups."""
    rng = rng or random.Random()
    if not fields:
        return []
    pool = [t for t in TEMPLATES if not families or t.family in families]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        t = rng.choice(pool)
        fid, ftype = rng.choice(fields)
        f = wrap_field(fid, ftype)
        f2 = None
        if t.needs_f2:
            others = [x for x in fields if x[0] != fid]
            if not others:
                continue
            fid2, ftype2 = rng.choice(others)
            f2 = wrap_field(fid2, ftype2)
        expr = instantiate(t, f, f2, rng.choice(WINDOWS[1:]), rng.choice(GROUPS[:3]))
        if expr not in seen:
            seen.add(expr)
            out.append((expr, t.family))
    return out


# ---------------------------------------------------------------- deterministic enumeration (research tasks)

# extra two-field templates for pairing fields from two datasets (or two fields of one dataset)
PAIR_TEMPLATES: list[Template] = [
    Template("pair_rank_diff", "ts_rank({f}, {d}) - ts_rank({f2}, {d})", needs_f2=True),
    Template("pair_residual", "group_rank(ts_regression({f}, {f2}, {d}), {g})", needs_f2=True,
             note="part of f not explained by f2"),
    Template("pair_gate", "trade_when(ts_rank({f2}, {d}) > 0.5, group_rank(ts_rank({f}, {d}), {g}), -1)",
             needs_f2=True, note="trade f only when f2 is high"),
    Template("pair_product", "group_rank(rank({f}) * rank({f2}), {g})", needs_f2=True),
]
PV_PARTNERS = [("returns", "MATRIX"), ("volume", "MATRIX"), ("cap", "MATRIX"), ("vwap", "MATRIX")]
ENUM_WINDOWS = (66, 126, 22, 252)
ENUM_GROUPS = ("subindustry", "industry", "sector")


def _first_order() -> list[Template]:
    return [t for t in TEMPLATES if not t.needs_f2 and "{g}" not in t.text]


def _group_order() -> list[Template]:
    return [t for t in TEMPLATES if not t.needs_f2 and "{g}" in t.text]


def _pair_order() -> list[Template]:
    return [t for t in TEMPLATES if t.needs_f2] + PAIR_TEMPLATES


def enumerate_single(fields: list[tuple[str, str]], windows=ENUM_WINDOWS):  # noqa: ANN001, ANN201
    """First-order expressions over every field: window-major, so a small budget still covers every field."""
    seen: set[str] = set()
    for d in windows:
        for t in _first_order():
            for fid, ftype in fields:
                expr = instantiate(t, wrap_field(fid, ftype), d=d)
                if expr not in seen:
                    seen.add(expr)
                    yield expr, t.family


def enumerate_group(fields: list[tuple[str, str]], windows=(66, 126, 252), groups=ENUM_GROUPS):  # noqa: ANN001, ANN201
    seen: set[str] = set()
    for g in groups:
        for d in windows:
            for t in _group_order():
                for fid, ftype in fields:
                    expr = instantiate(t, wrap_field(fid, ftype), d=d, g=g)
                    if expr not in seen:
                        seen.add(expr)
                        yield expr, t.family


def enumerate_pairs(fields_a: list[tuple[str, str]], fields_b: list[tuple[str, str]], windows=ENUM_WINDOWS,  # noqa: ANN001
                    groups=("subindustry", "industry")):  # noqa: ANN201
    """f from A × f2 from B over every two-field template."""
    seen: set[str] = set()
    for d in windows:
        for g in groups:
            for t in _pair_order():
                if "{g}" not in t.text and g != groups[0]:
                    continue
                for fa, ta in fields_a:
                    for fb, tb in fields_b:
                        if fa == fb:
                            continue
                        expr = instantiate(t, wrap_field(fa, ta), wrap_field(fb, tb), d=d, g=g)
                        if expr not in seen:
                            seen.add(expr)
                            yield expr, t.family


def second_order(expr: str, groups=ENUM_GROUPS, rng: random.Random | None = None) -> list[tuple[str, str]]:  # noqa: ANN001
    """Group-relative and event-gated variants of a working first-order alpha."""
    out = []
    for g in groups:
        out.append((f"group_neutralize({expr}, {g})", "group_neutralize"))
        out.append((f"group_rank({expr}, {g})", "group_rank"))
    out += [(e, "trade_when gate") for e in third_order(expr, rng, k=2)]
    return out


def third_order(alpha: str, rng: random.Random | None = None, k: int = 3) -> list[str]:
    rng = rng or random.Random()
    combos = list(itertools.product(OPEN_EVENTS, EXIT_EVENTS))
    rng.shuffle(combos)
    return [f"trade_when({o}, {alpha}, {e})" for o, e in combos[:k]]


def template_catalog_text(limit: int = 40) -> str:
    lines = [f"- [{t.family}] {t.text}" + (f"  ({t.note})" if t.note else "") for t in TEMPLATES[:limit]]
    return "\n".join(lines)
