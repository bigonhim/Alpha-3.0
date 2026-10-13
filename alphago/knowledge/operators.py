"""Curated FASTEXPR operator table and catalog construction from BRAIN API data.

Signatures follow BRAIN documentation and community re-implementations.  Tier ``"B"``
operators are available to every account; tier ``"C"`` ones are usually
consultant-only.  When the real operator list is loaded from ``GET /operators``,
:func:`build_catalog` marks what the account can actually use.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import replace

from alphago.fastexpr.catalog import INFIX_TO_OPERATOR, ArgSpec, Catalog, FieldInfo, OperatorSpec

__all__ = [
    "CURATED_OPERATORS",
    "GROUP_ARG_OPERATORS",
    "INFIX_TO_OPERATOR",
    "VECTOR_OPERATORS",
    "build_catalog",
    "field_info_from_api",
]

ARITHMETIC = "Arithmetic"
LOGICAL = "Logical"
TIME_SERIES = "Time Series"
CROSS_SECTIONAL = "Cross Sectional"
VECTOR = "Vector"
TRANSFORMATIONAL = "Transformational"
GROUP = "Group"
SPECIAL = "Special"


# --------------------------------------------------------------------------- spec helpers


def _x(name: str = "x") -> ArgSpec:
    return ArgSpec(name, "expr")


def _d(name: str = "d") -> ArgSpec:
    return ArgSpec(name, "int")


def _g(name: str = "group") -> ArgSpec:
    return ArgSpec(name, "group")


def _v(name: str = "x") -> ArgSpec:
    return ArgSpec(name, "vector")


def _num(name: str, default: float) -> ArgSpec:
    return ArgSpec(name, "number", required=False, default=default)


def _bool(name: str, default: bool = False) -> ArgSpec:
    return ArgSpec(name, "bool", required=False, default=default)


def _str(name: str, default: str) -> ArgSpec:
    return ArgSpec(name, "string", required=False, default=default)


def _enum(name: str, default: str, choices: tuple[str, ...]) -> ArgSpec:
    return ArgSpec(name, "enum", required=False, default=default, choices=choices)


def _op(
    name: str,
    category: str,
    positional: Iterable[ArgSpec],
    keywords: Iterable[ArgSpec] = (),
    *,
    desc: str,
    tier: str = "B",
    variadic: bool = False,
    min_args: int | None = None,
    returns: str = "matrix",
    permissive: bool = False,
) -> OperatorSpec:
    spec = OperatorSpec(
        name=name,
        category=category,
        positional=tuple(positional),
        keywords=tuple(keywords),
        variadic=variadic,
        min_args=min_args,
        returns=returns,
        description=desc,
        tier=tier,
        permissive=permissive,
    )
    return replace(spec, definition=spec.signature())


def _unary(name: str, category: str, desc: str, tier: str = "B", returns: str = "matrix") -> OperatorSpec:
    return _op(name, category, [_x()], desc=desc, tier=tier, returns=returns)


def _ts(name: str, desc: str, *extra_kw: ArgSpec, tier: str = "B") -> OperatorSpec:
    """Time-series operator ``name(x, d, ...)``."""
    return _op(name, TIME_SERIES, [_x(), _d()], extra_kw, desc=desc, tier=tier)


def _group(name: str, desc: str, *extra_kw: ArgSpec, tier: str = "B") -> OperatorSpec:
    """Group operator ``name(x, group, ...)``."""
    return _op(name, GROUP, [_x(), _g()], extra_kw, desc=desc, tier=tier)


def _cmp(name: str, symbol: str) -> OperatorSpec:
    return _op(name, LOGICAL, [_x(), _x("y")], desc=f"x {symbol} y, element-wise.", returns="boolean")


_DRIVERS = ("gaussian", "uniform", "cauchy")

_SPECS: list[OperatorSpec] = [
    # ----------------------------------------------------------------- arithmetic
    _unary("abs", ARITHMETIC, "Absolute value of x."),
    _op("add", ARITHMETIC, [_x(), _x("y")], [_bool("filter")], variadic=True,
        desc="Sum of all inputs; filter=true treats NaN as 0."),
    _op("subtract", ARITHMETIC, [_x(), _x("y")], [_bool("filter")],
        desc="x - y; filter=true treats NaN as 0."),
    _op("multiply", ARITHMETIC, [_x(), _x("y")], [_bool("filter")], variadic=True,
        desc="Product of all inputs; filter=true treats NaN as 1."),
    _op("divide", ARITHMETIC, [_x(), _x("y")], desc="x / y."),
    _unary("inverse", ARITHMETIC, "1 / x."),
    _unary("reverse", ARITHMETIC, "-x."),
    _unary("log", ARITHMETIC, "Natural logarithm of x."),
    _unary("sqrt", ARITHMETIC, "Square root of x."),
    _op("power", ARITHMETIC, [_x(), _x("y")], desc="x raised to the power y."),
    _op("signed_power", ARITHMETIC, [_x(), _x("y")], desc="sign(x) * abs(x)^y; preserves the sign of x."),
    _unary("sign", ARITHMETIC, "Sign of x: -1, 0 or 1 (NaN stays NaN)."),
    _op("max", ARITHMETIC, [_x(), _x("y")], variadic=True, desc="Element-wise maximum of the inputs."),
    _op("min", ARITHMETIC, [_x(), _x("y")], variadic=True, desc="Element-wise minimum of the inputs."),
    _op("densify", ARITHMETIC, [_g("x")], returns="group",
        desc="Re-index a grouping field so only populated buckets remain."),
    _unary("exp", ARITHMETIC, "Natural exponential of x."),
    _unary("s_log_1p", ARITHMETIC, "sign(x) * log(1 + abs(x))."),
    _op("to_nan", ARITHMETIC, [_x()], [_num("value", 0), _bool("reverse")], tier="C",
        desc="Convert value to NaN (or everything except value when reverse=true)."),
    _op("nan_mask", ARITHMETIC, [_x(), _x("y")], tier="C", desc="NaN where y < 0, otherwise x."),
    _op("nan_out", ARITHMETIC, [_x()], [_num("lower", 0), _num("upper", 0)], tier="C",
        desc="Set values outside [lower, upper] to NaN."),
    _unary("purify", ARITHMETIC, "Replace infinities with NaN.", tier="C"),
    _unary("pasteurize", ARITHMETIC, "Set to NaN if x is infinite or the instrument is outside the universe."),
    _unary("floor", ARITHMETIC, "Largest integer <= x.", tier="C"),
    _unary("ceiling", ARITHMETIC, "Smallest integer >= x.", tier="C"),
    _unary("round", ARITHMETIC, "x rounded to the nearest integer.", tier="C"),
    _op("round_down", ARITHMETIC, [_x()], [_num("f", 1)], tier="C",
        desc="Round x down to the nearest multiple of f."),
    _unary("fraction", ARITHMETIC, "Fractional part of x (sign preserved).", tier="C"),
    _unary("log_diff", ARITHMETIC, "log(x) - log(ts_delay(x, 1)).", tier="C"),
    _unary("arc_sin", ARITHMETIC, "Arcsine of x.", tier="C"),
    _unary("arc_cos", ARITHMETIC, "Arccosine of x.", tier="C"),
    _unary("arc_tan", ARITHMETIC, "Arctangent of x.", tier="C"),
    _unary("tanh", ARITHMETIC, "Hyperbolic tangent of x.", tier="C"),
    _unary("sigmoid", ARITHMETIC, "1 / (1 + exp(-x)).", tier="C"),
    # ----------------------------------------------------------------- logical
    _op("if_else", LOGICAL, [_x("input1"), _x("input2"), _x("input3")],
        desc="input2 where input1 is true, otherwise input3 (same as c ? a : b)."),
    _op("and", LOGICAL, [_x(), _x("y")], returns="boolean", desc="Logical AND (x && y)."),
    _op("or", LOGICAL, [_x(), _x("y")], returns="boolean", desc="Logical OR (x || y)."),
    _unary("not", LOGICAL, "Logical negation (!x).", returns="boolean"),
    _unary("is_nan", LOGICAL, "1 where x is NaN, else 0.", returns="boolean"),
    _cmp("less", "<"),
    _cmp("less_equal", "<="),
    _cmp("greater", ">"),
    _cmp("greater_equal", ">="),
    _cmp("equal", "=="),
    _cmp("not_equal", "!="),
    # ----------------------------------------------------------------- time series
    _ts("ts_delay", "Value of x d days ago."),
    _ts("ts_delta", "x - ts_delay(x, d)."),
    _ts("ts_mean", "Average of x over the past d days."),
    _ts("ts_sum", "Sum of x over the past d days."),
    _ts("ts_product", "Product of x over the past d days."),
    _ts("ts_std_dev", "Standard deviation of x over the past d days."),
    _ts("ts_zscore", "(x - ts_mean(x, d)) / ts_std_dev(x, d)."),
    _ts("ts_rank", "Rank of today's x among the past d days, plus constant.", _num("constant", 0)),
    _ts("ts_scale", "(x - ts_min) / (ts_max - ts_min) over d days, plus constant.", _num("constant", 0)),
    _ts("ts_quantile", "ts_rank of x mapped through the inverse CDF of driver.",
        _enum("driver", "gaussian", _DRIVERS)),
    _ts("ts_arg_max", "Days since the maximum of x within the past d days."),
    _ts("ts_arg_min", "Days since the minimum of x within the past d days."),
    _ts("ts_min", "Minimum of x over the past d days."),
    _ts("ts_max", "Maximum of x over the past d days."),
    _ts("ts_median", "Median of x over the past d days."),
    _ts("ts_av_diff", "x - ts_mean(x, d), ignoring NaNs."),
    _op("ts_corr", TIME_SERIES, [_x(), _x("y"), _d()], desc="Pearson correlation of x and y over d days."),
    _op("ts_covariance", TIME_SERIES, [_x("y"), _x(), _d()], desc="Covariance of y and x over d days."),
    _ts("ts_decay_linear", "Linearly decayed average of x over d days.", _bool("dense")),
    _ts("ts_decay_exp_window", "Exponentially decayed average of x over d days.", _num("factor", 0.5)),
    _op("ts_backfill", TIME_SERIES, [_x(), _d("lookback")], [_num("k", 1), _str("ignore", "NAN")],
        desc="Replace NaN with the k-th most recent valid value within lookback days."),
    _ts("ts_count_nans", "Number of NaN values of x in the past d days."),
    _op("ts_regression", TIME_SERIES, [_x("y"), _x(), _d()], [_num("lag", 0), _num("rettype", 0)],
        desc="Rolling OLS of y on x over d days; rettype selects error/intercept/slope/... ."),
    _op("ts_step", TIME_SERIES, [_d("n")], desc="Day counter (1, 2, 3, ...) used as a time regressor."),
    _ts("ts_ir", "ts_mean(x, d) / ts_std_dev(x, d)."),
    _ts("ts_skewness", "Skewness of x over the past d days."),
    _ts("ts_kurtosis", "Kurtosis of x over the past d days."),
    _ts("ts_returns", "Relative change of x over d days (mode=1: x/delay - 1, mode=2: log).",
        _num("mode", 1)),
    _ts("ts_max_diff", "x - ts_max(x, d)."),
    _ts("ts_min_diff", "x - ts_min(x, d)."),
    _ts("ts_min_max_diff", "x - f * (ts_min(x, d) + ts_max(x, d)).", _num("f", 0.5)),
    _ts("ts_min_max_cps", "(ts_min(x, d) + ts_max(x, d)) - f * x.", _num("f", 2)),
    _ts("ts_percentage", "Percentile value of x over the past d days.", _num("percentage", 0.5)),
    _op("ts_weighted_decay", TIME_SERIES, [_x()], [_num("k", 0.5)],
        desc="Weighted decay: k * today + (1 - k) * yesterday."),
    _unary("days_from_last_change", TIME_SERIES, "Days since x last changed value."),
    _ts("last_diff_value", "Last value of x within d days that differs from today's value."),
    _ts("kth_element", "k-th most recent valid value of x within d days.",
        _num("k", 1), _str("ignore", "NaN")),
    _op("hump", TIME_SERIES, [_x()], [_num("hump", 0.01)],
        desc="Limit day-to-day change of x to reduce turnover."),
    _ts("ts_moment", "k-th central moment of x over d days.", _num("k", 0), tier="C"),
    _ts("ts_entropy", "Entropy of x over d days using buckets bins.", _num("buckets", 10), tier="C"),
    _ts("jump_decay", "Damp sudden jumps of x relative to its d-day std.",
        _num("sensitivity", 0.5), _num("force", 0.1), tier="C"),
    _op("ts_target_tvr_decay", TIME_SERIES, [_x()],
        [_num("lambda_min", 0), _num("lambda_max", 1), _num("target_tvr", 0.1)], tier="C",
        desc="Decay x to reach a target turnover."),
    _op("ts_target_tvr_hump", TIME_SERIES, [_x()],
        [_num("lambda_min", 0), _num("lambda_max", 1), _num("target_tvr", 0.1)], tier="C",
        desc="Hump x to reach a target turnover."),
    _op("ts_target_tvr_delta_limit", TIME_SERIES, [_x(), _x("y")],
        [_num("lambda_min", 0), _num("lambda_max", 1), _num("target_tvr", 0.1)], tier="C",
        desc="Delta-limit x (scaled by y) to reach a target turnover."),
    _op("ts_delta_limit", TIME_SERIES, [_x(), _x("y")], [_num("limit_volume", 0.1)], tier="C",
        desc="Limit the daily change of x to limit_volume * y."),
    _op("ts_partial_corr", TIME_SERIES, [_x(), _x("y"), _x("z"), _d()], tier="C",
        desc="Partial correlation of x and y controlling for z over d days."),
    _op("ts_triple_corr", TIME_SERIES, [_x(), _x("y"), _x("z"), _d()], tier="C",
        desc="Triple correlation of x, y and z over d days."),
    _op("ts_co_skewness", TIME_SERIES, [_x("y"), _x(), _d()], tier="C", desc="Co-skewness of y and x over d days."),
    _op("ts_co_kurtosis", TIME_SERIES, [_x("y"), _x(), _d()], tier="C", desc="Co-kurtosis of y and x over d days."),
    _op("ts_theilsen", TIME_SERIES, [_x(), _x("y"), _d()], tier="C",
        desc="Theil-Sen robust slope of x on y over d days."),
    _op("ts_poly_regression", TIME_SERIES, [_x("y"), _x(), _d()], [_num("k", 1)], tier="C",
        desc="Residual of a degree-k polynomial regression of y on x over d days."),
    _op("ts_vector_neut", TIME_SERIES, [_x(), _x("y"), _d()], tier="C",
        desc="x neutralised against y over the past d days."),
    _op("ts_vector_proj", TIME_SERIES, [_x(), _x("y"), _d()], tier="C",
        desc="Projection of x onto y over the past d days."),
    _ts("inst_tvr", "Instrument turnover of x over d days.", tier="C"),
    _op("hump_decay", TIME_SERIES, [_x()], [_num("p", 0)], tier="C",
        desc="Ignore changes of x smaller than p relative to yesterday."),
    # ----------------------------------------------------------------- cross sectional
    _op("rank", CROSS_SECTIONAL, [_x()], [_num("rate", 2)], desc="Cross-sectional rank of x in [0, 1]."),
    _unary("zscore", CROSS_SECTIONAL, "Cross-sectional z-score of x."),
    _op("scale", CROSS_SECTIONAL, [_x()], [_num("scale", 1), _num("longscale", 1), _num("shortscale", 1)],
        desc="Scale x so that sum(abs(x)) equals scale (separately for long/short sides)."),
    _op("normalize", CROSS_SECTIONAL, [_x()], [_bool("useStd"), _num("limit", 0.0)],
        desc="x minus its cross-sectional mean (divided by std when useStd=true), clipped to limit."),
    _op("quantile", CROSS_SECTIONAL, [_x()], [_enum("driver", "gaussian", _DRIVERS), _num("sigma", 1.0)],
        desc="Rank x and map through the inverse CDF of driver."),
    _op("winsorize", CROSS_SECTIONAL, [_x()], [_num("std", 4)], desc="Clip x to mean +/- std standard deviations."),
    _op("truncate", CROSS_SECTIONAL, [_x()], [_num("maxPercent", 0.01)],
        desc="Cap each weight at maxPercent of the book."),
    _op("scale_down", CROSS_SECTIONAL, [_x()], [_num("constant", 0)],
        desc="Scale x to [0, 1] by its cross-sectional min/max, minus constant."),
    _op("vector_neut", CROSS_SECTIONAL, [_x(), _x("y")], desc="x neutralised against y (orthogonal residual)."),
    _op("vector_proj", CROSS_SECTIONAL, [_x(), _x("y")], desc="Projection of x onto y."),
    _op("regression_neut", CROSS_SECTIONAL, [_x("y"), _x()], desc="Residual of the cross-sectional OLS of y on x."),
    _op("regression_proj", CROSS_SECTIONAL, [_x("y"), _x()], desc="Fitted value of the cross-sectional OLS of y on x."),
    _op("rank_by_side", CROSS_SECTIONAL, [_x()], [_num("rate", 2), _num("scale", 1)], tier="C",
        desc="Rank positive and negative values separately."),
    _op("generalized_rank", CROSS_SECTIONAL, [_x()], [_num("m", 1)], tier="C",
        desc="Rank based on pairwise differences raised to the power m."),
    _op("one_side", CROSS_SECTIONAL, [_x()], [_enum("side", "long", ("long", "short"))], tier="C",
        desc="Keep only the long (or short) side of x."),
    _op("rank_gmean_amean_diff", CROSS_SECTIONAL, [_x(), _x("y")], variadic=True, tier="C",
        desc="Geometric mean minus arithmetic mean of the ranks of the inputs."),
    # ----------------------------------------------------------------- vector
    _op("vec_avg", VECTOR, [_v()], desc="Mean of the vector field's values for each instrument."),
    _op("vec_sum", VECTOR, [_v()], desc="Sum of the vector field's values for each instrument."),
    _op("vec_max", VECTOR, [_v()], tier="C", desc="Maximum of the vector values."),
    _op("vec_min", VECTOR, [_v()], tier="C", desc="Minimum of the vector values."),
    _op("vec_count", VECTOR, [_v()], tier="C", desc="Number of vector values."),
    _op("vec_stddev", VECTOR, [_v()], tier="C", desc="Standard deviation of the vector values."),
    _op("vec_range", VECTOR, [_v()], tier="C", desc="max - min of the vector values."),
    _op("vec_ir", VECTOR, [_v()], tier="C", desc="Mean / std of the vector values."),
    _op("vec_skewness", VECTOR, [_v()], tier="C", desc="Skewness of the vector values."),
    _op("vec_kurtosis", VECTOR, [_v()], tier="C", desc="Kurtosis of the vector values."),
    _op("vec_norm", VECTOR, [_v()], tier="C", desc="Sum of absolute vector values."),
    _op("vec_percentage", VECTOR, [_v()], [_num("percentage", 0.5)], tier="C",
        desc="Percentile of the vector values."),
    _op("vec_powersum", VECTOR, [_v()], [_num("constant", 2)], tier="C",
        desc="Sum of vector values raised to constant."),
    _op("vec_choose", VECTOR, [_v()], [_num("nth", 0)], tier="C", desc="The nth vector value."),
    # ----------------------------------------------------------------- transformational
    _op("trade_when", TRANSFORMATIONAL, [_x(), _x("y"), _x("z")],
        desc="Trade y when x > 0, close the position when z > 0, otherwise hold yesterday's value."),
    _op("bucket", TRANSFORMATIONAL, [_x()],
        [_str("range", "0.1,1,0.1"), _str("buckets", ""), _bool("skipBoth"), _bool("NANGroup")],
        returns="group", desc="Assign x to buckets (a group) given a range 'start,end,step' or explicit buckets."),
    _op("tail", TRANSFORMATIONAL, [_x()], [_num("lower", 0), _num("upper", 0), _num("newval", 0)],
        desc="Replace values within [lower, upper] with newval."),
    _op("left_tail", TRANSFORMATIONAL, [_x()], [_num("maximum", 0)], desc="NaN where x > maximum."),
    _op("right_tail", TRANSFORMATIONAL, [_x()], [_num("minimum", 0)], desc="NaN where x < minimum."),
    _op("clamp", TRANSFORMATIONAL, [_x()],
        [_num("lower", 0), _num("upper", 0), _bool("inverse"), _str("mask", "nan")], tier="C",
        desc="Clamp x to [lower, upper] (or mask values inside when inverse=true)."),
    _op("filter", TRANSFORMATIONAL, [_x()], [_str("h", "1,2,3,4"), _str("t", "0.5")], tier="C",
        desc="Linear filter of x with coefficients h and t."),
    _op("keep", TRANSFORMATIONAL, [_x(), _x("f")], [_num("period", 5)], tier="C",
        desc="Keep x for period days after f becomes true."),
    # ----------------------------------------------------------------- group
    _group("group_neutralize", "x minus its group mean."),
    _group("group_rank", "Rank of x within its group, in [0, 1]."),
    _group("group_zscore", "Z-score of x within its group."),
    _group("group_scale", "Min-max scale x within its group."),
    _op("group_mean", GROUP, [_x(), _x("weight"), _g()], desc="Weighted mean of x within its group."),
    _op("group_backfill", GROUP, [_x(), _g(), _d()], [_num("std", 4.0)],
        desc="Fill NaN with the winsorized group mean of the past d days."),
    _group("group_median", "Median of x within its group."),
    _group("group_sum", "Sum of x within its group."),
    _group("group_count", "Number of valid x values within its group."),
    _group("group_max", "Maximum of x within its group."),
    _group("group_min", "Minimum of x within its group."),
    _group("group_std_dev", "Standard deviation of x within its group."),
    _group("group_normalize", "Normalise x within its group.",
           _bool("constantCheck"), _num("tolerance", 0.01), _num("scale", 1)),
    _group("group_percentage", "Percentile of x within its group.", _num("percentage", 0.5)),
    _op("group_vector_neut", GROUP, [_x(), _x("y"), _g()], desc="x neutralised against y within each group."),
    _op("group_vector_proj", GROUP, [_x(), _x("y"), _g()], desc="Projection of x onto y within each group."),
    _op("group_cartesian_product", GROUP, [_g("g1"), _g("g2")], returns="group",
        desc="Group formed by every combination of g1 and g2."),
    _op("group_coalesce", GROUP, [_g("g1"), _g("g2")], variadic=True, returns="group", tier="C",
        desc="First non-NaN group among the inputs."),
    _op("group_extra", GROUP, [_x(), _x("weight"), _g()], tier="C",
        desc="Fill NaN in x with the weighted group mean."),
    _op("group_multi_regression", GROUP, [_x("y"), _x(), _g()], tier="C", permissive=True,
        desc="Multi-factor regression of y on x within each group."),
    # ----------------------------------------------------------------- special
    _unary("inst_pnl", SPECIAL, "Per-instrument PnL of x.", tier="C"),
    _unary("self_corr", SPECIAL, "Self-correlation of x.", tier="C"),
    _op("convert", SPECIAL, [_x()], [_str("mode", "dollar2share")], tier="C",
        desc="Convert positions between dollar and share units."),
]

CURATED_OPERATORS: dict[str, OperatorSpec] = {spec.name: spec for spec in _SPECS}

#: Operators taking at least one group argument.
GROUP_ARG_OPERATORS: frozenset[str] = frozenset(n for n, s in CURATED_OPERATORS.items() if s.takes_group)

#: Operators taking a VECTOR argument.
VECTOR_OPERATORS: frozenset[str] = frozenset(
    n for n, s in CURATED_OPERATORS.items() if any(a.kind == "vector" for a in s.all_args)
)


# --------------------------------------------------------------------------- catalog construction


def _scopes(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {s.upper() for s in re.split(r"[\s,;|]+", value) if s}
    if isinstance(value, Iterable):
        return {str(s).upper() for s in value}
    return {str(value).upper()}


def _permissive_spec(item: Mapping[str, object]) -> OperatorSpec:
    name = str(item["name"])
    return OperatorSpec(
        name=name,
        category=str(item.get("category") or ""),
        positional=(ArgSpec("x", "any"),),
        variadic=True,
        min_args=0,
        definition=str(item.get("definition") or f"{name}(...)"),
        description=str(item.get("description") or ""),
        tier="API",
        permissive=True,
    )


def field_info_from_api(item: Mapping[str, object] | FieldInfo) -> FieldInfo:
    """Map a BRAIN ``/data-fields`` record to :class:`FieldInfo` (category = category id or name)."""
    if isinstance(item, FieldInfo):
        return item
    dataset = item.get("dataset") or {}
    dataset_id = str(dataset.get("id") or "") if isinstance(dataset, Mapping) else str(dataset)
    category = item.get("category") or ""
    if isinstance(category, Mapping):
        category = category.get("id") or category.get("name") or ""
    coverage = item.get("coverage")
    alpha_count = item.get("alphaCount", item.get("alpha_count"))
    return FieldInfo(
        id=str(item["id"]),
        type=str(item.get("type") or "MATRIX").upper(),
        dataset_id=dataset_id,
        category=str(category),
        coverage=float(coverage) if isinstance(coverage, (int, float)) else None,
        alpha_count=int(alpha_count) if isinstance(alpha_count, (int, float)) else None,
        description=str(item.get("description") or ""),
    )


def build_catalog(
    api_operators: list[dict] | None = None,
    fields: list[dict] | None = None,
) -> Catalog:
    """Build a :class:`Catalog` from the curated table plus optional BRAIN API data.

    ``api_operators`` is the ``GET /operators`` response.  Only operators whose scope
    includes ``REGULAR`` (or that have no scope) are kept.  Curated operators missing
    from the list are kept but marked ``available=False``; API operators without a
    curated signature get a permissive spec.  ``fields`` is a list of ``/data-fields``
    records; a non-empty list makes unknown identifiers errors.
    """
    operators: dict[str, OperatorSpec] = dict(CURATED_OPERATORS)
    strict_operators = False
    if api_operators is not None:
        strict_operators = True
        seen: set[str] = set()
        for item in api_operators:
            name = item.get("name")
            if not name:
                continue
            scopes = _scopes(item.get("scope"))
            if scopes and "REGULAR" not in scopes:
                continue
            name = str(name)
            seen.add(name)
            curated = CURATED_OPERATORS.get(name)
            if curated is None:
                operators[name] = _permissive_spec(item)
            else:
                operators[name] = replace(
                    curated,
                    available=True,
                    category=str(item.get("category") or curated.category),
                    definition=str(item.get("definition") or curated.definition),
                    description=str(item.get("description") or curated.description),
                )
        for name, spec in CURATED_OPERATORS.items():
            if name not in seen:
                operators[name] = replace(spec, available=False)

    field_infos = [field_info_from_api(f) for f in fields or ()]
    return Catalog(
        operators,
        {f.id: f for f in field_infos},
        strict_fields=bool(field_infos),
        strict_operators=strict_operators,
    )
