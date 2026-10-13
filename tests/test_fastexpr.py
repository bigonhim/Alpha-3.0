"""Tests for the FASTEXPR toolkit (parser, validator, metrics, canonical forms, transforms)."""

from __future__ import annotations

import pytest

from alphago.fastexpr import (
    Assign,
    BinOp,
    Call,
    Catalog,
    FastExprSyntaxError,
    FieldInfo,
    Ident,
    Num,
    Program,
    Str,
    Unary,
    add_backfill,
    canonical_hash,
    canonical_source,
    compute_metrics,
    inline,
    largest_common_subtree,
    negate,
    num,
    parse,
    replace_identifier,
    scale_windows,
    set_kwarg,
    set_windows,
    skeleton,
    skeleton_hash,
    strip_comments,
    trade_when_wrap,
    validate,
    walk,
    wrap,
)
from alphago.knowledge.operators import (
    CURATED_OPERATORS,
    GROUP_ARG_OPERATORS,
    INFIX_TO_OPERATOR,
    build_catalog,
)

CORPUS = [
    "rank(-ts_delta(close, 2))",
    "(high + low)/2 - close",
    "-ts_zscore(enterprise_value/ebitda, 63)",
    "avg_news = vec_avg(nws12_afterhsz_sl); rank(ts_sum(avg_news, 60)) > 0.5 ? 1 : rank(-ts_delta(close, 2))",
    "buzz = ts_backfill(-vec_sum(scl12_alltype_buzzvec), 20); ts_av_diff(buzz, 60)",
    "rel_days_since_max = rank(ts_arg_max(close, 30)); decline_pct = (vwap - close) / close; "
    "decline_pct / min(ts_decay_linear(rel_days_since_max, 1), 0.15)",
    "lookback=10; avg_ret=power(ts_product(returns+1,lookback),1/lookback); a=zscore(avg_ret); "
    "when=ts_rank(ts_std_dev(returns,60),126)>0.55; b=trade_when(when,a,-1); "
    "group_vector_neut(b, ts_mean(returns,120), subindustry)",
    "group_rank(ts_rank(operating_income/equity, 126), subindustry)",
    "ts_decay_exp_window(close, 20, factor=0.5)",
    'bucket(rank(cap), range="0.1,1,0.1")',
    "group_neutralize(rank(close), bucket(rank(cap), range='0.1,1,0.1'))",
    "winsorize(ts_backfill(assets, 120), std=4)",
    "event = volume > adv20; alpha = -ts_delta(close, 5); trade_when(event, alpha, -1)",
    "ts_regression(returns, ts_step(20), 20, lag=0, rettype=2) > 0",
    "!(close > open) && volume > adv20 || returns < 0",
    "quantile(ts_rank(pretax_income, 250))",
    "(close-open)/((high-low)+0.001)",
    "rank(close) /* comment */ // trailing",
    "rank(close);",
]


@pytest.fixture(scope="module")
def strict_catalog() -> Catalog:
    fields = [FieldInfo(name, "MATRIX") for name in ("close", "open", "high", "low", "volume", "returns", "cap", "adv20")]
    fields.append(FieldInfo("nws12_afterhsz_sl", "VECTOR"))
    fields.append(FieldInfo("pv13_h_min2_3000_sector", "GROUP"))
    return Catalog(fields=fields)


def codes(issues) -> set[str]:
    return {i.code for i in issues}


# --------------------------------------------------------------------------- parser


@pytest.mark.parametrize("src", CORPUS)
def test_round_trip(src: str) -> None:
    program = parse(src)
    regenerated = program.to_source()
    assert parse(regenerated) == program
    assert parse(regenerated).to_source() == regenerated


def test_parse_structure_and_precedence() -> None:
    p = parse("!(close > open) && volume > adv20 || returns < 0")
    expr = p.expression
    assert isinstance(expr, BinOp) and expr.op == "||"
    assert isinstance(expr.left, BinOp) and expr.left.op == "&&"
    assert isinstance(expr.left.left, Unary) and expr.left.left.op == "!"

    p = parse("a = 1; b = a * 2; c ? a + b * 2 : -b")
    assert list(p.assignments) == ["a", "b"]
    assert p.to_source() == "a = 1; b = a * 2; c ? a + b * 2 : -b"

    assert parse("a - (b - c)").to_source() == "a - (b - c)"
    assert parse("(a - b) - c").to_source() == "a - b - c"
    assert parse("-(a + b)").to_source() == "-(a + b)"
    assert parse("x ? y : z ? u : v").to_source() == "x ? y : z ? u : v"


def test_literals_and_kwargs() -> None:
    call = parse("f(.5, 1e-3, 5.0, TRUE, 'NAN', driver=gaussian, ignore=\"NAN\")").expression
    assert isinstance(call, Call)
    assert [a.to_source() for a in call.args] == [".5", "1e-3", "5.0", "true", '"NAN"']
    assert call.kwargs[0] == ("driver", Ident("gaussian"))
    assert call.kwargs[1] == ("ignore", Str("NAN"))
    assert num(5.0).text == "5" and num(0.25).text == "0.25"
    assert Num(5, "5") == Num(5.0, "5.0")


def test_strip_comments() -> None:
    assert strip_comments("rank(close) /* c */ // x\n# y") == "rank(close)"
    assert strip_comments('f(x, s="a#b//c")') == 'f(x, s="a#b//c")'
    assert parse("a = 1; # note\nrank(a)").to_source() == "a = 1; rank(a)"


def test_walk_preorder() -> None:
    names = [n.name for n in walk(parse("rank(ts_mean(close, 5))").expression) if isinstance(n, (Call, Ident))]
    assert names == ["rank", "ts_mean", "close"]


@pytest.mark.parametrize(
    ("src", "fragment"),
    [
        ("rank(close", ")"),
        ("close ^ 2", "power"),
        ("a = rank(close)", "assignment"),
        ("rank(close); rank(open)", "only the last statement"),
        ("", "empty"),
        ("rank(close) /* open comment", "unterminated"),
        ("f(a=1, b)", "positional argument follows keyword"),
        ("rank(" * 400 + "close" + ")" * 400, "nested too deeply"),
    ],
)
def test_syntax_errors(src: str, fragment: str) -> None:
    with pytest.raises(FastExprSyntaxError) as info:
        parse(src)
    assert fragment in info.value.message
    assert info.value.line >= 1 and info.value.column >= 1


def test_syntax_error_becomes_issue() -> None:
    result = validate("close ^ 2")
    assert not result.ok and codes(result.errors) == {"syntax"}
    assert "power" in result.errors[0].message
    assert result.program is None
    assert codes(validate(None).errors) == {"syntax"}  # type: ignore[arg-type]


# --------------------------------------------------------------------------- validator


def test_corpus_validates_without_errors() -> None:
    for src in CORPUS:
        if src.startswith("bucket("):  # a group is not a valid final alpha
            continue
        result = validate(src)
        assert result.ok, (src, result.errors)


def test_unknown_operator(strict_catalog: Catalog) -> None:
    result = validate("foo_op(close)", strict_catalog)
    assert "unknown_operator" in codes(result.errors)


def test_arity(strict_catalog: Catalog) -> None:
    assert "arity" in codes(validate("ts_mean(close)", strict_catalog).errors)
    assert "arity" in codes(validate("ts_mean(close, 5, 6)", strict_catalog).errors)
    assert validate("ts_rank(close, 20, 0)", strict_catalog).ok  # positional overflow into keywords
    assert "unknown_kwarg" in codes(validate("rank(close, foo=1)", strict_catalog).errors)
    assert "duplicate_kwarg" in codes(validate("ts_backfill(close, 20, lookback=30)", strict_catalog).errors)
    assert validate("ts_backfill(close, lookback=30)", strict_catalog).ok


def test_windows(strict_catalog: Catalog) -> None:
    assert "window" in codes(validate("ts_mean(close, -5)", strict_catalog).errors)
    assert "window" in codes(validate("ts_mean(close, 2.5)", strict_catalog).errors)
    assert "window" in codes(validate("ts_mean(close, 5000)", strict_catalog).errors)
    assert validate("lookback = 10; ts_mean(close, lookback)", strict_catalog).ok
    assert "window" in codes(validate("w = close; ts_mean(close, w)", strict_catalog).errors)


def test_vector_checks(strict_catalog: Catalog) -> None:
    result = validate("rank(nws12_afterhsz_sl)", strict_catalog)
    assert "vector_misuse" in codes(result.errors)
    assert "vec_avg" in result.errors[0].message
    assert validate("rank(vec_avg(nws12_afterhsz_sl))", strict_catalog).ok
    assert "vector_required" in codes(validate("vec_avg(close)", strict_catalog).errors)


def test_group_checks(strict_catalog: Catalog) -> None:
    assert "group_required" in codes(validate("group_rank(close, close)", strict_catalog).errors)
    assert validate("group_rank(close, densify(pv13_h_min2_3000_sector))", strict_catalog).ok
    assert validate("group_rank(close, pv13_h_min2_3000_sector)", strict_catalog).ok
    assert validate("group_rank(close, sector)", strict_catalog).ok
    assert validate('g = bucket(rank(cap), range="0.1,1,0.1"); group_neutralize(close, g)', strict_catalog).ok
    assert "group_misuse" in codes(validate("rank(pv13_h_min2_3000_sector)", strict_catalog).errors)


def test_unknown_identifier_strictness(strict_catalog: Catalog) -> None:
    strict = validate("rank(foo_bar)", strict_catalog)
    assert "unknown_identifier" in codes(strict.errors)
    loose = validate("rank(foo_bar)")
    assert loose.ok
    assert "unknown_identifier" in codes(loose.warnings)


def test_statement_checks(strict_catalog: Catalog) -> None:
    assert "use_before_assignment" in codes(validate(Program((Ident("a"), Assign("a", Ident("close")), Ident("a")))).errors)
    assert "assignment_last" in codes(validate(Program((Assign("a", Ident("close")),))).errors)
    assert "shadow_operator" in codes(validate("rank = close; ts_mean(rank, 5)").warnings)
    assert "constant_alpha" in codes(validate("rank(1)").warnings)
    deep = "rank(" * 10 + "close" + ")" * 10
    assert "complexity" in codes(validate(deep).warnings)


def test_unavailable_operator_from_api() -> None:
    api = [
        {"name": name, "category": spec.category, "scope": ["REGULAR"], "definition": spec.definition}
        for name, spec in CURATED_OPERATORS.items()
        if name != "ts_entropy"
    ]
    api.append({"name": "brand_new_op", "category": "Special", "scope": "REGULAR", "definition": "brand_new_op(x)"})
    api.append({"name": "combo_only", "category": "Special", "scope": ["COMBO"]})
    catalog = build_catalog(api)
    assert catalog.strict_operators
    assert catalog.operators["ts_entropy"].available is False
    assert catalog.operators["brand_new_op"].permissive
    assert "combo_only" not in catalog.operators
    result = validate("ts_entropy(close, 20)", catalog)
    assert "unavailable_operator" in codes(result.errors)
    assert "not enabled" in result.errors[0].message
    assert validate("brand_new_op(close, 3, mode=whatever)", catalog).ok


def test_build_catalog_fields() -> None:
    catalog = build_catalog(
        None,
        [
            {"id": "close", "type": "MATRIX", "dataset": {"id": "pv1", "name": "PV"}, "category": {"id": "pv"}},
            {"id": "sec_grp", "type": "GROUP", "dataset": {"id": "pv13"}, "category": "pv", "coverage": 0.9,
             "alphaCount": 12},
        ],
    )
    assert catalog.strict_fields and not catalog.strict_operators
    assert catalog.fields["sec_grp"].type == "GROUP" and catalog.fields["sec_grp"].alpha_count == 12
    assert validate("group_rank(close, sec_grp)", catalog).ok
    assert "group_rank" in GROUP_ARG_OPERATORS and INFIX_TO_OPERATOR["?:"] == "if_else"


# --------------------------------------------------------------------------- metrics


def test_metrics() -> None:
    m = compute_metrics(parse("group_rank(ts_rank(operating_income/equity, 126), subindustry)"))
    assert m.operator_count == 3
    assert m.unique_operators == {"group_rank", "ts_rank", "divide"}
    assert m.non_group_fields == {"operating_income", "equity"}
    assert m.group_fields == {"subindustry"}
    assert m.windows == (126,)
    assert compute_metrics(parse("rank(close) + rank(close)")).operator_count == 3


def test_metrics_variables_and_trade_when() -> None:
    m = compute_metrics(parse("event = volume > adv20; alpha = -ts_delta(close, 5); trade_when(event, alpha, -1)"))
    # greater, reverse, ts_delta, trade_when ("-1" is a literal, not an operator)
    assert m.operator_count == 4
    assert m.has_trade_when
    assert m.fields == {"volume", "adv20", "close"}
    assert compute_metrics("lookback = 10; ts_mean(close, lookback)").windows == (10,)
    m = compute_metrics("group_rank(close, pv13_unlisted_sector)")
    assert m.group_fields == {"pv13_unlisted_sector"} and m.non_group_fields == {"close"}


# --------------------------------------------------------------------------- canonical


def test_canonical_hashes() -> None:
    assert canonical_hash("close + open") == canonical_hash("open+close")
    assert canonical_hash("a + b + c") == canonical_hash("c + (b + a)")
    assert canonical_hash("ts_mean(close, 5)") != canonical_hash("ts_mean(close, 10)")
    assert skeleton_hash("ts_mean(close, 5)") == skeleton_hash("ts_mean(close, 10)")
    assert skeleton("ts_mean(close, 5)") == "ts_mean(close, #)"
    assert canonical_hash("a = rank(close); a") == canonical_hash("rank(close)")
    assert canonical_source("ts_rank(x, 20, 0.50)") == "ts_rank(x, 20, constant=0.5)"
    assert canonical_source("f(b=1.0, a=TRUE)") == "f(a=true, b=1)"
    assert len(canonical_hash("rank(close)")) == 16


def test_inline_chain() -> None:
    node = inline(parse("a = close; b = a + 1; c = rank(b); c * a"))
    assert node.to_source() == "rank(close + 1) * close"


def test_largest_common_subtree() -> None:
    a = "rank(ts_mean(close, 5)) + volume"
    b = "zscore(ts_mean(close, 5))"
    assert largest_common_subtree(a, b) == 3  # ts_mean, close, 5
    assert largest_common_subtree("rank(close)", "rank(close)") == 2
    assert largest_common_subtree("open", "close") == 0


# --------------------------------------------------------------------------- transforms


def _assert_valid(program) -> None:
    result = validate(program)
    assert result.ok, (program.to_source(), result.errors)


def test_scale_and_set_windows() -> None:
    p = scale_windows(parse("ts_mean(close, 5)"), 2)
    assert p.to_source() == "ts_mean(close, 10)"
    _assert_valid(p)
    p = scale_windows("lookback = 10; ts_mean(close, lookback) + ts_delta(close, 3)", 2)
    assert p.to_source() == "lookback = 20; ts_mean(close, lookback) + ts_delta(close, 6)"
    assert scale_windows("ts_decay_linear(close, 1)", 0.5).to_source() == "ts_decay_linear(close, 1)"
    assert scale_windows("ts_mean(close, 400)", 2).to_source() == "ts_mean(close, 512)"
    p = set_windows("ts_corr(close, volume, 5) + ts_rank(close, 3, 0)", 20)
    assert p.to_source() == "ts_corr(close, volume, 20) + ts_rank(close, 20, 0)"
    _assert_valid(p)


def test_wrap_and_negate() -> None:
    base = parse("ts_delta(close, 5)")
    p = wrap(base, "group_neutralize", "subindustry")
    assert p.to_source() == "group_neutralize(ts_delta(close, 5), subindustry)"
    _assert_valid(p)
    p = wrap(base, "ts_decay_linear", 5)
    assert p.to_source() == "ts_decay_linear(ts_delta(close, 5), 5)"
    _assert_valid(p)
    p = wrap(wrap(base, "rank"), "bucket", range="0.1,1,0.1")
    assert p.to_source() == 'bucket(rank(ts_delta(close, 5)), range="0.1,1,0.1")'

    neg = negate(base)
    assert neg.to_source() == "-ts_delta(close, 5)"
    _assert_valid(neg)
    assert negate(neg) == base
    assert negate("reverse(close)").to_source() == "close"
    assert negate("a + b").to_source() == "-(a + b)"


def test_add_backfill() -> None:
    p = add_backfill("rank(assets / ts_backfill(debt, 20)) + group_rank(assets, sector)", {"assets", "debt"}, 60)
    assert p.to_source() == (
        "rank(ts_backfill(assets, 60) / ts_backfill(debt, 20)) + group_rank(ts_backfill(assets, 60), sector)"
    )
    _assert_valid(p)


def test_trade_when_wrap() -> None:
    p = trade_when_wrap("alpha = rank(close); alpha", "volume > adv20")
    assert p.to_source() == "alpha = rank(close); trade_when(volume > adv20, alpha, -1)"
    _assert_valid(p)
    p = trade_when_wrap("rank(close)", "v = volume / adv20; v > 2", "abs(returns) > 0.1")
    assert p.to_source() == "v = volume / adv20; trade_when(v > 2, rank(close), abs(returns) > 0.1)"
    _assert_valid(p)


def test_replace_identifier_and_set_kwarg() -> None:
    p = replace_identifier("a = close; rank(a) + close", "close", "vwap")
    assert p.to_source() == "a = vwap; rank(a) + vwap"
    _assert_valid(p)
    p = set_kwarg("ts_decay_exp_window(close, 20, factor=0.5) + ts_decay_exp_window(open, 10)", "ts_decay_exp_window",
                  "factor", 0.9)
    assert p.to_source() == "ts_decay_exp_window(close, 20, factor=0.9) + ts_decay_exp_window(open, 10, factor=0.9)"
    _assert_valid(p)
    assert set_kwarg("ts_rank(close, 20, 0)", "ts_rank", "constant", 1).to_source() == "ts_rank(close, 20, 1)"
    _assert_valid(set_kwarg("quantile(close)", "quantile", "driver", "uniform"))


def test_transforms_do_not_mutate() -> None:
    src = "lookback = 10; ts_mean(close, lookback)"
    p = parse(src)
    scale_windows(p, 3)
    wrap(p, "rank")
    add_backfill(p, {"close"})
    assert p.to_source() == "lookback = 10; ts_mean(close, lookback)"
