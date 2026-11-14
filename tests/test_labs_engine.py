"""Pure engine rules of the Search, Template and Evolution labs (alpha-harness' self-checks, and more)."""

from __future__ import annotations

import random

import pytest

from alphago.labs import ga, search, study, template
from alphago.labs.search import RandomTrial
from alphago.labs.tree import ParseError, parse, render
from tests.test_labs_support import OPERATORS

OPS = [{"name": n, "category": c, "definition": d, "scope": ["REGULAR"], "level": "ALL"} for n, c, d in OPERATORS]


# --- tree -------------------------------------------------------------------

@pytest.mark.parametrize("text", ["rank(close)", "a = rank(close); b = a * a; b - a", "x > 0 ? 1 : -1",
                                  "-close ^ 0.5", "2 ^ -x", 'bucket(rank(cap), range="0, 1 # 2")',
                                  "group_rank($ts_op($f, $d), $g)", "if_else(x > 1, ..., ...)"])
def test_parse_render_round_trip(text: str) -> None:
    tree = parse(text)
    assert parse(render(tree)) == tree


def test_parse_errors_name_a_position() -> None:
    with pytest.raises(ParseError, match=r"at \d+"):
        parse("rank(close,, 5)")
    with pytest.raises(ParseError, match="nests too deeply"):
        parse("rank(" * 70 + "x" + ")" * 70)


# --- search -----------------------------------------------------------------

def test_catalogue_sorts_operators_into_slots() -> None:
    cat = search.catalogue(OPS)
    assert "rank" in cat.cs and "zscore" in cat.cs
    assert "ts_rank" in cat.ts and "ts_backfill" not in cat.ts           # cleaning is never a slot
    assert set(cat.group) == {"group_rank", "group_zscore", "group_neutralize"}
    assert set(cat.vector) == {"vec_avg", "vec_max"}
    assert search.families(cat) == search.FAMILIES
    assert search.families(search.Catalogue(("rank",), (), (), (), {})) == ("cs",)


def test_render_shapes_and_keywords() -> None:
    p = {"field": "f", "family": "ts_ts", "ts": "ts_mean", "d": 5, "ts_outer": "ts_rank", "d_outer": 21}
    assert search.render(p, 63) == "ts_rank(ts_mean(ts_backfill(f, 63), 5), 21)"
    p = {"field": "v", "vector_op": "vec_avg", "family": "group_ts", "ts": "ts_moment", "d": 5,
         "group_op": "group_rank", "group": "sector"}
    assert search.render(p, 21, {"ts_moment": "k=0"}) == \
        "group_rank(ts_moment(ts_backfill(vec_avg(v), 21), 5, k=0), sector)"


SPACE = {"fields": {"a": "MATRIX", "b": "MATRIX", "v": "VECTOR"}, "families": list(search.FAMILIES),
         "cs": ["rank"], "ts": ["ts_mean", "ts_rank"], "group": ["group_rank"], "vector": ["vec_avg"],
         "lookbacks": list(search.LOOKBACKS), "groups": list(search.GROUPS), "universes": ["TOP3000", "TOP1000"],
         "absent": {"TOP1000": ["v"]}, "neutralizations": ["MARKET", "SECTOR"], "decays": [0, 5]}


def test_suggest_never_pairs_a_field_with_a_universe_lacking_it() -> None:
    choices = search.field_choices(SPACE)
    for seed in range(300):
        p = search.suggest(RandomTrial(random.Random(seed)), SPACE, choices)
        if p["universe"] == "TOP1000":
            assert p["field"] != "v"
        assert ("vector_op" in p) == (p["field"] == "v")
        assert ("d" in p) == (p["family"] in ("ts", "cs_ts", "ts_ts", "group_ts"))
        assert p["decay"] in (0, 5)
    assert search.first_pass(SPACE, "v") == {"universe": "TOP3000", "field@TOP3000": "v"}
    assert search.choices_of(SPACE, "field@TOP1000") == ("a", "b")
    assert search.choices_of(SPACE, "d_outer") == search.LOOKBACKS


def test_suggest_through_optuna_and_rebuild() -> None:
    """Asked define-by-run on a real TPE study; the rows it would store rebuild the same history."""
    class Lab:
        choices_of = staticmethod(search.choices_of)

    study_ = study._new_study(5, 0)
    choices = search.field_choices(SPACE)
    history = []
    for i in range(12):
        trial = study_.ask()
        p = search.suggest(trial, SPACE, choices)
        study_.tell(trial, float(i))
        history.append({"state": "COMPLETE", "value": float(i), "implausible": False,
                        "params": {**p, "_asked": list(trial.distributions)}})
    rebuilt = study.rebuild(history, study.Distributions(Lab(), SPACE), 5)
    assert [t.params for t in rebuilt.get_trials(deepcopy=False)] == [t.params for t in study_.get_trials()]
    # an implausible trial is told FAILURE whatever it scored
    history[0]["implausible"] = True
    again = study.rebuild(history, study.Distributions(Lab(), SPACE), 5)
    assert again.get_trials(deepcopy=False)[0].values == [-10.0]


def test_trim_keeps_best_and_latest() -> None:
    told = [("COMPLETE", 50.0 - n) for n in range(study.KEEP_BEST)]
    told += [("COMPLETE", -10.0), ("FAIL", None)] + [("COMPLETE", -5.0)] * 1000
    told += [("PRUNED", None), ("COMPLETE", 0.0), ("FAIL", None)] * (study.KEEP_LATEST // 2 + 1)
    latest = [i for i, (s, _) in enumerate(told) if s != "FAIL"][-study.KEEP_LATEST:]
    assert study.kept(told) == sorted({*range(study.KEEP_BEST), *latest})
    assert study.kept([("COMPLETE", -10.0), *[("PRUNED", None)] * 3]) == [0, 1, 2, 3]


def test_startup_trials() -> None:
    from alphago.labs.launch import startup_trials
    assert startup_trials(fields=4, size=20, per_round=10) == 10
    assert startup_trials(fields=135, size=1000, per_round=40) == 160
    assert startup_trials(fields=5000, size=1000, per_round=80) == 560


# --- template ---------------------------------------------------------------

TABLE = __import__("alphago.labs.tree", fromlist=["operator_table"]).operator_table(OPS)


def test_values_and_limits() -> None:
    assert template.values("5, 21, ") == ["5", "21"]
    with pytest.raises(ValueError, match="another \\$variable"):
        template.values("$x, 5")
    with pytest.raises(ValueError, match="at most 64"):
        template.values(", ".join(str(i) for i in range(65)))
    with pytest.raises(ValueError, match="type the signal"):
        template.values("...")


def test_template_problems() -> None:
    tree = template.program("x = rank(close); $y = 2; nope(close) + ...")
    found = template.template_problems(tree, TABLE)
    assert "$y is chosen by the search, so it can't be defined with =. Name the step without $." in found
    assert "Replace each ... with a signal." in found
    assert "nope is not one of your operators." in found
    assert any(p.startswith("x is never used after it is set") for p in found)
    assert any("has no name" in p for p in template.template_problems(template.program("rank(close); close"), TABLE))
    assert "ts_rank does not take 1 input." in template.template_problems(template.program("ts_rank(close)"), TABLE)


def test_values_and_fields_problems() -> None:
    uses = template.uses(template.program("$op(close, 5) + vec_avg($v) + $f"), {"vec_avg"}.__contains__)
    assert template.values_problems(uses["op"], ["ts_rank", "rank"], TABLE) == ["rank does not take 2 inputs."]
    assert template.values_problems(uses["op"], ["bogus"], TABLE) == ["bogus is not one of your operators."]
    assert template.field_types(uses["v"], None) == ["VECTOR"] and template.field_types(uses["f"], None) == ["MATRIX"]
    assert template.fields_problems(uses["op"], ["MATRIX"], []) == [
        "Fields can't be an operator. Type operator names as its values instead."]
    assert template.fields_problems(uses["f"], ["VECTOR"], []) == [
        "It holds vector fields only, so write it inside a vector operator, e.g. vec_avg($f)."]
    assert template.fields_problems(uses["f"], ["MATRIX", "VECTOR"], []) == [
        "Choose the vector operators its vector fields are reduced with."]


def test_render_folds_numbers_and_drops_the_last_assignment() -> None:
    space = {"variables": [{"name": "d", "kind": "values", "values": ["21"]},
                           {"name": "f", "kind": "fields", "fields": {"v": "VECTOR"}, "vector": ["vec_max"]}]}
    out = template.render("a = ts_mean($f, $d / 2); alpha = rank(a)", space,
                          {"$d": "21", "$f": "v", "$f#vector": "vec_max"})
    assert out == "a = ts_mean(vec_max(v), 10.5); rank(a)"


def test_combinations_and_draw_refuses_one_field_twice() -> None:
    space = {"universes": ["TOP3000"], "neutralizations": ["SECTOR"], "investability": ["none", "max_trade"],
             "variables": [{"name": "a", "kind": "fields", "fields": {"f1": "MATRIX", "f2": "VECTOR"},
                            "vector": ["vec_max"]},
                           {"name": "b", "kind": "fields", "fields": {"f1": "MATRIX", "f3": "MATRIX"}},
                           {"name": "op", "kind": "values", "values": ["ts_rank", "ts_zscore"]}]}
    assert template.combinations(space) == 1 * 1 * 2 * 2 * 2 * 2
    run = {"region": "USA", "delay": 1, "template": "$op($a - $b, 5)"}
    picks = template.field_choices(space)
    drawn = [template.draw(RandomTrial(random.Random(s)), run, space, picks) for s in range(80)]
    assert any(spec is None for _, spec in drawn)
    assert all(p["$a"] == p["$b"] == "f1" for p, spec in drawn if spec is None)
    sent = [spec for _, spec in drawn if spec]
    assert any("vec_max(f2)" in s.expression for s in sent)
    assert {s.settings["maxTrade"] for s in sent} == {"OFF", "ON"}
    assert all(s.settings["maxPosition"] == "OFF" for s in sent if s.settings["maxTrade"] == "ON")


def test_size_stats_and_kinds() -> None:
    pp = template.program("r = rank($field); if_else(abs(ts_arg_min(low, $lookback) - ts_arg_max(high, $lookback))"
                          " > $lookback / 2, r, 1 - r)")
    sized = template.size(pp, {"lookback": ["21"]})
    assert (sized.operators, sized.fields) == ((8, 8), (3, 3)) and sized.power_pool == (0, 0)
    assert template.size(template.program("rank($clean($f, 21))"), {"clean": ["ts_backfill", "ts_mean"]}) \
        .operators == (1, 2)
    one = template.program("rank($x / $y)")
    s1 = template.size(one, {})
    assert template.single_dataset(one, s1, {}, {"x": frozenset({"d1"}), "y": frozenset({"d1"})})
    assert not template.single_dataset(one, s1, {}, {"x": frozenset({"d1"}), "y": frozenset({"d2"})})
    assert template.most_pyramids(one, s1, {}, {}, {"x": frozenset({"A"}), "y": frozenset({"B", "C"})}) == 2


def test_lint_points_at_problems() -> None:
    lint = template.lint("rank(close", TABLE)
    assert lint[0]["message"] == "This ( is never closed." and lint[0]["column"] == 5
    lint = template.lint("rank(close)\n+ nope(x)", TABLE)
    assert any(i["message"] == "nope is not one of your operators." and i["line"] == 2 for i in lint)
    assert template.lint("rank($x) + ...", TABLE)[0]["level"] == "warning"


def test_defaults_by_name_and_blocks() -> None:
    datasets = [{"id": "fundamental6", "category_id": "fundamental", "subcategory_id": "fundamental-x"},
                {"id": "pv1", "category_id": "pv", "subcategory_id": None}]
    assert template.default_of("lookback", template.PRESETS, datasets) == {"kind": "values",
                                                                           "values": template.PRESETS["lookback"]}
    assert template.default_of("pv1_b", template.PRESETS, datasets) == {"kind": "fields", "dataset_ids": ["pv1"]}
    assert template.default_of("fundamental", template.PRESETS, datasets) == {"kind": "fields",
                                                                              "dataset_ids": ["fundamental6"]}
    assert template.default_of("nothing", template.PRESETS, datasets) is None
    assert template.from_blocks("{ts_rank OR ts_zscore}(FIELD A, LOOKBACK B)") == (
        "$op1($field, $lookback_b)", {"op1": {"kind": "values", "values": "ts_rank, ts_zscore"}})


# --- evolution --------------------------------------------------------------

def _market(datasets: tuple[str, ...] = ("pv1", "pv1"), names: tuple[str, ...] = ("close", "open")) -> ga.Market:
    fields = [{"field_id": f, "dataset_id": d, "field_type": "MATRIX"} for f, d in zip(names, datasets)]
    return ga.build_market(OPS, fields, ["MARKET", "SECTOR"])


def test_genes_and_mutations() -> None:
    market = _market()
    tree = parse("group_rank(ts_mean(close, 21) - open, sector)")
    kinds = sorted(k for _, k in market.genes(tree))
    assert kinds == ["field", "field", "group", "op", "op", "op", "window"]
    rng = random.Random(3)
    for _ in range(50):
        child, settings = ga.mutate(rng, tree, {"neutralization": "MARKET", "decay": 3, "truncation": 0.08},
                                    market, 1.0)
        out = render(child)
        assert out.startswith("group_")
        assert settings["neutralization"] == "SECTOR" and settings["decay"] != 3
        window = int(out.split(", ")[1].split(")")[0]) if "ts_" in out else 21
        assert window in (10, 42, 21)


def test_mutation_swaps_a_field_read_twice_everywhere() -> None:
    market = _market()
    tree = parse("close / ts_mean(close, 21)")
    rng = random.Random(1)
    for _ in range(30):
        child, _ = ga.mutate(rng, tree, {"neutralization": "MARKET", "decay": 0, "truncation": 0.08}, market, 1.0)
        reads = [n.value for _, n in __import__("alphago.labs.tree", fromlist=["walk"]).walk(child)
                 if n.kind == "name"]
        assert len(set(reads)) == 1


def test_operator_peer_gets_forced_options() -> None:
    ops = OPS + [{"name": "ts_moment", "category": "Time Series", "definition": "ts_moment(x, d, k = 0)",
                  "scope": ["REGULAR"], "level": "ALL"}]
    market = ga.build_market(ops, [{"field_id": "close", "dataset_id": "pv1", "field_type": "MATRIX"}], ["MARKET"])
    assert "ts_moment" in market.peers(parse("ts_mean(close, 5)"))
    seen = set()
    rng = random.Random(0)
    for _ in range(200):
        child, _ = ga.mutate(rng, parse("ts_mean(close, 5)"), {"neutralization": "MARKET", "decay": 0,
                                                                 "truncation": 0.08}, market, 1.0, keep_decay=True)
        seen.add(render(child).split("(")[0])
        if render(child).startswith("ts_moment"):
            assert "k=0" in render(child)
    assert "ts_moment" in seen


def test_crossover_and_clone_rules() -> None:
    swaps, fixed = _market(("pv1", "pv1")), _market(("pv1", "pv2"))
    a = ga.Parent.of("a", "close", {"neutralization": "MARKET", "decay": 3})
    b = ga.Parent.of("b", "open", {"neutralization": "MARKET", "decay": 10})

    def decays(market: ga.Market, x: ga.Parent, y: ga.Parent, keep: bool) -> set[int]:
        rng, shapes = random.Random(7), {p.tree: ga.Shape.of(p.tree, market) for p in (x, y)}
        made = [ga.child(rng, x, y, market, 1.0, shapes, keep_decay=keep) for _ in range(50)]
        return {int(s["decay"]) for _, s in filter(None, made)}

    assert decays(swaps, a, b, True) == {3, 10}
    assert not decays(swaps, a, b, False) <= {3, 10}
    twins = ga.Parent.of("c", "close", {"neutralization": "MARKET", "decay": 3}), \
        ga.Parent.of("d", "close", {"neutralization": "MARKET", "decay": 10})
    one_neutral = ga.build_market(OPS, [{"field_id": "close", "dataset_id": "pv1", "field_type": "MATRIX"},
                                        {"field_id": "open", "dataset_id": "pv2", "field_type": "MATRIX"}], ["MARKET"])
    assert decays(one_neutral, *twins, True) == set()                     # nothing left to change: no child
    del fixed
    # different shapes graft, never growing past the larger parent
    p1 = ga.Parent.of("x", "rank(ts_mean(close, 21))", {"neutralization": "MARKET"})
    p2 = ga.Parent.of("y", "zscore(ts_delta(open, 5))", {"neutralization": "MARKET"})
    rng, shapes = random.Random(2), {p.tree: ga.Shape.of(p.tree, swaps) for p in (p1, p2)}
    for _ in range(30):
        made = ga.child(rng, p1, p2, swaps, 0.05, shapes)
        if made:
            from alphago.labs.tree import operator_count
            assert operator_count(made[0]) <= 2
            assert (made[0], made[1]["neutralization"], made[1]["decay"], made[1]["truncation"]) not in \
                {p1.key, p2.key}


def test_seed_score_population_and_stall() -> None:
    good = {"sharpe": 2.5, "fitness": 1.8, "turnover": 0.2, "margin": 0.001, "drawdown": 0.06, "returns": 0.1,
            "checks": [{"name": "LOW_SHARPE", "result": "PASS", "limit": 1.25}]}
    weak = {**good, "sharpe": 1.0, "fitness": 0.6, "checks": [{"name": "LOW_SHARPE", "result": "FAIL"}]}
    assert ga.seed_score(good) > ga.seed_score(weak)
    assert ga.pass_rate(good["checks"]) == 1.0 and ga.pass_rate(weak["checks"]) == 0.0
    assert ga.pass_rate([{"name": "LOW_2Y_SHARPE", "result": "FAIL"}]) is None   # test-year checks are not judged
    assert ga.utility({"train_sharpe": 3.0, "sharpe": 0.1}) == ga.utility({"sharpe": 3.0})
    assert ga.population_for(500) == 20 and ga.population_for(20_000) == 100 and ga.population_for(5000, 50) == 50
    flat = [1.0] + [0.5] * ga.PATIENCE
    assert ga.stalled(flat) and not ga.stalled([1.0] * ga.PATIENCE)
    assert not ga.stalled([1.0] + [0.5] * (ga.PATIENCE - 1) + [2.0])


def test_seed_problem_reasons() -> None:
    market = _market()
    row = {"alpha_id": "A", "expression": "rank(close)", "region": "USA", "delay": 1, "universe": "TOP3000",
           "fitness": 1.0, "sim_type": "REGULAR"}
    assert ga.seed_problem(row, "USA", 1, "TOP3000", market) is None
    assert ga.seed_problem(None, "USA", 1, "TOP3000", market) == "Not stored here. Sync from BRAIN first."
    assert ga.seed_problem({**row, "submitted": True}, "USA", 1, "TOP3000", market) == "Already submitted."
    assert ga.seed_problem({**row, "universe": "TOP1000"}, "USA", 1, "TOP3000", market) == \
        "It ran in USA D1 TOP1000."
    assert ga.seed_problem({**row, "fitness": None}, "USA", 1, "TOP3000", market) == "It has no Fitness yet."
    assert ga.seed_problem({**row, "expression": "rank(nowhere)"}, "USA", 1, "TOP3000", market) == \
        "nowhere is not a data field here."
    assert "SUPER" in ga.seed_problem({**row, "sim_type": "SUPER"}, "USA", 1, "TOP3000", market)


def test_independent_parents_are_kept_apart() -> None:
    import numpy as np
    rng = np.random.default_rng(0)
    dates = [f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(300)]
    base = rng.normal(0, 1, 300)
    profiles = {"a": (dates, list(base)), "b": (dates, list(base + rng.normal(0, 0.1, 300))),
                "c": (dates, list(rng.normal(0, 1, 300)))}
    assert ga.independent(["a", "b", "c", "d"], profiles, 10) == ["a", "c", "d"]
    assert ga.independent(["a", "b", "c"], profiles, 1) == ["a"]
