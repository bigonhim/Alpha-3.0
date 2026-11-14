"""Tools: Settings Sampler grids (truncation, investability, ordering), Correlation Breaker compress and recipes,
Fields Sampler swaps and buckets on a small synthetic catalogue, Copy Alphas Markdown and the description template."""

from __future__ import annotations

import random

import pytest

from alphago.brain import brain
from alphago.brain.options import SimOptions
from alphago.brain.platform import investability_pairs, truncation_for
from alphago.db import db
from alphago.db.models import Alpha, DataField, Dataset, Operator
from alphago.tools import common
from alphago.tools import correlation_breaker as cb
from alphago.tools import fields_sampler as fs
from alphago.tools import settings_sampler as ss
from alphago.tools.copy_alphas import alphas_markdown
from alphago.tools.description import assemble, description_problems, message, operators_in, short
from alphago.tools.field_buckets import FieldText, split

USA = {"region": "USA", "delay": 1, "universe": "TOP3000", "decay": 4, "neutralization": "SUBINDUSTRY",
       "truncation": 0.08, "instrumentType": "EQUITY", "language": "FASTEXPR", "pasteurization": "ON",
       "nanHandling": "OFF", "maxTrade": "OFF", "maxPosition": "OFF", "testPeriod": "P0Y0M0D"}


@pytest.fixture(autouse=True)
def fallback_options(monkeypatch):
    monkeypatch.setattr(brain, "options", SimOptions())


@pytest.fixture
async def catalog(tmp_path):
    await db.init(tmp_path / "tools.db")

    def field(fid, region="USA", delay=1, universe="TOP3000", dataset="pv1", kind="MATRIX", desc="", cov=0.9,
              users=1, alphas=1, cat="pv", sub="pv-price"):
        return DataField(id=fid, dataset_id=dataset, region=region, delay=delay, universe=universe, type=kind,
                         description=desc, coverage=cov, user_count=users, alpha_count=alphas, category=cat,
                         category_id=cat, subcategory_id=sub, subcategory=sub)

    rows = []
    for u in ("TOP3000", "TOP1000", "TOPSP500"):
        for f in ("close", "returns", "volume", "cap", "adv20"):
            rows.append(field(f, universe=u))
    rows.append(field("close", region="EUR", universe="TOP2500"))
    rows.append(field("close", region="CHN", universe="TOP2000U", cov=0.5))
    rows.append(field("returns", region="CHN", universe="TOP2000U"))
    rows += [field(f"fnd_{n}", dataset="fnd6", desc=d, cat="fundamental", sub="fundamental-bs", cov=c)
             for n, d, c in (("liabilities", "Total liabilities of the company", 0.95),
                             ("assets", "Total assets of the company", 0.97),
                             ("debt", "Total debt of the company, long term and current", 0.9),
                             ("rdq_date", "Report date of quarterly earnings", 1.0),
                             ("equity", "Shareholders equity of the company", 0.8))]
    rows += [field("anl_liabilities_est", dataset="anl4", desc="Analyst estimate of total liabilities",
                   cat="analyst", sub="analyst-est"),
             field("sta1_cluster", kind="GROUP", dataset="sta1", users=90, alphas=400),
             field("sta1_rare", kind="GROUP", dataset="sta1", users=10, alphas=20),
             field("sta1_cluster", kind="GROUP", dataset="sta1", region="EUR", universe="TOP2500"),
             field("close_fast_d1", dataset="pv1")]
    async with db.write() as s:
        s.add_all(rows)
        s.add(Dataset(id="fnd6", region="USA", delay=1, universe="TOP3000", name="Fundamentals",
                      description="Company balance sheets."))
        for name in ("ts_regression", "log", "vector_neut", "ts_delta", "ts_decay_linear", "ts_av_diff",
                     "trade_when", "densify", "group_cartesian_product", "group_neutralize", "bucket", "rank",
                     "group_zscore", "ts_std_dev"):
            s.add(Operator(name=name, category="Time Series", definition=f"{name}(x)",
                           description=f"{name} does its thing"))
    yield
    await db.close()


# ---------------------------------------------------------------- Settings Sampler

def test_truncation_agent_and_investability_grids() -> None:
    cases = {("USA", 1, "TOP3000"): 0.08, ("USA", 1, "TOP1000"): 0.06, ("USA", 1, "TOP200"): 0.05,
             ("USA", 1, "TOPSP500"): 0.05, ("USA", 0, "TOP3000"): 0.05, ("CHN", 1, "TOP2000U"): 0.08,
             ("JPN", 1, "TOP1600"): 0.06, ("EUR", 1, "TOPCS1600"): 0.05, ("GLB", 1, "MINVOL1M"): 0.05,
             ("GBR", 1, "TOP700"): 0.05, ("ALL", 1, "LARGE"): 0.05}
    assert {k: truncation_for(*k) for k in cases} == cases
    assert investability_pairs("USA") == [("OFF", "OFF"), ("OFF", "ON"), ("ON", "OFF")]
    assert investability_pairs("CHN") == [("OFF", "OFF"), ("ON", "OFF")]
    assert ss.decay_ladder(4) == [4, 6, 9, 14, 24] and ss.decay_ladder(500) == [500, 502, 505, 510, 512]


def test_expand_holds_settings_shuffles_tens_and_drops_unhedged() -> None:
    usa = {"region": "USA", "neutralizations": ["NONE", "SUBINDUSTRY", "INDUSTRY"],
           "pairs": [{"maxTrade": "OFF", "maxPosition": "OFF"}, {"maxTrade": "ON", "maxPosition": "OFF"}],
           "markets": [{"region": "USA", "delay": 1, "universe": u} for u in ("TOP3000", "TOP1000")]}
    chn = {"region": "CHN", "neutralizations": ["INDUSTRY"], "pairs": [{"maxTrade": "OFF", "maxPosition": "OFF"}],
           "markets": [{"region": "CHN", "delay": 1, "universe": "TOP2000U"}]}
    source = {"expression": "rank(close)", "decay": 4, "truncation": 0.07}
    reqs = ss.expand([usa, chn], set(), set(), set(), source, decays=[0, 10, 10], rng=random.Random(1))
    # USA: 2 markets x 3 neutralizations x 2 pairs x 2 decays, minus NONE/OFF/OFF x 2 markets x 2 decays = 20
    assert sum(r["settings"]["region"] == "USA" for r in reqs) == 20
    assert not any(r["settings"]["neutralization"] == "NONE" and r["settings"]["maxTrade"] == "OFF" for r in reqs)
    assert {r["settings"]["decay"] for r in reqs} == {0, 10} and {r["settings"]["truncation"] for r in reqs} == {0.07}
    assert [r["settings"]["region"] for r in reqs][-2:] == ["CHN", "CHN"]       # the partial tail goes last
    for i in range(0, 20, 10):                                               # each full ten keeps one market
        assert len({(r["settings"]["region"], r["settings"]["delay"]) for r in reqs[i:i + 10]}) == 1
    agent = ss.expand([usa], {("USA", 1, "TOP1000")}, {"SUBINDUSTRY"}, set(), source, truncation_agent=True)
    assert {r["settings"]["truncation"] for r in agent} == {0.06}


async def test_settings_plan_places_every_field_and_counts_cost(catalog) -> None:
    found = await ss.plan(expression="rank(close / returns)")
    assert found["problems"] == []
    markets = {(m["region"], m["delay"], m["universe"]) for r in found["regions"] for m in r["markets"]}
    assert ("CHN", 1, "TOP2000U") in markets and ("EUR", 1, "TOP2500") not in markets     # EUR lacks returns
    chn = next(r for r in found["regions"] if r["region"] == "CHN")
    assert chn["markets"][0]["coverage"] == 0.5 and chn["position_available"] is False
    assert any("not downloaded" in w for w in found["warnings"])
    preview = await ss.preview({"expression": "rank(close / returns)", "neutralizations": ["INDUSTRY"],
                                "pairs": [["OFF", "OFF"]], "markets": [["USA", 1, "TOP3000"], ["CHN", 1, "TOP2000U"]],
                                "decays": [0, 2]})
    assert preview["simulations"] == 4 and preview["cost"] == 4 and preview["batches"] == 2
    with pytest.raises(common.ToolError):
        ss.requests_for(found, {"neutralizations": []})
    missing = await ss.plan(expression="rank(nothing_here)")
    assert missing["problems"] == ["nothing_here is not downloaded in any market. Sync its dataset on the Datasets "
                                   "page first."]


async def test_settings_presets(catalog) -> None:
    async with db.write() as s:
        s.add(Alpha(id="SS1", code="rank(close)", status="ACTIVE", settings=USA))
    found = await ss.plan("SS1")
    own = ss.own_choice(found)
    assert own["markets"] == [("USA", 1, "TOP3000")] and own["neutralizations"] == ["SUBINDUSTRY"]
    assert ss.ladder_choice(found)["decays"] == [4, 6, 9, 14, 24]
    defaults = ss.defaults_choice(found)
    assert ("EUR", 1, "TOP2500") in defaults["markets"] and ["OFF", "ON"] not in defaults["pairs"]
    assert "SUBINDUSTRY" in defaults["neutralizations"] and defaults["decays"] == []


def test_alpha_id_validation() -> None:
    assert common.alpha_id_of(" abC123 ") == "abC123"
    assert common.alpha_id_of("https://platform.worldquantbrain.com/alpha/XyZ987") == "XyZ987"
    with pytest.raises(common.ToolError, match="looks like an expression"):
        common.alpha_id_of("rank(close)")


# ---------------------------------------------------------------- Correlation Breaker

def test_compress_binds_alpha_once() -> None:
    one = cb.compress("rank(close)")
    assert one.prelude == () and one.binding == "alpha = rank(close)"
    many = cb.compress("x = ts_mean(close, 5); rank(x)")
    assert many.prelude == ("x = ts_mean(close, 5)",) and many.binding == "alpha = rank(x)"
    assigned = cb.compress("x = rank(close); y = -x")
    assert assigned.binding == "alpha = y" and assigned.prelude == ("x = rank(close)", "y = -x")
    already = cb.compress("alpha = rank(close)")
    assert already.binding == ""
    recipe = next(r for r in cb.RECIPES if r.id == "neut_both")
    assert one.program(recipe.tail, recipe.setup) == (
        "alpha = rank(close);\nno_size = vector_neut(alpha, log(cap));\n"
        "orthogonal = vector_neut(no_size, ts_delta(close, 5));\nts_decay_linear(orthogonal, 3)")
    assert many.bound == "x = ts_mean(close, 5);\nalpha = rank(x);"


def test_the_ten_recipes_and_custom_validation() -> None:
    assert [r.id for r in cb.RECIPES] == ["resid_returns", "resid_size", "neut_momentum", "neut_size", "neut_both",
                                          "fast_innovation", "event_gate", "cluster_country", "cluster_zscore",
                                          "cluster_volatility"]
    ok = cb.custom_recipe("a1", "Rank inside sector", "group_rank(alpha, sector)")
    assert not ok.missing and ok.operators == ("group_rank",) and ok.fields == ("sector",)
    assert cb.custom_recipe("a2", "", "rank(alpha)").missing == "has no name: give it one"
    assert cb.custom_recipe("a3", "x", "rank(close)").missing == "never reads alpha: write it around alpha"
    assert "last line assigns" in cb.custom_recipe("a4", "x", "y = rank(alpha)").missing
    assert cb.custom_recipe("a5", "x" * 121, "rank(alpha)").missing.startswith("name is 121 characters")
    assert cb.over_power_pool(9, 4) == "9 operators (limit 8) and 4 data fields (limit 3)"


async def test_breaker_plan_blocks_ranks_and_groups(catalog) -> None:
    checks = [{"name": "PROD_CORRELATION", "result": "FAIL", "limit": 0.7, "value": 0.81},
              {"name": "POWER_POOL_CORRELATION", "result": "PENDING"}]
    async with db.write() as s:
        s.add(Alpha(id="CB1", code="rank(fnd_liabilities / fnd_assets)", status="UNSUBMITTED", settings=USA,
                    checks=checks, classifications=[{"id": "DATA_USAGE:SINGLE_DATA_SET", "name": "Single"}]))
    found = await cb.plan("CB1", custom=[("k1", "Mine", "group_rank(alpha, sector)")])
    assert found["problems"] == [] and found["power_pool"] and found["single_dataset"]
    by_id = {r["id"]: r for r in found["recipes"]}
    assert by_id["cluster_country"]["blocked"].startswith("only for regions spanning countries")
    assert by_id["custom:k1"]["blocked"] == "needs group_rank, not on this account"
    assert by_id["group_top"]["name"] == "Neutralize inside sta1_cluster"
    assert by_id["group_exclusive"]["name"] == "Neutralize inside sta1_rare"        # EUR has sta1_cluster too
    assert by_id["neut_size"]["over_single_dataset"] == "cap (pv1)"
    assert by_id["resid_returns"]["transform"] == "ts_regression(alpha, returns, 63, rettype = 0)"
    reqs = cb.chosen_requests(found, ["neut_size", "cluster_country"])
    assert len(reqs) == 1 and reqs[0]["settings"]["simulationMode"] == "FULL"
    assert reqs[0]["settings"]["decay"] == 4 and reqs[0]["expression"].endswith("vector_neut(alpha, log(cap))")


def test_recipe_ranking_puts_what_worked_first() -> None:
    rows = [{"id": "a", "judged": 0, "worked": 0}, {"id": "b", "judged": 3, "worked": 0},
            {"id": "c", "judged": 2, "worked": 1}, {"id": "d", "judged": 4, "worked": 4}]
    assert [r["id"] for r in cb.ranked(rows)] == ["d", "c", "a", "b"]


# ---------------------------------------------------------------- Fields Sampler

def test_buckets_split_a_small_catalogue() -> None:
    def make(name: str, about: str) -> FieldText:
        return FieldText(name, about, 0.9)
    toy = [make("est_earnings_short_2", "Estimated earnings, short horizon"),
           make("est_earnings_long_3", "Estimated earnings, long horizon"),
           make("eps_change_30d", "Change in earnings per share over 30 days"),
           make("est_revenue_short_2", "Estimated revenue, short horizon"),
           make("est_revenue_long_4", "Estimated revenue, long horizon"),
           make("sales_growth_long", "Growth in revenue and sales, long horizon"),
           make("est_earnings_medium_3", "Estimated earnings, medium horizon"),
           make("est_revenue_medium_4", "Estimated revenue, medium horizon"),
           make("price_change_30d", "Price change over 30 days"),
           make("price_change_60d", "Price change over 60 days")]
    earnings, revenue = split(toy, ["est_earnings_medium_3", "est_revenue_medium_4"])
    assert sorted(earnings + revenue) == sorted(f.field for f in toy
                                                if f.field not in {"est_earnings_medium_3", "est_revenue_medium_4"})
    assert {"est_earnings_short_2", "est_earnings_long_3"} <= set(earnings)
    assert {"est_revenue_short_2", "est_revenue_long_4"} <= set(revenue)
    assert ("price_change_30d" in earnings) == ("price_change_60d" in earnings)
    assert len(split(toy, ["est_earnings_medium_3"])[0]) == len(toy) - 1


def test_variants_take_turns_and_grid_pairs_best_first() -> None:
    def c(*names: str) -> list[fs.Candidate]:
        return [fs.Candidate(n, "d", "dataset", 1.0) for n in names]
    turns = fs.variants("liabilities / assets", USA, {"liabilities": c("a", "b"), "assets": c("c")}, set())
    assert [v.expression for v in turns] == ["a / assets", "liabilities / c", "b / assets"]
    seen: set[str] = set()
    first = fs.variants("liabilities / assets", USA, {"liabilities": c("debt", "debt", "assets")}, seen)
    assert [v.expression for v in first] == ["debt / assets"]
    assert fs.variants("liabilities / assets", USA, {"liabilities": c("debt")}, seen) == []
    pair = fs.grid("rank(earnings) + rank(revenue)", USA, {"earnings": c("eps", "pe", "roe"),
                                                           "revenue": c("sales", "ebitda")}, 2, set())
    assert [v.expression for v in pair] == ["rank(eps) + rank(sales)", "rank(eps) + rank(ebitda)",
                                            "rank(pe) + rank(sales)", "rank(pe) + rank(ebitda)"]
    g1 = [fs.Variant("e", USA, str(n), "", "") for n in range(3)]
    g2 = [fs.Variant("e", USA, str(n), "", "") for n in range(3, 5)]
    assert [v.field for v in fs.taking_turns([g1, g2], 2)] == ["0", "1", "3", "4", "2"]
    assert fs.technical("fnd6_rdq_date") and fs.technical("isin") and not fs.technical("fnd6_acodo")


async def test_fields_plan_sweep_similar_twins_and_elsewhere(catalog) -> None:
    async with db.write() as s:
        s.add(Alpha(id="FS1", code="rank(fnd_liabilities / fnd_assets)", status="ACTIVE", settings=USA))
        s.add(Alpha(id="FS2", code="rank(fnd_liabilities / fnd_assets)", status="ACTIVE",
                    settings={**USA, "universe": "TOP1000"}))
        s.add(Alpha(id="FS3", code="rank(close)", status="UNSUBMITTED", settings=USA))
    sweep = await fs.plan("FS1", "sweep")
    slot = {r["field"]: [c["field"] for c in r["candidates"]] for r in sweep["slots"]}
    assert slot["fnd_liabilities"] == ["fnd_assets", "fnd_debt", "fnd_equity"]      # by coverage, no dates
    assert sweep["bucketable"] and [o["alpha_id"] for o in sweep["others"]] == ["FS2"]
    similar = await fs.plan("FS1", "similar", 2)
    tiers = [(c["field"], c["tier"]) for c in similar["slots"][0]["candidates"]]
    assert ("anl_liabilities_est", "market") in tiers
    assert sum(t == "dataset" for _, t in tiers) <= 2
    twins = await fs.plan("FS3", "sweep")
    first = twins["slots"][0]["candidates"][:2]
    assert [c["field"] for c in first] == ["close_fast_d1", "close_fast_d1 - close"]
    chosen, budget, fields = await fs.chosen_variants(sweep, {"budget": 4, "alphas": ["FS2"]})
    assert budget == 4 and len(chosen) == 4 and fields == ["fnd_liabilities", "fnd_assets"]
    assert {v.settings["universe"] for v in chosen} == {"TOP3000"}         # the first ten keep one market
    buckets = await fs.plan("FS1", "buckets", 2)
    assert buckets["problems"] == []
    # every other usable field of the dataset lands in exactly one bucket (the date field is never offered)
    assert sorted(c["field"] for r in buckets["slots"] for c in r["candidates"]) == ["fnd_debt", "fnd_equity"]


# ---------------------------------------------------------------- Copy Alphas and the description

async def test_copy_alphas_markdown(catalog) -> None:
    checks = [{"name": "LOW_SHARPE", "result": "PASS", "limit": 1.25, "value": 1.6},
              {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS", "limit": 0.6, "value": 0.91},
              {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
              {"name": "LOW_FITNESS", "result": "FAIL", "limit": 1.0, "value": 0.8}]
    async with db.write() as s:
        s.add(Alpha(id="CP1", code="rank(fnd_liabilities / fnd_assets)", status="UNSUBMITTED", settings=USA,
                    region="USA", sharpe=1.6, fitness=0.8, turnover=0.25, returns=0.1, drawdown=0.05, margin=0.0007,
                    long_count=1500, short_count=500, k_ratio=1.23, checks=checks, language="FASTEXPR",
                    raw={"investabilityConstrained": {"sharpe": 1.2, "turnover": 0.2, "longCount": 1400}},
                    classifications=[{"id": "X", "name": "Regular Alpha"}]))
        s.add(Alpha(id="CP2", code="rank(close)", status="ACTIVE", settings=USA, region="USA", sharpe=2.1))
    async with db.session() as s:
        alphas = [await s.get(Alpha, "CP1"), await s.get(Alpha, "CP2")]
    text = await alphas_markdown(alphas)
    assert text.index("rank(close)") < text.index("rank(fnd_liabilities")             # highest Sharpe first
    for line in ("Classifications:\nRegular Alpha", "Neutralization: Subindustry", "Language: Fast Expression",
                 "Calmar: 2.00", "Margin: 7.00‱", "K-Ratio: 1.23", "Long/Short Imbalance: +50%",
                 "Sharpe: 1.60\nSub-Universe Sharpe: 0.91\nTurnover: 25.00%", "Investability Constrained Metrics:",
                 "Long Count: 1,400", "Checks: FAIL (LOW_FITNESS)", "- CONCENTRATED_WEIGHT: PASS",
                 "# Metric Glossary", "# Category: fundamental", "### Dataset: fnd6 (Fundamentals)",
                 "Company balance sheets.", "- `fnd_assets` (MATRIX): Total assets of the company"):
        assert line in text, line
    bare = await alphas_markdown(alphas, glossary=False)
    assert "Metric Glossary" not in bare and bare.count("Alpha Expression:") == 2


def test_description_template_and_problems() -> None:
    long = "x" * 120
    text = assemble({"idea": f"Idea: {long}", "data": long, "operators": long})
    assert text.startswith("Idea: xxx") and "\nRationale for data used: " in text and short(text) == []
    assert short(assemble({"idea": "short", "data": long, "operators": long})) == ["Idea"]
    assert description_problems(text) == []
    assert description_problems("Idea: something") == [
        "Ensure this field has at least 100 characters.",
        'Ensure the description is following the provided template. "Rationale for data used:" is missing, out of '
        "order or empty."]
    assert operators_in("rank(-x / y) + ts_mean(z, 5) * -1") == ["add", "rank", "divide", "reverse", "multiply",
                                                                   "ts_mean"]
    msg = message("rank(x)", [{"name": "x", "type": "MATRIX", "description": "X things"}],
                  [{"name": "rank", "definition": "rank(x)", "description": "Ranks", "category": "Cross Sectional"},
                   {"name": "add", "category": "Arithmetic"}])
    assert "x (MATRIX): X things" in msg and "rank(x): Ranks" in msg and "add" not in msg.split("\n\n")[1]


async def test_write_description_retries_a_short_part(catalog, monkeypatch) -> None:
    from alphago.llm import llm
    from alphago.tools import description
    async with db.write() as s:
        s.add(Alpha(id="WD1", code="rank(fnd_assets)", status="UNSUBMITTED", settings=USA))
    answers = [description.DescriptionParts(idea="too short", data="d" * 120, operators="o" * 120),
               description.DescriptionParts(idea="i" * 120, data="d" * 120, operators="o" * 120)]
    prompts: list[str] = []

    async def fake(prompt, schema, **kwargs):  # noqa: ANN001, ANN003, ANN202
        prompts.append(prompt)
        return answers.pop(0)
    monkeypatch.setattr(llm, "configure", lambda *a: True)
    monkeypatch.setattr(llm, "generate_json", fake)
    text = await description.write_description("WD1")
    assert description_problems(text) == [] and len(prompts) == 2
    assert "fnd_assets (MATRIX): Total assets of the company" in prompts[0]
    assert "too short for Idea" in prompts[1]
