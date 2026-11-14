"""LLM Power Pool Lab and Super Alpha Lab: prompt layout and field rotation, tolerant parsing, the offline checks,
the settings draw, stop rules and model fallback with a fake LLM, the Super Alpha grid and recipe rules, the SUPER
send path against the fake BRAIN, a Power Pool task end to end through the runner, and both tabs rendering."""

from __future__ import annotations

import asyncio
import json
import random
import re

import pytest
import respx
from sqlalchemy import select

from alphago.brain import brain
from alphago.brain import client as client_mod
from alphago.brain import simulations as sims_mod
from alphago.db import db
from alphago.db.models import DataField, Dataset, Operator
from alphago.db.models_labs import LabTrial
from alphago.labs import power_pool_lab as pp
from alphago.labs import runner as runner_mod
from alphago.labs import super_lab as sl
from alphago.labs.runner import LabRunner, RefillContext
from alphago.llm.base import LLMError
from tests.fakebrain import FakeBrain

OPS = [
    ("rank", "Cross Sectional", "rank(x, rate=2)"), ("ts_delta", "Time Series", "ts_delta(x, d)"),
    ("ts_rank", "Time Series", "ts_rank(x, d, constant = 0)"), ("ts_mean", "Time Series", "ts_mean(x, d)"),
    ("ts_zscore", "Time Series", "ts_zscore(x, d)"), ("ts_backfill", "Time Series", "ts_backfill(x, lookback = d)"),
    ("group_rank", "Group", "group_rank(x, group)"), ("vec_avg", "Vector", "vec_avg(x)"),
    ("add", "Arithmetic", "add(x, y, filter = false)"), ("subtract", "Arithmetic", "subtract(x, y, filter=false)"),
    ("multiply", "Arithmetic", "multiply(x ,y, ... , filter=false)"), ("divide", "Arithmetic", "divide(x, y)"),
    ("reverse", "Arithmetic", "reverse(x)"), ("abs", "Arithmetic", "abs(x)"),
]


# ---------------------------------------------------------------- fixtures

@pytest.fixture
async def catalog_db(tmp_path):
    await db.init(tmp_path / "llm_labs.db")
    async with db.write() as s:
        for name, cat, definition in OPS:
            s.add(Operator(name=name, category=cat, scope=["REGULAR"], definition=definition,
                           description=f"{name} does its thing"))
        s.add(Operator(name="ts_ir", category="Time Series", scope=["COMBO"], definition="ts_ir(x, d)"))
        for u in ("TOP3000", "TOP1000"):
            s.add(Dataset(id="ds1", region="USA", delay=1, universe=u, name="Analyst estimates",
                          description="What analysts expect.", category_name="Analyst", subcategory="Estimates"))
            for i in range(30):
                s.add(DataField(id=f"f{i:03d}", dataset_id="ds1", region="USA", delay=1, universe=u, type="MATRIX",
                                coverage=round(1 - i / 100, 2), description=f"estimate {i}"))
            s.add(DataField(id="vec1", dataset_id="ds1", region="USA", delay=1, universe=u, type="VECTOR",
                            coverage=0.5, description="news vector"))
            for f in ("close", "volume", "returns"):
                s.add(DataField(id=f, dataset_id="pv1", region="USA", delay=1, universe=u, type="MATRIX",
                                coverage=1.0, description=f"daily {f}"))
            s.add(DataField(id="industry", dataset_id="pv1", region="USA", delay=1, universe=u, type="GROUP",
                            coverage=1.0, description="industry"))
        s.add(DataField(id="only3000", dataset_id="ds1", region="USA", delay=1, universe="TOP3000", type="MATRIX",
                        coverage=0.3, description="only in the broad universe"))
    yield
    await db.close()


class FakeProvider:
    """An LLM provider answering from a script: dicts are answers, exceptions are raised."""

    local = False
    label = "Fake"

    def __init__(self, name: str, answers: list, model: str = "m1") -> None:
        self.name = name
        self.answers = list(answers)
        self.models = [{"name": model, "input_limit": 100_000}]
        self.model = model
        self.prompts: list[tuple[str, str]] = []

    def configure(self, api_key=None) -> bool:  # noqa: ANN001
        return True

    def pick(self, role: str) -> str:
        return self.model

    async def restore_models(self) -> None:
        return None

    async def generate_json(self, prompt, schema, *, system=None, role="ideation", purpose="x", thinking=None):  # noqa: ANN001, ANN201
        self.prompts.append((system, prompt))
        answer = self.answers.pop(0) if self.answers else {"alphas": []}
        if isinstance(answer, Exception):
            raise answer
        return schema.model_validate(answer)


@pytest.fixture
def providers(monkeypatch):
    from alphago.config import settings
    from alphago.llm import router

    def install(*fakes: FakeProvider) -> None:
        for f in fakes:
            monkeypatch.setitem(router.PROVIDERS, f.name, f)
        monkeypatch.setattr(settings.llm, "provider", fakes[0].name)
        monkeypatch.setattr(settings.llm, "fallbacks", [f.name for f in fakes[1:]])

    return install


def good_batch(start: int = 0, n: int = 20) -> dict:
    return {"alphas": [{"expression": f"rank(ts_delta(f{(start + i) % 30:03d}, {5 + i}))"} for i in range(n)]}


def params(**over) -> dict:  # noqa: ANN003
    base = {"region": "USA", "delay": 1, "universe": "TOP3000", "dataset_ids": ["ds1"], "model": "fake:m1",
            "neutralizations": ["INDUSTRY", "SUBINDUSTRY"], "universes": [], "cores": 2, "simulations": 40,
            "hypothesis": "", "favour_operators": []}
    return base | over


# ---------------------------------------------------------------- prompt layout and rotation

def made(i: int) -> pp.Field:
    return pp.Field(f"f{i:03}", "MATRIX", 1 - i / 1000, f"field {i}", None)


def dataset_ctx(name: str, size: int) -> pp.Context:
    own = tuple(made(i) for i in range(size))
    return pp.Context(id=name, name=name.title(), category="Fundamental › Ratios", description="What it holds.",
                      fields=own, basics=(pp.Field("close", "MATRIX", 1.0, "Close price", None),),
                      groups=("industry", "sector"), types={f.id: f.type for f in own},
                      own=frozenset(f.id for f in own),
                      names=frozenset(f.id for f in own) | {"close", "industry", "sector"}, held={})


OPERATORS = [{"name": "rank", "definition": "rank(x)", "category": "Cross Sectional"},
             {"name": "ts_delta", "definition": "ts_delta(x, d)", "category": "Time Series"}]
RUN = pp.Run("USA", 1, ["TOP3000"], ["MARKET"], "Quiet firms drift.", ["ts_delta"])


def ask(ctx: pp.Context, memory: str = "None yet.", offset: int = 0) -> tuple[str, int]:
    return pp.user_prompt(ctx, OPERATORS, RUN, memory, offset, 40_000)


def shared(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))


def walk(ctx: pp.Context, calls: int) -> tuple[list[int], set[str]]:
    starts, seen, offset = [], set(), 0
    for _ in range(calls):
        prompt, shown = ask(ctx, offset=offset)
        starts.append(offset)
        seen |= set(re.findall(r"^(f\d{3}) ·", prompt, re.MULTILINE))
        offset = pp.next_offset(offset, shown, len(ctx.fields))
    return starts, seen


def test_prompt_sections_are_in_cache_order():
    big, small = dataset_ctx("big", 450), dataset_ctx("small", 150)
    prompt, _ = ask(big)
    order = ["MARKET\n", "OPERATORS\n", "OPERATORS NOT USED", "PRICE AND VOLUME", "GROUPING_FIELDS", "DATASET\n",
             "FIELDS OF big", "YOUR EARLIER ALPHAS", "HYPOTHESIS TO TEST", "Write 20 new"]
    places = [prompt.find(h) for h in order]
    assert -1 not in places and places == sorted(places)
    later, _ = ask(big, memory="Simulated, best Sharpe first:\n1.20 | rank(close)")
    assert prompt.index("YOUR EARLIER") < shared(prompt, later) < prompt.index("HYPOTHESIS")
    assert shared(prompt, ask(small)[0]) >= prompt.index("DATASET\n")
    without = pp.user_prompt(big, OPERATORS, pp.Run("USA", 1, ["TOP3000"], ["MARKET"]), "None yet.", 0, 40_000)[0]
    assert "HYPOTHESIS" not in without and "OPERATORS NOT USED" not in without
    assert "Write 20 new Power Pool Alphas that use big." in without
    agnostic = pp.user_prompt(big, OPERATORS, pp.Run("ALL", 1, ["TOP3000"], ["MARKET"]), "None yet.", 0, 40_000)[0]
    assert "runs in USA, Europe, Asia and Global at once" in agnostic


def test_fields_rotate_at_fixed_places():
    big, small = dataset_ctx("big", 450), dataset_ctx("small", 150)
    starts, seen = walk(big, 6)
    assert starts == [0, 200, 400] * 2
    assert len(walk(big, 3)[1]) == 450 == len(seen)
    assert walk(small, 3)[0] == [0, 0, 0]
    assert pp.next_offset(1050, 200, 450) == 350 and pp.next_offset(1250, 200, 450) == 0
    assert pp.next_offset(7, 0, 0) == 0


def test_a_small_budget_shows_fewer_fields():
    big = dataset_ctx("big", 450)
    _, full = pp.user_prompt(big, OPERATORS, RUN, "None yet.", 0, 40_000)
    _, few = pp.user_prompt(big, OPERATORS, RUN, "None yet.", 0, 1_400)
    assert full == 200 and 0 < few < full


def test_system_prompt_is_alpha_harness_verbatim():
    assert pp.POWER_POOL_LAB.startswith("You write Alphas for WorldQuant BRAIN in Fast Expression, and every one")
    assert "trade_when(volume > adv20, rank(vec_avg(z)), -1)   counts 4" in pp.POWER_POOL_LAB
    assert pp.POWER_POOL_LAB.rstrip().endswith('holding exactly 20 different expressions.')


# ---------------------------------------------------------------- parsing

def test_answers_parse_tolerantly():
    fenced = '```json\n{"alphas": [{"expression": "rank(close)"}]}\n```'
    assert pp.parse_alphas(json.dumps({"alphas": [{"expression": "a"}] * 2})) == [{"expression": "a"}] * 2
    assert pp.parse_alphas(fenced) == pp.parse_alphas('[{"expression": "rank(close)"}]') == [
        {"expression": "rank(close)"}]
    assert pp.parse_alphas("Here are some ideas.") == []
    assert pp.parse_alphas('{"alphas": [{"note": "x"}, {"expression": "rank(open)"}]}') == [
        {"expression": "rank(open)"}]
    batch = pp.PowerPoolBatch.model_validate(["rank(close)", {"expression": "rank(open)", "why": "x"}])
    assert [a.expression for a in batch.alphas] == ["rank(close)", "rank(open)"]


# ---------------------------------------------------------------- the offline checks

async def test_check_rejects_what_breaks_a_rule(catalog_db):
    ops = await pp.synced_operators()
    ctx = await pp.context_for("USA", 1, ["TOP3000", "TOP1000"], "ds1")
    cat = pp.catalog_for(ctx, ops)

    def why(text: str) -> str:
        with pytest.raises(pp.Rejected) as exc:
            pp.check(text, ctx, cat)
        return str(exc.value)

    assert why("rank(") .startswith("Could not be read")
    assert "nosuch" in why("rank(nosuch)")
    assert "9 operators; Power Pool allows 8" in why("rank(" * 9 + "f001" + ")" * 9)
    assert "4 data fields" in why("rank(f001 + f002 + f003 + f004)")
    assert why("rank(close / volume)") == "Uses no field of ds1."
    assert "VECTOR field" in why("x = vec1; vec_avg(x) + rank(f001)")
    expression, used = pp.check("group_rank(ts_backfill(f001, 60) / close, industry)", ctx, cat)
    assert used == {"f001", "close", "industry"} and expression.startswith("group_rank(")
    # backfills are not counted: 8 counted operators plus two backfills pass
    pp.check("rank(ts_delta(ts_backfill(f001, 5), 5)) + rank(ts_delta(ts_backfill(f002, 5), 5)) "
             "+ rank(ts_delta(f003, 5))", ctx, cat)
    assert pp.check("vec_avg(vec1)", ctx, cat)[1] == {"vec1"}


async def test_context_holds_fields_per_universe_and_sorts(catalog_db):
    ctx = await pp.context_for("USA", 1, ["TOP1000", "TOP3000"], "ds1")
    assert [f.id for f in ctx.fields[:3]] == ["f000", "f001", "f002"]
    assert ctx.held["only3000"] == {"TOP3000"} and ctx.held["f001"] == {"TOP1000", "TOP3000"}
    assert [f.id for f in ctx.basics] == ["close", "volume", "returns"]
    assert "industry" in ctx.groups and ctx.name == "Analyst estimates" and "Analyst" in ctx.category
    narrowed = await pp.context_for("USA", 1, ["TOP3000"], "ds1", {"types": ["VECTOR"]})
    assert [f.id for f in narrowed.fields] == ["vec1"]
    assert await pp.context_for("USA", 1, ["TOP3000"], "nope") is None


async def test_settings_draw_fits_every_used_field(catalog_db):
    ctx = await pp.context_for("USA", 1, ["TOP1000", "TOP3000"], "ds1")
    run = pp.Run("USA", 1, ["TOP1000", "TOP3000"], ["INDUSTRY", "SUBINDUSTRY"])
    rng = random.Random(1)
    draws = [pp.draw(rng, frozenset({"only3000", "close"}), ctx, run) for _ in range(30)]
    assert {d["universe"] for d in draws} == {"TOP3000"}
    assert {d["decay"] for d in draws} <= {0, 3, 5, 7, 10} and all(d["truncation"] == 0.08 for d in draws)
    assert {d["neutralization"] for d in draws} <= {"INDUSTRY", "SUBINDUSTRY"}
    only1000 = pp.Run("USA", 1, ["TOP1000"], ["INDUSTRY"])
    with pytest.raises(pp.Rejected, match="No downloaded universe has all of"):
        pp.draw(rng, frozenset({"only3000"}), ctx, only1000)


# ---------------------------------------------------------------- planning, calls, stops

async def test_preview_shows_the_prompt_and_refuses_what_cannot_run(catalog_db, providers):
    providers(FakeProvider("fake", []))
    plan = await pp.plan_task(params(favour_operators=["ts_rank", "not_an_operator"]))
    assert plan["problems"] == [], plan["problems"]
    assert plan["universes"] == ["TOP3000", "TOP1000"] and plan["llmCalls"] == 2 and plan["fields"] == 32
    assert plan["favourOperators"] == ["ts_rank"]
    assert plan["prompt"]["system"] == pp.POWER_POOL_LAB and "FIELDS OF ds1 · 1-32 of 32" in plan["prompt"]["user"]
    assert plan["prompt"]["tokens"] == pp.estimate_tokens(pp.POWER_POOL_LAB + plan["prompt"]["user"])
    bad = await pp.plan_task(params(dataset_ids=[], neutralizations=[], model="fake:m1", universes=["NOPE"]))
    assert "Choose at least one dataset." in bad["problems"] and pp.NO_NEUTRALIZATION in bad["problems"]
    assert any("None of the chosen universes" in p for p in bad["problems"])
    many = await pp.plan_task(params(dataset_ids=[f"d{i}" for i in range(51)]))
    assert any("at most 50 datasets" in p for p in many["problems"])


async def _task(lab_runner: LabRunner, **over) -> int:  # noqa: ANN003
    p = params(**over)
    tid = await lab_runner.create("power_pool", p, cores=p["cores"], max_trials=p["simulations"])
    await lab_runner._set(tid, status="RUNNING")
    return tid


@pytest.fixture
def lab_runner(monkeypatch):
    r = LabRunner()
    r.register(pp.PowerPoolLab())
    r.register(sl.SuperLab())
    monkeypatch.setattr(runner_mod, "runner", r)
    pp._calls.clear()
    pp._retry.clear()
    return r


async def test_a_call_parks_valid_drafts_and_prunes_rejected_ones(catalog_db, providers, lab_runner):
    answer = good_batch(0, 18)
    answer["alphas"] += [{"expression": "rank(close)"}, {"expression": answer["alphas"][0]["expression"]}]
    fake = FakeProvider("fake", [answer])
    providers(fake)
    tid = await _task(lab_runner)
    await pp.write(tid)
    trials = await lab_runner.trials(tid)
    parked = [t for t in trials if t.state == "PARKED"]
    pruned = [t for t in trials if t.state == "PRUNED"]
    assert len(parked) == 18 and len(pruned) == 2
    assert {t.message for t in pruned} == {"Uses no field of ds1.", "Already written in this task."}
    assert all((t.params or {}).get("rejected") for t in pruned)
    task = await lab_runner.get(tid)
    tally = task.state["llm"]
    assert tally["calls"] == 1 and tally["empty"] == 0 and tally["byDataset"]["ds1"]["calls"] == 1
    assert tally["byModel"] == {"m1": 1} and task.state["calls"][-1]["valid"] == 18
    system, user = fake.prompts[0]
    assert system == pp.POWER_POOL_LAB and "YOUR EARLIER ALPHAS ON ds1\nNone yet." in user
    # the next call is shown what is waiting and what was thrown away
    fake.answers.append(good_batch(18, 2))
    await pp.write(tid)
    _, second = fake.prompts[1]
    assert "Not simulated yet:" in second and "Thrown away:" in second and "Uses no field of ds1." in second


async def test_three_empty_calls_stop_the_task(catalog_db, providers, lab_runner):
    providers(FakeProvider("fake", [{"alphas": [{"expression": "rank(close)"}]}] * 3))
    tid = await _task(lab_runner)
    for _ in range(3):
        await pp.write(tid)
    task = await lab_runner.get(tid)
    assert pp.stop_reason(task) == "the last 3 LLM calls wrote no new valid Alpha"
    ctx = await lab_runner._context(task)
    assert pp.PowerPoolLab().done_early(task, ctx) == "Stopped: the last 3 LLM calls wrote no new valid Alpha."
    assert not await pp.PowerPoolLab().busy(task)


async def test_call_cap(catalog_db, providers, lab_runner):
    assert pp.call_cap(40) == 4 and pp.call_cap(1) == 3 and pp.call_cap(1000) == 100
    providers(FakeProvider("fake", [good_batch(i * 3, 3) for i in range(4)]))
    tid = await _task(lab_runner, simulations=40)
    for _ in range(4):
        await pp.write(tid)
    task = await lab_runner.get(tid)
    assert pp.stop_reason(task) == "it reached 4 LLM calls"
    ctx = await lab_runner._context(task)
    assert pp.PowerPoolLab().done_early(task, ctx) is None      # parked drafts are still to be sent


async def test_a_busy_provider_moves_to_the_next_model(catalog_db, providers, lab_runner):
    first = FakeProvider("fake", [LLMError("Overloaded", 529, transient=True)])
    second = FakeProvider("spare", [good_batch()], model="m2")
    providers(first, second)
    tid = await _task(lab_runner)
    await pp.write(tid)
    task = await lab_runner.get(tid)
    assert task.state["llm"]["byModel"] == {"m2": 1}
    assert task.state["calls"][-1]["model"] == "spare:m2"


async def test_only_model_waits_and_three_failures_pause(catalog_db, providers, lab_runner):
    first = FakeProvider("fake", [LLMError("Overloaded", 529, transient=True)] +
                         [LLMError("Invalid API key", 401)] * 3)
    providers(first, FakeProvider("spare", [good_batch()], model="m2"))
    tid = await _task(lab_runner, only_model=True)
    await pp.write(tid)
    task = await lab_runner.get(tid)
    assert task.state["calls"][-1]["error"] == "Overloaded" and task.state["llm"].get("calls", 0) == 0
    for _ in range(3):
        await pp.write(tid)
    task = await lab_runner.get(tid)
    assert task.status == "PAUSED" and task.message.startswith("The LLM failed 3 times in a row")
    assert task.state["llm"]["failed"] == 0


async def test_budget_exhausted_waits_for_the_reset(catalog_db, providers, lab_runner):
    providers(FakeProvider("fake", [LLMError("Daily Fake call budget (400) reached")]))
    tid = await _task(lab_runner, only_model=True)
    await pp.write(tid)
    task = await lab_runner.get(tid)
    assert task.message.startswith("Waiting for LLM budget") and pp._retry[tid] > 0
    assert await lab_runner.trials(tid) == []


async def test_background_call_never_runs_inside_the_refill(catalog_db, providers, lab_runner, monkeypatch):
    gate = asyncio.Event()
    fake = FakeProvider("fake", [good_batch()])
    original = fake.generate_json

    async def slow(*a, **k):  # noqa: ANN002, ANN003, ANN202
        await gate.wait()
        return await original(*a, **k)

    fake.generate_json = slow
    providers(fake)
    monkeypatch.setattr(pp, "runner_hooks", lambda: True)
    tid = await _task(lab_runner)
    task = await lab_runner.get(tid)
    lab = lab_runner.lab("power_pool")
    assert await lab.refill(task, 20, RefillContext({}, [], 0, set())) == []
    assert pp.calling(tid) and await lab.busy(task)
    assert await lab.refill(task, 20, RefillContext({}, [], 0, set())) == []    # one call at a time
    gate.set()
    await pp._calls[tid]
    assert len(await lab_runner.trials(tid, ("PARKED",))) == 20


# ---------------------------------------------------------------- end to end through the runner

@pytest.fixture
async def brain_env(monkeypatch):
    real_sleep = asyncio.sleep

    async def fast(_s):  # noqa: ANN001, ANN202
        await real_sleep(0)

    monkeypatch.setattr(client_mod.asyncio, "sleep", fast)
    monkeypatch.setattr(sims_mod.asyncio, "sleep", fast)
    brain.client._credentials = lambda: ("me@example.com", "pw")
    from alphago.engine.simulator import SimulationService
    sim = SimulationService()
    monkeypatch.setattr("alphago.engine.simulator.simulator", sim)
    yield sim
    await sim.stop()
    brain.client.observers.clear()
    await brain.client.close()


async def test_power_pool_task_end_to_end(catalog_db, providers, lab_runner, brain_env):
    providers(FakeProvider("fake", [good_batch(0), good_batch(10)]))
    fake = FakeBrain(lambda e, st: (1.0 + (len(e or "") % 7) / 10, 1.1, 0.2))
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await brain_env.start()
        tid = await lab_runner.create("power_pool", params(simulations=20), cores=2, max_trials=20)
        await lab_runner.run(tid)
        for _ in range(200):
            await lab_runner.tick(tid)
            if pp.calling(tid):
                await pp._calls[tid]
            t = await lab_runner.get(tid)
            if t.status in ("COMPLETE", "FAILED"):
                break
            await asyncio.sleep(0.02)
    t = await lab_runner.get(tid)
    assert t.status == "COMPLETE", t.message
    summary = await lab_runner.summary(tid)
    assert summary["simulated"] == 20 and fake.posted == 20
    top = await lab_runner.top(tid)
    assert top and all(r["settings"]["truncation"] == 0.08 for r in top)
    assert t.state["llm"]["calls"] >= 1


# ---------------------------------------------------------------- Super Alpha Lab

def test_super_grid_and_operator_cap():
    found = sl.candidates(region="USA", delay=1, universe="TOP3000", neutralization="SUBINDUSTRY",
                          selection_limit=40, activations=["IS", "OS"])
    assert len(found) == 5 * 4 * 2 * 2
    assert len({c.label for c in found}) == len(found)
    assert all("(operator_count <= 200)" in r.code for r in sl.selections(40))
    assert all("(operator_count <= 800)" in r.code for r in sl.selections(5))       # cap uses max(limit, 10)
    s = found[0].settings
    assert (s["truncation"], s["selectionHandling"], s["selectionLimit"], s["simulationMode"]) == (
        0.05, "POSITIVE", 40, "FULL")
    assert {c.settings["decay"] for c in found} == {4, 10}
    assert {c.settings["componentActivation"] for c in found} == {"IS", "OS"}
    assert sl.activations_of("BOTH") == ["IS", "OS"] and sl.activations_of("OS") == ["OS"]


def test_rolling_ir_is_portable_without_ts_ir():
    with_ir = sl.combos_for([{"name": "ts_ir", "scope": ["COMBO"], "definition": "ts_ir(x, d)"}])
    regular_only = sl.combos_for([{"name": "ts_ir", "scope": ["REGULAR"], "definition": "ts_ir(x, d)"}])
    assert next(r for r in with_ir if r.name == "Rolling IR").code == sl.ROLLING_IR
    assert next(r for r in regular_only if r.name == "Rolling IR").code == sl.ROLLING_IR_PORTABLE
    assert next(r for r in sl.combos_for([]) if r.name == "Rolling IR").code == sl.ROLLING_IR_PORTABLE


async def test_super_recipe_rules(catalog_db, monkeypatch):
    base = {"region": "USA", "delay": 1, "universe": "TOP3000", "neutralization": "SUBINDUSTRY",
            "selection_limit": 30, "activation": "BOTH"}
    found, problems = await sl.build(base)
    assert problems == [] and len(found) == 80
    assert (await sl.build({**base, "region": "ALL"}))[1][0].startswith("SuperAlphas combine Alphas of one region")
    assert "Tick at least one selection." in (await sl.build({**base, "selections": []}))[1]
    dup = [{"name": "A", "code": "1"}, {"name": "A", "code": "2"}]
    assert "Two ticked combos share a name: rename one." in (await sl.build({**base, "combos": dup}))[1]
    assert any("Selection Limit" in p for p in (await sl.build({**base, "selection_limit": 9}))[1])
    assert any("Selection Limit" in p for p in (await sl.build({**base, "selection_limit": 30.5}))[1])
    summary = await sl.plan_summary({**base, "activation": "IS"})
    assert summary["simulations"] == 40 and summary["decays"] == [4, 10] and summary["truncation"] == 0.05
    assert next(r for r in summary["combos"] if r.name == "Rolling IR").code == sl.ROLLING_IR     # ts_ir is COMBO


async def test_user_recipes_are_kept_and_ticked(catalog_db):
    prefs = await sl.load_prefs()
    rid = sl.add_own(prefs, "selections", "Mine", "(turnover < 0.2)")
    prefs["off"] = ["builtin:Slow core", f"custom:{rid}"]
    await sl.save_prefs(prefs)
    again = await sl.load_prefs()
    assert again["selections"][0]["name"] == "Mine"
    picked = sl.chosen(sl.selections(30), again["selections"], again["off"])
    assert [r.name for r in picked] == ["Least correlated", "Tradeable turnover", "Broad books", "Many sources"]
    again["off"] = []
    assert sl.chosen(sl.selections(30), again["selections"], again["off"])[-1] == sl.Recipe("Mine",
                                                                                            "(turnover < 0.2)")


async def test_super_task_is_written_up_front_and_sent_alone(catalog_db, lab_runner, brain_env, monkeypatch):
    monkeypatch.setattr(sl, "runner_hooks", lambda: True)
    monkeypatch.setattr(sl, "_recovered", True)
    p = {"region": "USA", "delay": 1, "universe": "TOP3000", "neutralization": "SUBINDUSTRY", "selection_limit": 30,
         "activation": "IS", "selections": [{"name": "Least correlated", "code": "(1 - prod_correlation)"}],
         "combos": [{"name": "Equal weight", "code": "1"}, {"name": "Rolling", "code": sl.ROLLING_IR}]}
    tid = await lab_runner.create("super", p, cores=5, max_trials=999)
    task = await lab_runner.get(tid)
    assert task.cores == 3 and task.max_trials == 4 and task.space["count"] == 4
    parked = await lab_runner.trials(tid, ("PARKED",))
    assert [t.expression for t in parked][:2] == ["Least correlated · Equal weight · Decay 4 · IS",
                                                  "Least correlated · Rolling · Decay 4 · IS"]
    fake = FakeBrain(lambda e, st: (1.4, 1.2, 0.1))
    bodies: list = []
    real = fake.post_sim

    def spy(request):  # noqa: ANN001, ANN202
        bodies.append(json.loads(request.content))
        return real(request)

    fake.post_sim = spy
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        ctx = await lab_runner._context(task)
        assert await lab_runner.lab("super").send(task, parked, ctx) is True
        running = await lab_runner.trials(tid, ("RUNNING",))
        assert len(running) == 3 and len(await lab_runner.trials(tid, ("PARKED",))) == 1
        await asyncio.gather(*list(sl._inflight.values()))
        harvested = await lab_runner._harvest(task)
    assert len(harvested) == 3 and all(t.state == "COMPLETE" and t.value > 0 for t in harvested)
    assert all(isinstance(b, dict) and b["type"] == "SUPER" for b in bodies)      # one per request, never batched
    assert bodies[0]["selection"] == "(1 - prod_correlation)" and bodies[0]["combo"] == "1"
    assert bodies[0]["settings"]["selectionLimit"] == 30 and bodies[0]["settings"]["componentActivation"] == "IS"


async def test_super_needs_the_runner_send_hook(catalog_db, lab_runner, monkeypatch):
    monkeypatch.setattr(sl, "runner_hooks", lambda: False)
    p = {"region": "USA", "delay": 1, "universe": "TOP3000", "neutralization": "SUBINDUSTRY", "activation": "IS"}
    with pytest.raises(ValueError, match="cannot send SuperAlpha"):
        await lab_runner.create("super", p)


async def test_super_recovery_never_resends(catalog_db):
    from alphago.db.models import Candidate
    async with db.write() as s:
        s.add(Candidate(expression="x", settings={}, settings_hash="h", region="USA", delay=1, origin=sl.ORIGIN,
                        status=sl.SENDING))
    counts = await sl.recover()
    assert counts == {"resumed": 0, "closed": 1}
    async with db.session() as s:
        c = (await s.scalars(select(Candidate))).one()
    assert c.status == "error" and "not resent" in c.error


# ---------------------------------------------------------------- the tabs render

async def test_both_tabs_render(catalog_db, providers, lab_runner):
    from nicegui import ui
    from nicegui.testing.user_simulation import user_simulation

    from alphago.ui.pages.labs_llm import power_pool_tab, super_tab
    providers(FakeProvider("fake", []))
    await _task(lab_runner)

    async def root() -> None:
        with ui.column():
            await power_pool_tab()
            await super_tab()

    async with user_simulation(root=root) as user:
        await user.open("/")
        for text in ("LLM Power Pool Lab", "Favour operators unused this quarter", "Add to Tasks", "Run now",
                     "Super Alpha Lab", "What runs", "Least correlated", "Rolling IR", "Write your own selection",
                     "SuperAlpha simulations", "LLM calls"):
            await user.should_see(text)
        trials = await lab_runner.trials(1)
        assert trials == [] or all(isinstance(t, LabTrial) for t in trials)
