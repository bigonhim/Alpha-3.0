"""Paste Alphas: the Markdown-tolerant parser on realistic LLM outputs, the resolver against BRAIN's options, batch
ordering, and an end-to-end paste task (plus copy-back) against the fake BRAIN through the lab runner."""

from __future__ import annotations

import asyncio

import pytest
import respx

from alphago.brain import brain
from alphago.brain import client as client_mod
from alphago.brain import simulations as sims_mod
from alphago.brain.options import SimOptions
from alphago.db import db
from alphago.labs import runner as runner_mod
from alphago.labs.runner import LabRunner
from alphago.tools import paste_alphas as paste
from alphago.tools.register import register
from tests.fakebrain import FakeBrain


@pytest.fixture(autouse=True)
def fallback_options(monkeypatch):
    monkeypatch.setattr(brain, "options", SimOptions())


COPY_FORMAT = """\
Here are two ideas.

Alpha Expression:
rank(-ts_delta(close, 5))
Simulation Settings:
Region: USA
Universe: TOP3000
Language: Fast Expression
Decay: 4
Delay: 1
Truncation: 0.08
Neutralization: Subindustry
Pasteurization: On
NaN Handling: Off
Max Trade: Off
Max Position: Off

---

Alpha Expression:
group_rank(ts_zscore(volume, 20), industry)
Simulation Settings:
Region: USA
Universe: TOP1000
Delay: 1
Neutralization: INDUSTRY
"""

MARKDOWN = """\
### Alpha 1: Short-term reversal

**Alpha Expression:**
```
rank(-returns)
```

**Simulation Settings:**
- **Region:** `USA`
- **Universe:** `TOP3000`
- **Delay:** `1`
- **Decay:** `6`
- **Neutralization:** `SUBINDUSTRY`

Rationale: reversal works best with Decay: 10 in volatile names.

### Alpha 2: Volume surprise

| Setting | Value |
|---|---|
| Region | USA |
| Universe | TOP500 |
| Delay | 0 |
| Truncation | 5% |

```fastexpr
a = ts_mean(volume, 5);
rank(a / adv20)
```
"""

INLINE = """\
1. Alpha Expression: rank(ts_rank(close, 20))
   Region: USA, Delay: 1, Universe: TOP3000, Decay: 3
2. Alpha Expression: -rank(ts_std_dev(returns, 20))
   Region = USA; Universe = TOP3000; Delay = 1; maxTrade = ON
"""


def test_copy_format_reads_every_setting() -> None:
    drafts = paste.parse(COPY_FORMAT)
    assert [d["expression"] for d in drafts] == ["rank(-ts_delta(close, 5))",
                                                 "group_rank(ts_zscore(volume, 20), industry)"]
    first = drafts[0]["given"]
    assert first["region"] == "USA" and first["decay"] == "4" and first["neutralization"] == "Subindustry"
    assert first["nanHandling"] == "Off" and first["language"] == "Fast Expression"
    assert drafts[1]["given"] == {"region": "USA", "universe": "TOP1000", "delay": "1", "neutralization": "INDUSTRY"}


def test_markdown_bold_fences_tables_and_passing_mentions() -> None:
    drafts = paste.parse(MARKDOWN)
    assert len(drafts) == 2
    assert drafts[0]["expression"] == "rank(-returns)"
    # "Decay: 10" said in passing in a rationale is not a setting; the first value written stands
    assert drafts[0]["given"]["decay"] == "6" and drafts[0]["given"]["universe"] == "TOP3000"
    # settings written above the code (a table) wait for it and are its own
    assert drafts[1]["expression"] == "a = ts_mean(volume, 5);\nrank(a / adv20)"
    assert drafts[1]["given"] == {"region": "USA", "universe": "TOP500", "delay": "0", "truncation": "5%"}


def test_inline_settings_and_numbered_lists() -> None:
    drafts = paste.parse(INLINE)
    assert [d["expression"] for d in drafts] == ["rank(ts_rank(close, 20))", "-rank(ts_std_dev(returns, 20))"]
    assert drafts[0]["given"] == {"region": "USA", "delay": "1", "universe": "TOP3000", "decay": "3"}
    assert drafts[1]["given"]["maxTrade"] == "ON"


def test_an_example_code_block_without_settings_is_left_out() -> None:
    text = "Remember the syntax:\n```\nrank(x)\n```\n\nAlpha Expression:\nrank(close)\nRegion: USA\nDelay: 1\n" \
           "Universe: TOP3000\n"
    assert [d["expression"] for d in paste.parse(text)] == ["rank(close)"]


def test_resolver_requires_the_market_and_defaults_the_rest() -> None:
    ok = paste.resolve({"region": "usa", "delay": "1", "universe": "top3000", "neutralization": "Slow Factors"})
    assert ok["problems"] == []
    s = ok["settings"]
    assert (s["region"], s["delay"], s["universe"], s["neutralization"]) == ("USA", 1, "TOP3000", "SLOW")
    assert s["decay"] == 0 and s["truncation"] == 0.08 and "Decay" in ok["defaulted"]
    assert s["maxTrade"] == "OFF"
    missing = paste.resolve({"universe": "TOP3000"})
    assert "No Region given." in missing["problems"] and "No Delay given." in missing["problems"]


def test_resolver_errors_are_plain() -> None:
    bad = paste.resolve({"region": "USA", "delay": "1", "universe": "TOP9999", "decay": "600", "truncation": "2",
                         "testPeriod": "two years"})
    text = " ".join(bad["problems"])
    assert "Universe 'TOP9999' is not offered in USA delay 1. BRAIN offers TOP3000" in text
    assert "Decay '600' is not a whole number from 0 to 512." in text
    assert "Truncation '2' is not a number from 0 to 1." in text
    assert "Test Period 'two years' is not like P2Y0M0D." in text
    both = paste.resolve({"region": "USA", "delay": "1", "universe": "TOP3000", "maxTrade": "On",
                          "maxPosition": "ON"})
    assert "Max Trade and Max Position cannot both be On." in both["problems"]
    nowhere = paste.resolve({"region": "MARS", "delay": "1", "universe": "TOP3000"})
    assert nowhere["problems"][0].startswith("Region 'MARS' is not offered. BRAIN offers USA")


def test_max_position_left_out_where_brain_does_not_offer_it() -> None:
    chn = paste.resolve({"region": "CHN", "delay": "1", "universe": "TOP2000U"})
    assert chn["problems"] == [] and "maxPosition" not in chn["settings"]
    usa = paste.resolve({"region": "USA", "delay": "1", "universe": "TOP3000"})
    assert usa["settings"]["maxPosition"] == "OFF"


async def test_plan_dedups_and_orders_full_batches_first(tmp_path) -> None:
    await db.init(tmp_path / "p.db")
    try:
        blocks = [f"Alpha Expression:\nrank(ts_delta(close, {n}))\nRegion: USA\nDelay: 1\nUniverse: TOP3000\n"
                  for n in range(1, 13)]
        blocks += ["Alpha Expression:\nrank(ts_delta(close, 1))\nRegion: USA\nDelay: 1\nUniverse: TOP3000\n"]
        blocks += [f"Alpha Expression:\nrank(volume / {n})\nRegion: CHN\nDelay: 1\nUniverse: TOP2000U\n"
                   for n in range(1, 11)]
        found = await paste.plan("\n---\n".join(blocks))
        assert found["problems"] == []
        assert found["alphas"][12]["duplicate_of"] == 1
        assert found["simulations"] == 22 and found["batches"] == 3
        regions = [r["settings"]["region"] for r in found["requests"]]
        # the two full tens first (USA's ten and CHN's ten), the USA tail of two last
        assert regions[-2:] == ["USA", "USA"] and regions[:20].count("CHN") == 10
        empty = await paste.plan("Nothing here")
        assert empty["problems"] == ['No Alpha found. Each one starts with a line reading "Alpha Expression:".']
    finally:
        await db.close()


@pytest.fixture
async def env(tmp_path, monkeypatch):
    real_sleep = asyncio.sleep

    async def fast(_s):  # noqa: ANN001, ANN202
        await real_sleep(0)

    monkeypatch.setattr(client_mod.asyncio, "sleep", fast)
    monkeypatch.setattr(sims_mod.asyncio, "sleep", fast)
    await db.init(tmp_path / "paste.db")
    brain.client._credentials = lambda: ("me@example.com", "pw")
    from alphago.engine.simulator import SimulationService
    sim = SimulationService()
    monkeypatch.setattr("alphago.engine.simulator.simulator", sim)
    r = LabRunner()
    register(r)
    monkeypatch.setattr(runner_mod, "runner", r)
    yield r, sim
    await sim.stop()
    brain.client.observers.clear()
    await brain.client.close()
    await db.close()


async def _drive(r: LabRunner, tid: int, rounds: int = 120) -> None:
    for _ in range(rounds):
        await r.tick(tid)
        t = await r.get(tid)
        if t.status in ("COMPLETE", "FAILED"):
            return
        await asyncio.sleep(0.02)


async def test_paste_task_runs_end_to_end_and_copies_back(env) -> None:
    r, sim = env
    fake = FakeBrain(lambda e, st: (1.6 if "close" in e else 0.4, 1.2, 0.2))
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await sim.start()
        tid = await paste.add_paste_task(COPY_FORMAT, cores=1, run=True)
        await _drive(r, tid)
        t = await r.get(tid)
        assert t.status == "COMPLETE", t.message
        assert t.name == "Paste Alphas · 2 Alphas"
        summary = await r.summary(tid)
        assert summary["simulated"] == 2 and fake.posted == 2
        sent = [a["settings"] for a in fake.alphas.values()]
        assert {s["universe"] for s in sent} == {"TOP3000", "TOP1000"}
        assert any(s["decay"] == 4 and s["neutralization"] == "SUBINDUSTRY" for s in sent)
        copied = await paste.copy_results(tid)
    assert copied["alphas"] == 2 and copied["failed"] == 0 and copied["waiting"] == 0
    text = copied["text"]
    assert text.index("rank(-ts_delta(close, 5))") < text.index("group_rank(")       # highest Sharpe first
    assert "Neutralization: Subindustry" in text and "Metric Glossary" not in text
    # the copy reads back as the same two Alphas
    again = await paste.plan(text)
    assert again["problems"] == [] and again["simulations"] == 2


async def test_flip_sign_queues_one_full_mode_negation(env) -> None:
    r, sim = env
    from alphago.db.models import Alpha
    async with db.write() as s:
        s.add(Alpha(id="SRC1", code="x = rank(close); ts_mean(x, 5)", status="UNSUBMITTED",
                    settings={"region": "USA", "delay": 1, "universe": "TOP3000", "decay": 2,
                              "neutralization": "INDUSTRY", "truncation": 0.05}))
    fake = FakeBrain(lambda e, st: (1.3, 1.1, 0.2))
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await sim.start()
        tid = await paste.flip_sign("SRC1")
        await _drive(r, tid)
    t = await r.get(tid)
    assert t.name == "Flip Sign · SRC1" and t.status == "COMPLETE", t.message
    (alpha,) = fake.alphas.values()
    assert alpha["regular"]["code"] == "x = rank(close); -ts_mean(x, 5)"
    assert alpha["settings"]["simulationMode"] == "FULL" and alpha["settings"]["decay"] == 2


async def test_fields_sampler_grows_a_tree_from_winners(env) -> None:
    r, sim = env

    from alphago.db.models import Alpha, DataField
    from alphago.tools import fields_sampler as fs
    settings = {"region": "USA", "delay": 1, "universe": "TOP3000", "decay": 0, "neutralization": "INDUSTRY",
                "truncation": 0.08}
    async with db.write() as s:
        for name in ("a", "b", "c", "d"):
            s.add(DataField(id=f"fnd_{name}", dataset_id="fnd6", region="USA", delay=1, universe="TOP3000",
                            type="MATRIX", description=f"Field {name}", coverage=0.9))
        s.add(Alpha(id="SEED1", code="rank(fnd_a) - rank(fnd_b)", status="ACTIVE", settings=settings))
    fake = FakeBrain(lambda e, st: (1.6 if "fnd_c" in e else 0.5, 1.2, 0.2))
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await sim.start()
        tid = await r.create("fields_sampler", {"alpha_id": "SEED1", "mode": "sweep"}, cores=1, run=True)
        await _drive(r, tid, rounds=200)
    t = await r.get(tid)
    assert t.status == "COMPLETE", t.message
    trials = await r.trials(tid)
    grown = [x for x in trials if x.generation == 2]
    assert grown and all((x.params or {}).get("parent") for x in grown)
    parents = {x.number: x for x in trials}
    assert all("fnd_c" in parents[x.params["parent"]].expression for x in grown)    # only winners grow
    assert len({x.expression for x in trials}) == len(trials)                       # nothing tried twice
    assert "rank(fnd_a) - rank(fnd_b)" not in {x.expression for x in trials}       # the seed is not re-run
    nodes = fs.tree(trials)
    assert any(n["children"] for n in nodes)


async def test_correlation_breaker_task_and_its_history(env) -> None:
    r, sim = env
    from sqlalchemy import update

    from alphago.db.models import Alpha, DataField, Operator
    from alphago.tools import correlation_breaker as cb
    settings = {"region": "USA", "delay": 1, "universe": "TOP3000", "decay": 3, "neutralization": "SECTOR",
                "truncation": 0.06}
    async with db.write() as s:
        for f in ("close", "cap", "returns"):
            s.add(DataField(id=f, dataset_id="pv1", region="USA", delay=1, universe="TOP3000", type="MATRIX"))
        for name in ("rank", "ts_regression", "log", "vector_neut", "ts_delta", "ts_av_diff"):
            s.add(Operator(name=name))
        s.add(Alpha(id="PC1", code="rank(close)", status="UNSUBMITTED", settings=settings,
                    checks=[{"name": "PROD_CORRELATION", "result": "FAIL", "value": 0.8}]))
    fake = FakeBrain(lambda e, st: (1.5, 1.2, 0.2))
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await sim.start()
        tid = await r.create("correlation_breaker", {"alpha_id": "PC1", "recipes": ["neut_size", "fast_innovation"]},
                             cores=1, name="Correlation Breaker · PC1", run=True)
        await _drive(r, tid)
    assert (await r.get(tid)).status == "COMPLETE", (await r.get(tid)).message
    sent = list(fake.alphas.values())
    assert len(sent) == 2 and all(a["settings"]["simulationMode"] == "FULL" and a["settings"]["decay"] == 3
                                  for a in sent)
    assert {a["regular"]["code"] for a in sent} == {"alpha = rank(close);\nvector_neut(alpha, log(cap))",
                                                    "alpha = rank(close);\nts_av_diff(alpha, 20)"}
    done = [t for t in await r.trials(tid) if t.alpha_id]
    worked = next(t.alpha_id for t in done if t.params["recipe"] == "neut_size")
    async with db.write() as s:
        await s.execute(update(Alpha).where(Alpha.id.in_([t.alpha_id for t in done])).values(
            checks=[{"name": "PROD_CORRELATION", "result": "FAIL"}]))
        await s.execute(update(Alpha).where(Alpha.id == worked).values(
            checks=[{"name": "PROD_CORRELATION", "result": "PASS"}, {"name": "LOW_SHARPE", "result": "PASS"}]))
    assert await cb.history() == {"neut_size": (1, 1), "fast_innovation": (1, 0)}
