"""Search, Template and Evolution tasks end-to-end against the fake BRAIN, through the lab runner; the TPE study
rebuilt after a restart."""

from __future__ import annotations

import respx
from sqlalchemy import select

from alphago.brain import brain
from alphago.db import db
from alphago.db.models import Alpha
from alphago.db.models_labs import LabTrial
from alphago.labs import results as results_mod
from alphago.labs import search as search_mod
from alphago.labs import template as template_mod
from tests import test_labs_support as support
from tests.fakebrain import FakeBrain
from tests.test_labs_support import drive, score

labenv = support.labenv          # the shared fixture

SEARCH = {"region": "USA", "delay": 1, "universe": "TOP3000", "dataset_ids": ["fnd1", "anl1"],
          "vector_operators": ["vec_avg"], "neutralizations": ["INDUSTRY", "SUBINDUSTRY"], "decays": [0, 5],
          "cores": 1}
TEMPLATE = {"region": "USA", "delay": 1, "template": "group_rank($ts_op($field, $lookback), $group)",
            "variables": {"field": {"kind": "fields", "dataset_ids": ["fnd1"]},
                          "ts_op": {"kind": "values", "values": "ts_rank, ts_zscore"},
                          "lookback": {"kind": "values", "values": "5, 21, 63"},
                          "group": {"kind": "values", "values": "sector, industry"}},
            "universes": ["TOP3000", "TOP1000"], "neutralizations": ["MARKET"], "investability": ["none"],
            "cores": 1}


async def test_search_preview_and_plan(labenv):
    r, _ = labenv
    plan = await search_mod.plan(SEARCH)
    assert plan["problems"] == [], plan["problems"]
    assert plan["fields"] == {"total": 4, "matrix": 3, "vector": 1}
    assert plan["universes"] == ["TOP3000", "TOP1000"] and plan["round"] == 10
    assert sorted(plan["space"]["absent"]["TOP1000"]) == ["anl1_v", "fnd1_c"]
    assert plan["space"]["backfill"]["fnd1_a"] == 63 and plan["space"]["backfill"]["anl1_v"] == 63
    assert len(plan["sample"]) == 5 and all(s["settings"]["testPeriod"] == "P2Y0M0D" for s in plan["sample"])
    no_vector = await search_mod.plan({**SEARCH, "vector_operators": []})
    assert no_vector["left_out"]["vector"] == 1 and "vector fields are left out" in no_vector["warnings"][0]
    refused = await search_mod.plan({**SEARCH, "neutralizations": [], "dataset_ids": []})
    assert "Choose at least one Neutralization." in refused["problems"]
    assert "Choose at least one dataset." in refused["problems"]


async def test_search_task_runs_and_rebuilds_after_restart(labenv):
    r, sim = labenv
    fake = FakeBrain(score)
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await sim.start()
        tid = await r.create("search", SEARCH, cores=1, max_trials=20)
        t = await r.get(tid)
        assert t.objective == "train_sharpe" and t.params["n_startup_trials"] == 10
        await r.run(tid)
        await drive(r, tid)
    t = await r.get(tid)
    assert t.status == "COMPLETE", t.message
    trials = await r.trials(tid)
    sent = [x for x in trials if x.state == "COMPLETE"]
    assert len(sent) == 20 and fake.posted == 20
    # the first pass gave every field one trial
    assert {x.params["field"] for x in sent} >= {"fnd1_a", "fnd1_b", "fnd1_c", "anl1_v"}
    assert all(x.expression.count("ts_backfill(") >= 1 for x in sent)
    assert all(x.params.get("_asked") for x in sent)
    # a restart: the study comes back from the rows alone, with every scored trial replayed
    lab = r.labs["search"]
    lab.forget(tid)
    run = await lab.driver.run_for(t)
    replayed = run.study.get_trials(deepcopy=False)
    assert len(replayed) == len(sent)
    assert {round(x.values[0], 6) for x in replayed} == {round(x.value, 6) for x in sent}
    res = await results_mod.results(tid)
    assert res["rows"] and res["regions"] and res["check_sets"]


async def test_template_task_end_to_end(labenv):
    r, sim = labenv
    plan = await template_mod.plan(TEMPLATE)
    assert plan["problems"] == [], plan["problems"]
    # 2 universes x 1 x 1 x fields(3) x 2 x 3 x 2; fnd1_c is absent in TOP1000 but still a choice of the variable
    assert plan["combinations"] == 2 * 3 * 2 * 3 * 2
    assert plan["stats"]["operators"] == [2, 2] and plan["stats"]["single_dataset"] is True
    assert all(s["expression"].startswith("group_rank(ts_") for s in plan["sample"])
    fake = FakeBrain(score)
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await sim.start()
        tid = await r.create("template", TEMPLATE, cores=1, max_trials=10, run=True)
        await drive(r, tid)
    t = await r.get(tid)
    assert t.status == "COMPLETE", t.message
    done = await r.trials(tid, ("COMPLETE",))
    assert len(done) == 10
    assert all(tr.settings["maxTrade"] == "OFF" and tr.settings["testPeriod"] == "P2Y0M0D" for tr in done)


async def test_template_plan_refusals(labenv):
    bad = await template_mod.plan({**TEMPLATE, "template": "rank($x) + nope(close)", "variables": {}})
    assert "nope is not one of your operators." in bad["template_problems"]
    assert any(p.startswith("$x: Define it") for p in bad["problems"])
    lacking = await template_mod.plan({**TEMPLATE, "template": "rank(fnd1_c - $field)"})
    assert any("TOP1000 is left out" in w for w in lacking["warnings"])


async def test_evolution_breeds_from_seeds(labenv):
    r, sim = labenv
    async with db.write() as s:
        for i, (code, fit) in enumerate([("rank(ts_mean(fnd1_a, 21))", 1.4), ("zscore(ts_delta(fnd1_b, 63))", 1.2),
                                         ("rank(fnd1_a - fnd1_b)", 0.9)]):
            s.add(Alpha(id=f"S{i}", code=code, type="REGULAR", region="USA", delay=1, universe="TOP3000",
                        neutralization="INDUSTRY", decay=3, truncation=0.08, status="UNSUBMITTED", fitness=fit,
                        sharpe=fit * 1.3, turnover=0.2, checks=[{"name": "LOW_SHARPE", "result": "PASS",
                                                                 "limit": 1.25}]))
    from alphago.labs import ga
    plan = await ga.plan({"region": "USA", "delay": 1, "universe": "TOP3000", "alpha_ids": ["S0", "S1", "S2", "X"],
                          "neutralizations": ["INDUSTRY", "SECTOR"], "simulations": 20})
    assert plan["problems"] == [], plan["problems"]
    assert [s["alpha_id"] for s in plan["seeds"]][0] == "S0"
    assert plan["skipped"] == [{"alpha_id": "X", "reason": "Not stored here. Sync from BRAIN first."}]
    assert plan["population"] == 20 and plan["sample"]
    fake = FakeBrain(score)
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        await brain.client.login("me@example.com", "pw")
        await sim.start()
        tid = await r.create("evolution", {"region": "USA", "delay": 1, "universe": "TOP3000",
                                           "alpha_ids": ["S0", "S1", "S2"], "neutralizations": ["INDUSTRY", "SECTOR"]},
                             cores=1, max_trials=10, run=True)
        await drive(r, tid)
    t = await r.get(tid)
    assert t.status == "COMPLETE", t.message
    async with db.session() as s:
        rows = list(await s.scalars(select(LabTrial).where(LabTrial.task_id == tid)))
    seeds = [x for x in rows if x.generation == 0]
    children = [x for x in rows if x.generation > 0 and x.state == "COMPLETE"]
    assert len(seeds) == 3 and all(x.free for x in seeds)
    assert len(children) == 10 and all(x.settings["testPeriod"] == "P2Y0M0D" for x in children)
    assert (await r.summary(tid))["simulated"] == 10
    assert t.objective == "train_fitness"


async def test_saved_templates_are_unique_whatever_the_case(labenv):
    from alphago.labs import templates_gallery as tg
    first = await tg.create("Momentum", "rank($field)", {"field": {"kind": "fields", "dataset_ids": ["fnd1"]}})
    import pytest
    with pytest.raises(ValueError, match="already exists"):
        await tg.create("momentum", "rank(close)")
    other = await tg.create("Reversal", "-rank(ts_delta(close, $lookback))")
    with pytest.raises(ValueError, match="already exists"):
        await tg.update(other["id"], "MOMENTUM", "x")
    renamed = await tg.update(first["id"], "momentum", "rank($field) * 2", first["variables"])  # its own name
    assert renamed["name"] == "momentum" and renamed["variables"]["field"]["dataset_ids"] == ["fnd1"]
    assert [t["name"] for t in await tg.saved()][0] in ("momentum", "Reversal")
    await tg.delete(other["id"])
    assert [t["name"] for t in await tg.saved()] == ["momentum"]
    stats = await template_mod.stats([{"text": f.text, "variables": {}} for f in tg.GALLERY], "USA", 1)
    assert stats[0]["holes"] == 2 and stats[2]["missing"] == []


async def test_auto_select_rejects_correlated_seeds(labenv):
    r, sim = labenv
    async with db.write() as s:
        for i in range(4):
            s.add(Alpha(id=f"FB0000{i + 1}", code=f"rank(ts_mean(fnd1_a, {5 + i}))", type="REGULAR", region="USA",
                        delay=1, universe="TOP3000", status="UNSUBMITTED", fitness=1.0 + i / 10, sharpe=1.5 + i / 10,
                        turnover=0.2))
    from alphago.labs import ga
    fake = FakeBrain(score)
    for i in range(4):          # the fake answers PnL for alphas it knows; its PnL is seeded by the id
        fake.alphas[f"FB0000{i + 1}"] = {"is": {"sharpe": 1.5}}
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        router.get(url__regex=r".*/alphas/\w+/recordsets/\w+$").mock(side_effect=fake.pnl)
        await brain.client.login("me@example.com", "pw")
        market = await ga.market_for("USA", 1, "TOP3000", ["INDUSTRY"])
        found = await ga.auto_seeds(market, "USA", 1, "TOP3000", 3)
    assert found["examined"] <= 4 and found["pool"] == 4
    assert len(found["seeds"]) + len(found["reasons"]) == found["examined"]
