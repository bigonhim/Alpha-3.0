"""The Research Labs page builds every tab (NiceGUI's browserless user simulation)."""

from __future__ import annotations

import pytest
from nicegui import ui
from nicegui.testing.user_simulation import user_simulation

from alphago.db import db
from alphago.db.models import Alpha
from tests.test_labs_support import seed_catalogue


@pytest.fixture
async def pageenv(tmp_path, monkeypatch):
    from alphago import background
    monkeypatch.setattr(background, "autostart_enabled", lambda: False)
    await db.init(tmp_path / "ui.db")
    await seed_catalogue()
    async with db.write() as s:
        s.add(Alpha(id="UI1", code="rank(fnd1_a)", type="REGULAR", sharpe=1.6, fitness=1.2, turnover=0.2,
                    status="UNSUBMITTED", region="USA", delay=1, universe="TOP3000", neutralization="INDUSTRY"))
    import alphago.labs.labs  # noqa: F401  (registers the labs)
    yield
    await db.close()


@pytest.mark.parametrize("tab,texts", [
    ("Search", ["Search Lab", "Datasets", "Risk (FAST, SLOW, … RAM)", "Truncation Agent", "Add to Tasks"]),
    ("Template", ["Template Lab", "Built-in", "Two fields, two datasets", "Investability", "Save As"]),
    ("Evolution", ["Evolution Lab", "Auto Select", "Mutation rate", "Population"]),
    ("Tasks", ["Run All", "Lend idle cores", "Click a task to act on it."]),
    ("Submittable", ["Bounds", "Show implausible (|Sharpe| > 10)"]),
])
async def test_labs_tabs_build(pageenv, tab: str, texts: list[str]) -> None:
    from alphago.ui.layout import root
    async with user_simulation(root=root) as user:
        await user.open("/labs")
        await user.should_see("Research Labs")
        user.find(kind=ui.tab, content=tab).click()
        for text in texts:
            await user.should_see(text, retries=60)


async def test_template_preview_through_the_page(pageenv) -> None:
    from alphago.labs import template as template_lab
    plan = await template_lab.plan({"region": "USA", "delay": 1, "template": "rank($fnd1)",
                                    "variables": template_lab.resolve("rank($fnd1)", {}, [{"id": "fnd1"}]),
                                    "universes": ["TOP3000"], "neutralizations": ["INDUSTRY"],
                                    "investability": ["none"]})
    assert plan["problems"] == [] and plan["combinations"] == 3


async def test_search_handoff_prefills_and_clears(pageenv) -> None:
    from alphago.ui.layout import root
    await db.kv_set("handoff:search", {"region": "USA", "delay": 1, "universe": "TOP3000",
                                       "dataset_ids": ["fnd1"], "field_ids": ["fnd1_a", "fnd1_b"]})
    async with user_simulation(root=root) as user:
        await user.open("/labs")
        await user.should_see("Only 2 chosen fields", retries=60)
    assert await db.kv_get("handoff:search") is None


async def test_task_link_opens_its_results(pageenv) -> None:
    from alphago.db.models_labs import LabTask
    from alphago.labs import runner as runner_mod
    from alphago.labs.runner import TrialSpec
    from alphago.ui.layout import root
    async with db.write() as s:
        task = LabTask(lab="search", name="linked task", cores=1, params={}, max_trials=10, status="COMPLETE")
        s.add(task)
        await s.flush()
        tid = task.id
    await runner_mod.runner.add_trials(tid, [TrialSpec("rank(fnd1_a)", {"region": "USA", "universe": "TOP3000"},
                                                       state="COMPLETE", alpha_id="UI1", value=1.6)])
    async with user_simulation(root=root) as user:
        await user.open(f"/labs?task={tid}")
        await user.should_see("linked task — results", retries=60)
        await user.should_see("Checks Failed Together", retries=30)
        await user.should_see("Across Regions")
