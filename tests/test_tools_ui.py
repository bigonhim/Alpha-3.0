"""The /tools page builds, shows every tool, refuses an expression in the Alpha ID box, and previews a paste."""

from __future__ import annotations

import pytest
from nicegui import ui
from nicegui.testing.user_simulation import user_simulation

from alphago.brain import brain
from alphago.brain.options import SimOptions
from alphago.db import db
from alphago.db.models import Alpha, DataField


@pytest.fixture
async def dbenv(tmp_path, monkeypatch):
    from alphago import background
    monkeypatch.setattr(background, "autostart_enabled", lambda: False)
    monkeypatch.setattr(brain, "options", SimOptions())
    await db.init(tmp_path / "ui_tools.db")
    async with db.write() as s:
        s.add(Alpha(id="TU1", code="rank(close)", status="ACTIVE", region="USA", delay=1, sharpe=1.5,
                    settings={"region": "USA", "delay": 1, "universe": "TOP3000", "decay": 2,
                              "neutralization": "INDUSTRY", "truncation": 0.08}))
        s.add(DataField(id="close", dataset_id="pv1", region="USA", delay=1, universe="TOP3000", type="MATRIX"))
    yield
    await db.close()


async def test_tools_page_shows_every_tool(dbenv) -> None:
    from alphago.ui.layout import root
    async with user_simulation(root=root) as user:
        await user.open("/tools")
        for text in ("Settings Sampler", "Correlation Breaker", "Fields Sampler", "Paste Alphas", "Copy & Describe",
                     "Tasks of this tool", "Override the held settings"):
            await user.should_see(text)


async def test_settings_sampler_analyses_an_alpha_and_refuses_an_expression(dbenv) -> None:
    from alphago.ui.layout import root
    async with user_simulation(root=root) as user:
        await user.open("/tools")
        user.find(marker="alpha-settings").type("rank(close)")
        await user.should_see("looks like an expression, not an Alpha ID")
        user.find(marker="alpha-settings").clear().type("TU1")
        user.find(marker="analyse-settings").click()
        await user.should_see("Where to run it", retries=20)
        await user.should_see("Preview (free: nothing is simulated)")
        await user.should_see("Run now")


async def test_paste_preview_reads_the_text(dbenv) -> None:
    from alphago.ui.layout import root
    async with user_simulation(root=root) as user:
        await user.open("/tools")
        user.find("Paste Alphas").elements.pop()
        area = next(e for e in user.find(kind=ui.textarea).elements
                    if getattr(e, "_props", {}).get("label") == "The LLM's output")
        area.set_value("Alpha Expression:\nrank(close)\nRegion: USA\nDelay: 1\nUniverse: TOP3000\n")
        user.find("Read it").click()
        await user.should_see("Alphas read", retries=20)
        await user.should_see("Ready")


async def test_breaker_and_fields_panels_render_their_plans(dbenv) -> None:
    from alphago.db.models import Operator
    from alphago.ui.layout import root
    async with db.write() as s:
        s.add(DataField(id="open", dataset_id="pv1", region="USA", delay=1, universe="TOP3000", type="MATRIX",
                        coverage=0.9))
        for name in ("rank", "vector_neut", "log", "ts_av_diff"):
            s.add(Operator(name=name))
    async with user_simulation(root=root) as user:
        await user.open("/tools")
        for tab in ("breaker", "fields"):
            user.find(marker=f"alpha-{tab}").type("TU1")
            user.find(marker=f"analyse-{tab}").click()
        await user.should_see("Re-shapes (ranked by what worked on this account)", retries=20)
        await user.should_see("Trade the innovation, not the level")
        await user.should_see("Your own re-shapes (kept on this computer)")
        await user.should_see("Its fields and what each could become", retries=20)
        await user.should_see("First sweep", retries=20)
