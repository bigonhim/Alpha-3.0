"""The new and changed pages build and show their new sections (NiceGUI's browserless user simulation)."""

from __future__ import annotations

import pytest
from nicegui.testing.user_simulation import user_simulation

from alphago.db import db
from alphago.db.models import Alpha

PAGES = [
    ("/health", ["AI provider errors per day", "Which repairs work", "Self-correlation estimate vs BRAIN"]),
    ("/audit", ["Read-only record", "Simulations, last 30 days"]),
    ("/planner", ["Suggested split", "How to read it"]),
    ("/llm", ["Fallback providers", "AI self-check before simulating", "Model tournament (ideation)",
              "Versions — every save is kept"]),
    ("/sync", ["Email alerts", "Run in the background", "Start with Windows"]),
    ("/oos", ["What predicts a gap in your alphas"]),
    ("/autopilot", ["Robustness test"]),
    ("/alphas", ["BRAIN check", "Robust"]),
    ("/datasets", ["Dataset Intelligence"]),
    ("/", ["Submittable Alphas"]),
]


@pytest.fixture
async def dbenv(tmp_path, monkeypatch):
    from alphago import background
    monkeypatch.setattr(background, "autostart_enabled", lambda: False)     # no schtasks call from a test
    await db.init(tmp_path / "ui.db")
    async with db.write() as s:
        s.add(Alpha(id="UI1", code="rank(close)", sharpe=1.6, fitness=1.2, turnover=0.2, verdict="ready",
                    status="UNSUBMITTED", check_state="pass", robust_state="fragile", robust_ratio=0.72,
                    region="USA", delay=1, est_self_corr=0.42, cluster="reversal"))
    yield
    await db.close()


@pytest.mark.parametrize("path,texts", PAGES, ids=[p for p, _ in PAGES])
async def test_page_shows_its_sections(dbenv, path: str, texts: list[str]) -> None:
    from alphago.ui.layout import root
    async with user_simulation(root=root) as user:
        await user.open(path)
        for text in texts:
            await user.should_see(text)


async def test_alpha_dialog_shows_the_fragile_badge(dbenv) -> None:
    from nicegui import ui

    from alphago.ui.alpha_detail import open_alpha
    from alphago.ui.layout import root
    async with user_simulation(root=root) as user:
        await user.open("/alphas")
        with user.client:
            await open_alpha("UI1")
        await user.should_see("FRAGILE 72%")
        await user.should_see("EST. SELF-CORR 0.42")
        await user.should_see("BRAIN CHECK PASSED")
        assert ui is not None
