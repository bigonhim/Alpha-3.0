"""BRAIN themes (GLB): theme matches, task settings and planning; Python alphas: support detection, local check, AI
fixes and the guarded simulation; the screening settings card."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from alphago.brain import brain
from alphago.config import ThemeDef, settings
from alphago.db import db
from alphago.db.models import Alpha, Dataset, ResearchTask
from alphago.engine import python_alpha as pa
from alphago.engine import themes


@pytest.fixture
async def database(tmp_path):
    await db.init(tmp_path / "themes.db")
    yield
    await db.close()


# ---------------------------------------------------------------- themes

def test_theme_hits_read_every_passing_matches_themes_check() -> None:
    a = Alpha(id="A", checks=[
        {"name": "MATCHES_THEMES", "result": "PASS", "themes": [{"name": "GLB Theme"}]},
        {"name": "MATCHES_THEMES", "result": "PASS", "themes": [{"name": "Sentiment"}]},     # an alpha can match two
        {"name": "MATCHES_THEMES", "result": "FAIL", "themes": [{"name": "Nope"}]},
        {"name": "LOW_SHARPE", "result": "PASS"}])
    assert themes.theme_hits(a) == ["GLB Theme", "Sentiment"]
    assert themes.theme_hits(Alpha(id="B", themes=[{"name": "X"}])) == ["X"]


def test_default_theme_is_glb_and_dates_limit_it() -> None:
    glb = settings.themes.items[0]
    assert glb.name == "GLB" and glb.region == "GLB" and glb.neutralization == "COUNTRY"
    t = ThemeDef(start="2026-11-01", end="2026-11-30")
    assert themes.is_active(t, date(2026, 11, 11)) and not themes.is_active(t, date(2026, 12, 1))
    assert not themes.is_active(ThemeDef(enabled=False))


def test_theme_tasks_use_the_theme_settings() -> None:
    glb = settings.themes.items[0]
    assert themes.overrides(themes.task_theme(glb), "GLB") == {"neutralization": "COUNTRY"}
    assert themes.overrides(themes.task_theme(glb), "USA") == {}          # wrong region: nothing forced
    assert themes.overrides("Fundamental", "GLB") == {}                    # an ordinary research focus


async def test_plan_task_picks_a_theme_dataset_and_waits_for_approval(database, monkeypatch) -> None:
    monkeypatch.setattr(settings.research, "mode", "assisted")
    synced = datetime.now(timezone.utc)
    async with db.write() as s:
        s.add_all([Dataset(id="glbsent", region="GLB", delay=1, universe="TOP3000", name="Sentiment",
                           themes=[{"name": "GLB"}], fields_synced_at=synced, field_count=40),
                   Dataset(id="other", region="GLB", delay=1, universe="TOP3000", name="Other", themes=None,
                           fields_synced_at=synced, field_count=40),
                   Dataset(id="nofields", region="GLB", delay=1, universe="TOP3000", themes=[{"name": "GLB"}])])
    glb = settings.themes.items[0]
    assert await themes.theme_dataset_ids(glb) == ["glbsent", "nofields"]
    tid = await themes.plan_task(glb)
    async with db.session() as s:
        t = await s.get(ResearchTask, tid)
    assert t.status == "planned" and t.dataset_ids == ["glbsent"] and t.theme == themes.task_theme(glb)
    assert t.region == "GLB" and t.planned_sims == settings.themes.task_sims and "GLB theme" in t.reason


async def test_plan_task_without_synced_data_says_what_to_do(database) -> None:
    with pytest.raises(RuntimeError, match="Sync theme data"):
        await themes.plan_task(settings.themes.items[0])


async def test_daily_theme_planning_only_when_switched_on(database, monkeypatch) -> None:
    monkeypatch.setattr(settings.themes, "auto_daily", False)
    assert await themes.maybe_plan_today() == []


# ---------------------------------------------------------------- Python alphas

GOOD = "import numpy as np\n\ndef alpha(data):\n    return -data['close'].pct_change(5)\n"


def test_local_check_blocks_unsafe_or_broken_code() -> None:
    assert pa.check_code(GOOD).ok
    assert not pa.check_code("def alpha(:\n").ok
    bad = pa.check_code("import os\n\ndef alpha(d):\n    eval('1')\n    return open('x')\n")
    assert not bad.ok and len(bad.errors) == 3
    assert not pa.check_code("").ok
    assert any("function" in w for w in pa.check_code("x = 1\n").warnings)


def _options(monkeypatch, langs: list[str], live: bool = True) -> None:
    monkeypatch.setattr(brain.options, "live", live)
    monkeypatch.setattr(brain.options, "global_choices", {**brain.options.global_choices, "language": langs})


def test_support_is_read_from_brains_allowed_settings(monkeypatch) -> None:
    _options(monkeypatch, ["FASTEXPR"])
    sup = pa.support()
    assert not sup["supported"] and "FASTEXPR" in sup["why"]
    _options(monkeypatch, ["FASTEXPR", "PYTHON"])
    assert pa.support()["supported"] and pa.python_language() == "PYTHON"
    _options(monkeypatch, ["FASTEXPR", "PYTHON"], live=False)
    assert not pa.support()["supported"]                                  # never guess before BRAIN answered


async def test_simulate_refuses_without_python_support(monkeypatch, database) -> None:
    _options(monkeypatch, ["FASTEXPR"])
    with pytest.raises(RuntimeError, match="cannot be simulated"):
        await pa.simulate(GOOD, {"region": "USA"})
    with pytest.raises(ValueError, match="Not simulated"):
        await pa.simulate("import os\n", {"region": "USA"})


async def test_simulate_sends_the_python_language_and_stores_the_alpha(monkeypatch, database) -> None:
    _options(monkeypatch, ["FASTEXPR", "PYTHON"])
    sent: list[dict] = []

    async def run(payloads):  # noqa: ANN001
        sent.extend(payloads)
        return [SimpleNamespace(status="COMPLETE", alpha_id="PY1", message=None)]

    async def get(aid: str) -> dict:
        return {"id": aid, "type": "REGULAR", "settings": {"region": "USA", "delay": 1, "language": "PYTHON"},
                "regular": {"code": GOOD}, "is": {"sharpe": 1.3, "fitness": 1.0, "turnover": 0.2}, "status": "UNSUBMITTED"}

    monkeypatch.setattr(brain.sims, "run", run)
    monkeypatch.setattr(brain.alphas, "get", get)
    entry = await pa.simulate(GOOD, {"region": "USA", "delay": 1, "universe": "TOP3000"}, name="Rev")
    assert sent[0]["settings"]["language"] == "PYTHON" and sent[0]["regular"] == GOOD
    assert entry["alpha_id"] == "PY1" and (await db.kv_get(pa.HISTORY_KEY))[0]["name"] == "Rev"
    async with db.session() as s:
        assert await s.get(Alpha, "PY1") is not None


async def test_ai_code_that_fails_the_check_is_fixed(monkeypatch) -> None:
    calls: list[str] = []

    class FakeLLM:
        def configure(self, *a, **k) -> bool:  # noqa: ANN002, ANN003
            return True

        async def generate_json(self, prompt, schema, **kw):  # noqa: ANN001, ANN003
            calls.append(prompt)
            code = "```python\nimport os\ndef alpha(d):\n    return d\n```" if len(calls) == 1 else GOOD
            return pa.PythonAlphaDraft(name="Reversal", hypothesis="h", code=code)

    monkeypatch.setattr(pa, "llm", FakeLLM())
    out = await pa.write_code("short-term reversal")
    assert out["check"]["ok"] and out["rounds"] == 2 and "import of os" in calls[1] and not out["verified"]
    assert "```" not in out["code"]


# ---------------------------------------------------------------- screening defaults

def test_every_screen_is_on_by_default() -> None:
    from alphago.config import LLMSettings, ResearchSettings
    r = ResearchSettings()
    assert r.field_screening and r.duplicate_filter and r.cluster_share > 0 and LLMSettings().self_check
