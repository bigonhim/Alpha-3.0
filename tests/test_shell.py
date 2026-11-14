"""The shell: diagnostics scrubbing, the command palette's filtering, the core tiles, pause reasons, the sidebar's
sections and the Live Simulations page with a running simulation."""

from __future__ import annotations

import zipfile
from datetime import datetime, timezone
from pathlib import PurePath, PureWindowsPath

import pytest
from nicegui import ui
from nicegui.testing.user_simulation import user_simulation

from alphago import diagnostics
from alphago.brain import brain
from alphago.db import db
from alphago.db.models import Candidate
from alphago.db.models_labs import LabTask
from alphago.ui import layout
from alphago.ui.layout import SECTIONS, alpha_id_of, palette_entries
from alphago.ui.pages import live

# ── diagnostics ──────────────────────────────────────────────────────────────────────────────


def test_scrubber_removes_home_user_and_ids() -> None:
    windows = diagnostics.scrubber(PureWindowsPath(r"C:\Users\Wanjiru"), "Wanjiru")
    line = (r"opened C:\Users\Wanjiru\AppData\Local\AlphaGo and c:/users/wanjiru/x, "
            r"'C:\\Users\\WANJIRU\\y' for wanjiru, inquiry=inq_AbC123 alpha_id=QPKp9zVr "
            r"GET /alphas/QPKp9zVr")
    out = windows(line)
    assert "Wanjiru" not in out and "wanjiru" not in out.lower().replace("<user>", "")
    assert out.count("~") == 3 and r"~\AppData\Local\AlphaGo" in out
    assert "for <user>," in out
    assert "inq_AbC123" not in out and "inq#" in out
    assert "QPKp9zVr" not in out
    hashes = {m.group("id") for m in diagnostics._ALPHA.finditer(out)}
    assert len(hashes) == 1                                  # the same Alpha hashes the same


def test_scrubber_posix_home_and_short_username() -> None:
    posix = diagnostics.scrubber(PurePath("/home/wanjiru"), "wanjiru")
    assert posix("/home/wanjiru/.local/share/AlphaGo") == "~/.local/share/AlphaGo"
    assert diagnostics.scrubber(PurePath("/home/al"), "al")("also al") == "also al"


def test_export_writes_about_and_the_scrubbed_log_tail(tmp_path, monkeypatch) -> None:
    from alphago import config
    logs = tmp_path / "logs"
    logs.mkdir()
    home = PurePath(str(tmp_path / "home" / "Zed"))
    (logs / "alphago.log").write_text("x" * 100 + f"\nopened {home}\\data alpha_id=ABCDE123 by zedd\n",
                                      encoding="utf-8")
    monkeypatch.setattr(config, "log_dir", lambda: logs)
    monkeypatch.setattr(diagnostics.Path, "home", staticmethod(lambda: diagnostics.Path(str(home))))
    monkeypatch.setattr(diagnostics, "_user", lambda: "zedd")
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    target = diagnostics.export(downloads)
    assert target.parent == downloads and target.name.startswith("alphago-diagnostics-")
    with zipfile.ZipFile(target) as z:
        about = z.read("about.txt").decode()
        log = z.read("alphago.log").decode()
    assert "AlphaGo" in about and "Python:" in about and "System:" in about
    assert "ABCDE123" not in log and "alpha#" in log and "<user>" in log and "~" in log


def test_tail_keeps_only_the_last_four_megabytes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(diagnostics, "TAIL_BYTES", 10)
    (tmp_path / "alphago.log").write_bytes(b"0123456789abcdefghij")
    assert diagnostics._tail(tmp_path) == "abcdefghij"


# ── palette ──────────────────────────────────────────────────────────────────────────────────

SCREENS = [("/", "Dashboard"), ("/live", "Live Simulations"), ("/data", "Data Explorer"), ("/sync", "Sync with BRAIN"),
           ("/alphas", "Alphas")]


def test_palette_alpha_id_opens_that_alpha() -> None:
    entries = palette_entries("QPKp9zVr", [], SCREENS)
    assert entries[0].kind == "alpha" and entries[0].target == "QPKp9zVr"
    assert alpha_id_of("https://platform.worldquantbrain.com/alpha/AbC1234") == "AbC1234"
    assert palette_entries("https://platform.worldquantbrain.com/alpha/AbC1234", [], SCREENS)[0].target == "AbC1234"
    assert all(e.kind != "alpha" for e in palette_entries("sync", [], SCREENS))     # lower-case word: not an id


def test_palette_offers_a_data_field_search_for_any_text() -> None:
    entries = palette_entries("close price", [], SCREENS)
    search = [e for e in entries if e.label.startswith("Search Data Fields")]
    assert search and search[0].target == "/data?q=close%20price"
    assert search[0].label == "Search Data Fields for “close price”"


def test_palette_screens_match_synonyms_and_tasks_by_name() -> None:
    entries = palette_entries("login", [], SCREENS)
    assert any(e.target == "/sync" for e in entries)                # "login" is a synonym of Sync with BRAIN
    tasks = [("Task 7 · Momentum hunt", "/tasks?task=7&kind=lab"), ("Task 8 · Value", "/tasks?task=8&kind=lab")]
    found = palette_entries("momentum", tasks, SCREENS)
    assert [e.label for e in found if e.group == "Tasks"] == ["Task 7 · Momentum hunt"]
    assert not any(e.group == "Screens" and e.target == "/sync" for e in found)
    everything = palette_entries("", tasks, SCREENS)
    assert {e.group for e in everything} == {"Actions", "Tasks", "Screens"}
    assert everything[0].group == "Actions"


def test_palette_ranks_label_prefix_first() -> None:
    entries = [e for e in palette_entries("data", [], SCREENS) if e.group == "Screens"]
    assert entries[0].target == "/data"


def test_every_screen_is_in_exactly_one_sidebar_section() -> None:
    paths = [p for _, ps in SECTIONS for p in ps]
    assert len(paths) == len(set(paths)) == 32


# ── cores and pauses ─────────────────────────────────────────────────────────────────────────


def test_core_blocks_glb_holds_two_cores() -> None:
    t = datetime(2026, 11, 1, tzinfo=timezone.utc)
    running = {"S1": {"region": "USA", "started": t, "progress": 0.5, "ids": [1]},
               "S2": {"region": "GLB", "started": t.replace(minute=1), "progress": None, "ids": [2]}}
    cores = live.core_blocks(running, 4)
    assert [(b.start, b.span, b.sim_id) for b in cores.blocks] == [(0, 1, "S1"), (1, 2, "S2"), (3, 1, None)]
    assert cores.blocks[1].label == "C2–3"
    assert [live.core_state(b, False) for b in cores.blocks] == ["running", "pending", "idle"]
    assert all(live.core_state(b, True) == "warning" for b in cores.blocks)
    assert live.core_blocks(running, 2).overflow == 1


def test_pauses_most_pressing_first(monkeypatch) -> None:
    from alphago.engine.simulator import simulator
    monkeypatch.setattr(brain.client, "state", "persona_required")
    monkeypatch.setattr(simulator, "paused", True)
    monkeypatch.setattr(simulator, "pause_reason", "quota: DAILY_SIMULATION_LIMIT_EXCEEDED")
    keys = [p.key for p in live.pauses()]
    assert keys[:2] == ["identity", "limit"]
    monkeypatch.setattr(brain.client, "state", "logged_out")
    monkeypatch.setattr(simulator, "pause_reason", None)
    monkeypatch.setattr(simulator, "paused", False)
    assert live.pauses()[0].chip == "Signed out"


# ── UI ───────────────────────────────────────────────────────────────────────────────────────


async def _stub_page() -> None:
    ui.label("stub screen")


@pytest.fixture
async def shell(tmp_path, monkeypatch):
    """The real shell with only the screens these tests need (other screens are built by other modules)."""
    await db.init(tmp_path / "shell.db")
    items = [layout.NavItem("/", "Dashboard", "space_dashboard", "#fff", _stub_page),
             layout.NavItem("/live", "Live Simulations", "memory", "#fff", live.page),
             layout.NavItem("/alphas", "Alphas", "layers", "#fff", _stub_page),
             layout.NavItem("/sync", "Sync with BRAIN", "sync", "#fff", _stub_page)]
    monkeypatch.setattr(layout, "_nav", lambda: items)
    yield
    from alphago.engine.simulator import simulator
    simulator.running.clear()
    await db.close()


async def test_live_page_shows_a_running_simulation(shell, monkeypatch) -> None:
    from alphago.engine.simulator import simulator
    async with db.write() as s:
        s.add(Candidate(id=11, expression="rank(close)", settings={"region": "USA", "delay": 1, "universe": "TOP3000",
                                                                    "neutralization": "SUBINDUSTRY"},
                        settings_hash="h", region="USA", delay=1, status="simulating", task_id=-3))
        s.add(LabTask(id=3, lab="search", name="Momentum hunt"))
    simulator.running["SIMabcdef1234"] = {"ids": [11], "region": "USA", "progress": 0.42, "task": -3,
                                          "started": datetime.now(timezone.utc), "size": 1,
                                          "expressions": ["rank(close)"]}
    async with user_simulation(root=layout.root) as user:
        await user.open("/live")
        await user.should_see("Live Simulations")
        await user.should_see("Task 3 · Momentum hunt")
        await user.should_see("SIMabcde…")
        await user.should_see("42%")
        await user.should_see("IDLE")
        await user.should_see("queued")


async def test_shell_has_sections_stop_all_diagnostics_and_palette(shell) -> None:
    async with user_simulation(root=layout.root) as user:
        await user.open("/")
        await user.should_see("Overview")
        await user.should_see("Stop all")
        await user.should_see("Export diagnostics")
        await user.should_see("Simulations Left Today")
        await user.should_see("Signed out")                     # the pause chip: no BRAIN session in tests
        await user.should_see("Live Simulations")


async def test_palette_opens_and_lists_tasks(shell) -> None:
    async with db.write() as s:
        s.add(LabTask(id=5, lab="template", name="Value template"))
    async with user_simulation(root=layout.root) as user:
        await user.open("/")
        with user.client:
            opener = layout._palette(layout._nav())
            await opener()
        await user.should_see("Task 5 · Value template")
        await user.should_see("Export diagnostics")
