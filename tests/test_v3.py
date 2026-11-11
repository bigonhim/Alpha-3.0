"""v3: field screening + family learning, and captured SPC request import (SPC/AQP: test_spc_auto, test_aqp)."""

from __future__ import annotations

import asyncio
import json
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from alphago.brain import spc as brain_spc
from alphago.config import SPCSettings, settings
from alphago.db import db
from alphago.db.models import Alpha, Candidate, ResearchTask
from alphago.engine.families import family_stats, thompson_order
from alphago.knowledge.templates import (
    FAST_WINDOWS,
    SLOW_WINDOWS,
    enumerate_single,
    probe_expression,
    windows_for_category,
    wrap_field,
)

API = "https://api.worldquantbrain.com"


@pytest.fixture
async def dbenv(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_write_lock", asyncio.Lock())
    await db.init(tmp_path / "v3.db")
    yield
    await db.close()


# ---------------------------------------------------------------- templates, screening, families

def test_signed_fields_and_category_windows() -> None:
    assert wrap_field("-eps") == "-winsorize(ts_backfill(eps, 120), std=4)"
    assert wrap_field("est", "VECTOR") == "winsorize(ts_backfill(vec_avg(est), 120), std=4)"
    assert windows_for_category("Fundamental") == SLOW_WINDOWS and windows_for_category("Price Volume") == FAST_WINDOWS
    probe = probe_expression("eps", "MATRIX", "Fundamental")
    assert probe == "ts_rank(winsorize(ts_backfill(eps, 120), std=4), 252)"
    # the probe is the first thing the search would simulate anyway (no wasted simulation)
    assert next(enumerate_single([("eps", "MATRIX")], windows=SLOW_WINDOWS))[0] == probe


def test_family_order_follows_the_record() -> None:
    stats = {"good": (40, 50), "bad": (0, 50)}
    rng = random.Random(1)
    firsts = Counter(thompson_order(["bad", "good", "new"], stats, rng)[0] for _ in range(200))
    assert firsts["good"] > 150 and firsts["bad"] == 0


async def test_family_stats_count_hits(dbenv) -> None:
    async with db.write() as s:
        for i, (fam, verdict) in enumerate([("ts_rank", "strong"), ("ts_rank", "rejected"), ("ts_zscore", "weak")]):
            s.add(Alpha(id=f"F{i}", verdict=verdict, region="USA", delay=1))
            s.add(Candidate(expression=f"x{i}", settings={}, settings_hash="h", region="USA", delay=1,
                            rationale={"family": fam}, alpha_id=f"F{i}"))
    assert await family_stats("USA", 1) == {"ts_rank": (1, 2), "ts_zscore": (0, 1)}


async def test_screening_keeps_signal_fields_and_flips_negative_ones(dbenv, monkeypatch) -> None:
    from alphago.engine import research as research_mod
    sharpe = {"good": 1.1, "neg": -0.9, "noise": 0.05, "weak": 0.2, "meh": 0.3, "flat": 0.0}
    async with db.write() as s:
        t = ResearchTask(kind="single", region="USA", delay=1, universe="TOP3000", dataset_ids=["ds"],
                         planned_sims=400, cores=2, status="running")
        s.add(t)
        await s.flush()
        tid = t.id

    async def prepare(specs, **_kw):  # noqa: ANN001, ANN003
        created = []
        async with db.write() as s:
            for sp in specs:
                fid = sp.rationale["field"]
                s.add(Alpha(id=f"P{fid}", sharpe=sharpe[fid], fitness=sharpe[fid] * 0.8, region="USA", delay=1))
                c = Candidate(expression=sp.expression, settings={}, settings_hash="h", region="USA", delay=1,
                              task_id=tid, alpha_id=f"P{fid}", rationale=sp.rationale)
                s.add(c)
                await s.flush()
                created.append(SimpleNamespace(id=c.id))
        return SimpleNamespace(created=created, left_out={})

    async def nothing(*_a, **_k):  # noqa: ANN002, ANN003
        return None

    monkeypatch.setattr(research_mod.simulator, "prepare", prepare)
    monkeypatch.setattr(research_mod.simulator, "enqueue", nothing)
    monkeypatch.setattr(research_mod.simulator, "wait_for", nothing)
    mgr = research_mod.ResearchManager()
    monkeypatch.setattr(mgr, "_refresh", nothing)
    async with db.session() as s:
        task = await s.get(ResearchTask, tid)
    fields = {"ds": [(f, "MATRIX") for f in sharpe]}
    done: set[str] = set()
    out, probed, ids = await mgr._screen_fields(task, fields, {"region": "USA"}, "Fundamental", done, 400,
                                                Counter(), 0)
    assert probed == 6 and len(ids) == 6 and len(done) == 6
    kept = [f for f, _ in out["ds"]]
    assert kept[:2] == ["good", "-neg"]                         # strongest first; the negative one sign-flipped
    assert "noise" not in kept[:2] and len(kept) == settings.research.screen_keep_min
    # a resumed task reads the probes back instead of simulating them again
    out2, probed2, _ = await mgr._screen_fields(task, fields, {}, "Fundamental", done, 400, Counter(), 0)
    assert probed2 == 0 and out2 == out


# ---------------------------------------------------------------- SPC

CURL_BASH = r"""curl 'https://api.worldquantbrain.com/competitions/SPC2026/predictions' \
  -H 'accept: application/json;version=2.0' \
  -H 'content-type: application/json' \
  --data-raw '{"name":"My prompt","prompt":"Pick 20 stocks. It'\''s \"systematic\".","model":"gpt","version":"5.6","frequency":"Daily","weight":0.2,"competition":"SPC2026","draft":false}'"""  # noqa: E501  (a verbatim browser capture)

FETCH = """fetch("https://api.worldquantbrain.com/spc/submissions", {
  "headers": {"content-type": "application/json"},
  "body": "{\\"title\\":\\"T\\",\\"instruction\\":\\"Pick stocks\\",\\"weight\\":\\"0.20\\"}",
  "method": "POST"
});"""


def test_spc_imports_a_captured_curl_request() -> None:
    got = brain_spc.parse_captured_request(CURL_BASH)
    assert got["path"] == "/competitions/{competition}/predictions" and got["method"] == "POST"
    assert got["competition_id"] == "SPC2026"
    tpl = json.loads(got["submit_body"])
    assert tpl == {"name": "{name}", "prompt": "{prompt}", "model": "{model}", "version": "{version}",
                   "frequency": "{frequency}", "weight": "{weight:num}", "competition": "{competition}",
                   "draft": False}
    rules = SPCSettings(submit_path=got["path"], submit_body=got["submit_body"], competition_id=got["competition_id"])
    req = brain_spc.build_request(rules, name="N", prompt='say "hi" {model}', form={"weight": "0.25"})
    body = req.body
    assert req.path == "/competitions/SPC2026/predictions"
    assert body["weight"] == 0.25 and body["prompt"] == 'say "hi" {model}' and body["competition"] == "SPC2026"


def test_spc_imports_a_captured_fetch_request() -> None:
    got = brain_spc.parse_captured_request(FETCH)
    assert got["path"] == "/spc/submissions" and got["method"] == "POST" and got["competition_id"] == ""
    assert json.loads(got["submit_body"]) == {"title": "{name}", "instruction": "{prompt}", "weight": "{weight}"}


def test_spc_import_rejects_requests_without_a_prompt() -> None:
    with pytest.raises(ValueError, match="no prompt"):
        brain_spc.parse_captured_request("curl 'https://api.worldquantbrain.com/users/self' --data-raw '{\"a\": 1}'")
    with pytest.raises(ValueError, match="No URL"):
        brain_spc.parse_captured_request("hello")


# ---------------------------------------------------------------- Autopilot reuses field screening

async def test_screened_fields_steer_later_template_ideas(dbenv, monkeypatch) -> None:
    from alphago.db.models import DataField
    from alphago.engine import generators
    scope = {"region": "USA", "delay": 1, "universe": "TOP3000"}
    async with db.write() as s:
        for f in ("good", "neg", "noise", "ok"):
            s.add(DataField(id=f, dataset_id="ds", region="USA", delay=1, universe="TOP3000", type="MATRIX",
                            coverage=0.9))
        for f, sh in (("good", 1.2), ("neg", -0.8), ("noise", 0.1), ("ok", 0.6)):
            s.add(Alpha(id=f"S{f}", sharpe=sh, fitness=abs(sh) * 0.7, region="USA", delay=1))
            s.add(Candidate(expression=f"probe {f}", settings={}, settings_hash="h", region="USA", delay=1,
                            rationale={"family": "ts_rank", "probe": True, "field": f}, alpha_id=f"S{f}"))
    assert await generators.screened_fields(scope, "ds") == [("good", "MATRIX"), ("-neg", "MATRIX"), ("ok", "MATRIX")]

    async def no_catalog(*_a):  # noqa: ANN002, ANN202
        from alphago.fastexpr import default_catalog
        return default_catalog()

    monkeypatch.setattr(generators, "get_catalog", no_catalog)
    monkeypatch.setattr(generators, "validate", lambda expr, cat: SimpleNamespace(ok=True))
    specs = await generators.template_ideas(scope, dataset_id="ds", n=20, rng=random.Random(3), families=None)
    used = " ".join(s.expression for s in specs)
    assert "noise" not in used and ("good" in used or "neg" in used)
