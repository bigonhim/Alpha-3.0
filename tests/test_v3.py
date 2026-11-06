"""v3: field screening + family learning, SPC generate/verify/submit, Alpha Quality Project."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import Counter
from types import SimpleNamespace

import httpx
import pytest
import respx

from alphago.brain import brain
from alphago.config import SPCSettings, settings
from alphago.db import db
from alphago.db.models import Alpha, Candidate, ResearchTask
from alphago.engine import aqp, spc
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

def _rules(**kw) -> SPCSettings:
    return SPCSettings(stability_runs=2, min_overlap=0.3, **kw)


GOOD_PROMPT = ('Pick 3 liquid US stocks with rising estimates. Output ONLY a JSON array; each element has '
               '"company_name" and "ticker".')
RUN_A = '[{"company_name": "Apple", "ticker": "AAPL"}, {"company_name": "Microsoft", "ticker": "MSFT"}, ' \
        '{"company_name": "Nvidia", "ticker": "NVDA"}]'
RUN_B = '[{"company_name": "Apple", "ticker": "AAPL"}, {"company_name": "Microsoft", "ticker": "MSFT"}, ' \
        '{"company_name": "Amazon", "ticker": "AMZN"}]'


def test_spc_gates_block_until_everything_passes() -> None:
    r = _rules()
    assert not spc.gates_pass(spc.gates("", [], r))
    one = spc.gates(GOOD_PROMPT, [RUN_A], r)
    assert not spc.gates_pass(one) and any(g.name == "Stable across runs" and not g.ok for g in one)
    assert spc.gates_pass(spc.gates(GOOD_PROMPT, [RUN_A, RUN_B], r))
    unstable = '[{"company_name": "X", "ticker": "XOM"}, {"company_name": "Y", "ticker": "CVX"}]'
    assert not spc.gates_pass(spc.gates(GOOD_PROMPT, [RUN_A, unstable], r))
    assert not spc.gates_pass(spc.gates("x" * 20_000 + GOOD_PROMPT, [RUN_A, RUN_B], r))   # over the limit
    assert not spc.gates_pass(spc.gates(GOOD_PROMPT + " {{ idea }}", [RUN_A, RUN_B], r))  # placeholder left


def test_spc_body_template_is_filled_safely() -> None:
    r = _rules(competition_id="SPC9")
    body = spc.submission_body(r, name='Q "v1"', prompt='Line 1\n"quoted"', form={"model": "gpt", "weight": "0.2"})
    assert body["prompt"] == 'Line 1\n"quoted"' and body["name"] == 'Q "v1"' and body["model"] == "gpt"


@respx.mock
async def test_spc_submit_refuses_then_posts(monkeypatch) -> None:
    r = _rules(submit_path="/competitions/{competition}/predictions", competition_id="SPC9")
    with pytest.raises(ValueError, match="not passed"):
        await spc.submit(name="n", prompt=GOOD_PROMPT, form={}, rules=r, gate_results=spc.gates(GOOD_PROMPT, [], r))
    monkeypatch.setattr(brain.client, "state", "logged_in")
    monkeypatch.setattr(brain.client, "expiry_at", time.time() + 14400)
    route = respx.post(f"{API}/competitions/SPC9/predictions").mock(return_value=httpx.Response(201, json={"id": 7}))
    res = await spc.submit(name="n", prompt=GOOD_PROMPT, form={}, rules=r,
                           gate_results=spc.gates(GOOD_PROMPT, [RUN_A, RUN_B], r))
    assert res["status"] == 201 and route.called
    await brain.client.close()


async def test_spc_generate_and_dry_run_use_the_llm(monkeypatch) -> None:
    from alphago.llm import router
    from alphago.llm.schemas import SPCPromptDraft

    class FakeLLM:
        def configure(self, *_a):  # noqa: ANN002, ANN202
            return True

        async def generate_json(self, prompt, schema, **_k):  # noqa: ANN001, ANN003, ANN202
            assert "JSON array" in prompt and '"ticker"' in prompt
            return SPCPromptDraft(name="Revisions", hypothesis="h", prompt=GOOD_PROMPT)

        async def generate_text(self, prompt, **_k):  # noqa: ANN001, ANN003, ANN202
            return "```json\n" + RUN_A + "\n```"

    monkeypatch.setattr(router, "llm", FakeLLM())
    monkeypatch.setattr("alphago.llm.llm", router.llm)
    draft = await spc.generate_prompt("revisions", _rules())
    assert draft["prompt"] == GOOD_PROMPT
    outs = await spc.dry_run(GOOD_PROMPT, 2)
    assert spc.gates_pass(spc.gates(GOOD_PROMPT, outs, _rules()))


# ---------------------------------------------------------------- AQP

def test_cycle_tag_and_old_tags() -> None:
    from datetime import date
    assert aqp.cycle_tag(date(2026, 11, 6)) == "AQP202611"
    a = Alpha(id="A", tags=["AQP202610", "x"])
    assert aqp.tagged_before(a, "AQP202611") and not aqp.tagged_before(Alpha(id="B", tags=["AQP202611"]),
                                                                          "AQP202611")


async def _seed_aqp() -> None:
    async with db.write() as s:
        rows = [("U1", "momentum", 1.8, 0.2, None), ("U2", "momentum", 1.6, 0.3, None), ("U3", "value", 1.2, 0.4, 0.9),
                ("U4", "quality", 1.0, 0.1, None), ("OLD", "value", 3.0, 0.1, None)]
        for aid, cluster, sh, corr, os_sh in rows:
            s.add(Alpha(id=aid, status="ACTIVE", region="USA", type="REGULAR", sharpe=sh, self_corr=corr,
                        os_sharpe=os_sh, cluster=cluster, code=f"rank(ts_rank(f_{aid.lower()}, 252))",
                        fields=[f"f_{aid.lower()}"], tags=["AQP202610"] if aid == "OLD" else [],
                        settings={"neutralization": "SUBINDUSTRY"}))
        s.add(Alpha(id="EU", status="ACTIVE", region="EUR", type="REGULAR", sharpe=2.0, code="rank(close)"))


async def test_aqp_eligibility_and_diverse_pick(dbenv) -> None:
    await _seed_aqp()
    cands = await aqp.eligible("AQP202611")
    ids = {c.alpha.id for c in cands}
    assert ids == {"U1", "U2", "U3", "U4"}                       # no earlier-cycle alpha, no non-USA alpha
    picks = [c.alpha.id for c in aqp.pick(cands)]
    assert len(picks) == 3 and not {"U1", "U2"} <= set(picks)   # two momentum alphas are not both picked


def test_aqp_match_check_finds_what_is_not_described() -> None:
    facts = aqp.Facts(fields=["eps"], operators=["group_rank", "ts_rank", "multiply"], windows=[252],
                      groups=["subindustry"], field_notes={}, datasets=[])
    probs = aqp.match_check("eps ranked over time with ts_rank", facts, {"neutralization": "INDUSTRY"})
    assert "operator group_rank is not explained" in probs and "window 252 is not justified" in probs
    assert "grouping by subindustry is not explained" in probs and "industry neutralization is not mentioned" in probs
    assert not any("multiply" in p for p in probs) and not any("field eps" in p for p in probs)


async def test_aqp_write_rewrites_until_the_grade_stops_improving(dbenv, monkeypatch) -> None:
    from alphago.llm.schemas import AQPDescription, DescriptionGrade
    await _seed_aqp()
    calls = {"describe": 0, "grade": 0}

    class FakeLLM:
        def configure(self, *_a):  # noqa: ANN002, ANN202
            return True

        async def generate_json(self, prompt, schema, **_k):  # noqa: ANN001, ANN003, ANN202
            if schema is AQPDescription:
                calls["describe"] += 1
                better = calls["describe"] > 1
                return AQPDescription(title="Slow trend", hypothesis="h", mechanism="m",
                                      data="f_u1 measures earnings" if better else "data",
                                      construction="rank of ts_rank over 252 days, subindustry neutralization"
                                      if better else "ranked", novelty="n", risks="r", expected_behaviour="e")
            calls["grade"] += 1
            return DescriptionGrade(match=6 + 3 * (calls["grade"] > 1), uniqueness=7, hypothesis_quality=7,
                                    improvements=["name the field"])

    monkeypatch.setattr(aqp, "llm", FakeLLM())
    out = await aqp.write_description("U1")
    assert out["rounds"] >= 2 and out["problems"] == [] and "f_u1" in out["text"]
    assert (await aqp.draft("U1"))["text"] == out["text"]


async def test_aqp_apply_enforces_the_rules_and_patches(dbenv, monkeypatch) -> None:
    await _seed_aqp()
    sent = []

    async def patch(alpha_id, **kw):  # noqa: ANN001, ANN003, ANN202
        sent.append((alpha_id, kw))
        return {"tags": kw["tags"]}

    monkeypatch.setattr(brain.alphas, "patch", patch)
    desc = "Idea: " + "a detailed hypothesis. " * 15
    with pytest.raises(ValueError, match="earlier AQP cycle"):
        await aqp.apply("OLD", desc, "AQP202611")
    with pytest.raises(ValueError, match="detailed description"):
        await aqp.apply("U1", "short", "AQP202611")
    for aid in ("U1", "U3", "U4"):
        await aqp.apply(aid, desc, "AQP202611")
    with pytest.raises(ValueError, match="already has 3"):
        await aqp.apply("U2", desc, "AQP202611")
    assert sent[0] == ("U1", {"tags": ["AQP202611"], "regular_description": desc.strip()})
    async with db.session() as s:
        assert (await s.get(Alpha, "U1")).tags == ["AQP202611"]


# ---------------------------------------------------------------- SPC: import the endpoint from a captured request

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
    got = spc.parse_captured_request(CURL_BASH)
    assert got["path"] == "/competitions/{competition}/predictions" and got["method"] == "POST"
    assert got["competition_id"] == "SPC2026"
    tpl = json.loads(got["body_template"])
    assert tpl == {"name": "{name}", "prompt": "{prompt}", "model": "{model}", "version": "{version}",
                   "frequency": "{frequency}", "weight": "{weight:num}", "competition": "{competition}",
                   "draft": False}
    rules = SPCSettings(body_template=got["body_template"], competition_id=got["competition_id"])
    body = spc.submission_body(rules, name="N", prompt='say "hi"', form={"weight": "0.25"})
    assert body["weight"] == 0.25 and body["prompt"] == 'say "hi"' and body["competition"] == "SPC2026"


def test_spc_imports_a_captured_fetch_request() -> None:
    got = spc.parse_captured_request(FETCH)
    assert got["path"] == "/spc/submissions" and got["method"] == "POST" and got["competition_id"] == ""
    assert json.loads(got["body_template"]) == {"title": "{name}", "instruction": "{prompt}", "weight": "{weight}"}


def test_spc_import_rejects_requests_without_a_prompt() -> None:
    with pytest.raises(ValueError, match="no prompt"):
        spc.parse_captured_request("curl 'https://api.worldquantbrain.com/users/self' --data-raw '{\"a\": 1}'")
    with pytest.raises(ValueError, match="No URL"):
        spc.parse_captured_request("hello")


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
