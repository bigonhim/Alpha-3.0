"""SPC generation and submission: gates, the rewrite loop, and the gated, once-only submit (BRAIN mocked with respx)."""

from __future__ import annotations

import json
from datetime import date

import httpx
import pandas as pd
import pytest
import respx
from sqlalchemy import select

from alphago.brain import brain
from alphago.brain import spc as brain_spc
from alphago.config import SPCSettings
from alphago.db import db
from alphago.db.models import SPCPrompt, SPCRun
from alphago.engine import audit, spc, spc_auto
from alphago.engine.spc_auto import Brief, PromptDraft, RunOutcome

API = "https://api.worldquantbrain.com"
AUTH_OK = {"user": {"id": "AB12345"}, "token": {"expiry": 14400.0}, "permissions": ["CONSULTANT"]}
GOOD_PROMPT = 'Pick stocks by a ranked rule. Reply ONLY with a JSON array of {"company_name": "...", "ticker": "..."}.'


def rows(*tickers: str) -> str:
    return json.dumps([{"company_name": f"{t} Inc.", "ticker": t} for t in tickers])


class FakeLLM:
    """Round 1 prompt: every run returns different stocks. A rewritten prompt (marked STABLE) returns the same ones."""

    def __init__(self, *, rewrite_fixes: bool = True) -> None:
        self.rewrite_fixes = rewrite_fixes
        self.n = 0
        self.revise_feedback: list[str] = []

    async def generate_json(self, prompt: str, schema, **kw):  # noqa: ANN001, ANN003
        if "WHAT FAILED" in prompt:
            self.revise_feedback.append(prompt.split("WHAT FAILED (measured):")[1])
            tag = " STABLE" if self.rewrite_fixes else ""
            return PromptDraft(name="Momentum v2", prompt=GOOD_PROMPT + tag, rationale="fixed")
        return PromptDraft(name="Momentum", prompt=GOOD_PROMPT, rationale="first")

    async def generate_text(self, prompt: str, **kw) -> str:  # noqa: ANN003
        if "STABLE" in prompt:
            return rows("AAA", "BBB", "CCC", "DDD")
        self.n += 1
        return rows(f"X{self.n}A", f"X{self.n}B", f"X{self.n}C")


@pytest.fixture
def fake_prices(monkeypatch):
    async def closes(tickers, start, end):  # noqa: ANN001
        idx = pd.bdate_range("2026-10-20", periods=5)
        return pd.DataFrame({t: [10.0, 10.1, 10.2, 10.3, 10.4] for t in tickers if t != "BAD"}, index=idx)

    monkeypatch.setattr(spc, "fetch_closes", closes)


@pytest.fixture
async def database(tmp_path):
    await db.init(tmp_path / "spc_auto.db")
    yield
    await db.close()


# ---------------------------------------------------------------- request building

def test_build_request_substitutes_placeholders_without_breaking_json() -> None:
    rules = SPCSettings(submit_path="/spc/entries",
                        submit_body='{"p": "{prompt}", "meta": {"m": "{model}", "n": ["{name}"]}}')
    nasty = 'Say "hi"\n{ticker} \\ and {prompt} {model}'
    req = brain_spc.build_request(rules, name="N 1", prompt=nasty, form={"model": "gpt"})
    assert req.method == "POST" and req.path == "/spc/entries"
    assert req.body == {"p": nasty, "meta": {"m": "gpt", "n": ["N 1"]}}     # text inside the prompt is never substituted


def test_competition_and_numeric_placeholders() -> None:
    rules = SPCSettings(submit_path="/competitions/{competition}/predictions", competition_id="SPC1",
                        submit_body='{"prompt": "{prompt}", "weight": "{weight:num}", "c": "{competition}"}')
    req = brain_spc.build_request(rules, name="n", prompt="p", form={"weight": "0.25"})
    assert req.path == "/competitions/SPC1/predictions" and req.body == {"prompt": "p", "weight": 0.25, "c": "SPC1"}
    assert brain_spc.validate_endpoint(SPCSettings(submit_path="/competitions/{competition}/x"))   # id missing


@pytest.mark.parametrize("path, method, body, expect", [
    ("", "POST", "{}", "No submit path"),
    ("https://evil.example/x", "POST", "{}", "relative"),
    ("//evil.example/x", "POST", "{}", "relative"),
    ("spc", "POST", "{}", "relative"),
    ("/spc", "DELETE", "{}", "method"),
    ("/spc", "POST", "[1]", "JSON object"),
    ("/spc", "POST", "{nope", "not valid JSON"),
    ("/spc", "POST", '{"a": "b"}', "{prompt}"),
])
def test_endpoint_validation_blocks_unsafe_or_unset_endpoints(path, method, body, expect) -> None:
    problems = brain_spc.validate_endpoint(SPCSettings(submit_path=path, submit_method=method, submit_body=body))
    assert problems and expect in problems[0]
    with pytest.raises(ValueError):
        brain_spc.build_request(SPCSettings(submit_path=path, submit_method=method, submit_body=body), name="n", prompt="p")


def test_default_settings_send_nothing() -> None:
    assert brain_spc.validate_endpoint(SPCSettings())          # no path: submission is not set up


def test_old_settings_files_migrate() -> None:
    old = SPCSettings.model_validate({"body_template": '{"prompt": "{prompt}"}', "min_overlap": 0.3,
                                      "stability_runs": 4})
    assert old.submit_body == '{"prompt": "{prompt}"}' and old.auto_min_overlap == 0.3 and old.auto_runs == 4


# ---------------------------------------------------------------- run parsing and gates

def test_parse_run_cuts_the_array_out_of_surrounding_text() -> None:
    rules = SPCSettings()
    run = spc_auto.parse_run('Sure! Here you go:\n[{"company_name": "A", "ticker": "AAA"}]\nHope it helps.', rules)
    assert run.ok and run.extracted and run.tickers == {"AAA"}
    assert not spc_auto.parse_run("no json here", rules).ok
    assert not spc_auto.parse_run('[{"ticker": "AAA"}]', rules).ok       # company_name missing


def _runs(*ticker_sets: tuple[str, ...]) -> list[RunOutcome]:
    return [spc_auto.parse_run(rows(*t), SPCSettings()) for t in ticker_sets]


def test_gates_pass_and_block() -> None:
    rules = SPCSettings()
    good = spc_auto.evaluate_gates(GOOD_PROMPT, rules, _runs(("A", "B", "C"), ("A", "B", "C")),
                                   {"n": 3, "unpriced": []}, None)
    assert spc_auto.ready(good) and {g.name: g.status for g in good}["Backtest"] == "off"

    unstable = spc_auto.evaluate_gates(GOOD_PROMPT, rules, _runs(("A", "B"), ("C", "D")), {"n": 4, "unpriced": []}, None)
    assert not spc_auto.ready(unstable) and next(g for g in unstable if g.name == "Stability").status == "fail"

    unpriced = spc_auto.evaluate_gates(GOOD_PROMPT, rules, _runs(("A", "B"), ("A", "B")), {"n": 2, "unpriced": ["B"]}, None)
    assert next(g for g in unpriced if g.name == "Tickers priced").status == "fail"

    # a check that could not run is not a pass
    unknown = spc_auto.evaluate_gates(GOOD_PROMPT, rules, _runs(("A",), ("A",)), {"error": "Yahoo down"}, None)
    assert next(g for g in unknown if g.name == "Tickers priced").status == "error" and not spc_auto.ready(unknown)

    too_long = spc_auto.evaluate_gates("x" * 11_000, rules, _runs(("A",), ("A",)), {"n": 1, "unpriced": []}, None)
    assert next(g for g in too_long if g.name == "Prompt length").status == "fail"

    bad_form = spc_auto.evaluate_gates(GOOD_PROMPT, rules, [spc_auto.parse_run("[]", SPCSettings(min_rows=1))] * 2,
                                       None, None)
    assert next(g for g in bad_form if g.name == "Form rules").status == "fail"

    left = spc_auto.evaluate_gates(GOOD_PROMPT + " [your universe here]", rules, _runs(("A",), ("A",)),
                                   {"n": 1, "unpriced": []}, None)
    assert next(g for g in left if g.name == "No placeholders").status == "fail" and not spc_auto.ready(left)


def test_failure_feedback_carries_the_evidence() -> None:
    rules = SPCSettings()
    runs = _runs(("A", "B", "C"), ("A", "B", "D"))
    gates = spc_auto.evaluate_gates(GOOD_PROMPT, SPCSettings(auto_min_overlap=0.9), runs, {"n": 4, "unpriced": ["D"]}, None)
    text = spc_auto.failure_feedback(gates, runs, rules)
    assert "Stability" in text and "C, D" in text and "Tickers priced" in text and "BRK-B" in text


def test_output_spec_follows_the_form_rules() -> None:
    spec = spc_auto.output_spec(SPCSettings(side_field="side", weight_field="weight"))
    assert '"side": "long" or "short"' in spec and '"weight"' in spec and '"ticker"' in spec


# ---------------------------------------------------------------- pipeline

async def test_failed_gate_triggers_a_targeted_rewrite_that_passes(monkeypatch, database, fake_prices) -> None:
    fake = FakeLLM()
    monkeypatch.setattr(spc_auto, "llm", fake)
    seen: list[str] = []

    async def progress(frac: float, msg: str) -> None:
        seen.append(msg)

    report = await spc_auto.run_pipeline(brief=Brief(theme="momentum"), rules=SPCSettings(auto_runs=3, auto_max_rounds=2),
                                         progress=progress, run_date=date(2026, 10, 26))
    assert report["ready"] and report["rounds"] == 2 and report["name"] == "Momentum v2"
    assert "Stability" in fake.revise_feedback[0] and "shaky" in fake.revise_feedback[0]
    assert any("rewriting" in m for m in seen)
    async with db.session() as s:
        row = await s.get(SPCPrompt, report["prompt_id"])
        runs = list(await s.scalars(select(SPCRun).where(SPCRun.prompt_id == report["prompt_id"])))
    assert row.stats["ready"] and not row.submitted and "STABLE" in row.prompt and len(runs) == 3
    assert spc_auto.summarize(report).startswith("Ready:")


async def test_a_failed_ai_run_is_a_failed_run_not_a_crash(monkeypatch, database, fake_prices) -> None:
    class Flaky(FakeLLM):
        async def generate_text(self, prompt: str, **kw) -> str:  # noqa: ANN003
            raise RuntimeError("429")

    monkeypatch.setattr(spc_auto, "llm", Flaky())
    report = await spc_auto.run_pipeline(prompt=GOOD_PROMPT, rules=SPCSettings(auto_runs=2), run_date=date(2026, 10, 26))
    assert not report["ready"] and "AI call failed" in report["runs"][0]["errors"][0]


async def test_unfixable_prompt_is_saved_as_not_ready(monkeypatch, database, fake_prices) -> None:
    monkeypatch.setattr(spc_auto, "llm", FakeLLM(rewrite_fixes=False))
    report = await spc_auto.run_pipeline(brief=Brief(), rules=SPCSettings(auto_runs=2, auto_max_rounds=1),
                                         run_date=date(2026, 10, 26))
    assert not report["ready"] and report["rounds"] == 2
    assert "Stability" in spc_auto.summarize(report)


async def test_editor_prompt_is_checked_but_not_rewritten(monkeypatch, database, fake_prices) -> None:
    fake = FakeLLM()
    monkeypatch.setattr(spc_auto, "llm", fake)
    report = await spc_auto.run_pipeline(prompt=GOOD_PROMPT, name="Mine", rules=SPCSettings(auto_runs=2),
                                         run_date=date(2026, 10, 26))
    assert report["rounds"] == 1 and not fake.revise_feedback and report["name"] == "Mine"
    async with db.session() as s:
        assert (await s.get(SPCPrompt, report["prompt_id"])).stats["source"] == "editor"


async def test_unpriced_tickers_fail_the_gate(monkeypatch, database, fake_prices) -> None:
    class BadTicker(FakeLLM):
        async def generate_text(self, prompt: str, **kw) -> str:  # noqa: ANN003
            return rows("AAA", "BAD")

    monkeypatch.setattr(spc_auto, "llm", BadTicker())
    report = await spc_auto.run_pipeline(prompt=GOOD_PROMPT, rules=SPCSettings(auto_runs=2, auto_min_priced=0.9),
                                         run_date=date(2026, 10, 26))
    gate = next(g for g in report["gates"] if g["name"] == "Tickers priced")
    assert gate["status"] == "fail" and "BAD" in gate["detail"] and not report["ready"]


# ---------------------------------------------------------------- submission

async def _ready_prompt(monkeypatch, rules: SPCSettings) -> int:
    monkeypatch.setattr(spc_auto, "llm", FakeLLM())
    report = await spc_auto.run_pipeline(brief=Brief(), rules=rules, run_date=date(2026, 10, 26))
    assert report["ready"]
    return report["prompt_id"]


@respx.mock
async def test_submit_needs_confirmation_and_a_passed_report_and_sends_once(monkeypatch, database, fake_prices) -> None:
    rules = SPCSettings(auto_runs=2, submit_path="/spc/entries", max_submits_per_day=5)
    pid = await _ready_prompt(monkeypatch, rules)
    brain.client._credentials = lambda: ("me@example.com", "pw")
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    route = respx.post(f"{API}/spc/entries").mock(return_value=httpx.Response(201, json={"id": "S1"}))

    with pytest.raises(PermissionError):
        await spc_auto.submit_prompt(pid, confirmed=False, rules=rules)
    assert not route.called

    preview = await spc_auto.preview_request(pid, rules)
    assert preview["path"] == "/spc/entries" and "STABLE" in preview["body"]["prompt"] and not route.called

    out = await spc_auto.submit_prompt(pid, confirmed=True, rules=rules)
    assert out.startswith("accepted") and route.call_count == 1
    sent = json.loads(route.calls[0].request.content)
    assert sent["prompt"].endswith("STABLE") and sent["model"] == rules.model
    async with db.session() as s:
        row = await s.get(SPCPrompt, pid)
    assert row.submitted and row.stats["submission"]["status"] == "accepted"
    assert [e["status"] for e in await spc_auto.submit_log()] == ["accepted"]      # the "sending" entry was replaced

    with pytest.raises(RuntimeError, match="already submitted"):
        await spc_auto.submit_prompt(pid, confirmed=True, rules=rules)
    assert route.call_count == 1
    await brain.client.close()


async def test_edited_prompt_text_needs_a_new_check(monkeypatch, database, fake_prices) -> None:
    rules = SPCSettings(auto_runs=2, submit_path="/spc/entries")
    pid = await _ready_prompt(monkeypatch, rules)
    async with db.write() as s:
        (await s.get(SPCPrompt, pid)).prompt += " edited"
    with pytest.raises(RuntimeError, match="changed after it passed"):
        await spc_auto.submit_prompt(pid, confirmed=True, rules=rules)


@respx.mock
async def test_a_prompt_that_did_not_pass_is_never_sent(monkeypatch, database, fake_prices) -> None:
    rules = SPCSettings(auto_runs=2, auto_max_rounds=0, submit_path="/spc/entries")
    monkeypatch.setattr(spc_auto, "llm", FakeLLM(rewrite_fixes=False))
    report = await spc_auto.run_pipeline(brief=Brief(), rules=rules, run_date=date(2026, 10, 26))
    route = respx.post(f"{API}/spc/entries").mock(return_value=httpx.Response(201, json={}))
    assert not report["ready"]
    with pytest.raises(RuntimeError, match="not passed"):
        await spc_auto.submit_prompt(report["prompt_id"], confirmed=True, rules=rules)
    assert not route.called


@respx.mock
async def test_unset_endpoint_sends_nothing(monkeypatch, database, fake_prices) -> None:
    rules = SPCSettings(auto_runs=2)                      # no submit_path
    pid = await _ready_prompt(monkeypatch, rules)
    with pytest.raises(ValueError, match="No submit path"):
        await spc_auto.submit_prompt(pid, confirmed=True, rules=rules)
    assert not respx.calls


@respx.mock
@pytest.mark.parametrize("status, body, expect, submitted", [
    (400, {"detail": "Prompt too similar"}, "rejected", False),
    (429, {"detail": "THROTTLED"}, "throttled", False),
    (500, {"detail": "boom"}, "error", False),
    (303, {}, "unknown", False),
])
async def test_non_2xx_is_never_reported_as_accepted(monkeypatch, database, fake_prices, status, body, expect,
                                                     submitted) -> None:
    rules = SPCSettings(auto_runs=2, submit_path="/spc/entries", max_submits_per_day=5)
    pid = await _ready_prompt(monkeypatch, rules)
    brain.client._credentials = lambda: ("me@example.com", "pw")
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    route = respx.post(f"{API}/spc/entries").mock(return_value=httpx.Response(status, json=body))
    out = await spc_auto.submit_prompt(pid, confirmed=True, rules=rules)
    assert out.startswith(expect) and route.call_count == 1        # a 5xx is not retried: it could double-submit
    async with db.session() as s:
        assert (await s.get(SPCPrompt, pid)).submitted is submitted
    await brain.client.close()


@respx.mock
async def test_network_failure_has_unknown_outcome_and_counts_against_the_daily_guard(monkeypatch, database,
                                                                                      fake_prices) -> None:
    rules = SPCSettings(auto_runs=2, submit_path="/spc/entries", max_submits_per_day=1)
    pid = await _ready_prompt(monkeypatch, rules)
    brain.client._credentials = lambda: ("me@example.com", "pw")
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    respx.post(f"{API}/spc/entries").mock(side_effect=httpx.ConnectError("reset"))
    out = await spc_auto.submit_prompt(pid, confirmed=True, rules=rules)
    assert out.startswith("unknown") and "check the SPC page on BRAIN" in out
    assert await spc_auto.submissions_today() == 1
    pid2 = await _ready_prompt(monkeypatch, rules)
    with pytest.raises(RuntimeError, match="guard"):
        await spc_auto.submit_prompt(pid2, confirmed=True, rules=rules)
    await brain.client.close()


# ---------------------------------------------------------------- the job, end to end

async def _run_job(kind: str, title: str, params: dict):  # noqa: ANN202
    import asyncio

    from alphago.db.models import Job
    from alphago.jobs.queue import jobs
    jobs._pending, jobs._running = asyncio.Queue(), {}       # the singleton's queue binds to the first test's event loop
    await jobs.start()
    try:
        jid = await jobs.submit(kind, title, params)
        for _ in range(400):
            async with db.session() as s:
                job = await s.get(Job, jid)
            if job.status not in ("queued", "running"):
                return job
            await asyncio.sleep(0.05)
        raise AssertionError("job did not finish")
    finally:
        await jobs.stop()


@respx.mock
async def test_generate_job_stops_at_ready_and_never_submits(monkeypatch, database, fake_prices) -> None:
    from alphago.config import settings
    monkeypatch.setattr(spc_auto, "llm", FakeLLM())
    monkeypatch.setattr(settings.spc, "auto_runs", 2)
    monkeypatch.setattr(settings.spc, "submit_path", "/spc/entries")           # even with the endpoint fully set up
    route = respx.post(f"{API}/spc/entries").mock(return_value=httpx.Response(201, json={}))
    job = await _run_job("spc_generate", "gen", {"brief": {"theme": "momentum"}})
    assert job.status == "done" and job.message.startswith("Ready:")
    async with db.session() as s:
        row = (await s.scalars(select(SPCPrompt))).first()
    assert row.stats["ready"] and not row.submitted and not route.called       # waits for the confirmation click


@respx.mock
async def test_auto_submit_sends_only_when_switched_on_and_every_gate_passed(monkeypatch, database, fake_prices) -> None:
    from alphago.config import settings
    monkeypatch.setattr(spc_auto, "llm", FakeLLM())
    monkeypatch.setattr(settings.spc, "auto_runs", 2)
    monkeypatch.setattr(settings.spc, "auto_submit", True)
    monkeypatch.setattr(settings.spc, "submit_path", "/spc/entries")
    brain.client._credentials = lambda: ("me@example.com", "pw")
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    route = respx.post(f"{API}/spc/entries").mock(return_value=httpx.Response(201, json={"id": "S1"}))
    job = await _run_job("spc_generate", "gen", {"brief": {"theme": "momentum"}})
    assert job.status == "done" and "accepted" in job.message and route.call_count == 1
    async with db.session() as s:
        assert (await s.scalars(select(SPCPrompt))).first().submitted
    await brain.client.close()


@respx.mock
async def test_auto_submit_never_sends_a_prompt_that_failed_a_gate(monkeypatch, database, fake_prices) -> None:
    from alphago.config import settings
    monkeypatch.setattr(spc_auto, "llm", FakeLLM(rewrite_fixes=False))
    monkeypatch.setattr(settings.spc, "auto_runs", 2)
    monkeypatch.setattr(settings.spc, "auto_max_rounds", 0)
    monkeypatch.setattr(settings.spc, "auto_submit", True)
    monkeypatch.setattr(settings.spc, "submit_path", "/spc/entries")
    route = respx.post(f"{API}/spc/entries").mock(return_value=httpx.Response(201, json={}))
    job = await _run_job("spc_generate", "gen", {"brief": {"theme": "momentum"}})
    assert job.status == "done" and job.message.startswith("Not ready") and not route.called


@respx.mock
async def test_auto_submit_refusal_keeps_the_prompt_ready(monkeypatch, database, fake_prices) -> None:
    from alphago.config import settings
    monkeypatch.setattr(spc_auto, "llm", FakeLLM())
    monkeypatch.setattr(settings.spc, "auto_runs", 2)
    monkeypatch.setattr(settings.spc, "auto_submit", True)
    monkeypatch.setattr(settings.spc, "submit_path", "")                    # endpoint not set up
    job = await _run_job("spc_generate", "gen", {"brief": {"theme": "momentum"}})
    assert job.status == "done" and "Not submitted" in job.message and not respx.calls
    async with db.session() as s:
        row = (await s.scalars(select(SPCPrompt))).first()
    assert row.stats["ready"] and not row.submitted


@respx.mock
async def test_confirmed_submit_job_sends_the_checked_prompt(monkeypatch, database, fake_prices) -> None:
    from alphago.config import settings
    monkeypatch.setattr(settings.spc, "submit_path", "/spc/entries")
    monkeypatch.setattr(settings.spc, "max_submits_per_day", 5)
    pid = await _ready_prompt(monkeypatch, SPCSettings(auto_runs=2))
    brain.client._credentials = lambda: ("me@example.com", "pw")
    respx.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH_OK))
    route = respx.post(f"{API}/spc/entries").mock(return_value=httpx.Response(201, json={"id": "S1"}))
    job = await _run_job("spc_submit", "sub", {"prompt_id": pid, "confirmed": True})
    assert job.status == "done" and job.message.startswith("accepted") and route.call_count == 1
    await brain.client.close()


@respx.mock
async def test_submit_job_without_confirmation_fails_and_sends_nothing(monkeypatch, database, fake_prices) -> None:
    rules = SPCSettings(auto_runs=2, submit_path="/spc/entries")
    pid = await _ready_prompt(monkeypatch, rules)
    route = respx.post(f"{API}/spc/entries").mock(return_value=httpx.Response(201, json={}))
    job = await _run_job("spc_submit", "sub", {"prompt_id": pid})            # no "confirmed"
    assert job.status == "failed" and "confirmation" in (job.error or "") and not route.called


# ---------------------------------------------------------------- wiring

def test_audit_names_the_spc_submission(monkeypatch) -> None:
    from alphago.config import settings
    assert audit.classify("POST", "/spc/entries") == "post"       # not set up: classified like any other POST
    monkeypatch.setattr(settings.spc, "submit_path", "/spc/entries")
    assert audit.classify("POST", "/spc/entries") == "spc_submit"
    assert audit.classify("GET", "/spc/entries") is None
    monkeypatch.setattr(settings.spc, "submit_path", "/competitions/{competition}/submit")
    monkeypatch.setattr(settings.spc, "competition_id", "SPC1")
    assert audit.classify("POST", "/competitions/SPC1/submit") == "spc_submit"   # not counted as an alpha submission


def test_job_kinds_are_registered() -> None:
    from alphago.jobs.queue import jobs
    assert {"spc_generate", "spc_submit", "aqp_describe", "aqp_apply", "aqp_untag"} <= set(jobs._handlers)


# ---------------------------------------------------------------- page

@pytest.fixture
async def ui_db(tmp_path, monkeypatch):
    from alphago import background
    monkeypatch.setattr(background, "autostart_enabled", lambda: False)
    await db.init(tmp_path / "spc_ui.db")
    yield
    await db.close()


async def test_spc_lab_shows_the_generation_panel_and_keeps_its_old_sections(ui_db) -> None:
    from nicegui.testing.user_simulation import user_simulation

    from alphago.ui.layout import root
    async with user_simulation(root=root) as user:
        await user.open("/spc")
        for text in ("Generate with AI, check, and submit", "Generate & check", "Check the prompt in the editor",
                     "Gate settings", "Submit settings", "1. Prompt", "2. Output", "Saved runs"):
            await user.should_see(text)


async def test_generate_without_an_ai_provider_says_so(ui_db, monkeypatch) -> None:
    from nicegui.testing.user_simulation import user_simulation

    from alphago.llm import llm
    from alphago.ui.layout import root
    monkeypatch.setattr(llm, "configure", lambda *a, **k: False)
    async with user_simulation(root=root) as user:
        await user.open("/spc")
        user.find("Generate & check").click()
        await user.should_see("Set up an AI provider")
