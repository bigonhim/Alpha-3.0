"""Phases 3–4: provider failover, error kinds, versioned prompts, model tournament, audit log, health counters,
email alerts, background running, quota planner and the BRAIN recorder."""

from __future__ import annotations

import asyncio
import json
import random
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from alphago.config import settings
from alphago.db import db
from alphago.db.models import Alpha, Candidate, LLMCall
from alphago.llm import LLMError
from alphago.llm import router as router_mod

BASE = {"region": "USA", "delay": 1, "universe": "TOP3000"}


@pytest.fixture
async def dbenv(tmp_path):
    await db.init(tmp_path / "ops.db")
    yield
    await db.close()


class FakeProvider:
    def __init__(self, name: str, fail: LLMError | None = None) -> None:
        self.name, self.label, self.local = name, name.title(), False
        self.fail, self.calls = fail, 0

    def configure(self, api_key=None) -> bool:  # noqa: ANN001
        return True

    def pick(self, role: str) -> str:
        return f"{self.name}-model"

    async def generate_json(self, prompt, schema, **kw):  # noqa: ANN001, ANN003, ANN201
        self.calls += 1
        if self.fail:
            raise self.fail
        return f"{self.name} answered"


# ---------------------------------------------------------------- 3.1 failover

async def test_failover_moves_to_the_next_provider_and_cools_down(monkeypatch) -> None:
    primary = FakeProvider("gemini", LLMError("Gemini error 429: rate limited", 429))
    backup = FakeProvider("claude")
    monkeypatch.setitem(router_mod.PROVIDERS, "gemini", primary)
    monkeypatch.setitem(router_mod.PROVIDERS, "claude", backup)
    monkeypatch.setattr(settings.llm, "provider", "gemini")
    monkeypatch.setattr(settings.llm, "fallbacks", ["claude"])
    events = []
    monkeypatch.setattr(router_mod.bus, "publish", lambda *a, **k: events.append((a, k)))
    r = router_mod.LLMRouter()
    assert await r.generate_json("p", dict) == "claude answered"
    assert events and events[0][1]["kind"] == "llm_failover"
    assert await r.generate_json("p", dict) == "claude answered"
    assert primary.calls == 1                       # in its cooldown the failed primary is tried last
    assert [p.name for p in r.chain()] == ["claude", "gemini"]


async def test_no_failover_without_fallbacks_or_for_budget(monkeypatch) -> None:
    primary = FakeProvider("gemini", LLMError("Gemini error 503: overloaded", 503))
    backup = FakeProvider("claude")
    monkeypatch.setitem(router_mod.PROVIDERS, "gemini", primary)
    monkeypatch.setitem(router_mod.PROVIDERS, "claude", backup)
    monkeypatch.setattr(settings.llm, "provider", "gemini")
    monkeypatch.setattr(settings.llm, "fallbacks", [])
    with pytest.raises(LLMError):                   # the default: exactly the old behaviour
        await router_mod.LLMRouter().generate_json("p", dict)
    monkeypatch.setattr(settings.llm, "fallbacks", ["claude"])
    primary.fail = LLMError("Daily Gemini call budget (400) reached")
    with pytest.raises(LLMError):                   # the budget is shared: another provider would not help
        await router_mod.LLMRouter().generate_json("p", dict)
    assert backup.calls == 0


async def test_tournament_call_forces_the_model(monkeypatch) -> None:
    from alphago.llm.base import MODEL_OVERRIDE, chosen_model
    seen = []

    class Recording(FakeProvider):
        async def generate_json(self, prompt, schema, **kw):  # noqa: ANN001, ANN003, ANN201
            seen.append(chosen_model(self, "ideation"))
            return "ok"

    monkeypatch.setitem(router_mod.PROVIDERS, "claude", Recording("claude"))
    await router_mod.LLMRouter().generate_json_with("claude", "claude-haiku-5-5", "p", dict)
    assert seen == ["claude-haiku-5-5"] and MODEL_OVERRIDE.get() is None


# ---------------------------------------------------------------- 3.2 error kinds

async def test_errors_are_logged_with_their_kind(dbenv) -> None:
    import time

    from alphago.llm.base import log_call
    await log_call("gemini", "m", "ideation", time.monotonic(), status="error", error="429: Resource exhausted")
    await log_call("claude", "m", "ideation", time.monotonic(), status="error", error="Request timed out")
    await log_call("claude", "m", "ideation", time.monotonic())
    async with db.session() as s:
        kinds = [r.error_kind for r in await s.scalars(select(LLMCall).order_by(LLMCall.id))]
    assert kinds == ["rate_limited", "timeout", None]
    from alphago.engine import health
    day = await health.llm_by_day(3)
    claude = next(r for r in day["rows"] if r["provider"] == "claude")
    assert claude["calls"] == 2 and claude["timeout"] == 1 and claude["ok"] == 1


# ---------------------------------------------------------------- 3.4 versioned prompts

async def test_prompt_versions_and_their_results(dbenv) -> None:
    from alphago.engine.tournament import prompt_results
    from alphago.llm import prompts
    v1 = await prompts.current_version("ideation")
    assert v1 == await prompts.current_version("ideation")          # unchanged text = same version
    v2 = await prompts.set_template("ideation", prompts.DEFAULTS["ideation"] + "\nBe bold.")
    assert v2 != v1 and await prompts.current_version("ideation") == v2
    async with db.write() as s:
        s.add(Alpha(id="PV1", verdict="strong"))
        s.add(Candidate(expression="x", settings=BASE, settings_hash="h", region="USA", delay=1, status="done",
                        alpha_id="PV1", prompt_version=v2, origin="llm"))
        s.add(Candidate(expression="y", settings=BASE, settings_hash="h", region="USA", delay=1, status="invalid",
                        prompt_version=v1, origin="llm"))
    res = {r["id"]: r for r in await prompt_results("ideation")}
    assert res[v2]["strong_plus"] == 1 and res[v2]["simulated"] == 1 and res[v1]["valid"] == 0
    v3 = await prompts.restore_version(v1)
    assert await prompts.get_template("ideation") == prompts.DEFAULTS["ideation"] and v3 not in (v1, v2)
    assert "+Be bold." in prompts.diff(prompts.DEFAULTS["ideation"], prompts.DEFAULTS["ideation"] + "\nBe bold.")


async def test_ideas_record_model_and_prompt_version(dbenv, monkeypatch) -> None:
    from alphago.engine import generators
    from alphago.engine.simulator import SimulationService
    from alphago.llm.schemas import AlphaIdea, IdeaBatch

    async def fake(prompt, schema, **kw):  # noqa: ANN001, ANN003, ANN202
        return IdeaBatch(ideas=[AlphaIdea(expression="rank(ts_delta(close, 7))", idea="i", mechanism="reversal",
                                          data_rationale="d", operator_rationale="o")])

    monkeypatch.setattr(generators.llm, "generate_json", fake)
    specs = await generators.llm_ideas(BASE, dataset_id=None, n=1, goal="g")
    res = await SimulationService().prepare(specs)
    c = res.created[0]
    assert c.prompt_version and c.llm_model and c.llm_model.endswith(generators.llm.pick("ideation"))
    assert c.cluster == "reversal"


# ---------------------------------------------------------------- 3.3 tournament

async def test_tournament_rotates_then_follows_results(dbenv, monkeypatch) -> None:
    from alphago.engine import tournament
    monkeypatch.setitem(router_mod.PROVIDERS, "gemini", FakeProvider("gemini"))
    monkeypatch.setitem(router_mod.PROVIDERS, "claude", FakeProvider("claude"))
    t = settings.llm.tournament
    monkeypatch.setattr(t, "enabled", True)
    monkeypatch.setattr(t, "contestants", ["gemini:g-flash", "claude:c-sonnet"])
    monkeypatch.setattr(t, "min_ideas", 10)
    monkeypatch.setattr(t, "locked", "")
    from alphago.config import AppSettings
    monkeypatch.setattr(AppSettings, "save", lambda self: None)
    first = await tournament.choose(random.Random(0))
    assert first in (("gemini", "g-flash"), ("claude", "c-sonnet"))
    async with db.write() as s:
        for i in range(40):
            for label, strong in (("gemini:g-flash", i < 2), ("claude:c-sonnet", i < 16)):
                aid = f"{label[0]}{i}"
                s.add(Alpha(id=aid, verdict="strong" if strong else "rejected"))
                s.add(Candidate(expression="x", settings=BASE, settings_hash="h", region="USA", delay=1,
                                status="done", alpha_id=aid, llm_model=label, origin="llm"))
    rng = random.Random(1)
    picks = [await tournament.choose(rng) for _ in range(200)]
    assert picks.count(("claude", "c-sonnet")) > 150             # the better model gets most calls
    board = {r["label"]: r for r in await tournament.scoreboard()}
    assert board["claude:c-sonnet"]["strong_per_100"] == 40.0 and board["gemini:g-flash"]["strong_plus"] == 2
    assert await tournament.lock_winner() == "claude:c-sonnet"
    assert await tournament.choose() == ("claude", "c-sonnet")


# ---------------------------------------------------------------- 4.3 audit log

async def test_audit_records_actions_with_actor_and_quota(dbenv) -> None:
    from alphago.engine import audit
    resp = SimpleNamespace(headers={})
    audit.ACTOR.set("task:12")
    audit.observe("POST", "/simulations", 201, [{"type": "REGULAR"}] * 10, True, resp)
    audit.observe("POST", "/simulations", 429, [{"type": "REGULAR"}] * 3, True, resp)       # refused: no cost
    audit.observe("GET", "/alphas/AB123/check", 200, None, True, SimpleNamespace(headers={"Retry-After": "2"}))
    audit.observe("GET", "/alphas/AB123/check", 200, None, True, resp)                     # finished check
    audit.observe("GET", "/alphas/AB123", 200, None, True, resp)                           # reads are not logged
    audit.ACTOR.set("you (confirmed)")
    audit.observe("POST", "/alphas/AB123/submit", 201, None, True, resp)
    audit.observe("PATCH", "/alphas/AB123", 200, {"osmosisPoints": 500}, True, resp)
    audit.observe("POST", "/simulations", 500, [{}], False, resp)                          # a retried attempt
    await audit.flush()
    rows = await audit.entries()
    got = sorted((r.action, r.actor, r.quota_cost) for r in rows)
    assert got == sorted([("simulate", "task:12", 10), ("simulate", "task:12", 0), ("check", "task:12", 0),
                          ("submit", "you (confirmed)", 0), ("properties", "you (confirmed)", 0)])
    props = next(r for r in rows if r.action == "properties")
    assert props.alpha_ids == ["AB123"] and "osmosis points 500" in props.detail
    s = await audit.summary()
    assert s["submissions"] == 1 and {r["actor"]: r["simulations"] for r in s["by_actor"]}["task:12"] == 10
    assert "time (UTC)" in audit.to_csv(rows)


async def test_audit_and_health_hooks_see_real_client_requests(dbenv) -> None:
    import httpx
    import respx

    from alphago.brain import brain
    from alphago.engine import audit, health
    audit.install()
    health.install()
    api = "https://api.worldquantbrain.com"
    try:
        with respx.mock:
            respx.post(f"{api}/authentication").mock(return_value=httpx.Response(
                201, json={"user": {"id": "U"}, "token": {"expiry": 9000}}))
            respx.patch(f"{api}/alphas/ZZ123").mock(return_value=httpx.Response(200, json={"id": "ZZ123"}))
            brain.client._credentials = lambda: ("me@x.com", "pw")
            await brain.client.login()
            await brain.alphas.patch("ZZ123", name="hello")
        await audit.flush()
        actions = sorted(r.action for r in await audit.entries())
        assert actions == ["properties"]               # sign-in goes through http.post directly, not request()
        await health.flush()
        today = (await db.kv_get(health.BRAIN_STATS_KEY))[next(iter(await db.kv_get(health.BRAIN_STATS_KEY)))]
        assert today["requests"] >= 1
    finally:
        brain.client.observers.clear()
        await brain.client.close()


# ---------------------------------------------------------------- 4.1 alerts

async def test_alerts_map_events_and_respect_the_quiet_period(dbenv, monkeypatch) -> None:
    from alphago.engine import alerts
    from alphago.events import Event
    sent = []

    async def fake_send(subject, body):  # noqa: ANN001, ANN202
        sent.append((subject, body))

    a = settings.alerts
    for k, v in {"enabled": True, "smtp_host": "smtp.example.com", "recipient": "me@example.com",
                 "quiet_minutes": 30}.items():
        monkeypatch.setattr(a, k, v)
    monkeypatch.setattr(alerts, "send", fake_send)
    monkeypatch.setattr(alerts, "_last_sent", {})
    monkeypatch.setattr(alerts, "_suppressed", {})
    assert alerts.trigger_for(Event("check", "A1 passed every submission check", "success", {"kind": "ready"})) \
        == "ready"
    assert alerts.trigger_for(Event("llm", "Gemini failed — switching", "warning", {"kind": "llm_failover"})) \
        == "llm_failing"
    assert alerts.trigger_for(Event("simulate", "Queue paused — BRAIN reports the daily simulation limit: x",
                                    "warning")) == "quota"
    assert alerts.trigger_for(Event("simulate", "Simulated 10 alpha(s)", "success")) is None
    assert await alerts.notify("ready", "Ready", "A1 passed")
    assert not await alerts.notify("ready", "Ready", "A2 passed")          # held back: quiet period
    alerts._last_sent["ready"] = -1e9
    assert await alerts.notify("ready", "Ready", "A3 passed")
    assert "+1 similar" in sent[-1][1] and len(sent) == 2
    monkeypatch.setattr(a, "triggers", {**a.triggers, "task_finished": False})
    assert not await alerts.notify("task_finished", "Done", "x")           # switched off


def test_alerts_are_off_by_default() -> None:
    from alphago.config import AlertSettings
    assert AlertSettings().enabled is False


# ---------------------------------------------------------------- 4.2 background

def test_only_one_engine_at_a_time(tmp_path) -> None:
    from alphago import background
    path = tmp_path / "engine.lock"
    assert background.acquire_instance_lock(path)
    held = background._lock_handle
    background._lock_handle = None                                         # pretend a second process asks
    assert not background.acquire_instance_lock(path)
    background._lock_handle = held
    background.release_instance_lock()
    assert background.acquire_instance_lock(path)
    background.release_instance_lock()


def test_logon_task_command(monkeypatch) -> None:
    from alphago import background
    calls = []
    monkeypatch.setattr(background, "_schtasks", lambda *a: calls.append(a) or SimpleNamespace(
        returncode=0, stdout="SUCCESS", stderr=""))
    monkeypatch.setattr(background.os, "name", "nt")
    assert "--headless" in background.headless_command()
    background.set_autostart(True)
    background.set_autostart(False)
    assert calls[0][:6] == ("/Create", "/F", "/SC", "ONLOGON", "/RL", "LIMITED") and calls[1][0] == "/Delete"


def test_web_mode_needs_a_token_off_localhost(monkeypatch) -> None:
    from alphago import __main__ as entry
    monkeypatch.setattr(entry.sys, "argv", ["alphago", "--web", "--host", "0.0.0.0"])
    monkeypatch.setattr(entry.sys, "platform", "linux")                    # no message box in tests
    with pytest.raises(SystemExit):
        entry.main()


# ---------------------------------------------------------------- 4.4 planner

async def test_planner_rates_and_allocates(dbenv) -> None:
    from alphago.engine.planner import allocate, yield_table
    async with db.write() as s:
        for i in range(100):
            for ds, origin, ready in (("dsA", "search", i < 3), ("dsB", "pair", False)):
                aid = f"{ds}{i}"
                s.add(Alpha(id=aid, verdict="ready" if ready else "rejected"))
                s.add(Candidate(expression="x", settings=BASE, settings_hash="h", region="USA", delay=1,
                                status="done", alpha_id=aid, dataset_id=ds, origin=origin))
    rows = await yield_table("USA", 1)
    top = rows[0]
    assert (top["dataset"], top["strategy"], top["passed_per_100"]) == ("dsA", "search", 3.0)
    assert top["passed_lo"] < 3.0 < top["passed_hi"]
    plan = allocate(rows, 500)
    assert plan[0]["dataset"] == "dsA" and plan[0]["allocate"] > plan[-1]["allocate"]
    assert sum(p["allocate"] for p in plan) <= 501


# ---------------------------------------------------------------- 4.5 recorder

def test_recorder_strips_personal_data(tmp_path, monkeypatch) -> None:
    from alphago.brain import recorder
    monkeypatch.setenv(recorder.ENV, str(tmp_path))
    monkeypatch.setattr(recorder, "_counts", {})
    body = {"id": "AB12345", "author": {"email": "me@x.com", "id": "U7"}, "token": "secret",
            "is": {"sharpe": 1.5}}
    resp = SimpleNamespace(content=b"x", headers={}, json=lambda: body)
    recorder.observe("GET", "/alphas/AB12345", 200, None, True, resp)
    recorder.observe("POST", "/simulations", 201, None, True, resp)        # only GETs are recorded
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["shape"] == "/alphas/{id}" and saved["json"]["token"] == "REDACTED"
    assert saved["json"]["author"]["email"] == "REDACTED" and saved["json"]["is"]["sharpe"] == 1.5
    assert asyncio.iscoroutinefunction(recorder.observe) is False
