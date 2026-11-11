"""Alpha Quality Project: eligibility, ranking, the description pre-checks and the confirmed, all-or-nothing apply."""

from __future__ import annotations

import pytest

from alphago.config import settings
from alphago.db import db
from alphago.db.models import Alpha
from alphago.engine import actions, aqp
from alphago.engine.aqp import AQPDescription, AQPGrade

CYCLE = "AQP202611"
CODE_A = "rank(ts_delta(close, 5)) * group_rank(ts_mean(volume, 20), industry)"


@pytest.fixture
async def database(tmp_path):
    await db.init(tmp_path / "aqp.db")
    yield
    await db.close()


def make_alpha(aid: str, code: str = CODE_A, **kw) -> Alpha:
    base = dict(id=aid, type="REGULAR", status="ACTIVE", region="USA", delay=1, universe="TOP3000", code=code, tags=[],
                sharpe=1.8, os_sharpe=1.0, turnover=0.2, op_count=4, field_count=2, fields=["close", "volume"],
                cluster="momentum", settings={"region": "USA", "neutralization": "SUBINDUSTRY", "decay": 4})
    base.update(kw)
    return Alpha(**base)


def description_for(code: str, extra: str = "", tail: bool = True) -> str:
    """A description long enough that names every field and operator of `code`."""
    ops, names, nums = aqp.expression_terms(code)
    body = " ".join(f"The term {t} matters here because it shapes the signal." for t in ops + names + nums)
    return "Idea: " + body + " " + extra + (" It should work because slow-moving capital reacts late." * 8 if tail else "")


# ---------------------------------------------------------------- eligibility and ranking

async def test_candidates_mark_used_selected_and_excluded(database) -> None:
    async with db.write() as s:
        s.add_all([make_alpha("OK1"), make_alpha("OLD", tags=["AQP202610"]), make_alpha("NOW", tags=[CYCLE]),
                   make_alpha("EUR", region="EUR"), make_alpha("UNSUB", status="UNSUBMITTED"),
                   make_alpha("SUPER", type="SUPER"), make_alpha("LEDGER")])
    await db.kv_set(aqp.LEDGER_KEY, [{"cycle": "AQP202609", "alpha_id": "LEDGER"}])
    rows = {r["id"]: r for r in await aqp.candidates(CYCLE)}
    assert "EUR" not in rows                                            # other regions are not listed
    assert rows["OK1"]["state"] == "eligible" and rows["NOW"]["state"] == "selected"
    assert rows["OLD"]["state"] == "used" and "cannot be tagged again" in rows["OLD"]["reason"]
    assert rows["LEDGER"]["state"] == "used"                            # the app's own ledger catches unsynced tags
    assert rows["SUPER"]["state"] == "excluded" and "UNSUB" not in rows        # only submitted (ACTIVE) alphas are listed
    unsub = make_alpha("U", status="UNSUBMITTED")
    assert aqp.eligibility(unsub, CYCLE)[0] == "excluded"                      # and apply re-checks it anyway


def test_ranking_weights_description_over_performance() -> None:
    simple = make_alpha("S", os_sharpe=0.3, op_count=4)
    complex_ = make_alpha("C", os_sharpe=2.0, op_count=15, field_count=5, cluster=None, est_self_corr=0.8)
    assert aqp.score_alpha(simple)["score"] > aqp.score_alpha(complex_)["score"]
    assert aqp.score_alpha(make_alpha("E", os_sharpe=None, sharpe=2.0))["os_basis"] == "est."


def test_os_estimate_uses_the_predicted_retention_when_there_is_one() -> None:
    assert aqp.performance(make_alpha("E", os_sharpe=None, sharpe=2.0))[1] == pytest.approx(1.0)       # 50% haircut
    assert aqp.performance(make_alpha("R", os_sharpe=None, sharpe=2.0, os_retention_pred=0.8))[1] == pytest.approx(1.6)


def test_pick_best_takes_different_idea_families() -> None:
    rows = [{"id": i, "state": "eligible", "score": sc, "cluster": cl}
            for i, sc, cl in [("A", .9, "momentum"), ("B", .85, "momentum"), ("C", .8, "value"), ("D", .7, "quality"),
                              ("U", .99, "value")]]
    rows[-1]["state"] = "used"
    assert aqp.pick_best(rows) == ["A", "C", "D"]
    assert aqp.pick_best(rows[:2]) == ["A", "B"]                         # fewer families than slots: fill with the next best


def test_pick_best_keeps_this_cycles_picks() -> None:
    rows = [{"id": "A", "state": "eligible", "score": .9, "cluster": "momentum"},
            {"id": "T", "state": "selected", "score": .2, "cluster": "value"}]
    assert aqp.pick_best(rows, 1) == ["T"]


# ---------------------------------------------------------------- the literal match check

def test_expression_terms_skip_assignments_and_keyword_arguments() -> None:
    ops, names, nums = aqp.expression_terms("x = ts_zscore(close, 20); group_rank(x, industry) + rank(volume, rate=2)")
    assert ops == ["ts_zscore", "group_rank", "rank"]
    assert names == ["close", "industry", "volume"] and "x" not in names and "rate" not in names
    assert nums == ["20", "2"]


def test_coverage_and_check_description() -> None:
    text = ("ts_delta of close over 5 days, ranked, times a group_rank of ts_mean of volume (20 days) by industry. "
            + "x" * 800)
    cov = aqp.coverage(text, CODE_A)
    assert cov["missing"] == ["rank"]                                    # `rank` appears only inside group_rank
    full = aqp.check_description(description_for(CODE_A), CODE_A)
    assert full["ok"] and not full["missing"]
    bad = aqp.check_description("short. mentions close only", CODE_A)
    assert not bad["ok"] and any("characters" in p for p in bad["problems"]) and "ts_delta" in bad["missing"]
    assert not aqp._mentions("group_rank only", "rank")                  # a term inside another name does not count


def test_similarity_flags_copy_paste() -> None:
    a = description_for(CODE_A)
    assert aqp.similarity(a, a) == 1.0
    assert aqp.similarity("momentum reversal liquidity premium", "dividend yield quality profitability") == 0.0


# ---------------------------------------------------------------- writing the description

class FakeLLM:
    def __init__(self, *, forget_first: bool = True, grade: int = 8) -> None:
        self.calls, self.revisions, self.forget_first, self.grade = 0, [], forget_first, grade

    async def generate_json(self, prompt: str, schema, **kw):  # noqa: ANN001, ANN003
        if schema is AQPGrade:
            return AQPGrade(match=self.grade, uniqueness=self.grade, hypothesis=self.grade, gaps=["be more specific"])
        self.calls += 1
        if "WHAT MUST CHANGE" in prompt:
            self.revisions.append(prompt.split("WHAT MUST CHANGE")[1])
        text = description_for(CODE_A)
        if self.forget_first and self.calls == 1:
            text = "Idea: a generic momentum story. " + "It works because of slow reaction. " * 30
        return AQPDescription(idea=text, rationale_data="close and volume", rationale_operators="ranked",
                              different="unlike the textbook version", risks="crowding")


async def test_failed_precheck_is_rewritten_with_the_missing_terms(monkeypatch, database) -> None:
    async with db.write() as s:
        s.add(make_alpha("A1"))
    fake = FakeLLM()
    monkeypatch.setattr(aqp, "llm", fake)
    res = await aqp.write_description("A1", others=["A1"])
    assert res["ok"] and res["rounds"] == 2 and "Does not name" in fake.revisions[0] and "ts_delta" in fake.revisions[0]
    assert "subindustry-neutralized" in fake.revisions[0]                # the neutralization is asked for too
    assert (await db.kv_get(aqp.DRAFT_KEY.format("A1")))["ok"]


async def test_low_judge_scores_block_and_are_reported(monkeypatch, database) -> None:
    async with db.write() as s:
        s.add(make_alpha("A1"))
    monkeypatch.setattr(aqp, "llm", FakeLLM(forget_first=False, grade=4))
    res = await aqp.write_description("A1", rounds=1)
    assert not res["ok"] and res["rounds"] == 2 and res["grade"]["match"] == 4


async def test_editing_a_draft_rechecks_the_text(monkeypatch, database) -> None:
    async with db.write() as s:
        s.add(make_alpha("A1"))
    monkeypatch.setattr(aqp, "llm", FakeLLM(forget_first=False))
    await aqp.write_description("A1")
    res = await aqp.recheck_text("A1", "now it says almost nothing")
    assert not res["ok"] and res["edited"] and res["checks"]["missing"]


# ---------------------------------------------------------------- apply

@pytest.fixture
def fake_brain(monkeypatch):
    """refresh_alpha reads the DB; set_properties records the PATCH and updates the DB like BRAIN would."""
    patches: list[tuple[str, dict]] = []

    async def refresh(aid: str):
        async with db.session() as s:
            return await s.get(Alpha, aid)

    async def set_props(aid: str, **props):
        patches.append((aid, props))
        async with db.write() as s:
            a = await s.get(Alpha, aid)
            if "tags" in props:
                a.tags = props["tags"]
            if "regular_description" in props:
                a.description = props["regular_description"]

    monkeypatch.setattr(actions, "refresh_alpha", refresh)
    monkeypatch.setattr(actions, "set_properties", set_props)
    return patches


def _vocab(i: int, n: int = 90) -> str:
    """Words that no other alpha's text shares (a different story per alpha)."""
    return " ".join("".join(chr(97 + (i * 5 + k * 3 + j * (i + 2)) % 26) for j in range(6)) + chr(97 + (k + i) % 26) * 2
                    for k in range(n))


def distinct_items(ids: list[str]) -> list[dict]:
    return [{"alpha_id": aid, "description": description_for(CODE_A, _vocab(i + 1), tail=False)}
            for i, aid in enumerate(ids)]


async def test_apply_requires_confirmation(database, fake_brain) -> None:
    async with db.write() as s:
        s.add(make_alpha("A1"))
    with pytest.raises(PermissionError):
        await aqp.apply_cycle(CYCLE, distinct_items(["A1"]), confirmed=False)
    assert not fake_brain


async def test_apply_writes_description_and_tag_together_and_keeps_other_tags(database, fake_brain) -> None:
    async with db.write() as s:
        s.add_all([make_alpha("A1", tags=["mine"]), make_alpha("A2", description="old text")])
    out = await aqp.apply_cycle(CYCLE, distinct_items(["A1", "A2"]), confirmed=True)
    assert "tagged 2" in out and len(fake_brain) == 2
    aid, props = fake_brain[0]
    assert props["tags"] == ["mine", CYCLE] and props["regular_description"].startswith("Idea:")
    assert (await db.kv_get(aqp.BACKUP_KEY.format("A2")))["description"] == "old text"
    assert {e["alpha_id"] for e in await db.kv_get(aqp.LEDGER_KEY)} == {"A1", "A2"}


async def test_a_failed_write_says_what_was_already_written(database, fake_brain, monkeypatch) -> None:
    async with db.write() as s:
        s.add_all([make_alpha("A1"), make_alpha("A2")])
    ok_props = actions.set_properties

    async def flaky(aid: str, **props):
        if aid == "A2":
            raise RuntimeError("BRAIN down")
        await ok_props(aid, **props)

    monkeypatch.setattr(actions, "set_properties", flaky)
    with pytest.raises(RuntimeError, match="Already written: A1"):
        await aqp.apply_cycle(CYCLE, distinct_items(["A1", "A2"]), confirmed=True)
    assert {e["alpha_id"] for e in await db.kv_get(aqp.LEDGER_KEY)} == {"A1"}


async def test_apply_is_all_or_nothing(database, fake_brain) -> None:
    async with db.write() as s:
        s.add_all([make_alpha("A1"), make_alpha("OLD", tags=["AQP202610"])])
    items = distinct_items(["A1", "OLD"])
    with pytest.raises(RuntimeError, match="Nothing was written.*cannot be tagged again"):
        await aqp.apply_cycle(CYCLE, items, confirmed=True)
    assert not fake_brain                                              # A1 was fine, but nothing is written


async def test_apply_refuses_a_fourth_alpha_and_weak_or_copied_descriptions(database, fake_brain) -> None:
    async with db.write() as s:
        s.add_all([make_alpha(f"A{i}") for i in range(1, 5)])
    with pytest.raises(RuntimeError, match="allows 3"):
        await aqp.apply_cycle(CYCLE, distinct_items(["A1", "A2", "A3", "A4"]), confirmed=True)
    weak = [{"alpha_id": "A1", "description": "too short"}]
    with pytest.raises(RuntimeError, match="characters|Does not name"):
        await aqp.apply_cycle(CYCLE, weak, confirmed=True)
    same = description_for(CODE_A)
    with pytest.raises(RuntimeError, match="overlap"):
        await aqp.apply_cycle(CYCLE, [{"alpha_id": "A1", "description": same}, {"alpha_id": "A2", "description": same}],
                              confirmed=True)
    assert not fake_brain


async def test_already_tagged_this_cycle_counts_against_the_limit(database, fake_brain) -> None:
    async with db.write() as s:
        s.add_all([make_alpha("T1", tags=[CYCLE]), make_alpha("T2", tags=[CYCLE]), make_alpha("T3", tags=[CYCLE]),
                   make_alpha("N1")])
    with pytest.raises(RuntimeError, match="allows 3"):
        await aqp.apply_cycle(CYCLE, distinct_items(["N1"]), confirmed=True)


async def test_bad_cycle_tag_is_refused(database, fake_brain) -> None:
    async with db.write() as s:
        s.add(make_alpha("A1"))
    for bad in ("PowerPoolSelected", "AQP2026", "AQP202611x", ""):
        with pytest.raises(RuntimeError, match="cycle tag"):
            await aqp.apply_cycle(bad, distinct_items(["A1"]), confirmed=True)


async def test_untag_removes_only_the_cycle_tag(database, fake_brain) -> None:
    async with db.write() as s:
        s.add(make_alpha("A1", tags=["mine", CYCLE, "AQP202610"]))
    with pytest.raises(PermissionError):
        await aqp.untag_cycle(CYCLE, ["A1"], confirmed=False)
    await aqp.untag_cycle(CYCLE, ["A1"], confirmed=True)
    assert fake_brain[0][1]["tags"] == ["mine", "AQP202610"]


# ---------------------------------------------------------------- wiring

async def test_apply_job_without_confirmation_fails(database, fake_brain) -> None:
    import asyncio

    from alphago.db.models import Job
    from alphago.jobs.queue import jobs
    async with db.write() as s:
        s.add(make_alpha("A1"))
    jobs._pending, jobs._running = asyncio.Queue(), {}
    await jobs.start()
    try:
        jid = await jobs.submit("aqp_apply", "x", {"cycle": CYCLE, "items": distinct_items(["A1"])})
        for _ in range(200):
            async with db.session() as s:
                job = await s.get(Job, jid)
            if job.status not in ("queued", "running"):
                break
            await asyncio.sleep(0.05)
    finally:
        await jobs.stop()
    assert job.status == "failed" and "confirmation" in job.error and not fake_brain


async def test_aqp_page_lists_alphas_and_explains_the_rules(tmp_path, monkeypatch) -> None:
    from nicegui.testing.user_simulation import user_simulation

    from alphago import background
    from alphago.ui.layout import root
    monkeypatch.setattr(background, "autostart_enabled", lambda: False)
    await db.init(tmp_path / "aqp_ui.db")
    try:
        async with db.write() as s:
            s.add(make_alpha("A1"))
        async with user_simulation(root=root) as user:
            await user.open("/aqp")
            for text in ("Alpha Quality Project", "Suggest the best 3", "Write descriptions (AI)", "Review & apply…",
                         "Pre-check settings", "can never be tagged again"):
                await user.should_see(text)
    finally:
        await db.close()


def test_settings_defaults() -> None:
    assert settings.aqp.region == "USA" and settings.aqp.max_picks == 3 and settings.aqp.tag_prefix == "AQP"
    assert aqp.current_cycle(__import__("datetime").date(2026, 11, 5)) == CYCLE
