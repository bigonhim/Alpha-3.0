"""Phase 2: learning from your review, AI self-check, walk-forward, search-adjusted (deflated) Sharpe and the
in-sample → out-of-sample gap model."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import select

from alphago.config import settings
from alphago.db import db
from alphago.db.models import Alpha, AlphaPnL, AutopilotRun, Batch, Candidate
from alphago.engine import honesty

BASE = {"region": "USA", "delay": 1, "universe": "TOP3000"}


@pytest.fixture
async def dbenv(tmp_path):
    await db.init(tmp_path / "honesty.db")
    yield
    await db.close()


# ---------------------------------------------------------------- walk-forward and deflated Sharpe

def test_walk_forward_flags_a_fading_alpha() -> None:
    rng = np.random.default_rng(1)
    steady = list(rng.normal(0.08, 1, 1250))
    fading = list(rng.normal(0.12, 1, 1000)) + list(rng.normal(-0.02, 1, 250))
    r_steady, _, _ = honesty.walk_forward_ratio(steady)
    r_fading, first, last = honesty.walk_forward_ratio(fading)
    assert r_fading < 0.3 < r_steady and first > 0 > last
    assert honesty.walk_forward_ratio(steady[:100]) == (None, None, None)     # too short to say


def test_deflated_sharpe_falls_as_the_search_grows() -> None:
    trials = list(np.random.default_rng(2).normal(0.5, 0.6, 400))
    one = honesty.deflated_sharpe(1.6, 1250, 1, trials)
    some = honesty.deflated_sharpe(1.6, 1250, 20, trials)
    many = honesty.deflated_sharpe(1.6, 1250, 400, trials)
    assert one > some > many and one > 0.99 and many < 0.95
    assert honesty.expected_max_sharpe(1, 0.1) == 0.0


async def test_search_size_counts_ideas_not_repairs(dbenv) -> None:
    async with db.write() as s:
        for i in range(6):
            origin = "optimize" if i >= 4 else "search"
            s.add(Alpha(id=f"A{i}", sharpe=0.2 * i, candidate_id=i + 1))
            s.add(Candidate(id=i + 1, expression="x", settings=BASE, settings_hash="h", region="USA", delay=1,
                            origin=origin, status="done", alpha_id=f"A{i}", task_id=7))
    size = await honesty.search_size("A5")
    assert size.n_trials == 6 and size.n_eff == 4 and len(size.sharpes) == 4


# ---------------------------------------------------------------- the gap model

async def test_gap_model_learns_that_turnover_predicts_a_fall(dbenv) -> None:
    rng = np.random.default_rng(4)
    async with db.write() as s:
        for i in range(30):
            turnover = float(rng.uniform(0.05, 0.6))
            retention = 0.9 - turnover + float(rng.normal(0, 0.03))      # higher turnover, bigger fall
            s.add(Alpha(id=f"G{i}", status="ACTIVE", sharpe=1.8, os_sharpe=1.8 * retention, turnover=turnover,
                        op_count=int(rng.integers(2, 9)), decay=int(rng.integers(0, 10)), code="rank(close)",
                        region="USA", delay=1))
    model = await honesty.fit_gap_model()
    assert model["ready"] and model["n"] == 30 and model["r2"] > 0.8
    top = honesty.explain(model)[0]
    assert top["feature"] == "turnover" and top["effect"] < 0
    calm = honesty.predict_retention(model, Alpha(id="n1", sharpe=1.8, turnover=0.1, code="rank(close)"), None)
    busy = honesty.predict_retention(model, Alpha(id="n2", sharpe=1.8, turnover=0.55, code="rank(close)"), None)
    assert calm > busy


async def test_gap_model_waits_for_enough_alphas(dbenv) -> None:
    async with db.write() as s:
        s.add(Alpha(id="G0", status="ACTIVE", sharpe=1.5, os_sharpe=1.0, turnover=0.2))
    model = await honesty.fit_gap_model()
    assert not model["ready"] and honesty.predict_retention(model, Alpha(id="x"), None) is None


async def test_assess_stores_the_figures(dbenv) -> None:
    rng = np.random.default_rng(5)
    daily = list(rng.normal(0.1, 1, 1250))
    async with db.write() as s:
        s.add(Alpha(id="Z1", sharpe=1.6, region="USA", delay=1))
        s.add(AlphaPnL(alpha_id="Z1", dates=[str(i) for i in range(1250)], daily=daily, n_days=1250))
    out = await honesty.assess("Z1")
    async with db.session() as s:
        a = await s.get(Alpha, "Z1")
    assert a.wf_ratio == out["wf_ratio"] and a.dsr == out["dsr"] and a.n_trials == 1 and 0 < a.dsr <= 1


# ---------------------------------------------------------------- learning from your review

async def test_review_examples_carry_approvals_rejections_and_edits(dbenv) -> None:
    from alphago.knowledge.pack import EDIT_MARK, review_examples
    async with db.write() as s:
        s.add(AutopilotRun(id=1, mode="assisted"))
        s.add(Batch(id=1, run_id=1))
        s.add(Alpha(id="AP1", verdict="strong", sharpe=1.7, score=110))
        rows = [
            Candidate(id=1, expression="rank(ts_delta(sales, 66))", status="done", alpha_id="AP1", batch_id=1),
            Candidate(id=2, expression="rank(close ^ 3)", status="rejected_by_user", batch_id=1),
            Candidate(id=3, expression="rank(volume)", status="rejected_by_user", error=EDIT_MARK, batch_id=1),
            Candidate(id=4, expression="group_rank(ts_rank(volume, 22), sector)", status="done", parent_id=3,
                      batch_id=1),
        ]
        for c in rows:
            c.settings, c.settings_hash, c.region, c.delay = BASE, "h", "USA", 1
            s.add(c)
    text = await review_examples("USA")
    assert "rank(ts_delta(sales, 66))  → strong, Sharpe 1.70" in text
    assert "Rejected before any simulation" in text and "rank(close ^ 3)" in text
    assert "rank(volume)\n  → group_rank(ts_rank(volume, 22), sector)" in text
    assert await review_examples("EUR") == ""


async def test_ideation_prompt_includes_the_review(dbenv, monkeypatch) -> None:
    from alphago.engine import generators
    from alphago.llm.schemas import AlphaIdea, IdeaBatch
    prompts: list[str] = []

    async def fake_review(region, per_kind=6, max_chars=180):  # noqa: ANN001, ANN202
        return "How the researcher reviewed earlier AI ideas — MARKER"

    async def fake_generate_json(prompt, schema, **kw):  # noqa: ANN001, ANN202
        prompts.append(prompt)
        return IdeaBatch(ideas=[AlphaIdea(expression="rank(close)", idea="i", mechanism="m", data_rationale="d",
                                          operator_rationale="o")])

    monkeypatch.setattr(generators.pack, "review_examples", fake_review)
    monkeypatch.setattr(generators.llm, "generate_json", fake_generate_json)
    await generators.llm_ideas(BASE, dataset_id=None, n=1, goal="g")
    assert "MARKER" in prompts[0]


# ---------------------------------------------------------------- AI self-check

async def test_self_check_drops_at_most_a_quarter(dbenv, monkeypatch) -> None:
    from alphago.engine import generators
    from alphago.llm.schemas import AlphaIdea, BatchCritique, IdeaReview
    ideas = [AlphaIdea(expression=f"rank(ts_delta(close, {w}))", idea="i", mechanism="m", data_rationale="d",
                       operator_rationale="o") for w in range(2, 10)]

    async def critic(prompt, schema, **kw):  # noqa: ANN001, ANN202
        assert kw["purpose"] == "self_check" and "[7] rank(ts_delta(close, 9))" in prompt
        return BatchCritique(reviews=[IdeaReview(index=i, keep=False, issues=["duplicate"] * (i % 3 + 1),
                                                 reason=f"dup {i}") for i in range(6)])

    monkeypatch.setattr(generators.llm, "generate_json", critic)
    report: dict = {}
    kept = await generators.self_check(ideas, "ctx", None, report)
    assert len(kept) == 6 and len(report["self_check"]) == 2          # 6 flagged, but only 25% of 8 dropped
    assert all(len(d["issues"]) == 3 for d in report["self_check"])  # the worst ones go first


async def test_self_check_failure_keeps_every_idea(dbenv, monkeypatch) -> None:
    from alphago.engine import generators
    from alphago.llm import LLMError
    from alphago.llm.schemas import AlphaIdea

    async def broken(prompt, schema, **kw):  # noqa: ANN001, ANN202
        raise LLMError("rate limited", 429)

    monkeypatch.setattr(generators.llm, "generate_json", broken)
    ideas = [AlphaIdea(expression="rank(close)", idea="i", mechanism="m", data_rationale="d",
                       operator_rationale="o")] * 4
    assert len(await generators.self_check(ideas, "ctx", None)) == 4
    assert settings.llm.self_check is True


async def test_v1_database_gets_the_new_columns(tmp_path) -> None:
    import sqlite3
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE alphas (id VARCHAR(32) PRIMARY KEY, code TEXT, sharpe FLOAT)")
    con.execute("CREATE TABLE optimization_attempts (id INTEGER PRIMARY KEY, parent_alpha_id VARCHAR(32), "
                "candidate_id INTEGER, kind VARCHAR(48), detail TEXT)")
    con.execute("INSERT INTO alphas (id, code, sharpe) VALUES ('OLD1', 'rank(close)', 1.4)")
    con.commit()
    con.close()
    await db.init(path)
    try:
        async with db.session() as s:
            a = await s.get(Alpha, "OLD1")
            assert a.sharpe == 1.4 and a.robust_state is None and a.expanded_at is None and a.is_checks is None
            from alphago.db.models import OptimizationAttempt
            assert await s.scalar(select(OptimizationAttempt)) is None
    finally:
        await db.close()
    con = sqlite3.connect(path)
    alpha_cols = {r[1] for r in con.execute('PRAGMA table_info("alphas")')}
    opt_cols = {r[1] for r in con.execute('PRAGMA table_info("optimization_attempts")')}
    con.close()
    assert {"robust_state", "est_self_corr", "wf_ratio", "dsr", "cluster", "expanded_at"} <= alpha_cols
    assert {"fix", "target_check", "outcome", "parent_snapshot"} <= opt_cols
