"""BRAIN-exact metrics and platform rules (engine/metrics.py, brain/platform.py, labs/fastexpr.py).

These rules were measured against BRAIN by alpha-harness; the port was also checked differentially against its
`vault/metrics.py` on 200 random alphas (identical to 1e-9). The cases below pin the rules themselves.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from alphago.brain import platform as P
from alphago.engine import metrics as M
from alphago.labs import fastexpr as F


def test_daily_series_keeps_first_day_and_stamps_turnover_a_day_early():
    pnl = [{"date": "2024-01-02", "pnl": 100.0}, {"date": "2024-01-03", "pnl": 150.0},
           {"date": "2024-01-04", "pnl": 120.0}]
    turn = [{"date": "2024-01-02", "turnover": 0.4}, {"date": "2024-01-03", "turnover": 0.2}]
    dates, daily, t = M.daily_series(pnl, turn)
    assert daily == [100.0, 50.0, -30.0]
    assert t == [1.0, 0.4, 0.2]                       # day 0 builds the book; then the previous day's row
    # no row after a held position: the book is sold whole once
    _, _, t2 = M.daily_series(pnl, turn[:1])
    assert t2 == [1.0, 0.4, 1.0]


def test_closing_days_spread_the_unexported_pnl():
    dates = ["2024-03-06", "2024-03-07", "2024-03-08"]           # Wed..Fri
    d, p, t, mask = M.closing_days(dates, [1.0, 1.0, 1.0], [0.1, 0.1, 0.1], end_date="2024-03-12", is_pnl=13.0)
    assert d[3:] == ["2024-03-11", "2024-03-12", "2024-03-13"]   # Mon, Tue (end), then the next weekday
    assert p[3:] == [pytest.approx(10 / 3)] * 3 and mask == [False] * 3 + [True] * 3
    # end on a weekend: the extra stamp is the end date itself
    d2, *_ = M.closing_days(dates, [1.0] * 3, [0.1] * 3, end_date="2024-03-09", is_pnl=4.0)
    assert d2[3:] == ["2024-03-09"]


def test_stats_follow_brain_formulas():
    rng = np.random.default_rng(1)
    pnl = rng.normal(5_000, 40_000, 500)
    turn = np.full(500, 0.05)
    st = M.stats(pnl, turn)
    assert st["returns"] == pytest.approx(pnl.mean() * 250 / 10_000_000)
    assert st["sharpe"] == pytest.approx(pnl.mean() / pnl.std(ddof=1) * math.sqrt(250))
    # turnover below 12.5 % is charged as 12.5 %
    assert st["fitness"] == pytest.approx(round(st["sharpe"], 2) * math.sqrt(abs(st["returns"]) / 0.125))
    assert st["margin"] == pytest.approx(pnl.sum() / (turn.sum() * 20_000_000))


def test_idle_days_before_the_first_trade_are_trimmed():
    st = M.stats([0, 0, 0, 10, -5, 7], [0, 0, 0, 0.1, 0.1, 0.1])
    assert st["days"] == 3


def test_after_cost_charges_each_days_turnover():
    net = M.after_cost([1000.0, 1000.0], [0.1, 0.0], bps=5)
    assert net == [pytest.approx(1000 - 0.1 * 20_000_000 * 5 / 10_000), 1000.0]


def test_k_ratio_rewards_steady_growth():
    steady = M.k_ratio([10.0] * 100 + [11.0] * 100)
    jumpy = M.k_ratio([0.0] * 99 + [1000.0] + [0.0] * 99 + [1100.0])
    assert steady is not None and jumpy is not None and steady > jumpy


def test_correlation_window_is_four_calendar_years():
    assert M.window_start(date(2026, 3, 1)) == date(2023, 1, 1)


def test_correlation_matrix_matches_pairwise_and_needs_250_days():
    rng = np.random.default_rng(2)
    m = rng.normal(0, 1, (400, 4))
    m[:200, 3] = np.nan                                   # only 200 shared days with the others
    m[:, 2] = m[:, 0] * 0.5 + rng.normal(0, 1, 400) * 0.5
    c = M.correlation_matrix(m)
    assert c[0, 2] == pytest.approx(M.pair_correlation(m[:, 0], m[:, 2]))
    assert np.isnan(c[0, 3]) and np.isnan(c[0, 0])


def test_portfolio_averages_only_alphas_already_trading():
    book, traded = M.combine(np.array([[1.0, 0.0], [1.0, 3.0]]), np.array([[0.1, 0.0], [0.1, 0.1]]))
    assert list(book) == [1.0, 2.0]                       # day 1: one alpha live; day 2: both


@pytest.mark.parametrize("region,delay,universe,expected", [
    ("USA", 1, "TOP3000", 0.08), ("USA", 1, "TOP1000", 0.06), ("USA", 1, "TOPSP500", 0.05),
    ("USA", 0, "TOP3000", 0.05), ("JPN", 1, "TOP1600", 0.06), ("EUR", 1, "TOPCS1600", 0.05),
    ("GLB", 1, "MINVOL1M", 0.05)])
def test_truncation_agent(region, delay, universe, expected):
    assert P.truncation_for(region, delay, universe) == expected


def test_investability_pairs_never_both_on():
    assert P.investability_pairs("USA") == [("OFF", "OFF"), ("OFF", "ON"), ("ON", "OFF")]
    assert P.investability_pairs("CHN") == [("OFF", "OFF"), ("ON", "OFF")]


def test_brain_verdict_ignores_informational_and_quota_checks():
    base = [{"name": "LOW_SHARPE", "result": "PASS"}, {"name": "PROD_CORRELATION", "result": "FAIL"},
            {"name": "REGULAR_SUBMISSION", "result": "FAIL"}]
    assert P.brain_verdict(base) == "submittable"
    assert P.brain_verdict(base + [{"name": "SELF_CORRELATION", "result": "PENDING"}]) == "pending"
    assert P.brain_verdict(base + [{"name": "LOW_FITNESS", "result": "ERROR"}]) == "refused"
    assert P.brain_verdict(base, "QUICK") == "refused"
    assert P.brain_verdict([]) is None
    assert P.is_near_miss([{"name": "LOW_FITNESS", "result": "FAIL"}, {"name": "LOW_SHARPE", "result": "PASS"}])


def test_power_pool_counting_rules():
    expr = "x = ts_backfill(close, 21); rank(-ts_delta(x, 5)) + group_rank(volume, sector)"
    assert F.operator_count(expr) == 5                    # ts_backfill is not counted
    assert F.data_fields(expr) == {"close", "volume"}     # sector is a grouping field
    assert "sector" in F.data_fields(expr, grouping=True)
    assert F.negated("a = rank(close); -a") == "a = rank(close); a"
    assert F.swap_field("ts_regression(close, open, 5, rettype = open)", "open", "vwap").count("vwap") == 1
