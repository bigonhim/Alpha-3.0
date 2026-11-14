"""BRAIN-exact figures from an alpha's daily PnL and turnover (ported from alpha-harness vault/metrics.py).

BRAIN annualises with 250 days on a $20M book (returns are on half the book), charges fitness against a turnover
floor of 12.5 %, measures correlations over the last four calendar years on at least 250 shared days, and counts a
few "closing days" after the last row its record sets export. These rules were measured against BRAIN; the older
`engine/pnl.py` statistics (252 days, $ figures) stay for the pages built on them.

Everything here is pure numpy on plain lists, so it is cheap to test and to reuse.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np

from ..brain.platform import (
    BOOK_SIZE,
    CORRELATION_WINDOW_YEARS,
    FITNESS_TURNOVER_FLOOR,
    MIN_OVERLAP,
    TRADING_DAYS,
)

HALF_BOOK = BOOK_SIZE / 2


# ---------------------------------------------------------------- record sets -> daily series

def _column(rows: list[dict[str, Any]], *names: str) -> tuple[str | None, str | None]:
    if not rows:
        return None, None
    keys = list(rows[0].keys())
    dkey = next((k for k in keys if k.lower() == "date"), None)
    vkey = next((k for k in keys for n in names if k.lower() == n), None)
    if vkey is None:
        vkey = next((k for k in keys if k != dkey and isinstance(rows[0].get(k), (int, float))), None)
    return dkey, vkey


def _by_date(rows: list[dict[str, Any]], *names: str) -> dict[str, float]:
    dkey, vkey = _column(rows, *names)
    out: dict[str, float] = {}
    if dkey is None or vkey is None:
        return out
    for r in rows:
        d, v = r.get(dkey), r.get(vkey)
        if d is None or v is None:
            continue
        try:
            out[str(d)[:10]] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def daily_series(pnl_rows: list[dict[str, Any]],
                 turnover_rows: list[dict[str, Any]] | None = None) -> tuple[list[str], list[float], list[float]]:
    """(dates, daily PnL, daily turnover) from BRAIN's cumulative `pnl` and daily `turnover` record sets.

    Daily PnL is the difference of the cumulative column starting from 0, so the first day keeps its value.
    BRAIN stamps each turnover row one trading day early: day i trades the turnover recorded on day i-1. The first
    day is charged 1.0 (the book being built) when it has a turnover row and the first recorded value is not 1.0;
    a day with no recorded turnover after a held position sells the book whole (1.0) once."""
    cum = _by_date(pnl_rows, "pnl")
    dates = sorted(cum)
    if not dates:
        return [], [], []
    pnl, prev = [], 0.0
    for d in dates:
        pnl.append(cum[d] - prev)
        prev = cum[d]
    recorded = _by_date(turnover_rows or [], "turnover")
    first_value = recorded[min(recorded)] if recorded else None
    turnover: list[float] = []
    holding = False
    for i, d in enumerate(dates):
        if i == 0:
            built = d in recorded and first_value != 1.0
            turnover.append(1.0 if built else 0.0)
            holding = built
        else:
            value = recorded.get(dates[i - 1])
            if value is None:
                turnover.append(1.0 if holding else 0.0)
                holding = False
            else:
                turnover.append(value)
                holding = True
    return dates, pnl, turnover


ROUNDING = 0.00005           # half the last decimal of BRAIN's four-decimal turnover


def calibrate(dates: Sequence[str], turnover: Sequence[float], yearly_rows: list[dict[str, Any]] | None,
              test_start: str | None = None) -> list[float]:
    """Scale each yearly-stats segment's turnover to the figure BRAIN reports for it.

    For delay-0 alphas with a group (or no) neutralization, and for CHN and GLB, the turnover record set differs
    from what BRAIN's figures use; its yearly figures are exact, so each year (the split year cut at the split day)
    is scaled to match. A segment already within BRAIN's rounding is left alone."""
    out = list(turnover)
    split_year = int(test_start[:4]) if test_start else None
    for row in yearly_rows or []:
        try:
            year = int(str(row.get("year")))
            target = float(row.get("turnover"))
        except (TypeError, ValueError):
            continue
        test = str(row.get("stage") or "").upper() == "TEST"
        idx = [i for i, d in enumerate(dates) if int(d[:4]) == year
               and (test_start is None or year != split_year or (d >= test_start) == test)]
        have = sum(out[i] for i in idx)
        if idx and have > 0 and abs(have / len(idx) - target) > ROUNDING:
            factor = target * len(idx) / have
            for i in idx:
                out[i] *= factor
    return out


def closing_days(dates: list[str], pnl: list[float], turnover: list[float], *, end_date: str | None,
                 is_pnl: float | None, test_start: str | None = None,
                 test_turnover: float | None = None) -> tuple[list[str], list[float], list[float], list[bool]]:
    """Append the days BRAIN counts after the last exported row, so totals match its IS figures.

    Stamps: the weekdays in (last, end], then one more on the end date, or on the next weekday when the end date
    was itself one of them. The PnL BRAIN reports (`is.pnl`) but the record set lacks is spread evenly over them;
    their turnover makes the test-period mean match BRAIN's. Returns the extended series and a mask of added days."""
    mask = [False] * len(dates)
    if not dates or not end_date or is_pnl is None:
        return list(dates), list(pnl), list(turnover), mask
    last = date.fromisoformat(dates[-1])
    end = date.fromisoformat(str(end_date)[:10])
    between = [last + timedelta(days=k) for k in range(1, (end - last).days + 1)]
    between = [d for d in between if d.weekday() < 5]
    after = end + timedelta(days=1)
    while after.weekday() >= 5:
        after += timedelta(days=1)
    stamps = [*between, after if end in between else end]
    if stamps[-1] <= last:
        return list(dates), list(pnl), list(turnover), mask
    n = len(stamps)
    each_pnl = (float(is_pnl) - sum(pnl)) / n
    each_turn = 0.0
    if test_turnover is not None and test_start:
        tested = [t for d, t in zip(dates, turnover) if d >= test_start]
        each_turn = min(2.0, max(0.0, (float(test_turnover) * (len(tested) + n) - sum(tested)) / n))
    return (list(dates) + [d.isoformat() for d in stamps], list(pnl) + [each_pnl] * n,
            list(turnover) + [each_turn] * n, mask + [True] * n)


# ---------------------------------------------------------------- statistics

def _trim(pnl: np.ndarray, turnover: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = np.flatnonzero((pnl != 0) | (turnover != 0))
    if active.size == 0:
        return pnl[:0], turnover[:0]
    return pnl[active[0]:], turnover[active[0]:]


def sharpe(pnl: Iterable[float]) -> float | None:
    arr = np.asarray(list(pnl), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    sd = arr.std(ddof=1)
    if not sd > 0:
        return None
    return float(arr.mean() / sd * math.sqrt(TRADING_DAYS))


def stats(pnl: Sequence[float], turnover: Sequence[float] | None = None, *, trim: bool = True) -> dict[str, Any]:
    """BRAIN's summary figures: sharpe, fitness, returns, turnover, drawdown, margin (and days)."""
    p = np.asarray(pnl, dtype=float)
    t = np.asarray(turnover if turnover is not None else np.zeros_like(p), dtype=float)
    p, t = np.nan_to_num(p), np.nan_to_num(t)
    if trim:
        p, t = _trim(p, t)
    n = int(p.size)
    out: dict[str, Any] = {"sharpe": None, "fitness": None, "returns": None, "turnover": None, "drawdown": None,
                           "margin": None, "days": n}
    if n == 0:
        return out
    returns = float(p.mean() * TRADING_DAYS / HALF_BOOK)
    mean_turn = float(t.sum() / n)
    cum = np.cumsum(p)
    drawdown = float(np.max(np.maximum.accumulate(cum) - cum) / HALF_BOOK)
    traded = float(t.sum())
    sh = sharpe(p)
    out.update(returns=returns, turnover=mean_turn, drawdown=drawdown,
               margin=float(p.sum() / (traded * BOOK_SIZE)) if traded > 0 else 0.0, sharpe=sh)
    if sh is not None:
        out["fitness"] = round(sh, 2) * math.sqrt(abs(returns) / max(mean_turn, FITNESS_TURNOVER_FLOOR))
    return out


def k_ratio(pnl: Sequence[float], turnover: Sequence[float] | None = None) -> float | None:
    """Kestner's K-ratio (2013 form): slope of cumulative PnL against the day number over its standard error times
    the number of days, annualised with 250 days. From the first trade, like every other figure here."""
    p = np.nan_to_num(np.asarray(pnl, dtype=float))
    t = np.nan_to_num(np.asarray(turnover, dtype=float)) if turnover is not None else np.zeros_like(p)
    active = np.flatnonzero((p != 0) | (t != 0))
    if active.size == 0:
        return None
    y = np.cumsum(p[active[0]:])
    n = y.size
    if n < 3:
        return None
    x = np.arange(1, n + 1, dtype=float)
    dx = x - x.mean()
    sxx = float(dx @ dx)
    slope = float(dx @ (y - y.mean())) / sxx
    resid = y - y.mean() - slope * dx
    error = math.sqrt(float(resid @ resid) / (n - 2) / sxx)
    return slope / (error * n) * math.sqrt(TRADING_DAYS) if error > 0 else None


def after_cost(pnl: Sequence[float], turnover: Sequence[float], bps: float = 5.0) -> list[float]:
    """Daily PnL net of trading cost: each day pays its own turnover × book × bps."""
    p = np.asarray(pnl, dtype=float)
    t = np.nan_to_num(np.asarray(turnover, dtype=float))
    return list(p - t * BOOK_SIZE * float(bps) / 10_000)


def windows(dates: Sequence[str], pnl: Sequence[float], turnover: Sequence[float],
            split: str | None) -> dict[str, dict[str, Any]]:
    """IS / train (≤ split) / test (≥ split) figures; both halves include the split day, as BRAIN's do."""
    p, t = list(pnl), list(turnover)
    out = {"is": stats(p, t)}
    if split:
        tr = [i for i, d in enumerate(dates) if d <= split]
        te = [i for i, d in enumerate(dates) if d >= split]
        out["train"] = stats([p[i] for i in tr], [t[i] for i in tr])
        out["test"] = stats([p[i] for i in te], [t[i] for i in te])
    return out


def yearly(dates: Sequence[str], pnl: Sequence[float], turnover: Sequence[float],
           split: str | None = None) -> list[dict[str, Any]]:
    """Per calendar year (and stage around the split), untrimmed, newest last."""
    groups: dict[tuple[int, str], list[int]] = {}
    for i, d in enumerate(dates):
        stage = "IS" if not split else ("TEST" if d >= split else "TRAIN")
        groups.setdefault((int(d[:4]), stage), []).append(i)
    rows = []
    for (year, stage), idx in sorted(groups.items()):
        st = stats([pnl[i] for i in idx], [turnover[i] for i in idx], trim=False)
        rows.append({"year": year, "stage": stage, **st})
    return rows


def figures(dates: Sequence[str], pnl: Sequence[float], turnover: Sequence[float], *, split: str | None = None,
            bps: float = 5.0) -> dict[str, Any]:
    """Everything a table row needs, gross and net of `bps`."""
    w = windows(dates, pnl, turnover, split)
    net = after_cost(pnl, turnover, bps)
    wn = windows(dates, net, turnover, split)
    return {"is": w["is"], "train": w.get("train"), "test": w.get("test"), "k_ratio": k_ratio(pnl, turnover),
            "after_cost": {"bps": bps, **wn["is"], "train_sharpe": (wn.get("train") or {}).get("sharpe"),
                           "test_sharpe": (wn.get("test") or {}).get("sharpe"), "k_ratio": k_ratio(net, turnover)}}


# ---------------------------------------------------------------- correlation

def window_start(last: date) -> date:
    """First day of BRAIN's correlation window: Jan 1 of the 4th calendar year back (inclusive)."""
    return date(last.year - CORRELATION_WINDOW_YEARS + 1, 1, 1)


def pair_correlation(a: Sequence[float], b: Sequence[float], min_overlap: int = MIN_OVERLAP) -> float | None:
    """Pearson on the days both have a value; None below `min_overlap` shared days or when one is flat."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both = np.isfinite(x) & np.isfinite(y)
    if both.sum() < min_overlap:
        return None
    x, y = x[both], y[both]
    sx, sy = x.std(), y.std()
    if not (sx > 0 and sy > 0):
        return None
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))


def correlation_matrix(matrix: np.ndarray, min_overlap: int = MIN_OVERLAP) -> np.ndarray:
    """Pairwise Pearson of the columns of a days × alphas matrix (NaN = no value that day), vectorised.

    Each pair uses only the days both columns have; pairs with fewer than `min_overlap` shared days, or a flat
    side, are NaN. The diagonal is NaN."""
    m = np.asarray(matrix, dtype=float)
    have = np.isfinite(m).astype(float)
    z = np.where(have > 0, m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n = have.T @ have
        sx = z.T @ have                     # Σx over days shared with column j  (row i = x of column i)
        sy = sx.T
        sxx = (z * z).T @ have
        syy = sxx.T
        sxy = z.T @ z
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        corr = cov / np.sqrt(vx * vy)
    corr[(n < min_overlap) | ~(vx > 1e-18) | ~(vy > 1e-18)] = np.nan
    np.fill_diagonal(corr, np.nan)
    return np.clip(corr, -1.0, 1.0)


# ---------------------------------------------------------------- portfolios

class Frame:
    """Days × alphas of daily PnL and turnover on a union calendar (NaN where an alpha has no row)."""

    def __init__(self, series: dict[str, tuple[list[str], list[float], list[float]]],
                 closing: dict[str, list[bool]] | None = None) -> None:
        self.ids = [k for k, v in series.items() if v[0]]
        self.dates = sorted({d for k in self.ids for d in series[k][0]})
        pos = {d: i for i, d in enumerate(self.dates)}
        shape = (len(self.dates), len(self.ids))
        self.pnl = np.full(shape, np.nan)
        self.turnover = np.full(shape, np.nan)
        self.closing = np.zeros(shape, dtype=bool)
        for j, k in enumerate(self.ids):
            ds, p, t = series[k]
            rows = [pos[d] for d in ds]
            self.pnl[rows, j] = p
            self.turnover[rows, j] = t
            if closing and k in closing:
                self.closing[rows, j] = closing[k]

    def exported(self) -> np.ndarray:
        """PnL with closing days hidden (correlations use only what the record sets export)."""
        return np.where(self.closing, np.nan, self.pnl)

    def since(self, start: date) -> slice:
        iso = start.isoformat()
        first = next((i for i, d in enumerate(self.dates) if d >= iso), len(self.dates))
        return slice(first, None)


def combine(pnl: np.ndarray, turnover: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight book over the alphas already trading each day: (daily PnL, daily turnover)."""
    p, t = np.nan_to_num(pnl), np.nan_to_num(turnover)
    started = np.maximum.accumulate((p != 0) | (t != 0), axis=0)
    live = np.maximum(started.sum(axis=1), 1)
    return p.sum(axis=1) / live, t.sum(axis=1) / live


def incremental_sharpe(pnl: np.ndarray, turnover: np.ndarray) -> list[float | None]:
    """Sharpe of the equal-weight book minus the Sharpe without each alpha."""
    whole = sharpe(combine(pnl, turnover)[0])
    out: list[float | None] = []
    for j in range(pnl.shape[1]):
        keep = [i for i in range(pnl.shape[1]) if i != j]
        if not keep or whole is None:
            out.append(None)
            continue
        rest = sharpe(combine(pnl[:, keep], turnover[:, keep])[0])
        out.append(None if rest is None else whole - rest)
    return out


def portfolio_correlation(pnl: np.ndarray) -> list[float | None]:
    """Each alpha against the equal-weight book of the others (averaged over those already trading)."""
    p = np.nan_to_num(pnl)
    started = np.maximum.accumulate(p != 0, axis=0)
    total = p.sum(axis=1)
    live = started.sum(axis=1)
    out: list[float | None] = []
    for j in range(pnl.shape[1]):
        others = live - started[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            rest = np.where(others > 0, (total - p[:, j]) / others, np.nan)
        own = np.where(np.isfinite(pnl[:, j]), pnl[:, j], np.nan)
        out.append(pair_correlation(own, rest))
    return out


def portfolio(series: dict[str, tuple[list[str], list[float], list[float]]], *,
              closing: dict[str, list[bool]] | None = None, split: str | None = None,
              bps: float = 5.0, grid_limit: int = 150) -> dict[str, Any]:
    """An equal-weight portfolio of the given alphas, the way alpha-harness' Portfolio screen computes it."""
    fr = Frame(series, closing)
    if not fr.ids:
        return {"alphas": [], "dates": []}
    book, traded = combine(fr.pnl, fr.turnover)
    net = np.asarray(after_cost(book, traded, bps))
    last = date.fromisoformat(fr.dates[-1])
    window = fr.since(window_start(last))
    corr = correlation_matrix(fr.exported()[window])
    pairs = [(fr.ids[i], fr.ids[j], float(corr[i, j])) for i in range(len(fr.ids)) for j in range(i + 1, len(fr.ids))
             if np.isfinite(corr[i, j])]
    pairs.sort(key=lambda x: -x[2])
    out: dict[str, Any] = {
        "alphas": fr.ids, "dates": fr.dates, "split": split,
        **{k: v for k, v in windows(fr.dates, list(book), list(traded), split).items()},
        "after_cost": windows(fr.dates, list(net), list(traded), split),
        "yearly": yearly(fr.dates, list(book), list(traded), split),
        "curve": list(np.cumsum(book)), "after_cost_curve": list(np.cumsum(net)),
        "k_ratio": k_ratio(book, traded),
        "measured_pairs": len(pairs),
        "highest": pairs[0] if pairs else None, "lowest": pairs[-1] if pairs else None,
        "max_pairwise": {fr.ids[j]: (None if not np.isfinite(corr[:, j]).any() else float(np.nanmax(corr[:, j])))
                         for j in range(len(fr.ids))},
        "portfolio_corr": dict(zip(fr.ids, portfolio_correlation(fr.exported()[window]))),
        "incremental_sharpe": dict(zip(fr.ids, incremental_sharpe(fr.pnl, fr.turnover))),
    }
    if len(fr.ids) <= grid_limit:
        out["correlation"] = [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in corr]
    else:
        out["top_pairs"] = pairs[:20]
    return out
