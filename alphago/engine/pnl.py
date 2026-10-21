"""Daily in-sample PnL of submitted alphas: fetch once, derive statistics, combine portfolios.

This is the backbone of Osmosis planning, the out-of-sample outlook and Power Pool picking. BRAIN shows little
out-of-sample detail, so these pages reason from each alpha's in-sample daily PnL ("last year" = the last year of
that data, not the last calendar year).
"""

from __future__ import annotations

import asyncio
import logging
import math
import warnings
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import select

from ..brain import brain
from ..db import db
from ..db.models import Alpha, AlphaPnL
from ..events import bus
from ..jobs.queue import JobContext, jobs

log = logging.getLogger(__name__)

TRADING_DAYS = 252
SPARK_POINTS = 60


# ---------------------------------------------------------------- parsing and statistics

def parse_records(rows: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
    """BRAIN `pnl` record set (cumulative $ PnL by date) -> (dates, daily increments)."""
    if not rows:
        return [], []
    keys = list(rows[0].keys())
    dkey = next((k for k in keys if k.lower() == "date"), None)
    if dkey is None:
        return [], []
    value_keys = [k for k in keys if k != dkey]
    vkey = next((k for k in value_keys if k.lower() == "pnl"), None) or next(
        (k for k in value_keys if isinstance(rows[0].get(k), (int, float))), None)
    if vkey is None:
        return [], []
    cum: dict[str, float] = {}
    for r in rows:
        d, v = r.get(dkey), r.get(vkey)
        if d is None or v is None:
            continue
        try:
            cum[str(d)[:10]] = float(v)
        except (TypeError, ValueError):
            continue
    dates = sorted(cum)
    if len(dates) < 2:
        return [], []
    daily = [cum[d] - cum[p] for p, d in zip(dates, dates[1:])]
    return dates[1:], daily


def sharpe(values: Iterable[float]) -> float | None:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 20:
        return None
    sd = arr.std(ddof=1)
    if sd <= 0:
        return None
    return float(math.sqrt(TRADING_DAYS) * arr.mean() / sd)


def max_drawdown(daily: np.ndarray) -> float:
    if daily.size == 0:
        return 0.0
    cum = np.cumsum(daily)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float(np.max(peak - cum)) if cum.size else 0.0


def spark(daily: np.ndarray, points: int = SPARK_POINTS) -> list[float]:
    if daily.size == 0:
        return []
    cum = np.cumsum(daily)
    idx = np.linspace(0, cum.size - 1, num=min(points, cum.size)).round().astype(int)
    return [round(float(cum[i]), 2) for i in idx]


def compute_stats(dates: list[str], daily: list[float]) -> dict[str, Any]:
    arr = np.asarray(daily, dtype=float)
    out: dict[str, Any] = {"n_days": int(arr.size), "sharpe": None, "sharpe_early": None, "sharpe_2y": None,
                           "sharpe_1y": None, "ann_pnl": None, "pnl_1y": None, "max_dd": None, "calmar": None,
                           "spark": []}
    if arr.size < 2:
        return out
    last = date.fromisoformat(dates[-1])
    cut2 = (last - timedelta(days=730)).isoformat()
    cut1 = (last - timedelta(days=365)).isoformat()
    in2 = np.array([d > cut2 for d in dates])
    in1 = np.array([d > cut1 for d in dates])
    ann = float(arr.mean() * TRADING_DAYS)
    dd = max_drawdown(arr)
    out.update(
        sharpe=sharpe(arr), sharpe_early=sharpe(arr[~in2]) if (~in2).sum() >= 60 else None,
        sharpe_2y=sharpe(arr[in2]), sharpe_1y=sharpe(arr[in1]), ann_pnl=ann, pnl_1y=float(arr[in1].sum()),
        max_dd=dd, calmar=(ann / dd) if dd > 0 else None, spark=spark(arr))
    return out


# ---------------------------------------------------------------- storage

async def stored(alpha_ids: Iterable[str]) -> dict[str, AlphaPnL]:
    ids = list(alpha_ids)
    if not ids:
        return {}
    out: dict[str, AlphaPnL] = {}
    async with db.session() as s:
        for i in range(0, len(ids), 500):
            for row in await s.scalars(select(AlphaPnL).where(AlphaPnL.alpha_id.in_(ids[i:i + 500]))):
                out[row.alpha_id] = row
    return out


async def save(alpha_id: str, dates: list[str], daily: list[float]) -> AlphaPnL:
    st = compute_stats(dates, daily)
    async with db.write() as s:
        row = await s.get(AlphaPnL, alpha_id)
        if row is None:
            row = AlphaPnL(alpha_id=alpha_id, dates=dates, daily=daily)
            s.add(row)
        row.dates, row.daily, row.fetched_at = dates, [round(x, 4) for x in daily], datetime.now(timezone.utc)
        for k, v in st.items():
            setattr(row, k, v)
    return row


async def fetch(alpha_id: str, *, force: bool = False, keep_raw: bool = True) -> AlphaPnL | None:
    """Daily PnL for one alpha: stored row, else the legacy KV cache, else BRAIN's `pnl` record set.

    keep_raw=False skips the raw copy in the KV table (used for research candidates, which are many)."""
    if not force:
        have = (await stored([alpha_id])).get(alpha_id)
        if have is not None:
            return have
    rows = None if force else await db.kv_get(f"pnl:{alpha_id}")
    if not rows:
        rows = await brain.alphas.recordset(alpha_id, "pnl")
        if keep_raw:
            await db.kv_set(f"pnl:{alpha_id}", rows)
    dates, daily = parse_records(rows or [])
    if not dates:
        return None
    return await save(alpha_id, dates, daily)


async def fetch_many(alpha_ids: list[str], *, force: bool = False, concurrency: int = 3,
                     progress=None) -> tuple[int, list[str]]:  # noqa: ANN001  (async callback(done, total))
    sem = asyncio.Semaphore(concurrency)
    done = 0
    errors: list[str] = []

    async def one(aid: str) -> None:
        nonlocal done
        async with sem:
            try:
                await fetch(aid, force=force)
            except Exception as exc:
                errors.append(f"{aid}: {getattr(exc, 'message', exc)}")
            done += 1
            if progress and (done % 5 == 0 or done == len(alpha_ids)):
                await progress(done, len(alpha_ids))

    await asyncio.gather(*(one(a) for a in alpha_ids))
    return done - len(errors), errors


async def active_alphas(region: str | None = None, delay: int | None = None) -> list[Alpha]:
    stmt = select(Alpha).where(Alpha.status == "ACTIVE")
    if region:
        stmt = stmt.where(Alpha.region == region)
    if delay is not None:
        stmt = stmt.where(Alpha.delay == delay)
    async with db.session() as s:
        return list(await s.scalars(stmt))


async def coverage() -> tuple[int, int]:
    """(ACTIVE alphas with daily PnL, ACTIVE alphas)."""
    alphas = await active_alphas()
    have = await stored(a.id for a in alphas)
    return len(have), len(alphas)


# ---------------------------------------------------------------- portfolios

def frame(rows: dict[str, AlphaPnL]) -> pd.DataFrame:
    """Dates × alphas of daily PnL; NaN where an alpha has no data that day."""
    cols = {aid: pd.Series(r.daily, index=pd.Index(r.dates, name="date"), dtype=float)
            for aid, r in rows.items() if r.dates}
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def risk_weights(df: pd.DataFrame, ids: list[str]) -> dict[str, float]:
    """Equal-risk weights (1 / daily PnL volatility), normalised to sum to 1."""
    vol = df[ids].std(ddof=1).replace(0, np.nan)
    w = (1.0 / vol).fillna(0.0)
    total = float(w.sum())
    return {k: (float(v) / total if total > 0 else 1.0 / len(ids)) for k, v in w.items()}


def combo_series(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    ids = [k for k in weights if k in df.columns]
    if not ids:
        return pd.Series(dtype=float)
    sub = df[ids]
    active = sub.notna().any(axis=1)
    combo = (sub.fillna(0.0) * pd.Series(weights)[ids]).sum(axis=1)
    return combo[active]


def combo_sharpe(df: pd.DataFrame, weights: dict[str, float]) -> float | None:
    return sharpe(combo_series(df, weights).to_numpy())


def combo_sharpe_last(df: pd.DataFrame, weights: dict[str, float], days: int = 730) -> float | None:
    series = combo_series(df, weights)
    if series.empty:
        return None
    last = date.fromisoformat(str(series.index[-1]))
    cut = (last - timedelta(days=days)).isoformat()
    return sharpe(series[series.index > cut].to_numpy())


def correlations(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return df[ids].corr(min_periods=60) if ids else pd.DataFrame()


def avg_correlation(df: pd.DataFrame, ids: list[str]) -> float | None:
    if len(ids) < 2:
        return None
    c = correlations(df, ids).to_numpy()
    mask = ~np.eye(len(ids), dtype=bool) & np.isfinite(c)
    return float(c[mask].mean()) if mask.any() else None


def greedy_select(df: pd.DataFrame, candidates: list[str], max_n: int, *, min_n: int = 1,
                  score: dict[str, float] | None = None) -> list[str]:
    """Forward selection: start from the best single alpha, add whichever alpha raises the combo Sharpe most
    (equal-risk weights); stop when nothing improves it (after `min_n`) or `max_n` is reached."""
    pool = [c for c in candidates if c in df.columns and df[c].notna().sum() >= 60]
    if not pool:
        return []
    mask = df[pool].notna().to_numpy()
    x = df[pool].fillna(0.0).to_numpy()
    vol = df[pool].std(ddof=1).to_numpy()
    iv = np.where(vol > 0, 1.0 / np.where(vol > 0, vol, 1.0), 0.0)
    xw = x * iv                          # equal-risk contributions; Sharpe is scale-free, so no renormalising needed

    def sharpe_cols(values: np.ndarray, active: np.ndarray) -> np.ndarray:
        v = np.where(active, values, np.nan)
        n = active.sum(axis=0)
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(v, axis=0)
            sd = np.nanstd(v, axis=0, ddof=1)
            s = math.sqrt(TRADING_DAYS) * mean / sd
        return np.where((n >= 20) & (sd > 0), s, -np.inf)

    singles = sharpe_cols(x, mask)
    if score:
        start = max(range(len(pool)), key=lambda i: score.get(pool[i], singles[i]))
    else:
        start = int(np.argmax(singles))
    chosen = [start]
    acc, act = xw[:, start].copy(), mask[:, start].copy()
    best = float(singles[start])
    while len(chosen) < max_n:
        trial = sharpe_cols(acc[:, None] + xw, act[:, None] | mask)
        trial[chosen] = -np.inf
        j = int(np.argmax(trial))
        if not np.isfinite(trial[j]):
            break
        if trial[j] <= best and len(chosen) >= min_n:
            break
        chosen.append(j)
        acc += xw[:, j]
        act |= mask[:, j]
        best = max(best, float(trial[j]))
    return [pool[i] for i in chosen]


# ---------------------------------------------------------------- job

async def prepare_data(ctx: JobContext | None, params: dict) -> str:
    """Sync submitted alphas from BRAIN (fresh points/tags), then fetch daily PnL for any that lack it."""
    from .sync import sync_alphas
    if params.get("sync", True):
        if ctx:
            await ctx.update(0.02, "Reading submitted alphas from BRAIN")
        await sync_alphas(None, {"active_only": True})
    alphas = await active_alphas()
    have = await stored(a.id for a in alphas)
    force = bool(params.get("force"))
    todo = [a.id for a in alphas if force or a.id not in have]

    async def progress(done: int, total: int) -> None:
        if ctx:
            await ctx.update(0.1 + 0.9 * done / max(1, total), f"Daily PnL {done}/{total}")

    ok, errors = await fetch_many(todo, force=force, progress=progress)
    msg = f"{len(alphas)} submitted alphas · {len(have) + ok} with daily PnL" + (
        f" · {len(errors)} failed" if errors else "")
    bus.publish("sync", f"Prepared data: {msg}.", "success" if not errors else "warning")
    if errors and ok == 0 and todo:
        raise RuntimeError("; ".join(errors[:5]))
    return msg


jobs.register("prepare_pnl")(prepare_data)
