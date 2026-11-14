"""BRAIN-exact daily series per alpha: download once, keep, and derive the vault figures from them.

Every alpha-harness analysis (portfolio, Power Pool correlation, submission order, after-cost columns, K-ratio)
reads these rows. They come from three record sets — `pnl` (cumulative), `turnover` and `yearly-stats` (to
calibrate turnover) — and are stored on `AlphaPnL.s_*`; the closing days BRAIN counts after the last exported
row are added when a series is read, from the alpha's own `end_date`, `is_pnl`, `test_start` and `test_turnover`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from ..brain import brain
from ..brain.errors import BrainError
from ..brain.platform import QUICK_MODE
from ..db import db
from ..db.models import Alpha, AlphaPnL
from ..jobs.queue import JobContext, jobs
from . import metrics

log = logging.getLogger(__name__)

Series = tuple[list[str], list[float], list[float]]


async def _recordset(alpha_id: str, name: str, *, cache: bool) -> list[dict[str, Any]]:
    key = f"{name}:{alpha_id}" if name != "yearly-stats" else f"yearly:{alpha_id}"
    if cache:
        have = await db.kv_get(key)
        if have:
            return have
    rows = await brain.alphas.recordset(alpha_id, name)
    if cache and rows:
        await db.kv_set(key, rows)
    return rows or []


async def fetch_series(alpha_id: str, *, force: bool = False, turnover: bool = True) -> AlphaPnL | None:
    """Download (or reuse) an alpha's daily PnL and turnover and store the BRAIN-exact series.

    `turnover=False` downloads the PnL only (one request), enough for correlations."""
    async with db.session() as s:
        row = await s.get(AlphaPnL, alpha_id)
        alpha = await s.get(Alpha, alpha_id)
    if row is not None and row.s_dates and not force and (row.has_turnover or not turnover):
        return row
    pnl_rows = await _recordset(alpha_id, "pnl", cache=not force)
    if not pnl_rows:
        return None
    turn_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    if turnover:
        try:
            turn_rows = await _recordset(alpha_id, "turnover", cache=False)
        except BrainError as exc:
            log.info("turnover of %s unavailable: %s", alpha_id, exc.message)
        if turn_rows and not (alpha and str(alpha.simulation_mode or "").upper() == QUICK_MODE):
            try:
                yearly_rows = await _recordset(alpha_id, "yearly-stats", cache=True)
            except BrainError:
                yearly_rows = []
    dates, pnl, turn = metrics.daily_series(pnl_rows, turn_rows)
    if turn_rows:
        turn = metrics.calibrate(dates, turn, yearly_rows, alpha.test_start if alpha else None)
    from .pnl import save as save_legacy
    if len(dates) > 1 and (row is None or not row.dates):       # the older pages' statistics, same download
        await save_legacy(alpha_id, dates[1:], pnl[1:])
    async with db.write() as s:
        row = await s.get(AlphaPnL, alpha_id)
        if row is None:
            row = AlphaPnL(alpha_id=alpha_id, dates=[], daily=[])
            s.add(row)
        row.s_dates, row.s_pnl = dates, [round(x, 4) for x in pnl]
        row.s_turnover = [round(x, 6) for x in turn]
        row.has_turnover = bool(turn_rows)
        row.series_at = datetime.now(timezone.utc)
    await refresh_figures([alpha_id])
    async with db.session() as s:
        return await s.get(AlphaPnL, alpha_id)


def with_closing(row: AlphaPnL, alpha: Alpha | None) -> tuple[list[str], list[float], list[float], list[bool]]:
    dates, pnl, turn = list(row.s_dates or []), list(row.s_pnl or []), list(row.s_turnover or [])
    if alpha is None:
        return dates, pnl, turn, [False] * len(dates)
    return metrics.closing_days(dates, pnl, turn, end_date=alpha.end_date, is_pnl=alpha.is_pnl,
                                test_start=alpha.test_start, test_turnover=alpha.test_turnover)


async def load(alpha_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Stored series with closing days, per alpha: {dates, pnl, turnover, closing, alpha, has_turnover}."""
    ids = list(dict.fromkeys(alpha_ids))
    out: dict[str, dict[str, Any]] = {}
    async with db.session() as s:
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            rows = {r.alpha_id: r for r in await s.scalars(select(AlphaPnL).where(AlphaPnL.alpha_id.in_(chunk)))}
            alphas = {a.id: a for a in await s.scalars(select(Alpha).where(Alpha.id.in_(chunk)))}
            for aid in chunk:
                row = rows.get(aid)
                if row is None or not row.s_dates:
                    continue
                d, p, t, c = with_closing(row, alphas.get(aid))
                out[aid] = {"dates": d, "pnl": p, "turnover": t, "closing": c, "alpha": alphas.get(aid),
                            "has_turnover": bool(row.has_turnover)}
    return out


async def refresh_figures(alpha_ids: Iterable[str], bps: float = 5.0) -> int:
    """Recompute the stored after-cost Sharpe (5 bps) and K-ratio of alphas whose series has turnover."""
    loaded = await load(alpha_ids)
    done = 0
    async with db.write() as s:
        for aid, ser in loaded.items():
            alpha = await s.get(Alpha, aid)
            if alpha is None:
                continue
            alpha.k_ratio = metrics.k_ratio(ser["pnl"], ser["turnover"])
            if ser["has_turnover"]:
                net = metrics.after_cost(ser["pnl"], ser["turnover"], bps)
                alpha.after_cost_5bps = metrics.stats(net, ser["turnover"])["sharpe"]
            done += 1
    return done


async def fetch_many(alpha_ids: list[str], *, turnover: bool = True, force: bool = False, concurrency: int = 4,
                     ctx: JobContext | None = None) -> tuple[int, list[str]]:
    sem = asyncio.Semaphore(concurrency)
    done = 0
    errors: list[str] = []

    async def one(aid: str) -> None:
        nonlocal done
        async with sem:
            try:
                await fetch_series(aid, force=force, turnover=turnover)
            except BrainError as exc:
                errors.append(f"{aid}: {exc.message}")
            except Exception as exc:
                log.exception("series download failed for %s", aid)
                errors.append(f"{aid}: {exc}")
            done += 1
            if ctx and (done % max(1, len(alpha_ids) // 50) == 0 or done == len(alpha_ids)):
                await ctx.update(done / max(1, len(alpha_ids)), f"Daily PnL and turnover {done}/{len(alpha_ids)}")

    await asyncio.gather(*(one(a) for a in alpha_ids))
    return done - len(errors), errors


async def missing(alpha_ids: Iterable[str], *, turnover: bool = True) -> list[str]:
    """Which of these alphas have no stored series yet (absent ones are retried after a day)."""
    ids = list(alpha_ids)
    have: set[str] = set()
    async with db.session() as s:
        for i in range(0, len(ids), 500):
            for r in await s.scalars(select(AlphaPnL).where(AlphaPnL.alpha_id.in_(ids[i:i + 500]))):
                if r.s_dates and (r.has_turnover or not turnover):
                    have.add(r.alpha_id)
    return [a for a in ids if a not in have]


async def _job_fetch(ctx: JobContext, params: dict) -> str:
    ids = list(params.get("alpha_ids") or [])
    if params.get("submitted"):
        async with db.session() as s:
            ids = list(await s.scalars(select(Alpha.id).where(Alpha.status.in_(["ACTIVE", "DECOMMISSIONED"]),
                                                              Alpha.sim_type != "RA_PARENT")))
    todo = ids if params.get("force") else await missing(ids, turnover=params.get("turnover", True))
    ok, errors = await fetch_many(todo, turnover=params.get("turnover", True), force=bool(params.get("force")),
                                  ctx=ctx)
    msg = f"Daily PnL and turnover: {ok} downloaded, {len(ids) - len(todo)} already stored"
    if errors:
        msg += f", {len(errors)} failed ({errors[0]})"
    if errors and not ok and todo:
        raise RuntimeError(msg)
    return msg


jobs.register("fetch_series")(_job_fetch)

