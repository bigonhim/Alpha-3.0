"""Health: how the AI providers and BRAIN behaved over time, so settings can be tuned with evidence.

AI providers — every call is already logged (`LLMCall`); errors carry a kind (rate_limited, timeout,
model_unavailable, server_error, auth, budget, bad_json, other).
BRAIN — request outcomes are counted in memory by a client observer (429 throttles, 5xx, network errors) and written
to the database every few minutes with the day's quota numbers; simulation errors come from the candidates table.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from ..brain.quota import EASTERN, eastern_day
from ..db import db
from ..db.models import Candidate, LLMCall
from ..llm.base import ERROR_KINDS, classify_error

log = logging.getLogger(__name__)

BRAIN_STATS_KEY = "brain_stats"
FLUSH_EVERY_S = 120
_counts: dict[str, Counter] = defaultdict(Counter)


def observe(method: str, url: str, status: int, body: Any, final: bool, response: Any = None) -> None:
    """BrainClient observer: count every response (retried attempts included) by day and outcome."""
    c = _counts[eastern_day()]
    c["requests"] += 1
    if status == 0:
        c["network_errors"] += 1
    elif status == 429:
        c["throttled"] += 1
    elif status >= 500:
        c["server_errors"] += 1
    elif status in (401, 403):
        c["auth_errors"] += 1
    if method.upper() == "POST" and url.rstrip("/").endswith("/simulations") and final:
        if 200 <= status < 300:
            c["simulations_started"] += len(body) if isinstance(body, list) else 1
        elif status == 429:
            c["concurrency_rejections"] += 1


def install() -> None:
    from ..brain import brain
    if observe not in brain.client.observers:
        brain.client.observers.append(observe)


async def flush() -> None:
    """Merge the in-memory counters into the stored per-day stats (kept 60 days), with today's quota."""
    from ..brain import brain
    stored: dict[str, dict[str, int]] = await db.kv_get(BRAIN_STATS_KEY) or {}
    for day, c in list(_counts.items()):
        row = stored.setdefault(day, {})
        for k, v in c.items():
            row[k] = row.get(k, 0) + v
        c.clear()
    q = brain.client.quota
    today = stored.setdefault(eastern_day(), {})
    today.update(quota_limit=q.limit, quota_remaining=q.remaining, sent_by_app=q.sent_today)
    keep = sorted(stored)[-60:]
    await db.kv_set(BRAIN_STATS_KEY, {d: stored[d] for d in keep})


async def flush_loop() -> None:
    while True:
        await asyncio.sleep(FLUSH_EVERY_S)
        try:
            await flush()
        except Exception:
            log.exception("health flush failed")


def _since(days: int) -> datetime:
    return (datetime.now(EASTERN) - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0,
                                                                       microsecond=0).astimezone(timezone.utc)


def _day(ts: datetime) -> str:
    return ts.astimezone(EASTERN).strftime("%Y-%m-%d")


async def llm_by_day(days: int = 14) -> dict[str, Any]:
    """Per provider and BRAIN day: calls, errors by kind, average latency, tokens."""
    async with db.session() as s:
        rows = (await s.execute(select(LLMCall.ts, LLMCall.provider, LLMCall.status, LLMCall.error,
                                       LLMCall.error_kind, LLMCall.latency_ms, LLMCall.prompt_tokens,
                                       LLMCall.output_tokens).where(LLMCall.ts >= _since(days)))).all()
    labels = [(_since(days) + timedelta(days=i)).astimezone(EASTERN).strftime("%Y-%m-%d") for i in range(days)]
    table: dict[tuple[str, str], Counter] = defaultdict(Counter)
    latency: dict[tuple[str, str], list[int]] = defaultdict(list)
    for ts, provider, status, error, kind, lat, pt, ot in rows:
        key = (provider or "gemini", _day(ts))
        c = table[key]
        c["calls"] += 1
        if status == "ok":
            c["ok"] += 1
        else:
            c[kind or classify_error(error) or "other"] += 1
        c["tokens"] += (pt or 0) + (ot or 0)
        if lat is not None:
            latency[key].append(lat)
    providers = sorted({p for p, _ in table})
    out = []
    for p in providers:
        for d in labels:
            c = table.get((p, d))
            if not c:
                continue
            lats = latency.get((p, d)) or []
            out.append({"provider": p, "day": d, **{k: c.get(k, 0) for k in ("calls", "ok", "tokens")},
                        **{k: c.get(k, 0) for k in ERROR_KINDS},
                        "avg_latency_s": round(sum(lats) / len(lats) / 1000, 1) if lats else None})
    return {"labels": labels, "providers": providers, "rows": out}


async def llm_by_model(days: int = 14) -> list[dict[str, Any]]:
    async with db.session() as s:
        rows = (await s.execute(select(LLMCall.provider, LLMCall.model, LLMCall.status, func.count(),
                                       func.avg(LLMCall.latency_ms))
                                .where(LLMCall.ts >= _since(days))
                                .group_by(LLMCall.provider, LLMCall.model, LLMCall.status))).all()
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for provider, model, status, n, lat in rows:
        r = agg.setdefault((provider or "gemini", model), {"provider": provider or "gemini", "model": model,
                                                           "calls": 0, "errors": 0, "lat_sum": 0.0})
        r["calls"] += n
        r["errors"] += 0 if status == "ok" else n
        r["lat_sum"] += (lat or 0) * n
    out = []
    for r in agg.values():
        out.append({"provider": r["provider"], "model": r["model"], "calls": r["calls"], "errors": r["errors"],
                    "error_rate": round(r["errors"] / r["calls"], 3) if r["calls"] else 0,
                    "avg_latency_s": round(r["lat_sum"] / r["calls"] / 1000, 1) if r["calls"] else None})
    return sorted(out, key=lambda r: r["calls"], reverse=True)


async def brain_by_day(days: int = 14) -> list[dict[str, Any]]:
    await flush()
    stored: dict[str, dict[str, int]] = await db.kv_get(BRAIN_STATS_KEY) or {}
    async with db.session() as s:
        rows = (await s.execute(select(Candidate.finished_at, Candidate.status).where(
            Candidate.finished_at >= _since(days), Candidate.status.in_(["done", "error"])))).all()
    sims: dict[str, Counter] = defaultdict(Counter)
    for ts, status in rows:
        if ts:
            sims[_day(ts)][status] += 1
    labels = [(_since(days) + timedelta(days=i)).astimezone(EASTERN).strftime("%Y-%m-%d") for i in range(days)]
    out = []
    for d in labels:
        st = stored.get(d, {})
        out.append({"day": d, "simulated": sims[d].get("done", 0), "sim_errors": sims[d].get("error", 0),
                    "requests": st.get("requests", 0), "throttled": st.get("throttled", 0),
                    "concurrency_rejections": st.get("concurrency_rejections", 0),
                    "server_errors": st.get("server_errors", 0), "network_errors": st.get("network_errors", 0),
                    "quota_limit": st.get("quota_limit"), "quota_remaining": st.get("quota_remaining"),
                    "sent_by_app": st.get("sent_by_app")})
    return out


async def repair_table(min_n: int = 1) -> list[dict[str, Any]]:
    """Which repairs actually worked, per failing check (the learned order the repair loop uses)."""
    from .refine import FIX_BY_KEY, load_fix_stats
    stats = await load_fix_stats()
    out = []
    for (fix, target), (succ, fail) in stats.overall.items():
        n = succ + fail
        if n >= min_n:
            prior = FIX_BY_KEY[fix].prior if fix in FIX_BY_KEY else 0.3
            out.append({"fix": fix, "target": target, "tried": n, "worked": succ,
                        "rate": round(succ / n, 3), "prior": prior})
    return sorted(out, key=lambda r: (r["target"], -r["rate"]))
