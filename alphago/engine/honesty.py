"""Honesty figures for an alpha that passed BRAIN's check — all computed locally, none costs simulation quota.

walk_forward     Sharpe of the last 20% of the in-sample daily PnL ÷ Sharpe of the first 80%. The search saw these
                 years, so this measures recent stability, not true out-of-sample performance.
deflated_sharpe  Bailey & López de Prado (2014): the probability that the Sharpe beats the best of N luck-only trials,
                 where N is the number of distinct ideas (lineages) the search tried and the trials' Sharpe spread
                 sets how lucky "the best" can get. Advisory: it ranks finalists, it never blocks one.
gap model        A ridge regression on your submitted alphas that have BRAIN out-of-sample numbers: which formula
                 features predicted a fall from in-sample to out-of-sample Sharpe, and the predicted OS/IS ratio of a
                 new alpha. It needs at least MIN_GAP_ALPHAS such alphas.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import NormalDist
from typing import Any

import numpy as np
from sqlalchemy import func, select

from ..db import db
from ..db.models import Alpha, AlphaPnL, Candidate
from . import pnl

log = logging.getLogger(__name__)

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649
WF_SPLIT = 0.8
MIN_GAP_ALPHAS = 20
GAP_MODEL_KEY = "gap_model"
_N = NormalDist()


# ---------------------------------------------------------------- walk-forward

def walk_forward_ratio(daily: list[float], split: float = WF_SPLIT) -> tuple[float | None, float | None, float | None]:
    """(ratio, Sharpe of the first part, Sharpe of the last part) of a daily PnL series."""
    arr = np.asarray(daily, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 250:
        return None, None, None
    k = int(arr.size * split)
    first, last = pnl.sharpe(arr[:k]), pnl.sharpe(arr[k:])
    if first is None or last is None or first <= 0.1:
        return None, first, last
    return round(last / first, 3), first, last


# ---------------------------------------------------------------- deflated Sharpe

def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Expected maximum (per-period) Sharpe of n_trials luck-only strategies whose Sharpes have this variance."""
    if n_trials <= 1 or var_sharpe <= 0:
        return 0.0
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(var_sharpe) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(sharpe_annual: float, n_days: int, n_trials: int, trial_sharpes_annual: list[float],
                    skew: float = 0.0, kurtosis: float = 3.0) -> float | None:
    """Probability (0..1) that the true Sharpe is above what the best of n_trials lucky trials would show."""
    if n_days < 30 or sharpe_annual is None:
        return None
    sr = sharpe_annual / math.sqrt(TRADING_DAYS)                     # per day
    trials = [s / math.sqrt(TRADING_DAYS) for s in trial_sharpes_annual if s is not None and math.isfinite(s)]
    var = float(np.var(trials, ddof=1)) if len(trials) >= 2 else 0.0
    sr0 = expected_max_sharpe(n_trials, var)
    denom = 1 - skew * sr + (kurtosis - 1) / 4 * sr ** 2
    if denom <= 0:
        return None
    z = (sr - sr0) * math.sqrt(n_days - 1) / math.sqrt(denom)
    return round(_N.cdf(z), 4)


@dataclass
class SearchSize:
    n_trials: int          # every simulation of the search, repairs included
    n_eff: int             # distinct ideas (first-pass simulations): repair children share their root's luck
    sharpes: list[float]   # Sharpe of every first-pass trial


async def search_size(alpha_id: str) -> SearchSize:
    """The search (research task or Autopilot batch) that produced an alpha. A manual alpha is a search of one."""
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
        cand = await s.get(Candidate, a.candidate_id) if a and a.candidate_id else None
        if cand is None or (cand.task_id is None and cand.batch_id is None):
            return SearchSize(1, 1, [a.sharpe] if a and a.sharpe is not None else [])
        col, key = (Candidate.task_id, cand.task_id) if cand.task_id is not None else (Candidate.batch_id,
                                                                                      cand.batch_id)
        rows = (await s.execute(select(Candidate.origin, Alpha.sharpe).join(Alpha, Alpha.id == Candidate.alpha_id)
                                .where(col == key, Candidate.status.in_(["done", "reused"])))).all()
    first = [sh for origin, sh in rows if origin not in ("optimize", "robustness")]
    return SearchSize(len(rows), max(1, len(first)), [sh for sh in first if sh is not None])


# ---------------------------------------------------------------- the IS → OS gap model

FEATURES = ["operators", "fields", "turnover", "decay", "trade_when", "longest_window", "is_sharpe", "holds",
            "risk_neutralization"]
FEATURE_LABELS = {"operators": "more operators", "fields": "more data fields", "turnover": "higher turnover",
                  "decay": "more decay", "trade_when": "trade_when gating", "longest_window": "longer windows",
                  "is_sharpe": "higher in-sample Sharpe", "holds": "recent years holding up",
                  "risk_neutralization": "risk-model neutralization"}
RISK_NEUTRALIZATIONS = {"STATISTICAL", "CROWDING", "FAST", "SLOW", "SLOW_AND_FAST", "REVERSION_AND_MOMENTUM"}


def features(a: Alpha, p: AlphaPnL | None) -> list[float]:
    longest = 0
    if a.code:
        try:
            from ..fastexpr import compute_metrics, parse
            m = compute_metrics(parse(a.code))
            longest = max(m.windows) if m.windows else 0
        except Exception:
            longest = 0
    holds = 1.0
    if p is not None and p.sharpe_early and p.sharpe_2y is not None and p.sharpe_early > 0.1:
        holds = max(-2.0, min(2.0, p.sharpe_2y / p.sharpe_early))
    return [float(a.op_count or 0), float(len(a.fields or []) or (a.field_count or 0)), float(a.turnover or 0),
            float(a.decay or 0), 1.0 if "trade_when" in (a.code or "") else 0.0, math.log1p(longest),
            float(a.sharpe or 0), holds, 1.0 if (a.neutralization or "") in RISK_NEUTRALIZATIONS else 0.0]


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> dict[str, Any]:
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z = (x - mean) / std
    yc = y - y.mean()
    coef = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ yc)
    pred = z @ coef + y.mean()
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {"mean": mean.tolist(), "std": std.tolist(), "coef": coef.tolist(), "intercept": float(y.mean()),
            "r2": round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else None, "n": int(len(y))}


async def fit_gap_model() -> dict[str, Any]:
    """Fit (and store) the OS/IS model on submitted alphas that have BRAIN out-of-sample Sharpe."""
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE", Alpha.os_sharpe.is_not(None),
                                                        Alpha.sharpe > 0.3)))
    stored = await pnl.stored([a.id for a in rows])
    if len(rows) < MIN_GAP_ALPHAS:
        model = {"n": len(rows), "ready": False, "fitted_at": datetime.now(timezone.utc).isoformat()}
        await db.kv_set(GAP_MODEL_KEY, model)
        return model
    x = np.array([features(a, stored.get(a.id)) for a in rows])
    y = np.array([max(-2.0, min(2.0, a.os_sharpe / a.sharpe)) for a in rows])
    model = {**fit_ridge(x, y), "ready": True, "features": FEATURES,
             "fitted_at": datetime.now(timezone.utc).isoformat()}
    await db.kv_set(GAP_MODEL_KEY, model)
    return model


async def gap_model(max_age: timedelta = timedelta(days=1)) -> dict[str, Any]:
    model = await db.kv_get(GAP_MODEL_KEY)
    if model:
        try:
            if datetime.now(timezone.utc) - datetime.fromisoformat(model["fitted_at"]) < max_age:
                return model
        except (KeyError, ValueError):
            pass
    return await fit_gap_model()


def predict_retention(model: dict[str, Any], a: Alpha, p: AlphaPnL | None) -> float | None:
    if not model.get("ready"):
        return None
    z = (np.array(features(a, p)) - np.array(model["mean"])) / np.array(model["std"])
    return round(float(z @ np.array(model["coef"]) + model["intercept"]), 3)


def explain(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Features ordered by how strongly they predicted a gap (standardised coefficients)."""
    if not model.get("ready"):
        return []
    out = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "effect": round(c, 3),
            "reading": ("predicted a BIGGER fall out of sample" if c < 0 else "predicted holding up better")}
           for f, c in zip(model.get("features", FEATURES), model["coef"])]
    return sorted(out, key=lambda r: abs(r["effect"]), reverse=True)


# ---------------------------------------------------------------- one call for certification

async def assess(alpha_id: str) -> dict[str, Any]:
    """Walk-forward, deflated Sharpe and predicted OS/IS for an alpha; stored on the alpha's columns."""
    row = (await pnl.stored([alpha_id])).get(alpha_id)
    if row is None:
        try:
            row = await pnl.fetch(alpha_id, keep_raw=False)
        except Exception as exc:
            log.info("no daily PnL for %s: %s", alpha_id, getattr(exc, "message", exc))
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
    if a is None:
        return {}
    out: dict[str, Any] = {}
    size = await search_size(alpha_id)
    out["n_trials"], out["n_eff"] = size.n_trials, size.n_eff
    if row is not None and row.daily:
        out["wf_ratio"] = walk_forward_ratio(row.daily)[0]
        arr = np.asarray(row.daily, dtype=float)
        sd = arr.std()
        skew = float(((arr - arr.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
        kurt = float(((arr - arr.mean()) ** 4).mean() / sd ** 4) if sd > 0 else 3.0
        if a.sharpe is not None:
            out["dsr"] = deflated_sharpe(a.sharpe, len(arr), size.n_eff, size.sharpes, skew, kurt)
    try:
        out["os_retention_pred"] = predict_retention(await gap_model(), a, row)
    except Exception:
        log.exception("gap model failed")
    async with db.write() as s:
        target = await s.get(Alpha, alpha_id)
        if target is not None:
            for k, v in out.items():
                setattr(target, k, v)
    return out


async def gap_summary() -> dict[str, Any]:
    model = await gap_model()
    async with db.session() as s:
        n_os = await s.scalar(select(func.count(Alpha.id)).where(Alpha.status == "ACTIVE",
                                                                 Alpha.os_sharpe.is_not(None))) or 0
    return {"model": model, "explain": explain(model), "n_with_os": n_os, "needed": MIN_GAP_ALPHAS}
