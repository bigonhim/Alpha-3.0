"""PnL-driven program logic: alpha ratings, the Osmosis plan, the Osmosis review and the Power Pool pick.

Everything here works on in-sample daily PnL (see `pnl.py`): a guide, not a promise of Osmosis rank or Power Pool
results.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from ..config import settings
from ..db import db
from ..db.models import Alpha, AlphaPnL
from . import pnl
from .programs import PP_TAG, _constrain, alpha_quality, power_pool_shape

RATINGS = ["very_good", "good", "average", "weak"]
RATING_LABEL = {"very_good": "Very good", "good": "Good", "average": "Average", "weak": "Weak"}


# ---------------------------------------------------------------- rating

def rate(a: Alpha, p: AlphaPnL | None) -> tuple[str, str]:
    """(rating, why). Weak = the Osmosis leave-out rule (last-2-year Sharpe < 0.5 or turnover > 50%)."""
    pr = settings.programs
    t = a.turnover or 0.0
    if p is None or p.sharpe_2y is None:
        return "weak", "No daily PnL yet (press Prepare data)."
    s2, s, cal = p.sharpe_2y, (a.sharpe if a.sharpe is not None else p.sharpe) or 0.0, p.calmar
    bits = f"last-2-year Sharpe {s2:.2f}, Sharpe {s:.2f}, turnover {t:.0%}" + (f", Calmar {cal:.2f}" if cal else "")
    if s2 < pr.osmosis_leave_out_sharpe2y or t > pr.osmosis_leave_out_turnover:
        return "weak", f"Weak: {bits}. Left out when last-2-year Sharpe < {pr.osmosis_leave_out_sharpe2y} or " \
                       f"turnover > {pr.osmosis_leave_out_turnover:.0%}."
    if s2 >= 1.5 and s >= 1.5 and t <= 0.3:
        return "very_good", f"Very good: {bits}."
    if s2 >= 1.0:
        return "good", f"Good: {bits}."
    return "average", f"Average: {bits}."


def rating_ok(rating: str, minimum: str) -> bool:
    return RATINGS.index(rating) <= RATINGS.index(minimum)


# ---------------------------------------------------------------- Osmosis plan

@dataclass
class ScopePlan:
    region: str
    delay: int
    submitted: int
    with_pnl: int
    picked: list[str] = field(default_factory=list)
    points: dict[str, int] = field(default_factory=dict)
    left_out: list[dict[str, str]] = field(default_factory=list)
    ratings: dict[str, int] = field(default_factory=dict)       # ratings of the picked alphas
    combo_sharpe: float | None = None
    combo_2y: float | None = None
    best_single: float | None = None
    best_single_id: str | None = None
    avg_corr: float | None = None
    datasets: int = 0
    now_points: int = 0
    now_alphas: int = 0
    rules_ok: bool = False
    recommended: bool = False
    include: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.region}/D{self.delay}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _scope_alphas() -> dict[tuple[str, int], list[Alpha]]:
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE", Alpha.type == "REGULAR")))
    groups: dict[tuple[str, int], list[Alpha]] = defaultdict(list)
    for a in rows:
        groups[(a.region or "?", int(a.delay or 1))].append(a)
    return groups


def _weights(method: str, df: pd.DataFrame, picked: list[str], alphas: dict[str, Alpha]) -> dict[str, float]:
    if method == "equal":
        return {a: 1.0 for a in picked}
    if method == "score":
        return {a: max(0.01, alpha_quality(alphas[a])) for a in picked}
    sub = df[picked].dropna()
    if method in ("kelly", "inverse_corr") and len(sub) >= 60 and len(picked) >= 2:
        if method == "kelly":
            cov = np.cov(sub.to_numpy(), rowvar=False) + np.eye(len(picked)) * 1e-9
            f = np.linalg.solve(cov, sub.mean().to_numpy()) * 0.5
            w = {a: float(max(0.0, x)) for a, x in zip(picked, f)}
            if sum(w.values()) > 0:
                return w
        else:
            c = sub.corr().to_numpy()
            avg = (c.sum(axis=0) - 1) / max(1, len(picked) - 1)
            return {a: float(max(0.05, 1 - x)) for a, x in zip(picked, avg)}
    return pnl.risk_weights(df, picked)


def _datasets(alphas: list[Alpha]) -> int:
    ids = set()
    for a in alphas:
        for d in a.dataset_ids or []:
            ids.update(x for x in str(d).split(",") if x)
    return len(ids)


async def plan_scope(region: str, delay: int, alphas: list[Alpha], max_n: int, method: str = "risk",
                     rows: dict[str, AlphaPnL] | None = None) -> tuple[ScopePlan, pd.DataFrame]:
    pr = settings.programs
    rows = rows if rows is not None else await pnl.stored(a.id for a in alphas)
    sp = ScopePlan(region, delay, submitted=len(alphas), with_pnl=sum(1 for a in alphas if a.id in rows),
                   now_points=sum(a.osmosis_points or 0 for a in alphas),
                   now_alphas=sum(1 for a in alphas if (a.osmosis_points or 0) > 0))
    by_id = {a.id: a for a in alphas}
    eligible: list[str] = []
    for a in alphas:
        p = rows.get(a.id)
        rating, why = rate(a, p)
        if p is None:
            sp.left_out.append({"id": a.id, "reason": "no daily PnL yet"})
        elif rating == "weak":
            sp.left_out.append({"id": a.id, "reason": why})
        else:
            eligible.append(a.id)
    df = pnl.frame({k: rows[k] for k in eligible})
    if not eligible or df.empty:
        sp.notes.append("No alphas pass the rules in this scope.")
        return sp, df
    singles = {k: rows[k].sharpe or -9 for k in eligible}
    sp.best_single_id = max(singles, key=singles.get)
    sp.best_single = rows[sp.best_single_id].sharpe
    picked = pnl.greedy_select(df, eligible, max_n, min_n=min(pr.osmosis_min_alphas, len(eligible)))
    w = _weights(method, df, picked, by_id)
    sp.picked = picked
    sp.points = _constrain(w, pr.osmosis_points_total, pr.osmosis_min_points, pr.osmosis_max_points)
    weights = {k: v / pr.osmosis_points_total for k, v in sp.points.items()}
    sp.combo_sharpe = pnl.combo_sharpe(df, weights)
    sp.combo_2y = pnl.combo_sharpe_last(df, weights)
    sp.avg_corr = pnl.avg_correlation(df, picked)
    sp.datasets = _datasets([by_id[k] for k in picked])
    for k in picked:
        r, _ = rate(by_id[k], rows.get(k))
        sp.ratings[r] = sp.ratings.get(r, 0) + 1
    total = sum(sp.points.values())
    sp.rules_ok = (total == pr.osmosis_points_total and len(picked) >= min(pr.osmosis_min_alphas, len(eligible))
                   and all(pr.osmosis_min_points <= v <= pr.osmosis_max_points for v in sp.points.values()))
    if total != pr.osmosis_points_total:
        sp.notes.append(f"Only {total:,} points fit: {len(picked)} alphas × max {pr.osmosis_max_points:,}. "
                        "Raise the max per alpha or add alphas.")
    if len(picked) < pr.osmosis_min_alphas:
        sp.notes.append(f"Only {len(picked)} alphas qualify (aim for ≥ {pr.osmosis_min_alphas}).")
    sp.recommended = bool(sp.rules_ok and sp.combo_sharpe is not None and sp.best_single is not None
                          and sp.combo_sharpe >= sp.best_single)
    sp.include = sp.rules_ok
    return sp, df


async def make_plan(max_n: int | None = None, method: str = "risk") -> dict[str, Any]:
    max_n = max_n or settings.programs.osmosis_plan_max
    groups = await _scope_alphas()
    all_rows = await pnl.stored(a.id for g in groups.values() for a in g)
    scopes = []
    for (region, delay), alphas in sorted(groups.items()):
        sp, _ = await plan_scope(region, delay, alphas, max_n, method, rows=all_rows)
        scopes.append(sp.to_dict())
    plan = {"created": datetime.now(timezone.utc).isoformat(), "max_n": max_n, "method": method, "scopes": scopes}
    await db.kv_set("osmosis_plan", plan)
    return plan


async def saved_plan() -> dict[str, Any] | None:
    return await db.kv_get("osmosis_plan")


async def scope_chart(region: str, delay: int, points: dict[str, int], best_id: str | None) -> dict[str, list]:
    """Cumulative combo PnL of the plan vs the best single alpha (for the per-scope chart)."""
    ids = list(points) + ([best_id] if best_id and best_id not in points else [])
    rows = await pnl.stored(ids)
    df = pnl.frame(rows)
    if df.empty:
        return {"dates": [], "combo": [], "best": []}
    total = sum(points.values()) or 1
    combo = pnl.combo_series(df, {k: v / total for k, v in points.items() if k in df.columns}).cumsum()
    out = {"dates": [str(d) for d in combo.index], "combo": [round(float(x), 2) for x in combo], "best": []}
    if best_id and best_id in df.columns:
        best = df[best_id].reindex(combo.index).fillna(0).cumsum()
        out["best"] = [round(float(x), 2) for x in best]
    step = max(1, len(out["dates"]) // 400)
    return {k: v[::step] for k, v in out.items()}


# ---------------------------------------------------------------- Osmosis review

@dataclass
class ScopeReview:
    region: str
    delay: int
    alphas: int
    points: int
    combo_sharpe: float | None
    combo_2y: float | None
    est_os: float | None
    avg_corr: float | None
    big: list[str]
    some: list[str]

    @property
    def key(self) -> str:
        return f"{self.region}/D{self.delay}"


def _est_os(combo: float | None, combo_2y: float | None, combo_early: float | None) -> float | None:
    if combo is None:
        return None
    holds = (combo_2y / combo_early) if (combo_2y is not None and combo_early and combo_early > 0.1) else None
    hf = 0.85 if holds is None else max(0.7, min(1.0, 0.85 + 0.15 * holds))
    return round(combo * 0.5 * hf, 2)


async def review() -> list[ScopeReview]:
    pr = settings.programs
    groups = await _scope_alphas()
    out = []
    for (region, delay), alphas in sorted(groups.items()):
        held = [a for a in alphas if (a.osmosis_points or 0) > 0]
        if not held:
            continue
        rows = await pnl.stored(a.id for a in alphas)
        df = pnl.frame({a.id: rows[a.id] for a in held if a.id in rows})
        total = sum(a.osmosis_points or 0 for a in held)
        weights = {a.id: (a.osmosis_points or 0) / total for a in held if a.id in df.columns}
        combo = pnl.combo_sharpe(df, weights) if weights else None
        combo2 = pnl.combo_sharpe_last(df, weights) if weights else None
        early = None
        if weights:
            series = pnl.combo_series(df, weights)
            if not series.empty:
                cut = (pd.Timestamp(str(series.index[-1])) - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
                early = pnl.sharpe(series[series.index <= cut].to_numpy())
        big, some = [], []
        weak = []
        avg_or_weak = []
        for a in held:
            r, _ = rate(a, rows.get(a.id))
            if r == "weak":
                weak.append(a.id)
            elif r == "average":
                avg_or_weak.append(a.id)
        if weak:
            big.append(f"{len(weak)} alpha(s) with points fail the rules (last-2-year Sharpe < "
                       f"{pr.osmosis_leave_out_sharpe2y} or turnover > {pr.osmosis_leave_out_turnover:.0%}, or no PnL).")
        if total != pr.osmosis_points_total:
            big.append(f"Scope holds {total:,} points instead of {pr.osmosis_points_total:,}.")
        if len(held) < pr.osmosis_min_alphas:
            big.append(f"Only {len(held)} alphas hold points (aim for ≥ {pr.osmosis_min_alphas}).")
        corr = pnl.avg_correlation(df, list(weights))
        if corr is not None and corr > 0.5:
            big.append(f"Average correlation {corr:.2f} > 0.5: the points sit on very similar alphas.")
        heavy = [a.id for a in held if (a.osmosis_points or 0) > pr.osmosis_max_points]
        if heavy:
            big.append(f"{len(heavy)} alpha(s) above {pr.osmosis_max_points:,} points.")
        if avg_or_weak:
            some.append(f"{len(avg_or_weak)} alpha(s) with points are only rated Average.")
        best, _ = await plan_scope(region, delay, alphas, pr.osmosis_plan_max, rows=rows)
        if combo is not None and best.combo_sharpe and combo < 0.8 * best.combo_sharpe:
            some.append(f"Combo Sharpe {combo:.2f} is more than 20% below the best plan ({best.combo_sharpe:.2f}).")
        out.append(ScopeReview(region, delay, len(held), total, combo, combo2, _est_os(combo, combo2, early), corr,
                               big, some))
    return out


async def ai_predict(scopes: list[ScopeReview], score_now: float | None, language: str):  # noqa: ANN201
    from ..llm import llm
    from ..llm.prompts import render
    from ..llm.schemas import OsmosisPrediction
    prompt = await render("osmosis_predict", score_now="not given" if score_now is None else score_now,
                          numbers=review_numbers(scopes), language=language)
    return await llm.generate_json(prompt, OsmosisPrediction, role="refine", purpose="osmosis")


async def ai_explain(scopes: list[ScopeReview], language: str):  # noqa: ANN201
    from ..llm import llm
    from ..llm.prompts import render
    from ..llm.schemas import OsmosisExplanation
    pr = settings.programs
    prompt = await render("osmosis_explain", numbers=review_numbers(scopes), language=language,
                          min_alphas=pr.osmosis_min_alphas, min_sharpe2y=pr.osmosis_leave_out_sharpe2y,
                          max_turnover=f"{pr.osmosis_leave_out_turnover:.0%}")
    return await llm.generate_json(prompt, OsmosisExplanation, role="refine", purpose="osmosis")


def review_numbers(scopes: list[ScopeReview]) -> str:
    """Numbers-only summary for the AI (never expressions or alpha ids)."""
    lines = []
    for s in scopes:
        lines.append(f"{s.key}: {s.alphas} alphas hold {s.points:,} pts; combo Sharpe {s.combo_sharpe}; last-2y "
                     f"combo Sharpe {s.combo_2y}; est. OS Sharpe {s.est_os}; avg corr "
                     f"{None if s.avg_corr is None else round(s.avg_corr, 2)}; big issues: {s.big or 'none'}; "
                     f"smaller issues: {s.some or 'none'}")
    return "\n".join(lines) or "No scopes hold points."


# ---------------------------------------------------------------- Power Pool pick

def month_of(a: Alpha) -> str | None:
    return a.date_submitted.strftime("%Y-%m") if a.date_submitted else None


def pp_shape(a: Alpha) -> tuple[bool, int | None, int | None]:
    try:
        ops, fields = power_pool_shape(a.code or "")
        return ops <= 8 and len(fields) <= 3, ops, len(fields)
    except Exception:
        return False, None, None


async def pp_months() -> list[tuple[str, int]]:
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE", Alpha.type == "REGULAR")))
    counts: dict[str, int] = defaultdict(int)
    for a in rows:
        m = month_of(a)
        if m:
            counts[m] += 1
    return sorted(counts.items(), reverse=True)


async def pp_rows(month: str) -> list[dict[str, Any]]:
    async with db.session() as s:
        alphas = [a for a in await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE", Alpha.type == "REGULAR"))
                  if month_of(a) == month]
    rows = await pnl.stored(a.id for a in alphas)
    out = []
    for a in alphas:
        ok, ops, nf = pp_shape(a)
        p = rows.get(a.id)
        r, why = rate(a, p)
        out.append({"id": a.id, "tagged": PP_TAG in (a.tags or []), "shape": ok, "rating": r,
                    "rating_label": RATING_LABEL[r], "why": why, "scope": f"{a.region}/D{a.delay}",
                    "region": a.region, "delay": a.delay, "submitted": a.date_submitted.strftime("%Y-%m-%d"),
                    "universe": a.universe, "spark": (p.spark if p else None) or [], "sharpe": a.sharpe,
                    "sharpe_2y": p.sharpe_2y if p else None, "calmar": p.calmar if p else None,
                    "turnover": a.turnover, "pnl_1y": p.pnl_1y if p else None,
                    "ops_fields": f"{ops if ops is not None else '—'} / {nf if nf is not None else '—'}",
                    "max_corr": None})
    return out


async def pp_pick(month: str, max_per_scope: int, min_rating: str, max_corr: float) -> dict[str, Any]:
    """Per scope: best-rated PP-shaped alphas first, skipping any that correlate ≥ max_corr with one already picked."""
    rows = await pp_rows(month)
    pnl_rows = await pnl.stored(r["id"] for r in rows)
    by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["shape"] and rating_ok(r["rating"], min_rating):
            by_scope[r["scope"]].append(r)
    picked: list[str] = []
    max_corrs: dict[str, float] = {}
    for scope, items in by_scope.items():
        items.sort(key=lambda r: (RATINGS.index(r["rating"]), -(r["sharpe_2y"] or -9), -(r["sharpe"] or -9)))
        df = pnl.frame({r["id"]: pnl_rows[r["id"]] for r in items if r["id"] in pnl_rows})
        chosen: list[str] = []
        for r in items:
            if len(chosen) >= max_per_scope:
                break
            aid = r["id"]
            worst = None
            if chosen and aid in df.columns:
                c = df[[aid] + [x for x in chosen if x in df.columns]].corr(min_periods=60)[aid].drop(aid)
                worst = float(c.max()) if c.notna().any() else None
            if worst is not None and worst >= max_corr:
                continue
            chosen.append(aid)
            if worst is not None and not math.isnan(worst):
                max_corrs[aid] = round(worst, 2)
        picked += chosen
    result = {"month": month, "created": datetime.now(timezone.utc).isoformat(), "max": max_per_scope,
              "min_rating": min_rating, "picked": picked, "max_corr": max_corrs}
    await db.kv_set("pp_pick", result)
    return result


async def saved_pick() -> dict[str, Any] | None:
    return await db.kv_get("pp_pick")
