"""Out-of-sample outlook from in-sample daily PnL (BRAIN shows little OS detail, so this is an estimate).

Holds        = Sharpe of the last 2 IS years ÷ Sharpe of the earlier IS years.
Est. OS      = IS Sharpe × 0.5 × turnover haircut × holds factor (a rule of thumb: OS Sharpe is often about half of IS).
Risk (0–100) = decay of the recent period + weak recent Sharpe + very high IS Sharpe that fades + turnover +
               complexity + short history.
Outlook      = Overfit risk (risk ≥ 50) · Fading (35–49) · Likely fine (20–34) · Robust (< 20).
These are AlphaGo's own heuristics; the reference app's exact formulas are not public.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import select

from ..db import db
from ..db.models import Alpha, AlphaPnL
from .programs import PP_TAG

OUTLOOKS = {"overfit": "Overfit risk", "fading": "Fading", "fine": "Likely fine", "robust": "Robust"}

EXPLAIN = {
    "outlook": "Robust: recent IS years hold up. Likely fine: small warning signs. Fading: the recent period is clearly "
               "weaker. Overfit risk: the alpha's edge has mostly disappeared in its last IS years.",
    "risk": "0–100. Higher = more likely to disappoint out of sample. Built from Holds, last-2-year Sharpe, turnover, "
            "complexity and history length.",
    "holds": "Last-2-year IS Sharpe ÷ earlier IS Sharpe. 100% = no decay; negative = the recent years lost money.",
    "est_os": "Rough estimate: IS Sharpe × 0.5, reduced for high turnover and for decay. Not a BRAIN number.",
    "early": "Sharpe of the in-sample period before its last 2 years.",
    "last2": "Sharpe of the last 2 years of the in-sample period.",
}


@dataclass
class Outlook:
    alpha_id: str
    kind: str                  # Regular | Power Pool | Super
    outlook: str               # overfit | fading | fine | robust
    risk: int
    is_sharpe: float | None
    early: float | None
    last2: float | None
    holds: float | None
    est_os: float | None
    reason: str

    def row(self) -> dict[str, Any]:
        return asdict(self)


def alpha_kind(a: Alpha) -> str:
    if (a.type or "REGULAR") == "SUPER":
        return "Super"
    return "Power Pool" if PP_TAG in (a.tags or []) else "Regular"


def assess(a: Alpha, p: AlphaPnL | None) -> Outlook:
    is_s = a.sharpe if a.sharpe is not None else (p.sharpe if p else None)
    early = p.sharpe_early if p else None
    last2 = p.sharpe_2y if p else None
    holds = (last2 / early) if (early is not None and last2 is not None and early > 0.1) else None
    t = a.turnover or 0.0
    ops = a.op_count or 0
    n_days = p.n_days if p else 0

    risk = 0.0
    if holds is None:
        risk += 15
    elif holds < 0:
        risk += 50
    elif holds < 0.2:
        risk += 40
    elif holds < 0.5:
        risk += 30
    elif holds < 0.8:
        risk += 15
    if last2 is not None:
        risk += 15 if last2 < 0 else (8 if last2 < 0.5 else 0)
    if is_s is not None and is_s > 3 and holds is not None and holds < 0.5:
        risk += 10
    risk += 10 if t > 0.4 else (5 if t > 0.25 else 0)
    risk += 8 if ops > 12 else (4 if ops > 8 else 0)
    risk += 4 if (a.field_count or 0) > 3 else 0
    if p is not None and n_days < 750:
        risk += 5
    risk_i = int(max(0, min(100, round(risk))))
    outlook = "overfit" if risk_i >= 50 else "fading" if risk_i >= 35 else "fine" if risk_i >= 20 else "robust"

    est = None
    if is_s is not None:
        haircut = 1.0 - min(0.4, max(0.0, t - 0.2))
        hf = 0.85 if holds is None else max(0.7, min(1.0, 0.85 + 0.15 * holds))
        est = round(is_s * 0.5 * haircut * hf, 2)

    if p is None:
        reason = "No daily PnL yet: press Refresh data."
    elif last2 is not None and last2 < 0:
        reason = f"Last 2 IS years lost money (Sharpe {last2:.2f}) after {early or 0:.2f} earlier."
    elif holds is not None and holds < 0.5:
        reason = f"Last 2 IS years kept only {holds:.0%} of the earlier Sharpe."
    elif t > 0.4:
        reason = f"High turnover ({t:.0%}) usually loses more out of sample."
    elif ops > 12:
        reason = f"Complex expression ({ops} operators) leaves room for overfitting."
    elif n_days < 750:
        reason = "Short in-sample history; little evidence either way."
    elif holds is not None:
        reason = f"Recent IS years hold up ({holds:.0%} of the earlier Sharpe)."
    else:
        reason = "Not enough history to compare early and recent years."
    return Outlook(a.id, alpha_kind(a), outlook, risk_i, is_s, early, last2, holds, est, reason)


async def outlook_rows() -> list[dict[str, Any]]:
    async with db.session() as s:
        alphas = list(await s.scalars(select(Alpha).where(Alpha.status == "ACTIVE")))
        pnl = {r.alpha_id: r for r in await s.scalars(select(AlphaPnL).where(
            AlphaPnL.alpha_id.in_([a.id for a in alphas])))} if alphas else {}
    rows = []
    for a in alphas:
        p = pnl.get(a.id)
        o = assess(a, p)
        rows.append({**o.row(), "outlook_label": OUTLOOKS[o.outlook], "turnover": a.turnover,
                     "scope": f"{a.region}/D{a.delay}", "submitted": a.date_submitted.strftime("%Y-%m-%d")
                     if a.date_submitted else "", "brain_os": a.os_sharpe, "spark": (p.spark if p else None) or [],
                     "id": a.id})
    rows.sort(key=lambda r: r["risk"], reverse=True)
    return rows
