"""Copy Alphas: Alphas as Markdown for an LLM to read (alpha-harness api/portfolio.copied_alphas).

Each Alpha is one block, highest Sharpe first: its expression, classifications, Simulation Settings by BRAIN's labels,
Metrics (BRAIN's own figures to BRAIN's places, plus Calmar, Margin in ‱, K-Ratio, Long/Short Imbalance and the
correlations worked out here), each check value that measures something the metrics do not show inserted right after
the metric it belongs with, the Investability Constrained block when BRAIN reported one, and a verdict line. Then the
Metric Glossary and every data field read, by category, subcategory and dataset. Paste Alphas reads this format
back, so an LLM's answer in it runs as written.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
from sqlalchemy import select

from ..brain.platform import (
    CORRELATION_WINDOW_YEARS,
    IGNORED_CHECKS,
    QUICK_MODE,
    TRADING_DAYS,
    brain_verdict,
    strip_quota_checks,
)
from ..db import db
from ..db.models import Alpha, Dataset

COPIED_SETTINGS = (
    ("region", "region", "Region"), ("universe", "universe", "Universe"), ("language", "language", "Language"),
    ("decay", "decay", "Decay"), ("delay", "delay", "Delay"), ("truncation", "truncation", "Truncation"),
    ("neutralization", "neutralization", "Neutralization"), ("pasteurization", "pasteurization", "Pasteurization"),
    ("nanHandling", "nan_handling", "NaN Handling"), ("maxTrade", "max_trade", "Max Trade"),
    ("maxPosition", "max_position", "Max Position"),
)


def _pct(v: float) -> str:
    return f"{v:.2%}"


def _ratio(v: float) -> str:
    return f"{v:.2f}"


def _margin(v: float) -> str:
    return f"{v * 10_000:.2f}‱"


def _count(v: float) -> str:
    return f"{v:,.0f}"


COPIED_METRICS = (
    ("sharpe", "Sharpe", _ratio), ("turnover", "Turnover", _pct), ("fitness", "Fitness", _ratio),
    ("returns", "Returns", _pct), ("drawdown", "Drawdown", _pct), ("calmar", "Calmar", _ratio),
    ("margin", "Margin", _margin), ("k_ratio", "K-Ratio", _ratio), ("long_count", "Long Count", _count),
    ("short_count", "Short Count", _count), ("imbalance", "Long/Short Imbalance", lambda v: f"{v:+.0%}"),
    ("prod_correlation", "Production Correlation", _ratio),
    ("max_pairwise_correlation", "Max Pairwise Correlation", _ratio),
    ("portfolio_correlation", "Portfolio Correlation", _ratio),
)
CONSTRAINED_FIGURES = (
    ("sharpe", "Sharpe", _ratio), ("turnover", "Turnover", _pct), ("fitness", "Fitness", _ratio),
    ("returns", "Returns", _pct), ("drawdown", "Drawdown", _pct), ("margin", "Margin", _margin),
    ("longCount", "Long Count", _count), ("shortCount", "Short Count", _count),
)
# a check's value read as a metric, listed right after the metric it belongs with
CHECK_METRICS = {
    "LOW_SUB_UNIVERSE_SHARPE": ("sharpe", "Sub-Universe Sharpe", _ratio),
    "CLUSTER_TEST": ("sharpe", "Cluster Sharpe", _ratio),
    "LOW_2Y_SHARPE": ("sharpe", "2Y Sharpe", _ratio),
    "LOW_GLB_AMER_SHARPE": ("sharpe", "GLB AMER Sharpe", _ratio),
    "LOW_GLB_APAC_SHARPE": ("sharpe", "GLB APAC Sharpe", _ratio),
    "LOW_GLB_EMEA_SHARPE": ("sharpe", "GLB EMEA Sharpe", _ratio),
    "LOW_ASI_JPN_SHARPE": ("sharpe", "ASI JPN Sharpe", _ratio),
    "LOW_ROBUST_UNIVERSE_SHARPE": ("sharpe", "Robust Universe Sharpe", _ratio),
    "LOW_ROBUST_UNIVERSE_SHARPE.WITH_RATIO": ("sharpe", "Robust Universe Sharpe", _ratio),
    "LOW_ROBUST_UNIVERSE_RETURNS": ("returns", "Robust Universe Returns", _pct),
}
COPIED_CHECKS = ("IS_LADDER_SHARPE", "CONCENTRATED_WEIGHT")    # what they report has no single figure

METRIC_GLOSSARY = "\n".join((
    "# Metric Glossary",
    "",
    "Every figure is over the in-sample period on BRAIN's $20,000,000 book. Sharpe, Turnover, Fitness, Returns, "
    "Drawdown, Margin, Long Count, Short Count, Production Correlation, the Sub-Universe, Cluster, 2Y, regional and "
    "Robust Universe figures, the Investability Constrained Metrics (the same Alpha within BRAIN's liquidity limits on "
    "positions and trades) and the Checks are BRAIN's own. The rest are worked out locally from each Alpha's daily "
    "PnL, before cost:",
    "",
    "- **K-Ratio**: Kestner's K-ratio (2013 form). Fit a line to cumulative PnL against the day number, from the "
    f"first day traded; K = slope / (standard error of the slope * number of days) * sqrt({TRADING_DAYS}).",
    "- **Calmar**: Returns / Drawdown, from BRAIN's own two figures.",
    "- **Long/Short Imbalance**: (Long Count - Short Count) / (Long Count + Short Count). +100% is all long, -100% "
    "all short.",
    "- **Max Pairwise Correlation**: the highest Pearson correlation of daily PnL with any other submitted Alpha, on "
    f"the days both traded, over the last {CORRELATION_WINDOW_YEARS} calendar years of data, the latest included, as "
    "BRAIN measures its correlations.",
    "- **Portfolio Correlation**: the Pearson correlation of daily PnL with the equal-weighted portfolio of every other "
    "submitted Alpha (their daily PnL averaged over those trading that day), over the same window.",
))


def _imbalance(long: float | None, short: float | None) -> float | None:
    if long is None or short is None or not long + short:
        return None
    return (long - short) / (long + short)


def _check_text(check: dict[str, Any]) -> str:
    """One check as BRAIN reports it: its result, then every figure it gives."""
    parts = [f"{re.sub(r'(?<!^)(?=[A-Z])', ' ', key).lower()} {value}" for key, value in check.items()
             if key not in ("name", "result") and value is not None]
    return f"{check.get('name')}: {check.get('result')}" + (f" ({', '.join(parts)})" if parts else "")


def settings_of(a: Alpha) -> dict[str, Any]:
    raw = dict(a.settings or {})
    return {name: raw.get(name, getattr(a, column, None)) for name, column, _ in COPIED_SETTINGS}


def settings_lines(settings: dict[str, Any]) -> list[str]:
    """Simulation Settings as BRAIN labels them, from settings keyed by BRAIN's names."""
    from .paste_alphas import label
    return [f"{title}: {'—' if settings.get(name) is None else label(name, settings[name])}"
            for name, _, title in COPIED_SETTINGS]


def settled_checks(a: Alpha) -> list[dict[str, Any]]:
    """The more settled of the /check result and the simulation's checks: fewer still PENDING (the /check one on a
    tie, as it may have spoken since)."""
    stored = strip_quota_checks(a.checks)
    body = strip_quota_checks(a.is_checks)

    def pending(checks: list[dict[str, Any]]) -> int:
        return sum(str(c.get("result", "")).upper() == "PENDING" for c in checks)
    return body if body and pending(body) < pending(stored) else stored


def checks_line(checks: list[dict[str, Any]], a: Alpha) -> str:
    """BRAIN's verdict, and the checks behind a refusal."""
    if a.status and a.status != "UNSUBMITTED":
        return "Checks:"                 # submitted says it all; past submission some checks stay PENDING for good
    failed = [str(c.get("name")) for c in checks if str(c.get("result", "")).upper() in ("FAIL", "ERROR")
              and str(c.get("name", "")).upper() not in IGNORED_CHECKS]
    if failed:
        return f"Checks: FAIL ({', '.join(failed)})"
    if str(a.simulation_mode or "").upper() == QUICK_MODE:
        return "Checks: QUICK (Quick mode runs no submission checks)"
    found = brain_verdict(checks, a.simulation_mode)
    return {"submittable": "Checks: PASS", "pending": "Checks: PENDING"}.get(str(found), "Checks: —")


def classification_names(a: Alpha) -> list[str]:
    out = []
    for c in a.classifications or []:
        name = c.get("name") if isinstance(c, dict) else c
        if name:
            out.append(str(name))
    return sorted(out, key=str.casefold)


# ---------------------------------------------------------------- figures worked out here

async def local_figures(alpha_ids: list[str]) -> dict[str, dict[str, Any]]:
    """K-Ratio, Max Pairwise Correlation and Portfolio Correlation, before cost. A submitted Alpha is measured against
    every other submitted one; any other Alpha against every submitted one."""
    from ..engine import metrics
    from ..engine import series as series_mod
    async with db.session() as s:
        members = list(await s.scalars(select(Alpha.id).where(Alpha.status.is_not(None),
                                                              Alpha.status != "UNSUBMITTED",
                                                              Alpha.sim_type.is_distinct_from("RA_PARENT"))))
    loaded = await series_mod.load([*members, *alpha_ids])
    out: dict[str, dict[str, Any]] = {}
    for a in alpha_ids:
        if a in loaded:
            v = loaded[a]
            pnl = [p for p, c in zip(v["pnl"], v["closing"], strict=False) if not c] or v["pnl"]
            out[a] = {"k_ratio": metrics.k_ratio(pnl)}
    if not loaded:
        return out

    def work() -> dict[str, dict[str, Any]]:
        frame = metrics.Frame({k: (v["dates"], v["pnl"], v["turnover"]) for k, v in loaded.items()},
                              {k: v["closing"] for k, v in loaded.items()})
        if not frame.dates:
            return {}
        window = frame.since(metrics.window_start(date.fromisoformat(frame.dates[-1])))
        ex = frame.exported()[window]
        if ex.shape[0] == 0:
            return {}
        corr = metrics.correlation_matrix(ex)
        col = {k: j for j, k in enumerate(frame.ids)}
        member_cols = [col[m] for m in members if m in col]
        p = np.nan_to_num(ex)
        started = np.maximum.accumulate(p != 0, axis=0)
        found: dict[str, dict[str, Any]] = {}
        for a in alpha_ids:
            j = col.get(a)
            if j is None:
                continue
            others = [c for c in member_cols if c != j]
            if not others:
                continue
            row = corr[j, others]
            highest = float(np.nanmax(row)) if np.isfinite(row).any() else None
            live = started[:, others].sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                rest = np.where(live > 0, p[:, others].sum(axis=1) / live, np.nan)
            found[a] = {"max_pairwise_correlation": highest, "portfolio_correlation": metrics.pair_correlation(
                ex[:, j], rest)}
        return found

    for a, figures in (await asyncio.to_thread(work)).items():
        out.setdefault(a, {}).update(figures)
    return out


async def constrained_of(alphas: list[Alpha]) -> dict[str, dict[str, Any]]:
    """BRAIN's Investability Constrained figures: from the stored body when it kept them, else read from BRAIN once
    (when signed in) and remembered."""
    from ..brain import brain
    out: dict[str, dict[str, Any]] = {}
    unread: list[str] = []
    for a in alphas:
        kept = (a.raw or {}).get("investabilityConstrained") or await db.kv_get(f"copy:ic:{a.id}")
        if isinstance(kept, dict):
            out[a.id] = kept
        elif kept is None:
            unread.append(a.id)
    if not unread or not brain.client.logged_in:
        return out
    gate = asyncio.Semaphore(4)

    async def one(alpha_id: str) -> None:
        async with gate:
            try:
                body = await brain.alphas.get(alpha_id)
            except Exception:
                return
        found = ((body or {}).get("is") or {}).get("investabilityConstrained")
        await db.kv_set(f"copy:ic:{alpha_id}", found if isinstance(found, dict) else {})
        if isinstance(found, dict):
            out[alpha_id] = found
    await asyncio.gather(*(one(a) for a in unread[:200]))
    return out


# ---------------------------------------------------------------- the field glossary

PLACE = ("category", "subcategory", "dataset_id")


def field_glossary(fields: list[str], known: dict[str, Any], datasets: dict[str, Dataset]) -> str:
    """Every field read, under its category, subcategory and dataset, each level alphabetical (Markdown headings,
    each with a blank line around it)."""
    def text(field: str, key: str) -> str:
        return str(getattr(known.get(field), key, None) or "")

    def order(field: str) -> tuple:
        return (field not in known, *(text(field, k).casefold() for k in PLACE), field.casefold())

    lines: list[str] = []
    shown: tuple[str, ...] | None = None
    first = True
    for field in sorted(fields, key=order):
        place = tuple(text(field, k) for k in PLACE) if field in known else None
        if place != shown or first:
            if place is None:
                lines += ["", "# Not in the local data catalog", ""]
            else:
                category, subcategory, dataset = place
                if shown is None or shown[:1] != place[:1]:
                    lines += ["", f"# Category: {category or '—'}"]
                if shown is None or shown[:2] != place[:2]:
                    lines += ["", f"## Subcategory: {subcategory or '—'}"]
                about = datasets.get(dataset)
                named = f" ({about.name})" if about is not None and about.name else ""
                lines += ["", f"### Dataset: {dataset or '—'}{named}", ""]
                if about is not None and about.description:
                    lines += [str(about.description), ""]
            shown, first = place, False
        if place is None:
            lines.append(f"- `{field}`")
        else:
            lines.append(f"- `{field}` ({text(field, 'type') or '—'}): {text(field, 'description') or '—'}")
    return "\n".join(lines).strip()


def _fields_in(expression: str) -> list[str]:
    from .common import fields_in_order
    try:
        return fields_in_order(expression, grouping=True)
    except ValueError:
        return []


async def markdown(alphas: Sequence[Alpha], *, glossary: bool = True,
                   failures: Sequence[dict[str, Any]] = ()) -> str:
    """The Copy Alphas text; `failures` (expression, settings, error) are simulations that made no Alpha, listed after
    the Alphas so a reader learns from those too."""
    unique = list({a.id: a for a in alphas}.values())
    local = await local_figures([a.id for a in unique])
    constrained = await constrained_of(unique)
    blocks: list[str] = []
    regions: dict[str, str] = {}

    def by_sharpe(a: Alpha) -> float:
        return math.inf if a.sharpe is None else -float(a.sharpe)

    for a in sorted(unique, key=by_sharpe):
        expression = a.code or ""
        lines = ["Alpha Expression:", expression, "Classifications:", ", ".join(classification_names(a)) or "None",
                 "Simulation Settings:", *settings_lines(settings_of(a))]
        figures: dict[str, Any] = {"sharpe": a.sharpe, "turnover": a.turnover, "fitness": a.fitness,
                                   "returns": a.returns, "drawdown": a.drawdown, "margin": a.margin,
                                   "k_ratio": a.k_ratio, "long_count": a.long_count, "short_count": a.short_count,
                                   "prod_correlation": a.prod_corr, "max_pairwise_correlation": None,
                                   "portfolio_correlation": None}
        figures.update({k: v for k, v in local.get(a.id, {}).items() if v is not None})
        figures["imbalance"] = _imbalance(figures["long_count"], figures["short_count"])
        returns, drawdown = figures["returns"], figures["drawdown"]
        figures["calmar"] = returns / drawdown if returns is not None and drawdown else None
        lines.append("Metrics:")
        checks = settled_checks(a)
        measured = {str(c.get("name")): v for c in checks if isinstance(v := c.get("value"), (int, float))
                    and not isinstance(v, bool)}
        for key, title, shown in COPIED_METRICS:
            value = figures.get(key)
            lines.append(f"{title}: {'—' if value is None else shown(value)}")
            added: set[str] = set()
            for name, (after, label_, form) in CHECK_METRICS.items():
                if after == key and name in measured and label_ not in added:
                    added.add(label_)
                    lines.append(f"{label_}: {form(measured[name])}")
        held = constrained.get(a.id)
        if isinstance(held, dict) and held.get("sharpe") is not None:
            lines.append("Investability Constrained Metrics:")
            for key, title, shown in CONSTRAINED_FIGURES:
                value = held.get(key)
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
                lines.append(f"{title}: {shown(value) if ok else '—'}")
        listed = [f"- {_check_text(c)}" for c in checks if c.get("name") in COPIED_CHECKS]
        verdict = checks_line(checks, a)
        if listed or verdict != "Checks:":
            lines += [verdict, *listed]
        blocks.append("\n".join(lines))
        for field in _fields_in(expression):
            regions.setdefault(field, str(a.region))
    for failure in failures:
        expression = str(failure.get("expression") or "")
        settings = dict(failure.get("settings") or {})
        blocks.append("\n".join(["Alpha Expression:", expression, "Simulation Settings:", *settings_lines(settings),
                                 f"Simulation Error: {failure.get('error') or 'BRAIN gave no reason.'}"]))
        for field in _fields_in(expression):
            regions.setdefault(field, str(settings.get("region")))
    if not glossary:
        return "\n\n---\n\n".join(blocks)
    from .common import described
    known: dict[str, Any] = {}
    for region in set(regions.values()):
        known |= await described([f for f, r in regions.items() if r == region], region)
    dataset_ids = sorted({r.dataset_id for r in known.values() if r.dataset_id})
    datasets: dict[str, Dataset] = {}
    if dataset_ids:
        async with db.session() as s:
            rows = list(await s.scalars(select(Dataset).where(Dataset.id.in_(dataset_ids))))
        rows.sort(key=lambda d: (d.description is None, d.region, d.delay, d.universe))
        for d in rows:
            datasets.setdefault(d.id, d)
    blocks.append(METRIC_GLOSSARY)
    if regions:
        blocks.append(field_glossary(list(regions), known, datasets))
    return "\n\n---\n\n".join(blocks)


async def alphas_markdown(alphas: list[Alpha], *, glossary: bool = True) -> str:
    """These Alphas as Copy Alphas writes them (see the module docstring). Without `glossary`, the Metric Glossary
    and the field glossary are left out."""
    return await markdown(alphas, glossary=glossary)
