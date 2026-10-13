"""Systematic Predictions Challenge (SPC) workbench: check a prompt's JSON output and test it on Yahoo Finance prices.

BRAIN does not publish the SPC form rules, so the checks follow `settings.spc` (editable on the SPC Lab page). The app
never submits anything: the prompt is pasted into BRAIN's form by the user. Only ticker symbols and dates are sent to
Yahoo Finance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..config import SPCSettings

log = logging.getLogger(__name__)

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,6}([.\-][A-Z0-9]{1,3})?$")
LONG = {"long", "buy", "l", "1", "+1"}
SHORT = {"short", "sell", "s", "-1"}


@dataclass
class CheckResult:
    ok: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def tickers(self) -> list[str]:
        return [norm_ticker(r.get("ticker")) for r in self.rows if r.get("ticker")]


def norm_ticker(t: Any) -> str:
    return str(t or "").strip().upper().replace("/", "-")


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", t, re.S)
    return m.group(1) if m else t


def parse_output(text: str) -> list[dict[str, Any]]:
    data = json.loads(_strip_fences(text))
    if not isinstance(data, list):
        raise ValueError("The output must be a JSON array: [ {...}, {...} ]")
    return data


def side_of(row: dict[str, Any], rules: SPCSettings) -> int:
    if not rules.side_field:
        return 1
    v = str(row.get(rules.side_field, "")).strip().lower()
    return -1 if v in SHORT else 1


def check_output(text: str, rules: SPCSettings) -> CheckResult:
    if not (text or "").strip():
        return CheckResult(False, errors=["Paste the JSON array the prompt returned."])
    try:
        data = parse_output(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return CheckResult(False, errors=[f"Not valid JSON: {exc}"])
    res = CheckResult(True)
    seen: dict[str, int] = {}
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            res.errors.append(f"Row {i} is not an object.")
            continue
        missing = [f for f in rules.required_fields if f and not str(item.get(f, "")).strip()]
        if missing:
            res.errors.append(f"Row {i} is missing {', '.join(missing)}.")
        t = norm_ticker(item.get("ticker"))
        if t:
            if not TICKER_RE.match(t):
                res.warnings.append(f"Row {i}: '{item.get('ticker')}' does not look like a ticker.")
            elif str(item.get("ticker")).strip() != t:
                res.warnings.append(f"Row {i}: ticker '{item.get('ticker')}' will be read as '{t}'.")
            if t in seen:
                res.errors.append(f"Row {i}: duplicate ticker {t} (also row {seen[t]}).")
            seen.setdefault(t, i)
        if rules.weight_field:
            w = item.get(rules.weight_field)
            try:
                if w is None or not math.isfinite(float(w)):
                    raise ValueError
            except (TypeError, ValueError):
                res.errors.append(f"Row {i}: {rules.weight_field} must be a number.")
        if rules.side_field:
            v = str(item.get(rules.side_field, "")).strip().lower()
            if v not in LONG | SHORT:
                res.errors.append(f"Row {i}: {rules.side_field} must be long/short (got '{item.get(rules.side_field)}').")
        res.rows.append(item)
    n = len(res.rows)
    if n < rules.min_rows:
        res.errors.append(f"{n} rows; at least {rules.min_rows} required.")
    if rules.max_rows and n > rules.max_rows:
        res.errors.append(f"{n} rows; at most {rules.max_rows} allowed.")
    if rules.weight_field and not res.errors:
        total = sum(abs(float(r[rules.weight_field])) for r in res.rows)
        if not (0.98 <= total <= 1.02 or 98 <= total <= 102):
            res.warnings.append(f"Weights add up to {total:g} (expected 1 or 100); the test normalises them.")
    res.ok = not res.errors
    return res


# ---------------------------------------------------------------- prices and PnL

async def fetch_closes(tickers: list[str], start: date, end: date | None) -> pd.DataFrame:
    """Daily adjusted closes from Yahoo Finance (yfinance runs in a worker thread)."""
    import yfinance as yf

    def run() -> pd.DataFrame:
        data = yf.download(sorted(set(tickers)), start=start.isoformat(),
                           end=(end + timedelta(days=1)).isoformat() if end else None,
                           auto_adjust=True, progress=False, threads=True)
        if data is None or len(data) == 0:
            return pd.DataFrame()
        closes = data["Close"] if "Close" in data.columns.get_level_values(0) else data
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(name=tickers[0])
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        return closes.dropna(how="all", axis=1)

    return await asyncio.to_thread(run)


def portfolio_returns(closes: pd.DataFrame, rows: list[dict[str, Any]], rules: SPCSettings,
                      run_date: date) -> tuple[pd.Series, dict[str, float], list[str]]:
    """Daily portfolio returns from the run date (entry at the first close on/after it), daily-rebalanced to the
    given weights (equal weight unless a weight field is configured; short rows count negative)."""
    weights: dict[str, float] = {}
    for r in rows:
        t = norm_ticker(r.get("ticker"))
        if not t:
            continue
        w = abs(float(r.get(rules.weight_field) or 0)) if rules.weight_field else 1.0
        weights[t] = weights.get(t, 0.0) + w * side_of(r, rules)
    missing = sorted(t for t in weights if t not in closes.columns or closes[t].dropna().empty)
    use = {t: w for t, w in weights.items() if t not in missing}
    gross = sum(abs(w) for w in use.values()) or 1.0
    use = {t: w / gross for t, w in use.items()}
    px = closes[list(use)].loc[closes.index >= pd.Timestamp(run_date)]
    rets = px.pct_change().iloc[1:]
    port = (rets.fillna(0.0) * pd.Series(use)).sum(axis=1)
    per_name = {t: float((1 + rets[t].fillna(0)).prod() - 1) for t in use}
    return port, per_name, missing


def summary(port: pd.Series, bench: pd.Series | None = None) -> dict[str, Any]:
    if port.empty:
        return {"days": 0}
    cum = (1 + port).cumprod()
    sd = float(port.std(ddof=1)) if len(port) > 1 else 0.0
    peak = cum.cummax()
    out = {"days": int(len(port)), "total_return": float(cum.iloc[-1] - 1),
           "sharpe": float(math.sqrt(252) * port.mean() / sd) if sd > 0 and len(port) >= 5 else None,
           "max_drawdown": float(((peak - cum) / peak).max()), "start": str(port.index[0].date()),
           "end": str(port.index[-1].date())}
    if bench is not None and not bench.empty:
        b = bench.reindex(port.index).fillna(0.0)
        out["benchmark_return"] = float((1 + b).prod() - 1)
        out["excess_return"] = out["total_return"] - out["benchmark_return"]
    return out


async def test_pnl(rows: list[dict[str, Any]], rules: SPCSettings, run_date: date,
                   test_until: date | None = None) -> dict[str, Any]:
    tickers = [norm_ticker(r.get("ticker")) for r in rows if r.get("ticker")]
    bench = rules.benchmark.strip().upper()
    closes = await fetch_closes(tickers + ([bench] if bench else []), run_date - timedelta(days=7), test_until)
    if closes.empty:
        raise RuntimeError("Yahoo Finance returned no prices (check the tickers, the dates and your connection).")
    port, per_name, missing = portfolio_returns(closes, rows, rules, run_date)
    bench_ret = None
    if bench and bench in closes.columns:
        bp = closes[bench].loc[closes.index >= pd.Timestamp(run_date)]
        bench_ret = bp.pct_change().iloc[1:]
    res = summary(port, bench_ret)
    ranked = sorted(per_name.items(), key=lambda kv: kv[1], reverse=True)
    n_side = min(5, len(ranked) // 2) or len(ranked)
    res.update(missing=missing, best=ranked[:n_side], worst=ranked[len(ranked) - n_side:][::-1] if len(ranked) > 1 else [],
               n_priced=len(per_name),
               curve={"dates": [str(d.date()) for d in port.index],
                      "portfolio": [round(float(x) - 1, 5) for x in (1 + port).cumprod()],
                      "benchmark": [round(float(x) - 1, 5) for x in (1 + bench_ret.reindex(port.index).fillna(0)).cumprod()]
                      if bench_ret is not None else []})
    return res


# ---------------------------------------------------------------- stability

def compare_runs(outputs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    sets = [{norm_ticker(r.get("ticker")) for r in out if r.get("ticker")} for out in outputs]
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            pairs.append({"runs": f"{i + 1}–{j + 1}", "overlap": len(sets[i] & sets[j]) / len(union) if union else None})
    common = sorted(set.intersection(*sets)) if sets else []
    return {"sizes": [len(s) for s in sets], "pairs": pairs, "common": common}


async def stability_pnl(outputs: list[list[dict[str, Any]]], rules: SPCSettings, run_date: date,
                        test_until: date | None) -> dict[str, Any]:
    tickers = sorted({norm_ticker(r.get("ticker")) for out in outputs for r in out if r.get("ticker")})
    closes = await fetch_closes(tickers, run_date - timedelta(days=7), test_until)
    series = [portfolio_returns(closes, out, rules, run_date)[0] for out in outputs]
    df = pd.DataFrame({f"run {i + 1}": s for i, s in enumerate(series)})
    corr = df.corr().round(3)
    totals = [float((1 + s).prod() - 1) if not s.empty else None for s in series]
    return {"totals": totals, "corr": corr.where(np.isfinite(corr), None).to_dict()}


# ---------------------------------------------------------------- starter prompts

STARTER_PROMPTS: dict[str, str] = {
    "Quality momentum (long only) v1": """\
You are a systematic equity analyst. Build a portfolio of 20 liquid US-listed stocks (market cap above $5B) to hold
from today. Rank candidates on: 6–12 month price momentum (excluding the last month), improving gross profitability,
and positive analyst estimate revisions over the last 3 months. Avoid stocks with earnings in the next 5 trading days
and avoid more than 4 stocks from one sector.
Output ONLY a JSON array, no prose, no code fences. Each element:
{"company_name": "<full company name>", "ticker": "<primary US ticker>"}""",
    "Earnings Season Drift LS v1": """\
You are a systematic equity analyst. Using post-earnings-announcement drift, pick 10 US-listed stocks to buy and 10 to
sell short from companies that reported in the last 10 trading days: long the largest positive surprises with raised
guidance, short the largest negative surprises with cut guidance. Market cap above $2B, average daily volume above
$20M, at most 3 names per sector on each side.
Output ONLY a JSON array, no prose, no code fences. Each element:
{"company_name": "<full company name>", "ticker": "<primary US ticker>", "side": "long" or "short"}""",
    "Value with revisions v1": """\
You are a systematic equity analyst. Choose 20 US-listed stocks that are cheap versus their sector on forward
earnings and free-cash-flow yield AND have positive analyst EPS revisions in the last 60 days. Exclude banks with
recent capital raises, stocks under $10, and companies with pending mergers.
Output ONLY a JSON array, no prose, no code fences. Each element:
{"company_name": "<full company name>", "ticker": "<primary US ticker>"}""",
}
