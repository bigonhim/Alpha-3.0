"""Systematic Predictions Challenge (SPC) workbench: write a prompt (by hand or with AI), dry-run it, check its JSON
output, test it on Yahoo Finance prices and submit it to BRAIN once every gate passes.

BRAIN does not publish the SPC form rules, so the checks follow `settings.spc` (editable on the SPC Lab page).
Submission only happens when you press Submit and confirm, and only after every blocking gate passed; the endpoint is
configurable because BRAIN publishes no SPC API. Only ticker symbols and dates are sent to Yahoo Finance.
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


# ---------------------------------------------------------------- full pipeline: generate → dry-run → gates → submit
#
# BRAIN runs an SPC prompt on its own model (the form's Model / Version). The app cannot run that model, so a
# "dry run" sends the prompt to your AI provider instead: it shows whether the prompt is unambiguous enough to return
# a valid, stable JSON portfolio, not what BRAIN's model will pick. Submission goes through BRAIN's API only after
# every gate passes and you confirm; the endpoint is configurable (settings.spc.submit_path / body_template) because
# BRAIN publishes no SPC API.

GENERATE_PROMPT = """\
TASK: Write a prompt for WorldQuant BRAIN's Systematic Predictions Challenge (SPC). The prompt is run by a large
language model ({model} {version}) {frequency_text}; the model's answer is read as a portfolio and scored on its
forward returns.
Strategy idea from the researcher: {idea}

The prompt you write must:
- describe a systematic, rules-based selection a model can apply from public information (factors, filters,
  ranking criteria, sector/size limits, liquidity floor); no discretionary "pick good stocks";
- give exact counts{rows_text};
- end with an exact output specification: ONLY a JSON array, no prose, no code fences, each element with the keys
  {keys};{side_text}{weight_text}
- stay under {max_chars} characters in total;
- avoid look-ahead, avoid illiquid names, and keep sector concentration bounded.
Return JSON matching the schema."""


@dataclass
class Gate:
    name: str
    ok: bool
    detail: str = ""
    blocking: bool = True


def _keys(rules: SPCSettings) -> list[str]:
    keys = [k for k in rules.required_fields if k]
    for extra in (rules.side_field, rules.weight_field):
        if extra and extra not in keys:
            keys.append(extra)
    return keys or ["company_name", "ticker"]


def generation_prompt(idea: str, rules: SPCSettings) -> str:
    rows = ""
    if rules.min_rows > 1 or rules.max_rows:
        rows = f" (between {max(1, rules.min_rows)} and {rules.max_rows or 'any number of'} stocks)"
    side = (f'\n  "{rules.side_field}" is "long" or "short";' if rules.side_field else "")
    weight = (f'\n  "{rules.weight_field}" is a positive number and the weights add up to 1;'
              if rules.weight_field else "")
    freq = {"Daily": "every trading day", "Weekly": "every week", "Monthly": "every month"}.get(rules.frequency,
                                                                                            rules.frequency)
    return GENERATE_PROMPT.format(model=rules.model, version=rules.version, frequency_text=freq,
                                  idea=idea.strip() or "a robust, diversified US equity factor portfolio",
                                  rows_text=rows, keys=", ".join(f'"{k}"' for k in _keys(rules)), side_text=side,
                                  weight_text=weight, max_chars=rules.max_chars)


async def generate_prompt(idea: str, rules: SPCSettings) -> dict[str, str]:
    """AI-written SPC prompt for a strategy idea: {name, hypothesis, prompt}."""
    from ..llm import llm
    from ..llm.schemas import SPCPromptDraft
    if not llm.configure():
        raise RuntimeError("Set up an AI provider on the LLM Integration page first.")
    draft = await llm.generate_json(generation_prompt(idea, rules), SPCPromptDraft, role="refine",
                                    purpose="spc_generate", thinking="medium")
    text = draft.prompt.strip()
    if len(text) > rules.max_chars:          # one tightening round
        draft = await llm.generate_json(
            generation_prompt(idea, rules) + f"\n\nYour previous prompt was {len(text)} characters; the limit is "
            f"{rules.max_chars}. Shorten it without losing the output specification.\n\nPrevious prompt:\n{text}",
            SPCPromptDraft, role="refine", purpose="spc_generate", thinking="low")
        text = draft.prompt.strip()
    return {"name": draft.name.strip()[:120], "hypothesis": draft.hypothesis.strip(), "prompt": text}


async def dry_run(prompt: str, runs: int = 3) -> list[str]:
    """Send the prompt to your AI provider `runs` times (one after another) and return the raw answers."""
    from ..llm import llm
    if not llm.configure():
        raise RuntimeError("Set up an AI provider on the LLM Integration page first.")
    out = []
    for _ in range(max(1, runs)):
        out.append(await llm.generate_text(prompt, role="ideation", purpose="spc_dry_run", thinking="low"))
    return out


def gates(prompt: str, outputs: list[str], rules: SPCSettings, pnl: dict[str, Any] | None = None) -> list[Gate]:
    """Every check a prompt must pass before it may be submitted (blocking unless marked otherwise)."""
    text = (prompt or "").strip()
    keys = _keys(rules)
    g: list[Gate] = [
        Gate("Prompt written", bool(text), "" if text else "The prompt is empty."),
        Gate("Character limit", 0 < len(text) <= rules.max_chars, f"{len(text):,} / {rules.max_chars:,} characters"),
        Gate("Output format specified", "json" in text.lower() and all(k in text for k in keys),
             "asks for a JSON array with " + ", ".join(keys)),
        Gate("No unfilled placeholders", not re.search(r"\{\{|\}\}|<insert|\bTODO\b|\[your", text, re.I),
             "no template placeholders left"),
    ]
    checked = [check_output(o, rules) for o in outputs if (o or "").strip()]
    if checked:
        bad = next((c.errors[0] for c in checked if not c.ok and c.errors), "")
        detail = f"{sum(c.ok for c in checked)}/{len(checked)} runs valid" + (f" — {bad}" if bad else "")
    else:
        detail = "Run a dry run (or paste outputs) first."
    g.append(Gate("Dry-run output passes the form rules", bool(checked) and all(c.ok for c in checked), detail))
    need = max(1, rules.stability_runs)
    valid = [c.rows for c in checked if c.ok]
    if need > 1:
        if len(valid) >= need:
            overlaps = [p["overlap"] for p in compare_runs(valid)["pairs"] if p["overlap"] is not None]
            worst = min(overlaps) if overlaps else 0.0
            g.append(Gate("Stable across runs", worst >= rules.min_overlap,
                          f"lowest overlap between runs {worst:.0%} (needs {rules.min_overlap:.0%})"))
        else:
            g.append(Gate("Stable across runs", False,
                          f"{len(valid)} valid run(s); {need} needed for the stability check."))
    if pnl is not None or rules.require_pnl:
        priced = (pnl or {}).get("n_priced") or 0
        total = priced + len((pnl or {}).get("missing") or [])
        share = priced / total if total else 0.0
        g.append(Gate("Tickers have prices", bool(pnl) and share >= 0.9,
                      f"{priced}/{total} tickers priced on Yahoo Finance" if pnl else "Run Test PnL first.",
                      blocking=rules.require_pnl))
    return g


def gates_pass(items: list[Gate]) -> bool:
    return bool(items) and all(x.ok for x in items if x.blocking)


def _fill(node: Any, values: dict[str, str]) -> Any:
    """Fill {placeholders} in a JSON template. A value that is exactly "{key}" is replaced whole (so prompts with quotes
    and newlines stay valid JSON); "{key:num}" becomes a number."""
    if isinstance(node, dict):
        return {k: _fill(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill(v, values) for v in node]
    if isinstance(node, str):
        m = re.fullmatch(r"\{(\w+)(:num)?\}", node)
        if m and m.group(1) in values:
            v = values[m.group(1)]
            if m.group(2):
                try:
                    f = float(v)
                    return int(f) if f.is_integer() else f
                except (TypeError, ValueError):
                    return v
            return v
        return re.sub(r"\{(\w+)\}", lambda mm: str(values.get(mm.group(1), mm.group(0))), node)
    return node


# key names in a captured request body -> the placeholder that fills them
_CAPTURE_KEYS = (("prompt", "prompt"), ("instruction", "prompt"), ("name", "name"), ("title", "name"),
                 ("version", "version"), ("model", "model"), ("frequency", "frequency"), ("weight", "weight"))


def _shell_args(text: str) -> list[str]:
    import shlex
    cleaned = text.replace("\\\r\n", " ").replace("\\\n", " ").replace("^\r\n", " ").replace("^\n", " ")
    if "^\"" in cleaned:                         # Chrome's "Copy as cURL (cmd)": ^ escapes everything
        cleaned = re.sub(r"\^(.)", r"\1", cleaned)
    try:
        return shlex.split(cleaned, posix=True)
    except ValueError:
        return cleaned.split()


def parse_captured_request(text: str) -> dict[str, Any]:
    """Turn a request copied from the browser (DevTools → Network → right-click → Copy as cURL, or Copy as fetch)
    into the SPC endpoint settings: {path, method, competition_id, body_template, mapped, unmapped}.

    String values whose key names a form field (prompt, name, model, version, frequency, weight) become placeholders;
    numbers become numeric placeholders; everything else is kept as captured."""
    from urllib.parse import urlparse
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Paste the request you copied from the browser.")
    url, method, body = "", "", None
    if raw.lower().startswith("fetch("):
        m = re.search(r"fetch\(\s*\"([^\"]+)\"", raw)
        url = m.group(1) if m else ""
        mm = re.search(r'"method"\s*:\s*"(\w+)"', raw)
        method = mm.group(1) if mm else ""
        mb = re.search(r'"body"\s*:\s*("(?:[^"\\]|\\.)*")', raw, re.S)
        if mb:
            body = json.loads(mb.group(1))
    else:
        args = _shell_args(raw)
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("-X", "--request") and i + 1 < len(args):
                method = args[i + 1]
                i += 1
            elif a in ("--data-raw", "--data", "-d", "--data-binary") and i + 1 < len(args):
                body = args[i + 1]
                i += 1
            elif a.startswith("http") and not url:
                url = a
            i += 1
    if not url:
        raise ValueError("No URL found. Copy the request with 'Copy as cURL' or 'Copy as fetch'.")
    parsed = urlparse(url)
    path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    comp = ""
    mc = re.search(r"/competitions/([^/?]+)", path)
    if mc:
        comp = mc.group(1)
        path = path.replace(f"/competitions/{comp}", "/competitions/{competition}", 1)
    try:
        data = json.loads(body) if isinstance(body, str) else (body or {})
    except json.JSONDecodeError as exc:
        raise ValueError(f"The request body is not JSON: {exc}") from exc
    mapped: dict[str, str] = {}
    unmapped: list[str] = []

    def walk(node: Any, key: str = "") -> Any:
        if isinstance(node, dict):
            return {k: walk(v, k) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, key) for v in node]
        low = key.lower()
        target = next((ph for name, ph in _CAPTURE_KEYS if name in low), None)
        if target and isinstance(node, (str, int, float)) and not isinstance(node, bool):
            mapped[key] = target
            return f"{{{target}}}" if isinstance(node, str) else f"{{{target}:num}}"
        if comp and node == comp:
            mapped[key] = "competition"
            return "{competition}"
        if key:
            unmapped.append(key)
        return node

    template = walk(data)
    if "prompt" not in mapped.values():
        raise ValueError("The captured body has no prompt field. Copy the request sent when you pressed Submit on "
                         "BRAIN's SPC page.")
    return {"path": path, "method": (method or ("POST" if body is not None else "GET")).upper(),
            "competition_id": comp, "body_template": json.dumps(template, ensure_ascii=False),
            "mapped": mapped, "unmapped": unmapped}


def submission_body(rules: SPCSettings, *, name: str, prompt: str, form: dict[str, Any]) -> dict[str, Any]:
    values = {"name": name, "prompt": prompt, "model": str(form.get("model", rules.model)),
              "version": str(form.get("version", rules.version)),
              "frequency": str(form.get("frequency", rules.frequency)),
              "weight": str(form.get("weight", rules.weight)), "competition": rules.competition_id}
    try:
        template = json.loads(rules.body_template or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"The SPC body template is not valid JSON: {exc}") from exc
    return _fill(template, values)


async def submit(*, name: str, prompt: str, form: dict[str, Any], rules: SPCSettings,
                 gate_results: list[Gate]) -> dict[str, Any]:
    """Send a prompt to BRAIN. Refuses unless every blocking gate passed and an endpoint is configured."""
    from ..brain import brain
    if not gates_pass(gate_results):
        bad = ", ".join(x.name for x in gate_results if x.blocking and not x.ok)
        raise ValueError(f"Not submitted: these checks have not passed yet: {bad}.")
    path = (rules.submit_path or "").strip()
    if not path:
        raise ValueError("No SPC submission endpoint is set. Open 'BRAIN submission endpoint' and paste the request "
                         "path your browser sends when you submit on BRAIN's SPC page.")
    path = path.replace("{competition}", rules.competition_id or "")
    body = submission_body(rules, name=name, prompt=prompt, form=form)
    response = await brain.client.request((rules.submit_method or "POST").upper(), path, json=body,
                                          retry_statuses=(500, 502, 504))
    payload = brain.client.raise_for(response)
    return {"status": response.status_code, "path": path,
            "response": payload if isinstance(payload, (dict, list)) else str(payload or "")[:500]}


async def find_competitions() -> list[dict[str, Any]]:
    """Competitions your account is enrolled in (to find the SPC id for the endpoint)."""
    from ..brain import brain
    body = await brain.users.competitions()
    items = body if isinstance(body, list) else (body or {}).get("results") or []
    return [{"id": c.get("id"), "name": c.get("name") or c.get("title"), "status": c.get("status"),
             "start": c.get("startDate") or c.get("dateStart"), "end": c.get("endDate") or c.get("dateEnd")}
            for c in items if isinstance(c, dict)]
