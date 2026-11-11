"""SPC generation and submission: write a Systematic Predictions Challenge prompt with the AI, run it, check it against
gates, rewrite it when a gate fails, and submit it to BRAIN only once every gate has passed.

Pipeline (one job, `spc_generate`):
  1. draft    the AI writes the prompt from a short brief and the form rules (output fields, limits);
  2. run      the prompt is run `auto_runs` times on the app's LLM. BRAIN runs the prompt on its own model with its own
              data, so these runs are a *proxy*: good for the form check, ticker validity and stability, not a forecast;
  3. gates    prompt length, prompt format, no placeholders, form rules, clean output, stability across runs, tickers
              priced on Yahoo, optional backtest;
  4. rewrite  a failed gate sends its specific evidence (the bad rows, the unstable names, the unpriced tickers) back to
              the AI for a rewrite, up to `auto_max_rounds` times;
  5. save     the prompt, its report and its runs are stored (they appear in SPC Lab's library and saved runs);
  6. submit   `spc_submit`, only for a prompt whose report says every gate passed, only after the user's confirmation
              click (or, if the user switched `auto_submit` on, right after the gates pass), at most
              `max_submits_per_day`, never twice for the same prompt, and never two submissions at the same time.

Nothing outside SPC is touched. The BRAIN call goes through `brain.spc` and the endpoint comes from settings.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ..brain import brain
from ..brain import spc as brain_spc
from ..brain.quota import EASTERN
from ..config import SPCSettings, settings
from ..db import db
from ..db.models import SPCPrompt, SPCRun
from ..events import bus
from ..jobs.queue import JobContext, jobs
from ..llm import llm
from . import spc

log = logging.getLogger(__name__)

Progress = Callable[[float, str], Awaitable[None]]
PASSING = {"pass", "warn", "off"}            # a gate in any other state (fail, error) blocks submission
SUBMIT_LOG_KEY = "spc_submit_log"
COUNTED = ("accepted", "unknown", "sending")  # these use up the daily guard (an unknown outcome may have gone through)
_PLACEHOLDER_LEFT = re.compile(r"\{\{|\}\}|<insert|\bTODO\b|\[your|\bLOREM\b", re.I)
_submit_lock = asyncio.Lock()


# ---------------------------------------------------------------- brief, draft schema

@dataclass
class Brief:
    theme: str = "quality momentum"
    n_stocks: int = 20
    style: str = "long"                      # long | long_short
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Brief:
        d = d or {}
        return cls(theme=str(d.get("theme") or cls.theme).strip(), n_stocks=max(1, int(d.get("n_stocks") or 20)),
                   style="long_short" if str(d.get("style")) == "long_short" else "long",
                   notes=str(d.get("notes") or "").strip())


class PromptDraft(BaseModel):
    name: str = Field(description="Short name for the prompt, 3 to 6 words")
    prompt: str = Field(description="The complete prompt text, ready to paste into BRAIN's SPC form")
    rationale: str = Field(default="", description="One or two sentences on the selection logic")


@dataclass
class Gate:
    name: str
    status: str                              # pass | warn | fail | error | off
    detail: str = ""

    @property
    def blocks(self) -> bool:
        return self.status not in PASSING


@dataclass
class RunOutcome:
    text: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    ok: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extracted: bool = False                  # the JSON array had to be cut out of surrounding text

    @property
    def tickers(self) -> set[str]:
        return {spc.norm_ticker(r.get("ticker")) for r in self.rows if r.get("ticker")}


# ---------------------------------------------------------------- prompt drafting

def output_fields(rules: SPCSettings) -> list[str]:
    fields = [f for f in rules.required_fields if f] or ["company_name", "ticker"]
    for extra in (rules.weight_field, rules.side_field):
        if extra and extra not in fields:
            fields.append(extra)
    return fields


def output_spec(rules: SPCSettings) -> str:
    """The exact JSON element the prompt must make the model return, derived from the form rules."""
    parts = []
    for f in output_fields(rules):
        if f == "ticker":
            parts.append('"ticker": "<primary US ticker, e.g. AAPL>"')
        elif f == "company_name":
            parts.append('"company_name": "<full company name>"')
        elif f == rules.weight_field:
            parts.append(f'"{f}": <number: decimal fraction, all weights add up to 1>')
        elif f == rules.side_field:
            parts.append(f'"{f}": "long" or "short"')
        else:
            parts.append(f'"{f}": "<{f}>"')
    return "{" + ", ".join(parts) + "}"


DRAFT_TEMPLATE = """\
You write prompts for the Systematic Predictions Challenge (SPC) on WorldQuant BRAIN. A language model reads the prompt
{frequency_l} and must answer with a list of US-listed stocks that is scored on forward returns.

Write ONE prompt for this brief.
Theme: {theme}
Portfolio: {portfolio}
{notes}
Requirements for the prompt you write:
- A role line, then the selection method as an explicit, ordered procedure: a defined universe (US-listed, a market-cap
  floor, a liquidity floor), 2 to 4 ranking criteria with how each is measured, tie-breakers, and exclusions (earnings
  inside the next 5 trading days, pending mergers, penny stocks, OTC, ADRs).
- Deterministic on purpose: the same prompt run twice must return mostly the same names, so use ranked rules with
  explicit cut-offs rather than "pick the best ideas".
- Only information a careful analyst could know on the run date. No future dates, no invented data.
- No template placeholders left for a human to fill in.
- The reply must be ONLY a JSON array, no prose, no code fences. State this at the end, with the exact element shape:
  {spec}
- At most {max_len} characters in total (a hard limit is {max_chars}).
Return JSON with: name, prompt, rationale."""

REVISE_TEMPLATE = """\
This Systematic Predictions Challenge prompt failed checks when it was run {runs} times. Rewrite it so the checks pass,
keeping its investment idea ({theme}). Fix exactly what is listed; do not make unrelated changes.

PROMPT:
{prompt}

WHAT FAILED (measured):
{feedback}

The reply must remain ONLY a JSON array of {spec}
At most {max_len} characters in total. Return JSON with: name, prompt, rationale."""


async def draft_prompt(brief: Brief, rules: SPCSettings) -> PromptDraft:
    portfolio = (f"{brief.n_stocks} stocks, long only" if brief.style == "long" else
                 f"about {brief.n_stocks} stocks in total, long and short (each row says which side)")
    text = DRAFT_TEMPLATE.format(
        frequency_l=rules.frequency.lower(), theme=brief.theme, portfolio=portfolio,
        notes=f"Notes: {brief.notes}\n" if brief.notes else "", spec=output_spec(rules),
        max_len=int(rules.max_chars * 0.6), max_chars=rules.max_chars)
    return await llm.generate_json(text, PromptDraft, role="ideation", purpose="spc_draft", thinking="medium")


async def revise_prompt(prompt: str, brief: Brief, feedback: str, runs: int, rules: SPCSettings) -> PromptDraft:
    text = REVISE_TEMPLATE.format(runs=runs, theme=brief.theme, prompt=prompt, feedback=feedback,
                                  spec=output_spec(rules), max_len=int(rules.max_chars * 0.6))
    return await llm.generate_json(text, PromptDraft, role="refine", purpose="spc_revise", thinking="medium")


# ---------------------------------------------------------------- running the prompt locally

def _array_slice(text: str) -> str | None:
    start, end = text.find("["), text.rfind("]")
    return text[start:end + 1] if 0 <= start < end else None


def parse_run(text: str, rules: SPCSettings) -> RunOutcome:
    """Form check of one reply. A reply with text around the array still gets checked, with a warning."""
    res = spc.check_output(text, rules)
    extracted = False
    if not res.ok and any(e.startswith("Not valid JSON") for e in res.errors):
        cut = _array_slice(text or "")
        if cut:
            alt = spc.check_output(cut, rules)
            if alt.ok or not any(e.startswith("Not valid JSON") for e in alt.errors):
                res, extracted = alt, True
    return RunOutcome(text=text, rows=res.rows, ok=res.ok, errors=list(res.errors), warnings=list(res.warnings),
                      extracted=extracted)


async def run_prompt_once(prompt: str, run_date: date) -> str:
    system = (f"Today's date is {run_date.isoformat()}. You have no live market data: use what you reliably know, "
              "and follow the user's output instructions exactly.")
    return await llm.generate_text(prompt, system=system, role="ideation", purpose="spc_run", thinking="low")


async def run_many(prompt: str, rules: SPCSettings, run_date: date, n: int,
                   progress: Progress | None = None, base: float = 0.0, span: float = 0.0) -> list[RunOutcome]:
    out: list[RunOutcome] = []
    for i in range(n):
        try:
            out.append(parse_run(await run_prompt_once(prompt, run_date), rules))
        except Exception as exc:              # one failed AI call is a failed run, not a failed job
            log.warning("SPC run %d failed: %s", i + 1, exc)
            out.append(RunOutcome(text="", errors=[f"The AI call failed: {getattr(exc, 'message', None) or exc}"]))
        if progress:
            await progress(base + span * (i + 1) / n, f"Ran the prompt {i + 1}/{n}")
    return out


# ---------------------------------------------------------------- gates

def mean_overlap(runs: list[RunOutcome]) -> float | None:
    sets = [r.tickers for r in runs if r.ok and r.tickers]
    if len(sets) < 2:
        return None
    vals = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    return sum(vals) / len(vals)


def evaluate_gates(prompt: str, rules: SPCSettings, runs: list[RunOutcome], prices: dict[str, Any] | None,
                   backtest: dict[str, Any] | None) -> list[Gate]:
    """Every gate, in order. `prices` is {"n": tickers, "unpriced": [...]} or {"error": msg} or None (check off);
    `backtest` is a spc.test_pnl result, {"error": msg} or None (off)."""
    gates: list[Gate] = []
    n = len(prompt)
    gates.append(Gate("Prompt length", "pass" if 0 < n <= rules.max_chars else "fail",
                      f"{n:,} / {rules.max_chars:,} characters"))

    missing = [f for f in output_fields(rules) if f.lower() not in prompt.lower()]
    if "json" not in prompt.lower() or missing:
        why = ["does not say the reply is JSON"] if "json" not in prompt.lower() else []
        if missing:
            why.append("does not name the field(s) " + ", ".join(missing))
        gates.append(Gate("Prompt format", "warn", "The prompt " + " and ".join(why) + "."))
    else:
        gates.append(Gate("Prompt format", "pass", "Asks for JSON and names every output field."))

    left = _PLACEHOLDER_LEFT.search(prompt)
    gates.append(Gate("No placeholders", "fail" if left else "pass",
                      f"Template text left in the prompt: {left.group(0)!r}." if left else
                      "No template placeholders left."))

    bad = [(i, r) for i, r in enumerate(runs, 1) if not r.ok]
    if not runs:
        gates.append(Gate("Form rules", "fail", "The prompt produced no replies."))
    elif bad:
        gates.append(Gate("Form rules", "fail", "; ".join(
            f"run {i}: {(r.errors or ['no rows'])[0]}" + (f" (+{len(r.errors) - 1} more)" if len(r.errors) > 1 else "")
            for i, r in bad[:3]) + f" — {len(bad)} of {len(runs)} runs fail the form."))
    else:
        sizes = ", ".join(str(len(r.rows)) for r in runs)
        gates.append(Gate("Form rules", "pass", f"All {len(runs)} runs pass ({sizes} stocks)."))

    wrapped = [i for i, r in enumerate(runs, 1) if r.extracted]
    gates.append(Gate("Clean output", "warn" if wrapped else "pass",
                      f"Run(s) {', '.join(map(str, wrapped))} had text around the JSON array; the form wants the array "
                      "alone." if wrapped else "Every reply was the bare JSON array."))

    ov = mean_overlap(runs)
    if ov is None:
        gates.append(Gate("Stability", "fail", "Needs at least two valid runs to compare."))
    else:
        gates.append(Gate("Stability", "pass" if ov >= rules.auto_min_overlap else "fail",
                          f"Mean stock overlap between runs {ov:.0%} (needs {rules.auto_min_overlap:.0%})."))

    if not rules.auto_check_prices:
        gates.append(Gate("Tickers priced", "off", "Switched off in the settings."))
    elif prices is None or "error" in (prices or {}):
        gates.append(Gate("Tickers priced", "error",
                          f"Could not check prices: {(prices or {}).get('error', 'no valid output')}."))
    else:
        total, unpriced = int(prices["n"]), list(prices["unpriced"])
        share = 1 - len(unpriced) / total if total else 0.0
        gates.append(Gate("Tickers priced", "pass" if share >= rules.auto_min_priced else "fail",
                          f"{share:.0%} of {total} tickers have Yahoo prices (needs {rules.auto_min_priced:.0%})"
                          + (f"; missing: {', '.join(unpriced[:12])}." if unpriced else ".")))

    if rules.auto_backtest_days <= 0:
        gates.append(Gate("Backtest", "off", "Switched off (the AI may remember how that period went)."))
    elif backtest is None or "error" in backtest:
        gates.append(Gate("Backtest", "error", f"Could not run: {(backtest or {}).get('error', 'no valid output')}."))
    else:
        got = backtest.get("excess_return", backtest.get("total_return"))
        label = "vs benchmark" if "excess_return" in backtest else "total return"
        ok = got is not None and got >= rules.auto_min_excess
        gates.append(Gate("Backtest", "pass" if ok else "fail",
                          f"{rules.auto_backtest_days} days back: {got:+.2%} {label} (needs {rules.auto_min_excess:+.2%}); "
                          "a hindsight-prone test, treat it as a smoke test." if got is not None else "No trading days."))
    return gates


def ready(gates: list[Gate]) -> bool:
    return bool(gates) and not any(g.blocks for g in gates)


def failure_feedback(gates: list[Gate], runs: list[RunOutcome], rules: SPCSettings) -> str:
    """Evidence for the rewrite: each failed gate with its own numbers and the rows behind them."""
    lines: list[str] = []
    for g in gates:
        if not g.blocks:
            continue
        lines.append(f"- {g.name}: {g.detail}")
        if g.name == "Form rules":
            for i, r in enumerate(runs, 1):
                for e in r.errors[:3]:
                    lines.append(f"    run {i}: {e}")
            lines.append(f"    Every element must have exactly: {output_spec(rules)}")
        elif g.name == "Stability":
            sets = [r.tickers for r in runs if r.ok]
            if sets:
                counts: dict[str, int] = {}
                for s_ in sets:
                    for t in s_:
                        counts[t] = counts.get(t, 0) + 1
                shaky = sorted(t for t, c in counts.items() if c < len(sets))
                lines.append(f"    Only {len(counts) - len(shaky)} names appeared in every run; the shaky ones: "
                             f"{', '.join(shaky[:20])}. Add explicit numeric cut-offs, a fixed ranking formula and "
                             "tie-breakers so the same names win every time.")
        elif g.name == "Tickers priced":
            lines.append("    Use only the primary ticker of currently listed NYSE/Nasdaq stocks: no ADRs, OTC, delisted "
                         "or renamed tickers, and write class shares like BRK-B.")
        elif g.name == "Prompt length":
            lines.append("    Shorten it: cut examples and repetition, keep the procedure and the output shape.")
        elif g.name == "No placeholders":
            lines.append("    Replace every placeholder with the concrete rule or number it stands for.")
        elif g.name == "Backtest":
            lines.append("    Revisit the ranking criteria; do not add look-ahead.")
    return "\n".join(lines) or "- (no detail)"


# ---------------------------------------------------------------- pipeline

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _price_check(runs: list[RunOutcome], run_date: date) -> dict[str, Any] | None:
    tickers = sorted(set().union(*(r.tickers for r in runs if r.ok))) if any(r.ok for r in runs) else []
    if not tickers:
        return None
    try:
        closes = await spc.fetch_closes(tickers, run_date - timedelta(days=10), None)
    except Exception as exc:  # network, throttling, yfinance changes
        return {"error": str(exc)[:200] or exc.__class__.__name__}
    if closes is None or closes.empty:
        return {"error": "Yahoo Finance returned no prices"}
    unpriced = [t for t in tickers if t not in closes.columns or closes[t].dropna().empty]
    return {"n": len(tickers), "unpriced": unpriced}


async def _backtest(runs: list[RunOutcome], rules: SPCSettings, run_date: date) -> dict[str, Any] | None:
    first = next((r for r in runs if r.ok), None)
    if first is None:
        return None
    try:
        return await spc.test_pnl(first.rows, rules, run_date - timedelta(days=rules.auto_backtest_days), None)
    except Exception as exc:
        return {"error": str(exc)[:200] or exc.__class__.__name__}


async def run_pipeline(*, brief: Brief | None = None, prompt: str | None = None, name: str | None = None,
                       rules: SPCSettings | None = None, progress: Progress | None = None,
                       revise: bool | None = None, run_date: date | None = None) -> dict[str, Any]:
    """Draft (or take) a prompt, run it, gate it, rewrite on failure, save the result. Returns the report."""
    rules = rules or settings.spc
    brief = brief or Brief()
    run_date = run_date or date.today()
    do_revise = (prompt is None) if revise is None else revise
    rounds = (max(0, rules.auto_max_rounds) if do_revise else 0) + 1
    n_runs = max(2, rules.auto_runs)
    step = 1.0 / (rounds * (n_runs + 2))

    async def tick(frac: float, msg: str) -> None:
        if progress:
            await progress(min(0.97, frac), msg)

    cur_prompt, cur_name, rationale = prompt, name, ""
    attempts: list[dict[str, Any]] = []
    for rnd in range(rounds):
        base = rnd / rounds
        if cur_prompt is None:
            await tick(base, "Writing the prompt")
            d = await draft_prompt(brief, rules)
            cur_prompt, cur_name, rationale = d.prompt.strip(), d.name.strip(), d.rationale
        elif rnd > 0:
            fb = failure_feedback(attempts[-1]["gate_objs"], attempts[-1]["runs"], rules)
            await tick(base, f"Round {rnd + 1}: rewriting the prompt for the failed checks")
            d = await revise_prompt(cur_prompt, brief, fb, n_runs, rules)
            cur_prompt, cur_name, rationale = d.prompt.strip(), (d.name.strip() or cur_name), d.rationale
        runs = await run_many(cur_prompt, rules, run_date, n_runs, progress=tick, base=base + step, span=step * n_runs)
        prices = await _price_check(runs, run_date) if rules.auto_check_prices else None
        await tick(base + step * (n_runs + 1), "Checking tickers and gates")
        backtest = await _backtest(runs, rules, run_date) if rules.auto_backtest_days > 0 else None
        gates = evaluate_gates(cur_prompt, rules, runs, prices, backtest)
        attempts.append({"round": rnd + 1, "prompt": cur_prompt, "name": cur_name, "rationale": rationale,
                         "runs": runs, "gate_objs": gates, "ready": ready(gates), "overlap": mean_overlap(runs)})
        if attempts[-1]["ready"]:
            break

    best = attempts[-1] if attempts[-1]["ready"] else max(
        attempts, key=lambda a: (sum(not g.blocks for g in a["gate_objs"]), a["round"]))
    return await _save(best, len(attempts), brief, rules, run_date, prompt_given=prompt is not None)


async def _save(best: dict[str, Any], rounds_used: int, brief: Brief, rules: SPCSettings, run_date: date,
                *, prompt_given: bool) -> dict[str, Any]:
    gates: list[Gate] = best["gate_objs"]
    runs: list[RunOutcome] = best["runs"]
    nm = (best["name"] or "").strip()[:120] or f"AI prompt {run_date.isoformat()}"
    stats = {"ready": best["ready"], "gates": [asdict(g) for g in gates], "overlap": best["overlap"],
             "rounds": rounds_used, "prompt_sha": _sha(best["prompt"]), "source": "editor" if prompt_given else "ai",
             "rationale": best["rationale"], "run_date": run_date.isoformat(), "runs": len(runs),
             "brief": asdict(brief), "generated_at": datetime.now(timezone.utc).isoformat()}
    async with db.write() as s:
        version = (await s.scalar(select(func.max(SPCPrompt.version)).where(SPCPrompt.name == nm)) or 0) + 1
        row = SPCPrompt(name=nm, version=version, prompt=best["prompt"], stats=stats, submitted=False,
                        form_settings={"model": rules.model, "version": rules.version, "frequency": rules.frequency,
                                       "weight": rules.weight})
        s.add(row)
        await s.flush()
        pid = row.id
        for i, r in enumerate(runs, 1):
            if r.ok:
                s.add(SPCRun(prompt_id=pid, label=f"{nm} v{version} (AI run {i})", run_date=run_date.isoformat(),
                             output=r.rows, check={"errors": r.errors, "warnings": r.warnings}))
    return {"prompt_id": pid, "name": nm, "version": version, "prompt": best["prompt"], "ready": best["ready"],
            "gates": stats["gates"], "overlap": best["overlap"], "rounds": rounds_used,
            "runs": [{"rows": len(r.rows), "ok": r.ok, "errors": r.errors[:3], "extracted": r.extracted} for r in runs],
            "rationale": best["rationale"]}


def summarize(report: dict[str, Any]) -> str:
    failed = [g["name"] for g in report["gates"] if g["status"] not in PASSING]
    head = f"{report['name']} v{report['version']}"
    if report["ready"]:
        return f"Ready: {head} passed every gate after {report['rounds']} round(s)."
    return f"Not ready: {head} failed {', '.join(failed)} after {report['rounds']} round(s)."


# ---------------------------------------------------------------- submission

def _day_start_utc() -> datetime:
    return datetime.now(EASTERN).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


async def submit_log() -> list[dict[str, Any]]:
    return list(await db.kv_get(SUBMIT_LOG_KEY, []) or [])


async def submissions_today() -> int:
    """Submissions counted against the daily guard: accepted, sent with an unknown outcome, or still being sent
    (a send the app never heard back from, for example because it was closed mid-request)."""
    start = _day_start_utc()
    return sum(1 for e in await submit_log() if e.get("status") in COUNTED and datetime.fromisoformat(e["ts"]) >= start)


async def _load_submittable(prompt_id: int, rules: SPCSettings) -> SPCPrompt:
    async with db.session() as s:
        row = await s.get(SPCPrompt, prompt_id)
    if row is None:
        raise RuntimeError("That prompt is not in the library.")
    stats = row.stats or {}
    if not stats.get("ready"):
        raise RuntimeError("This prompt has not passed the generation checks. Run 'Generate & check' (or 'Check the "
                           "prompt in the editor') first; a prompt is only submitted once every gate has passed.")
    if stats.get("prompt_sha") != _sha(row.prompt):
        raise RuntimeError("The prompt text changed after it passed the checks. Check it again.")
    if row.submitted:
        raise RuntimeError("This prompt version was already submitted.")
    return row


async def preview_request(prompt_id: int, rules: SPCSettings | None = None) -> dict[str, Any]:
    """The exact request that would be sent. Sends nothing."""
    rules = rules or settings.spc
    row = await _load_submittable(prompt_id, rules)
    req = brain_spc.build_request(rules, name=f"{row.name} v{row.version}", prompt=row.prompt, form=row.form_settings)
    return {"method": req.method, "path": req.path, "body": req.body}


async def _log_entry(entry: dict[str, Any], replace_ts: str | None = None) -> None:
    log_ = [e for e in await submit_log() if not (replace_ts and e.get("ts") == replace_ts
                                                   and e.get("prompt_id") == entry.get("prompt_id"))]
    await db.kv_set(SUBMIT_LOG_KEY, [*log_[-200:], entry])


async def submit_prompt(prompt_id: int, *, confirmed: bool, rules: SPCSettings | None = None) -> str:
    """Send a prompt that passed every gate to BRAIN. `confirmed` must come from the user's click in the dialog (or
    from the opt-in auto-submit setting)."""
    if not confirmed:
        raise PermissionError("Submission requires explicit user confirmation.")
    rules = rules or settings.spc
    async with _submit_lock:                  # two clicks never send two submissions at the same time
        row = await _load_submittable(prompt_id, rules)
        used = await submissions_today()
        if used >= rules.max_submits_per_day:
            raise RuntimeError(f"Daily SPC submission guard reached ({used}/{rules.max_submits_per_day}). Raise "
                               "'Max submissions per day' if BRAIN allows more.")
        nm = f"{row.name} v{row.version}"
        req = brain_spc.build_request(rules, name=nm, prompt=row.prompt, form=row.form_settings)
        ts = datetime.now(timezone.utc).isoformat()
        await _log_entry({"ts": ts, "prompt_id": prompt_id, "status": "sending", "http": None})
        bus.publish("submit", f"Submitting SPC prompt {nm} to BRAIN…")
        result = await brain_spc.SPCAPI(brain.client).submit(req)
        entry = {"ts": ts, "prompt_id": prompt_id, "status": result.status, "http": result.http_status}
        await _log_entry(entry, replace_ts=ts)
        async with db.write() as s:
            fresh = await s.get(SPCPrompt, prompt_id)
            stats = dict(fresh.stats or {})
            stats["submission"] = {**entry, "message": result.message[:500]}
            fresh.stats = stats
            fresh.submitted = result.accepted
    level = "success" if result.accepted else "error"
    bus.publish("submit", f"SPC prompt {nm}: {result.status} — {result.message}", level)
    return f"{result.status}: {result.message}"


# ---------------------------------------------------------------- jobs

@jobs.register("spc_generate")
async def _job_generate(ctx: JobContext, params: dict) -> str:
    async def progress(frac: float, msg: str) -> None:
        await ctx.update(frac, msg)

    report = await run_pipeline(brief=Brief.from_dict(params.get("brief")), prompt=params.get("prompt") or None,
                                name=params.get("name") or None, progress=progress,
                                revise=params.get("revise"))
    text = summarize(report)
    bus.publish("spc", text + (" Open SPC Lab to review and submit." if report["ready"] else ""),
                "success" if report["ready"] else "warning", kind="spc_generate", prompt_id=report["prompt_id"])
    if report["ready"] and settings.spc.auto_submit:
        # opt-in only (SPC Lab > Submit settings). Same guards as a click: every gate passed, endpoint set, daily limit,
        # never twice. Any refusal leaves the prompt saved and ready, so it can still be submitted by hand.
        from .audit import ACTOR
        token = ACTOR.set("spc auto-submit")
        await ctx.update(0.98, "Every gate passed: submitting (auto-submit is on)")
        try:
            text += " " + await submit_prompt(report["prompt_id"], confirmed=True)
        except Exception as exc:
            text += f" Not submitted: {getattr(exc, 'message', None) or exc}"
            bus.publish("spc", text, "warning", kind="spc_auto_submit", prompt_id=report["prompt_id"])
        finally:
            ACTOR.reset(token)
    return text        # without auto-submit this job never submits: a submission starts with the confirmation click


@jobs.register("spc_submit")
async def _job_submit(ctx: JobContext, params: dict) -> str:
    return await submit_prompt(int(params["prompt_id"]), confirmed=bool(params.get("confirmed")))
