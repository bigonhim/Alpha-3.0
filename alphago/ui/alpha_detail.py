"""The Alpha page (a dialog shared by Alphas, Simulation Matrix, Autopilot, Power Pool ...) and the submit confirmation.

The BRAIN half is ported from alpha-harness ``screens/alpha/**`` and ``api/alphas.py``: the Alpha as BRAIN's own page
shows it — status, type, created date on BRAIN's clock (New York), the test period hidden until asked for, Testing
Status grouped by result in BRAIN's own sentences, the Summary's six figures with BRAIN's tips, the yearly table
coloured by period, the OS table, per-region blocks, matches, the PnL / Sharpe / Turnover chart, regional children,
correlations and the before/after performance comparison. It is read from BRAIN on every open and kept in
``db.kv`` (``alpha:<id>``), so a copy still shows, said plainly, when BRAIN cannot answer. Correlations and the
comparison spend BRAIN's hourly budgets, so a kept answer shows for free and only a click asks BRAIN again.

The AlphaGo half (verdict, robustness and honesty badges, the repair plan, Optimise, Repair, Submit) stays as it was.
"""

from __future__ import annotations

import asyncio
import html
import inspect
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from nicegui import ui

from ..brain import brain
from ..brain.errors import BrainError
from ..brain.platform import POWER_POOL_CHECK, alpha_url
from ..db import db
from ..db.models import Alpha, Candidate
from ..engine import actions
from ..engine.evaluator import evaluate
from ..jobs.queue import jobs
from .check_messages import brain_value, by_result, cell, check_html, is_num, message, parts_html
from .components import badge, fmt_int, fmt_num, fmt_pct, notify_error, verdict_badge

BRAIN_ALPHA_URL = "https://platform.worldquantbrain.com/alpha/{}"
PLATFORM_TZ = ZoneInfo("America/New_York")

def _metric(label: str, value: str, color: str = "") -> None:
    with ui.column().classes("gap-0 min-w-[92px]"):
        ui.label(label).classes("ag-muted text-xs")
        ui.label(value).classes(f"ag-mono text-lg {color}")


def metrics_row(a: Alpha) -> None:
    ev = evaluate(sharpe=a.sharpe, fitness=a.fitness, turnover=a.turnover, checks=a.checks, delay=a.delay,
                  region=a.region, status=a.status)
    sc = "ag-green" if (a.sharpe or 0) >= ev.sharpe_limit else ("ag-amber" if (a.sharpe or 0) > 0 else "ag-red")
    fc = "ag-green" if (a.fitness or 0) >= ev.fitness_limit else "ag-amber"
    with ui.row().classes("gap-6 flex-wrap"):
        _metric("Sharpe", fmt_num(a.sharpe), sc)
        _metric("Fitness", fmt_num(a.fitness), fc)
        _metric("Turnover", fmt_pct(a.turnover))
        _metric("Returns", fmt_pct(a.returns))
        _metric("Drawdown", fmt_pct(a.drawdown))
        _metric("Margin", f"{fmt_num((a.margin or 0) * 10000, 1)} bps" if a.margin is not None else "—")
        _metric("Long / Short", f"{fmt_int(a.long_count)} / {fmt_int(a.short_count)}")
        _metric("Self corr", fmt_num(a.self_corr))
        _metric("Ops / fields", f"{fmt_int(a.op_count)} / {len(a.fields or []) or '—'}")


CHECK_LABEL = {"pass": ("BRAIN CHECK PASSED", "pass"), "fail": ("BRAIN CHECK FAILED", "fail"),
               "pending": ("BRAIN CHECK PENDING", "pending"), "stale": ("RE-CHECK NEEDED", "warning")}


def robust_label(a: Alpha) -> str:
    if a.robust_state == "robust":
        return f"robust {fmt_pct(a.robust_ratio, 0)}"
    if a.robust_state == "fragile":
        return f"fragile {fmt_pct(a.robust_ratio, 0)}"
    return "untested" if a.robust_state == "unknown" else ""


def quality_badges(a: Alpha) -> None:
    """BRAIN check state, robustness, estimated self-correlation, honesty figures and idea cluster as badges."""
    with ui.row().classes("gap-1 flex-wrap items-center"):
        label = CHECK_LABEL.get(a.check_state or "")
        if label:
            badge(*label).tooltip("Stale: you submitted another alpha in this region/delay since this check; "
                                  "self-correlation may have changed." if a.check_state == "stale" else
                                  "Result of BRAIN's full submission check")
        if a.check_state == "pass" and not a.robust_state:
            badge("ROBUSTNESS NOT TESTED").tooltip("The robustness test did not run for this alpha (switched off, or "
                                                   "the task's repair budget was used up).")
        if a.robust_state:
            kind = {"robust": "pass", "fragile": "fail"}.get(a.robust_state, "")
            badge(robust_label(a).upper(), kind).tooltip(
                "Robustness: the worst of three neighbours (decay +2, windows ×0.8, ×1.25) keeps this share of the "
                "Sharpe. Below 80% = fragile. AlphaGo's own test; BRAIN does not require it.")
        if a.est_self_corr is not None:
            kind = "fail" if a.est_self_corr >= 0.7 else ("warning" if a.est_self_corr >= 0.5 else "pass")
            badge(f"EST. SELF-CORR {a.est_self_corr:.2f}", kind).tooltip(
                f"Estimated from daily PnL against your submitted alphas (closest: {a.est_corr_with or '—'}). "
                "An estimate and a lower bound; BRAIN's check decides.")
        if a.wf_ratio is not None:
            kind = "pass" if a.wf_ratio >= 0.7 else ("warning" if a.wf_ratio >= 0.4 else "fail")
            badge(f"RECENT 20% KEEPS {fmt_pct(a.wf_ratio, 0)}", kind).tooltip(
                "Walk-forward: Sharpe of the last 20% of the in-sample period ÷ the first 80%. The search saw these "
                "years, so this measures recent stability, not true out-of-sample.")
        if a.dsr is not None:
            badge(f"SEARCH-ADJUSTED {a.dsr:.2f}", "pass" if a.dsr >= 0.95 else "warning").tooltip(
                f"Deflated-Sharpe confidence after {a.n_trials or '?'} simulations ({a.n_eff or '?'} distinct "
                "lineages) in the search that found it. Below 0.95 the Sharpe may be partly luck. Advisory only.")
        if a.os_retention_pred is not None:
            badge(f"PREDICTED OS/IS {a.os_retention_pred:.2f}").tooltip(
                "From your own submitted alphas: the out-of-sample share of Sharpe alphas like this kept.")
        if a.cluster:
            badge(a.cluster.upper()).tooltip("Economic idea (used for the daily cap per idea)")


def settings_line(settings: dict[str, Any] | None) -> str:
    s = settings or {}
    keys = ["region", "universe", "delay", "neutralization", "decay", "truncation", "pasteurization", "nanHandling",
            "unitHandling", "maxTrade", "testPeriod"]
    return " · ".join(f"{k}={s[k]}" for k in keys if k in s)


async def repair_block(a: Alpha) -> None:
    """What the repair engine sees in this alpha: why each check fails, the moves it would run next, what was tried."""
    from ..engine.refine import repair_plan, tried_lines
    if (a.type or "REGULAR") != "REGULAR" or not a.code or a.verdict in ("ready", "submitted"):
        return
    plan = repair_plan(a)
    tried = await tried_lines(a.id, 6)
    if not (plan["lines"] or plan["moves"] or tried):
        return
    with ui.expansion("Repair plan", icon="healing", value=a.verdict in ("promising", "weak")).classes("w-full"):
        if plan["state"] == "exhausted":
            ui.label(plan["note"] or "Every move that could help has been tried.").classes("ag-amber text-sm")
        if plan["out_of_reach"]:
            ui.label(f"Skipped by automatic repair. {plan['out_of_reach']} Select it yourself to try anyway.").classes(
                "ag-amber text-sm")
        for line in plan["lines"]:
            ui.label(line).classes("text-sm whitespace-pre-wrap")
        if plan["correlation_only"]:
            ui.label("Passes every metric: a repair task takes the decorrelation route.").classes("ag-cyan text-sm")
        if plan["moves"]:
            ui.label("Next moves").classes("ag-muted text-xs pt-1")
            for move in plan["moves"]:
                ui.label(f"• {move}").classes("text-sm ag-mono")
        if tried:
            ui.label("Tried so far").classes("ag-muted text-xs pt-1")
            for line in tried:
                ui.label(line).classes("text-xs ag-mono")


async def load_alpha(alpha_id: str) -> Alpha | None:
    async with db.session() as s:
        return await s.get(Alpha, alpha_id)


async def confirm_submit(alpha_id: str, on_done: Callable[[], Any] | None = None) -> None:
    """Explicit confirmation dialog — the only path that submits an alpha."""
    a = await load_alpha(alpha_id)
    if a is None:
        ui.notify("Alpha not found locally.", type="warning")
        return
    kind = a.type or "REGULAR"
    used = await actions.submissions_today(kind)
    from ..config import settings
    limit = settings.programs.daily_super_submissions if kind == "SUPER" else settings.programs.daily_regular_submissions
    with ui.dialog() as dialog, ui.card().classes("ag-card p-6 w-[720px] max-w-[90vw]"):
        ui.label(f"Submit {alpha_id} to WorldQuant BRAIN?").classes("text-lg font-semibold")
        ui.label(a.code or a.selection or "").classes("ag-mono text-sm whitespace-pre-wrap break-all")
        ui.label(settings_line(a.settings)).classes("ag-muted text-xs")
        metrics_row(a)
        quality_badges(a)
        if a.check_state == "stale":
            ui.label("⚠ You submitted another alpha in this region/delay since this alpha was checked, so its "
                     "self-correlation may have changed. Run 'Check' again before submitting.").classes(
                "ag-amber text-sm")
        elif a.check_state != "pass":
            ui.label("⚠ BRAIN's full submission check has not passed yet for this alpha. Run 'Check' first — BRAIN will "
                     "reject the submission if any test fails.").classes("ag-amber text-sm")
        if a.robust_state == "fragile":
            ui.label(f"⚠ Fragile: neighbouring settings keep only {fmt_pct(a.robust_ratio, 0)} of its Sharpe (AlphaGo's "
                     "own test). BRAIN accepts it, but fragile alphas tend to hold up worse out of sample.").classes(
                "ag-amber text-sm")
        with ui.row().classes("gap-1 flex-wrap"):
            for c in a.checks or []:
                check_chip(c)
        ui.label(f"Today's {kind.lower()} submissions: {used}/{limit}. Submission is final and visible to "
                 "WorldQuant.").classes("ag-muted text-sm")

        async def go() -> None:
            dialog.close()
            fresh = await load_alpha(alpha_id)
            if fresh is not None and fresh.check_state == "stale":
                ui.notify(f"Re-running BRAIN's check on {alpha_id} first (its last pass is stale)…", type="info")
                try:
                    state, _ = await actions.run_check(alpha_id)
                except BrainError as exc:
                    notify_error(exc, "Check failed, not submitted: ")
                    return
                if state != "pass":
                    ui.notify(f"{alpha_id} no longer passes BRAIN's check ({state}) — not submitted.", type="warning")
                    return
            await jobs.submit("submit_alpha", f"Submit {alpha_id}", {"alpha_id": alpha_id, "confirmed": True},
                              priority=1)
            ui.notify(f"Submitting {alpha_id} — follow progress in Tasks.", type="info")
            if on_done:
                on_done()

        with ui.row().classes("justify-end w-full gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Submit to BRAIN", icon="send", on_click=go).props(
                f"color=positive {'disable' if used >= limit else ''}")
    dialog.open()


def check_chip(check: dict[str, Any], stage: str = "IS") -> None:
    """A compact check badge whose tooltip is BRAIN's own sentence for it."""
    from .check_messages import check_message
    result = str(check.get("result") or "PENDING").upper()
    kind = {"PASS": "pass", "FAIL": "fail", "PENDING": "pending", "WARNING": "warning", "ERROR": "fail"}.get(result, "")
    badge(f"{check.get('name')} · {result}", kind).tooltip(check_message(check, stage, markdown=False))


# ── BRAIN's page, read the way BRAIN reads it ────────────────────────────────────────────────

#: BRAIN's short names for an Alpha's status.
STATUS_LABEL = {"UNSUBMITTED": "UNSUB", "SUBMITTED": "SUB", "ACTIVE": "ACTV", "DECOMMISSIONED": "DECM"}

#: The six figures every summary row shows, in BRAIN's order, with BRAIN's explanation of each.
METRICS: list[tuple[str, str, str, str]] = [
    ("sharpe", "Sharpe", "decimal",
     "Average measure of risk-adjusted returns; Sharpe = Avg. Annualized Returns /  Annualized Std. Dev. of Returns"),
    ("turnover", "Turnover", "percent",
     "Average measure of daily trading activity; Turnover = Value Traded / Value Held"),
    ("fitness", "Fitness", "decimal",
     "Hybrid metric that indicates overall performance. Higher values indicate better performance. Fitness = Sharpe "
     "* Sqrt( Abs( Returns ) / Max( Turnover, 0.125 ) )"),
    ("returns", "Returns", "percent",
     "Annualized average gain or loss as a fraction of the invested amount. In WorldQuant BRAIN the invested amount "
     "is equal to half the book size"),
    ("drawdown", "Drawdown", "percent",
     "Drawdown is the largest reduction in PnL during a given period, expressed as a percentage. Good "
     "Returns/Drawdown ratio >=1"),
    ("margin", "Margin", "permyriad",
     "Average gain or loss per dollar traded; calculated as PnL divided by total dollars traded in a given time "
     "period. Suggested: > 4 basis points"),
]
CHILD_METRICS = [(k, "Max Drawdown" if k == "drawdown" else t, kind, tip) for k, t, kind, tip in METRICS]

#: The yearly table's columns; ``stage`` only colours the row.
YEARLY: list[tuple[str, str, str]] = [
    ("year", "Year", "year"), ("sharpe", "Sharpe", "decimal"), ("turnover", "Turnover", "percent"),
    ("fitness", "Fitness", "decimal"), ("returns", "Returns", "percent"), ("drawdown", "Drawdown", "percent"),
    ("margin", "Margin", "permyriad"), ("longCount", "Long Count", "integer"), ("shortCount", "Short Count", "integer"),
]
#: The one-row table the OS period adds above the yearly one.
OS_TABLE: list[tuple[str, str, str]] = [
    ("sharpe60", "Sharpe 60", "decimal"), ("sharpe125", "Sharpe 125", "decimal"),
    ("sharpe250", "Sharpe 250", "decimal"), ("sharpe500", "Sharpe 500", "decimal"),
    ("osISSharpeRatio", "OS/IS Ratio", "decimal"), ("preCloseSharpe", "Pre-Close Sharpe", "decimal"),
    ("preCloseSharpeRatio", "Pre-Close Ratio", "decimal"), ("selfCorrelation", "Self-Correlation", "decimal"),
]
#: BRAIN's Simulation Settings, in its order.
SETTINGS: list[tuple[str, str]] = [
    ("instrumentType", "Instrument Type"), ("region", "Region"), ("universe", "Universe"), ("language", "Language"),
    ("decay", "Decay"), ("delay", "Delay"), ("truncation", "Truncation"), ("neutralization", "Neutralization"),
    ("pasteurization", "Pasteurization"), ("nanHandling", "NaN Handling"), ("unitHandling", "Unit Handling"),
    ("maxTrade", "Max Trade"), ("maxPosition", "Max Position"),
]
PERIODS = ("train", "test", "is", "os", "prod")
#: BRAIN's Performance series, the choices of its Chart.
SERIES = {"pnl": "PnL", "sharpe": "Sharpe", "turnover": "Turnover"}
#: BRAIN adds a submitted Alpha's out-of-sample days as they pass, so its series age.
GROWING_SERIES_AGE = timedelta(hours=12)
#: The Alpha page waits this long for BRAIN before showing the kept copy instead.
PAGE_TIMEOUT_S = 25.0

POWER_POOL_DESCRIPTION_MIN_LENGTH = 100
POWER_POOL_DESCRIPTION_HEADINGS = ("Idea", "Rationale for data used", "Rationale for operators used")
#: BRAIN's template, exactly as its Use the template button writes it.
POWER_POOL_TEMPLATE = "\n".join(f"{h}: " for h in POWER_POOL_DESCRIPTION_HEADINGS)
_TEMPLATE = re.compile(r"^\s*idea:(\s+(.+?)\n)\s*rationale for data used:\s+((.+?)\n)\s*rationale for operators "
                       r"used:(\s+(.+?))", re.IGNORECASE | re.DOTALL)

#: Each correlation: (kind, title, the words when nothing correlates, whom it compares with).
CORRELATIONS = [
    ("self", "Self Correlation", "There are no self-correlated alphas.",
     "all other alphas submitted by the same user"),
    ("power-pool", "Power Pool Correlation", "There are no self-correlated alphas.",
     "all other power pool alphas submitted by the same user"),
    ("prod", "Prod Correlation", "There are no prod-correlated alphas.",
     "all other alphas submitted by all WorldQuant BRAIN users"),
]
#: The columns BRAIN shows of each correlated Alpha, in the order the answer lists them.
CORRELATED_SHOWN = ("name", "universe", "correlation", "sharpe", "turnover", "fitness", "returns", "drawdown", "margin")

#: BRAIN's competition scores in the before/after comparison, with its explanation of before, after and change.
SCORES = [
    ("score", "Score", ("Your team’s current delay score.", "Expected team’s delay score after submitting this alpha.",
                        "Change in your team’s delay score if this alpha is submitted.")),
    ("datafieldsScore", "Data Fields Score", ("Your current data fields score.",
                                              "Expected data fields score after submitting this alpha.",
                                              "Change in your data fields score if this alpha is submitted.")),
    ("weeklyScore", "Weekly Score", ("Your current weekly score.", "Expected weekly score after submitting this alpha.",
                                     "Change in your weekly score if this alpha is submitted.")),
    ("robustnessScore", "Robustness Score", ("Your current robustness score.",
                                             "Expected robustness score after submitting this alpha.",
                                             "Change in your robustness score if this alpha is submitted.")),
]
PARTITION = {"REGULAR": "Regular alpha", "SUPER": "Super alpha", "EQUITY": "Equity", "0": "Delay-0", "1": "Delay-1"}


def _parse(iso: Any) -> datetime | None:
    if isinstance(iso, datetime):
        return iso if iso.tzinfo else iso.replace(tzinfo=timezone.utc)
    if not isinstance(iso, str) or not iso.strip():
        return None
    text = iso.strip()
    if len(text) == 10:                    # a bare day is that day in New York, not the UTC midnight before it
        text = f"{text}T12:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def platform_day(iso: Any) -> str:
    """``09/30/2026 EDT``: a day on BRAIN's own clock, as its Created and Start Date read."""
    when = _parse(iso)
    if when is None:
        return ""
    local = when.astimezone(PLATFORM_TZ)
    return f"{local:%m/%d/%Y} {local.tzname()}"


def stamp(iso: Any) -> str:
    """``Wed, 09/30/2026, 10:03 AM`` on this computer's clock: Last Run."""
    when = _parse(iso)
    if when is None:
        return ""
    local = when.astimezone()
    return f"{local:%a, %m/%d/%Y}, {local.hour % 12 or 12}:{local:%M %p}"


def words(value: str) -> str:
    return value[:1] + value[1:].lower().replace("_", " ")


def test_control(info: dict[str, Any]) -> bool:
    """An unsubmitted Alpha with a test period keeps it hidden until asked, as BRAIN does."""
    return info.get("status") == "UNSUBMITTED" and bool((info["periods"].get("test") or {}).get("startDate"))


def period_tabs(periods: dict[str, Any], control: bool, shown: bool) -> list[tuple[str, str, bool]]:
    """The Period tabs (period, label, available): TRAIN alone until the test period shows, then TRAIN, TEST, IS;
    any other Alpha IS and OS."""
    if control:
        return [("train", "TRAIN", True), ("test", "TEST", shown), ("is", "IS", shown), ("os", "OS", False)]
    return [("is", "IS", periods.get("is") is not None), ("os", "OS", periods.get("os") is not None)]


def first_period(stage: str | None, control: bool, shown: bool) -> str:
    """TEST once the test period shows, TRAIN while it hides, else the Alpha's own stage (production counts as OS)."""
    if control:
        return "test" if shown else "train"
    return "os" if stage in ("OS", "PROD") else "is"


def stage_of(period: str) -> str:
    """TRAIN and TEST are parts of IS: their checks and title are IS's."""
    return "is" if period in ("train", "test") else period


def regions_of(block: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """BRAIN's regions inside a region-agnostic (``subregions``) or global (``glbAmer`` ...) Alpha's block."""
    sub = block.get("subregions")
    if isinstance(sub, dict):
        return [(str(k), v) for k, v in sub.items() if isinstance(v, dict)]
    out = []
    for r in ("Amer", "Apac", "Emea"):
        found = block.get(f"glb{r}")
        if isinstance(found, dict):
            out.append((r.upper(), found))
    return out


def rows_of(recordset: dict[str, Any] | None) -> list[dict[str, Any]]:
    """A record set's rows as dicts keyed by its column names."""
    names = [p.get("name") for p in ((recordset or {}).get("schema") or {}).get("properties") or []]
    return [dict(zip(names, r)) for r in (recordset or {}).get("records") or [] if isinstance(r, list)]


def described_for_power_pool(info: dict[str, Any]) -> bool:
    """While its Power Pool check is open, BRAIN holds the Alpha's description to its template."""
    return any(c.get("name") == POWER_POOL_CHECK and c.get("result") in ("PASS", "PENDING")
               for c in (info["periods"].get("is") or {}).get("checks") or [])


def is_power_pool(info: dict[str, Any]) -> bool:
    """Power Pool Correlation applies: the check above is open, or BRAIN classes the Alpha eligible."""
    return described_for_power_pool(info) or any(
        c.get("id") == "POWER_POOL:POWER_POOL_ELIGIBLE" for c in info.get("classifications") or [])


def _template_gap(text: str) -> str | None:
    """The first heading of the template missing, out of order or with nothing after it."""
    lower = text.lower()
    starts: list[int] = []
    for heading in POWER_POOL_DESCRIPTION_HEADINGS:
        at = lower.find(f"{heading.lower()}:", starts[-1] if starts else 0)
        if at == -1:
            return heading
        starts.append(at)
    for i, heading in enumerate(POWER_POOL_DESCRIPTION_HEADINGS):
        end = starts[i + 1] if i + 1 < len(starts) else None
        if not text[starts[i] + len(heading) + 1:end].strip():
            return heading
    return None


def description_problems(text: str) -> list[str]:
    """What BRAIN says is wrong with a Power Pool description, in its words, and which heading."""
    text = text or ""
    found = _TEMPLATE.search(text)
    gap = _template_gap(text)
    problems = []
    if len(text.strip()) < POWER_POOL_DESCRIPTION_MIN_LENGTH:
        problems.append(f"Ensure this field has at least {POWER_POOL_DESCRIPTION_MIN_LENGTH} characters.")
    if not (found and all((part or "").strip() for part in (found.group(1), found.group(3), found.group(5)))):
        problems.append("Ensure the description is following the provided template."
                        + (f' "{gap}:" is missing, out of order or empty.' if gap else ""))
    return problems


def info_of(alpha_id: str, body: dict[str, Any], is_checks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """BRAIN's ``GET /alphas/{id}`` body in the page's shape. ``is_checks`` replaces the IS checks when a Check
    Submission's answer is kept (BRAIN's own page shows that one after a check)."""
    regular, combo, selection = (body.get(k) for k in ("regular", "combo", "selection"))
    code = regular.get("code") if isinstance(regular, dict) else regular     # an older Alpha sends a bare string
    described = next((p for p in (regular, combo, selection) if isinstance(p, dict)), {})
    periods = {p: dict(block) for p in PERIODS if isinstance(block := body.get(p), dict)}
    if "is" in periods and is_checks:
        periods["is"]["checks"] = is_checks
    return {
        "alphaId": alpha_id, "type": body.get("type"), "author": body.get("author"),
        "code": code if isinstance(code, str) else None,
        "selection": selection.get("code") if isinstance(selection, dict) else None,
        "combo": combo.get("code") if isinstance(combo, dict) else None,
        "description": described.get("description"), "name": body.get("name"), "category": body.get("category"),
        "color": body.get("color"), "tags": [t for t in body.get("tags") or [] if isinstance(t, str)],
        "stage": body.get("stage"), "status": body.get("status"), "grade": body.get("grade"),
        "dateCreated": body.get("dateCreated"), "dateModified": body.get("dateModified"),
        "settings": body.get("settings") or {},
        "classifications": [c for c in body.get("classifications") or [] if isinstance(c, dict) and c.get("id")],
        "themes": [t for t in body.get("themes") or [] if isinstance(t, dict)],
        "pyramidThemes": body.get("pyramidThemes") or None,
        "periods": periods, "children": [str(c) for c in body.get("children") or []],
        "brainUrl": alpha_url(alpha_id), "body": body,
    }


# ── kept copies of BRAIN's answers (db.kv) ─────────────────────────────────────────────────────


async def kept(key: str) -> dict[str, Any] | None:
    """A kept answer: ``{"body": ..., "fetchedAt": iso}``, or None."""
    row = await db.kv_get(key)
    return row if isinstance(row, dict) and "body" in row else None


async def keep(key: str, body: Any) -> str:
    now = datetime.now(timezone.utc).isoformat()
    await db.kv_set(key, {"body": body, "fetchedAt": now})
    return now


async def forget(*keys: str) -> None:
    for key in keys:
        await db.kv_set(key, None)


def copy_note(row: dict[str, Any]) -> str:
    when = _parse(row.get("fetchedAt"))
    return f"showing the copy from {when.astimezone(timezone.utc):%b %d, %H:%M} UTC." if when else \
        "showing the last copy kept."


def unanswered(exc: BaseException, row: dict[str, Any]) -> str:
    reason = getattr(exc, "message", None) or str(exc) or exc.__class__.__name__
    return f"BRAIN did not answer ({reason}); {copy_note(row)}"


async def _is_checks(alpha_id: str, simulated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The last Check Submission's answer when one is kept; before that, the Alpha's own checks with AlphaGo's
    resolved copy of each (from its last /check) in their place."""
    row = await kept(f"check:{alpha_id}")
    if row is not None:
        return [c for c in ((row["body"] or {}).get("is") or {}).get("checks") or [] if isinstance(c, dict)]
    local = await load_alpha(alpha_id)
    resolved = [c for c in (local.checks if local and local.checked_at else None) or [] if isinstance(c, dict)]
    merged: list[dict[str, Any]] = []
    for check in simulated:
        found = next((r for r in resolved if r.get("name") == check.get("name")), check)
        if found is not check:
            resolved.remove(found)
        merged.append(found)
    return merged + resolved


async def load_page(alpha_id: str) -> tuple[dict[str, Any] | None, list[str]]:
    """The Alpha as BRAIN's page shows it, read from BRAIN on every open, falling back to the kept copy (then the
    copy AlphaGo stored when it synced the Alpha) when BRAIN cannot answer. Returns (info or None, problems)."""
    problems: list[str] = []
    key = f"alpha:{alpha_id}"
    stale = await kept(key)
    try:
        body = await asyncio.wait_for(brain.alphas.get(alpha_id), PAGE_TIMEOUT_S)
        if not isinstance(body, dict) or not body:
            raise BrainError(f"BRAIN sent no details for {alpha_id}")
        await keep(key, body)
    except Exception as exc:  # noqa: BLE001 - any failure falls back to a copy, said plainly
        if stale is not None and isinstance(stale.get("body"), dict):
            body = stale["body"]
            problems.append(unanswered(exc, stale))
        else:
            local = await load_alpha(alpha_id)
            if local is None or not isinstance(local.raw, dict) or not local.raw:
                reason = getattr(exc, "message", None) or str(exc) or exc.__class__.__name__
                problems.append(f"Could not load {alpha_id} from BRAIN: {reason}")
                return None, problems
            body = local.raw
            reason = getattr(exc, "message", None) or str(exc) or exc.__class__.__name__
            problems.append(f"BRAIN did not answer ({reason}); showing the copy AlphaGo stored when it last synced "
                            "this alpha.")
    simulated = [c for c in ((body.get("is") or {}).get("checks") or []) if isinstance(c, dict)]
    checks = await _is_checks(alpha_id, simulated) if "is" in body else None
    return info_of(alpha_id, body, checks), problems


async def recordset(alpha_id: str, name: str, stage: str | None) -> dict[str, Any]:
    """One of the Alpha's series as BRAIN sends it. An in-sample series never changes, so it is read once; a
    submitted Alpha's grows a day at a time, so its copy is read again once it is 12 hours old. A copy that could not
    be refreshed comes back with ``problem`` saying so."""
    key = f"recordset:{name}:{alpha_id}"
    row = await kept(key)
    growing = stage not in (None, "IS")
    if row is not None:
        when = _parse(row.get("fetchedAt"))
        fresh = when is not None and datetime.now(timezone.utc) - when <= GROWING_SERIES_AGE
        if not growing or fresh:
            return dict(row["body"] or {})
    try:
        response = await brain.client.poll(f"/alphas/{alpha_id}/recordsets/{name}", max_wait_s=300)
        body = brain.client.raise_for(response) or {}
    except Exception as exc:  # noqa: BLE001
        if row is None:
            raise
        return dict(row["body"] or {}) | {"problem": unanswered(exc, row)}
    body = body if isinstance(body, dict) else {}
    await keep(key, body)
    return body


async def budgeted(key: str, fetch: Callable[[], Any], mode: str) -> dict[str, Any] | None:
    """A BRAIN job's kept answer stamped with ``fetchedAt``. ``cached`` reads only what is kept (None when nothing
    is), ``run`` asks BRAIN only when nothing is kept, ``refresh`` always asks."""
    row = await kept(key)
    if mode == "cached" or (mode == "run" and row is not None):
        return None if row is None else dict(row["body"] or {}) | {"fetchedAt": row["fetchedAt"]}
    body = await fetch()
    fetched = await keep(key, body)
    return dict(body or {}) | {"fetchedAt": fetched}


async def correlation(alpha_id: str, kind: str, mode: str) -> dict[str, Any] | None:
    """Self, Power Pool or production correlation, kept in ``correlation:<kind>:<id>`` until refreshed."""
    async def fetch() -> dict[str, Any]:
        response = await brain.client.poll(f"/alphas/{alpha_id}/correlations/{kind}", max_wait_s=600)
        body = brain.client.raise_for(response)
        return body if isinstance(body, dict) else {}
    return await budgeted(f"correlation:{kind}:{alpha_id}", fetch, mode)


async def performance(alpha_id: str, mode: str) -> dict[str, Any] | None:
    """Your pool's stats before and after this Alpha joins it, kept in ``performance:<id>`` until refreshed."""
    async def fetch() -> dict[str, Any]:
        body = await brain.alphas.before_after(alpha_id)
        return body if isinstance(body, dict) else {}
    return await budgeted(f"performance:{alpha_id}", fetch, mode)


async def recheck(alpha_id: str, alpha_type: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Re-run BRAIN's submission checks (never submits). The answer is kept for the page; a region-agnostic parent
    answers for its children too, each under its own id, so every child carries BRAIN's final word."""
    state, checks = await actions.run_check(alpha_id)
    if checks:
        await keep(f"check:{alpha_id}", {"is": {"checks": checks}})
    if alpha_type == "RA_PARENT":
        try:
            response = await brain.client.poll(f"/alphas/{alpha_id}/check", max_wait_s=300)
            body = brain.client.raise_for(response) or {}
        except BrainError:
            body = {}
        subregions = ((body.get("is") or {}).get("subregions") or {}) if isinstance(body, dict) else {}
        for child_id, child_checks in subregions.items():
            if not isinstance(child_checks, list) or not child_checks:
                continue
            from ..engine.evaluator import check_state_of, reverdict
            async with db.write() as s:
                row = await s.get(Alpha, str(child_id))
                if row is not None:
                    row.checks = child_checks
                    row.check_state = check_state_of(child_checks)
                    row.checked_at = datetime.now(timezone.utc)
                    reverdict(row)
            await keep(f"check:{child_id}", {"is": {"checks": child_checks}})
            await forget(f"alpha:{child_id}")
    await forget(f"alpha:{alpha_id}")
    return state, checks


# ── page parts ───────────────────────────────────────────────────────────────────────────────


def _tip(title: str, text: str) -> None:
    with ui.icon("info_outline").classes("text-xs ag-muted cursor-help"):
        ui.tooltip(f"{title}: {text}").classes("max-w-[320px]")


def _figures(label: str, block: dict[str, Any] | None, note: str = "", metrics=METRICS, blank: str = "") -> None:
    """BRAIN's six figures in a row, under the label it gives the row."""
    with ui.row().classes("w-full items-end gap-6 flex-wrap"):
        with ui.column().classes("gap-0 min-w-[110px]"):
            ui.label(label).classes("text-sm font-medium")
            if note:
                ui.label(note).classes("ag-muted text-xs ag-mono")
        with ui.element("div").classes("grid grid-cols-3 md:grid-cols-6 gap-3 flex-1"):
            for key, title, kind, tip in metrics:
                value = (block or {}).get(key)
                with ui.column().classes("gap-0 min-w-0"):
                    with ui.row().classes("items-center gap-1 no-wrap"):
                        ui.label(title).classes("ag-figure-label")
                        _tip(title, tip)
                    ui.label(brain_value(value, kind) if is_num(value) else blank).classes("ag-figure")


def _aggregate(period: str, block: dict[str, Any]) -> None:
    """A block's own figures: OS dates its row, every other period calls it Aggregate Data."""
    if period == "os":
        _figures("Start Date", block, platform_day(block.get("startDate")))
    else:
        _figures("Aggregate Data", block)


def table_html(columns: list[tuple[str, str, str]], rows: list[dict[str, Any]]) -> str:
    """A table of BRAIN's figures; a yearly row's first cell carries its period's edge colour."""
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title, _ in columns)
    body = []
    for row in rows:
        cells = []
        for j, (key, _, kind) in enumerate(columns):
            stage = f' class="stage-{html.escape(str(row.get("stage")))}"' if j == 0 and row.get("stage") else ""
            cells.append(f"<td{stage}>{html.escape(cell(row.get(key), kind))}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="ag-table-wrap"><table class="ag-table"><thead><tr>{head}</tr></thead>' \
           f'<tbody>{"".join(body)}</tbody></table></div>'


def _period_tabs(tabs: list[tuple[str, str, bool]], value: str, on_change: Callable[[str], Any]) -> None:
    with ui.row().classes("items-center gap-1 no-wrap"):
        ui.label("Period").classes("ag-muted text-xs pr-1")
        for period, label, available in tabs:
            props = "dense no-caps size=sm " + ("unelevated color=primary" if period == value else "flat")
            if not available:
                props += " disable"
            ui.button(label, on_click=lambda p=period: on_change(p)).props(props)


def _check_list(checks: list[dict[str, Any]], stage: str) -> None:
    with ui.column().classes("gap-1 w-full"):
        for check in checks:
            with ui.row().classes("gap-2 no-wrap items-start w-full"):
                ui.element("span").classes(f"ag-dot {html.escape(str(check.get('result') or 'PENDING'))}")
                ui.html(check_html(check, stage)).classes("text-sm min-w-0 break-words")


RESULT_CLASS = {"PASS": "ag-green", "FAIL": "ag-red", "ERROR": "ag-red", "WARNING": "ag-amber", "PENDING": "ag-muted"}


def testing_status(info: dict[str, Any], control: bool, shown: bool) -> None:
    """BRAIN's checks grouped by result (FAIL open: these are why it cannot be submitted), in BRAIN's sentences."""
    state = {"period": first_period(info.get("stage"), control, shown)}

    @ui.refreshable
    def body() -> None:
        stage = stage_of(state["period"])
        checks = (info["periods"].get(stage) or {}).get("checks")
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(f"{stage.upper()} Testing Status").classes("ag-panel-title")
            _period_tabs(period_tabs(info["periods"], control, shown), state["period"],
                         lambda p: (state.update(period=p), body.refresh()))
        quick = checks is not None and info.get("status") == "UNSUBMITTED" and \
            info["settings"].get("simulationMode") == "QUICK"
        if quick:
            ui.label("This alpha did not pass all quick checks. Fix the failures below, then run Full Simulate."
                     if any(c.get("result") == "FAIL" for c in checks) else
                     "Quick Simulate is not enough — Full Simulate is needed to submit this alpha.").classes(
                "ag-warn-box w-full text-sm")
        if not checks:
            ui.label("No checks reported for this period.").classes("ag-muted text-sm")
        for result, group in by_result(checks or []):
            with ui.expansion(value=result == "FAIL").classes("w-full").props("dense") as exp:
                with exp.add_slot("header"):
                    ui.label(f"{len(group)} {result}").classes(f"font-medium {RESULT_CLASS.get(result, '')}")
                _check_list(group, "IS" if stage == "is" else "OS")

    with ui.card().classes("ag-card w-full p-4 gap-3"):
        body()


def _matches(info: dict[str, Any]) -> None:
    """The grade, themes, classifications and pyramids BRAIN tags the Alpha with."""
    pyramids = (info.get("pyramidThemes") or {}).get("pyramids")
    effective = (info.get("pyramidThemes") or {}).get("effective")
    tagged: list[tuple[str, str]] = []
    for t in info.get("themes") or []:
        tagged.append((f"Theme: {t.get('name')} x{t.get('multiplier')}", parts_html(message(
            "IS.MATCHES_THEMES.PASS.single", {"name": t.get("name"), "multiplier": str(t.get("multiplier"))}))))
    for c in info.get("classifications") or []:
        tagged.append((str(c.get("name") or c.get("id")), parts_html(message(
            "IS.MATCHES_CLASSIFICATION.PASS.single", {"value": c.get("name")}))))
    for p in pyramids if isinstance(pyramids, list) else []:
        if isinstance(p, dict):
            tagged.append((f"Pyramid theme: {p.get('name')} x{p.get('multiplier')}", parts_html(message(
                "IS.MATCHES_PYRAMID.PASS.single", {"name": p.get("name"), "multiplier": str(p.get("multiplier")),
                                                   "effective": effective}))))
    if not tagged and not info.get("grade"):
        return
    with ui.row().classes("gap-2 flex-wrap"):
        if info.get("grade"):
            badge(words(str(info["grade"])))
        for text, about in tagged:
            with ui.label(text).classes("ag-badge"):
                with ui.tooltip().classes("max-w-[360px]"):
                    ui.html(about)


async def summary(alpha_id: str, info: dict[str, Any], control: bool, shown: bool) -> None:
    """The period's six figures with BRAIN's tips, the yearly table, the OS table and each region's figures."""
    state = {"period": first_period(info.get("stage"), control, shown), "years": True}
    yearly_set: dict[str, Any] = {}

    async def load_yearly() -> None:
        try:
            yearly_set["data"] = await recordset(alpha_id, "yearly-stats", info.get("stage"))
        except Exception as exc:  # noqa: BLE001
            yearly_set["error"] = getattr(exc, "message", None) or str(exc)

    @ui.refreshable
    def body() -> None:
        period = state["period"]
        block = info["periods"].get(period)
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(f"{stage_of(period).upper()} Summary").classes("ag-panel-title")
            _period_tabs(period_tabs(info["periods"], control, shown), period,
                         lambda p: (state.update(period=p), body.refresh()))
        _matches(info)
        if block is None:
            ui.label("Summary not available, please clone this alpha and re-simulate.").classes("ag-muted text-sm")
            return
        yearly = "subregions" not in block       # a region-agnostic block has only its regions' figures
        with ui.row().classes("w-full items-end no-wrap"):
            with ui.column().classes("flex-1 min-w-0"):
                _aggregate(period, block)
            if yearly:
                ui.button(icon="expand_less" if state["years"] else "expand_more",
                          on_click=lambda: (state.update(years=not state["years"]), body.refresh())).props(
                    "flat round dense").tooltip("Hide the yearly figures" if state["years"] else
                                                "Show the yearly figures")
        if yearly and state["years"]:
            if "error" in yearly_set:
                ui.label(f"The yearly figures did not load: {yearly_set['error']}").classes("ag-bad-box text-sm")
            elif "data" in yearly_set:
                data = yearly_set["data"]
                if data.get("problem"):
                    ui.label(data["problem"]).classes("ag-warn-box text-sm w-full")
                if period == "os":
                    ui.html(table_html(OS_TABLE, [block])).classes("w-full")
                train_only = control and not shown and period == "train"
                rows = [r for r in rows_of(data) if not train_only or r.get("stage") == "TRAIN"]
                if rows:
                    ui.html(table_html(YEARLY, rows)).classes("w-full")
                else:
                    ui.label("Yearly summary not available, please clone this alpha and re-simulate.").classes(
                        "ag-muted text-sm")
            else:
                ui.spinner(size="sm")
        for region, figures in regions_of(block):
            with ui.column().classes("w-full gap-2 pt-3").style("border-top:1px solid var(--ag-border)"):
                ui.label(region).classes("text-sm font-medium")
                _aggregate(period, figures)
        for key, title in (("riskNeutralized", "Risk neutralized"),
                           ("investabilityConstrained", "Investability constrained")):
            if isinstance(block.get(key), dict):
                with ui.column().classes("w-full gap-2 pt-3").style("border-top:1px solid var(--ag-border)"):
                    ui.label(title).classes("text-sm font-medium")
                    _aggregate(period, block[key])

    with ui.card().classes("ag-card w-full p-4 gap-3"):
        body()
    await load_yearly()
    body.refresh()


# ── chart ────────────────────────────────────────────────────────────────────────────────────

PERIOD_COLOURS = {"train": "#7c6cf0", "test": "#d9a21b", "os": "#4ade80"}
SIDE_LINES = {"risk-neutralized-pnl": ("#8b98ad", "dashed"), "investability-constrained-pnl": ("#5b6880", "dashed")}
REGION_COLOURS = ["#7c6cf0", "#d9a21b", "#4ade80", "#7dd3fc", "#f472b6"]


def chart_options(data: dict[str, Any], name: str, test_start: str | None, os_start: str | None,
                  test_shown: bool) -> dict[str, Any] | None:
    """BRAIN's chart as ECharts options: the main line coloured by period (train, test, out of sample), the
    risk-neutralized and investability-constrained lines dashed beside it, one line per region for a
    region-agnostic Alpha, and the series cut at the test start while the test period is hidden."""
    props = ((data.get("schema") or {}).get("properties")) or []
    names = [p.get("name") for p in props]
    if "date" not in names:
        return None
    at = names.index("date")
    cut = None if test_shown else test_start
    by_day: dict[str, list[Any]] = {}
    for r in data.get("records") or []:
        if isinstance(r, list) and len(r) > at and (cut is None or str(r[at]) < cut):
            by_day[str(r[at])] = r
    days = sorted(by_day)
    if not days:
        return None
    series: list[dict[str, Any]] = []
    region = 0

    def zone(day: str) -> str:
        if os_start and day >= os_start:
            return "os"
        if test_start and day >= test_start:
            return "test"
        return "train"

    for i, p in enumerate(props):
        key = p.get("name")
        if key == "date":
            continue
        title = p.get("title") or key
        scale = 100 if p.get("type") == "percent" else 1
        values = [by_day[d][i] * scale if i < len(by_day[d]) and is_num(by_day[d][i]) else None for d in days]
        if key in SIDE_LINES:
            colour, style = SIDE_LINES[key]
            series.append({"name": title, "type": "line", "showSymbol": False, "data": list(zip(days, values)),
                           "lineStyle": {"width": 1, "type": style, "color": colour}, "itemStyle": {"color": colour}})
        elif key == name:
            labels = {"train": "Train" if (test_start or os_start) else "IS", "test": "Test", "os": "OS"}
            for z in ("train", "test", "os"):
                # each period its own run, sharing the boundary day so the line stays unbroken
                points = []
                for j, d in enumerate(days):
                    inside = zone(d) == z or (j + 1 < len(days) and zone(days[j + 1]) == z and zone(d) != z)
                    points.append([d, values[j] if inside else None])
                if any(v is not None for _, v in points):
                    series.append({"name": labels[z], "type": "line", "showSymbol": False, "data": points,
                                   "connectNulls": False,
                                   "lineStyle": {"width": 1.5, "color": PERIOD_COLOURS[z]},
                                   "itemStyle": {"color": PERIOD_COLOURS[z]}})
        else:
            colour = REGION_COLOURS[region % len(REGION_COLOURS)]
            region += 1
            series.append({"name": title, "type": "line", "showSymbol": False, "data": list(zip(days, values)),
                           "lineStyle": {"width": 1.2, "color": colour}, "itemStyle": {"color": colour}})
    return {
        "backgroundColor": "transparent",
        "animation": False,
        "tooltip": {"trigger": "axis"},
        "legend": {"textStyle": {"color": "#8b98ad"}, "bottom": 0},
        "grid": {"left": 64, "right": 16, "top": 16, "bottom": 48},
        "xAxis": {"type": "time", "axisLabel": {"color": "#8b98ad"}},
        "yAxis": {"type": "value", "scale": True, "axisLabel": {"color": "#8b98ad"},
                  "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
        "dataZoom": [{"type": "inside"}],
        "series": series,
    }


async def chart_panel(alpha_id: str, info: dict[str, Any], shown: bool) -> None:
    state = {"series": "pnl"}
    test_start = (info["periods"].get("test") or {}).get("startDate")
    os_start = (info["periods"].get("os") or {}).get("startDate")
    with ui.card().classes("ag-card w-full p-4 gap-3"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("Chart").classes("ag-panel-title")
            pick = ui.select(SERIES, value="pnl").props("dense outlined").classes("w-40")
        box = ui.column().classes("w-full")

    async def draw() -> None:
        box.clear()
        with box:
            spinner = ui.spinner(size="lg")
        try:
            data = await recordset(alpha_id, state["series"], info.get("stage"))
        except Exception as exc:  # noqa: BLE001
            box.clear()
            with box:
                ui.label(f"The chart did not load: {getattr(exc, 'message', None) or exc}").classes(
                    "ag-bad-box text-sm w-full")
            return
        spinner.delete()
        options = chart_options(data, state["series"], test_start, os_start, shown)
        with box:
            if data.get("problem"):
                ui.label(data["problem"]).classes("ag-warn-box text-sm w-full")
            if options is None:
                ui.label("Chart not available, please clone this alpha and re-simulate.").classes(
                    "ag-muted text-sm py-10 w-full text-center")
            else:
                ui.echart(options).classes("w-full h-80")

    async def changed(e: Any) -> None:
        state["series"] = e.value
        await draw()

    pick.on_value_change(changed)
    await draw()


# ── regional children ────────────────────────────────────────────────────────────────────────


async def regional_children(info: dict[str, Any], opener: Callable[[str], Any]) -> None:
    """A region-agnostic parent's Alphas, one per region, as BRAIN lists them."""
    with ui.card().classes("ag-card w-full p-4 gap-0"):
        ui.label("Regional Child Alphas").classes("ag-panel-title pb-2")
        if not info["children"]:
            ui.label("No child alphas found").classes("ag-muted text-sm")
            return
        holders = [ui.column().classes("w-full gap-2 py-3").style("border-top:1px solid var(--ag-border)")
                   for _ in info["children"]]
    for holder in holders:
        with holder:
            ui.spinner(size="sm")
    pages = await asyncio.gather(*(load_page(c) for c in info["children"]))
    for holder, child_id, (child, problems) in zip(holders, info["children"], pages):
        holder.clear()
        with holder:
            if child is None:
                ui.label(f"Error loading child alphas: {'; '.join(problems)}").classes("ag-bad-box text-sm")
                continue
            checks = (child["periods"].get("is") or {}).get("checks") or []
            failed = [c for c in checks if c.get("result") == "FAIL"]
            pending = [c for c in checks if c.get("result") == "PENDING"]
            passed = sum(1 for c in checks if c.get("result") == "PASS")
            region = str(child["settings"].get("region") or "")
            sub = (child["periods"].get("is") or {}).get("subregions")
            figures = sub.get(region) if isinstance(sub, dict) and isinstance(sub.get(region), dict) else \
                child["periods"].get("is")
            with ui.row().classes("items-center justify-between w-full"):
                ui.label(region or child_id).classes("text-lg font-medium")
                with ui.row().classes("items-center gap-3"):
                    text = (f"Tests: {len(failed)} failed" if failed else
                            f"Tests: {passed} pass, {len(pending)} pending submission check")
                    with ui.label(text).classes(f"text-sm {'ag-red' if failed else 'ag-green'}"):
                        if failed or pending:
                            with ui.tooltip().classes("max-w-[420px]"):
                                if failed:
                                    ui.label(f"{len(failed)} Failed Test{'s' if len(failed) != 1 else ''}").classes(
                                        "font-medium")
                                    for c in failed:
                                        ui.html(check_html(c, "IS")).classes("text-xs")
                                if pending:
                                    ui.label(f"{len(pending)} Pending Test{'s' if len(pending) != 1 else ''}") \
                                        .classes("font-medium")
                                    for c in pending:
                                        ui.html(check_html(c, "IS")).classes("text-xs")
                    ui.button("View alpha details", on_click=lambda c=child_id: opener(c)).props(
                        "flat dense no-caps size=sm")
            with ui.element("div").classes("grid grid-cols-3 md:grid-cols-6 gap-3 w-full"):
                for key, title, kind, _tip_text in CHILD_METRICS:
                    value = (figures or {}).get(key)
                    with ui.column().classes("gap-0"):
                        ui.label(title).classes("ag-figure-label")
                        ui.label(brain_value(value, kind) if is_num(value) else "-").classes("ag-figure")


# ── correlations ─────────────────────────────────────────────────────────────────────────────


def _corr_figure(kind: str | None, name: str, value: Any) -> str:
    v = value if is_num(value) else 0
    if kind in ("percent", "permyriad"):
        return brain_value(v, kind)
    return f"{v:,.4f}" if name == "correlation" else f"{v:,.2f}"


def correlated_table(info: dict[str, Any], data: dict[str, Any], opener: Callable[[str], Any]) -> None:
    """Each correlated Alpha's figures, under a first row for this Alpha itself; names open that Alpha."""
    props = ((data.get("schema") or {}).get("properties")) or []
    columns = [(i, p) for i, p in enumerate(props) if p.get("name") in CORRELATED_SHOWN]
    id_at = next((i for i, p in enumerate(props) if p.get("name") == "id"), None)
    own = info["periods"].get("is") or {}
    rows: list[dict[str, Any]] = []
    first: dict[str, Any] = {}
    for _, p in columns:
        n = p.get("name")
        first[n] = (f"{info.get('name') or 'anonymous'} (current)" if n == "name" else
                    str(info["settings"].get("universe") or "") if n == "universe" else
                    "1.00" if n == "correlation" else _corr_figure(p.get("type"), n, own.get(n)))
    rows.append(first | {"_id": ""})
    for record in data.get("records") or []:
        if not isinstance(record, list):
            continue
        row: dict[str, Any] = {"_id": record[id_at] if id_at is not None and id_at < len(record) else ""}
        for i, p in columns:
            n = p.get("name")
            value = record[i] if i < len(record) else None
            row[n] = (value or "anonymous") if n == "name" else str(value or "") if p.get("type") == "string" else \
                _corr_figure(p.get("type"), n, value)
        rows.append(row)
    cols = [{"field": p.get("name"), "headerName": p.get("title") or p.get("name"),
             "type": None if p.get("type") == "string" else "rightAligned", "flex": 1, "minWidth": 90,
             "cellClass": "ag-num"} for _, p in columns]
    cols = [{k: v for k, v in c.items() if v is not None} for c in cols]
    grid = ui.aggrid({"columnDefs": cols, "rowData": rows, "domLayout": "normal",
                      "defaultColDef": {"sortable": True, "resizable": True}}, theme="quartz").classes("w-full h-72")

    def clicked(e: Any) -> None:
        aid = (e.args.get("data") or {}).get("_id")
        if aid:
            return opener(aid)
    grid.on("cellClicked", clicked)
    ui.label("Click a row to open that alpha.").classes("ag-muted text-xs")


def histogram_options(data: dict[str, Any]) -> dict[str, Any]:
    """How many production Alphas fall in each band of correlation, on a log scale."""
    props = ((data.get("schema") or {}).get("properties")) or []
    names = [p.get("name") for p in props]

    def at(n: str) -> int | None:
        return names.index(n) if n in names else None
    low, high, count = at("min"), at("max"), at("alphas")
    bands, values = [], []
    for r in data.get("records") or []:
        if not isinstance(r, list):
            continue
        lo = r[low] if low is not None and low < len(r) else None
        hi = r[high] if high is not None and high < len(r) else None
        n = r[count] if count is not None and count < len(r) else None
        bands.append(f"{float(lo or 0):.1f}…{float(hi or 0):.1f}")
        values.append(n if is_num(n) and n > 0 else None)
    return {
        "backgroundColor": "transparent", "animation": False,
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 64, "right": 12, "top": 12, "bottom": 64},
        "xAxis": {"type": "category", "data": bands, "axisLabel": {"color": "#8b98ad", "rotate": 45}},
        "yAxis": {"type": "log", "name": "Nº Prod Alphas", "nameLocation": "middle", "nameGap": 48, "min": 1,
                  "axisLabel": {"color": "#8b98ad"}, "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
        "series": [{"type": "bar", "name": "Alphas", "data": values, "itemStyle": {"color": "#4ade80"}}],
    }


def _run_bar(what: str, last_run: str | None, on_run: Callable[[], Any], on_toggle: Callable[[], Any],
             is_open: bool, running: bool) -> None:
    with ui.row().classes("items-center gap-1 no-wrap"):
        ui.label(f"Last Run: {stamp(last_run) or '-'}").classes("ag-muted text-xs ag-mono")
        btn = ui.button(icon="refresh", on_click=on_run).props("flat round dense size=sm").tooltip(
            f"Run {what} on BRAIN")
        if running:
            btn.props("loading")
        ui.button(icon="expand_less" if is_open else "expand_more", on_click=on_toggle).props(
            "flat round dense size=sm").tooltip(f"{'Hide' if is_open else 'Show'} {what}")


async def correlation_panel(alpha_id: str, info: dict[str, Any], opener: Callable[[str], Any]) -> None:
    """Self, Power Pool (for Power Pool alphas) and production correlation: kept answers show for free; the refresh
    button is what spends BRAIN's budget, and opening an answer never asked for asks for it."""
    kinds = [c for c in CORRELATIONS if c[0] != "power-pool" or is_power_pool(info)]
    with ui.card().classes("ag-card w-full p-4 gap-0"):
        ui.label("Correlation").classes("ag-panel-title pb-1")
        for kind, title, none, compared in kinds:
            state: dict[str, Any] = {"data": None, "open": False, "running": False, "error": None, "status": None}
            try:
                state["data"] = await correlation(alpha_id, kind, "cached")
            except Exception as exc:  # noqa: BLE001
                state["error"] = getattr(exc, "message", None) or str(exc)

            @ui.refreshable
            def row(kind=kind, title=title, none=none, compared=compared, state=state) -> None:
                data = state["data"]
                with ui.row().classes("items-end w-full gap-4 py-3 flex-wrap").style(
                        "border-top:1px solid var(--ag-border)"):
                    ui.label(title).classes("text-sm font-medium min-w-[170px] self-center")
                    for label, key in (("Maximum", "max"), ("Minimum", "min")):
                        with ui.column().classes("gap-0 min-w-[90px]"):
                            with ui.row().classes("items-center gap-1 no-wrap"):
                                ui.label(label).classes("ag-figure-label")
                                _tip(title, f"{label} Pearson correlation coefficient from comparing a given alpha to "
                                            f"{compared}")
                            value = (data or {}).get(key)
                            ui.label(f"{value:.4f}" if is_num(value) else "-").classes("ag-figure")
                    ui.space()
                    _run_bar(title, (data or {}).get("fetchedAt"), lambda: run(kind, state, row, "refresh"),
                             lambda: toggle(kind, state, row), state["open"], state["running"])
                status = state["status"]
                if status in (410, 412):
                    ui.label("Invalid alpha PnL." if status == 412 else none).classes("ag-muted text-sm")
                elif state["error"]:
                    ui.label(f"{title} did not run: {state['error']}").classes("ag-bad-box text-sm w-full")
                if state["open"] and status not in (410, 412):
                    if state["running"] and not data:
                        with ui.row().classes("items-center gap-2"):
                            ui.spinner(size="sm")
                            ui.label(f"BRAIN is working out {title}…").classes("ag-muted text-sm")
                    elif data:
                        if not is_num(data.get("max")):
                            ui.label(none).classes("ag-muted text-sm")
                        elif kind == "prod":
                            ui.echart(histogram_options(data)).classes("w-full h-64")
                        else:
                            correlated_table(info, data, opener)

            row()

    async def run(kind: str, state: dict[str, Any], row: Any, mode: str) -> None:
        if state["running"]:
            return
        state.update(running=True, error=None, status=None)
        row.refresh()
        try:
            state["data"] = await correlation(alpha_id, kind, "refresh" if state["data"] else mode)
        except BrainError as exc:
            state["status"] = exc.status
            state["error"] = exc.message
        except Exception as exc:  # noqa: BLE001
            state["error"] = str(exc) or exc.__class__.__name__
        finally:
            state["running"] = False
            row.refresh()

    async def toggle(kind: str, state: dict[str, Any], row: Any) -> None:
        state["open"] = not state["open"]
        row.refresh()
        if state["open"] and not state["data"] and not state["running"]:
            await run(kind, state, row, "run")


# ── performance comparison ───────────────────────────────────────────────────────────────────


def _unit(kind: str | None) -> str:
    return " %" if kind == "percent" else " bps" if kind == "permyriad" else ""


def _bare(value: float, kind: str | None) -> str:
    return re.sub(r"%$| bps$", "", brain_value(value, kind or "decimal"))


def change_html(value: float, kind: str | None) -> str:
    """A change with a sign and an arrow, coloured as a verdict (never by colour alone)."""
    cls = "ag-green" if value > 0 else "ag-red" if value < 0 else "ag-muted"
    arrow = "▲" if value > 0 else "▼" if value < 0 else "–"
    text = _bare(value, kind)
    if value > 0 and not text.startswith("+"):
        text = f"+{text}"
    return f'<span class="{cls} ag-mono text-xs">{arrow} {html.escape(text)}</span>'


def partitions_of(paths: list[Any], body: dict[str, Any]) -> str:
    out = []
    for path in paths or []:
        found: Any = body
        for key in str(path).split("."):
            found = found.get(key) if isinstance(found, dict) else None
        value = "" if found is None else str(found)
        out.append(PARTITION.get(value, value))
    return ", ".join(out)


def comparison(info: dict[str, Any], data: dict[str, Any]) -> None:
    delay = str(info["settings"].get("delay") or "")
    for key, title, tips in SCORES:
        pair = data.get(key)
        if not isinstance(pair, dict):
            continue
        before = pair.get("before") if is_num(pair.get("before")) else None
        after = pair.get("after") if is_num(pair.get("after")) else None
        change = None if after is None else after - (before or 0)
        ui.label(f"Delay-{delay} {title}").classes("text-sm font-medium")
        with ui.element("div").classes("grid grid-cols-3 gap-4 w-full"):
            for label, value, tip in (("Before Submission", before, tips[0]), ("After Submission", after, tips[1]),
                                      ("Change", change, tips[2])):
                with ui.column().classes("gap-0"):
                    with ui.row().classes("items-center gap-1 no-wrap"):
                        ui.label(label).classes("ag-figure-label")
                        _tip(label, tip)
                    cls = "" if label != "Change" or value is None else \
                        "ag-green" if value > 0 else "ag-red" if value < 0 else ""
                    sign = "+" if label == "Change" and value is not None and value > 0 else ""
                    ui.label("-" if value is None else f"{sign}{value:,}").classes(f"ag-figure {cls}")
    stats = data.get("stats") or {}
    before, after = stats.get("before") or {}, stats.get("after")
    after_years = (data.get("yearlyStats") or {}).get("after")
    if not after or not after_years:
        ui.label("No data available.").classes("ag-muted text-sm")
        return
    with ui.element("div").classes("grid grid-cols-3 md:grid-cols-6 gap-3 w-full"):
        for key, title, kind, tip in METRICS:
            now = after.get(key) if is_num(after.get(key)) else 0
            was = before.get(key) if is_num(before.get(key)) else 0
            with ui.column().classes("gap-0"):
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.label(title).classes("ag-figure-label")
                    _tip(title, tip)
                ui.label(f"{_bare(now, kind)}{_unit(kind)}").classes("ag-figure")
                ui.html(change_html(now - was, kind))
    parts = partitions_of(data.get("partition") or [], info.get("body") or {})
    ui.label(f"IS Performance After Submission ({parts} alphas)").classes("text-sm font-medium pt-2")
    ui.label(f"Combined performance data is shown for all {parts} alphas. Submitting this alpha will not effect the "
             "combined performance of your other alphas.").classes("ag-muted text-xs")
    props = ((after_years.get("schema") or {}).get("properties")) or []
    compared = [c for c, _, _ in YEARLY[:7]]
    columns = [(next(i for i, p in enumerate(props) if p.get("name") == cid), cid) for cid in compared
               if any(p.get("name") == cid for p in props)]
    before_rows = ((data.get("yearlyStats") or {}).get("before") or {}).get("records") or []
    head = "".join(f"<th>{html.escape(str(props[i].get('title') or cid))}</th>" for i, cid in columns)
    body_rows = []
    for r_i, record in enumerate(after_years.get("records") or []):
        cells = []
        for i, cid in columns:
            value = record[i] if i < len(record) else None
            kind = props[i].get("type")
            if is_num(value) and cid != "year":
                earlier = before_rows[r_i][i] if r_i < len(before_rows) and i < len(before_rows[r_i]) else None
                cells.append(f"<td><div>{html.escape(_bare(value, kind) + _unit(kind))}</div>"
                             f"{change_html(value - (earlier if is_num(earlier) else 0), kind)}</td>")
            else:
                cells.append(f"<td>{html.escape('' if value is None else str(value))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    ui.html(f'<div class="ag-table-wrap"><table class="ag-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body_rows)}</tbody></table></div>').classes("w-full")
    pnl = data.get("pnl") or {}
    if pnl.get("records"):
        ui.label("PnL").classes("text-sm font-medium pt-2")
        ui.echart(pair_chart_options(pnl)).classes("w-full h-64")


def pair_chart_options(data: dict[str, Any]) -> dict[str, Any]:
    """The comparison's PnL: the pool as it is, then with this Alpha in it, each its own line."""
    props = ((data.get("schema") or {}).get("properties")) or []
    names = [p.get("name") for p in props]
    at = names.index("date") if "date" in names else 0
    rows = sorted({str(r[at]): r for r in data.get("records") or [] if isinstance(r, list)}.items())
    colours = ["#d9a21b", "#7c6cf0"]
    series = []
    k = 0
    for i, p in enumerate(props):
        if p.get("name") == "date":
            continue
        colour = colours[k % 2]
        k += 1
        series.append({"name": p.get("title") or p.get("name"), "type": "line", "showSymbol": False,
                       "data": [[d, r[i] if i < len(r) and is_num(r[i]) else None] for d, r in rows],
                       "lineStyle": {"width": 1.2, "color": colour}, "itemStyle": {"color": colour}})
    return {"backgroundColor": "transparent", "animation": False, "tooltip": {"trigger": "axis"},
            "legend": {"textStyle": {"color": "#8b98ad"}, "bottom": 0},
            "grid": {"left": 64, "right": 16, "top": 16, "bottom": 40},
            "xAxis": {"type": "time", "axisLabel": {"color": "#8b98ad"}},
            "yAxis": {"type": "value", "scale": True, "axisLabel": {"color": "#8b98ad"},
                      "splitLine": {"lineStyle": {"color": "#1e2b40"}}},
            "series": series}


async def performance_panel(alpha_id: str, info: dict[str, Any]) -> None:
    state: dict[str, Any] = {"data": None, "open": False, "running": False, "error": None}
    try:
        state["data"] = await performance(alpha_id, "cached")
    except Exception as exc:  # noqa: BLE001
        state["error"] = str(exc)

    @ui.refreshable
    def body() -> None:
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("Performance Comparison").classes("ag-panel-title")
            _run_bar("Performance Comparison", (state["data"] or {}).get("fetchedAt"), lambda: run("refresh"),
                     toggle, state["open"], state["running"])
        if state["error"]:
            ui.label(f"The comparison did not run: {state['error']}").classes("ag-bad-box text-sm w-full")
        if state["open"]:
            if state["running"] and not state["data"]:
                with ui.row().classes("items-center gap-2"):
                    ui.spinner(size="sm")
                    ui.label("BRAIN is comparing…").classes("ag-muted text-sm")
            elif state["data"]:
                comparison(info, state["data"])

    async def run(mode: str) -> None:
        if state["running"]:
            return
        state.update(running=True, error=None)
        body.refresh()
        try:
            state["data"] = await performance(alpha_id, "refresh" if state["data"] else mode)
        except Exception as exc:  # noqa: BLE001
            state["error"] = getattr(exc, "message", None) or str(exc) or exc.__class__.__name__
        finally:
            state["running"] = False
            body.refresh()

    async def toggle() -> None:
        state["open"] = not state["open"]
        body.refresh()
        if state["open"] and not state["data"] and not state["running"]:
            await run("run")

    with ui.card().classes("ag-card w-full p-4 gap-3"):
        body()


def settings_grid(settings: dict[str, Any], super_alpha: bool = False) -> None:
    shown = SETTINGS + ([("componentActivation", "Component Activation")] if super_alpha else [])
    ui.label("Simulation Settings").classes("text-sm font-medium")
    with ui.row().classes("gap-x-6 gap-y-3 flex-wrap"):
        for key, title in shown:
            with ui.column().classes("gap-0"):
                ui.label(title).classes("ag-figure-label whitespace-nowrap")
                value = settings.get(key)
                ui.label("" if value is None else str(value)).classes("ag-mono text-sm whitespace-nowrap")


# ── properties (saved to BRAIN, never submits) ────────────────────────────────────────────────

CATEGORIES = ["NONE", "PRICE_REVERSION", "PRICE_MOMENTUM", "VOLUME", "FUNDAMENTAL", "ANALYST", "PRICE_VOLUME",
              "RELATION", "SENTIMENT"]


async def properties_dialog(alpha_id: str, info: dict[str, Any] | None = None) -> None:
    """Name, category, colour, tags and description, saved to the Alpha on BRAIN. While its Power Pool check is open,
    the description is held to BRAIN's template and BRAIN's own complaints show as you type."""
    a = await load_alpha(alpha_id)
    if a is None and info is None:
        ui.notify("Alpha not in the local database yet.", type="warning")
        return
    if info is None:
        row = await kept(f"alpha:{alpha_id}")
        body = (row or {}).get("body") or (a.raw if a is not None and isinstance(a.raw, dict) else {})
        info = info_of(alpha_id, body, a.checks if a is not None else None) if body else None
    power_pool = bool(info and described_for_power_pool(info)) or bool(a and any(
        isinstance(c, dict) and c.get("name") == POWER_POOL_CHECK and c.get("result") in ("PASS", "PENDING")
        for c in a.checks or []))
    kind = (info or {}).get("type") or (a.type if a else None) or "REGULAR"
    name0 = (info or {}).get("name") if info else a.name
    with ui.dialog() as dialog, ui.card().classes("ag-card p-6 w-[680px] max-w-[95vw]"):
        ui.label(f"Properties · {alpha_id}").classes("text-lg font-semibold")
        name = ui.input("Name", value=name0 or "").classes("w-full")
        category0 = (info or {}).get("category") or (a.category if a else None) or "NONE"
        category0 = category0 if category0 in CATEGORIES else "NONE"
        with ui.row().classes("w-full gap-3 no-wrap"):
            category = ui.select({c: ("None" if c == "NONE" else words(c)) for c in CATEGORIES},
                                 value=category0, label="Category").classes("flex-1")
            color = ui.select(["", "RED", "YELLOW", "GREEN", "BLUE", "PURPLE"],
                              value=((info or {}).get("color") or (a.color if a else None) or ""), label="Color") \
                .classes("w-40")
        tags = ui.input("Tags (comma separated)",
                        value=", ".join((info or {}).get("tags") or (a.tags if a else None) or [])).classes("w-full")
        desc = ui.textarea("Description (Idea / Rationale for data used / Rationale for operators used)",
                           value=(info or {}).get("description") or (a.description if a else None) or "") \
            .classes("w-full").props("autogrow")
        counter = ui.label().classes("ag-muted text-xs")
        problems_box = ui.column().classes("gap-0 w-full")

        def validate(text: str) -> None:
            counter.set_text(f"{len(text or '')} characters (Power Pool needs ≥ {POWER_POOL_DESCRIPTION_MIN_LENGTH})")
            problems_box.clear()
            if power_pool:
                with problems_box:
                    for problem in description_problems(text or ""):
                        ui.label(problem).classes("ag-red text-xs")

        desc.on_value_change(lambda e: validate(e.value))
        validate(desc.value)
        if power_pool:
            ui.label("This alpha's Power Pool check is open, so BRAIN holds its description to the template.").classes(
                "ag-muted text-xs")

        async def write_desc() -> None:
            from ..engine.describe import describe_alpha
            try:
                desc.value = await describe_alpha(alpha_id)
            except Exception as exc:  # noqa: BLE001
                notify_error(exc, "Description failed: ")

        async def save() -> None:
            if power_pool and description_problems(desc.value or "") and (desc.value or "").strip():
                ui.notify("BRAIN will refuse this description: " + " ".join(description_problems(desc.value)),
                          type="warning", multi_line=True)
                return
            try:
                await actions.set_properties(
                    alpha_id, name=name.value or None, color=color.value or None,
                    tags=[t.strip() for t in tags.value.split(",") if t.strip()],
                    regular_description=desc.value if kind == "REGULAR" and (desc.value or "").strip() else None)
                if category.value != category0:
                    try:
                        await brain.alphas.patch(alpha_id, category=None if category.value == "NONE" else category.value)
                    except BrainError as exc:
                        notify_error(exc, "Saved, but BRAIN refused the category: ")
                await forget(f"alpha:{alpha_id}")
                ui.notify("Saved to BRAIN.", type="positive")
                dialog.close()
            except BrainError as exc:
                notify_error(exc)

        with ui.row().classes("justify-between w-full"):
            with ui.row().classes("gap-1"):
                ui.button("Write with AI", icon="auto_awesome", on_click=write_desc).props("flat")
                if power_pool:
                    ui.button("Use the template", icon="description",
                              on_click=lambda: desc.set_value(POWER_POOL_TEMPLATE)).props("flat")
            with ui.row():
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button("Save to BRAIN", icon="save", on_click=save)
    dialog.open()


# ── the page ─────────────────────────────────────────────────────────────────────────────────


def _transient_alpha(alpha_id: str, info: dict[str, Any] | None) -> Alpha:
    """An unsaved Alpha built from BRAIN's page, for tools that take an Alpha when AlphaGo has none stored."""
    is_ = ((info or {}).get("periods") or {}).get("is") or {}
    settings = (info or {}).get("settings") or {}
    return Alpha(id=alpha_id, type=(info or {}).get("type") or "REGULAR", code=(info or {}).get("code"),
                 selection=(info or {}).get("selection"), combo=(info or {}).get("combo"), settings=settings,
                 region=settings.get("region"), universe=settings.get("universe"), delay=settings.get("delay"),
                 status=(info or {}).get("status"), name=(info or {}).get("name"),
                 description=(info or {}).get("description"), sharpe=is_.get("sharpe"), fitness=is_.get("fitness"),
                 turnover=is_.get("turnover"), returns=is_.get("returns"), drawdown=is_.get("drawdown"),
                 margin=is_.get("margin"), checks=is_.get("checks"), raw=(info or {}).get("body"))


class AlphaPage:
    """One open Alpha page. ``ready`` finishes once BRAIN's sections have loaded (tests and callers may await it)."""

    def __init__(self, alpha_id: str, ids: list[str] | None, on_change: Callable[[], Any] | None) -> None:
        self.alpha_id = alpha_id
        self.ids = [i for i in ids or [] if i]
        self.on_change = on_change
        self.ready: asyncio.Task | None = None
        self.info: dict[str, Any] | None = None
        self.test_shown = False

    def step(self, offset: int) -> str | None:
        if self.alpha_id not in self.ids:
            return None
        at = self.ids.index(self.alpha_id) + offset
        return self.ids[at] if 0 <= at < len(self.ids) else None


async def open_alpha(alpha_id: str, ids: list[str] | Callable[[], Any] | None = None, *,
                     on_change: Callable[[], Any] | None = None) -> AlphaPage | None:
    """Open the Alpha page. ``ids`` (the list it was opened from) adds previous/next stepping; a callable second
    argument is the old ``on_change`` and still works."""
    if callable(ids):
        on_change, ids = ids, None
    if not alpha_id:
        return None
    page = AlphaPage(str(alpha_id), ids if isinstance(ids, list) else None, on_change)
    a = await load_alpha(page.alpha_id)
    cand = None
    if a is not None and a.candidate_id:
        async with db.session() as s:
            cand = await s.get(Candidate, a.candidate_id)

    # built at the page's root, never inside the dialog it was opened from: a closed dialog renders nothing inside it
    root = ui.context.client.layout
    with root, ui.dialog() as dialog, ui.card().classes("ag-card p-6 w-[1440px] max-w-[97vw] gap-4"):
        async def go(other: str | None) -> None:
            if other:
                dialog.close()
                await open_alpha(other, page.ids, on_change=on_change)

        with ui.row().classes("items-center justify-between w-full no-wrap"):
            with ui.row().classes("items-center gap-3 flex-wrap"):
                head_status = ui.row().classes("items-center gap-2")
                ui.label(page.alpha_id).classes("text-xl font-semibold ag-mono")
                ui.button(icon="content_copy", on_click=lambda: (ui.clipboard.write(page.alpha_id),
                                                                 ui.notify(f"Copied {page.alpha_id}"))).props(
                    "flat round dense size=sm").tooltip("Copy the alpha id")
                if a is not None:
                    verdict_badge(a.verdict)
                    if a.type == "SUPER":
                        badge("SUPER", "pending")
            with ui.row().classes("gap-1 items-center no-wrap"):
                if page.ids and page.alpha_id in page.ids:
                    at = page.ids.index(page.alpha_id)
                    ui.label(f"{at + 1} of {len(page.ids)}").classes("ag-muted text-xs")
                    ui.button(icon="chevron_left", on_click=lambda: go(page.step(-1))).props(
                        "flat round dense" + ("" if page.step(-1) else " disable")).tooltip("Previous alpha")
                    ui.button(icon="chevron_right", on_click=lambda: go(page.step(1))).props(
                        "flat round dense" + ("" if page.step(1) else " disable")).tooltip("Next alpha")
                ui.button(icon="open_in_new", on_click=lambda: ui.navigate.to(BRAIN_ALPHA_URL.format(page.alpha_id),
                                                                             new_tab=True)).props(
                    "flat round").tooltip("Open on BRAIN")
                ui.button(icon="close", on_click=dialog.close).props("flat round")
        header = ui.column().classes("w-full gap-2")
        notices = ui.column().classes("w-full gap-2")

        # AlphaGo's own reading of the alpha
        if a is not None:
            if cand and cand.hypothesis:
                ui.label(f"Hypothesis: {cand.hypothesis}").classes("text-sm ag-cyan")
            metrics_row(a)
            quality_badges(a)
            await repair_block(a)

        async def run_check() -> None:
            check_btn.props("loading")
            try:
                state, _ = await recheck(page.alpha_id, (page.info or {}).get("type") or (a.type if a else None))
                ui.notify(f"Check: {state.upper()}", type="positive" if state == "pass" else "warning")
                dialog.close()
                await open_alpha(page.alpha_id, page.ids, on_change=on_change)
                if on_change:
                    on_change()
            except BrainError as exc:
                notify_error(exc)
            finally:
                check_btn.props(remove="loading")

        async def optimize() -> None:
            from ..engine.optimizer import optimize_alpha
            try:
                made = await optimize_alpha(page.alpha_id, rounds_hint=1)
                ui.notify(f"Queued {len(made)} optimisation variants — they appear in Alphas when done.", type="info")
            except Exception as exc:  # noqa: BLE001
                notify_error(exc, "Optimise failed: ")

        async def repair() -> None:
            from .pages.tasks import repair_dialog
            s = (page.info or {}).get("settings") or (a.settings if a else None) or {}
            await repair_dialog(alpha_ids=[page.alpha_id], scope={
                "region": s.get("region") or (a.region if a else None),
                "delay": s.get("delay") if s.get("delay") is not None else (a.delay if a else None),
                "universe": s.get("universe") or (a.universe if a else None)})

        async def flip() -> None:
            with ui.dialog() as confirm, ui.card().classes("ag-card p-5 w-[520px]"):
                ui.label(f"Flip the sign of {page.alpha_id}?").classes("ag-section-title")
                ui.label("Adds and starts a task simulating the expression negated at the same settings, in Full "
                         "mode: one simulation. Nothing is submitted.").classes("text-sm ag-muted")

                async def go_flip() -> None:
                    confirm.close()
                    try:
                        from ..tools.paste_alphas import flip_sign
                        result = flip_sign(page.alpha_id)
                        task_id = await result if inspect.isawaitable(result) else result
                    except Exception as exc:  # noqa: BLE001
                        notify_error(exc, "Flip Sign failed: ")
                        return
                    ui.notify(f"Task {task_id} is simulating {page.alpha_id} with its sign flipped. Follow it on "
                              "Tasks.", type="positive")

                with ui.row().classes("justify-end w-full gap-2"):
                    ui.button("Cancel", on_click=confirm.close).props("flat")
                    ui.button("Flip Sign", icon="swap_vert", on_click=go_flip)
            confirm.open()

        async def copy_markdown() -> None:
            try:
                from ..tools.copy_alphas import alphas_markdown
            except ImportError:
                ui.notify("Copy as Markdown is not available in this version.", type="warning")
                return
            alpha = a if a is not None else _transient_alpha(page.alpha_id, page.info)
            try:
                text = alphas_markdown([alpha])
                text = await text if inspect.isawaitable(text) else text
            except Exception as exc:  # noqa: BLE001
                notify_error(exc, "Copy as Markdown failed: ")
                return
            ui.clipboard.write(str(text))
            ui.notify(f"Copied {page.alpha_id} as Markdown.", type="positive")

        with ui.row().classes("gap-2 flex-wrap items-center"):
            check_btn = ui.button("Check Submission", icon="rule", on_click=run_check).tooltip(
                "Run BRAIN's submission checks again without submitting")
            ui.button("Submit…", icon="send", on_click=lambda: confirm_submit(page.alpha_id, on_change)).props(
                "color=positive" + ("" if (a is not None and a.status != "ACTIVE") else " disable"))
            ui.button("Optimise", icon="tune", on_click=optimize).props("flat")
            ui.button("Repair…", icon="healing", on_click=repair).props("flat")
            flip_btn = ui.button("Flip Sign", icon="swap_vert", on_click=flip).props("flat").tooltip(
                "Sharpe is -1 or below: the same idea with the opposite sign may well pass")
            flip_btn.set_visibility(a is not None and a.sharpe is not None and a.sharpe <= -1)
            ui.button("Copy as Markdown", icon="content_paste", on_click=copy_markdown).props("flat")
            ui.button("Properties", icon="edit", on_click=lambda: properties_dialog(page.alpha_id, page.info)).props(
                "flat")

        sections = ui.column().classes("w-full gap-4")
        with sections:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="md")
                ui.label(f"Loading {page.alpha_id} from BRAIN…").classes("ag-muted text-sm")

    def opener(other: str) -> Any:
        dialog.close()
        return open_alpha(other)

    def render_header(info: dict[str, Any], hiding: bool) -> None:
        head_status.clear()
        with head_status:
            badge(STATUS_LABEL.get(info.get("status") or "UNSUBMITTED", str(info.get("status") or "")),
                  "pass" if info.get("status") == "ACTIVE" else "")
            if info.get("type") and info.get("type") != "REGULAR":
                badge(str(info["type"]), "pending")
        header.clear()
        unsubmitted = info.get("status") == "UNSUBMITTED"
        quick = unsubmitted and info["settings"].get("simulationMode") == "QUICK"
        failing = any(c.get("result") == "FAIL" for c in (info["periods"].get("is") or {}).get("checks") or [])
        with header:
            with ui.row().classes("items-center gap-4 w-full flex-wrap"):
                ui.label(info.get("name") or "anonymous").classes("text-lg")
                ui.label(f"Created {platform_day(info.get('dateCreated'))}").classes("ag-muted text-sm italic")
                ui.space()
                if info.get("type") == "RA_CHILD":
                    ui.label("Only the parent region-agnostic alpha can be submitted.").classes("ag-muted text-sm")
                elif hiding:
                    ui.button("Show test period", on_click=lambda: show_test(True)).props("dense no-caps")
                elif unsubmitted and not failing and not quick:
                    ui.link("Submit on BRAIN ↗", info["brainUrl"], new_tab=True).classes("ag-link text-sm").tooltip(
                        "Submitting is permanent, so it happens on BRAIN: this opens the alpha there")
                else:
                    ui.link("Open on BRAIN ↗", info["brainUrl"], new_tab=True).classes("ag-link text-sm")
            if info.get("type") == "SUPER":
                ui.label("Selection Expression").classes("ag-muted text-xs")
                ui.code(info.get("selection") or "").classes("w-full text-sm")
                ui.label("Combo Expression").classes("ag-muted text-xs")
                ui.code(info.get("combo") or "").classes("w-full text-sm")
            else:
                ui.code(info.get("code") or "", language="javascript").classes("w-full text-sm")
            settings_grid(info["settings"], info.get("type") == "SUPER")
        disabled = info.get("type") == "RA_CHILD" or hiding or not unsubmitted or failing or quick
        if disabled:
            check_btn.props("disable")
        else:
            check_btn.props(remove="disable")
        sharpe = (info["periods"].get("is") or {}).get("sharpe")
        flip_btn.set_visibility(is_num(sharpe) and sharpe <= -1 and info.get("type") in (None, "REGULAR"))

    async def render_sections() -> None:
        info = page.info
        assert info is not None
        control = test_control(info)
        hiding = control and not page.test_shown
        regional = info.get("type") == "RA_PARENT"
        render_header(info, hiding)
        sections.clear()
        with sections:
            if control:
                with ui.row().classes("ag-warn-box w-full items-center justify-between"):
                    ui.label("Test period and overall stats are hidden by default when test period is specified.")
                    ui.button("Hide test period" if page.test_shown else "Show test period",
                              on_click=lambda: show_test(not page.test_shown)).props("dense no-caps")
            with ui.element("div").classes("grid grid-cols-1 xl:grid-cols-2 gap-4 w-full"):
                left = ui.column().classes("gap-4 min-w-0")
                right = ui.column().classes("gap-4 min-w-0")
        jobs_: list[tuple[ui.column, Callable[[], Any]]] = []
        with left:
            chart_box = ui.column().classes("w-full")
            if not hiding:
                testing_status(info, control, page.test_shown)
        jobs_.append((chart_box, lambda: chart_panel(page.alpha_id, info, page.test_shown)))
        with right:
            if not regional:
                jobs_.append((ui.column().classes("w-full"),
                              lambda: summary(page.alpha_id, info, control, page.test_shown)))
            if not regional and not hiding:
                jobs_.append((ui.column().classes("w-full"), lambda: correlation_panel(page.alpha_id, info, opener)))
            if not regional and not hiding and info.get("stage") == "IS":
                jobs_.append((ui.column().classes("w-full"), lambda: performance_panel(page.alpha_id, info)))
            if regional:
                jobs_.append((ui.column().classes("w-full"), lambda: regional_children(info, opener)))

        async def fill(box: ui.column, make: Callable[[], Any]) -> None:
            # each section loads in its own task, inside its own box, so a slow BRAIN read holds up nothing else
            try:
                with box:
                    await make()
            except Exception as exc:  # noqa: BLE001 - one failing section never blanks the page
                with box:
                    ui.label(f"This section did not load: {getattr(exc, 'message', None) or exc}").classes(
                        "ag-bad-box text-sm w-full")

        await asyncio.gather(*(fill(box, make) for box, make in jobs_))

    async def show_test(shown: bool) -> None:
        page.test_shown = shown
        await render_sections()

    async def load() -> None:
        info, problems = await load_page(page.alpha_id)
        page.info = info
        notices.clear()
        with notices:
            for problem in problems:
                ui.label(problem).classes("ag-warn-box text-sm w-full")
        if info is None:
            sections.clear()
            with sections:
                ui.label("BRAIN's page for this alpha could not be shown. AlphaGo's own figures are above.").classes(
                    "ag-muted text-sm")
            return
        page.test_shown = not test_control(info)
        await render_sections()

    def dispose(e: Any) -> None:
        """A closed page stops reading BRAIN and is removed a little later (once a handler that closed it, such as
        Check Submission or stepping to the next alpha, is done with its elements), so pages do not pile up."""
        if e.value:
            return
        if page.ready is not None and not page.ready.done():
            page.ready.cancel()
        with root:
            ui.timer(30.0, lambda: None if dialog.value or dialog.is_deleted else dialog.delete(), once=True)

    dialog.on_value_change(dispose)
    dialog.open()
    page.ready = asyncio.create_task(load())
    return page
