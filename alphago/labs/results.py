"""What the Tasks and Submittable tabs read: every lab task's progress, one task's results, and every submittable
Alpha any task found (alpha-harness api/lab_tasks.py, tasks/results.tsx and tasks/submittable.tsx).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from ..brain.platform import (
    IMPLAUSIBLE_SHARPE,
    QUICK_MODE,
    brain_verdict,
    failed_checks,
    investability_of,
)
from ..db import db
from ..db.models import Alpha
from ..db.models_labs import LabTask, LabTrial
from . import runner as _runner
from .objectives import label
from .runner import FAILURE

STATUS_LABELS = {"IDLE": "Not started", "QUEUED": "Waiting for cores", "RUNNING": "Running", "PAUSED": "Paused",
                 "COMPLETE": "Complete", "FAILED": "Failed"}
METRICS = ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin", "test_sharpe", "after_cost_5bps")
LOWER_IS_BETTER = {"turnover", "drawdown"}


def _aware(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def times(t: LabTask, now: datetime | None = None) -> tuple[float | None, float | None]:
    """(seconds waiting for cores, seconds running): a task waits for its cores before it runs, which is not
    running time, so the two are told apart rather than added together."""
    now = now or datetime.now(timezone.utc)
    queued, started, finished = _aware(t.queued_at), _aware(t.started_at), _aware(t.finished_at)
    waiting = None
    if queued is not None:
        waiting = max(0.0, ((started or (now if t.status == "QUEUED" else queued)) - queued).total_seconds())
        if started is not None and started < queued:
            waiting = 0.0 if t.status != "QUEUED" else max(0.0, (now - queued).total_seconds())
    running = max(0.0, ((finished or now) - started).total_seconds()) if started is not None else None
    return waiting, running


def fields_of(space: Any) -> int:
    if not isinstance(space, dict):
        return 0
    pooled = len(space.get("fields") or {})
    if pooled or not isinstance(space.get("variables"), list):
        return pooled
    return sum(len(v.get("fields") or {}) for v in space["variables"] if isinstance(v, dict))


async def task_rows() -> list[dict[str, Any]]:
    """Every lab task, newest first, with what the Tasks list shows."""
    out = []
    for t in await _runner.runner.list():
        summary = await _runner.runner.summary(t.id)
        waiting, running = times(t)
        lab = _runner.runner.labs.get(t.lab)
        params = t.params or {}
        out.append({
            "id": t.id, "name": t.name or "", "lab": t.lab, "lab_name": lab.title if lab else t.lab,
            "label": t.name or f"{lab.title if lab else t.lab} #{t.id}", "status": t.status,
            "status_label": ("Stopping" if t.stopping else STATUS_LABELS.get(t.status, t.status)),
            "message": t.message or "", "cores": t.cores, "target": t.max_trials,
            "simulated": summary["simulated"], "cached": summary["cached"], "queued": summary["queued"],
            "running": summary["running"], "failed": summary["failed"],
            "reasons": "; ".join(f"{m} ×{n}" for m, n in summary["reasons"]),
            "best": t.best_value, "objective": label(t.objective), "region": t.region, "delay": t.delay,
            "universe": t.universe, "fields": fields_of(t.space), "datasets": ", ".join(params.get("dataset_ids") or []),
            "waiting_s": waiting, "running_s": running, "stopping": t.stopping,
            "created": t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else "",
        })
    return out


def _row(r: dict[str, Any]) -> dict[str, Any]:
    checks = r.get("checks") if r.get("checks") is not None else r.get("is_checks")
    verdict = r.get("brain_verdict") or brain_verdict(checks, r.get("simulation_mode"))
    refused = failed_checks(checks, gating_only=True) if checks else list(r.get("failed") or [])
    settings = r.get("settings") or {}
    return {**{k: v for k, v in r.items() if k not in ("checks", "is_checks")},
            "region": r.get("region") or settings.get("region"), "delay": r.get("delay", settings.get("delay")),
            "universe": r.get("universe") or settings.get("universe"),
            "neutralization": r.get("neutralization") or settings.get("neutralization"),
            "investability": investability_of(settings), "failed_checks": ", ".join(refused), "refused_by": refused,
            "submittable": verdict == "submittable", "pending": verdict == "pending",
            "implausible": bool(r.get("implausible")) or any(
                v is not None and abs(float(v)) > IMPLAUSIBLE_SHARPE
                for v in (r.get("sharpe"), r.get("train_sharpe"), r.get("test_sharpe")))}


async def results(task_id: int, limit: int = 5000) -> dict[str, Any]:
    """One task's ranked Alphas, the Across Regions summary and the Checks Failed Together table."""
    rows = [_row(r) for r in await _runner.runner.top(task_id, limit)]
    return {"rows": rows, "regions": across_regions(rows), "check_sets": check_sets(rows)}


def across_regions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per region: Alphas, submittable (or still being judged), refused, and the best each metric reaches among the
    Alphas nothing refused: a best BRAIN will not take is not a best."""
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = str(r.get("region") or "—")
        row = seen.setdefault(key, {"region": key, "alphas": 0, "submittable": 0, "refused": 0,
                                    **{f"best_{m}": None for m in METRICS}})
        row["alphas"] += 1
        green = r["submittable"] or r["pending"]
        row["submittable" if green else "refused"] += 1
        if green:
            for m in METRICS:
                value = r.get(m)
                if value is None:
                    continue
                held = row[f"best_{m}"]
                if held is None or (value < held if m in LOWER_IS_BETTER else value > held):
                    row[f"best_{m}"] = value
    return sorted(seen.values(), key=lambda x: (-x["submittable"], -x["alphas"]))


def check_sets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every distinct set of refusing checks, and how many Alphas hit exactly that set: one Alpha usually trips
    several at once, so a per-check tally double-counts and every sweep looks like the same wall of failures."""
    counts: Counter[tuple[str, ...]] = Counter(tuple(sorted(r["refused_by"])) for r in rows)
    total = len(rows) or 1
    return [{"checks": " + ".join(k) if k else "— (none)", "count": len(k), "alphas": n, "share": n / total}
            for k, n in counts.most_common()]


async def submittable() -> list[dict[str, Any]]:
    """Every Alpha any lab task found that BRAIN would take: verdict submittable, not Quick, each once, from the
    task that found it first."""
    async with db.session() as s:
        tasks = {t.id: t for t in await s.scalars(select(LabTask))}
        rows = (await s.execute(select(LabTrial, Alpha).join(Alpha, Alpha.id == LabTrial.alpha_id).where(
            LabTrial.state == "COMPLETE", LabTrial.generation > 0, LabTrial.value > FAILURE,
            Alpha.brain_verdict == "submittable").order_by(LabTrial.task_id, LabTrial.number))).all()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tr, a in rows:
        if a.id in seen or str(a.simulation_mode or "").upper() == QUICK_MODE:
            continue
        seen.add(a.id)
        task = tasks.get(tr.task_id)
        lab = _runner.runner.labs.get(task.lab) if task else None
        settings = a.settings or tr.settings or {}
        out.append({"alpha_id": a.id, "expression": a.code or tr.expression, "task_id": tr.task_id,
                    "task_name": (task.name if task else None) or f"{lab.title if lab else 'Task'} #{tr.task_id}",
                    "region": a.region, "delay": a.delay, "universe": a.universe, "neutralization": a.neutralization,
                    "investability": investability_of(settings), "sharpe": a.sharpe, "fitness": a.fitness,
                    "turnover": a.turnover, "returns": a.returns, "drawdown": a.drawdown, "margin": a.margin,
                    "test_sharpe": a.test_sharpe, "test_fitness": a.test_fitness, "train_sharpe": a.train_sharpe,
                    "after_cost_5bps": a.after_cost_5bps, "prod_corr": a.prod_corr, "pp_corr": a.pp_corr,
                    "implausible": any(v is not None and abs(v) > IMPLAUSIBLE_SHARPE
                                       for v in (a.sharpe, a.train_sharpe, a.test_sharpe))})
    return out


#: Bounds the Submittable tab filters by; `div` converts the display unit to BRAIN's fraction.
BOUNDS = [("sharpe", "Sharpe", 1), ("fitness", "Fitness", 1), ("turnover", "Turnover %", 100),
          ("returns", "Returns %", 100), ("drawdown", "Drawdown %", 100), ("margin", "Margin bps", 10_000),
          ("test_sharpe", "Test Sharpe", 1), ("test_fitness", "Test Fitness", 1),
          ("after_cost_5bps", "After-Cost Sharpe", 1), ("prod_corr", "Production Correlation", 1),
          ("pp_corr", "Power Pool Correlation", 1)]
PICKS = ("region", "delay", "universe", "neutralization", "investability")


def filter_submittable(rows: list[dict[str, Any]], *, search: str = "", picked: dict[str, list[Any]] | None = None,
                       bounds: dict[str, tuple[float | None, float | None]] | None = None,
                       implausible: bool = False) -> list[dict[str, Any]]:
    """Every filter at once (AND). A bound leaves out the Alphas without that figure yet: they cannot be shown to
    meet it."""
    needle = search.strip().lower()
    divs = {k: d for k, _, d in BOUNDS}
    out = []
    for r in rows:
        if not implausible and r.get("implausible"):
            continue
        if needle and not any(needle in str(r.get(k) or "").lower() for k in ("alpha_id", "expression")):
            continue
        if any(v and str(r.get(k)) not in {str(x) for x in v} for k, v in (picked or {}).items()):
            continue
        ok = True
        for k, (lo, hi) in (bounds or {}).items():
            if lo is None and hi is None:
                continue
            value = r.get(k)
            if value is None or (lo is not None and value < lo / divs.get(k, 1)) or \
                    (hi is not None and value > hi / divs.get(k, 1)):
                ok = False
                break
        if ok:
            out.append(r)
    return out
