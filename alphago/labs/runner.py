"""Lab task runner: the alpha-harness task lifecycle on top of AlphaGo's simulator.

A lab (Search, Template, Evolution, ...) is a small class: it plans a task and, when the task has room, hands back
the next trials to simulate. The runner does the rest, the way alpha-harness' scheduler does:

- **cores**: a task runs only when its cores fit in BRAIN's 8 slots (oldest Run first; a big task does not block a
  small one behind it); each core keeps 10 simulations in flight, refilled in tens;
- **trials**: every request is a `LabTrial` row before it is sent, so a restart resumes where it was;
- **dedup**: the simulator answers an identical request from an alpha already simulated (FREE, no quota spent,
  not counted against the task's simulations);
- **scoring**: the lab's objective (train Sharpe / train Fitness / Sharpe) minus 1 per failed structural check;
  |Sharpe| ≥ 10 is implausible and told as a failure;
- **refusals**: when BRAIN refuses a setting value in a market, parked trials repeating it are closed unsent;
- **lifecycle**: Run (queues until cores fit), Pause, Stop (scores what is out), Stop again (cancels on BRAIN),
  resize, rename, delete, Run All; "Complete with nothing scored" ends Failed with the commonest reason;
- **Quick -> Full**: a Quick-mode Alpha that passed every check BRAIN ran is owed a Full run (what BRAIN submits):
  parked as `params.fullOf`, sent ahead of the search, never counted against the task's simulations, and the Quick
  trial is hidden from the results once its Full run is back;
- **lending idle cores** (pref `pref:labs` lend_idle_cores): a running task that has sent everything holds no
  cores, so a queued task starts beside its last batches, and free slots are lent to the tasks still sending;
- **withdrawn settings**: a parked trial whose settings BRAIN's OPTIONS no longer accept is pruned unsent.

Simulator cores are accounted per task: lab task N uses the simulator task key -N, so lab and research tasks can
never collide.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar

from sqlalchemy import func, select, update

from ..brain.platform import (
    FULL_MODE,
    IGNORED_CHECKS,
    IMPLAUSIBLE_SHARPE,
    MULTI_SIMULATION_SIZE,
    QUICK_MODE,
    SIMULATION_SLOTS,
    STRUCTURAL_CHECKS,
    strip_quota_checks,
)
from ..brain.simulations import normalize_settings, settings_hash
from ..db import db
from ..db.models import Alpha, Candidate
from ..db.models_labs import LabTask, LabTrial
from ..events import bus

log = logging.getLogger(__name__)

FAILURE = -10.0
POLL_S = 5.0
RUNNABLE = {"IDLE", "PAUSED", "FAILED"}
OPEN_TRIALS = ("PARKED", "RUNNING")
FREE_MESSAGE = "Matched an alpha already simulated; no quota spent."
PENDING_FULL = "Waiting to run in Full mode."
NOTHING_LEFT = "The lab had nothing more to try."
PREF_KEY = "pref:labs"
OBJECTIVES = {"sharpe": "Sharpe", "fitness": "Fitness", "train_sharpe": "Train Sharpe",
              "train_fitness": "Train Fitness"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def core_key(task_id: int) -> int:
    """The simulator's task key for a lab task (negative, so it never meets a research task's id)."""
    return -int(task_id)


@dataclass
class TrialSpec:
    """One simulation a lab wants: expression, full settings and what it chose (to learn from)."""

    expression: str
    settings: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    generation: int = 1
    parent: list[int] | None = None
    message: str | None = None
    state: str = "PARKED"            # PRUNED = record it without sending; COMPLETE = already scored (a seed)
    # only for a trial written already scored (Evolution's seeds, generation 0): nothing is simulated
    alpha_id: str | None = None
    value: float | None = None
    free: bool = False
    figures: dict[str, Any] | None = None


class Lab:
    """Base class of a lab. Subclasses set `name` and implement `plan` and `refill`."""

    name: ClassVar[str] = "lab"
    title: ClassVar[str] = "Lab"
    objective: ClassVar[str] = "sharpe"
    max_cores: ClassVar[int] = SIMULATION_SLOTS

    async def plan(self, task: LabTask) -> dict[str, Any]:
        """Check the inputs and return the frozen search space (stored on the task). Raise ValueError to refuse."""
        return {}

    async def initial(self, task: LabTask) -> list[TrialSpec]:
        """Trials written up front (Super Alpha's full grid, paste lists, seeds). Default: none."""
        return []

    async def refill(self, task: LabTask, want: int, ctx: RefillContext) -> list[TrialSpec]:
        """Up to `want` new trials to send now. Return [] when the lab has nothing more to offer."""
        return []

    def done_early(self, task: LabTask, ctx: RefillContext) -> str | None:
        """A message to finish the task before its budget (stalled search, ...), else None."""
        return None

    async def on_scored(self, task: LabTask, trials: list[LabTrial]) -> None:
        """Called after a tick harvested trials (Evolution breeds, Fields Sampler expands winners, ...)."""

    async def preview(self, params: dict[str, Any]) -> dict[str, Any]:
        """Free preview of a task: {problems, warnings, sample[{expression, settings}], combinations?}."""
        return {"problems": [], "warnings": [], "sample": []}

    def nothing_left(self, task: LabTask) -> str:
        """Why the task completes when `refill` had nothing more to offer ("Stopped early: ...")."""
        return NOTHING_LEFT

    def forget(self, task_id: int) -> None:
        """Drop the lab's memory of a task (deleted)."""

    async def busy(self, task: LabTask) -> bool:
        """True while the lab is still producing trials in the background (an LLM call): the task must not finish
        just because nothing is parked or running at this moment."""
        return False

    async def send(self, task: LabTask, trials: list[LabTrial], ctx: RefillContext) -> bool:
        """Send these parked trials itself and return True (SUPER simulations, which the simulator cannot build);
        False lets the runner send them through the simulator."""
        return False


@dataclass
class RefillContext:
    """What a lab may read while refilling: the task's memory of what it already tried."""

    seen: dict[str, float | bool]                  # key -> score (float), failed (False), or in flight (True)
    scored: list[LabTrial]                          # COMPLETE trials, best first
    spent: int                                      # simulations committed (free answers excluded)
    refusals: set[tuple]                            # settings BRAIN refused in this task


def trial_key(expression: str, settings: dict[str, Any]) -> str:
    """Identity of a request: canonical expression and the settings that change the result."""
    try:
        from ..fastexpr import canonical_hash
        expr = canonical_hash(expression)
    except Exception:
        expr = re.sub(r"\s+", "", expression)
    st = {k: v for k, v in normalize_settings(settings).items() if k != "simulationMode"}
    quick = "|Q" if str((settings or {}).get("simulationMode") or "").upper() == QUICK_MODE else ""
    return f"{expr}|{settings_hash(st)}{quick}"


def passed_quick(alpha: Alpha) -> bool:
    """A Quick Alpha that passed every check BRAIN ran on it: owed a run in Full mode."""
    if str(alpha.simulation_mode or "").upper() != QUICK_MODE:
        return False
    checks = alpha.is_checks if alpha.is_checks is not None else alpha.checks
    results = {str(c.get("result") or "").upper() for c in strip_quota_checks(checks)
               if str(c.get("name")) not in IGNORED_CHECKS}
    results.discard("PENDING")          # not run yet (self-correlation waits for a check): not a refusal
    return bool(results) and results <= {"PASS", "WARNING"}


# ---------------------------------------------------------------- scoring

def objective_value(alpha: Alpha, objective: str) -> float | None:
    if objective == "train_sharpe":
        return alpha.train_sharpe if alpha.train_sharpe is not None else alpha.sharpe
    if objective == "train_fitness":
        return alpha.train_fitness if alpha.train_fitness is not None else alpha.fitness
    if objective == "fitness":
        return alpha.fitness
    return alpha.sharpe


def structural_penalty(checks: list[dict[str, Any]] | None) -> float:
    return float(sum(1 for c in checks or [] if str(c.get("name")) in STRUCTURAL_CHECKS
                     and str(c.get("result") or "").upper() == "FAIL"))


def score(alpha: Alpha, objective: str) -> tuple[float, float, bool, dict[str, Any]]:
    """(value told to the search, penalty, implausible, figures) for a simulated alpha."""
    from ..brain.platform import failed_checks
    checks = alpha.is_checks if alpha.is_checks is not None else alpha.checks
    penalty = structural_penalty(checks)
    raw = objective_value(alpha, objective)
    search_sharpe = alpha.train_sharpe if alpha.train_sharpe is not None else alpha.sharpe
    implausible = search_sharpe is not None and abs(search_sharpe) >= IMPLAUSIBLE_SHARPE
    figures = {k: getattr(alpha, k) for k in ("sharpe", "fitness", "turnover", "returns", "drawdown", "margin",
                                              "train_sharpe", "train_fitness", "test_sharpe", "test_fitness")}
    figures["failed"] = failed_checks(checks)
    value = FAILURE if raw is None or implausible else float(raw) - penalty
    return value, penalty, implausible, figures


# ---------------------------------------------------------------- refusals

_SETTING = re.compile(r"\bsettings\.(\w+)")


def refusal_of(settings: dict[str, Any], message: str | None) -> tuple | None:
    """A setting BRAIN refused, learned from its message: (region, delay, universe, setting, value)."""
    m = _SETTING.search(message or "")
    if not m:
        return None
    name = m.group(1)
    s = normalize_settings(settings)
    return (s["region"], s["delay"], s["universe"], name, json.dumps(s.get(name), sort_keys=True, default=str))


def repeats_refusal(settings: dict[str, Any], refusals: set[tuple]) -> tuple | None:
    s = normalize_settings(settings)
    for r in refusals:
        if (s["region"], s["delay"], s["universe"]) == r[:3] and \
                json.dumps(s.get(r[3]), sort_keys=True, default=str) == r[4]:
            return r
    return None


# ---------------------------------------------------------------- the runner

class LabRunner:
    def __init__(self) -> None:
        self.labs: dict[str, Lab] = {}
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._tick_locks: dict[int, asyncio.Lock] = {}
        self.lend_idle_cores = False
        self.lendable = 0          # free cores lent to running tasks this tick (lending on)

    # ---- registry
    def register(self, lab: Lab) -> Lab:
        self.labs[lab.name] = lab
        return lab

    def lab(self, name: str) -> Lab:
        if name not in self.labs:
            raise ValueError(f"Unknown lab {name!r}")
        return self.labs[name]

    # ---- reading
    async def get(self, task_id: int) -> LabTask | None:
        async with db.session() as s:
            return await s.get(LabTask, task_id)

    async def list(self, lab: str | None = None, limit: int = 300) -> list[LabTask]:
        stmt = select(LabTask).order_by(LabTask.id.desc()).limit(limit)
        if lab:
            stmt = stmt.where(LabTask.lab == lab)
        async with db.session() as s:
            return list(await s.scalars(stmt))

    async def trials(self, task_id: int, states: tuple[str, ...] | None = None) -> list[LabTrial]:
        stmt = select(LabTrial).where(LabTrial.task_id == task_id).order_by(LabTrial.number)
        if states:
            stmt = stmt.where(LabTrial.state.in_(states))
        async with db.session() as s:
            return list(await s.scalars(stmt))

    async def summary(self, task_id: int) -> dict[str, Any]:
        """Counts for the task list: simulated, cached (free), queued, running, failed, top failure reasons."""
        async with db.session() as s:
            rows = (await s.execute(select(LabTrial.state, LabTrial.free, func.count()).where(
                LabTrial.task_id == task_id).group_by(LabTrial.state, LabTrial.free))).all()
            reasons = (await s.execute(select(LabTrial.message, func.count()).where(
                LabTrial.task_id == task_id, LabTrial.state == "FAIL").group_by(LabTrial.message)
                .order_by(func.count().desc()).limit(3))).all()
        counts = Counter()
        for state, free, n in rows:
            counts[state] += n
            if free:
                counts["free"] += n
        return {"simulated": counts["COMPLETE"] + counts["FAIL"] - counts["free"], "cached": counts["free"],
                "queued": counts["PARKED"], "running": counts["RUNNING"], "failed": counts["FAIL"],
                "pruned": counts["PRUNED"], "reasons": [(m or "failed", n) for m, n in reasons]}

    async def top(self, task_id: int, limit: int = 50) -> list[dict[str, Any]]:
        """Scored trials, best first, with the alpha's current figures."""
        async with db.session() as s:
            full_of = {int((p or {}).get("fullOf")) for p in await s.scalars(select(LabTrial.params).where(
                LabTrial.task_id == task_id, LabTrial.state == "COMPLETE")) if (p or {}).get("fullOf") is not None}
            trials = list(await s.scalars(select(LabTrial).where(
                LabTrial.task_id == task_id, LabTrial.state == "COMPLETE", LabTrial.generation > 0,
                LabTrial.value > FAILURE).order_by(LabTrial.value.desc(), LabTrial.number).limit(
                limit + len(full_of))))
            # once its Full run is back, that stands in for a Quick Alpha: the same figures, judged in full
            trials = [t for t in trials if t.number not in full_of][:limit]
            ids = [t.alpha_id for t in trials if t.alpha_id]
            alphas = {a.id: a for a in await s.scalars(select(Alpha).where(Alpha.id.in_(ids)))} if ids else {}
        out = []
        for t in trials:
            a = alphas.get(t.alpha_id or "")
            out.append({"trial": t.number, "alpha_id": t.alpha_id, "expression": t.expression,
                        "settings": t.settings, "value": t.value, "free": t.free,
                        **{k: getattr(a, k, None) for k in (
                            "sharpe", "fitness", "turnover", "returns", "drawdown", "margin", "train_sharpe",
                            "test_sharpe", "test_fitness", "after_cost_5bps", "prod_corr", "brain_verdict",
                            "status", "long_count", "short_count", "simulation_mode", "checks", "is_checks",
                            "region", "delay", "universe", "neutralization", "decay", "truncation", "max_trade",
                            "max_position", "train_fitness")},
                        "quick": str(getattr(a, "simulation_mode", "") or "").upper() == QUICK_MODE,
                        "full_of": (t.params or {}).get("fullOf"), "implausible": t.implausible,
                        "failed": (t.figures or {}).get("failed") or []})
        return out

    # ---- creating
    async def create(self, lab: str, params: dict[str, Any], *, cores: int = 4, max_trials: int = 500,
                     name: str | None = None, run: bool = False) -> int:
        impl = self.lab(lab)
        cores = max(1, min(int(cores), impl.max_cores, SIMULATION_SLOTS))
        task = LabTask(lab=lab, name=(name or "").strip()[:160] or None, cores=cores, params=params,
                       max_trials=max(1, int(max_trials)), objective=params.get("objective") or impl.objective,
                       region=params.get("region"), delay=params.get("delay"), universe=params.get("universe"))
        task.space = await impl.plan(task)          # ValueError refuses the task before anything is stored
        async with db.write() as s:
            s.add(task)
            await s.flush()
            tid = task.id
        initial = await impl.initial(await self.get(tid))
        if initial:
            await self.add_trials(tid, initial)
        if run:
            await self.run(tid)
        return tid

    async def add_trials(self, task_id: int, specs: list[TrialSpec]) -> list[int]:
        async with db.write() as s:
            n = await s.scalar(select(func.max(LabTrial.number)).where(LabTrial.task_id == task_id)) or 0
            ids = []
            for spec in specs:
                n += 1
                settings = normalize_settings(spec.settings)
                row = LabTrial(task_id=task_id, number=n, state=spec.state, params=spec.params,
                               expression=spec.expression.strip(), settings=settings,
                               settings_hash=settings_hash(settings), generation=spec.generation,
                               parent=spec.parent, message=spec.message, alpha_id=spec.alpha_id,
                               value=spec.value, free=spec.free, figures=spec.figures,
                               finished_at=utcnow() if spec.state in ("PRUNED", "COMPLETE", "FAIL") else None)
                s.add(row)
                await s.flush()
                ids.append(row.id)
        return ids

    # ---- lifecycle
    async def _set(self, task_id: int, **values: Any) -> None:
        async with db.write() as s:
            await s.execute(update(LabTask).where(LabTask.id == task_id).values(**values))

    async def run(self, task_id: int) -> None:
        t = await self.get(task_id)
        if t is None or t.status not in RUNNABLE:
            return
        extra = {"message": None, "finished_at": None} if t.status == "FAILED" else {}
        await self._set(task_id, status="QUEUED", queued_at=utcnow(), stopping=False, **extra)
        await self.start_waiting()

    async def run_all(self) -> int:
        n = 0
        for t in await self.list():
            if t.status == "IDLE":
                await self.run(t.id)
                n += 1
        return n

    async def pause(self, task_id: int) -> None:
        from ..engine.simulator import simulator
        t = await self.get(task_id)
        if t is None or t.status not in ("RUNNING", "QUEUED") or t.stopping:
            return
        await self._set(task_id, status="PAUSED")
        await self._unsend(task_id)
        simulator.pause_task(core_key(task_id))
        await self.start_waiting()

    async def stop(self, task_id: int) -> str:
        """First press: stop sending, score what is out. Second press: cancel on BRAIN and finish now."""
        from ..engine.simulator import simulator
        t = await self.get(task_id)
        if t is None or t.status in ("COMPLETE", "IDLE"):
            return "nothing to stop"
        await self._unsend(task_id)
        running = await self.trials(task_id, ("RUNNING",))
        if t.stopping or not running or t.status != "RUNNING":
            if running:
                for sid in {c for c in await self._sim_ids(running) if c}:
                    try:
                        await simulator.cancel_running(sid, force=True)
                    except Exception:
                        log.exception("cancel %s failed", sid)
                async with db.write() as s:
                    await s.execute(update(LabTrial).where(LabTrial.task_id == task_id,
                                                           LabTrial.state.in_(OPEN_TRIALS)).values(
                        state="PRUNED", message="The task was stopped before this finished.", finished_at=utcnow()))
            await self._finish(task_id, "COMPLETE", "Stopped by you.", by_user=True)
            return "stopped"
        await self._set(task_id, stopping=True)
        return "stopping"

    async def _sim_ids(self, trials: list[LabTrial]) -> list[str | None]:
        ids = [t.candidate_id for t in trials if t.candidate_id]
        if not ids:
            return []
        async with db.session() as s:
            return list(await s.scalars(select(Candidate.sim_id).where(Candidate.id.in_(ids))))

    async def _unsend(self, task_id: int) -> None:
        """Take this task's queued (not yet sent) simulations back from the simulator: they wait as PARKED."""
        from ..engine.simulator import simulator
        await simulator.cancel_queued(task_id=core_key(task_id))
        async with db.session() as s:
            rows = (await s.execute(select(LabTrial.id, Candidate.status).join(
                Candidate, Candidate.id == LabTrial.candidate_id).where(
                LabTrial.task_id == task_id, LabTrial.state == "RUNNING"))).all()
        back = [tid for tid, status in rows if status in ("cancelled", "new", "approved")]
        if back:
            async with db.write() as s:
                await s.execute(update(LabTrial).where(LabTrial.id.in_(back)).values(
                    state="PARKED", candidate_id=None))

    async def resize(self, task_id: int, *, cores: int | None = None, max_trials: int | None = None) -> None:
        from ..engine.simulator import simulator
        t = await self.get(task_id)
        if t is None or t.status in ("COMPLETE",):
            raise ValueError("This task has finished.")
        values: dict[str, Any] = {}
        if cores is not None:
            cores = max(1, min(int(cores), self.lab(t.lab).max_cores, SIMULATION_SLOTS))
            if t.status == "RUNNING" and cores > t.cores:
                free = SIMULATION_SLOTS - await self._cores_used(exclude=task_id)
                if cores > free:
                    raise ValueError(f"Only {max(0, free - t.cores)} more cores are free.")
            values["cores"] = cores
            simulator.set_task_cores(core_key(task_id), cores)
        if max_trials is not None:
            spent = (await self.summary(task_id))["simulated"]
            if max_trials < spent:
                raise ValueError(f"This task has already used {spent} simulations.")
            values["max_trials"] = int(max_trials)
        if values:
            await self._set(task_id, **values)

    async def rename(self, task_id: int, name: str | None) -> None:
        await self._set(task_id, name=(name or "").strip()[:160] or None)

    async def delete(self, task_id: int) -> None:
        t = await self.get(task_id)
        if t is None:
            return
        if t.status in ("RUNNING", "QUEUED") or await self.trials(task_id, ("RUNNING",)):
            raise ValueError("Pause or stop the task before removing it: simulations are still out.")
        async with db.write() as s:
            await s.execute(LabTrial.__table__.delete().where(LabTrial.task_id == task_id))
            await s.execute(LabTask.__table__.delete().where(LabTask.id == task_id))
        if t.lab in self.labs:
            self.labs[t.lab].forget(task_id)

    async def _cores_used(self, exclude: int | None = None) -> int:
        async with db.session() as s:
            stmt = select(func.coalesce(func.sum(LabTask.cores), 0)).where(LabTask.status == "RUNNING")
            if exclude is not None:
                stmt = stmt.where(LabTask.id != exclude)
            return int(await s.scalar(stmt) or 0)

    async def start_waiting(self) -> None:
        """Start queued tasks, oldest first, whose cores fit in the free slots."""
        from ..engine.simulator import simulator
        async with self._lock:
            used = await self._cores_used()
            lending = await self.lending()
            if lending:
                used = await self._cores_sending()
            async with db.session() as s:
                waiting = list(await s.scalars(select(LabTask).where(LabTask.status == "QUEUED").order_by(
                    LabTask.queued_at, LabTask.id)))
            for t in waiting:
                if used + t.cores > SIMULATION_SLOTS:
                    continue
                used += t.cores
                async with db.write() as s:
                    await s.execute(update(LabTask).where(LabTask.id == t.id, LabTask.status == "QUEUED").values(
                        status="RUNNING", started_at=t.started_at or utcnow()))
                simulator.set_task_cores(core_key(t.id), t.cores)
                simulator.resume_task(core_key(t.id))
                bus.publish("labs", f"{self.lab(t.lab).title} task {t.name or t.id} started on {t.cores} cores.")
            self.lendable = max(0, SIMULATION_SLOTS - used) if lending else 0
        self.ensure_running()

    async def lending(self) -> bool:
        """Whether idle cores are lent (the consultant's preference; off by default)."""
        try:
            pref = await db.kv_get(PREF_KEY) or {}
        except Exception:
            pref = {}
        return bool(pref.get("lend_idle_cores", self.lend_idle_cores))

    async def _cores_sending(self) -> int:
        """Cores held by running tasks that still have simulations to send: short of their target or holding
        parked work. The rest only wait on what is out, which the simulator never takes back."""
        used = 0
        async with db.session() as s:
            running = list(await s.scalars(select(LabTask).where(LabTask.status == "RUNNING")))
        for t in running:
            if t.stopping:
                continue
            if await self._count(t.id, ("PARKED",)) or (await self._context(t)).spent < t.max_trials:
                used += t.cores
        return used

    # ---- the loop
    def ensure_running(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._loop(), name="lab-runner")

    async def restore(self) -> None:
        """After a restart: running tasks carry on from their rows (nothing to rebuild)."""
        async with db.session() as s:
            n = await s.scalar(select(func.count(LabTask.id)).where(LabTask.status.in_(["RUNNING", "QUEUED"])))
        async with db.write() as s:          # running tasks queue again so cores are handed out afresh
            await s.execute(update(LabTask).where(LabTask.status == "RUNNING").values(status="QUEUED"))
        if n:
            await self.start_waiting()

    async def _loop(self) -> None:
        while True:
            try:
                async with db.session() as s:
                    ids = list(await s.scalars(select(LabTask.id).where(LabTask.status == "RUNNING")))
                if not ids:
                    async with db.session() as s:
                        waiting = await s.scalar(select(func.count(LabTask.id)).where(LabTask.status == "QUEUED"))
                    if not waiting:
                        return
                    await self.start_waiting()
                for tid in ids:
                    await self.tick(tid)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("lab runner tick failed")
            await asyncio.sleep(POLL_S)

    async def tick(self, task_id: int) -> None:
        """Harvest what finished, then send what the task has room for. One tick per task at a time: two
        concurrent ticks would both send the same parked trials and pay for them twice."""
        lock = self._tick_locks.setdefault(task_id, asyncio.Lock())
        async with lock:
            await self._tick(task_id)

    async def _tick(self, task_id: int) -> None:
        t = await self.get(task_id)
        if t is None or t.status != "RUNNING":
            return
        lab = self.lab(t.lab)
        harvested = await self._harvest(t)
        if harvested:
            await lab.on_scored(t, harvested)
            t = await self.get(task_id)
        ctx = await self._context(t)
        running = await self._count(task_id, ("RUNNING",))
        room = (t.cores + self.lendable) * MULTI_SIMULATION_SIZE
        if not t.stopping and room - running > 0:
            # Ahead of the task's own budget: a passing Quick Alpha is submittable only once run in Full, and a task
            # that had spent its simulations would otherwise strand it.
            owed = [tr for tr in await self.trials(task_id, ("PARKED",))
                    if (tr.params or {}).get("fullOf") is not None][: room - running]
            if owed:
                await self._send(t, owed, ctx)
                return
        if t.stopping or ctx.spent >= t.max_trials:
            if not running:
                await self._finish(task_id, "COMPLETE", "Stopped by you." if t.stopping else None,
                                   by_user=t.stopping)
            return
        early = lab.done_early(t, ctx)
        if early:
            if not running:
                await self._finish(task_id, "COMPLETE", early)
            return
        if self.lendable:
            from ..engine.simulator import simulator
            simulator.set_task_cores(core_key(task_id), t.cores + self.lendable)
        want = min(room - running, t.max_trials - ctx.spent)
        want = want // MULTI_SIMULATION_SIZE * MULTI_SIMULATION_SIZE or (want if want < MULTI_SIMULATION_SIZE and
                                                                          running == 0 else 0)
        if want <= 0:
            return
        parked = await self.trials(task_id, ("PARKED",))
        if len(parked) < want:
            fresh = await lab.refill(t, want - len(parked), ctx)
            fresh = self._dedup(fresh, ctx)
            if fresh:
                await self.add_trials(task_id, fresh)
                parked = await self.trials(task_id, ("PARKED",))
        if not parked:
            if not running and not await lab.busy(t):
                await self._finish(task_id, "COMPLETE", lab.nothing_left(t))
            return
        if not await lab.send(t, parked[:want], ctx):
            await self._send(t, parked[:want], ctx)

    def _dedup(self, specs: list[TrialSpec], ctx: RefillContext) -> list[TrialSpec]:
        out, here = [], set()
        for sp in specs:
            if sp.state == "PRUNED":
                out.append(sp)
                continue
            key = trial_key(sp.expression, sp.settings)
            if key in here or key in ctx.seen:
                continue
            here.add(key)
            out.append(sp)
        return out

    async def _count(self, task_id: int, states: tuple[str, ...]) -> int:
        async with db.session() as s:
            return int(await s.scalar(select(func.count(LabTrial.id)).where(
                LabTrial.task_id == task_id, LabTrial.state.in_(states))) or 0)

    async def _context(self, t: LabTask) -> RefillContext:
        trials = await self.trials(t.id)
        seen: dict[str, float | bool] = {}
        refusals: set[tuple] = set()
        spent = 0
        for tr in trials:
            if tr.state == "PRUNED":
                continue
            key = trial_key(tr.expression, tr.settings)
            if tr.state == "COMPLETE":
                seen[key] = tr.value if tr.value is not None else FAILURE
            elif tr.state == "FAIL":
                seen[key] = False
                r = refusal_of(tr.settings, tr.message)
                if r:
                    refusals.add(r)
            else:
                seen[key] = True
            if tr.state in ("COMPLETE", "FAIL", "RUNNING") and not tr.free and not (tr.params or {}).get("fullOf"):
                spent += 1
        scored = sorted((tr for tr in trials if tr.state == "COMPLETE" and tr.value is not None),
                        key=lambda x: (-(x.value or FAILURE), x.number))
        return RefillContext(seen=seen, scored=scored, spent=spent, refusals=refusals)

    async def _send(self, t: LabTask, parked: list[LabTrial], ctx: RefillContext) -> None:
        from ..brain import brain
        from ..engine.simulator import CandidateSpec, simulator
        sendable: list[LabTrial] = []
        for tr in parked:
            withdrawn = brain.options.problems(tr.settings) if brain.options.live else []
            if withdrawn:
                async with db.write() as s:
                    await s.execute(update(LabTrial).where(LabTrial.id == tr.id).values(
                        state="PRUNED", finished_at=utcnow(), message=f"Withdrawn: {withdrawn[0]}"[:2000]))
                continue
            r = repeats_refusal(tr.settings, ctx.refusals)
            if r:
                async with db.write() as s:
                    await s.execute(update(LabTrial).where(LabTrial.id == tr.id).values(
                        state="FAIL", finished_at=utcnow(),
                        message=f"Not sent: BRAIN refused {r[3]} {json.loads(r[4])} in {r[0]} D{r[1]} {r[2]} "
                                "earlier in this task."))
                continue
            sendable.append(tr)
        if not sendable:
            return
        specs = [CandidateSpec(tr.expression, tr.settings, origin=f"lab:{t.lab}"[:32],
                               rationale={"lab_task": t.id, "trial": tr.number}) for tr in sendable]
        result = await simulator.prepare(specs, task_id=core_key(t.id))
        by_key: dict[str, Candidate] = {}
        for c in result.all:
            by_key.setdefault(trial_key(c.expression, c.settings), c)
        queue: list[int] = []
        async with db.write() as s:
            for tr in sendable:
                c = by_key.get(trial_key(tr.expression, tr.settings))
                row = await s.get(LabTrial, tr.id)
                if c is None:
                    row.state, row.finished_at = "PRUNED", utcnow()
                    row.message = "Left out: the same simulation is already in this batch."
                elif c.status == "invalid":
                    row.state, row.candidate_id, row.finished_at = "FAIL", c.id, utcnow()
                    row.message = f"Not valid here: {c.error}"[:2000]
                elif c.status == "reused":
                    row.state, row.candidate_id, row.alpha_id, row.free = "RUNNING", c.id, c.alpha_id, True
                    row.message = FREE_MESSAGE
                else:
                    row.state, row.candidate_id = "RUNNING", c.id
                    queue.append(c.id)
        if queue:
            await simulator.enqueue(queue)

    async def _harvest(self, t: LabTask) -> list[LabTrial]:
        """Score trials whose simulation finished (or was answered for free)."""
        async with db.session() as s:
            rows = (await s.execute(select(LabTrial, Candidate).join(
                Candidate, Candidate.id == LabTrial.candidate_id).where(
                LabTrial.task_id == t.id, LabTrial.state == "RUNNING",
                Candidate.status.in_(["done", "reused", "error", "invalid", "cancelled"])))).all()
            ids = [c.alpha_id for _, c in rows if c.alpha_id]
            alphas = {a.id: a for a in await s.scalars(select(Alpha).where(Alpha.id.in_(ids)))} if ids else {}
        done: list[LabTrial] = []
        owed: list[TrialSpec] = []
        best: tuple[float, str] | None = None
        async with db.write() as s:
            owing = {int((p or {}).get("fullOf")) for p in await s.scalars(select(LabTrial.params).where(
                LabTrial.task_id == t.id)) if (p or {}).get("fullOf") is not None} if rows else set()
            for tr, c in rows:
                row = await s.get(LabTrial, tr.id)
                alpha = alphas.get(c.alpha_id or "")
                row.finished_at = utcnow()
                if c.status in ("done", "reused") and alpha is not None:
                    value, penalty, implausible, figures = score(alpha, t.objective)
                    row.state, row.alpha_id, row.value, row.penalty = "COMPLETE", alpha.id, value, penalty
                    row.implausible, row.figures = implausible, figures
                    if value > FAILURE and (best is None or value > best[0]):
                        best = (value, alpha.id)
                    if passed_quick(alpha) and (tr.params or {}).get("fullOf") is None and tr.number not in owing:
                        owing.add(tr.number)
                        owed.append(TrialSpec(tr.expression, {**(tr.settings or {}), "simulationMode": FULL_MODE},
                                              params={"fullOf": tr.number}, generation=tr.generation,
                                              parent=tr.parent, message=PENDING_FULL))
                elif c.status == "cancelled":
                    row.state, row.message = "PRUNED", c.error or "Cancelled"
                else:
                    row.state = "FAIL"
                    row.message = (c.error or "Simulation failed")[:2000]
                done.append(row)
            if best is not None:
                task = await s.get(LabTask, t.id)
                if task.best_value is None or best[0] > task.best_value:
                    task.best_value, task.best_alpha_id = best
        if owed:
            await self.add_trials(t.id, owed)
        return done

    async def _finish(self, task_id: int, status: str, message: str | None = None, *, by_user: bool = False) -> None:
        from ..engine.simulator import simulator
        summary = await self.summary(task_id)
        async with db.session() as s:
            scored = await s.scalar(select(func.count(LabTrial.id)).where(
                LabTrial.task_id == task_id, LabTrial.state == "COMPLETE", LabTrial.generation > 0,
                LabTrial.value > FAILURE)) or 0
        if status == "COMPLETE" and not by_user and not scored and summary["failed"]:
            reasons = summary["reasons"]
            refused = next((r for r in reasons if "BRAIN rejected" in r[0] or "refused" in r[0]), None)
            reason = (refused or reasons[0])[0] if reasons else "they failed"
            status = "FAILED"
            message = f"None of its {summary['failed']} simulations scored. {reason}"
        await self._set(task_id, status=status, message=message, stopping=False, finished_at=utcnow())
        simulator.task_cores.pop(core_key(task_id), None)
        t = await self.get(task_id)
        bus.publish("labs", f"{self.lab(t.lab).title} task {t.name or t.id}: {status.lower()}"
                            + (f" — {message}" if message else "") + ".",
                    "success" if status == "COMPLETE" else "warning")
        await self.start_waiting()


runner = LabRunner()
