"""Optuna TPE, driven by ask/tell in rounds that fill whole batches (alpha-harness labs/study.py and scheduler.py).

``study.optimize`` evaluates one trial at a time; here a trial takes minutes on BRAIN and eighty run at once, so the
study is driven by ask/tell: ask N points -> write N trials -> the runner sends them -> harvested -> tell -> ask.

Restart safety: the study is never persisted. It is rebuilt from the task's trial rows (each stores the names it
was asked under, ``_asked``; every distribution is categorical over the frozen space, so it is rebuilt from the
space) and replayed with ``add_trial`` of a FrozenTrial. A trial asked before a restart has no live Optuna trial;
when it finishes it is added as an already-finished trial instead. Past TRIM_AT trials a study keeps the 25 best and
the 5,000 latest: each trial more costs every suggestion time, and TPE weighs older ones down anyway.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol, Sequence

from sqlalchemy import select

from ..db import db
from ..db.models_labs import LabTask, LabTrial
from .launch import settings_problems
from .objectives import searched
from .runner import FAILURE, RefillContext, TrialSpec, repeats_refusal, trial_key

log = logging.getLogger(__name__)

KEEP_BEST = 25
KEEP_LATEST = 5_000
TRIM_AT = 6_000
#: The states TPE learns from, and counts toward its random start.
LEARNED = frozenset({"COMPLETE", "PRUNED"})
DROPPED = "dropped"
ASKED = "_asked"
TOKEN = "_t"


class SpaceLab(Protocol):
    """What a lab searched with TPE provides (Search and Template)."""

    def choices_of(self, space: dict[str, Any], name: str) -> tuple[Any, ...] | None: ...
    def coverage(self, space: dict[str, Any]) -> tuple[str, list[str]]: ...
    def first_pass(self, space: dict[str, Any], field_id: str) -> dict[str, Any]: ...
    def draw(self, trial: Any, task: LabTask) -> tuple[dict[str, Any], TrialSpec | None]: ...


class _Shared(dict):
    """A replayed trial's distributions: Optuna deep-copies every trial added, and a field slot holds thousands of
    choices that never change, so the copy shares them."""

    def __deepcopy__(self, _memo: dict[int, Any]) -> dict[str, Any]:
        return dict(self)


def _frozen(state: Any, params: dict[str, Any], distributions: dict[str, Any], values: list[float] | None) -> Any:
    """A trial already told. `create_trial` makes the same one, but first compares each distribution with itself
    choice by choice: 9 ms a trial for a field slot."""
    from optuna.trial import FrozenTrial
    now = datetime.now()  # noqa: DTZ005 - naive, as Optuna dates its own trials
    return FrozenTrial(number=-1, state=state, value=None, values=values, datetime_start=now, datetime_complete=now,
                       params=params, distributions=_Shared(distributions), user_attrs={}, system_attrs={},
                       intermediate_values={}, trial_id=-1)


class Distributions:
    """Each asked name's categorical distribution over the frozen space, made once and shared."""

    def __init__(self, lab: SpaceLab, space: dict[str, Any]) -> None:
        self.lab, self.space = lab, space
        self._made: dict[str, Any] = {}

    def load(self, names: Sequence[str]) -> dict[str, Any] | None:
        from optuna.distributions import CategoricalDistribution
        out: dict[str, Any] = {}
        for name in names:
            if name not in self._made:
                choices = self.lab.choices_of(self.space, name)
                if not choices:
                    return None
                self._made[name] = CategoricalDistribution(tuple(choices))
            out[name] = self._made[name]
        return out


def optuna_params(params: dict[str, Any], distributions: dict[str, Any]) -> dict[str, Any] | None:
    """A stored trial's values under the names Optuna asked them by.

    The labs store what a point means (``field``); Optuna names the slot it was drawn from (``field@TOP3000``)."""
    found: dict[str, Any] = {}
    for name, dist in distributions.items():
        key = name if name in params else name.rsplit("@", 1)[0]
        if key not in params or params[key] not in dist.choices:
            return None
        found[name] = params[key]
    return found


def kept(trials: Sequence[tuple[str, float | None]]) -> list[int]:
    """Which of a study's trials, as (state, value told) in order, a trim keeps: the KEEP_LATEST latest learned and
    the KEEP_BEST best Complete (highest first, the earlier on a tie). A FAILURE is never a best."""
    learned = [i for i, (state, _) in enumerate(trials) if state in LEARNED]
    scored = [(value, i) for i, (state, value) in enumerate(trials)
              if state == "COMPLETE" and value is not None and value > FAILURE]
    best = [i for _, i in sorted(scored, key=lambda pair: -pair[0])[:KEEP_BEST]]
    return sorted({*learned[-KEEP_LATEST:], *best})


def _sampler(startup: int, dropped: int) -> Any:
    from optuna.exceptions import ExperimentalWarning
    from optuna.samplers import TPESampler
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        return TPESampler(n_startup_trials=max(0, startup - dropped), multivariate=True, group=True,
                          constant_liar=True)


def _new_study(startup: int, dropped: int) -> Any:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study = optuna.create_study(direction="maximize", sampler=_sampler(startup, dropped))
    study.set_user_attr(DROPPED, dropped)
    return study


def trimmed(study: Any, startup: int) -> Any:
    """``study`` cut to what :func:`kept` keeps; what it dropped still counts toward the random start."""
    trials = study.get_trials(deepcopy=False)
    keep = kept([(t.state.name, t.value) for t in trials])
    learned = sum(t.state.name in LEARNED for t in trials)
    dropped = int(study.user_attrs.get(DROPPED, 0)) + learned - len(keep)
    out = _new_study(startup, dropped)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in keep:
            t = trials[i]
            out.add_trial(_frozen(t.state, t.params, t.distributions, t.values))
    return out


def rebuild(history: list[dict[str, Any]], loader: Distributions, startup: int) -> Any:
    """Recreate a study from its finished trials' rows ({state, params, value, implausible}), trimmed if outgrown.

    Only terminal trials asked by the search are replayed: a Full re-run or a seed asked it nothing."""
    from optuna.trial import TrialState
    study = _new_study(startup, 0)
    states = {"COMPLETE": TrialState.COMPLETE, "FAIL": TrialState.FAIL, "PRUNED": TrialState.PRUNED}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for entry in history:
            state = states.get(str(entry["state"]))
            asked = (entry.get("params") or {}).get(ASKED)
            if state is None or not asked:
                continue
            distributions = loader.load(asked)
            if distributions is None:
                continue
            params = optuna_params(entry["params"], distributions)
            if params is None:
                continue
            values = [searched(entry.get("value"), bool(entry.get("implausible")))] \
                if state is TrialState.COMPLETE else None
            try:
                study.add_trial(_frozen(state, params, distributions, values))
            except ValueError:
                log.warning("replay skipped a trial of %s", sorted(params))
    return trimmed(study, startup) if len(study.get_trials(deepcopy=False)) > TRIM_AT else study


@dataclass
class _Run:
    study: Any
    loader: Distributions
    startup: int
    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    live: dict[str, Any] = field(default_factory=dict)
    synced_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Driver:
    """The live TPE studies of one lab, by task id."""

    def __init__(self, lab: SpaceLab) -> None:
        self.lab = lab
        self.runs: dict[int, _Run] = {}

    def forget(self, task_id: int) -> None:
        self.runs.pop(task_id, None)

    async def run_for(self, task: LabTask) -> _Run:
        run = self.runs.get(task.id)
        if run is not None and run.loader.space is not task.space and run.loader.space != task.space:
            run = None
        if run is None:
            async with db.session() as s:
                rows = (await s.execute(select(LabTrial.state, LabTrial.params, LabTrial.value, LabTrial.implausible)
                                        .where(LabTrial.task_id == task.id).order_by(LabTrial.number))).all()
            history = [{"state": st, "params": p or {}, "value": v, "implausible": imp} for st, p, v, imp in rows]
            loader = Distributions(self.lab, task.space or {})
            startup = int((task.params or {}).get("n_startup_trials") or 20)
            run = _Run(study=rebuild(history, loader, startup), loader=loader, startup=startup)
            self.runs[task.id] = run
        return run

    def tell(self, run: _Run, trials: Sequence[LabTrial]) -> int:
        """Report finished trials: a live one is told directly, one asked before a restart is replayed."""
        from optuna.trial import TrialState
        told = 0
        for tr in trials:
            params = tr.params or {}
            token = params.get(TOKEN)
            live = run.live.pop(token, None) if token else None
            try:
                if live is not None:
                    if tr.state == "COMPLETE":
                        run.study.tell(live, searched(tr.value, tr.implausible), skip_if_finished=True)
                    elif tr.state == "FAIL":
                        run.study.tell(live, state=TrialState.FAIL, skip_if_finished=True)
                    else:
                        run.study.tell(live, state=TrialState.PRUNED, skip_if_finished=True)
                    told += 1
                elif tr.state == "COMPLETE" and params.get(ASKED) and token and not token.startswith(run.uid):
                    distributions = run.loader.load(params[ASKED])
                    found = optuna_params(params, distributions) if distributions else None
                    if found is not None:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            run.study.add_trial(_frozen(TrialState.COMPLETE, found, distributions,
                                                        [searched(tr.value, tr.implausible)]))
                        told += 1
            except Exception:
                log.warning("telling trial %s failed", tr.number, exc_info=True)
        return told

    async def tell_finished(self, task: LabTask, trials: Sequence[LabTrial]) -> int:
        return self.tell(await self.run_for(task), trials)

    async def sync(self, task: LabTask) -> None:
        """Tell trials that ended without being harvested: pruned by a stop, left out of a batch, refused unsent."""
        run = await self.run_for(task)
        if not run.live:
            return
        since = run.synced_at - timedelta(seconds=5)
        run.synced_at = datetime.now(timezone.utc)
        async with db.session() as s:
            rows = list(await s.scalars(select(LabTrial).where(
                LabTrial.task_id == task.id, LabTrial.state.in_(["PRUNED", "FAIL", "COMPLETE"]),
                LabTrial.finished_at >= since)))
        self.tell(run, [r for r in rows if (r.params or {}).get(TOKEN) in run.live])

    def ask(self, run: _Run, task: LabTask, want: int, ctx: RefillContext, cover: list[str],
            problems: Callable[[dict[str, Any]], list[str]] = settings_problems) -> tuple[list[TrialSpec], str | None]:
        """Ask for `want` new simulations (runs off the event loop).

        A point this task already scored is answered with its score, one that failed is told as failed, and a
        repeat, or a point its lab cannot write, is pruned; each time another point is asked, so quota is only
        spent on Alphas the task has not seen. So is a point repeating a setting BRAIN refused in its market earlier
        in the task, or one BRAIN no longer accepts. Returns the first such setting, so a task left with nothing to
        ask can say whether BRAIN withdrew what it searches or the search ran out of new points."""
        from optuna.trial import TrialState
        if len(run.study.get_trials(deepcopy=False)) > TRIM_AT and not run.live:
            run.study = trimmed(run.study, run.startup)
        study = run.study
        if cover and not study.get_trials(deepcopy=False, states=(TrialState.WAITING,)):
            for field_id in cover:
                study.enqueue_trial(self.lab.first_pass(task.space, field_id))
        picked: list[TrialSpec] = []
        keys: set[str] = set()
        withdrawn: str | None = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(want * 5):
                if len(picked) >= want:
                    break
                trial = study.ask()
                try:
                    params, spec = self.lab.draw(trial, task)
                except Exception as exc:          # a point the lab cannot write costs one draw, never the task
                    log.warning("draw failed: %s", exc)
                    study.tell(trial, state=TrialState.PRUNED)
                    continue
                if spec is not None:
                    found = problems(spec.settings)
                    if found:
                        withdrawn = withdrawn or found[0]
                        study.tell(trial, state=TrialState.PRUNED)
                        continue
                    hit = repeats_refusal(spec.settings, ctx.refusals)
                    if hit:
                        withdrawn = withdrawn or f"{hit[3]} {hit[4]} in {hit[0]} D{hit[1]} {hit[2]}"
                        study.tell(trial, state=TrialState.PRUNED)
                        continue
                key = trial_key(spec.expression, spec.settings) if spec is not None else None
                if spec is None or key in keys or key in ctx.seen:
                    known = ctx.seen.get(key) if key else None
                    if known is False:
                        study.tell(trial, state=TrialState.FAIL)
                    elif isinstance(known, float):
                        study.tell(trial, searched(known))
                    else:
                        study.tell(trial, state=TrialState.PRUNED)
                    continue
                keys.add(key)
                token = f"{run.uid}:{trial.number}"
                spec.params = {**params, ASKED: list(trial.distributions), TOKEN: token}
                run.live[token] = trial
                picked.append(spec)
        return picked, withdrawn


def tried(trials: Sequence[LabTrial], key: str) -> set[Any]:
    """The values of `key` (the field the first pass covers) among unpruned trials."""
    return {(t.params or {}).get(key) for t in trials if t.state != "PRUNED"}


def cover_for(lab: SpaceLab, task: LabTask, want: int, ctx: RefillContext, all_trials: Sequence[LabTrial]) -> list[str]:
    """First pass: every field one trial, budget permitting (up to half the task)."""
    key, coverable = lab.coverage(task.space or {})
    if not key:
        return []
    limit = task.max_trials // 2
    done = tried(all_trials, key)
    if len(done) >= limit:
        return []
    untried = [f for f in coverable if f not in done]
    return untried[: min(want, limit - len(done))]


def nothing_new(withdrawn: str | None) -> str:
    if withdrawn:
        return f"Stopped early: BRAIN no longer accepts a setting this task searches ({withdrawn})."
    return "Stopped early: every point the search drew had been simulated already."
