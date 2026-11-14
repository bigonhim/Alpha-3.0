"""Super Alpha Lab: SuperAlphas built from your own submitted Alphas, ready to submit (ported from alpha-harness
labs/super_alpha.py and api/super_lab.py).

A SuperAlpha is a selection expression, ranking every submitted ACTIVE Alpha of the region and delay and keeping the
top `selectionLimit`, and a combo expression, weighting the kept Alphas each day. Nothing here searches either
expression: SuperAlpha simulations are slow and BRAIN runs three at a time, so the lab crosses a small library of
economically grounded selections with proven weighting schemes, each at two decays to land Turnover inside the 2% to
40% a SuperAlpha must clear, and lets BRAIN's own checks say which are submittable.

Every selection caps each Alpha's operators at 8,000 / limit: BRAIN warns rather than fails past 8,000 operators in
all, and drops what does not fit.

Sending: AlphaGo's simulator builds REGULAR payloads only, so this lab sends its own SUPER simulations (one per
request, never batched with others) through `brain.sims.run` with `super_payload`, on the runner's trial rows. Each
send is a `Candidate` row (origin `lab:super`) the runner harvests like any other; its statuses `super_send` (POST
under way) and `super_run` (taken by BRAIN, polling) are this module's own, so the simulator never touches them.
After a restart a taken simulation is polled again; one whose POST may have been delivered is never resent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select, update

from ..brain.platform import (
    FULL_MODE,
    REGION_AGNOSTIC,
    SUPER_ALPHA_MIN_SELECTION,
    SUPER_ALPHA_OPERATOR_LIMIT,
    SUPER_ALPHA_SLOTS,
)
from ..db import db
from ..db.models import Candidate
from ..db.models_labs import LabTask, LabTrial
from ..events import bus
from .runner import Lab, RefillContext, core_key, repeats_refusal, utcnow

log = logging.getLogger(__name__)

#: Combo smoothing, each pairing at both: lighter keeps fast components' edge, heavier pulls an over-trading
#: combination under the 40% Turnover ceiling.
DECAYS = (4, 10)
#: Applied to the final weighted combination. Lower than an Alpha's usual 0.08: a SuperAlpha concentrates more
#: easily, and truncation is the documented brake on overfitting.
TRUNCATION = 0.05
MAX_LIMIT = 200
DEFAULT_LIMIT = 30
MAX_RECIPES = 50
ACTIVATIONS = {"BOTH": "IS and OS", "IS": "IS", "OS": "OS"}
PREF_KEY = "pref:super_recipes"
ORIGIN = "lab:super"
SENDING, TAKEN = "super_send", "super_run"
STARTER = {"selections": "(operator_count <= 100) * (1 - prod_correlation)",
           "combos": "stats = generate_stats(alpha);\nmax(ts_ir(stats.returns, 120), 0)"}
SINGULAR = {"selections": "selection", "combos": "combo"}

_inflight: dict[int, asyncio.Task] = {}      # candidate id -> its send / poll
_backoff_until = 0.0
_backoff_reason: str | None = None


def runner_hooks() -> bool:
    """True when the lab runner lets a lab send its own parked trials (`Lab.send`). The simulator cannot build a
    SUPER payload, so without the hook a Super Alpha task could not send anything and is refused at once."""
    return "send" in Lab.__dict__


@dataclass(frozen=True)
class Recipe:
    name: str
    code: str


def selections(limit: int) -> list[Recipe]:
    """Rankings over Alpha properties, each one keeping the pool diverse in its own way.

    `cap` bounds an Alpha's operators so the kept ones never pass 8,000 between them. Products of gates and scores:
    a gate that fails scores 0, which POSITIVE selection handling drops."""
    cap = SUPER_ALPHA_OPERATOR_LIMIT // max(int(limit), SUPER_ALPHA_MIN_SELECTION)
    budget = f"(operator_count <= {cap})"
    return [
        Recipe("Least correlated", f"{budget} * (self_correlation < 0.7) * (1 - prod_correlation)"),
        Recipe("Tradeable turnover", f"{budget} * (turnover >= 0.02) * (turnover < 0.35) * (1 - self_correlation)"),
        Recipe("Broad books", f"{budget} * (long_count + short_count) / sqrt(universe_size(universe))"),
        Recipe("Slow core", f"{budget} * (turnover < 0.25) / (turnover + 0.05)"),
        Recipe("Many sources", f"{budget} * (1 + datacategory_count) * (1 - prod_correlation)"),
    ]


#: Weighting schemes from BRAIN's own combo examples and their risk-aware variants.
ROLLING_IR = "stats = generate_stats(alpha);\nmax(ts_ir(stats.returns, 250), 0)"
#: The same ratio for an account without `ts_ir`, which BRAIN refuses there as an "inaccessible or unknown
#: operator". The guard follows Inverse volatility's.
ROLLING_IR_PORTABLE = ("stats = generate_stats(alpha);\n"
                       "max(ts_mean(stats.returns, 250) / (ts_std_dev(stats.returns, 250) + 0.0001), 0)")
COMBOS = [
    Recipe("Equal weight", "1"),
    Recipe("Rolling IR", ROLLING_IR),
    Recipe("Least crowded",
           "stats = generate_stats(alpha);\n"
           "innerCorr = self_corr(stats.returns, 500);\n"
           "ic = if_else(innerCorr == 1.0, nan, innerCorr);\n"
           "maxCorr = reduce_max(ic);\n"
           "max(1 - maxCorr, 0.05)"),
    Recipe("Inverse volatility", "stats = generate_stats(alpha);\n1 / (ts_std_dev(stats.returns, 120) + 0.0001)"),
]


def combos_for(operators: list[dict[str, Any]]) -> list[Recipe]:
    """The built-in combos as this account can run them. Rolling IR keeps `ts_ir` where the account has it as a
    COMBO operator, so its simulations still match the ones already run; elsewhere, and when the operators could
    not be read, it is written without."""
    from .power_pool_lab import operator_table
    if operators and "ts_ir" in operator_table(operators, "COMBO"):
        return list(COMBOS)
    return [Recipe(r.name, ROLLING_IR_PORTABLE) if r.code == ROLLING_IR else r for r in COMBOS]


def activations_of(choice: str) -> list[str]:
    return ["IS", "OS"] if str(choice).upper() == "BOTH" else [str(choice).upper()]


def super_settings(region: str, delay: int, universe: str, neutralization: str, decay: int, selection_limit: int,
                   activation: str) -> dict[str, Any]:
    """A Full SUPER simulation: only a Full one can be submitted, and Quick runs none of the correlation checks a
    SuperAlpha must also pass."""
    return {"instrumentType": "EQUITY", "region": region, "delay": int(delay), "universe": universe,
            "neutralization": neutralization, "decay": int(decay), "truncation": TRUNCATION,
            "selectionHandling": "POSITIVE", "selectionLimit": int(selection_limit),
            "componentActivation": activation, "simulationMode": FULL_MODE, "language": "FASTEXPR"}


@dataclass(frozen=True)
class SuperCandidate:
    selection: Recipe
    combo: Recipe
    settings: dict[str, Any]

    @property
    def name(self) -> str:
        return f"{self.selection.name} · {self.combo.name}"

    @property
    def label(self) -> str:
        """The trial's expression line: names the recipe, the line a results table shows."""
        return f"{self.name} · Decay {self.settings['decay']} · {self.settings['componentActivation']}"


def candidates(*, region: str, delay: int, universe: str, neutralization: str, selection_limit: int,
               activations: list[str], chosen_selections: list[Recipe] | None = None,
               combos: list[Recipe] | None = None) -> list[SuperCandidate]:
    """Every selection · combo · decay · activation, as Full SUPER simulations; the built-in selections and combos
    unless others are given."""
    out: list[SuperCandidate] = []
    for activation in activations:
        for decay in DECAYS:
            for selection in chosen_selections or selections(selection_limit):
                for combo in combos or COMBOS:
                    out.append(SuperCandidate(selection, combo, super_settings(
                        region, delay, universe, neutralization, decay, selection_limit, activation)))
    return out


def recipe_problem(kind: str, chosen: list[Recipe]) -> str | None:
    """Why a list cannot run as ticked: BRAIN needs both halves, and results are named for them, so a blank one or
    two of one name would read as one."""
    word = SINGULAR[kind]
    if not chosen:
        return f"Tick at least one {word}."
    if len(chosen) > MAX_RECIPES:
        return f"Tick at most {MAX_RECIPES} {word}s."
    if any(not r.name or not r.code for r in chosen):
        return f"Every ticked {word} needs a name and code."
    if any(len(r.name) > 120 or len(r.code) > 8000 for r in chosen):
        return f"A {word} name is at most 120 characters and its code at most 8,000."
    names = [r.name for r in chosen]
    if len(set(names)) < len(names):
        return f"Two ticked {word}s share a name: rename one."
    return None


def _recipes(given: Any, default: list[Recipe]) -> list[Recipe]:
    if given is None:
        return default
    return [Recipe(str(r.get("name") or "").strip(), str(r.get("code") or "").strip()) for r in given]


# ---------------------------------------------------------------- the user's own recipes

def empty_prefs() -> dict[str, Any]:
    return {"selections": [], "combos": [], "off": []}


async def load_prefs() -> dict[str, Any]:
    """Your own selections and combos, and which of every one are unticked (`builtin:{name}`, `custom:{id}`)."""
    stored = await db.kv_get(PREF_KEY) or {}
    prefs = empty_prefs()
    for kind in ("selections", "combos"):
        prefs[kind] = [{"id": str(r.get("id") or uuid.uuid4().hex), "name": str(r.get("name") or ""),
                        "code": str(r.get("code") or "")} for r in stored.get(kind) or [] if isinstance(r, dict)]
    prefs["off"] = [str(k) for k in stored.get("off") or []]
    return prefs


async def save_prefs(prefs: dict[str, Any]) -> None:
    await db.kv_set(PREF_KEY, {"selections": prefs.get("selections") or [], "combos": prefs.get("combos") or [],
                               "off": prefs.get("off") or []})


def add_own(prefs: dict[str, Any], kind: str, name: str | None = None, code: str | None = None) -> str:
    rid = uuid.uuid4().hex
    own = prefs.setdefault(kind, [])
    own.append({"id": rid, "name": name or f"My {SINGULAR[kind].title()} {len(own) + 1}",
                "code": code if code is not None else STARTER[kind]})
    return rid


def chosen(builtins: list[Recipe], own: list[dict[str, Any]], off: list[str]) -> list[Recipe]:
    """What a task crosses of one kind: the built-in ones and your own, as ticked."""
    return [*(r for r in builtins if f"builtin:{r.name}" not in off),
            *(Recipe(str(r["name"]).strip(), str(r["code"]).strip()) for r in own if f"custom:{r['id']}" not in off)]


async def account_operators() -> list[dict[str, Any]]:
    from .power_pool_lab import synced_operators
    return await synced_operators()


async def plan_summary(params: dict[str, Any]) -> dict[str, Any]:
    """What a task with these settings would run, without running anything; lists every built-in selection and
    combo whatever was chosen, so the form can offer them all."""
    limit = _limit(params)
    activations = activations_of(params.get("activation") or "BOTH")
    combos = combos_for(await account_operators())
    builtin_sel = selections(limit if limit is not None else SUPER_ALPHA_MIN_SELECTION)
    picked_sel = _recipes(params.get("selections"), builtin_sel)
    picked_combo = _recipes(params.get("combos"), combos)
    return {"selections": builtin_sel, "combos": combos, "decays": list(DECAYS), "truncation": TRUNCATION,
            "simulations": len(picked_sel) * len(picked_combo) * len(DECAYS) * len(activations)}


def _limit(params: dict[str, Any]) -> int | None:
    """The selection limit when it is a whole number from 10 to 200 (left out: 30), else None."""
    raw = params.get("selection_limit", DEFAULT_LIMIT)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not value.is_integer():
        return None
    value = int(value)
    return value if SUPER_ALPHA_MIN_SELECTION <= value <= MAX_LIMIT else None


async def build(params: dict[str, Any]) -> tuple[list[SuperCandidate], list[str]]:
    """The candidate grid for a form, and the problems that refuse it."""
    problems: list[str] = []
    region = str(params.get("region") or "")
    if region == REGION_AGNOSTIC:
        problems.append("SuperAlphas combine Alphas of one region. Pick a region other than All Regions.")
    if not region:
        problems.append("Choose a region.")
    limit = _limit(params)
    if limit is None:
        problems.append(f"Selection Limit must be a whole number from {SUPER_ALPHA_MIN_SELECTION} to {MAX_LIMIT}.")
    choice = str(params.get("activation") or "BOTH").upper()
    if choice not in ACTIVATIONS:
        problems.append("Component Activation is IS, OS or both.")
    if not params.get("neutralization"):
        problems.append("Choose a neutralization.")
    if not params.get("universe"):
        problems.append("Choose a universe.")
    sel = _recipes(params.get("selections"), selections(limit or SUPER_ALPHA_MIN_SELECTION))
    com = _recipes(params.get("combos"), combos_for(await account_operators()))
    for kind, picked in (("selections", sel), ("combos", com)):
        problem = recipe_problem(kind, picked)
        if problem:
            problems.append(problem)
    if problems:
        return [], problems
    found = candidates(region=region, delay=int(params.get("delay") or 0), universe=str(params["universe"]),
                       neutralization=str(params["neutralization"]), selection_limit=int(limit),
                       activations=activations_of(choice), chosen_selections=sel, combos=com)
    return found, []


# ---------------------------------------------------------------- the lab

class SuperLab(Lab):
    name = "super"
    title = "Super Alpha Lab"
    objective = "sharpe"
    max_cores = SUPER_ALPHA_SLOTS

    async def preview(self, params: dict[str, Any]) -> dict[str, Any]:
        found, problems = await build(params)
        if not problems and not runner_hooks():
            problems.append(NO_HOOK)
        return {"problems": problems, "warnings": [], "combinations": len(found),
                "sample": [{"expression": c.label, "settings": c.settings, "selection": c.selection.code,
                            "combo": c.combo.code} for c in found[:5]]}

    async def plan(self, task: LabTask) -> dict[str, Any]:
        if not runner_hooks():
            raise ValueError(NO_HOOK)
        found, problems = await build(dict(task.params or {}))
        if problems:
            raise ValueError(problems[0])
        task.max_trials = len(found)          # every pairing is written up front and each is sent once
        params = task.params or {}
        return {"recipes": sorted({c.name for c in found}), "count": len(found),
                "activation": activations_of(params.get("activation") or "BOTH"),
                "selection_limit": int(params.get("selection_limit") or DEFAULT_LIMIT),
                "neutralization": params.get("neutralization"), "decays": list(DECAYS), "truncation": TRUNCATION}

    async def initial(self, task: LabTask) -> list[Any]:
        from .runner import TrialSpec
        found, _ = await build(dict(task.params or {}))
        return [TrialSpec(c.label, c.settings, params={"recipe": c.name, "selection": c.selection.code,
                                                       "combo": c.combo.code}) for c in found]

    async def send(self, task: LabTask, trials: list[LabTrial], ctx: RefillContext) -> bool:
        """Send parked SuperAlphas ourselves, one per request, at most the task's cores at a time and three across
        the app (BRAIN's limit). The rest stay parked for the next tick."""
        await send_parked(task, trials, ctx)
        return True


NO_HOOK = ("This version of the lab runner cannot send SuperAlpha simulations (it has no Lab.send hook), so the task "
           "was not added. Update AlphaGo.")


# ---------------------------------------------------------------- sending SUPER simulations

async def send_parked(task: LabTask, trials: list[LabTrial], ctx: RefillContext) -> int:
    from ..brain import brain
    from ..config import settings
    global _backoff_until, _backoff_reason
    start_recovery()
    if not trials or not brain.client.logged_in or time.monotonic() < _backoff_until:
        return 0
    async with db.session() as s:
        out = int(await s.scalar(select(func.count(LabTrial.id)).where(
            LabTrial.task_id == task.id, LabTrial.state == "RUNNING")) or 0)
    live = sum(1 for t in _inflight.values() if not t.done())
    room = min(int(task.cores) - out, SUPER_ALPHA_SLOTS - live)
    sent = 0
    for tr in trials:
        if room <= 0:
            break
        refused = repeats_refusal(tr.settings, ctx.refusals)
        if refused:
            async with db.write() as s:
                await s.execute(update(LabTrial).where(LabTrial.id == tr.id).values(
                    state="FAIL", finished_at=utcnow(),
                    message=f"Not sent: BRAIN refused {refused[3]} {json.loads(refused[4])} in {refused[0]} "
                            f"D{refused[1]} {refused[2]} earlier in this task."))
            continue
        if not brain.client.quota.can_spend(1, settings.brain.quota_reserve):
            _backoff_until = time.monotonic() + 600
            _backoff_reason = "the day's simulation quota is down to its reserve"
            break
        params = tr.params or {}
        async with db.write() as s:
            cand = Candidate(expression=tr.expression, settings=tr.settings, settings_hash=tr.settings_hash or "",
                             region=tr.settings.get("region") or task.region or "USA",
                             delay=int(tr.settings.get("delay") or 0), origin=ORIGIN, status=SENDING,
                             task_id=core_key(task.id), sent_at=utcnow(),
                             rationale={"lab_task": task.id, "trial": tr.number, "selection": params.get("selection"),
                                        "combo": params.get("combo"), "type": "SUPER"})
            s.add(cand)
            await s.flush()
            row = await s.get(LabTrial, tr.id)
            row.state, row.candidate_id = "RUNNING", cand.id
            cid = cand.id
        _inflight[cid] = asyncio.get_running_loop().create_task(
            _run_one(cid, tr.id, str(params.get("selection") or ""), str(params.get("combo") or "1"),
                     dict(tr.settings)), name=f"super-{cid}")
        room -= 1
        sent += 1
    return sent


async def _set_candidate(cid: int, **values: Any) -> None:
    async with db.write() as s:
        await s.execute(update(Candidate).where(Candidate.id == cid).values(**values))


async def _unsend(cid: int, trial_id: int, reason: str) -> None:
    """Nothing reached BRAIN: the trial waits again as PARKED, and the candidate is closed."""
    async with db.write() as s:
        await s.execute(update(Candidate).where(Candidate.id == cid).values(
            status="cancelled", error=f"Not sent: {reason}", finished_at=utcnow()))
        await s.execute(update(LabTrial).where(LabTrial.id == trial_id, LabTrial.state == "RUNNING").values(
            state="PARKED", candidate_id=None))


async def _run_one(cid: int, trial_id: int, selection: str, combo: str, settings: dict[str, Any]) -> None:
    from ..brain import brain
    from ..brain.errors import (
        BrainError,
        ConcurrencyLimit,
        MaybeDelivered,
        QuotaExhausted,
        SimulationRejected,
        StartThrottled,
    )
    from ..brain.simulations import super_payload
    global _backoff_until, _backoff_reason
    variant = await db.kv_get("super_payload_variant", "top")
    shapes = [variant, "nested" if variant == "top" else "top"]
    limit = int(settings.get("selectionLimit") or DEFAULT_LIMIT)
    handling = str(settings.get("selectionHandling") or "POSITIVE")
    refusals: list[str] = []

    async def started(location: str) -> None:
        sim_id = location.rstrip("/").rsplit("/", 1)[-1]
        await _set_candidate(cid, status=TAKEN, sim_id=sim_id)

    try:
        outcome = None
        for shape in shapes:
            payload = super_payload(selection, combo, settings, selection_limit=limit, selection_handling=handling,
                                    nested=shape == "nested")
            try:
                outcome = (await brain.sims.run([payload], on_started=started))[0]
            except SimulationRejected as exc:          # refused before BRAIN took it: the other shape may pass
                refusals.append(exc.message)
                continue
            await db.kv_set("super_payload_variant", shape)
            break
        if outcome is None:
            await _fail(cid, f"BRAIN rejected: {refusals[0] if refusals else 'the SuperAlpha'}")
            return
        await _finish(cid, outcome.alpha_id, outcome.message or outcome.status)
    except QuotaExhausted as exc:
        _backoff_until, _backoff_reason = time.monotonic() + 3600, "BRAIN's daily simulation limit is reached"
        await _unsend(cid, trial_id, exc.message)
    except (ConcurrencyLimit, StartThrottled) as exc:
        _backoff_until, _backoff_reason = time.monotonic() + 60, "BRAIN is busy"
        await _unsend(cid, trial_id, exc.message)
    except MaybeDelivered as exc:
        await _fail(cid, f"{exc.message} Look for it in Alphas before running this pairing again.")
    except BrainError as exc:
        await _fail(cid, exc.message or "SuperAlpha simulation failed")
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a send never leaves its trial stuck
        log.exception("SuperAlpha send %s failed", cid)
        await _fail(cid, f"SuperAlpha simulation failed: {exc}")
    finally:
        _inflight.pop(cid, None)


async def _fail(cid: int, message: str) -> None:
    await _set_candidate(cid, status="error", error=message[:2000], finished_at=utcnow())


async def _finish(cid: int, alpha_id: str | None, message: str | None) -> None:
    """Store the SuperAlpha and mark its candidate done (the runner scores the trial on its next tick)."""
    from ..brain import brain
    from ..brain.errors import BrainError
    from ..engine.ingest import upsert_alpha
    if not alpha_id:
        await _fail(cid, message or "SuperAlpha simulation failed")
        return
    try:
        data = await brain.alphas.get(alpha_id)
        await upsert_alpha(data, candidate_id=cid, origin="app")
    except BrainError as exc:
        await _set_candidate(cid, status="error", alpha_id=alpha_id, finished_at=utcnow(),
                             error=f"Simulated as {alpha_id}, but reading it failed: {exc.message}")
        return
    await _set_candidate(cid, status="done", alpha_id=alpha_id, finished_at=utcnow(), error=None)
    bus.publish("labs", f"SuperAlpha simulated: {alpha_id}.", "success")


# ---------------------------------------------------------------- after a restart

_recovered = False


def start_recovery() -> None:
    """Recover once per run (see `recover`); a no-op outside a running event loop."""
    global _recovered
    if _recovered:
        return
    _recovered = True
    try:
        asyncio.get_running_loop().create_task(recover(), name="super-recover")
    except RuntimeError:
        _recovered = False


async def recover() -> dict[str, int]:
    """SuperAlphas that were out when the app closed: one BRAIN took (its id is known) is polled again; one whose
    POST was under way may have been delivered, so it is closed with a note instead of being paid for twice."""
    async with db.session() as s:
        rows = list(await s.scalars(select(Candidate).where(Candidate.origin == ORIGIN,
                                                           Candidate.status.in_((SENDING, TAKEN)))))
    counts = {"resumed": 0, "closed": 0}
    for c in rows:
        if c.id in _inflight:
            continue
        if c.status == TAKEN and c.sim_id:
            _inflight[c.id] = asyncio.get_running_loop().create_task(_resume(c.id, c.sim_id), name=f"super-{c.id}")
            counts["resumed"] += 1
        else:
            await _fail(c.id, "The app closed while this SuperAlpha was being sent. BRAIN may have received it, so "
                              "it is not resent; look for it in Alphas.")
            counts["closed"] += 1
    return counts


async def _resume(cid: int, sim_id: str) -> None:
    from ..brain import brain
    from ..brain.errors import BrainError
    try:
        while not brain.client.logged_in:
            await asyncio.sleep(5)
        outcome = (await brain.sims.wait(f"/simulations/{sim_id}", 1))[0]
        await _finish(cid, outcome.alpha_id, outcome.message or outcome.status)
    except BrainError as exc:
        await _fail(cid, f"Lost SuperAlpha simulation {sim_id}: {exc.message}")
    finally:
        _inflight.pop(cid, None)


def backoff() -> str | None:
    """Why SuperAlpha sends are waiting, for the tab."""
    return _backoff_reason if time.monotonic() < _backoff_until else None
