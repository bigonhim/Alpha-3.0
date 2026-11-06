"""Research tasks: large, systematic searches with their own share of BRAIN's concurrent simulation slots.

Kinds
- single       "Search Lab": first-order templates over every field of one dataset, then group / event variants of
               the best results.
- pair         "Double dataset · Pair": two-field templates combining a field of dataset A with one of dataset B.
- interaction  field × field templates inside one dataset, plus each field against price/volume.
- repair       "Repair near misses": no new ideas. BRAIN's check on strong alphas you already have, then targeted
               repairs of the near misses in scope, closest to passing first, until its budget ends. Runs only when
               you start it; the Autopilot never plans it.

Field screening (settings.research.field_screening): a search task first spends one simulation per field (its
time-series rank over the data category's main horizon), keeps the fields that showed a signal in either direction
(a negative one is used sign-flipped) and spends the rest of its budget on those fields only. On a typical dataset
most fields carry no usable signal, so this multiplies the share of simulations that reach "promising" or better.
Templates are tried in the order your own history ranks their families (engine/families.py).

A task runs in waves: generate → validate / de-duplicate (skipping shapes this market has already answered) →
simulate on its cores → optimise near misses → BRAIN submission check on strong alphas → robustness gate.
Submission is never automatic.

Modes (settings.research.mode)
- manual     the Autopilot never starts anything; you create and run tasks.
- assisted   the Autopilot proposes tasks (with a reason); each one waits for your approval.
- autopilot  proposed tasks start by themselves within the daily Autopilot cap (opt-in, behind a warning).
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update

from ..brain import brain
from ..brain.quota import eastern_day
from ..brain.simulations import normalize_settings, settings_hash
from ..config import settings
from ..db import db
from ..db.models import Alpha, Candidate, Dataset, DatasetRating, ResearchTask
from ..events import bus
from ..knowledge.templates import (
    ENUM_WINDOWS,
    PAIR_TEMPLATES,
    PV_PARTNERS,
    TEMPLATES,
    enumerate_group,
    enumerate_pairs,
    enumerate_single,
    probe_expression,
    second_order,
    settings_for_category,
    windows_for_category,
)
from ..llm import llm
from .evaluator import CORRELATION_CHECKS, blocking_failures
from .families import family_stats, thompson_order
from .generators import dataset_fields
from .refine import RefineContext, best_first, certify, closest_first, refine, repair_spent
from .simulator import CandidateSpec, simulator
from .stats import brain_day_start

log = logging.getLogger(__name__)

SEARCH_KINDS = {"single": "Search Lab", "pair": "Double dataset · Pair", "interaction": "Interaction"}
KINDS = {**SEARCH_KINDS, "repair": "Repair near misses"}
STAGES = ["analyzing", "generating", "simulating", "optimizing", "completed"]
ACTIVE = ("queued", "running")
FINISHED = ("complete", "stopped", "skipped", "failed")
HISTORY_CAP = 3          # skip skeletons this market has already simulated this many times
ROBUST_RESERVE = 3       # repair rounds leave room in the repair budget for one robustness test (3 simulations)
REPAIR_POOL = 200        # near misses one repair task takes on, nearest to passing first


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def task_label(t: ResearchTask) -> str:
    if t.kind == "repair":
        picked = f" · {len(t.alpha_ids)} selected" if t.alpha_ids else ""
        return f"{KINDS['repair']} · {t.region} D{t.delay}{picked}"
    n = len(t.dataset_ids or [])
    return f"{KINDS.get(t.kind, t.kind)} · {t.region} D{t.delay} · {n} dataset{'s' if n != 1 else ''}"


async def autopilot_sims_today() -> int:
    """Simulations the Autopilot sent today (its research tasks plus fully automatic AI batches)."""
    from ..db.models import AutopilotRun, Batch
    start = brain_day_start()
    sent = ("queued", "simulating", "done", "error")
    async with db.session() as s:
        by_task = await s.scalar(select(func.count(Candidate.id)).join(ResearchTask, ResearchTask.id == Candidate.task_id)
                                 .where(ResearchTask.source == "autopilot", Candidate.created_at >= start,
                                        Candidate.status.in_(sent))) or 0
        by_batch = await s.scalar(select(func.count(Candidate.id)).join(Batch, Batch.id == Candidate.batch_id)
                                  .join(AutopilotRun, AutopilotRun.id == Batch.run_id)
                                  .where(AutopilotRun.mode == "autopilot", Candidate.created_at >= start,
                                         Candidate.task_id.is_(None), Candidate.status.in_(sent))) or 0
    return by_task + by_batch


class ResearchManager:
    def __init__(self) -> None:
        self._runners: dict[int, asyncio.Task] = {}
        self.version = 0

    # ------------------------------------------------------------ queries

    async def get(self, task_id: int) -> ResearchTask | None:
        async with db.session() as s:
            return await s.get(ResearchTask, task_id)

    async def list(self, limit: int = 300) -> list[ResearchTask]:
        async with db.session() as s:
            return list(await s.scalars(select(ResearchTask).order_by(ResearchTask.id.desc()).limit(limit)))

    def is_running(self, task_id: int) -> bool:
        t = self._runners.get(task_id)
        return t is not None and not t.done()

    async def _set(self, task_id: int, **values: Any) -> None:
        async with db.write() as s:
            await s.execute(update(ResearchTask).where(ResearchTask.id == task_id).values(**values))
        self.version += 1

    # ------------------------------------------------------------ lifecycle

    async def create(self, *, kind: str, region: str, delay: int, universe: str, dataset_ids: list[str],
                     planned_sims: int | None = None, cores: int | None = None, source: str = "manual",
                     reason: str = "", theme: str | None = None, status: str = "queued",
                     plan_day: str | None = None, alpha_ids: list[str] | None = None) -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown task kind {kind}")
        need = 0 if kind == "repair" else 2 if kind == "pair" else 1
        if len(dataset_ids) < need:
            raise ValueError(f"{KINDS[kind]} needs {need} dataset(s)")
        r = settings.research
        planned_sims = planned_sims or (r.repair_task_sims if kind == "repair" else r.task_sims)
        async with db.write() as s:
            t = ResearchTask(kind=kind, source=source, region=region, delay=int(delay), universe=universe,
                             dataset_ids=list(dataset_ids)[:need], alpha_ids=list(alpha_ids) if alpha_ids else None,
                             theme=theme, reason=reason, planned_sims=int(planned_sims),
                             cores=int(cores or r.task_cores),
                             status=status, plan_day=plan_day or eastern_day(), stage=None)
            s.add(t)
            await s.flush()
            tid = t.id
        self.version += 1
        bus.publish("autopilot" if source != "manual" else "simulate",
                    f"Research task #{tid} {status}: {KINDS[kind]} on {', '.join(dataset_ids[:need]) or 'existing alphas'} "
                    f"({region} D{delay}, {planned_sims} simulations).")
        return tid

    async def start(self, task_id: int) -> None:
        t = await self.get(task_id)
        if t is None or self.is_running(task_id):
            return
        if not brain.client.logged_in:
            raise RuntimeError("Sign in to BRAIN first")
        simulator.set_task_cores(task_id, t.cores)
        simulator.resume_task(task_id)
        await self._set(task_id, status="running", message=None, started_at=t.started_at or utcnow())
        self._runners[task_id] = asyncio.create_task(self._run(task_id), name=f"research-{task_id}")

    async def approve(self, task_id: int) -> None:
        if not brain.client.logged_in:
            raise RuntimeError("Sign in to BRAIN first")
        await self._set(task_id, status="queued")
        await self.start(task_id)

    async def pause(self, task_id: int) -> None:
        simulator.pause_task(task_id)
        await self._set(task_id, status="paused", message="Paused — queued simulations wait; running ones finish.")

    async def resume(self, task_id: int) -> None:
        if self.is_running(task_id):
            simulator.resume_task(task_id)
            await self._set(task_id, status="running", message=None)
        else:
            await self.start(task_id)

    async def stop(self, task_id: int, reason: str = "stopped by user") -> None:
        runner = self._runners.pop(task_id, None)
        if runner and not runner.done() and runner is not asyncio.current_task():
            runner.cancel()
        removed = await simulator.cancel_queued(task_id=task_id)
        simulator.resume_task(task_id)
        await self._set(task_id, status="stopped", message=f"{reason}; {removed} queued simulations removed",
                        finished_at=utcnow())

    async def delete(self, task_id: int) -> bool:
        t = await self.get(task_id)
        if t is None or t.status not in FINISHED + ("planned",):
            return False
        async with db.write() as s:
            row = await s.get(ResearchTask, task_id)
            if row:
                await s.delete(row)
        self.version += 1
        return True

    async def run_all(self) -> int:
        """Resume the simulation queue, every paused task and every approved (queued) task."""
        simulator.resume()
        n = 0
        for t in await self.list():
            if t.status in ("paused", "queued"):
                try:
                    await self.resume(t.id)
                    n += 1
                except RuntimeError as exc:
                    bus.publish("simulate", f"Task #{t.id} not started: {exc}", "warning")
                    break
        return n

    async def stop_all(self) -> int:
        """Header 'Stop all': stop the Autopilot, pause every active task and the queue. Nothing is deleted."""
        from .autopilot import autopilot
        await autopilot.stop("Stop all")
        n = 0
        for t in await self.list():
            if t.status in ACTIVE:
                await self.pause(t.id)
                n += 1
        simulator.pause("Stop all — press Run All on Tasks to continue")
        bus.publish("simulate", f"Stop all: Autopilot stopped, {n} task(s) and the simulation queue paused.", "warning")
        return n

    async def restore(self) -> None:
        """After an app restart: tasks that were running wait for the user (no silent restarts)."""
        async with db.session() as s:
            rows = list(await s.scalars(select(ResearchTask).where(ResearchTask.status.in_(ACTIVE))))
        for t in rows:
            simulator.set_task_cores(t.id, t.cores)
            simulator.pause_task(t.id)
            await self._set(t.id, status="paused", message="Interrupted by an app restart — press Run to continue.")

    # ------------------------------------------------------------ runner

    async def _run(self, task_id: int) -> None:
        from .audit import ACTOR
        ACTOR.set(f"task:{task_id}")
        try:
            await self._run_inner(task_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("research task %s failed", task_id)
            await self._set(task_id, status="failed", message=str(exc)[:500], finished_at=utcnow())
            bus.publish("simulate", f"Research task #{task_id} failed: {exc}", "error")
        finally:
            self._runners.pop(task_id, None)

    async def _counts(self, task_id: int) -> dict[str, Any]:
        async with db.session() as s:
            cands = (await s.execute(select(Candidate.status, Candidate.origin, Candidate.alpha_id)
                                     .where(Candidate.task_id == task_id))).all()
            ids = [a for _, _, a in cands if a]
            alphas = {a.id: a for a in await s.scalars(select(Alpha).where(Alpha.id.in_(ids)))} if ids else {}
        verdicts = [alphas[a].verdict for _, _, a in cands if a in alphas]
        best = max((a for a in alphas.values() if a.sharpe is not None), key=lambda a: a.sharpe, default=None)
        return {
            "sent": sum(st in ("queued", "simulating", "done", "error") and o not in ("optimize", "robustness")
                        for st, o, _ in cands),
            "n_simulated": sum(st == "done" for st, _, _ in cands),
            "n_optimized": sum(o == "optimize" for _, o, _ in cands),
            "n_strong": verdicts.count("strong"), "n_ready": verdicts.count("ready"),
            "n_rejected": verdicts.count("rejected") + verdicts.count("weak")
            + sum(st in ("error", "invalid") for st, _, _ in cands),
            "best_sharpe": best.sharpe if best else None, "best_alpha_id": best.id if best else None,
        }

    async def _refresh(self, task_id: int, left_out: Counter, generated: int, stage: str | None = None) -> dict:
        c = await self._counts(task_id)
        values = {k: v for k, v in c.items() if k != "sent"}
        values.update(left_out=dict(left_out), n_generated=generated)
        if stage:
            values["stage"] = stage
        await self._set(task_id, **values)
        return c

    async def _top(self, task_id: int, verdicts: tuple[str, ...], limit: int) -> list[Alpha]:
        async with db.session() as s:
            ids = list(await s.scalars(select(Candidate.alpha_id).where(Candidate.task_id == task_id,
                                                                        Candidate.alpha_id.is_not(None))))
            if not ids:
                return []
            return list(await s.scalars(select(Alpha).where(Alpha.id.in_(ids), Alpha.verdict.in_(verdicts))
                                        .order_by(Alpha.score.desc().nulls_last()).limit(limit)))

    def _expressions(self, t: ResearchTask, fields: dict[str, list[tuple[str, str]]],  # noqa: ANN202
                     windows: tuple[int, ...] = ENUM_WINDOWS, family_order: list[str] | None = None):
        ds = t.dataset_ids or []
        if t.kind == "single":
            yield from enumerate_single(fields[ds[0]], windows=windows, family_order=family_order)
            yield from enumerate_group(fields[ds[0]][:12], windows=windows[:3], family_order=family_order)
        elif t.kind == "pair":
            a, b = fields[ds[0]][:12], fields[ds[1]][:12]
            yield from enumerate_pairs(a, b, windows=windows, family_order=family_order)
            yield from enumerate_pairs(b, a, windows=windows, family_order=family_order)
        else:
            f = fields[ds[0]][:16]
            yield from enumerate_pairs(f, f, windows=windows, family_order=family_order)
            yield from enumerate_pairs(f, PV_PARTNERS, windows=windows, family_order=family_order)
            yield from enumerate_group(f[:8], windows=windows[:3], family_order=family_order)

    async def _screen_fields(self, t: ResearchTask, fields: dict[str, list[tuple[str, str]]], base: dict[str, Any],
                             category: str | None, done_exprs: set[str], budget: int, left_out: Counter,
                             generated: int) -> tuple[dict[str, list[tuple[str, str]]], int, list[int]]:
        """Probe every field once; keep the ones with a signal (sign-flipped when it runs the wrong way).

        Returns (screened fields per dataset, probes generated, candidate ids). Fields whose probe produced no result
        are kept at the end of the list (they were not proven useless). Already-probed fields (a resumed task) are
        read back from the database instead of being simulated again."""
        r = settings.research
        dataset_tag = ",".join(t.dataset_ids or [])
        probes: dict[str, tuple[str, str, str]] = {}          # expression -> (dataset, field id, field type)
        for d, flist in fields.items():
            for fid, ftype in flist:
                probes[probe_expression(fid, ftype, category)] = (d, fid, ftype)
        todo = [e for e in probes if e not in done_exprs][:max(0, budget)]
        ids: list[int] = []
        if todo:
            specs = [CandidateSpec(expression=e, settings=base, origin="search", dataset_id=dataset_tag,
                                   task_id=t.id, hypothesis=f"{KINDS[t.kind]}: field screening",
                                   rationale={"family": "ts_rank", "probe": True, "field": probes[e][1]})
                     for e in todo]
            done_exprs.update(todo)
            res = await simulator.prepare(specs, initial_status="approved", task_id=t.id)
            for k, v in res.left_out.items():
                left_out[k] += v
            ids = [x.id for x in res.created]
            await simulator.enqueue(ids)
            await self._set(t.id, message=f"Screening {len(todo)} fields: one simulation each before the search.")
            await self._refresh(t.id, left_out, generated + len(todo), stage="simulating")
            await simulator.wait_for(ids)
        # read every probe's result (also the ones simulated by an earlier run of this task)
        async with db.session() as s:
            rows = (await s.execute(select(Candidate.expression, Alpha.sharpe, Alpha.fitness)
                                    .join(Alpha, Alpha.id == Candidate.alpha_id)
                                    .where(Candidate.task_id == t.id, Candidate.expression.in_(list(probes))))).all()
        result = {e: (sh, fi) for e, sh, fi in rows if sh is not None}
        out: dict[str, list[tuple[str, str]]] = {}
        kept_total = 0
        for d, flist in fields.items():
            scored, unknown = [], []
            for fid, ftype in flist:
                e = probe_expression(fid, ftype, category)
                if e not in result:
                    unknown.append((fid, ftype))
                    continue
                sh, fi = result[e]
                strength = abs(sh) + 0.5 * abs(fi or 0.0)
                scored.append((strength, abs(sh), ("-" + fid) if sh < 0 else fid, ftype))
            scored.sort(reverse=True)
            keep_n = max(r.screen_keep_min, math.ceil(len(flist) * r.screen_keep_share))
            keep = [(fid, ft) for _st, ash, fid, ft in scored if ash >= r.screen_min_sharpe][:keep_n]
            if len(keep) < min(r.screen_keep_min, len(scored)):           # nothing cleared the bar: best few anyway
                keep = [(fid, ft) for _st, _a, fid, ft in scored[:r.screen_keep_min]]
            out[d] = keep + unknown
            kept_total += len(keep)
        flipped = sum(1 for flist in out.values() for fid, _ in flist if fid.startswith("-"))
        await self._set(t.id, message=f"Screening kept {kept_total} of {sum(len(v) for v in fields.values())} fields "
                                      f"({flipped} used sign-flipped); searching those.")
        return out, len(todo), ids

    async def _run_inner(self, task_id: int) -> None:
        t = await self.get(task_id)
        if t is None:
            return
        if t.kind == "repair":
            await self._run_repair(t)
            return
        scope = {"region": t.region, "delay": t.delay, "universe": t.universe}
        await self._set(task_id, stage="analyzing")
        fields: dict[str, list[tuple[str, str]]] = {}
        category = None
        async with db.session() as s:
            for d in t.dataset_ids or []:
                ds = await s.scalar(select(Dataset).where(Dataset.id == d, Dataset.region == t.region,
                                                          Dataset.delay == t.delay, Dataset.universe == t.universe))
                category = category or (ds.category_name if ds else None)
        for d in t.dataset_ids or []:
            fields[d] = await dataset_fields(scope, d)
            if not fields[d]:
                raise RuntimeError(f"No fields synced for {d} in {t.region}/D{t.delay}/{t.universe} — run "
                                   "'Sync everything for scope' first.")
        decay, neut, trunc, nan = settings_for_category(category)
        allowed = brain.options.neutralizations(t.region)
        base = normalize_settings({**scope, "decay": decay, "neutralization": neut if neut in allowed else allowed[0],
                                   "truncation": trunc, "nanHandling": nan})
        dataset_tag = ",".join(t.dataset_ids or [])
        async with db.session() as s:
            done_exprs = set(await s.scalars(select(Candidate.expression).where(Candidate.task_id == task_id)))
        left_out: Counter = Counter(t.left_out or {})
        generated = t.n_generated or 0
        wave = max(20, min(200, t.cores * 30))
        exhausted = False
        await self._set(task_id, stage="generating")

        # ---------------- budget: first-pass simulations + a share reserved for repairing near misses
        r = settings.research
        repair_budget = int(round(t.planned_sims * max(0.0, min(0.9, r.optimize_share))))
        first_budget = t.planned_sims - repair_budget
        ctx = RefineContext(task_id=task_id, use_llm=llm.configure(), max_depth=r.repair_depth,
                            allowance=lambda: self._repair_allowance(t, repair_budget))

        async def repair_only() -> int:
            return await self._repair_allowance(t, repair_budget) - ROBUST_RESERVE

        # ---------------- field screening, then templates in the order your history ranks their families
        windows = windows_for_category(category)
        try:
            fstats = await family_stats(t.region, t.delay)
        except Exception:
            fstats = {}
        families = [x.family for x in TEMPLATES + PAIR_TEMPLATES]
        family_order = thompson_order(families, fstats, random.Random(task_id)) if fstats else None
        n_fields = sum(len(v) for v in fields.values())
        if r.field_screening and n_fields >= 6 and first_budget >= 2 * n_fields:
            fields, probed, probe_ids = await self._screen_fields(t, fields, base, category, done_exprs,
                                                                  first_budget, left_out, generated)
            generated += probed
            await self._refresh(task_id, left_out, generated)
            await self._repair_wave(t, probe_ids, ctx, repair_budget, first_budget, left_out, generated)
        exprs = self._expressions(t, fields, windows=windows, family_order=family_order)

        # ---------------- waves of first-pass expressions, each followed by repairs and BRAIN checks
        while True:
            c = await self._counts(task_id)
            remaining = first_budget - c["sent"]
            if remaining <= 0 or exhausted:
                break
            if t.source == "autopilot":
                cap_left = settings.research.autopilot_daily_cap - await autopilot_sims_today()
                if cap_left <= 0:
                    await self._refresh(task_id, left_out, generated)
                    await self._set(task_id, status="paused", message="Daily Autopilot cap reached — continues after "
                                                                      "the quota reset.")
                    simulator.pause_task(task_id)
                    return
                remaining = min(remaining, cap_left)
            n = min(wave, remaining)
            specs: list[CandidateSpec] = []
            while len(specs) < n:
                try:
                    expr, family = next(exprs)
                except StopIteration:
                    exhausted = True
                    break
                if expr in done_exprs:
                    continue
                done_exprs.add(expr)
                specs.append(CandidateSpec(expression=expr, settings=base, origin=t.kind if t.kind != "single"
                                           else "search", dataset_id=dataset_tag, task_id=task_id,
                                           hypothesis=f"{KINDS[t.kind]}: {family}", rationale={"family": family}))
            if not specs:
                break
            generated += len(specs)
            res = await simulator.prepare(specs, initial_status="approved", task_id=task_id, history_cap=HISTORY_CAP,
                                          cluster_cap=t.source != "manual")
            for k, v in res.left_out.items():
                left_out[k] += v
            ids = [x.id for x in res.created]
            await simulator.enqueue(ids)
            await self._refresh(task_id, left_out, generated, stage="simulating")
            await simulator.wait_for(ids)
            await self._refresh(task_id, left_out, generated)
            await self._repair_wave(t, ids, ctx, repair_budget, first_budget, left_out, generated)

        # ---------------- Search Lab: group / event variants of the best first-order results
        if t.kind == "single":
            c = await self._counts(task_id)
            budget = first_budget - c["sent"]
            best = await self._top(task_id, ("promising", "strong", "ready"), 8)
            if budget > 0 and best:
                specs = []
                rng = random.Random(task_id)
                for a in best:
                    for expr, family in second_order(a.code or "", rng=rng):
                        if expr not in done_exprs and len(specs) < budget:
                            done_exprs.add(expr)
                            specs.append(CandidateSpec(expression=expr, settings=normalize_settings(a.settings or base),
                                                       origin="search", dataset_id=dataset_tag, task_id=task_id,
                                                       parent_alpha_id=a.id, hypothesis=f"Search Lab: {family} of {a.id}",
                                                       rationale={"family": family}))
                if specs:
                    generated += len(specs)
                    res = await simulator.prepare(specs, initial_status="approved", task_id=task_id,
                                                  cluster_cap=t.source != "manual")
                    for k, v in res.left_out.items():
                        left_out[k] += v
                    ids = [x.id for x in res.created]
                    await simulator.enqueue(ids)
                    await simulator.wait_for(ids)
                    await self._repair_wave(t, ids, ctx, repair_budget, first_budget, left_out, generated)

        # ---------------- what is left of the repair budget, then BRAIN checks on every strong alpha
        await self._refresh(task_id, left_out, generated, stage="optimizing")
        seeds = await self._seeds(task_id, None, 3)
        if seeds and await ctx.budget() > ROBUST_RESERVE:
            await refine(seeds, replace(ctx, allowance=repair_only))
        strong = [a.id for a in await self._top(task_id, ("strong",), 500)]
        await certify(strong, ctx, cap=max(0, r.check_cap - await self._checks_done(task_id)))
        c = await self._refresh(task_id, left_out, generated, stage="completed")
        await self._set(task_id, status="complete", finished_at=utcnow(),
                        message=f"Done: {c['n_simulated']} simulated, {c['n_optimized']} repaired, {c['n_strong']} "
                                f"strong, {c['n_ready']} passed BRAIN's check (ready for your manual submission).")
        bus.publish("autopilot" if t.source != "manual" else "simulate",
                    f"Research task #{task_id} complete: {c['n_simulated']} simulated, {c['n_strong']} strong, "
                    f"{c['n_ready']} ready" + (f", best Sharpe {c['best_sharpe']:.2f}" if c["best_sharpe"] else "")
                    + ".", "success" if c["n_ready"] or c["n_strong"] else "info", task_id=task_id,
                    kind="task_finished")

    # ------------------------------------------------------------ repair tasks

    async def _scope_alphas(self, t: ResearchTask, verdicts: tuple[str, ...]) -> list[Alpha]:
        """Alphas this repair task may work on: the ones you picked, else every unsubmitted one in its scope."""
        async with db.session() as s:
            stmt = select(Alpha).where(Alpha.region == t.region, Alpha.delay == t.delay, Alpha.universe == t.universe,
                                       Alpha.type == "REGULAR", Alpha.verdict.in_(verdicts),
                                       (Alpha.status.is_(None)) | (Alpha.status == "UNSUBMITTED"))
            if t.alpha_ids:
                stmt = stmt.where(Alpha.id.in_(t.alpha_ids))
            return list(await s.scalars(stmt))

    async def _adopt(self, t: ResearchTask, alphas: list[Alpha]) -> None:
        """Record the alphas a repair task works on as its own (no simulation), so its counts and BRAIN-check totals
        include them."""
        if not alphas:
            return
        async with db.write() as s:
            have = set(await s.scalars(select(Candidate.alpha_id).where(Candidate.task_id == t.id,
                                                                        Candidate.origin == "seed")))
            s.add_all(Candidate(expression=a.code or "", settings=normalize_settings(a.settings or {}),
                                settings_hash=settings_hash(a.settings or {}), region=a.region or t.region,
                                delay=a.delay if a.delay is not None else t.delay, origin="seed", status="reused",
                                alpha_id=a.id, task_id=t.id, finished_at=utcnow(), hypothesis="Repair task seed")
                      for a in alphas if a.id not in have)

    async def _run_repair(self, t: ResearchTask) -> None:
        """Turn the alphas you already have into Ready ones, cheapest first: BRAIN's check on strong alphas costs no
        simulation, correlation failures come next, then one shared budget goes wherever a near miss is closest to
        passing (refine.best_first)."""
        from .decorrelate import decorrelate
        task_id, r = t.id, settings.research
        left_out: Counter = Counter(t.left_out or {})
        budget = t.planned_sims
        ctx = RefineContext(task_id=task_id, use_llm=llm.configure(), max_depth=r.repair_depth, revisit=True,
                            resume=True, allowance=lambda: self._repair_allowance(t, budget))

        async def repair_only() -> int:
            return await self._repair_allowance(t, budget) - ROBUST_RESERVE

        async def refresh() -> None:
            await self._refresh(task_id, left_out, 0, stage="optimizing")

        await self._set(task_id, stage="analyzing")
        strong = await self._scope_alphas(t, ("strong",))
        await self._adopt(t, strong)
        await self._set(task_id, stage="optimizing")
        await certify([a.id for a in strong], ctx, cap=r.check_cap)
        await refresh()

        hard = [a for a in await self._scope_alphas(t, ("promising",))
                if a.check_state == "fail" and a.repair_state != "exhausted"
                and set(blocking_failures(a.checks)) and set(blocking_failures(a.checks)) <= CORRELATION_CHECKS]
        await self._adopt(t, hard)
        for a in hard:
            if await repair_only() <= 0:
                break
            await decorrelate(a, replace(ctx, resume=False, allowance=repair_only), cap=r.check_cap)
            await refresh()

        pool = await self._scope_alphas(t, ("promising", "weak"))
        ordered = await asyncio.to_thread(closest_first, pool)          # thousands of alphas: keep the window alive
        seeds = [a for a in ordered if a.id not in {x.id for x in hard}][:REPAIR_POOL]
        if t.alpha_ids:                       # you chose these: try the ones automatic repair would skip as well
            seeds += [a for a in pool if a.code and a.id not in {x.id for x in seeds}]
        await self._adopt(t, seeds)

        async def check(ids: list[str]):  # noqa: ANN202
            report = await certify(ids, ctx, cap=r.check_cap)
            await refresh()
            return report

        if seeds and await repair_only() > 0:
            await best_first(seeds, replace(ctx, allowance=repair_only), check=check, on_round=refresh)
        c = await self._refresh(task_id, left_out, 0, stage="completed")
        await self._set(task_id, status="complete", finished_at=utcnow(),
                        message=f"Done: {len(seeds) + len(hard) + len(strong)} alphas worked on, "
                                f"{c['n_simulated']} simulated, "
                                f"{c['n_strong']} strong, {c['n_ready']} passed BRAIN's check (ready for your manual "
                                "submission).")
        bus.publish("simulate", f"Repair task #{task_id} complete: {c['n_simulated']} simulated, {c['n_ready']} ready.",
                    "success" if c["n_ready"] else "info", task_id=task_id, kind="task_finished")

    async def _repair_allowance(self, t: ResearchTask, repair_budget: int) -> int:
        """Repair simulations the task may still spend: its reserved share, and the daily Autopilot cap."""
        left = repair_budget - await repair_spent(t.id)
        if t.source == "autopilot":
            left = min(left, settings.research.autopilot_daily_cap - await autopilot_sims_today())
        return left

    async def _seeds(self, task_id: int, candidate_ids: list[int] | None, limit: int) -> list[Alpha]:
        """Near misses worth repairing and never expanded, nearest to passing every check first."""
        async with db.session() as s:
            stmt = select(Candidate.alpha_id).where(Candidate.task_id == task_id, Candidate.alpha_id.is_not(None))
            if candidate_ids is not None:
                stmt = stmt.where(Candidate.id.in_(candidate_ids or [-1]))
            ids = list(await s.scalars(stmt))
            if not ids:
                return []
            rows = list(await s.scalars(select(Alpha).where(Alpha.id.in_(ids), Alpha.expanded_at.is_(None),
                                                            Alpha.verdict.in_(["promising", "weak"]))))
        return closest_first(rows)[:limit]

    async def _checks_done(self, task_id: int) -> int:
        async with db.session() as s:
            ids = list(await s.scalars(select(Candidate.alpha_id).where(Candidate.task_id == task_id,
                                                                        Candidate.alpha_id.is_not(None))))
            if not ids:
                return 0
            return await s.scalar(select(func.count(Alpha.id)).where(Alpha.id.in_(ids),
                                                                     Alpha.checked_at.is_not(None))) or 0

    async def _repair_wave(self, t: ResearchTask, candidate_ids: list[int], ctx: RefineContext, repair_budget: int,
                           first_budget: int, left_out: Counter, generated: int) -> None:
        """After a wave: repair its best near misses (a slice of the repair budget proportional to the wave) and
        run BRAIN's check on its strong alphas, so good alphas surface while the task is still running."""
        if not candidate_ids:
            return
        seeds = await self._seeds(t.id, candidate_ids, 2)
        if seeds and repair_budget > 0:
            wave_cap = max(12, math.ceil(repair_budget * len(candidate_ids) / max(1, first_budget) * 1.5))
            start = await repair_spent(t.id)

            async def wave_allowance() -> int:
                return min(await self._repair_allowance(t, repair_budget) - ROBUST_RESERVE,
                           wave_cap - (await repair_spent(t.id) - start))

            await self._refresh(t.id, left_out, generated, stage="optimizing")
            await refine(seeds, replace(ctx, allowance=wave_allowance))
        strong = [a.id for a in await self._top(t.id, ("strong",), 200)]
        cap = settings.research.check_cap - await self._checks_done(t.id)
        if strong and cap > 0:
            await certify(strong, ctx, cap=cap)
        await self._refresh(t.id, left_out, generated, stage="simulating")

    # ------------------------------------------------------------ planning

    async def strong_by_theme(self, region: str, delay: int, universe: str) -> dict[str, int]:
        async with db.session() as s:
            alphas = list(await s.scalars(select(Alpha).where(Alpha.region == region, Alpha.delay == delay,
                                                              Alpha.verdict.in_(["strong", "ready", "submitted"]))))
            ratings = {r.dataset_id: r.theme for r in await s.scalars(select(DatasetRating).where(
                DatasetRating.region == region, DatasetRating.delay == delay, DatasetRating.universe == universe))}
            cats = {d.id: d.category_name for d in await s.scalars(select(Dataset).where(
                Dataset.region == region, Dataset.delay == delay, Dataset.universe == universe))}
        out: Counter = Counter()
        for a in alphas:
            first = str((a.dataset_ids or [""])[0] or "").split(",")[0]
            out[ratings.get(first) or cats.get(first) or "Other"] += 1
        return dict(out.most_common())

    async def plan(self, *, source: str = "assisted", region: str | None = None, delay: int | None = None,
                   universe: str | None = None, exclude: set[str] | None = None) -> dict[str, Any]:
        """Choose the next research task: focus theme → dataset(s) → kind, with a human-readable reason."""
        from .datasets import rated_datasets
        r = settings.research
        region, delay, universe = region or r.region, int(delay if delay is not None else r.delay), universe or r.universe
        rows = [x for x in await rated_datasets(region, delay, universe) if (x["field_count"] or 0) > 0
                and x["id"] not in (exclude or set())]
        if not rows:
            raise RuntimeError(f"No datasets with fields for {region}/D{delay}/{universe} — sync the catalogue first.")
        strong = await self.strong_by_theme(region, delay, universe)
        themes = sorted({x["theme"] for x in rows if x["theme"]})
        if r.research_focus and r.research_focus in themes:
            focus, why = r.research_focus, f"Focus set to {r.research_focus} in the settings."
        else:
            focus = min(themes, key=lambda th: (strong.get(th, 0), -max((x["ai_score"] or x["opportunity"] or 0)
                                                                       for x in rows if x["theme"] == th)))
            why = f"{focus} has the fewest strong alphas so far ({strong.get(focus, 0)}); new themes break correlation."
        pool = [x for x in rows if x["theme"] == focus] or rows
        plan = None
        if llm.configure() and any(x["rated"] for x in rows):
            try:
                plan = await self._ai_plan(region, delay, universe, focus, strong, pool + [x for x in rows
                                                                                       if x not in pool][:15])
            except Exception as exc:
                log.warning("AI research plan failed: %s", exc)
        if plan is None:
            top = max(pool, key=lambda x: ((x["ai_score"] or 0) + x["opportunity"]) / (1 + x["tried"] / 200))
            kind = {"double": "pair"}.get(top.get("recommended") or "single", top.get("recommended") or "single")
            ids = [top["id"]]
            if kind == "pair":
                partner = max((x for x in rows if x["id"] != top["id"] and x["theme"] != top["theme"]),
                              key=lambda x: (x["double"] or 0) + x["opportunity"], default=None)
                if partner is None:
                    kind = "single"
                else:
                    ids.append(partner["id"])
            plan = {"kind": kind, "dataset_ids": ids,
                    "reason": f"{top['id']} scores highest in {focus} "
                              f"(AI {top['ai_score'] if top['ai_score'] is not None else '—'}, "
                              f"{top['tried']} of your simulations so far)."}
        plan.update(region=region, delay=delay, universe=universe, theme=focus, focus_reason=why)
        return plan

    async def _ai_plan(self, region: str, delay: int, universe: str, focus: str, strong: dict[str, int],
                       pool: list[dict[str, Any]]) -> dict[str, Any] | None:
        from ..llm.prompts import render
        from ..llm.schemas import ResearchPlan
        lines = "\n".join(f"- {x['id']} · {x['theme']} · AI {x['ai_score']} · {x['single']}/{x['double']}/"
                          f"{x['interaction']} · {x['alpha_count']} alphas · {x['tried']} mine · ×{x['pyramid_multiplier']}"
                          for x in pool[:40])
        res = await llm.generate_json(await render("research_plan", scope=f"{region} / D{delay} / {universe}",
                                                      focus=focus, strong_by_theme=strong or "none", datasets=lines),
                                         ResearchPlan, role="ideation", purpose="research_plan", thinking="low")
        valid = {x["id"] for x in pool}
        ids = [d for d in res.dataset_ids if d in valid]
        kind = res.kind if res.kind in SEARCH_KINDS else "single"
        if kind == "pair" and len(ids) < 2:
            kind = "single"
        if not ids:
            return None
        return {"kind": kind, "dataset_ids": ids[:2 if kind == "pair" else 1], "reason": res.reason}

    async def plan_and_create(self, *, source: str, start: bool, exclude: set[str] | None = None) -> int:
        p = await self.plan(source=source, exclude=exclude)
        r = settings.research
        status = "queued" if start else "planned"
        if not brain.client.quota.can_spend(r.task_sims, settings.brain.quota_reserve):
            status = "skipped"
        tid = await self.create(kind=p["kind"], region=p["region"], delay=p["delay"], universe=p["universe"],
                                dataset_ids=p["dataset_ids"], source=source, reason=f"{p['focus_reason']} {p['reason']}",
                                theme=p["theme"], status=status)
        if status == "skipped":
            await self._set(tid, message="Not enough simulations left today.")
        elif start:
            await self.start(tid)
        return tid

    async def plan_day(self) -> list[int]:
        """Plan today's research (and start it in Full Autopilot mode)."""
        r = settings.research
        start = r.mode == "autopilot"
        ids: list[int] = []
        used: set[str] = set()
        for _ in range(max(1, r.tasks_per_day)):
            try:
                tid = await self.plan_and_create(source="autopilot" if start else "assisted", start=start,
                                                 exclude=used)
            except Exception as exc:
                bus.publish("autopilot", f"Could not plan research: {exc}", "warning")
                break
            ids.append(tid)
            t = await self.get(tid)
            used.update(t.dataset_ids or [])
        await db.kv_set("autopilot:planned_day", eastern_day())
        return ids


research = ResearchManager()


async def daily_scheduler() -> None:
    """Runs in the background: after each BRAIN quota reset, plan the day's research if the user turned it on."""
    from .audit import ACTOR
    ACTOR.set("autopilot")
    await asyncio.sleep(20)
    while True:
        try:
            today = eastern_day()
            if simulator.paused and (simulator.pause_reason or "").startswith("quota"):
                paused_day = await db.kv_get("quota_pause_day")
                if paused_day is None:
                    await db.kv_set("quota_pause_day", today)
                elif paused_day != today:
                    simulator.resume()
                    await db.kv_set("quota_pause_day", None)
                    bus.publish("simulate", "New BRAIN day: simulation queue resumed after the quota pause.")
            if settings.research.auto_daily and brain.client.logged_in and settings.research.mode != "manual":
                if await db.kv_get("autopilot:planned_day") != today:
                    async with db.write() as s:
                        await s.execute(update(ResearchTask).where(ResearchTask.status == "planned",
                                                                   ResearchTask.plan_day != today)
                                        .values(status="skipped", message="Not approved on its day."))
                    for t in await research.list():
                        if t.status == "paused" and (t.message or "").startswith("Daily Autopilot cap"):
                            await research.resume(t.id)
                    ids = await research.plan_day()
                    bus.publish("autopilot", f"New BRAIN day: planned {len(ids)} research task(s)"
                                + (" and started them." if settings.research.mode == "autopilot"
                                   else " — approve them on the Autopilot page."), "success" if ids else "info")
        except Exception:
            log.exception("daily scheduler failed")
        await asyncio.sleep(300)
