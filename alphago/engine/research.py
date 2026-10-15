"""Research tasks: large, systematic searches with their own share of BRAIN's concurrent simulation slots.

Kinds
- single       "Search Lab": first-order templates over every field of one dataset, then group / event variants of
               the best results.
- pair         "Double dataset · Pair": two-field templates combining a field of dataset A with one of dataset B.
- interaction  field × field templates inside one dataset, plus each field against price/volume.

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
import random
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update

from ..brain import brain
from ..brain.quota import eastern_day
from ..brain.simulations import normalize_settings
from ..config import settings
from ..db import db
from ..db.models import Alpha, Candidate, Dataset, DatasetRating, ResearchTask
from ..events import bus
from ..knowledge.templates import (PV_PARTNERS, enumerate_group, enumerate_pairs, enumerate_single, second_order,
                                   settings_for_category)
from . import actions
from .generators import dataset_fields
from .simulator import CandidateSpec, simulator
from .stats import brain_day_start

log = logging.getLogger(__name__)

KINDS = {"single": "Search Lab", "pair": "Double dataset · Pair", "interaction": "Interaction"}
STAGES = ["analyzing", "generating", "simulating", "optimizing", "completed"]
ACTIVE = ("queued", "running")
FINISHED = ("complete", "stopped", "skipped", "failed")
HISTORY_CAP = 3          # skip skeletons this market has already simulated this many times


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def task_label(t: ResearchTask) -> str:
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
                     plan_day: str | None = None) -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown task kind {kind}")
        need = 2 if kind == "pair" else 1
        if len(dataset_ids) < need:
            raise ValueError(f"{KINDS[kind]} needs {need} dataset(s)")
        r = settings.research
        async with db.write() as s:
            t = ResearchTask(kind=kind, source=source, region=region, delay=int(delay), universe=universe,
                             dataset_ids=list(dataset_ids)[:need], theme=theme, reason=reason,
                             planned_sims=int(planned_sims or r.task_sims), cores=int(cores or r.task_cores),
                             status=status, plan_day=plan_day or eastern_day(), stage=None)
            s.add(t)
            await s.flush()
            tid = t.id
        self.version += 1
        bus.publish("autopilot" if source != "manual" else "simulate",
                    f"Research task #{tid} {status}: {KINDS[kind]} on {', '.join(dataset_ids[:need])} "
                    f"({region} D{delay}, {planned_sims or r.task_sims} simulations).")
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
                                        .order_by(Alpha.sharpe.desc().nulls_last()).limit(limit)))

    def _expressions(self, t: ResearchTask, fields: dict[str, list[tuple[str, str]]]):  # noqa: ANN202
        ds = t.dataset_ids or []
        if t.kind == "single":
            yield from enumerate_single(fields[ds[0]])
        elif t.kind == "pair":
            a, b = fields[ds[0]][:12], fields[ds[1]][:12]
            yield from enumerate_pairs(a, b)
            yield from enumerate_pairs(b, a)
        else:
            f = fields[ds[0]][:16]
            yield from enumerate_pairs(f, f)
            yield from enumerate_pairs(f, PV_PARTNERS)
            yield from enumerate_group(f[:8])

    async def _run_inner(self, task_id: int) -> None:
        t = await self.get(task_id)
        if t is None:
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
        exprs = self._expressions(t, fields)
        exhausted = False
        await self._set(task_id, stage="generating")

        # ---------------- waves of first-pass expressions
        while True:
            c = await self._counts(task_id)
            remaining = t.planned_sims - c["sent"]
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
            res = await simulator.prepare(specs, initial_status="approved", task_id=task_id, history_cap=HISTORY_CAP)
            for k, v in res.left_out.items():
                left_out[k] += v
            ids = [x.id for x in res.created]
            await simulator.enqueue(ids)
            await self._refresh(task_id, left_out, generated, stage="simulating")
            await simulator.wait_for(ids)
            await self._refresh(task_id, left_out, generated)

        # ---------------- Search Lab: group / event variants of the best first-order results
        if t.kind == "single":
            c = await self._counts(task_id)
            budget = t.planned_sims - c["sent"]
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
                    res = await simulator.prepare(specs, initial_status="approved", task_id=task_id)
                    for k, v in res.left_out.items():
                        left_out[k] += v
                    ids = [x.id for x in res.created]
                    await simulator.enqueue(ids)
                    await simulator.wait_for(ids)

        # ---------------- optimise near misses, check strong alphas, robustness
        await self._refresh(task_id, left_out, generated, stage="optimizing")
        from ..llm import llm
        from .optimizer import optimize_alpha
        near = await self._top(task_id, ("promising",), 3)
        opt_ids: list[int] = []
        for a in near:
            try:
                opt_ids += await optimize_alpha(a.id, use_llm=llm.configure(), task_id=task_id)
            except Exception as exc:
                log.warning("optimise %s failed: %s", a.id, exc)
        if opt_ids:
            await simulator.wait_for(opt_ids)
        from .autopilot import check_robustness
        for a in await self._top(task_id, ("strong",), 3):
            try:
                state, _ = await actions.run_check(a.id)
                if state == "pass" and settings.research.robustness_gate:
                    await check_robustness(a.id, None, task_id=task_id)
            except Exception as exc:
                bus.publish("check", f"Check failed for {a.id}: {getattr(exc, 'message', exc)}", "warning")
        c = await self._refresh(task_id, left_out, generated, stage="completed")
        await self._set(task_id, status="complete", finished_at=utcnow(),
                        message=f"Done: {c['n_simulated']} simulated, {c['n_optimized']} optimised, {c['n_strong']} "
                                f"strong, {c['n_ready']} ready for manual submission.")
        bus.publish("autopilot" if t.source != "manual" else "simulate",
                    f"Research task #{task_id} complete: {c['n_simulated']} simulated, {c['n_strong']} strong, "
                    f"{c['n_ready']} ready" + (f", best Sharpe {c['best_sharpe']:.2f}" if c["best_sharpe"] else "")
                    + ".", "success" if c["n_ready"] or c["n_strong"] else "info")

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
        from ..llm import llm
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
        from ..llm import llm
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
        kind = res.kind if res.kind in KINDS else "single"
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
