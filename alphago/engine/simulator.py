"""Simulation queue: validate → de-duplicate/reuse → pack into multi-simulations → run under slot/quota limits →
ingest results → classify. Every caller (matrix, autopilot, research tasks, optimiser) goes through this service.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import func, select, update

from ..brain import brain
from ..brain.errors import BrainError, ConcurrencyLimit, PersonaRequired, QuotaExhausted, SimulationRejected
from ..brain.simulations import group_key, normalize_settings, regular_payload, settings_hash
from ..config import settings as app_settings
from ..db import db
from ..db.models import Alpha, Candidate, Simulation
from ..events import bus
from ..fastexpr import canonical_hash, compute_metrics, skeleton_hash, validate
from .catalogs import field_dataset_map, get_catalog
from .ingest import journal, upsert_alpha

log = logging.getLogger(__name__)

TERMINAL = {"done", "error", "invalid", "reused", "cancelled", "rejected_by_user"}
# candidate origins that count as "alphas generated" on the Dashboard
GENERATED_ORIGINS = ["llm", "template", "mutation", "spc", "genius", "search", "pair", "interaction"]


@dataclass
class CandidateSpec:
    expression: str
    settings: dict[str, Any]
    origin: str = "manual"
    hypothesis: str | None = None
    rationale: dict[str, Any] | None = None
    dataset_id: str | None = None
    parent_id: int | None = None
    parent_alpha_id: str | None = None
    task_id: int | None = None


@dataclass
class PrepareResult:
    created: list[Candidate] = field(default_factory=list)
    reused: list[Candidate] = field(default_factory=list)
    invalid: list[Candidate] = field(default_factory=list)
    skipped: int = 0
    left_out: dict[str, int] = field(default_factory=dict)      # reason -> count ("Left out on purpose")

    def leave(self, reason: str) -> None:
        self.left_out[reason] = self.left_out.get(reason, 0) + 1

    @property
    def all(self) -> list[Candidate]:
        return self.created + self.reused + self.invalid


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SimulationService:
    def __init__(self) -> None:
        self._queue: deque[int] = deque()
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()
        self._inflight_candidates = 0
        self._waiters: dict[int, list[asyncio.Future]] = {}
        self._listeners: list[Callable[[Candidate, Alpha | None], Any]] = []
        self._quota_warned = False
        self.paused = False
        self.pause_reason: str | None = None
        # research tasks: each gets a cap of concurrent multi-simulations ("cores"); packs never mix tasks
        self._task_of: dict[int, int] = {}          # queued candidate id -> task id
        self.task_cores: dict[int, int] = {}
        self.paused_tasks: set[int] = set()
        self._inflight_task: dict[int | None, int] = {}
        self._rr: list[int | None] = []             # round-robin order of task keys (None = work without a task)

    # ------------------------------------------------------------ status

    @property
    def queued_count(self) -> int:
        return len(self._queue)

    @property
    def running_count(self) -> int:
        return self._inflight_candidates

    def add_listener(self, fn: Callable[[Candidate, Alpha | None], Any]) -> None:
        self._listeners.append(fn)

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        async with db.write() as s:
            # simulations that were running when the app closed: resume polling if we know where, else requeue
            sims = list(await s.scalars(select(Simulation).where(Simulation.status == "SIMULATING")))
            resumable = [(sim.location, list(sim.candidate_ids or [])) for sim in sims if sim.location]
            for sim in sims:
                if not sim.location:
                    sim.status = "LOST"
            await s.execute(update(Candidate).where(Candidate.status == "simulating",
                                                    Candidate.sim_id.is_(None)).values(status="queued"))
            rows = (await s.execute(select(Candidate.id, Candidate.task_id).where(Candidate.status == "queued")
                                    .order_by(Candidate.id))).all()
        queued = [cid for cid, _ in rows]
        self._task_of.update({cid: tid for cid, tid in rows if tid is not None})
        self._queue.extend(queued)
        for location, ids in resumable:
            self._spawn(self._resume(location, ids))
        self._task = asyncio.create_task(self._loop(), name="simulator-loop")
        if queued or resumable:
            bus.publish("simulate", f"Resumed {len(queued)} queued and {len(resumable)} in-flight simulations.")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        for task in list(self._inflight):
            task.cancel()

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._inflight.add(task)

        def _done(t: asyncio.Task) -> None:
            self._inflight.discard(t)
            self._wake.set()
            if not t.cancelled() and t.exception():
                log.error("simulation task failed", exc_info=t.exception())

        task.add_done_callback(_done)
        return task

    def pause(self, reason: str = "paused by user") -> None:
        self.paused, self.pause_reason = True, reason

    def resume(self) -> None:
        self.paused, self.pause_reason = False, None
        self._quota_warned = False
        self._wake.set()

    # ------------------------------------------------------------ preparing candidates

    async def prepare(self, specs: list[CandidateSpec], *, batch_id: int | None = None, family_cap: int | None = None,
                      initial_status: str = "new", task_id: int | None = None,
                      history_cap: int | None = None) -> PrepareResult:
        """Validate, de-duplicate and store candidates.

        family_cap:  at most N candidates per expression skeleton in this call.
        history_cap: skip skeletons already simulated N+ times in this region/delay ("shapes this market has already
                     answered"). Skip reasons are counted in `PrepareResult.left_out`.
        """
        result = PrepareResult()
        staged: list[tuple[CandidateSpec, dict[str, Any], Any]] = []
        for spec in specs:
            settings = normalize_settings(spec.settings)
            region, delay, universe = settings["region"], settings["delay"], settings["universe"]
            catalog = await get_catalog(region, delay, universe)
            staged.append((spec, settings, validate(spec.expression, catalog)))

        hashes = [(canonical_hash(vr.program), settings_hash(st)) for _, st, vr in staged if vr.ok and vr.program]
        existing = await self._existing_alphas(hashes)
        history: dict[tuple[str, str, int], int] = {}
        if history_cap:
            skels = {(skeleton_hash(vr.program), st["region"], st["delay"]) for _, st, vr in staged
                     if vr.ok and vr.program}
            history = await self._skeleton_history(skels)

        seen: set[tuple[str, str]] = set()
        families: dict[str, int] = {}
        rows: list[Candidate] = []
        for spec, settings, vr in staged:
            region, delay, universe = settings["region"], settings["delay"], settings["universe"]
            sh = settings_hash(settings)
            base = dict(expression=spec.expression.strip(), settings=settings, settings_hash=sh, region=region,
                        delay=delay, origin=spec.origin, parent_id=spec.parent_id,
                        parent_alpha_id=spec.parent_alpha_id, batch_id=batch_id, hypothesis=spec.hypothesis,
                        rationale=spec.rationale, dataset_id=spec.dataset_id,
                        task_id=task_id if task_id is not None else spec.task_id)
            if not vr.ok or vr.program is None:
                cand = Candidate(**base, status="invalid",
                                 error="; ".join(i.message for i in vr.errors)[:2000] or "invalid expression")
                rows.append(cand)
                result.invalid.append(cand)
                result.leave("failed local validation")
                continue
            ch, sk = canonical_hash(vr.program), skeleton_hash(vr.program)
            if (ch, sh) in seen:
                result.skipped += 1
                result.leave("duplicate in this batch")
                continue
            seen.add((ch, sh))
            if family_cap:
                families[sk] = families.get(sk, 0) + 1
                if families[sk] > family_cap:
                    result.skipped += 1
                    result.leave("too many variants of one shape in this batch")
                    continue
            if history_cap and history.get((sk, region, delay), 0) >= history_cap:
                result.skipped += 1
                result.leave("shape this market has already answered")
                continue
            m = vr.metrics or compute_metrics(vr.program)
            dataset_id = spec.dataset_id
            if not dataset_id and m.non_group_fields:
                fmap = await field_dataset_map(region, delay, universe)
                ds = sorted({fmap[f] for f in m.non_group_fields if f in fmap})
                dataset_id = ds[0] if len(ds) == 1 else (",".join(ds) if ds else None)
            base.update(canonical_hash=ch, skeleton_hash=sk, fields=sorted(m.fields),
                        operators=sorted(m.unique_operators), op_count=m.operator_count,
                        field_count=len(m.non_group_fields), dataset_id=dataset_id)
            reuse_alpha = existing.get((ch, sh))
            if reuse_alpha:
                cand = Candidate(**base, status="reused", alpha_id=reuse_alpha, finished_at=utcnow())
                rows.append(cand)
                result.reused.append(cand)
                result.leave("already simulated (result reused)")
                continue
            cand = Candidate(**base, status=initial_status)
            rows.append(cand)
            result.created.append(cand)
        if rows:
            async with db.write() as s:
                s.add_all(rows)
        return result

    async def _existing_alphas(self, hashes: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
        """Alpha already simulated for (canonical hash, settings hash), looked up in batches."""
        out: dict[tuple[str, str], str] = {}
        wanted = set(hashes)
        canon = sorted({c for c, _ in wanted})
        async with db.session() as s:
            for i in range(0, len(canon), 400):
                chunk = canon[i:i + 400]
                for aid, ch, sh in (await s.execute(select(Alpha.id, Alpha.canonical_hash, Alpha.settings_hash).where(
                        Alpha.canonical_hash.in_(chunk)))).all():
                    if (ch, sh) in wanted:
                        out.setdefault((ch, sh), aid)
                for aid, ch, sh in (await s.execute(select(Candidate.alpha_id, Candidate.canonical_hash,
                                                           Candidate.settings_hash).where(
                        Candidate.canonical_hash.in_(chunk), Candidate.alpha_id.is_not(None)))).all():
                    if (ch, sh) in wanted:
                        out.setdefault((ch, sh), aid)
        return out

    async def _skeleton_history(self, keys: set[tuple[str, str, int]]) -> dict[tuple[str, str, int], int]:
        skels = sorted({k[0] for k in keys})
        out: dict[tuple[str, str, int], int] = {}
        async with db.session() as s:
            for i in range(0, len(skels), 400):
                rows = (await s.execute(select(Candidate.skeleton_hash, Candidate.region, Candidate.delay, func.count())
                                        .where(Candidate.skeleton_hash.in_(skels[i:i + 400]),
                                               Candidate.status == "done")
                                        .group_by(Candidate.skeleton_hash, Candidate.region, Candidate.delay))).all()
                for sk, region, delay, n in rows:
                    out[(sk, region, delay)] = n
        return out

    async def enqueue(self, candidate_ids: list[int]) -> int:
        if not candidate_ids:
            return 0
        async with db.write() as s:
            await s.execute(update(Candidate).where(Candidate.id.in_(candidate_ids),
                                                    Candidate.status.in_(["new", "approved"])).values(status="queued"))
            rows = (await s.execute(select(Candidate.id, Candidate.task_id).where(
                Candidate.id.in_(candidate_ids), Candidate.status == "queued").order_by(Candidate.id))).all()
        ids = [cid for cid, _ in rows]
        self._task_of.update({cid: tid for cid, tid in rows if tid is not None})
        self._queue.extend(ids)
        self._wake.set()
        return len(ids)

    async def submit(self, specs: list[CandidateSpec], *, batch_id: int | None = None,
                     family_cap: int | None = None) -> PrepareResult:
        result = await self.prepare(specs, batch_id=batch_id, family_cap=family_cap)
        await self.enqueue([c.id for c in result.created])
        return result

    async def cancel_queued(self, batch_id: int | None = None, task_id: int | None = None) -> int:
        async with db.write() as s:
            stmt = update(Candidate).where(Candidate.status == "queued")
            if batch_id is not None:
                stmt = stmt.where(Candidate.batch_id == batch_id)
            if task_id is not None:
                stmt = stmt.where(Candidate.task_id == task_id)
            res = await s.execute(stmt.values(status="cancelled", finished_at=utcnow()))
            if batch_id is None and task_id is None:
                cancelled_ids = set(self._queue)
            else:
                q = select(Candidate.id).where(Candidate.status == "cancelled")
                if batch_id is not None:
                    q = q.where(Candidate.batch_id == batch_id)
                if task_id is not None:
                    q = q.where(Candidate.task_id == task_id)
                cancelled_ids = set(await s.scalars(q))
        self._queue = deque(i for i in self._queue if i not in cancelled_ids)
        for cid in cancelled_ids:
            self._task_of.pop(cid, None)
            self._resolve(cid)
        return res.rowcount or 0

    # ------------------------------------------------------------ research-task controls

    def set_task_cores(self, task_id: int, cores: int) -> None:
        self.task_cores[task_id] = max(1, int(cores))
        self._wake.set()

    def pause_task(self, task_id: int) -> None:
        self.paused_tasks.add(task_id)

    def resume_task(self, task_id: int) -> None:
        self.paused_tasks.discard(task_id)
        self._wake.set()

    def task_queued(self, task_id: int) -> int:
        return sum(1 for t in self._task_of.values() if t == task_id)

    def task_inflight(self, task_id: int) -> int:
        return self._inflight_task.get(task_id, 0)

    def cores_in_use(self) -> int:
        return sum(v for k, v in self._inflight_task.items() if k is not None)

    async def wait_for(self, candidate_ids: list[int], timeout: float | None = None) -> None:
        loop = asyncio.get_running_loop()
        futures = []
        for cid in candidate_ids:
            fut = loop.create_future()
            self._waiters.setdefault(cid, []).append(fut)
            futures.append((cid, fut))
        async with db.session() as s:
            done = set(await s.scalars(select(Candidate.id).where(Candidate.id.in_(candidate_ids),
                                                                  Candidate.status.in_(TERMINAL))))
        for cid in done:
            self._resolve(cid)
        pending = [f for _cid, f in futures if not f.done()]
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), timeout)

    def _resolve(self, cid: int) -> None:
        for fut in self._waiters.pop(cid, []):
            if not fut.done():
                fut.set_result(None)

    # ------------------------------------------------------------ running

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            try:
                await self._dispatch()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("simulator dispatch failed")
                await asyncio.sleep(5)

    async def _dispatch(self) -> None:
        if self.paused or not self._queue or not brain.client.logged_in:
            return
        cfg = app_settings.brain
        while self._queue and len(self._inflight) < max(1, brain.client.slots.limit):
            pack = await self._next_pack(cfg.multi_size)
            if not pack:
                return
            if not brain.client.quota.can_spend(len(pack), cfg.quota_reserve):
                self._requeue_front(pack)
                if not self._quota_warned:
                    bus.publish("simulate", f"Daily simulation quota reserve reached ({cfg.quota_reserve} kept); "
                                            "queue paused until the quota resets.", "warning")
                    self._quota_warned = True
                return
            self._spawn(self._run_pack(pack))

    def _capped(self, key: int | None) -> bool:
        """Work without a task only waits for global slots; a task also respects its pause flag and its cores."""
        if key is None:
            return False
        if key in self.paused_tasks:
            return True
        return self._inflight_task.get(key, 0) >= self.task_cores.get(key, app_settings.research.task_cores)

    async def _next_pack(self, size: int) -> list[Candidate]:
        """Next multi-simulation (≤ 10 alphas sharing region/delay/type/language), taken round-robin from the
        research tasks that have a free core; a pack never mixes tasks."""
        by_key: dict[int | None, list[int]] = {}
        for cid in self._queue:
            lst = by_key.setdefault(self._task_of.get(cid), [])
            if len(lst) < 200:
                lst.append(cid)
        keys = [k for k in by_key if not self._capped(k)]
        for k in keys:
            if k not in self._rr:
                self._rr.append(k)
        for key in [k for k in self._rr if k in keys]:
            head = by_key[key]
            async with db.session() as s:
                rows = {c.id: c for c in await s.scalars(select(Candidate).where(Candidate.id.in_(head)))}
            valid = [rows[i] for i in head if i in rows and rows[i].status == "queued"]
            stale = {i for i in head if i not in rows or rows[i].status != "queued"}
            if stale:
                self._queue = deque(i for i in self._queue if i not in stale)
                for i in stale:
                    self._task_of.pop(i, None)
            if not valid:
                continue
            gk = group_key(valid[0].settings)
            pack = [c for c in valid if group_key(c.settings) == gk][:max(1, min(10, size))]
            ids = {c.id for c in pack}
            self._queue = deque(i for i in self._queue if i not in ids)
            for i in ids:
                self._task_of.pop(i, None)
            self._rr.remove(key)
            self._rr.append(key)
            return pack
        return []

    def _requeue_front(self, pack: list[Candidate]) -> None:
        self._queue.extendleft(reversed([c.id for c in pack]))
        for c in pack:
            if c.task_id is not None:
                self._task_of[c.id] = c.task_id

    async def _set_status(self, ids: list[int], **values: Any) -> None:
        async with db.write() as s:
            await s.execute(update(Candidate).where(Candidate.id.in_(ids)).values(**values))

    async def _run_pack(self, pack: list[Candidate]) -> None:
        key = pack[0].task_id
        self._inflight_task[key] = self._inflight_task.get(key, 0) + 1
        try:
            await self._run_pack_inner(pack)
        finally:
            self._inflight_task[key] = max(0, self._inflight_task.get(key, 1) - 1)
            self._wake.set()

    async def _run_pack_inner(self, pack: list[Candidate]) -> None:
        ids = [c.id for c in pack]
        self._inflight_candidates += len(pack)
        await self._set_status(ids, status="simulating")
        payloads = [regular_payload(c.expression, c.settings) for c in pack]
        sim_holder: dict[str, str] = {}

        def on_started(location: str) -> None:
            sim_holder["location"] = location
            sim_id = location.rstrip("/").rsplit("/", 1)[-1]
            sim_holder["id"] = sim_id
            asyncio.get_running_loop().create_task(self._record_sim(sim_id, location, ids, pack[0].region,
                                                                    "multi" if len(pack) > 1 else "single"))

        try:
            outcomes = await brain.sims.run(payloads, on_started=on_started)
        except (ConcurrencyLimit, PersonaRequired) as exc:
            await self._set_status(ids, status="queued")
            self._requeue_front(pack)
            if isinstance(exc, PersonaRequired):
                self.pause("BRAIN biometric verification required")
                bus.publish("simulate", "Queue paused: BRAIN needs biometric verification (see Sync with BRAIN).",
                            "warning")
            self._inflight_candidates -= len(pack)
            return
        except QuotaExhausted as exc:
            await self._set_status(ids, status="queued")
            self._requeue_front(pack)
            self.pause(f"quota: {exc.message}")
            bus.publish("simulate", f"Queue paused — BRAIN reports the daily simulation limit: {exc.message}", "warning")
            self._inflight_candidates -= len(pack)
            return
        except (SimulationRejected, BrainError) as exc:
            await self._set_status(ids, status="error", error=exc.message[:2000], finished_at=utcnow())
            bus.publish("simulate", f"{len(pack)} simulation(s) failed: {exc.message}", "error")
            for cid in ids:
                self._resolve(cid)
            self._inflight_candidates -= len(pack)
            return
        try:
            await self._ingest(pack, outcomes, sim_holder.get("id"))
        finally:
            self._inflight_candidates -= len(pack)
            self._wake.set()

    async def _record_sim(self, sim_id: str, location: str, ids: list[int], region: str | None, kind: str) -> None:
        async with db.write() as s:
            if await s.get(Simulation, sim_id) is None:
                s.add(Simulation(id=sim_id, kind=kind, location=location, candidate_ids=ids, region=region))
            await s.execute(update(Candidate).where(Candidate.id.in_(ids)).values(sim_id=sim_id))

    async def _resume(self, location: str, ids: list[int]) -> None:
        while not brain.client.logged_in:
            await asyncio.sleep(5)
        async with db.session() as s:
            pack = [c for c in await s.scalars(select(Candidate).where(Candidate.id.in_(ids)).order_by(Candidate.id))]
        if not pack:
            return
        self._inflight_candidates += len(pack)
        try:
            outcomes = await brain.sims.wait(location, len(pack))
            await self._ingest(pack, outcomes, location.rstrip("/").rsplit("/", 1)[-1])
        except BrainError as exc:
            await self._set_status(ids, status="queued", sim_id=None)
            self._queue.extend(ids)
            self._task_of.update({c.id: c.task_id for c in pack if c.task_id is not None})
            log.warning("could not resume %s: %s", location, exc)
        finally:
            self._inflight_candidates -= len(pack)

    async def _ingest(self, pack: list[Candidate], outcomes, sim_id: str | None) -> None:  # noqa: ANN001
        ok = 0
        best: tuple[float, str] | None = None
        for cand, outcome in zip(pack, outcomes):
            alpha_row: Alpha | None = None
            if outcome.alpha_id:
                try:
                    data = await brain.alphas.get(outcome.alpha_id)
                    alpha_row = await upsert_alpha(data, candidate_id=cand.id, origin="app")
                    await self._set_status([cand.id], status="done", alpha_id=outcome.alpha_id,
                                           finished_at=utcnow(), error=None)
                    await journal(alpha_row, cand)
                    ok += 1
                    if alpha_row.sharpe is not None and (best is None or alpha_row.sharpe > best[0]):
                        best = (alpha_row.sharpe, alpha_row.id)
                except BrainError as exc:
                    await self._set_status([cand.id], status="error", alpha_id=outcome.alpha_id,
                                           error=f"fetch failed: {exc.message}", finished_at=utcnow())
            else:
                await self._set_status([cand.id], status="error",
                                       error=(outcome.message or outcome.status or "simulation failed")[:2000],
                                       finished_at=utcnow())
            for fn in list(self._listeners):
                try:
                    res = fn(cand, alpha_row)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    log.exception("simulation listener failed")
            self._resolve(cand.id)
        if sim_id:
            async with db.write() as s:
                sim = await s.get(Simulation, sim_id)
                if sim is not None:
                    sim.status = "COMPLETE" if ok else "ERROR"
                    sim.finished_at = utcnow()
        best_txt = f" · best Sharpe {best[0]:.2f}" if best else ""
        bus.publish("simulate", f"Simulated {len(pack)} alpha(s): {ok} completed, {len(pack) - ok} failed{best_txt}.",
                    "success" if ok else "warning")

    # ------------------------------------------------------------ stats

    async def counts_today(self) -> dict[str, int]:
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        async with db.session() as s:
            simulated = await s.scalar(select(func.count(Candidate.id)).where(
                Candidate.status == "done", Candidate.finished_at >= start))
            generated = await s.scalar(select(func.count(Candidate.id)).where(
                Candidate.created_at >= start, Candidate.origin.in_(GENERATED_ORIGINS)))
        return {"simulated": simulated or 0, "generated": generated or 0}


simulator = SimulationService()
