"""Alpha Autopilot: dataset choice → generation → (approval) → simulation → optimisation rounds → check → robustness.

Modes:
- assisted (default): every generated batch waits for the user to approve/edit it before any simulation is spent.
- autopilot: fully automatic loop (explicit opt-in; BRAIN's consultant guidelines discourage automated idea
  generation, so the UI warns before enabling it).
Submission is never automatic.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update

from ..brain import brain
from ..brain.simulations import normalize_settings
from ..config import settings as app_settings
from ..db import db
from ..db.models import Alpha, AutopilotRun, Batch, Candidate
from ..events import bus
from ..fastexpr import scale_windows
from ..llm import LLMError, llm
from . import actions
from .datasets import choose_dataset
from .evaluator import promising_or_better
from .generators import MOVES, llm_ideas, mutation_ideas, template_ideas
from .optimizer import optimize_alpha
from .simulator import CandidateSpec, simulator

log = logging.getLogger(__name__)

GOALS = {
    "general": "Find alphas that pass every BRAIN submission test and are uncorrelated with existing alphas.",
    "power_pool": "Power Pool alphas: Sharpe ≥ 1.0 with ≤ 8 operators (repeats counted) and ≤ 3 data fields; "
                  "simple, robust, low correlation.",
    "pyramid": "Light new pyramids: alphas in this dataset category that pass every test.",
    "genius": "Advance the Genius level: simple (few operators/fields), diverse, submittable alphas.",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AutopilotConfig:
    mode: str = "assisted"
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    batch_size: int = 12
    max_rounds: int = 3
    quota_share: float = 0.6
    stop_after_empty_batches: int = 6
    generator_mix: dict[str, float] = field(default_factory=lambda: {"llm": 0.5, "template": 0.3, "mutation": 0.2})
    dataset_strategy: str = "opportunity"
    pinned_datasets: list[str] = field(default_factory=list)
    goal: str = "general"
    family_cap: int = 3
    robustness_gate: bool = True
    max_batches: int = 0          # 0 = unlimited
    check_top_k: int = 3          # run /check on up to k strong alphas per batch
    neutralization: str | None = None

    @classmethod
    def from_settings(cls) -> "AutopilotConfig":
        r = app_settings.research
        return cls(mode=r.mode, region=r.region, universe=r.universe, delay=r.delay, batch_size=r.batch_size,
                   max_rounds=r.max_rounds, quota_share=r.quota_share,
                   stop_after_empty_batches=r.stop_after_empty_batches, generator_mix=dict(r.generator_mix),
                   dataset_strategy=r.dataset_strategy, pinned_datasets=list(r.pinned_datasets),
                   family_cap=r.family_cap_per_batch, robustness_gate=r.robustness_gate)


class Autopilot:
    def __init__(self) -> None:
        self.state = "stopped"          # stopped | running | awaiting_approval | stopping
        self.config: AutopilotConfig | None = None
        self.run_id: int | None = None
        self.batch_id: int | None = None
        self._task: asyncio.Task | None = None
        self._approval: asyncio.Future | None = None
        self._stop = asyncio.Event()
        self.rng = random.Random()
        self.status_line = "Stopped"

    @property
    def running(self) -> bool:
        return self.state in ("running", "awaiting_approval", "stopping")

    # ------------------------------------------------------------ control

    async def start(self, config: AutopilotConfig) -> int:
        if self.running:
            raise RuntimeError("Autopilot is already running")
        if not brain.client.logged_in:
            raise RuntimeError("Sign in to BRAIN first")
        self.config = config
        self._stop.clear()
        async with db.write() as s:
            run = AutopilotRun(mode=config.mode, goal=config.goal, config=asdict(config), status="running")
            s.add(run)
            await s.flush()
            self.run_id = run.id
        self.state = "running"
        self._task = asyncio.create_task(self._run(), name="autopilot")
        bus.publish("autopilot", f"Autopilot started ({config.mode} mode, {config.region} D{config.delay} "
                                 f"{config.universe}, goal: {config.goal}).", "success")
        return self.run_id

    async def stop(self, reason: str = "stopped by user") -> None:
        if not self.running:
            return
        self.state = "stopping"
        self._stop.set()
        if self._approval and not self._approval.done():
            self._approval.set_result([])
        removed = 0
        halted = 0
        if self.run_id is not None:
            async with db.session() as s:
                open_batches = list(await s.scalars(select(Batch.id).where(
                    Batch.run_id == self.run_id, Batch.status.in_(["pending", "awaiting_approval", "running"]))))
            for bid in open_batches:
                removed += await simulator.cancel_queued(batch_id=bid)
                halted += 1
            async with db.write() as s:
                await s.execute(update(Batch).where(Batch.id.in_(open_batches)).values(status="halted",
                                                                                      finished_at=utcnow()))
                run = await s.get(AutopilotRun, self.run_id)
                if run:
                    run.status, run.reason, run.stopped_at = "stopped", reason, utcnow()
        bus.publish("autopilot", f"Autopilot stopped: {halted} batches halted, {removed} queued simulations removed.")
        if self._task and self._task is not asyncio.current_task():
            self._task.cancel()
        self.state = "stopped"
        self.status_line = f"Stopped — {reason}"

    def approve(self, candidate_ids: list[int]) -> None:
        if self._approval and not self._approval.done():
            self._approval.set_result(list(candidate_ids))

    async def replace_edited(self, edits: dict[int, str]) -> tuple[dict[int, int], list[str]]:
        """Expressions edited in the approval grid become new, re-validated candidates (hashes, fields and
        operator counts recomputed); the originals are set aside. Returns ({old_id: new_id}, problems)."""
        mapping: dict[int, int] = {}
        problems: list[str] = []
        async with db.session() as s:
            originals = {c.id: c for c in await s.scalars(select(Candidate).where(Candidate.id.in_(list(edits))))}
        for cid, expr in edits.items():
            c = originals.get(cid)
            if c is None or expr.strip() == (c.expression or "").strip():
                continue
            res = await simulator.prepare([CandidateSpec(expression=expr, settings=c.settings, origin=c.origin,
                                                         hypothesis=c.hypothesis, rationale=c.rationale,
                                                         dataset_id=c.dataset_id, parent_id=c.id)],
                                          batch_id=c.batch_id, initial_status="approved")
            new = (res.created or res.reused or [None])[0]
            if res.invalid or new is None:
                problems.append(f"#{cid}: {(res.invalid[0].error if res.invalid else 'duplicate')}")
                continue
            mapping[cid] = new.id
            async with db.write() as s:
                await s.execute(update(Candidate).where(Candidate.id == cid).values(
                    status="rejected_by_user", error="replaced by an edited expression", finished_at=utcnow()))
        return mapping, problems

    # ------------------------------------------------------------ main loop

    async def _run(self) -> None:
        cfg = self.config
        assert cfg is not None
        start_remaining = brain.client.quota.effective_remaining
        budget = int(start_remaining * cfg.quota_share)
        empty = 0
        n_batches = 0
        reason = "finished"
        try:
            while not self._stop.is_set():
                spent = await self._run_spent()   # counted from the database, so it survives midnight and restarts
                if spent >= budget:
                    reason = f"quota budget reached ({spent}/{budget} simulations)"
                    break
                if cfg.mode == "autopilot":
                    from .research import autopilot_sims_today
                    used, cap = await autopilot_sims_today(), app_settings.research.autopilot_daily_cap
                    if used + cfg.batch_size > cap:
                        reason = f"daily Autopilot cap reached ({used}/{cap} simulations)"
                        break
                if not brain.client.quota.can_spend(cfg.batch_size, app_settings.brain.quota_reserve):
                    reason = "daily quota reserve reached"
                    break
                if cfg.max_batches and n_batches >= cfg.max_batches:
                    reason = f"{n_batches} batches done"
                    break
                if not brain.client.logged_in:
                    self.status_line = "Waiting for BRAIN session…"
                    await asyncio.sleep(10)
                    continue
                n_batches += 1
                found = await self._batch()
                empty = 0 if found else empty + 1
                if empty >= cfg.stop_after_empty_batches:
                    reason = f"{empty} batches in a row without a strong alpha"
                    break
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log.exception("autopilot crashed")
            reason = f"error: {exc}"
            bus.publish("autopilot", f"Autopilot error: {exc}", "error")
        if self.state != "stopped":
            await self.stop(reason)

    async def _run_spent(self) -> int:
        """Simulations this run has sent (queued, running or finished)."""
        if self.run_id is None:
            return 0
        from sqlalchemy import func
        async with db.session() as s:
            return await s.scalar(select(func.count(Candidate.id)).join(Batch, Batch.id == Candidate.batch_id).where(
                Batch.run_id == self.run_id, Candidate.status.in_(["queued", "simulating", "done", "error"]))) or 0

    async def _generate(self, scope: dict[str, Any], dataset_id: str | None, n: int, move: str) -> list[CandidateSpec]:
        cfg = self.config
        assert cfg is not None
        mix = cfg.generator_mix or {"llm": 1.0}
        total = sum(mix.values()) or 1.0
        n_llm = round(n * mix.get("llm", 0) / total)
        n_mut = round(n * mix.get("mutation", 0) / total)
        n_tpl = max(0, n - n_llm - n_mut)
        base = {"neutralization": cfg.neutralization} if cfg.neutralization else {}
        specs: list[CandidateSpec] = []
        if n_llm:
            if llm.configure():
                try:
                    self.status_line = f"Asking {llm.label} for {n_llm} ideas ({move})…"
                    specs += await llm_ideas(scope, dataset_id=dataset_id, n=n_llm, goal=GOALS.get(cfg.goal, cfg.goal),
                                             move=move, base_settings=base)
                except LLMError as exc:
                    bus.publish("generate", f"{llm.label} unavailable ({exc.message}); using templates instead.",
                                "warning")
                    n_tpl += n_llm
            else:
                n_tpl += n_llm
        if n_mut:
            muts = await mutation_ideas(scope, n=n_mut, dataset_id=dataset_id, rng=self.rng)
            specs += muts
            n_tpl += n_mut - len(muts)
        if n_tpl > 0:
            specs += await template_ideas(scope, dataset_id=dataset_id, n=n_tpl, base_settings=base, rng=self.rng)
        if cfg.goal == "power_pool":
            from .programs import power_pool_shape_ok
            specs = [s for s in specs if power_pool_shape_ok(s.expression)]
        return specs

    async def _wait_approval(self, batch_id: int) -> list[int]:
        loop = asyncio.get_running_loop()
        self._approval = loop.create_future()
        self.state = "awaiting_approval"
        self.status_line = f"Batch #{batch_id} is waiting for your approval"
        bus.publish("autopilot", f"Batch #{batch_id} generated — review and approve it on the Autopilot page.")
        try:
            return await self._approval
        finally:
            self._approval = None
            if self.state == "awaiting_approval":
                self.state = "running"

    async def _batch(self) -> bool:
        cfg = self.config
        assert cfg is not None
        scope = {"region": cfg.region, "delay": cfg.delay, "universe": cfg.universe}
        ds = await choose_dataset(cfg.region, cfg.delay, cfg.universe, cfg.dataset_strategy, cfg.pinned_datasets,
                                  self.rng)
        dataset_id = ds.id if ds else None
        move = self.rng.choices(list(MOVES), weights=[0.45, 0.25, 0.15, 0.15])[0]
        if cfg.goal in ("power_pool", "genius"):
            move = "SIMPLIFY"
        async with db.write() as s:
            batch = Batch(run_id=self.run_id, source="autopilot", region=cfg.region, universe=cfg.universe,
                          delay=cfg.delay, dataset_id=dataset_id, strategy=f"{move.lower()}/{cfg.goal}", status="pending")
            s.add(batch)
            await s.flush()
            bid = batch.id
        self.batch_id = bid
        self.status_line = f"Batch #{bid}: generating ideas for {dataset_id or 'price/volume'}"
        specs = await self._generate(scope, dataset_id, cfg.batch_size, move)
        initial = "new" if cfg.mode == "assisted" else "approved"
        result = await simulator.prepare(specs, batch_id=bid, family_cap=cfg.family_cap, initial_status=initial)
        bus.publish("generate", f"{len(result.created)} new alpha candidates generated for batch #{bid} "
                                f"({dataset_id or 'pv'}, {move.lower()}); {len(result.invalid)} invalid, "
                                f"{len(result.reused)} cached.")
        ids = [c.id for c in result.created]
        if cfg.mode == "assisted" and ids:
            async with db.write() as s:
                await s.execute(update(Batch).where(Batch.id == bid).values(status="awaiting_approval"))
            approved = await self._wait_approval(bid)
            if self._stop.is_set():
                return False
            approved_set = set(approved)
            rejected = [i for i in ids if i not in approved_set]
            if rejected:
                async with db.write() as s:
                    await s.execute(update(Candidate).where(Candidate.id.in_(rejected)).values(
                        status="rejected_by_user", finished_at=utcnow()))
            # approved ids may include re-validated replacements of expressions the user edited
            ids = list(dict.fromkeys(approved))
        async with db.write() as s:
            await s.execute(update(Batch).where(Batch.id == bid).values(
                status="running", n_generated=len(specs), n_valid=len(result.created) + len(result.reused),
                n_reused=len(result.reused)))
        queued = await simulator.enqueue(ids)
        bus.publish("simulate", f"{queued} simulations queued for batch #{bid} (round 1); {len(result.reused)} reused.")
        self.status_line = f"Batch #{bid}: simulating {queued} alphas (round 1)"
        await simulator.wait_for(ids)

        # optimisation rounds on near misses
        for rnd in range(2, cfg.max_rounds + 1):
            if self._stop.is_set():
                return False
            near = await self._batch_alphas(bid, verdicts=("promising",), limit=3)
            if not near:
                break
            new_ids: list[int] = []
            for a in near:
                new_ids += await optimize_alpha(a.id, batch_id=bid, use_llm=cfg.generator_mix.get("llm", 0) > 0)
            self.status_line = f"Batch #{bid}: optimisation round {rnd} ({len(new_ids)} variants)"
            bus.publish("optimize", f"Batch #{bid} round {rnd}: {len(new_ids)} optimisation variants of "
                                    f"{len(near)} near-misses.")
            await simulator.wait_for(new_ids)

        # submission checks on the best strong alphas (+ robustness gate)
        strong = await self._batch_alphas(bid, verdicts=("strong",), limit=cfg.check_top_k)
        for a in strong:
            if self._stop.is_set():
                break
            try:
                state, _ = await actions.run_check(a.id)
                if state == "pass" and cfg.robustness_gate:
                    await self._robustness(a.id, bid)
            except Exception as exc:
                bus.publish("check", f"Check failed for {a.id}: {getattr(exc, 'message', exc)}", "warning")
        counts = await self._batch_counts(bid)
        async with db.write() as s:
            await s.execute(update(Batch).where(Batch.id == bid).values(
                status="done", finished_at=utcnow(), n_simulated=counts["simulated"], n_promising=counts["promising"],
                n_strong=counts["strong"], n_ready=counts["ready"], n_rejected=counts["rejected"]))
        bus.publish("autopilot", f"Batch #{bid} complete: {counts['strong']} strong, {counts['ready']} ready, "
                                 f"{counts['rejected']} rejected.", "success" if counts["ready"] else "info")
        return counts["strong"] + counts["ready"] > 0

    async def _batch_alphas(self, bid: int, verdicts: tuple[str, ...], limit: int) -> list[Alpha]:
        async with db.session() as s:
            ids = list(await s.scalars(select(Candidate.alpha_id).where(Candidate.batch_id == bid,
                                                                        Candidate.alpha_id.is_not(None))))
            if not ids:
                return []
            rows = list(await s.scalars(select(Alpha).where(Alpha.id.in_(ids), Alpha.verdict.in_(verdicts))
                                        .order_by(Alpha.score.desc().nulls_last()).limit(limit)))
        return rows

    async def _batch_counts(self, bid: int) -> dict[str, int]:
        async with db.session() as s:
            cands = list(await s.scalars(select(Candidate).where(Candidate.batch_id == bid)))
            ids = [c.alpha_id for c in cands if c.alpha_id]
            verdicts = list(await s.scalars(select(Alpha.verdict).where(Alpha.id.in_(ids)))) if ids else []
        return {
            "simulated": sum(c.status == "done" for c in cands),
            "promising": sum(promising_or_better(v) for v in verdicts),
            "strong": verdicts.count("strong"),
            "ready": verdicts.count("ready"),
            "rejected": verdicts.count("rejected") + sum(c.status in ("error", "invalid") for c in cands),
        }

    async def _robustness(self, alpha_id: str, bid: int) -> None:
        await check_robustness(alpha_id, bid)


async def check_robustness(alpha_id: str, bid: int | None, task_id: int | None = None) -> None:
    """Chan-style sensitivity check: neighbouring settings/windows should keep ≥ 80% of the Sharpe."""
    async with db.session() as s:
        a = await s.get(Alpha, alpha_id)
    if a is None or not a.code or a.sharpe is None:
        return
    st = normalize_settings(a.settings or {})
    specs = [
        CandidateSpec(expression=a.code, settings={**st, "decay": int(st["decay"]) + 2}, origin="robustness",
                      parent_alpha_id=a.id, hypothesis="robustness: decay +2", task_id=task_id),
        CandidateSpec(expression=scale_windows(a.code, 0.8).to_source(), settings=st, origin="robustness",
                      parent_alpha_id=a.id, hypothesis="robustness: windows ×0.8", task_id=task_id),
        CandidateSpec(expression=scale_windows(a.code, 1.25).to_source(), settings=st, origin="robustness",
                      parent_alpha_id=a.id, hypothesis="robustness: windows ×1.25", task_id=task_id),
    ]
    result = await simulator.submit(specs, batch_id=bid)
    ids = [c.id for c in result.created + result.reused]
    await simulator.wait_for(ids)
    async with db.session() as s:
        alpha_ids = list(await s.scalars(select(Candidate.alpha_id).where(Candidate.id.in_(ids))))
        sharpes = [x for x in await s.scalars(select(Alpha.sharpe).where(Alpha.id.in_(alpha_ids))) if x is not None]
    ratio = (min(sharpes) / a.sharpe) if sharpes and a.sharpe else 0.0
    robust = ratio >= 0.8
    await db.kv_set(f"robust:{alpha_id}", {"ratio": round(ratio, 3), "robust": robust, "n": len(sharpes)})
    if robust:
        bus.publish("check", f"{alpha_id} is ready and robust (neighbours keep {ratio:.0%} of Sharpe) — review it "
                             "on the Alphas page.", "success")
    else:
        async with db.write() as s:
            row = await s.get(Alpha, alpha_id)
            if row and row.verdict == "ready":
                row.verdict = "strong"
        bus.publish("check", f"{alpha_id} passed checks but is fragile (neighbours keep only {ratio:.0%} of "
                             "Sharpe) — kept as 'strong'.", "warning")


autopilot = Autopilot()
