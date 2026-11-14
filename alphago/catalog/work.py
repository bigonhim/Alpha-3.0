"""Work in flight: every research and lab task holding queued simulations or cores (alpha-harness dashboard/work).

Pausing a task takes its waiting simulations off the queue; they are sent when it is run again from Tasks. Anything
already sent to BRAIN keeps running.
"""

from __future__ import annotations

from dataclasses import dataclass

RESEARCH_ACTIVE = ("queued", "running", "paused", "analyzing", "generating", "simulating", "optimizing")
LAB_ACTIVE = ("QUEUED", "RUNNING", "PAUSED")


@dataclass
class WorkRow:
    kind: str            # research | lab
    task_id: int
    name: str
    status: str
    in_flight: int
    waiting: int

    @property
    def running(self) -> bool:
        return self.in_flight > 0

    @property
    def pausable(self) -> bool:
        return self.status.lower() not in ("paused",) and (self.waiting > 0 or self.in_flight > 0)


async def rows() -> list[WorkRow]:
    from ..engine.research import KINDS, research
    from ..engine.simulator import simulator
    from ..labs.runner import core_key, runner
    out: list[WorkRow] = []
    for t in await research.list(limit=200):
        if t.status not in RESEARCH_ACTIVE:
            continue
        sets = ", ".join(t.dataset_ids or []) or "your alphas"
        out.append(WorkRow("research", t.id, f"#{t.id} {KINDS.get(t.kind, t.kind)} · {sets} ({t.region} D{t.delay})",
                           t.status, simulator.task_inflight(t.id), simulator.task_queued(t.id)))
    for t in await runner.list(limit=200):
        if t.status not in LAB_ACTIVE:
            continue
        summary = await runner.summary(t.id)
        key = core_key(t.id)
        out.append(WorkRow("lab", t.id, t.name or f"{t.lab.replace('_', ' ').title()} #{t.id}", t.status,
                           max(simulator.task_inflight(key), summary["running"]),
                           simulator.task_queued(key) + summary["queued"]))
    return [r for r in out if r.waiting or r.in_flight or r.status.lower() == "paused"]


async def pause(row: WorkRow) -> None:
    if row.kind == "research":
        from ..engine.research import research
        await research.pause(row.task_id)
    else:
        from ..labs.runner import runner
        await runner.pause(row.task_id)


async def pause_all() -> int:
    n = 0
    for row in await rows():
        if row.pausable:
            await pause(row)
            n += 1
    return n
