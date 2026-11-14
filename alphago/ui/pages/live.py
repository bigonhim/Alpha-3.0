"""Live Simulations: what every BRAIN simulation core is doing right now, with cancel.

Ported from alpha-harness ``screens/matrix/index.tsx`` (the live cores x Alphas matrix) and ``shell/pause.tsx`` (why
nothing is being sent). One row per running (multi-)simulation from ``simulator.running``: the cores it holds (a GLB
run holds two), the batch key every Alpha in it shares, its task, BRAIN's progress, the time since it started and one
cell per Alpha. Queued work is counted, never drawn. The header's core tiles read the same :func:`core_blocks` and
:func:`pauses`, so the top bar and this page always agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nicegui import ui

from ..components import fmt_int, notify_error, page_header

#: The fields BRAIN requires every child of one multi-simulation to share.
BATCH_KEY = (("Type", "type"), ("Region", "region"), ("Delay", "delay"), ("Instrument Type", "instrumentType"),
             ("Language", "language"))
QUEUED = "Queued work stays queued."


# ── the cores ──────────────────────────────────────────────────────────────────────────────


def core_span(info: dict[str, Any]) -> int:
    """Cores one run holds: a GLB simulation counts twice against BRAIN's concurrency limit."""
    return 2 if str(info.get("region") or "").upper() == "GLB" else 1


@dataclass
class CoreBlock:
    """A run of adjacent cores held by one simulation, or one idle core."""

    start: int
    span: int = 1
    sim_id: str | None = None
    info: dict[str, Any] | None = None

    @property
    def label(self) -> str:
        return f"C{self.start + 1}" if self.span == 1 else f"C{self.start + 1}–{self.start + self.span}"


@dataclass
class Cores:
    blocks: list[CoreBlock] = field(default_factory=list)
    #: Running simulations that do not fit in the cores (BRAIN let more through than the slot count says).
    overflow: int = 0


def core_blocks(running: dict[str, dict[str, Any]], slots: int) -> Cores:
    """Lay the running simulations onto ``slots`` cores, oldest first, each over the cores it holds; the rest idle."""
    slots = max(1, int(slots))
    out = Cores()
    at = 0

    def started(item: tuple[str, dict[str, Any]]) -> str:
        value = item[1].get("started")
        return value.isoformat() if isinstance(value, datetime) else str(value or "")

    for sim_id, info in sorted(running.items(), key=started):
        span = min(core_span(info), slots)
        if at + span > slots:
            out.overflow += 1
            continue
        out.blocks.append(CoreBlock(at, span, sim_id, info))
        at += span
    out.blocks.extend(CoreBlock(i) for i in range(at, slots))
    return out


def core_state(block: CoreBlock, paused: bool) -> str:
    """running | pending | warning | idle — the four states of a core tile."""
    if paused:
        return "warning"
    if block.info is None:
        return "idle"
    return "running" if block.info.get("progress") is not None else "pending"


def slot_count() -> int:
    from ...brain import brain
    return max(1, int(brain.client.slots.capacity))


# ── why nothing is being sent ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pause:
    key: str          # identity | offline | session | limit | busy | paused
    chip: str
    title: str
    body: str


def pauses() -> list[Pause]:
    """Why sending is paused, most pressing first; empty while sending runs."""
    from ...brain import brain
    from ...engine.simulator import simulator
    c = brain.client
    out: list[Pause] = []
    reason = simulator.pause_reason or ""
    if c.state == "persona_required" or reason.startswith("BRAIN biometric"):
        out.append(Pause("identity", "Identity check", "BRAIN needs to verify your identity",
                         f"Sending is paused until BRAIN accepts the check: finish it on Sync with BRAIN. {QUEUED}"))
    elif getattr(c, "persona_expired", False):
        out.append(Pause("identity", "Identity check", "BRAIN's identity check expired",
                         f"It closed before it was finished: start a new check on Sync with BRAIN to resume sending. "
                         f"{QUEUED}"))
    elif c.state == "error":
        out.append(Pause("offline", "Offline", "BRAIN cannot be reached",
                         "This computer has no connection to BRAIN. Sending and reading results resume by themselves "
                         f"once it is back. {QUEUED}"))
    elif not c.logged_in:
        out.append(Pause("session", "Signed out", "BRAIN is not accepting your session",
                         f"Sending is paused until you sign in again on Sync with BRAIN. {QUEUED}"))
    if c.quota.effective_remaining <= 0 or reason.startswith("quota"):
        out.append(Pause("limit", "Daily limit", "BRAIN's daily simulation limit is reached",
                         f"Nothing more will be sent until the allowance resets at midnight US Eastern. {QUEUED}"))
    if c.slots.limit < c.slots.capacity:
        out.append(Pause("busy", "Busy elsewhere", "BRAIN is busy with simulations started elsewhere",
                         f"BRAIN refused a start because too many simulations are running on your account (website or "
                         f"notebook). AlphaGo uses {c.slots.limit} of {c.slots.capacity} cores and takes the rest back "
                         "one a minute."))
    if simulator.paused and not any(p.key in ("identity", "limit") for p in out):
        out.append(Pause("paused", "Paused", "Sending is paused",
                         f"{(reason or 'Paused by you').rstrip('.')}. Run All on Tasks resumes it. {QUEUED}"))
    return out


# ── reading a run ──────────────────────────────────────────────────────────────────────────


async def task_names(keys: set[int]) -> dict[int, str]:
    """A readable name for each task key the simulator uses: negative = a lab task, positive = a research task."""
    from ...db import db
    from ...db.models import ResearchTask
    from ...db.models_labs import LabTask
    names: dict[int, str] = {}
    async with db.session() as s:
        for key in keys:
            if key < 0:
                t = await s.get(LabTask, -key)
                names[key] = f"Task {-key} · {t.name or t.lab}" if t else f"Task {-key}"
            else:
                r = await s.get(ResearchTask, key)
                names[key] = (f"Research {key} · {r.kind} {r.region} D{r.delay} {r.universe}" if r
                              else f"Research {key}")
    return names


async def children_of(ids: list[int]) -> list[dict[str, Any]]:
    """Each Alpha in a run: expression and the settings it was sent with."""
    from sqlalchemy import select

    from ...db import db
    from ...db.models import Candidate
    if not ids:
        return []
    async with db.session() as s:
        rows = {c.id: c for c in await s.scalars(select(Candidate).where(Candidate.id.in_(ids)))}
    out = []
    for cid in ids:
        c = rows.get(cid)
        out.append({"id": cid, "expression": c.expression if c else None,
                    "settings": dict(c.settings or {}) if c else {}})
    return out


def batch_key(info: dict[str, Any], children: list[dict[str, Any]]) -> dict[str, str]:
    settings = next((c["settings"] for c in children if c.get("settings")), {})
    values = {"type": "REGULAR", "region": settings.get("region") or info.get("region"),
              "delay": settings.get("delay"), "instrumentType": settings.get("instrumentType") or "EQUITY",
              "language": settings.get("language") or "FASTEXPR"}
    return {k: "—" if v is None else str(v) for k, v in values.items()}


def elapsed(started: Any) -> str:
    if not isinstance(started, datetime):
        return "—"
    when = started if started.tzinfo else started.replace(tzinfo=timezone.utc)
    seconds = max(0, int((datetime.now(timezone.utc) - when).total_seconds()))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def short_id(sim_id: str) -> str:
    return f"{sim_id[:8]}…" if len(sim_id) > 8 else sim_id


def progress_text(info: dict[str, Any]) -> str:
    p = info.get("progress")
    return "pending" if p is None else f"{max(0.0, min(1.0, float(p))) * 100:.0f}%"


# ── cancel ───────────────────────────────────────────────────────────────────────────────────


async def cancel_dialog(sim_id: str, info: dict[str, Any]) -> None:
    """Confirm, ask BRAIN to cancel, and when BRAIN refuses offer to free the core here."""
    from ...engine.simulator import simulator
    size = int(info.get("size") or len(info.get("ids") or []) or 1)
    what = f"batch {short_id(sim_id)}" if size > 1 else f"simulation {short_id(sim_id)}"

    async def run(force: bool, dialog: ui.dialog) -> None:
        dialog.close()
        try:
            result = await simulator.cancel_running(sim_id, force=force)
        except Exception as exc:  # noqa: BLE001 - a network failure is told, never raised into the page
            notify_error(exc, f"Could not cancel {what}: ")
            return
        if result == "cancelled":
            ui.notify(f"BRAIN acknowledged the cancel of {what}.", type="positive")
        elif result == "forced":
            ui.notify(f"Freed the core held by {what}. Its Alphas are marked cancelled.", type="positive")
        else:
            with ui.dialog() as again, ui.card().classes("ag-card p-5 w-[520px]"):
                ui.label(f"BRAIN did not acknowledge the cancel of {what}").classes("ag-section-title")
                ui.label("It may still be running on BRAIN. A batch BRAIN has wedged refuses every cancel: freeing "
                         "its core here stops AlphaGo waiting for it and marks its Alphas cancelled.").classes(
                    "ag-muted text-sm")
                with ui.row().classes("justify-end w-full gap-2"):
                    ui.button("Keep waiting", on_click=again.close).props("flat")
                    ui.button("Free its core", icon="link_off",
                              on_click=lambda: run(True, again)).props("color=negative")
            again.open()

    with ui.dialog() as dialog, ui.card().classes("ag-card p-5 w-[560px]"):
        ui.label("Cancel this batch?" if size > 1 else "Cancel this simulation?").classes("ag-section-title")
        if size > 1:
            ui.label(f"Asks BRAIN to cancel batch {sim_id}. None of its {size} Alphas will finish.").classes("text-sm")
        else:
            ui.label(f"Asks BRAIN to cancel simulation {sim_id}. It will not finish.").classes("text-sm")
        with ui.row().classes("justify-end w-full gap-2"):
            ui.button("Keep running", on_click=dialog.close).props("flat")
            ui.button("Cancel on BRAIN", icon="cancel", on_click=lambda: run(False, dialog)).props("color=negative")
    dialog.open()


# ── page ─────────────────────────────────────────────────────────────────────────────────────


async def page() -> None:
    from ...brain import brain
    from ...config import settings as app_settings
    from ...engine.simulator import simulator

    page_header("Live Simulations", "What every simulation core is doing right now, with cancel.")
    with ui.row().classes("gap-2 items-center flex-wrap"):
        busy_chip = ui.label().classes("ag-chip")
        queued_chip = ui.label().classes("ag-chip")
        batch_chip = ui.label().classes("ag-chip")
    notices = ui.column().classes("w-full gap-2")
    ui.label("Every Alpha in one batch shares the same Type, Region, Delay, Instrument Type and Language. Only "
             "simulations that have started are shown here; waiting ones start as soon as a core is free.").classes(
        "ag-muted text-sm")
    live: dict[str, dict[str, Any]] = {}          # sim id -> {"progress": label, "elapsed": label, "bar": element}
    shown: dict[str, Any] = {"key": None, "pauses": None}

    @ui.refreshable
    async def matrix() -> None:
        live.clear()
        running = dict(simulator.running)
        cores = core_blocks(running, slot_count())
        paused = bool(pauses())
        names = await task_names({int(i["task"]) for i in running.values() if i.get("task") is not None})
        size = max(2, int(app_settings.brain.multi_size))
        if cores.overflow:
            ui.label(f"{fmt_int(cores.overflow)} running simulation(s) do not fit in the {slot_count()} cores and "
                     "are not drawn.").classes("ag-warn-box w-full")
        with ui.column().classes("w-full gap-0"):
            for block in cores.blocks:
                info = block.info
                state = core_state(block, paused)
                with ui.column().classes("w-full gap-2 py-3").style("border-bottom:1px solid var(--ag-border)"):
                    with ui.row().classes("items-center gap-3 w-full flex-wrap"):
                        tile = ui.label(block.label).classes(f"ag-core-label {state}")
                        if block.span > 1:
                            tile.tooltip(f"One simulation holding {block.span} cores (GLB counts twice)")
                        children = await children_of(list((info or {}).get("ids") or []))
                        key = batch_key(info, children) if info else {}
                        with ui.row().classes("ag-batch-key items-center gap-4 flex-wrap" + (" on" if info else "")):
                            for label, k in BATCH_KEY:
                                with ui.row().classes("items-center gap-1 no-wrap"):
                                    ui.label(label).classes("ag-muted text-[11px] uppercase")
                                    ui.label(key.get(k, "—")).classes("ag-mono text-sm")
                        ui.space()
                        if info is None:
                            ui.label("IDLE").classes("ag-muted text-xs tracking-wider")
                            continue
                        task = info.get("task")
                        ui.label(names.get(int(task), f"Task {task}") if task is not None else "Manual / batch") \
                            .classes("text-sm ag-muted max-w-64 truncate")
                        ui.label(short_id(block.sim_id or "")).classes("ag-mono text-xs ag-muted").tooltip(
                            block.sim_id or "")
                        with ui.row().classes("ag-chip items-center gap-2 no-wrap").tooltip(
                                "Progress reported by BRAIN"):
                            with ui.element("div").classes("ag-progress"):
                                bar = ui.element("div").classes("ag-progress-fill")
                            prog = ui.label(progress_text(info)).classes("ag-mono w-12 text-right")
                        with ui.row().classes("ag-chip items-center gap-1 no-wrap").tooltip(
                                "Time since this batch started"):
                            ui.icon("schedule").classes("text-sm")
                            el = ui.label(elapsed(info.get("started"))).classes("ag-mono")
                        sid = block.sim_id or ""
                        ui.button(icon="cancel", on_click=lambda s=sid, i=info: cancel_dialog(s, i)).props(
                            "flat round dense color=negative").tooltip(
                            "Cancel batch" if len(children) > 1 else "Cancel simulation")
                        live[sid] = {"progress": prog, "elapsed": el, "bar": bar, "info": info}
                    cells = children or []
                    with ui.element("div").classes("w-full grid gap-1").style(
                            f"grid-template-columns:repeat({size}, minmax(0,1fr))"):
                        for j in range(size):
                            child = cells[j] if j < len(cells) else None
                            if child is None or info is None:
                                ui.element("div").classes("ag-cell empty")
                                continue
                            s = child.get("settings") or {}
                            cell_state = "running" if info.get("progress") is not None else "pending"
                            with ui.element("div").classes(f"ag-cell {cell_state}"):
                                if cell_state == "running":
                                    ui.element("span").classes("ag-cell-dot")
                                ui.tooltip(f"{child.get('expression') or 'Not linked to its batch yet'}\n"
                                           f"{'Running' if cell_state == 'running' else 'Pending'} · Universe "
                                           f"{s.get('universe', '—')} · Neutralization {s.get('neutralization', '—')}"
                                           ).classes("whitespace-pre-wrap ag-mono text-xs max-w-[520px]")
        _tick()

    def _tick() -> None:
        for entry in live.values():
            info = entry["info"]
            entry["progress"].set_text(progress_text(info))
            entry["elapsed"].set_text(elapsed(info.get("started")))
            p = info.get("progress")
            entry["bar"].style(f"width:{0 if p is None else max(0.0, min(1.0, float(p))) * 100:.0f}%")

    def render_status() -> None:
        c = brain.client
        busy_chip.set_text(f"{fmt_int(c.slots.in_use)}/{fmt_int(c.slots.limit)} cores busy")
        queued_chip.set_text(f"{fmt_int(simulator.queued_count)} queued")
        batch_chip.set_text(f"{fmt_int(app_settings.brain.multi_size)} per batch")
        now = pauses()
        said = [(p.key, p.body) for p in now]
        if said != shown["pauses"]:
            shown["pauses"] = said
            notices.clear()
            with notices:
                for p in now:
                    with ui.column().classes("ag-warn-box w-full gap-0"):
                        ui.label(p.title).classes("font-semibold")
                        ui.label(p.body).classes("text-sm")

    def signature() -> tuple:
        """What changes the drawing (runs, cores, pause, pending vs running); progress and time only tick."""
        return (tuple(sorted(simulator.running)), slot_count(), bool(pauses()),
                tuple(i.get("progress") is None for _, i in sorted(simulator.running.items())))

    async def refresh() -> None:
        render_status()
        key = signature()
        if key != shown["key"]:
            shown["key"] = key
            await matrix.refresh()
        else:
            _tick()

    render_status()
    with ui.card().classes("ag-card w-full p-4"):
        with ui.row().classes("items-center w-full"):
            ui.label(f"{slot_count()} cores × {app_settings.brain.multi_size} Alphas").classes("ag-section-title")
        shown["key"] = signature()
        await matrix()
    ui.timer(1.0, refresh)
