"""Application shell: header (core tiles, clocks, pause chip, search), sidebar navigation in collapsible sections,
the Ctrl+K command palette and client-side routed pages.

The header and the palette follow alpha-harness ``shell/header.tsx`` and ``shell/command-menu.tsx``: one tile per
BRAIN simulation core (running fills along its bottom as BRAIN reports progress, pending, warning while sending is
paused, idle as a hollow dashed socket), the clocks a consultant runs against, the most pressing reason sending is
paused, and a palette that jumps to any screen, task, Alpha id or data-field search.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from nicegui import ui

from ..brain import brain
from .components import fmt_duration, fmt_int
from .theme import SIDEBAR_WIDTH, apply_theme


class NavItem:
    def __init__(self, path: str, label: str, icon: str, color: str, builder: Callable) -> None:
        self.path, self.label, self.icon, self.color, self.builder = path, label, icon, color, builder


def _nav() -> list[NavItem]:
    from .pages import (
        alphas,
        analytics,
        aqp,
        assistant,
        audit,
        autopilot,
        competitions,
        dashboard,
        data_explorer,
        dataset_intel,
        genius,
        health,
        labs,
        live,
        llm,
        oos,
        osmosis,
        osmosis_review,
        planner,
        pool,
        portfolio,
        power_pool,
        pyramids,
        python_alpha,
        research_history,
        simulation_matrix,
        spc_lab,
        super_alphas,
        sync,
        tasks,
        themes,
        tools,
    )
    return [
        NavItem("/", "Dashboard", "space_dashboard", "#a78bfa", dashboard.page),
        NavItem("/live", "Live Simulations", "memory", "#60a5fa", live.page),
        NavItem("/matrix", "Simulation Matrix", "grid_view", "#818cf8", simulation_matrix.page),
        NavItem("/labs", "Research Labs", "science", "#a78bfa", labs.page),
        NavItem("/tools", "Tools", "construction", "#7dd3fc", tools.page),
        NavItem("/data", "Data Explorer", "dataset", "#5eead4", data_explorer.page),
        NavItem("/datasets", "Dataset Intelligence", "psychology", "#f472b6", dataset_intel.page),
        NavItem("/autopilot", "Alpha Autopilot", "smart_toy", "#7dd3fc", autopilot.page),
        NavItem("/genius", "Genius AI", "workspace_premium", "#fbbf24", genius.page),
        NavItem("/osmosis", "Osmosis", "water_drop", "#5eead4", osmosis.page),
        NavItem("/osmosis-review", "Osmosis Review", "fact_check", "#f472b6", osmosis_review.page),
        NavItem("/power-pool", "Power Pool", "diamond", "#c084fc", power_pool.page),
        NavItem("/oos", "Out of Sample", "travel_explore", "#7dd3fc", oos.page),
        NavItem("/themes", "BRAIN Themes", "public", "#7dd3fc", themes.page),
        NavItem("/python", "Python Alphas", "code", "#fbbf24", python_alpha.page),
        NavItem("/aqp", "Alpha Quality Project", "verified", "#5eead4", aqp.page),
        NavItem("/spc", "SPC Lab", "emoji_events", "#fbbf24", spc_lab.page),
        NavItem("/super", "Super Alphas", "hub", "#a78bfa", super_alphas.page),
        NavItem("/tasks", "Tasks", "checklist", "#5eead4", tasks.page),
        NavItem("/alphas", "Alphas", "layers", "#818cf8", alphas.page),
        NavItem("/pool", "Alpha Pool", "inventory_2", "#c084fc", pool.page),
        NavItem("/portfolio", "Portfolio", "donut_large", "#4ade80", portfolio.page),
        NavItem("/pyramids", "Pyramids", "change_history", "#fbbf24", pyramids.page),
        NavItem("/competitions", "Competitions", "emoji_events", "#f472b6", competitions.page),
        NavItem("/history", "Research History", "history", "#7dd3fc", research_history.page),
        NavItem("/analytics", "Analytics", "insights", "#818cf8", analytics.page),
        NavItem("/planner", "Quota Planner", "calculate", "#fbbf24", planner.page),
        NavItem("/health", "Health", "monitor_heart", "#4ade80", health.page),
        NavItem("/audit", "Audit Log", "receipt_long", "#8b98ad", audit.page),
        NavItem("/llm", "LLM Integration", "auto_awesome", "#c084fc", llm.page),
        NavItem("/assistant", "AI Assistant", "forum", "#c084fc", assistant.page),
        NavItem("/sync", "Sync with BRAIN", "sync", "#4ade80", sync.page),
    ]


#: The sidebar's sections, in order; every NavItem path belongs to exactly one.
SECTIONS: list[tuple[str, list[str]]] = [
    ("Overview", ["/", "/live", "/matrix", "/tasks"]),
    ("Research", ["/labs", "/tools", "/data", "/datasets", "/autopilot", "/genius", "/history"]),
    ("Alphas", ["/alphas", "/pool", "/portfolio", "/oos", "/super", "/python"]),
    ("Programs", ["/osmosis", "/osmosis-review", "/power-pool", "/themes", "/aqp", "/spc", "/pyramids",
                  "/competitions"]),
    ("Insights", ["/analytics", "/planner", "/health", "/audit"]),
    ("System", ["/llm", "/assistant", "/sync"]),
]
SIDEBAR_PREF = "pref:sidebar"

#: Other words for a screen, so a search for what it does finds it too.
SYNONYMS: dict[str, list[str]] = {
    "/": ["home", "overview", "today", "submittable alphas", "dispatch today's simulations"],
    "/live": ["cores", "running", "cancel simulation", "in flight", "progress"],
    "/matrix": ["view live simulation matrix", "grid", "settings matrix"],
    "/labs": ["search lab", "template lab", "evolution lab", "llm power pool lab", "super alpha lab", "optuna"],
    "/tools": ["settings sampler", "fields sampler", "correlation breaker", "paste alphas", "flip sign"],
    "/data": ["explore data catalog", "data fields", "datasets", "fields", "catalogue"],
    "/datasets": ["dataset ranking", "coverage", "which dataset"],
    "/autopilot": ["automatic research", "robot", "auto"],
    "/genius": ["genius program", "level", "pyramid count"],
    "/osmosis": ["osmosis points", "allocation"],
    "/osmosis-review": ["osmosis review", "points review"],
    "/power-pool": ["power pool alphas", "ppac", "description template"],
    "/oos": ["out of sample", "decay", "os sharpe", "is/os gap"],
    "/themes": ["themes", "multiplier", "glb theme"],
    "/python": ["python alphas", "code alpha"],
    "/aqp": ["alpha quality project", "quality"],
    "/spc": ["super power challenge", "spc", "competition submit"],
    "/super": ["super alphas", "combo", "selection"],
    "/tasks": ["jobs", "running tasks", "repair", "queue"],
    "/alphas": ["browse submittable alphas", "submit", "check", "my alphas", "ready to submit"],
    "/pool": ["alpha pool", "stored alphas", "vault", "submitted alphas"],
    "/portfolio": ["portfolio correlation", "combined pnl", "incremental sharpe"],
    "/pyramids": ["pyramids", "pyramid multipliers", "category coverage"],
    "/competitions": ["competitions", "iqc", "contest"],
    "/history": ["research history", "past research"],
    "/analytics": ["charts", "statistics", "hit rate"],
    "/planner": ["quota planner", "simulations left", "daily budget"],
    "/health": ["health", "rate limit", "pacing", "errors"],
    "/audit": ["audit log", "what happened", "record"],
    "/llm": ["ai providers", "api keys", "models", "openai", "claude", "ollama", "openrouter"],
    "/assistant": ["ask ai assistant", "chat"],
    "/sync": ["sync data fields with brain", "sign in", "login", "credentials", "download"],
}

#: Shaped like a BRAIN Alpha id: 7 or 8 letters and digits, with a digit or a capital past the first.
ALPHA_ID = re.compile(r"^(?=[A-Za-z0-9]{7,8}$).(?=.*[0-9A-Z])")
_ALPHA_URL = re.compile(r"worldquantbrain\.com/alphas?/([A-Za-z0-9]+)")


def alpha_id_of(text: str) -> str:
    """The Alpha id in a pasted BRAIN link, or the text itself."""
    found = _ALPHA_URL.search(text)
    return found.group(1) if found else text


@dataclass
class PaletteEntry:
    group: str              # Alpha | Actions | Tasks | Screens
    label: str
    kind: str               # alpha | navigate | action
    target: str             # alpha id, path, or action name
    keywords: list[str] = field(default_factory=list)


def palette_entries(query: str, tasks: Iterable[tuple[str, str]], screens: Iterable[tuple[str, str]],
                    limit: int = 60) -> list[PaletteEntry]:
    """What the palette lists for ``query``: an Alpha id opens that Alpha, any text searches the data fields, then
    actions, tasks and screens whose label or synonyms hold every word typed (labels starting with it first).

    ``tasks`` are (label, path) pairs; ``screens`` are (path, label) pairs."""
    typed = query.strip()
    words = [w for w in typed.lower().split() if w]
    out: list[PaletteEntry] = []
    alpha = alpha_id_of(typed)
    if ALPHA_ID.match(alpha):
        out.append(PaletteEntry("Alpha", f"Open Alpha {alpha}", "alpha", alpha))
    pool: list[PaletteEntry] = [
        PaletteEntry("Actions", "Export diagnostics", "action", "diagnostics",
                     ["support", "help", "log", "bug report", "zip"]),
        PaletteEntry("Actions", "Stop all", "action", "stop_all", ["pause everything", "halt", "emergency"]),
    ]
    pool += [PaletteEntry("Tasks", label, "navigate", path, ["task"]) for label, path in tasks]
    pool += [PaletteEntry("Screens", label, "navigate", path, [path, *SYNONYMS.get(path, [])])
             for path, label in screens]

    def rank(entry: PaletteEntry) -> int | None:
        label = entry.label.lower()
        hay = " ".join([label, *(k.lower() for k in entry.keywords)])
        if not all(w in hay for w in words):
            return None
        if not words:
            return 1
        joined = " ".join(words)
        return 0 if label.startswith(joined) else 1 if joined in label else 2

    ranked = [(r, i, e) for i, e in enumerate(pool) if (r := rank(e)) is not None]
    group_order = {"Actions": 0, "Tasks": 1, "Screens": 2}
    ranked.sort(key=lambda t: (t[0], group_order.get(t[2].group, 3), t[1]))
    matched = [e for _, _, e in ranked]
    if typed:
        # after the Alpha, before the rest: a search is always on offer for anything typed
        out.append(PaletteEntry("Actions", f"Search Data Fields for “{typed}”", "navigate",
                                f"/data?q={_quote(typed)}"))
    out += matched
    return out[:limit]


def _quote(text: str) -> str:
    from urllib.parse import quote
    return quote(text, safe="")


async def task_destinations() -> list[tuple[str, str]]:
    """Every lab task and research task by name, newest first, for the palette."""
    from sqlalchemy import select

    from ..db import db
    from ..db.models import ResearchTask
    from ..db.models_labs import LabTask
    out: list[tuple[str, str]] = []
    async with db.session() as s:
        for t in await s.scalars(select(LabTask).order_by(LabTask.id.desc()).limit(200)):
            out.append((f"Task {t.id} · {t.name or t.lab}", f"/labs?task={t.id}"))
        for r in await s.scalars(select(ResearchTask).order_by(ResearchTask.id.desc()).limit(200)):
            label = f"Research task {r.id} · {r.kind} {r.region} D{r.delay} {r.universe}"
            if r.theme:
                label += f" · {r.theme}"
            out.append((label, f"/tasks?task={r.id}&kind=research"))
    return out


# ── actions shared by the palette and the sidebar ─────────────────────────────────────────────


async def stop_all() -> None:
    from ..engine.research import research
    n = await research.stop_all()
    ui.notify(f"Stopped: Autopilot off, {n} task(s) and the simulation queue paused. Run All on Tasks resumes.",
              type="warning", multi_line=True)


async def export_diagnostics() -> None:
    """Write the scrubbed diagnostics zip to Downloads and say where it is."""
    import asyncio

    from .. import diagnostics
    try:
        path = await asyncio.to_thread(diagnostics.export)
    except OSError as exc:
        ui.notify(f"Could not write the diagnostics file: {exc}", type="negative", multi_line=True)
        return
    ui.notify(f"Saved {path.name} in {path.parent}. It holds the app version and the end of the log, with your "
              "username, home folder and ids removed: attach it when you ask for help.", type="positive",
              multi_line=True, timeout=10000, close_button=True)


# ── palette ──────────────────────────────────────────────────────────────────────────────────


def _palette(items: list[NavItem]) -> Callable[[], Any]:
    """Build the palette dialog once per client; returns the function that opens it."""
    state: dict[str, Any] = {"entries": [], "selected": 0, "tasks": []}
    screens = [(i.path, i.label) for i in items]

    with ui.dialog().props("position=top") as dialog, ui.card().classes("ag-card ag-palette p-0 w-[560px] max-w-[92vw]"):
        with ui.row().classes("items-center gap-2 px-3 w-full no-wrap").style("border-bottom:1px solid var(--ag-border)"):
            ui.icon("search").classes("ag-muted")
            box = ui.input(placeholder="Go to a screen, task or Alpha id…").props(
                "borderless dense autofocus").classes("flex-1")
            ui.label("Esc").classes("ag-kbd")
        listing = ui.column().classes("w-full gap-0 p-1 max-h-[60vh] overflow-y-auto")

    def render() -> None:
        listing.clear()
        entries = state["entries"]
        with listing:
            if not entries:
                ui.label("No destination matches.").classes("ag-muted text-sm px-3 py-6 w-full text-center")
                return
            group = None
            for i, entry in enumerate(entries):
                if entry.group != group:
                    group = entry.group
                    ui.label(group).classes("ag-muted text-[11px] uppercase tracking-wider px-2 pt-2 pb-1")
                row = ui.row().classes("ag-palette-item items-center w-full no-wrap" +
                                       (" selected" if i == state["selected"] else ""))
                row.on("click", lambda _e, e=entry: choose(e))
                with row:
                    ui.icon({"alpha": "layers", "action": "bolt"}.get(entry.kind, "arrow_forward")).classes(
                        "text-base ag-muted")
                    ui.label(entry.label).classes("truncate text-sm")

    def recompute() -> None:
        state["entries"] = palette_entries(box.value or "", state["tasks"], screens)
        state["selected"] = 0
        render()

    def move(step: int) -> None:
        n = len(state["entries"])
        if n:
            state["selected"] = (state["selected"] + step) % n
            render()

    async def choose(entry: PaletteEntry | None = None) -> None:
        if entry is None:
            if not state["entries"]:
                return
            entry = state["entries"][state["selected"]]
        dialog.close()
        if entry.kind == "alpha":
            from .alpha_detail import open_alpha
            await open_alpha(entry.target)
        elif entry.kind == "navigate":
            ui.navigate.to(entry.target)
        elif entry.target == "diagnostics":
            await export_diagnostics()
        elif entry.target == "stop_all":
            await stop_all()

    box.on_value_change(lambda _e: recompute())
    box.on("keydown.down", lambda _e: move(1))
    box.on("keydown.up", lambda _e: move(-1))
    box.on("keydown.enter", lambda _e: choose())

    async def open_palette() -> None:
        box.value = ""
        try:
            state["tasks"] = await task_destinations()
        except Exception:  # noqa: BLE001 - the palette still opens on screens when the tasks cannot be read
            state["tasks"] = []
        recompute()
        dialog.open()
        box.run_method("focus")

    return open_palette


_HOTKEY = """
<script>
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    emitEvent('ag-palette');
  }
});
</script>
"""


# ── header ───────────────────────────────────────────────────────────────────────────────────


def _core_tiles() -> None:
    """One tile per BRAIN simulation core, refreshed every second (DESIGN.md core-slot)."""
    from ..engine.simulator import simulator
    from .pages.live import core_blocks, core_state, pauses, progress_text, slot_count

    holder = ui.row().classes("ag-cores items-center no-wrap cursor-pointer").on(
        "click", lambda: ui.navigate.to("/live"))
    tiles: list[dict[str, Any]] = []

    def build(n: int) -> None:
        holder.clear()
        tiles.clear()
        with holder:
            for i in range(n):
                with ui.element("div").classes("ag-core idle") as tile:
                    text = ui.label(f"C{i + 1}").classes("ag-core-num")
                    fill = ui.element("div").classes("ag-core-fill")
                    tip = ui.tooltip("")
                tiles.append({"tile": tile, "text": text, "fill": fill, "tip": tip})

    def refresh() -> None:
        n = slot_count()
        if len(tiles) != n:
            build(n)
        paused = pauses()
        cores = core_blocks(dict(simulator.running), n)
        for block in cores.blocks:
            state = core_state(block, bool(paused))
            where = f"Core {block.start + 1}" if block.span == 1 else f"Cores {block.start + 1}–{block.start + block.span}"
            word = {"idle": "Idle", "running": "Running", "pending": "Pending", "warning": "Warning"}[state]
            info = block.info or {}
            lines = [f"{where}: {word}"]
            if paused:
                lines.append(paused[0].title)
            if block.info is not None:
                lines.append(f"{info.get('region') or '—'} · {info.get('size') or len(info.get('ids') or [])} "
                             f"Alpha(s) · {progress_text(info)}")
                if block.span > 1:
                    lines.append(f"One simulation holding {block.span} cores")
            for k in range(block.span):
                t = tiles[block.start + k]
                edge = "" if block.span == 1 else " first" if k == 0 else " last" if k == block.span - 1 else " mid"
                t["tile"].classes(replace=f"ag-core {state}{edge}")
                t["text"].set_text(f"C{block.start + k + 1}" + (" !" if state == "warning" else ""))
                p = info.get("progress")
                width = 0 if state != "running" or p is None else max(0.0, min(1.0, float(p))) * 100
                t["fill"].style(f"width:{width:.0f}%")
                t["tip"].set_text("\n".join(lines))

    refresh()
    ui.timer(1.0, refresh)


def _status_chips() -> None:
    from ..engine.simulator import simulator
    from .pages.live import pauses

    session = ui.label().classes("ag-chip")
    left = ui.label().classes("ag-chip")
    with left:
        left_tip = ui.tooltip("")
    queued = ui.label().classes("ag-chip")

    ui.button("Stop all", icon="stop_circle", on_click=stop_all).props(
        "outline dense no-caps color=negative").classes("ag-stop-all").tooltip(
        "Stop the Autopilot and pause every task and the simulation queue. Nothing is deleted.")
    reset = ui.label().classes("ag-chip")
    pause_chip = ui.row().classes("ag-chip warn items-center gap-1 no-wrap cursor-pointer").on(
        "click", lambda: ui.navigate.to("/live"))
    with pause_chip:
        ui.icon("pause").classes("text-sm")
        pause_text = ui.label()
        pause_tip = ui.tooltip("").classes("max-w-[340px]")

    def refresh() -> None:
        c = brain.client
        if c.state == "persona_required":
            session.set_text("⚠ BRAIN verification required")
            session.classes(replace="ag-chip warn")
        elif c.logged_in:
            session.set_text(f"🛡 BRAIN Session  {fmt_duration(c.seconds_left)}")
            session.classes(replace="ag-chip" if c.seconds_left > 1800 else "ag-chip warn")
        else:
            session.set_text("🛡 BRAIN  signed out")
            session.classes(replace="ag-chip bad")
        q = c.quota
        exact = q.updated_at is not None
        left.set_text(f"⚡ {'' if exact else '~'}{fmt_int(q.effective_remaining)}  Simulations Left Today")
        left_tip.set_text("Simulations left in today's quota, as BRAIN's own headers report it." if exact else
                          "An estimate until BRAIN's first simulation today reports the exact quota.")
        left.classes(replace="ag-chip" if q.effective_remaining > 500 else "ag-chip warn")
        queued.set_text(f"☰ {fmt_int(simulator.queued_count)}  Queued · {c.slots.in_use}/{c.slots.limit} slots")
        reset.set_text(f"⏱ Simulation Quota Reset in  {fmt_duration(q.seconds_to_reset)}")
        now = pauses()
        pause_chip.set_visibility(bool(now))
        if now:
            pause_text.set_text(now[0].chip)
            pause_tip.set_text(f"{now[0].title}. {now[0].body}")

    refresh()
    ui.timer(1.0, refresh)


# ── sidebar ──────────────────────────────────────────────────────────────────────────────────


def section_of(path: str) -> str | None:
    for name, paths in SECTIONS:
        if path in paths:
            return name
    return None


def root() -> None:
    apply_theme()
    items = _nav()
    by_path = {item.path: item for item in items}
    nav_rows: dict[str, ui.row] = {}
    open_palette = _palette(items)
    ui.add_body_html(_HOTKEY)
    ui.on("ag-palette", lambda _e: open_palette())

    def go(path: str) -> None:
        ui.navigate.to(path)

    with ui.header().classes("ag-header items-center px-5 py-2 gap-3 no-wrap"):
        _core_tiles()
        ui.space()
        _status_chips()
        ui.button(icon="search", on_click=open_palette).props("flat round dense color=grey-5").tooltip(
            "Go to a screen, task or Alpha (Ctrl+K)")

    with ui.left_drawer(value=True, fixed=True).props(f"width={SIDEBAR_WIDTH} :breakpoint=0").classes("ag-drawer p-3"):
        with ui.row().classes("items-center gap-2 px-2 pb-4 pt-1 no-wrap"):
            ui.icon("change_history").classes("text-3xl").style("color:#8b5cf6")
            ui.label("AlphaGo").classes("text-xl font-semibold")
            ui.label("Quant").classes("text-xl font-semibold ag-cyan")
        badges: dict[str, ui.label] = {}
        sections: dict[str, ui.expansion] = {}
        prefs: dict[str, Any] = {"collapsed": set(), "loaded": False}

        async def remember(name: str, opened: bool) -> None:
            if not prefs["loaded"]:
                return
            collapsed = prefs["collapsed"]
            if opened:
                collapsed.discard(name)
            else:
                collapsed.add(name)
            from ..db import db
            await db.kv_set(SIDEBAR_PREF, {"collapsed": sorted(collapsed)})

        with ui.column().classes("w-full gap-1 ag-nav-sections"):
            placed: set[str] = set()
            for name, paths in SECTIONS:
                exp = ui.expansion(name, value=True).props("dense dense-toggle switch-toggle-side expand-separator") \
                    .classes("w-full ag-nav-section")
                exp.on_value_change(lambda e, n=name: remember(n, bool(e.value)))
                sections[name] = exp
                with exp:
                    for path in paths:
                        item = by_path.get(path)
                        if item is None:
                            continue
                        placed.add(path)
                        with ui.row().classes("ag-nav-item items-center gap-3 no-wrap w-full").on(
                                "click", lambda _e, p=item.path: go(p)) as row:
                            ui.icon(item.icon).style(f"color:{item.color}").classes("text-[20px]")
                            ui.label(item.label).classes("text-[15px]")
                            if item.path == "/alphas":
                                badges[item.path] = ui.label().classes("ag-nav-badge").tooltip("Ready to submit")
                        nav_rows[item.path] = row
            for item in items:                       # a screen added without a section still shows
                if item.path not in placed:
                    with ui.row().classes("ag-nav-item items-center gap-3 no-wrap w-full").on(
                            "click", lambda _e, p=item.path: go(p)) as row:
                        ui.icon(item.icon).style(f"color:{item.color}").classes("text-[20px]")
                        ui.label(item.label).classes("text-[15px]")
                    nav_rows[item.path] = row
        ui.space()
        with ui.row().classes("items-center gap-1 px-2 pt-3 w-full no-wrap"):
            ui.button("Export diagnostics", icon="medical_information", on_click=export_diagnostics).props(
                "flat dense no-caps size=sm color=grey-5").tooltip(
                "Save a zip with the app version and the end of the log (username, home folder and ids removed) "
                "to Downloads, to attach when you ask for help.")
            ui.space()
            ui.button(icon="search", on_click=open_palette).props("flat round dense size=sm color=grey-5").tooltip(
                "Command palette (Ctrl+K)")
        with ui.row().classes("items-center gap-3 no-wrap px-2 pt-3 w-full").style(
                "border-top:1px solid var(--ag-border)"):
            avatar = ui.label().classes("rounded-full w-8 h-8 flex items-center justify-center text-sm font-semibold") \
                .style("background:#5b8def;color:#fff")
            with ui.column().classes("gap-0 min-w-0"):
                who = ui.label().classes("text-sm")
                mail = ui.label().classes("ag-muted text-xs truncate")

        async def load_sections() -> None:
            from ..db import db
            try:
                saved = await db.kv_get(SIDEBAR_PREF, {}) or {}
            except Exception:  # noqa: BLE001 - before the database is up every section simply stays open
                saved = {}
            collapsed = {n for n in saved.get("collapsed") or [] if n in sections}
            active = section_of(_current_path())
            collapsed.discard(active or "")
            prefs["collapsed"] = collapsed
            for name, exp in sections.items():
                exp.value = name not in collapsed
            prefs["loaded"] = True

        async def refresh_side() -> None:
            from sqlalchemy import func, select

            from .. import credentials
            from ..db import db
            from ..db.models import Alpha
            async with db.session() as s:
                n = await s.scalar(select(func.count(Alpha.id)).where(Alpha.verdict == "ready",
                                                                      Alpha.status == "UNSUBMITTED")) or 0
            if "/alphas" in badges:
                badges["/alphas"].set_text(str(n) if n else "")
            email = credentials.brain_email() or ""
            who.set_text(brain.client.user_id or ("Signed in" if brain.client.logged_in else "Not signed in"))
            mail.set_text(email)
            avatar.set_text((email[:1] or "?").upper())

        ui.timer(0.05, load_sections, once=True)
        ui.timer(0.1, refresh_side, once=True)
        ui.timer(15.0, refresh_side)

        router = ui.context.client.sub_pages_router

        def _current_path() -> str:
            return (router.current_path or "/").split("?", 1)[0]

        def highlight(path: str) -> None:
            path = (path or "/").split("?", 1)[0]
            for p, row in nav_rows.items():
                if p == path:
                    row.classes(add="active")
                else:
                    row.classes(remove="active")
            name = section_of(path)
            if name and name in sections and not sections[name].value:
                sections[name].value = True

        highlight(router.current_path)
        router.on_path_changed(highlight)

    with ui.column().classes("w-full px-8 py-6 gap-5"):
        ui.sub_pages({item.path: item.builder for item in items}).classes("w-full")
