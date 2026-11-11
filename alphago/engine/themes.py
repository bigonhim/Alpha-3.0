"""BRAIN themes (the GLB theme first): find the theme's data, research it, and see which alphas match.

BRAIN announces themes in the platform's notifications and publishes no theme API. What the API does give:
  - dataset / data-field searches accept `theme=true` (the datasets that count for the current themes);
  - the submission check reports MATCHES_THEMES with a `themes` list (an alpha can match more than one);
  - datasets and fields carry a `themes` list once synced.
So a theme here is a scope (region / delay / universe, GLB by default) plus optional dates, multiplier and datasets from
the announcement, all editable on the Themes page. Everything else is automated:
  - `sync_theme`   pulls the scope's theme datasets (theme=true) and syncs their fields;
  - `plan_task`    creates a Search Lab task on the best theme dataset not yet researched today, with the theme's
                   neutralization (COUNTRY for GLB), in the Autopilot's mode (Assisted: waits for approval);
  - `plan_day`     does that for every active theme after the quota reset when `themes.auto_daily` is on;
  - `check_scope`  runs BRAIN's check (no simulations) on strong theme-scope alphas, which reads MATCHES_THEMES.
Nothing is ever submitted from here.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sqlalchemy import func, select

from ..brain import brain
from ..brain.quota import eastern_day
from ..config import ThemeDef, settings
from ..db import db
from ..db.models import Alpha, Candidate, Dataset, DatasetRating, ResearchTask
from ..events import bus
from ..jobs.queue import JobContext, jobs

log = logging.getLogger(__name__)

TAG_PREFIX = "BRAIN theme: "
PLANNED_KEY = "themes:planned_day"
DATASETS_KEY = "themes:datasets:{}"


def task_theme(t: ThemeDef) -> str:
    return f"{TAG_PREFIX}{t.name}"[:64]


def find(name: str | None) -> ThemeDef | None:
    if not name:
        return None
    name = name[len(TAG_PREFIX):] if name.startswith(TAG_PREFIX) else name
    return next((t for t in settings.themes.items if t.name == name), None)


def is_active(t: ThemeDef, today: date | None = None) -> bool:
    if not t.enabled:
        return False
    day = (today or date.today()).isoformat()
    return (not t.start or t.start <= day) and (not t.end or day <= t.end)


def active(today: date | None = None) -> list[ThemeDef]:
    return [t for t in settings.themes.items if is_active(t, today)]


def overrides(task_theme_name: str | None, region: str) -> dict[str, Any]:
    """Settings a theme task must use (applied by research tasks on top of the category defaults)."""
    t = find(task_theme_name) if (task_theme_name or "").startswith(TAG_PREFIX) else None
    if t is None or t.region != region:
        return {}
    out: dict[str, Any] = {}
    if t.neutralization:
        out["neutralization"] = t.neutralization
    if t.decay >= 0:
        out["decay"] = t.decay
    return out


def theme_hits(a: Alpha) -> list[str]:
    """Theme names an alpha matches: from every MATCHES_THEMES check that passed, plus the alpha's own `themes`."""
    names: list[str] = []
    for c in a.checks or []:
        if isinstance(c, dict) and c.get("name") == "MATCHES_THEMES" and str(c.get("result")).upper() == "PASS":
            for th in c.get("themes") or []:
                names.append(str(th.get("name") if isinstance(th, dict) else th))
    for th in a.themes or []:
        names.append(str(th.get("name") if isinstance(th, dict) else th))
    return list(dict.fromkeys(n for n in names if n and n != "None"))


def _in_scope(a: Alpha, t: ThemeDef) -> bool:
    return a.region == t.region and int(a.delay or 0) == t.delay and (not t.universe or a.universe == t.universe)


# ---------------------------------------------------------------- data

async def theme_dataset_ids(t: ThemeDef) -> list[str]:
    """The theme's datasets: the ones named in the theme, else BRAIN's theme=true list (synced by `sync_theme`), else
    datasets of the scope whose synced `themes` list is not empty."""
    if t.dataset_ids:
        return list(t.dataset_ids)
    cached = await db.kv_get(DATASETS_KEY.format(t.name))
    if cached:
        return list(cached)
    async with db.session() as s:
        rows = list(await s.scalars(select(Dataset).where(Dataset.region == t.region, Dataset.delay == t.delay,
                                                          Dataset.universe == t.universe)))
    return [d.id for d in rows if d.themes]


async def sync_theme(t: ThemeDef, ctx: JobContext | None = None) -> str:
    from .sync import sync_datasets, sync_fields
    params = {"region": t.region, "delay": t.delay, "universe": t.universe}
    await sync_datasets(ctx, params)                  # the scope's catalogue (names, categories, themes)
    ids: list[str] = []
    async for d in brain.data.datasets(t.region, t.delay, t.universe, theme="true"):
        ids.append(str(d.get("id")))
    await db.kv_set(DATASETS_KEY.format(t.name), ids)
    if ctx:
        await ctx.update(0.4, f"{len(ids)} theme datasets; syncing their fields")
    use = t.dataset_ids or ids
    if use:
        await sync_fields(ctx, {**params, "dataset_ids": use})
    msg = (f"{t.name} theme: {len(ids)} dataset(s) flagged for the theme in {t.region} D{t.delay} {t.universe}"
           + ("" if ids else " (BRAIN flags none right now: the theme may not restrict data; all datasets count)") + ".")
    bus.publish("sync", msg, "success")
    return msg


async def ranked_datasets(t: ThemeDef) -> list[dict[str, Any]]:
    """Theme datasets with synced fields, best first: AI rating, then fewest of your own simulations."""
    ids = await theme_dataset_ids(t)
    async with db.session() as s:
        q = select(Dataset).where(Dataset.region == t.region, Dataset.delay == t.delay, Dataset.universe == t.universe)
        if ids:
            q = q.where(Dataset.id.in_(ids))
        rows = list(await s.scalars(q))
        ratings = {r.dataset_id: r for r in await s.scalars(select(DatasetRating).where(
            DatasetRating.region == t.region, DatasetRating.delay == t.delay, DatasetRating.universe == t.universe))}
        tested = dict((await s.execute(select(Candidate.dataset_id, func.count()).where(
            Candidate.region == t.region, Candidate.delay == t.delay, Candidate.status.in_(("done", "reused")))
            .group_by(Candidate.dataset_id))).all())
    out = []
    for d in rows:
        if not d.fields_synced_at:
            continue
        r = ratings.get(d.id)
        out.append({"id": d.id, "name": d.name, "category": d.category_name, "fields": d.field_count,
                    "ai_score": r.ai_score if r else None, "tested": int(tested.get(d.id, 0) or 0),
                    "pyramid": d.pyramid_multiplier, "theme_flag": bool(d.themes) or d.id in ids})
    out.sort(key=lambda x: (-(x["ai_score"] or 50), x["tested"], -(x["pyramid"] or 1)))
    return out


# ---------------------------------------------------------------- research

async def plan_task(t: ThemeDef, *, start: bool | None = None, source: str | None = None,
                    exclude: set[str] | None = None) -> int:
    """One Search Lab task on the best theme dataset. Assisted mode: planned (waits for approval); Full Autopilot or an
    explicit start: queued and started. Manual clicks are not capped by the Autopilot's daily cap."""
    from .research import research
    mode = settings.research.mode
    start = (mode == "autopilot") if start is None else start
    source = source or ("autopilot" if mode == "autopilot" else "assisted")
    ranked = [d for d in await ranked_datasets(t) if d["id"] not in (exclude or set())]
    if not ranked:
        raise RuntimeError(f"No {t.name} theme dataset has synced fields yet: press 'Sync theme data' first.")
    async with db.session() as s:
        today = list(await s.scalars(select(ResearchTask).where(ResearchTask.theme == task_theme(t),
                                                                ResearchTask.plan_day == eastern_day())))
    used_today = {d for task in today for d in (task.dataset_ids or [])}
    pick = next((d for d in ranked if d["id"] not in used_today), ranked[0])
    cfg = settings.themes
    status = "queued" if start else "planned"
    if not brain.client.quota.can_spend(cfg.task_sims, settings.brain.quota_reserve):
        status = "skipped"
    reason = (f"{t.name} theme ({t.region} D{t.delay} {t.universe}"
              + (f", {t.neutralization} neutralization" if t.neutralization else "")
              + (f", multiplier {t.multiplier:g}" if t.multiplier and t.multiplier != 1 else "")
              + f"): {pick['id']} is the best-rated theme dataset"
              + (f" with {pick['tested']} of your simulations so far." if pick["tested"] else " you have not tried yet."))
    tid = await research.create(kind="single", region=t.region, delay=t.delay, universe=t.universe,
                                dataset_ids=[pick["id"]], planned_sims=cfg.task_sims, cores=cfg.task_cores,
                                source=source, reason=reason, theme=task_theme(t), status=status)
    if status == "skipped":
        from sqlalchemy import update
        async with db.write() as s:
            await s.execute(update(ResearchTask).where(ResearchTask.id == tid).values(
                message="Not enough simulations left today."))
    elif start:
        await research.start(tid)
    return tid


async def plan_day() -> list[int]:
    """After the quota reset: one task per active theme (the Autopilot's mode decides approval)."""
    ids: list[int] = []
    for t in active():
        try:
            ids.append(await plan_task(t))
        except Exception as exc:
            bus.publish("autopilot", f"{t.name} theme: could not plan research: {exc}", "warning")
    await db.kv_set(PLANNED_KEY, eastern_day())
    return ids


async def maybe_plan_today() -> list[int]:
    """Called by the daily scheduler; plans once per BRAIN day when the user switched theme planning on."""
    if not settings.themes.auto_daily or settings.research.mode == "manual" or not brain.client.logged_in:
        return []
    if await db.kv_get(PLANNED_KEY) == eastern_day():
        return []
    ids = await plan_day()
    if ids:
        bus.publish("autopilot", f"Themes: planned {len(ids)} theme task(s)"
                    + (" and started them." if settings.research.mode == "autopilot"
                       else " — approve them on the Autopilot or Tasks page."), "success")
    return ids


# ---------------------------------------------------------------- matches

async def scope_alphas(t: ThemeDef) -> list[Alpha]:
    async with db.session() as s:
        rows = list(await s.scalars(select(Alpha).where(Alpha.region == t.region, Alpha.delay == t.delay)))
    return [a for a in rows if _in_scope(a, t)]


async def summary(t: ThemeDef) -> dict[str, Any]:
    alphas = await scope_alphas(t)
    verdicts: dict[str, int] = {}
    for a in alphas:
        verdicts[a.verdict or "—"] = verdicts.get(a.verdict or "—", 0) + 1
    matched = [a for a in alphas if theme_hits(a)]
    return {"alphas": len(alphas), "verdicts": verdicts, "matched": len(matched),
            "matched_unsubmitted": sum(1 for a in matched if a.status == "UNSUBMITTED"),
            "datasets": len(await theme_dataset_ids(t)), "active": is_active(t)}


async def check_scope(t: ThemeDef, limit: int | None = None, ctx: JobContext | None = None) -> str:
    """BRAIN's check (no simulations) on strong/ready unsubmitted alphas of the scope, to read MATCHES_THEMES."""
    from .actions import run_check
    limit = limit or settings.research.check_cap
    todo = [a for a in await scope_alphas(t) if a.status == "UNSUBMITTED" and a.verdict in ("strong", "ready")
            and not any(isinstance(c, dict) and c.get("name") == "MATCHES_THEMES" for c in (a.checks or []))]
    todo.sort(key=lambda a: -(a.sharpe or 0))
    todo = todo[:limit]
    hits = 0
    for i, a in enumerate(todo, 1):
        if ctx:
            await ctx.update((i - 1) / max(1, len(todo)), f"Checking {a.id} ({i}/{len(todo)})")
        try:
            await run_check(a.id)
        except Exception as exc:
            log.warning("theme check of %s failed: %s", a.id, exc)
            continue
        async with db.session() as s:
            fresh = await s.get(Alpha, a.id)
        hits += bool(fresh and theme_hits(fresh))
    msg = f"{t.name} theme: checked {len(todo)} alpha(s), {hits} match a theme."
    bus.publish("check", msg, "success" if hits else "info")
    return msg


# ---------------------------------------------------------------- jobs

def _theme_from(params: dict) -> ThemeDef:
    t = find(str(params.get("theme") or ""))
    if t is None:
        raise ValueError(f"Unknown theme {params.get('theme')!r}")
    return t


@jobs.register("theme_sync")
async def _job_sync(ctx: JobContext, params: dict) -> str:
    return await sync_theme(_theme_from(params), ctx)


@jobs.register("theme_check")
async def _job_check(ctx: JobContext, params: dict) -> str:
    return await check_scope(_theme_from(params), ctx=ctx)
