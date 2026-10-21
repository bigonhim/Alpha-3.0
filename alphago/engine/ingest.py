"""Convert BRAIN alpha JSON into `Alpha` rows (shared by simulation results and account sync)."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ..brain.simulations import settings_hash
from ..db import db
from ..db.models import Alpha, Candidate, JournalEntry
from .evaluator import blocking_failures, reverdict

log = logging.getLogger(__name__)


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _code(part: Any) -> str | None:
    if isinstance(part, dict):
        return part.get("code")
    return part if isinstance(part, str) else None


def _description(part: Any) -> str | None:
    return part.get("description") if isinstance(part, dict) else None


def expression_hashes(code: str | None) -> tuple[str | None, str | None, int | None, list[str] | None]:
    """(canonical hash, skeleton hash, operator count, fields) — tolerant of unparsable code."""
    if not code:
        return None, None, None, None
    try:
        from ..fastexpr import canonical_hash, compute_metrics, parse, skeleton_hash
        program = parse(code)
        metrics = compute_metrics(program)
        return canonical_hash(program), skeleton_hash(program), metrics.operator_count, sorted(metrics.fields)
    except Exception:
        return None, None, None, None


def apply_alpha_json(row: Alpha, data: dict[str, Any], origin: str | None = None) -> Alpha:
    settings = data.get("settings") or {}
    is_ = data.get("is") or {}
    row.type = data.get("type") or row.type or "REGULAR"
    row.code = _code(data.get("regular")) or row.code
    row.selection = _code(data.get("selection")) or row.selection
    row.combo = _code(data.get("combo")) or row.combo
    row.settings = settings or row.settings
    row.region = settings.get("region", row.region)
    row.universe = settings.get("universe", row.universe)
    row.delay = settings.get("delay", row.delay)
    row.neutralization = settings.get("neutralization", row.neutralization)
    row.decay = settings.get("decay", row.decay)
    row.truncation = settings.get("truncation", row.truncation)
    row.status = data.get("status", row.status)
    row.stage = data.get("stage", row.stage)
    row.grade = data.get("grade", row.grade)
    row.name = data.get("name", row.name)
    row.tags = data.get("tags", row.tags)
    row.color = data.get("color", row.color)
    row.category = data.get("category", row.category)
    row.description = _description(data.get("regular")) or row.description
    row.classifications = data.get("classifications", row.classifications)
    row.pyramids = data.get("pyramids", row.pyramids)
    row.themes = data.get("themes", row.themes)
    row.competitions = data.get("competitions", row.competitions)
    row.osmosis_points = data.get("osmosisPoints", row.osmosis_points)
    row.date_created = _dt(data.get("dateCreated")) or row.date_created
    row.date_submitted = _dt(data.get("dateSubmitted")) or row.date_submitted
    row.date_modified = _dt(data.get("dateModified")) or row.date_modified
    for attr, key in (("sharpe", "sharpe"), ("fitness", "fitness"), ("turnover", "turnover"), ("returns", "returns"),
                      ("drawdown", "drawdown"), ("margin", "margin"), ("long_count", "longCount"),
                      ("short_count", "shortCount")):
        if key in is_:
            setattr(row, attr, is_.get(key))
    if is_.get("checks") is not None:
        row.is_checks = is_.get("checks")
        if row.checked_at is None:      # once BRAIN's full /check ran, `checks` keeps that result
            row.checks = is_.get("checks")
    if is_.get("selfCorrelation") is not None:
        row.self_corr = is_.get("selfCorrelation")
    if is_.get("prodCorrelation") is not None:
        row.prod_corr = is_.get("prodCorrelation")
    os_ = data.get("os")
    if os_:
        row.os = os_
        row.os_sharpe = os_.get("sharpe") or os_.get("osISSharpeRatio") or row.os_sharpe
    if origin:
        row.origin = origin
    if row.code and not row.canonical_hash:
        row.canonical_hash, row.skeleton_hash, row.op_count, row.fields = expression_hashes(row.code)
    if row.settings:
        row.settings_hash = settings_hash(row.settings)
    reverdict(row)
    row.raw = {k: v for k, v in data.items() if k not in ("is", "os", "train", "test", "prod")}
    return row


async def upsert_alpha(data: dict[str, Any], *, candidate_id: int | None = None, origin: str = "app") -> Alpha:
    alpha_id = str(data["id"])
    async with db.write() as s:
        row = await s.get(Alpha, alpha_id)
        if row is None:
            row = Alpha(id=alpha_id, origin=origin)
            s.add(row)
        apply_alpha_json(row, data, origin=None if row.origin == "app" and origin == "synced" else origin)
        if candidate_id is not None:
            row.candidate_id = candidate_id
            cand = await s.get(Candidate, candidate_id)
            if cand is not None:
                row.dataset_ids = [cand.dataset_id] if cand.dataset_id else row.dataset_ids
        await s.flush()
        return row


async def journal(alpha: Alpha, cand: Candidate | None) -> None:
    failing = blocking_failures(alpha.is_checks if alpha.is_checks is not None else alpha.checks)
    async with db.write() as s:
        s.add(JournalEntry(
            skeleton_hash=alpha.skeleton_hash, expression=alpha.code or "", hypothesis=cand.hypothesis if cand else None,
            dataset_id=cand.dataset_id if cand else None, region=alpha.region, verdict=alpha.verdict,
            sharpe=alpha.sharpe, fitness=alpha.fitness, turnover=alpha.turnover, failing_checks=failing,
            alpha_id=alpha.id))
