"""ORM models. JSON columns hold BRAIN payload fragments that we do not need to query on."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """SQLite drops tzinfo: store naive UTC, hand back aware UTC (also normalises comparison parameters)."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):  # noqa: ANN001, ANN201
        if value is not None and value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):  # noqa: ANN001, ANN201
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class KV(Base):
    """Small key/value store for app state (last sync times, persona state, caches)."""

    __tablename__ = "kv"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


# ---------------------------------------------------------------- catalogue

class Operator(Base):
    __tablename__ = "operators"
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    category: Mapped[str | None] = mapped_column(String(64))
    scope: Mapped[Any] = mapped_column(JSON, nullable=True)
    definition: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    documentation: Mapped[str | None] = mapped_column(Text)
    level: Mapped[str | None] = mapped_column(String(32))
    synced_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class DataCategory(Base):
    __tablename__ = "data_categories"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    payload: Mapped[Any] = mapped_column(JSON, nullable=True)


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("id", "region", "delay", "universe", name="uq_dataset_scope"),)
    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(64), index=True)
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))
    name: Mapped[str | None] = mapped_column(String(256))
    description: Mapped[str | None] = mapped_column(Text)
    category_id: Mapped[str | None] = mapped_column(String(64))
    category_name: Mapped[str | None] = mapped_column(String(128))
    subcategory: Mapped[str | None] = mapped_column(String(128))
    coverage: Mapped[float | None] = mapped_column(Float)
    value_score: Mapped[float | None] = mapped_column(Float)
    user_count: Mapped[int | None] = mapped_column(Integer)
    alpha_count: Mapped[int | None] = mapped_column(Integer)
    field_count: Mapped[int | None] = mapped_column(Integer)
    pyramid_multiplier: Mapped[float | None] = mapped_column(Float)
    themes: Mapped[Any] = mapped_column(JSON, nullable=True)
    fields_synced_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    synced_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class DataField(Base):
    __tablename__ = "datafields"
    __table_args__ = (
        UniqueConstraint("id", "region", "delay", "universe", name="uq_field_scope"),
        Index("ix_field_dataset", "dataset_id", "region", "delay", "universe"),
    )
    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(128), index=True)
    dataset_id: Mapped[str] = mapped_column(String(64))
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))
    type: Mapped[str] = mapped_column(String(16), default="MATRIX")
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(128))
    coverage: Mapped[float | None] = mapped_column(Float)
    user_count: Mapped[int | None] = mapped_column(Integer)
    alpha_count: Mapped[int | None] = mapped_column(Integer)
    themes: Mapped[Any] = mapped_column(JSON, nullable=True)
    synced_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class DatasetInsight(Base):
    __tablename__ = "dataset_insights"
    __table_args__ = (UniqueConstraint("dataset_id", "region", "delay", "universe", name="uq_insight_scope"),)
    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(64))
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))
    summary: Mapped[str | None] = mapped_column(Text)
    ideas: Mapped[Any] = mapped_column(JSON, nullable=True)
    field_notes: Mapped[Any] = mapped_column(JSON, nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


# ---------------------------------------------------------------- research

class AutopilotRun(Base):
    __tablename__ = "autopilot_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mode: Mapped[str] = mapped_column(String(16))
    goal: Mapped[str | None] = mapped_column(String(64))
    config: Mapped[Any] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")
    reason: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    stopped_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class Batch(Base):
    __tablename__ = "batches"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("autopilot_runs.id"))
    source: Mapped[str] = mapped_column(String(32), default="autopilot")   # autopilot | matrix | manual | spc | genius
    round: Mapped[int] = mapped_column(Integer, default=1)
    region: Mapped[str | None] = mapped_column(String(16))
    universe: Mapped[str | None] = mapped_column(String(32))
    delay: Mapped[int | None] = mapped_column(Integer)
    dataset_id: Mapped[str | None] = mapped_column(String(64))
    strategy: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending")   # pending | awaiting_approval | running | done | halted
    n_generated: Mapped[int] = mapped_column(Integer, default=0)
    n_valid: Mapped[int] = mapped_column(Integer, default=0)
    n_simulated: Mapped[int] = mapped_column(Integer, default=0)
    n_reused: Mapped[int] = mapped_column(Integer, default=0)
    n_promising: Mapped[int] = mapped_column(Integer, default=0)
    n_strong: Mapped[int] = mapped_column(Integer, default=0)
    n_ready: Mapped[int] = mapped_column(Integer, default=0)
    n_rejected: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[Any] = mapped_column(JSON, nullable=True)
    task_id: Mapped[int | None] = mapped_column(Integer)                  # research task this batch belongs to
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class ResearchTask(Base):
    """A unit of research with its own share of BRAIN's concurrent simulation slots ("cores")."""

    __tablename__ = "research_tasks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16))                         # single | pair | interaction | ai_batch
    source: Mapped[str] = mapped_column(String(16), default="manual")    # manual | assisted | autopilot
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))
    dataset_ids: Mapped[Any] = mapped_column(JSON, nullable=True)
    theme: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str | None] = mapped_column(Text)
    planned_sims: Mapped[int] = mapped_column(Integer, default=500)
    cores: Mapped[int] = mapped_column(Integer, default=4)
    status: Mapped[str] = mapped_column(String(16), default="planned", index=True)
    # planned | queued | running | paused | complete | stopped | skipped | failed
    stage: Mapped[str | None] = mapped_column(String(16))                 # analyzing | generating | simulating | optimizing | completed
    plan_day: Mapped[str | None] = mapped_column(String(10), index=True)  # BRAIN (US/Eastern) day it was planned for
    n_generated: Mapped[int] = mapped_column(Integer, default=0)
    n_simulated: Mapped[int] = mapped_column(Integer, default=0)
    n_optimized: Mapped[int] = mapped_column(Integer, default=0)
    n_strong: Mapped[int] = mapped_column(Integer, default=0)
    n_ready: Mapped[int] = mapped_column(Integer, default=0)
    n_rejected: Mapped[int] = mapped_column(Integer, default=0)
    left_out: Mapped[Any] = mapped_column(JSON, nullable=True)            # {reason: count}
    best_sharpe: Mapped[float | None] = mapped_column(Float)
    best_alpha_id: Mapped[str | None] = mapped_column(String(32))
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class Candidate(Base):
    __tablename__ = "candidates"
    __table_args__ = (Index("ix_cand_hash", "canonical_hash", "settings_hash"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    expression: Mapped[str] = mapped_column(Text)
    canonical_hash: Mapped[str | None] = mapped_column(String(32))
    skeleton_hash: Mapped[str | None] = mapped_column(String(32), index=True)
    settings: Mapped[Any] = mapped_column(JSON)
    settings_hash: Mapped[str] = mapped_column(String(32))
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    origin: Mapped[str] = mapped_column(String(32), default="manual")   # llm | template | mutation | optimize | manual | matrix | spc
    parent_id: Mapped[int | None] = mapped_column(Integer)
    parent_alpha_id: Mapped[str | None] = mapped_column(String(32))
    batch_id: Mapped[int | None] = mapped_column(ForeignKey("batches.id"), index=True)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    rationale: Mapped[Any] = mapped_column(JSON, nullable=True)
    dataset_id: Mapped[str | None] = mapped_column(String(64))
    fields: Mapped[Any] = mapped_column(JSON, nullable=True)
    operators: Mapped[Any] = mapped_column(JSON, nullable=True)
    op_count: Mapped[int | None] = mapped_column(Integer)
    field_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="new", index=True)
    # new | invalid | approved | queued | simulating | done | error | reused | rejected_by_user
    error: Mapped[str | None] = mapped_column(Text)
    sim_id: Mapped[str | None] = mapped_column(String(64))
    alpha_id: Mapped[str | None] = mapped_column(String(32), index=True)
    task_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class Simulation(Base):
    __tablename__ = "simulations"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)    # BRAIN simulation id (parent for multi)
    kind: Mapped[str] = mapped_column(String(16))                     # single | multi | super
    location: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="SIMULATING")
    progress: Mapped[float | None] = mapped_column(Float)
    candidate_ids: Mapped[Any] = mapped_column(JSON, nullable=True)
    children: Mapped[Any] = mapped_column(JSON, nullable=True)
    message: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str | None] = mapped_column(String(16))
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class Alpha(Base):
    __tablename__ = "alphas"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)     # BRAIN alpha id
    candidate_id: Mapped[int | None] = mapped_column(Integer, index=True)
    type: Mapped[str] = mapped_column(String(16), default="REGULAR")
    code: Mapped[str | None] = mapped_column(Text)
    selection: Mapped[str | None] = mapped_column(Text)
    combo: Mapped[str | None] = mapped_column(Text)
    settings: Mapped[Any] = mapped_column(JSON, nullable=True)
    region: Mapped[str | None] = mapped_column(String(16), index=True)
    universe: Mapped[str | None] = mapped_column(String(32))
    delay: Mapped[int | None] = mapped_column(Integer)
    neutralization: Mapped[str | None] = mapped_column(String(32))
    decay: Mapped[int | None] = mapped_column(Integer)
    truncation: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str | None] = mapped_column(String(24), index=True)   # UNSUBMITTED | ACTIVE | DECOMMISSIONED
    stage: Mapped[str | None] = mapped_column(String(8))
    grade: Mapped[str | None] = mapped_column(String(32))
    name: Mapped[str | None] = mapped_column(String(256))
    tags: Mapped[Any] = mapped_column(JSON, nullable=True)
    color: Mapped[str | None] = mapped_column(String(32))
    category: Mapped[str | None] = mapped_column(String(64))
    description: Mapped[str | None] = mapped_column(Text)
    classifications: Mapped[Any] = mapped_column(JSON, nullable=True)
    pyramids: Mapped[Any] = mapped_column(JSON, nullable=True)
    themes: Mapped[Any] = mapped_column(JSON, nullable=True)
    competitions: Mapped[Any] = mapped_column(JSON, nullable=True)
    osmosis_points: Mapped[int | None] = mapped_column(Integer)
    date_created: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)
    date_submitted: Mapped[datetime | None] = mapped_column(UTCDateTime)
    date_modified: Mapped[datetime | None] = mapped_column(UTCDateTime)
    sharpe: Mapped[float | None] = mapped_column(Float)
    fitness: Mapped[float | None] = mapped_column(Float)
    turnover: Mapped[float | None] = mapped_column(Float)
    returns: Mapped[float | None] = mapped_column(Float)
    drawdown: Mapped[float | None] = mapped_column(Float)
    margin: Mapped[float | None] = mapped_column(Float)
    long_count: Mapped[int | None] = mapped_column(Integer)
    short_count: Mapped[int | None] = mapped_column(Integer)
    checks: Mapped[Any] = mapped_column(JSON, nullable=True)            # list of {name,result,limit,value,message}
    check_state: Mapped[str | None] = mapped_column(String(16))         # none | pending | pass | fail
    checked_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    self_corr: Mapped[float | None] = mapped_column(Float)
    prod_corr: Mapped[float | None] = mapped_column(Float)
    os: Mapped[Any] = mapped_column(JSON, nullable=True)
    os_sharpe: Mapped[float | None] = mapped_column(Float)
    verdict: Mapped[str | None] = mapped_column(String(16), index=True)  # rejected | weak | promising | strong | ready | submitted
    score: Mapped[float | None] = mapped_column(Float)
    origin: Mapped[str] = mapped_column(String(16), default="app")      # app | synced
    canonical_hash: Mapped[str | None] = mapped_column(String(32), index=True)
    skeleton_hash: Mapped[str | None] = mapped_column(String(32))
    settings_hash: Mapped[str | None] = mapped_column(String(32))
    dataset_ids: Mapped[Any] = mapped_column(JSON, nullable=True)
    fields: Mapped[Any] = mapped_column(JSON, nullable=True)
    op_count: Mapped[int | None] = mapped_column(Integer)
    field_count: Mapped[int | None] = mapped_column(Integer)
    raw: Mapped[Any] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class OptimizationAttempt(Base):
    __tablename__ = "optimization_attempts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent_alpha_id: Mapped[str] = mapped_column(String(32), index=True)
    candidate_id: Mapped[int | None] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(48))
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alpha_id: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="REGULAR")
    status: Mapped[str] = mapped_column(String(16), default="requested")   # requested | accepted | active | rejected | error
    result: Mapped[Any] = mapped_column(JSON, nullable=True)
    requested_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class JournalEntry(Base):
    """Idea journal: every evaluated idea and its verdict (negative memory for the LLM)."""

    __tablename__ = "journal"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    skeleton_hash: Mapped[str | None] = mapped_column(String(32), index=True)
    expression: Mapped[str] = mapped_column(Text)
    hypothesis: Mapped[str | None] = mapped_column(Text)
    dataset_id: Mapped[str | None] = mapped_column(String(64))
    region: Mapped[str | None] = mapped_column(String(16))
    verdict: Mapped[str | None] = mapped_column(String(16))
    sharpe: Mapped[float | None] = mapped_column(Float)
    fitness: Mapped[float | None] = mapped_column(Float)
    turnover: Mapped[float | None] = mapped_column(Float)
    failing_checks: Mapped[Any] = mapped_column(JSON, nullable=True)
    alpha_id: Mapped[str | None] = mapped_column(String(32))


# ---------------------------------------------------------------- operations

class EventRow(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    source: Mapped[str] = mapped_column(String(24))
    level: Mapped[str] = mapped_column(String(12), default="info")
    message: Mapped[str] = mapped_column(Text)
    payload: Mapped[Any] = mapped_column(JSON, nullable=True)


class LLMCall(Base):
    __tablename__ = "llm_calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    provider: Mapped[str | None] = mapped_column(String(16))     # gemini | claude | openai | ollama (NULL = gemini)
    model: Mapped[str] = mapped_column(String(64))
    purpose: Mapped[str] = mapped_column(String(32))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cached_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    error: Mapped[str | None] = mapped_column(Text)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(48))
    title: Mapped[str] = mapped_column(String(256))
    params: Mapped[Any] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)  # queued | running | done | failed | cancelled
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, default=5)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class QuotaDaily(Base):
    __tablename__ = "quota_daily"
    day: Mapped[str] = mapped_column(String(10), primary_key=True)    # YYYY-MM-DD (US/Eastern)
    limit: Mapped[int | None] = mapped_column(Integer)
    remaining: Mapped[int | None] = mapped_column(Integer)
    used_by_app: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


# ---------------------------------------------------------------- programs

class Pyramid(Base):
    __tablename__ = "pyramids"
    __table_args__ = (UniqueConstraint("category_id", "region", "delay", "quarter", name="uq_pyramid"),)
    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_id: Mapped[str] = mapped_column(String(64))
    category_name: Mapped[str | None] = mapped_column(String(128))
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    quarter: Mapped[str] = mapped_column(String(8))                    # e.g. 2026Q4
    multiplier: Mapped[float | None] = mapped_column(Float)
    alpha_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class ConsultantSnapshot(Base):
    __tablename__ = "consultant_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    payload: Mapped[Any] = mapped_column(JSON)


class OsmosisAllocation(Base):
    __tablename__ = "osmosis_allocations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    week: Mapped[str] = mapped_column(String(10), index=True)          # ISO week e.g. 2026-W42
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    alpha_id: Mapped[str] = mapped_column(String(32))
    points: Mapped[int] = mapped_column(Integer)
    method: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="draft")   # draft | pushed | failed
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    pushed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class AlphaPnL(Base):
    """Daily in-sample PnL of an alpha (from the BRAIN `pnl` record set) and statistics derived from it."""

    __tablename__ = "alpha_pnl"
    alpha_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    dates: Mapped[Any] = mapped_column(JSON)                 # ISO dates of the daily increments
    daily: Mapped[Any] = mapped_column(JSON)                 # daily PnL increments ($)
    n_days: Mapped[int] = mapped_column(Integer, default=0)
    sharpe: Mapped[float | None] = mapped_column(Float)      # whole period
    sharpe_early: Mapped[float | None] = mapped_column(Float)  # all but the last 2 years
    sharpe_2y: Mapped[float | None] = mapped_column(Float)
    sharpe_1y: Mapped[float | None] = mapped_column(Float)
    ann_pnl: Mapped[float | None] = mapped_column(Float)
    pnl_1y: Mapped[float | None] = mapped_column(Float)
    max_dd: Mapped[float | None] = mapped_column(Float)      # $ (positive number)
    calmar: Mapped[float | None] = mapped_column(Float)
    spark: Mapped[Any] = mapped_column(JSON, nullable=True)  # ~60 points of cumulative PnL for sparklines


class DatasetRating(Base):
    """AI rating of a dataset for single-field, two-dataset and field-interaction research."""

    __tablename__ = "dataset_ratings"
    __table_args__ = (UniqueConstraint("dataset_id", "region", "delay", "universe", name="uq_rating_scope"),)
    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(String(64))
    region: Mapped[str] = mapped_column(String(16))
    delay: Mapped[int] = mapped_column(Integer)
    universe: Mapped[str] = mapped_column(String(32))
    ai_score: Mapped[float | None] = mapped_column(Float)
    single: Mapped[float | None] = mapped_column(Float)
    double: Mapped[float | None] = mapped_column(Float)
    interaction: Mapped[float | None] = mapped_column(Float)
    recommended: Mapped[str | None] = mapped_column(String(16))   # single | double | interaction
    theme: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str | None] = mapped_column(Text)
    rated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class SPCPrompt(Base):
    __tablename__ = "spc_prompts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[int] = mapped_column(Integer, default=1)
    prompt: Mapped[str] = mapped_column(Text)
    stats: Mapped[Any] = mapped_column(JSON, nullable=True)
    submitted: Mapped[bool] = mapped_column(Boolean, default=False)
    form_settings: Mapped[Any] = mapped_column(JSON, nullable=True)   # {model, version, frequency, weight}
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class SPCRun(Base):
    """One pasted output of an SPC prompt, kept so it can be re-tested later."""

    __tablename__ = "spc_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt_id: Mapped[int | None] = mapped_column(Integer, index=True)
    label: Mapped[str | None] = mapped_column(String(128))
    run_date: Mapped[str] = mapped_column(String(10))                  # YYYY-MM-DD
    test_until: Mapped[str | None] = mapped_column(String(10))
    output: Mapped[Any] = mapped_column(JSON)                          # the parsed JSON array
    extra_runs: Mapped[Any] = mapped_column(JSON, nullable=True)       # stability check: 2nd and 3rd outputs
    check: Mapped[Any] = mapped_column(JSON, nullable=True)
    result: Mapped[Any] = mapped_column(JSON, nullable=True)           # last PnL test summary
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    tested_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
