"""Tables of the research labs and tools ported from alpha-harness (Search, Template, Evolution, Power Pool and
Super Alpha labs; Settings / Fields Samplers, Correlation Breaker, Paste Alphas).

A `LabTask` is one run of a lab with its own share of the simulation cores; each `LabTrial` is one simulation it
asked for (or chose not to send). Trials link to the simulator's `Candidate` rows, so dedup, packing, quota and
recovery are the simulator's, not repeated here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base, UTCDateTime, utcnow


class LabTask(Base):
    __tablename__ = "lab_tasks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # search | template | evolution | power_pool | super | settings_sampler | fields_sampler |
    # correlation_breaker | paste
    lab: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str | None] = mapped_column(String(160))
    # IDLE (added) | QUEUED (waiting for cores) | RUNNING | PAUSED | COMPLETE | FAILED
    status: Mapped[str] = mapped_column(String(12), default="IDLE", index=True)
    stopping: Mapped[bool] = mapped_column(Boolean, default=False)     # Stop pressed: only score what is out
    cores: Mapped[int] = mapped_column(Integer, default=4)
    max_trials: Mapped[int] = mapped_column(Integer, default=500)     # simulations to spend (free answers excluded)
    region: Mapped[str | None] = mapped_column(String(16))
    delay: Mapped[int | None] = mapped_column(Integer)
    universe: Mapped[str | None] = mapped_column(String(32))
    objective: Mapped[str] = mapped_column(String(24), default="sharpe")   # sharpe | train_sharpe | train_fitness
    params: Mapped[Any] = mapped_column(JSON, nullable=True)          # the lab's inputs, as the form sent them
    space: Mapped[Any] = mapped_column(JSON, nullable=True)           # what the lab planned (frozen at start)
    state: Mapped[Any] = mapped_column(JSON, nullable=True)           # the lab's own running memory (tallies, ...)
    message: Mapped[str | None] = mapped_column(Text)
    best_value: Mapped[float | None] = mapped_column(Float)
    best_alpha_id: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    queued_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class LabTrial(Base):
    __tablename__ = "lab_trials"
    __table_args__ = (Index("ix_trial_task_state", "task_id", "state"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(Integer, index=True)
    number: Mapped[int] = mapped_column(Integer)                      # order within the task
    # PARKED (written, waiting for cores) | RUNNING (sent) | COMPLETE | FAIL | PRUNED (not sent)
    state: Mapped[str] = mapped_column(String(10), default="PARKED")
    params: Mapped[Any] = mapped_column(JSON, nullable=True)          # what the lab chose (for the search to learn)
    expression: Mapped[str] = mapped_column(Text)
    settings: Mapped[Any] = mapped_column(JSON)
    settings_hash: Mapped[str | None] = mapped_column(String(32))
    candidate_id: Mapped[int | None] = mapped_column(Integer, index=True)
    alpha_id: Mapped[str | None] = mapped_column(String(32), index=True)
    value: Mapped[float | None] = mapped_column(Float)                # objective minus structural penalty
    penalty: Mapped[float | None] = mapped_column(Float)
    free: Mapped[bool] = mapped_column(Boolean, default=False)        # answered from an alpha already simulated
    implausible: Mapped[bool] = mapped_column(Boolean, default=False)
    generation: Mapped[int] = mapped_column(Integer, default=1)       # 0 = seed (Evolution)
    parent: Mapped[Any] = mapped_column(JSON, nullable=True)          # trial numbers it came from
    message: Mapped[str | None] = mapped_column(Text)
    figures: Mapped[Any] = mapped_column(JSON, nullable=True)         # sharpe, fitness, ... failed checks at harvest
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class LabTemplate(Base):
    """A Template Lab template the consultant saved: its text and its variables' definitions.

    Names are unique case-insensitively (`name_key` is the lowered name), so "Momentum" and "momentum" cannot both
    exist and a lookup by name is exact."""

    __tablename__ = "lab_templates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    name_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(String(500))
    text: Mapped[str] = mapped_column(Text, default="")
    variables: Mapped[Any] = mapped_column(JSON, nullable=True)      # {name: {kind: fields|values, ...}}
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
