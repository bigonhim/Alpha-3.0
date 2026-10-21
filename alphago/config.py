"""Application settings, stored as JSON in %APPDATA%\\AlphaGo\\settings.json."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

APP_NAME = "AlphaGo"

log = logging.getLogger(__name__)


def app_dir() -> Path:
    base = Path(os.environ.get("ALPHAGO_HOME") or Path(os.environ.get("APPDATA", Path.home())) / APP_NAME)
    base.mkdir(parents=True, exist_ok=True)
    return base


def db_path() -> Path:
    return app_dir() / "alphago.db"


def log_dir() -> Path:
    path = app_dir() / "logs"
    path.mkdir(exist_ok=True)
    return path


class BrainSettings(BaseModel):
    base_url: str = "https://api.worldquantbrain.com"
    max_slots: int = 8                # concurrent multi-simulations allowed by BRAIN for consultants
    multi_size: int = 10              # alphas per multi-simulation (2..10)
    quota_reserve: int = 200          # never spend the last N simulations of the day
    relogin_threshold_s: int = 1000   # re-login when session has fewer seconds left
    request_timeout_s: float = 60.0
    max_poll_minutes: int = 45        # give up polling a single simulation after this long
    catalog_concurrency: int = 4      # data-fields requests in flight during a catalogue sync (1 = one at a time)


class ProviderSettings(BaseModel):
    """Connection and per-role models for one non-Gemini LLM provider (empty model = provider's automatic choice)."""
    base_url: str = ""
    ideation_model: str = ""
    refine_model: str = ""
    describe_model: str = ""
    timeout_s: float = 0              # 0 = provider default
    num_ctx: int = 32_768             # Ollama only: context window sent with every request (capped at the model's)
    num_predict: int = 8_192          # Ollama only: max tokens per reply (bounds a runaway generation)


class TournamentSettings(BaseModel):
    """Run the ideation prompt on several models and learn which one yields more promising/strong alphas per call."""
    enabled: bool = False
    contestants: list[str] = Field(default_factory=list)   # "provider:model", e.g. "gemini:gemini-3-flash"
    min_ideas: int = 60               # simulated ideas per contestant before calls follow the results
    locked: str = ""                  # "provider:model" chosen as the winner (empty = still competing)


class LLMSettings(BaseModel):
    provider: str = "gemini"          # the one active provider: gemini | claude | openai | ollama
    # Gemini's models per role (kept at the top level so older settings.json files load unchanged)
    ideation_model: str = ""          # empty = auto-pick the best available "flash" model
    refine_model: str = ""            # empty = auto-pick the strongest available model
    describe_model: str = ""
    thinking_level: str = "low"       # minimal | low | medium | high
    start_rpm: float = 6.0            # adaptive pacing starts here and learns from 429s
    ideas_per_call: int = 8
    daily_call_budget: int = 400
    ai_language: str = "English"      # English | Hinglish (AI explanations only; numbers/expressions unchanged)
    self_check: bool = True           # one extra AI call per batch: critique the ideas, drop the worst (≤ 25%)
    fallbacks: list[str] = Field(default_factory=list)   # providers tried in order when the primary fails (empty = off)
    tournament: TournamentSettings = Field(default_factory=TournamentSettings)
    claude: ProviderSettings = Field(default_factory=ProviderSettings)
    openai: ProviderSettings = Field(default_factory=lambda: ProviderSettings(base_url="https://api.openai.com/v1"))
    ollama: ProviderSettings = Field(default_factory=lambda: ProviderSettings(base_url="http://localhost:11434"))


class ResearchSettings(BaseModel):
    mode: str = "assisted"            # manual | assisted | autopilot
    goal: str = "general"             # general | power_pool | pyramid | genius
    region: str = "USA"
    universe: str = "TOP3000"
    delay: int = 1
    batch_size: int = 12
    max_rounds: int = 3
    quota_share: float = 0.6          # share of remaining daily quota autopilot may spend
    stop_after_empty_batches: int = 6
    generator_mix: dict[str, float] = Field(default_factory=lambda: {"llm": 0.5, "template": 0.3, "mutation": 0.2})
    dataset_strategy: str = "opportunity"   # opportunity | pyramid_gap | under_used | pinned
    pinned_datasets: list[str] = Field(default_factory=list)
    robustness_gate: bool = True
    family_cap_per_batch: int = 3
    # research tasks (Search Lab / Double dataset · Pair / Interaction)
    autopilot_daily_cap: int = 500    # simulations the Autopilot may spend per BRAIN day (manual tasks are not capped)
    auto_daily: bool = False          # plan (and in Full Autopilot, start) the day's research after the quota reset
    research_focus: str = ""          # dataset theme to prioritise; empty = choose automatically
    task_cores: int = 4               # default cores (concurrent multi-simulations) per task
    task_sims: int = 500              # default planned simulations per task
    tasks_per_day: int = 2
    # repair loop and certification (engine/refine.py)
    optimize_share: float = 0.2       # share of a task's planned simulations reserved for repairing near misses
    repair_depth: int = 3             # rounds of stacked fixes per near miss
    check_cap: int = 15               # BRAIN /check runs per task or batch (a check costs no simulation quota)
    duplicate_filter: bool = True     # leave out exact shape+field copies of your submitted alphas before simulating
    cluster_share: float = 0.4        # max share of the day's simulations per economic idea (0 = no cap)
    cluster_min: int = 150            # ... but never cap an idea below this many simulations a day


class ProgramSettings(BaseModel):
    # Genius thresholds (community-reported, editable): signals, pyramids, combined performance
    genius_thresholds: dict[str, list[float]] = Field(default_factory=lambda: {
        "EXPERT": [20, 10, 0.5],
        "MASTER": [120, 30, 1.0],
        "GRANDMASTER": [220, 60, 2.0],
    })
    osmosis_points_total: int = 100_000
    osmosis_min_points: int = 1_000
    osmosis_max_points: int = 15_000
    osmosis_min_alphas: int = 10
    osmosis_max_alphas: int = 35
    osmosis_lock_weekday: int = 6     # 0=Mon .. 6=Sun (inferred from a screenshot; editable)
    osmosis_lock_hour_et: int = 23
    osmosis_lock_minute: int = 59     # Sun 23:59 New York = "Mon 9:29 AM" India time in the reference app
    osmosis_plan_max: int = 30        # max alphas per scope in "Make best plan"
    osmosis_leave_out_sharpe2y: float = 0.5   # left out: last-2-year Sharpe below this ...
    osmosis_leave_out_turnover: float = 0.5   # ... or turnover above this
    pp_max_per_scope: int = 15
    pp_min_rating: str = "good"       # very_good | good | average
    pp_max_corr: float = 0.5          # Power Pool: picked alphas must correlate below this with each other
    daily_regular_submissions: int = 4
    daily_super_submissions: int = 1


class SPCSettings(BaseModel):
    """Systematic Predictions Challenge. BRAIN does not publish the form rules, so everything here is editable."""
    model: str = "gpt"
    version: str = "5.6"
    frequency: str = "Daily"
    weight: str = "0.20"
    max_chars: int = 10_000
    required_fields: list[str] = Field(default_factory=lambda: ["company_name", "ticker"])
    min_rows: int = 1
    max_rows: int = 0                 # 0 = no limit
    weight_field: str = ""            # if rows carry a weight, name the field (else equal weight)
    side_field: str = ""              # if rows carry long/short, name the field (values: long/short or 1/-1)
    benchmark: str = "SPY"


class AlertSettings(BaseModel):
    """Email alerts (SMTP). The password is kept in Windows Credential Manager, never in this file."""
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    security: str = "starttls"        # starttls | ssl | none
    smtp_user: str = ""
    sender: str = ""                  # empty = the SMTP user
    recipient: str = ""
    triggers: dict[str, bool] = Field(default_factory=lambda: {
        "ready": True, "task_finished": True, "stuck": True, "quota": True, "persona": True, "login_failed": True,
        "llm_failing": True, "daily_summary": False})
    quiet_minutes: int = 30           # the same alert at most once per this many minutes
    stuck_minutes: int = 30           # queued simulations but none finished for this long = stuck
    summary_hour: int = 18            # local hour of the daily summary (when switched on)


class ServiceSettings(BaseModel):
    """Running without the window (`--headless`)."""
    keep_awake: bool = True           # keep the PC from sleeping while simulations are queued or running


class AppSettings(BaseModel):
    brain: BrainSettings = Field(default_factory=BrainSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    programs: ProgramSettings = Field(default_factory=ProgramSettings)
    spc: SPCSettings = Field(default_factory=SPCSettings)
    autopilot_warning_accepted: bool = False

    @classmethod
    def path(cls) -> Path:
        return app_dir() / "settings.json"

    @classmethod
    def load(cls) -> "AppSettings":
        path = cls.path()
        if path.exists():
            try:
                return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except Exception:  # corrupted file: keep a backup and start fresh
                log.exception("settings.json unreadable; using defaults")
                path.replace(path.with_suffix(".bad.json"))
        return cls()

    def save(self) -> None:
        tmp = self.path().with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.path())


settings = AppSettings.load()
