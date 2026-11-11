"""Application settings, stored as JSON in %APPDATA%\\AlphaGo\\settings.json."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

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
    catalog_concurrency: int = 2      # data-fields requests in flight during a catalogue sync (1 = one at a time)
    catalog_rps: float = 2.0          # starting pace of catalogue requests per second (halves on every 429, 0 = unpaced)
    fields_resync_hours: int = 72     # a field sync skips datasets synced more recently than this (0 = always re-sync)


class ProviderSettings(BaseModel):
    """Connection and per-role models for one non-Gemini LLM provider (empty model = provider's automatic choice)."""
    base_url: str = ""
    ideation_model: str = ""
    refine_model: str = ""
    describe_model: str = ""
    timeout_s: float = 0              # 0 = provider default
    num_ctx: int = 32_768             # Ollama only: context window sent with every request (capped at the model's)
    num_predict: int = 8_192          # Ollama only: max tokens per reply (bounds a runaway generation)
    free_only: bool = True            # OpenRouter only: list and auto-pick free models only


class TournamentSettings(BaseModel):
    """Run the ideation prompt on several models and learn which one yields more promising/strong alphas per call."""
    enabled: bool = False
    contestants: list[str] = Field(default_factory=list)   # "provider:model", e.g. "gemini:gemini-3-flash"
    min_ideas: int = 60               # simulated ideas per contestant before calls follow the results
    locked: str = ""                  # "provider:model" chosen as the winner (empty = still competing)


class LLMSettings(BaseModel):
    provider: str = "gemini"          # the one active provider: gemini | claude | openai | openrouter | ollama
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
    openrouter: ProviderSettings = Field(default_factory=lambda: ProviderSettings(base_url="https://openrouter.ai/api/v1"))


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
    repair_task_sims: int = 150       # default budget of a "Repair near misses" task
    duplicate_filter: bool = True     # leave out exact shape+field copies of your submitted alphas before simulating
    cluster_share: float = 0.4        # max share of the day's simulations per economic idea (0 = no cap)
    cluster_min: int = 150            # ... but never cap an idea below this many simulations a day
    # field screening (search tasks): one probe per field, then the search spends its budget on the fields that worked
    field_screening: bool = True
    screen_min_sharpe: float = 0.5    # |Sharpe| a probe needs for its field to be kept (negative = used sign-flipped)
    screen_keep_share: float = 0.35   # keep at most this share of a dataset's fields ...
    screen_keep_min: int = 4          # ... but at least this many (the best ones, even below the bar)


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
    # --- AI generation (SPC Lab "Generate & check"): how many local runs, how strict the gates are
    auto_runs: int = 3                # local runs of the prompt, for the form check and the stability gate
    auto_max_rounds: int = 2          # AI rewrites of a prompt that fails a gate (0 = never rewrite)
    auto_min_overlap: float = 0.5     # stability gate: mean pairwise stock overlap (Jaccard) between runs
    auto_check_prices: bool = True    # ticker gate: every ticker must have Yahoo prices
    auto_min_priced: float = 0.9      # ... at least this share of the tickers
    auto_backtest_days: int = 0       # >0: also test the portfolio as of N days ago (the AI may know that period)
    auto_min_excess: float = 0.0      # backtest gate: return minus benchmark must be at least this
    # --- submission to BRAIN. BRAIN publishes no SPC endpoint: copy the path and body from the browser's network
    # tab when you submit the form by hand once (or paste "Copy as cURL" into Import). Placeholders in string values:
    # {prompt} {name} {model} {version} {frequency} {weight} {competition}; "{weight:num}" sends a number.
    submit_path: str = ""             # e.g. "/competitions/{competition}/submissions" (empty = not set up)
    submit_method: str = "POST"
    competition_id: str = ""          # fills {competition} in the path and body
    submit_body: str = ('{"name": "{name}", "prompt": "{prompt}", "model": "{model}", "version": "{version}", '
                        '"frequency": "{frequency}", "weight": "{weight}"}')
    max_submits_per_day: int = 1      # app-side guard against double submitting
    auto_submit: bool = False         # OFF by default: submit without the confirmation click once every gate passed

    @model_validator(mode="before")
    @classmethod
    def _migrate(cls, data: Any) -> Any:
        """settings.json files from before the split: body_template -> submit_body, min_overlap -> auto_min_overlap."""
        if isinstance(data, dict):
            data = dict(data)
            if "body_template" in data and "submit_body" not in data:
                data["submit_body"] = data.pop("body_template")
            if "min_overlap" in data and "auto_min_overlap" not in data:
                data["auto_min_overlap"] = data.pop("min_overlap")
            if "stability_runs" in data and "auto_runs" not in data:
                data["auto_runs"] = max(2, int(data.pop("stability_runs") or 2))
        return data


class AQPSettings(BaseModel):
    """Alpha Quality Project: a monthly cycle where up to 3 USA alphas are tagged AQP<YYYYMM> and described. The scoring
    (80% description, 20% out-of-sample performance) is BRAIN's; these knobs only set the app's own pre-checks."""
    tag_prefix: str = "AQP"
    region: str = "USA"
    max_picks: int = 3
    min_description_chars: int = 700      # pre-check: long enough to explain every field and operator
    min_grade: int = 7                    # pre-check: each AI-judge score (match, uniqueness, hypothesis) out of 10
    max_rounds: int = 2                   # AI rewrites of a description that fails a pre-check
    max_similarity: float = 0.55          # the picked alphas' descriptions must be this different from each other


class ThemeDef(BaseModel):
    """One BRAIN theme (announced in the platform's notifications; BRAIN publishes no theme API). Alphas whose settings
    and data match a theme earn its multiplier, and BRAIN's check reports the match as MATCHES_THEMES."""
    name: str = "GLB"
    enabled: bool = True
    region: str = "GLB"
    delay: int = 1
    universe: str = "TOP3000"
    neutralization: str = "COUNTRY"   # used by the theme's research tasks ("" = the dataset category's default)
    decay: int = -1                   # -1 = the dataset category's default
    dataset_ids: list[str] = Field(default_factory=list)   # empty = BRAIN's theme datasets (theme=true), else best rated
    start: str = ""                   # YYYY-MM-DD, from the announcement (empty = open)
    end: str = ""
    multiplier: float = 1.0           # from the announcement (informational: BRAIN applies it, the app only ranks by it)
    note: str = ""


class ThemeSettings(BaseModel):
    items: list[ThemeDef] = Field(default_factory=lambda: [ThemeDef()])
    auto_daily: bool = False          # after each quota reset, plan one task per active theme (Assisted: for approval)
    task_sims: int = 300              # simulations per theme task
    task_cores: int = 2               # GLB allows fewer concurrent simulations than other regions
    check_matches: bool = True        # run BRAIN's check on strong theme-scope alphas to read MATCHES_THEMES


class PythonAlphaSettings(BaseModel):
    """Alphas written in Python instead of FASTEXPR. BRAIN publishes no public spec: the app reads the allowed
    `settings.language` values from OPTIONS /simulations and only simulates when your account offers a Python one."""
    language: str = ""                # empty = detect from OPTIONS /simulations (a value containing "PYTHON")
    reference: str = ""               # paste BRAIN's Python Alpha docs / an example here: the AI writes code against it
    starter: str = ""                 # your template; the AI keeps its structure
    max_chars: int = 20_000
    ai_rounds: int = 2                # AI fixes of code that fails the local syntax/safety check


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
    aqp: AQPSettings = Field(default_factory=AQPSettings)
    themes: ThemeSettings = Field(default_factory=ThemeSettings)
    python_alpha: PythonAlphaSettings = Field(default_factory=PythonAlphaSettings)
    autopilot_warning_accepted: bool = False

    @classmethod
    def path(cls) -> Path:
        return app_dir() / "settings.json"

    @classmethod
    def load(cls) -> AppSettings:
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
