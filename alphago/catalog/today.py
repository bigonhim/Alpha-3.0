"""Today: what you have, and what you are about to waste (alpha-harness api/today.py).

The true daily simulation limit reaches the app only in the response headers of a simulation POST, so the figure
starts as the assumed allowance minus today's runs and is replaced by BRAIN's own number once the first batch comes
back. `exact` says which one is shown; an estimate is written with a leading "~".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..brain.catalog_api import PLATFORM_TZ

DAY_SECONDS = 24 * 60 * 60


def headline(remaining: int, limit: int, queued: int = 0) -> str:
    """The sentence at the top of the Dashboard. Names what is still unused while there is some, and stops nagging
    once the day is claimed. No clock in it: the reset countdown is shown on its own."""
    if limit <= 0:
        return "No simulations available today."
    if remaining <= 0:
        return "You have used every simulation today. Nothing was wasted."
    unspoken = max(0, remaining - queued)
    if unspoken <= 0:
        return "Every simulation left today is queued. Nothing is going to waste."
    share = unspoken / limit
    tail = f" {queued:,} more are already queued." if queued else ""
    if share > 0.9:
        return f"{unspoken:,} simulations are unused today. They cannot be saved for tomorrow.{tail}"
    if share > 0.5:
        return f"{unspoken:,} simulations left today.{tail}"
    return f"{unspoken:,} left of {limit:,}. Good day so far.{tail}"


def same_platform_day(epoch: float | None, now: float | None = None) -> bool:
    if epoch is None:
        return False
    day = datetime.fromtimestamp(epoch, PLATFORM_TZ).date()
    return day == datetime.fromtimestamp(now or time.time(), PLATFORM_TZ).date()


@dataclass
class Simulations:
    limit: int
    used: int
    remaining: int
    queued: int
    #: What is genuinely still going to waste: not yet run and not yet claimed.
    unspoken: int
    #: False means "assumed from the allowance", True means "BRAIN told us today".
    exact: bool
    resets_in_seconds: int
    headline: str

    @property
    def day_gone(self) -> float:
        """How much of BRAIN's quota day has passed (the mark on the depletion bar)."""
        return min(1.0, max(0.0, 1 - self.resets_in_seconds / DAY_SECONDS))

    @property
    def used_share(self) -> float:
        return self.used / self.limit if self.limit else 0.0

    @property
    def remaining_text(self) -> str:
        return f"{'' if self.exact else '~'}{self.remaining:,}"


def simulations_today(quota: Any = None, queued: int | None = None) -> Simulations:
    """From the client's quota tracker: BRAIN's own figure wins whenever one arrived today."""
    from ..brain import brain
    from ..engine.simulator import simulator
    q = quota or brain.client.quota
    allowance = int(q.effective_limit)
    exact = q.remaining is not None and same_platform_day(q.updated_at)
    if exact:
        limit = int(q.limit or allowance)
        remaining = int(q.remaining)
        used = max(0, limit - remaining)
    else:
        limit = allowance
        used = int(q.sent_today)
        remaining = max(0, allowance - used)
    # queued work has not been sent, so it is not in `used` — but it is spoken for, and counting it as waste would
    # tell someone who just queued their day that they did nothing
    queued = int(simulator.queued_count if queued is None else queued)
    return Simulations(limit=limit, used=used, remaining=remaining, queued=queued,
                       unspoken=max(0, remaining - queued), exact=exact,
                       resets_in_seconds=round(q.seconds_to_reset), headline=headline(remaining, limit, queued))


@dataclass
class Step:
    key: str
    title: str
    done: bool
    detail: str
    action: str
    target: str


async def getting_started() -> list[Step]:
    """What a new consultant sets up before anything else, ticked off from real state (never a stored flag). The
    Dashboard hides the list once every step is done."""
    from ..llm import llm
    from . import queries
    from . import sync as catalog_sync
    synced = bool(await queries.synced_tuples())
    run = catalog_sync.live.payload() if catalog_sync.live is not None else None
    syncing = bool(run and run["status"] == "RUNNING")
    # done the moment the first market lands, not when the whole catalogue has
    landed = bool(run and any(m["state"] in ("fields", "details", "done") for m in run["markets"]))
    data_detail = ("Downloading from BRAIN…" if syncing else
                   (run["error"] or "") if run and run["status"] == "FAILED" else
                   "Downloads every market's data fields, so research and the Data Explorer have something to use.")
    keyed = bool(llm.configure())
    try:
        models = bool(llm.models) or llm.status == "ok"
    except Exception:
        models = False
    return [
        Step("data", "Sync BRAIN Datasets", synced or landed, data_detail, "View" if syncing else "Sync", "/sync"),
        Step("ai", "Add a Free API Key", keyed,
             "A free key (Gemini, OpenRouter) is what writes ideas and explains the data in plain English.",
             "Add a Key", "/llm"),
        # a key alone runs nothing: the assistant and the AI labs need a model on it
        Step("model", "Set Up a Model", keyed and models,
             "Load the provider's models so AlphaGo can pick one for ideas, repairs and descriptions.",
             "Set Up a Model", "/llm"),
    ]
