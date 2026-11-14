"""Facts about the BRAIN platform that the research tools rely on (measured against BRAIN, as in alpha-harness).

Keep numbers here rather than scattered through pages: when BRAIN changes a rule, it changes in one place.
"""

from __future__ import annotations

import re
from typing import Any

INSTRUMENT_TYPE = "EQUITY"
LANGUAGE = "FASTEXPR"
DEFAULT_DECAY = 0
DEFAULT_NEUTRALIZATION = "NONE"
DEFAULT_TRUNCATION = 0.08
DEFAULT_PASTEURIZATION = "ON"
DEFAULT_NAN_HANDLING = "ON"
MAX_DECAY = 512

TEST_PERIOD = "P2Y0M0D"                       # the held-out last two years labs train against
NO_TEST_PERIOD = "P0Y0M0D"
TEST_PERIOD_PATTERN = re.compile(r"^P(?:[0-5]Y(?:[0-9]|1[01])|6Y0)M0D$")

REGION_AGNOSTIC = "ALL"                       # a region-agnostic simulation costs 4 of the daily quota, sent alone
RA_CHILDREN = 4
RA_MIN_PASSING_CHILDREN = 2
MULTI_SIMULATION_SIZE = 10
SIMULATION_SLOTS = 8
MULTI_COUNTRY_REGIONS = frozenset({"GLB", "EUR", "ASI", "AMR"})
MAX_SIMULATIONS = 100_000

FAST_D1_SUFFIX = "_fast_d1"
FAST_D1_MARKET = ("USA", 1)

QUICK_MODE = "QUICK"
FULL_MODE = "FULL"

# Power Pool
POWER_POOL_MAX_OPERATORS = 8
POWER_POOL_MAX_FIELDS = 3
POWER_POOL_CHECK = "POWER_POOL_CORRELATION"   # its presence (not its result) marks an alpha as Power Pool eligible
POWER_POOL_CORRELATION_CEILING = 0.5
POWER_POOL_ESCAPE = 1.1                       # a collision is allowed when Sharpe ≥ 1.1 × the collided alpha's
POWER_POOL_TAG = "PowerPoolSelected"
SELF_CORRELATION_CEILING = 0.7
PROD_CORRELATION_CEILING = 0.7
PROD_CORRELATION_CHECK = "PROD_CORRELATION"
POWER_POOL_CLASS = "Power Pool Alpha"
REGULAR_CLASS = "Regular Alpha"
SINGLE_DATASET_CLASS = "DATA_USAGE:SINGLE_DATA_SET"

# Super Alpha
SUPER_ALPHA_SLOTS = 3
SUPER_ALPHA_MIN_SELECTION = 10
SUPER_ALPHA_OPERATOR_LIMIT = 8000

# metrics
BOOK_SIZE = 20_000_000
TRADING_DAYS = 250                            # BRAIN annualises with 250 days (measured), not 252
FITNESS_TURNOVER_FLOOR = 0.125
CORRELATION_WINDOW_YEARS = 4
MIN_OVERLAP = 250
IMPLAUSIBLE_SHARPE = 10.0
PYRAMID_MIN_ALPHAS = 3

GROUPING_FIELDS = frozenset({"market", "sector", "industry", "subindustry", "country", "exchange", "currency",
                             "split", "adjfactor"})
UNCOUNTED_OPERATORS = frozenset({"ts_backfill", "group_backfill"})   # not counted by the Power Pool operator limit

# checks that only meter the daily allowance: stripped from stored checks
QUOTA_CHECKS = frozenset({"REGULAR_SUBMISSION", "D0_SUBMISSION"})
# checks that never gate a submission
IGNORED_CHECKS = frozenset({
    "PROD_CORRELATION", "MATCHES_COMPETITION", "MATCHES_PYRAMID", "MATCHES_THEMES", "CLUSTER_TEST",
    "OSMOSIS_ALLOCATION", "POWER_POOL_DESCRIPTION_LENGTH", "POWER_POOL_DESCRIPTION_FORMAT",
})
NEAR_MISS_CHECKS = frozenset({"HIGH_TURNOVER", "LOW_TURNOVER", "LOW_FITNESS", "CONCENTRATED_WEIGHT",
                              "LOW_SUB_UNIVERSE_SHARPE", "SELF_CORRELATION"})
STRUCTURAL_CHECKS = frozenset({"HIGH_TURNOVER", "LOW_TURNOVER", "CONCENTRATED_WEIGHT", "LOW_SUB_UNIVERSE_SHARPE"})

# investability: where Max Position may be used; elsewhere only Max Trade
MAX_POSITION_REGIONS = frozenset({"USA", "EUR", "GLB", "ASI", "ALL"})
MAX_TRADE_REGIONS = frozenset({"ASI", "JPN", "HKG", "TWN", "KOR"})   # Max Trade keeps ≥ 70% of Sharpe here

PLATFORM_URL = "https://platform.worldquantbrain.com"


def alpha_url(alpha_id: str) -> str:
    return f"{PLATFORM_URL}/alpha/{alpha_id}"


_BREADTH = re.compile(r"TOP[A-Z]*(\d+)[A-Z]*")


def breadth(universe: str | None) -> int | None:
    """Names in a universe from its label (TOP3000 -> 3000, TOPSP500 -> 500); None when it does not say."""
    m = _BREADTH.search(str(universe or "").upper())
    return int(m.group(1)) if m else None


def truncation_for(region: str, delay: int, universe: str | None) -> float:
    """Truncation Agent: the largest single-name weight that keeps a book of this breadth diversified.

    Broad books (≥2000 names) take 0.08, ≥1000 names 0.06, narrower ones 0.05; multi-country regions and delay 0
    never exceed 0.05. USA D1 TOP3000 → 0.08, TOP1000 → 0.06, TOPSP500 → 0.05, USA D0 → 0.05, GLB → 0.05."""
    names = breadth(universe)
    value = 0.08 if names is None or names >= 2000 else 0.06 if names >= 1000 else 0.05
    if str(region).upper() in MULTI_COUNTRY_REGIONS | {REGION_AGNOSTIC} or int(delay) == 0:
        value = min(value, 0.05)
    return min(value, 0.08)


def investability_pairs(region: str) -> list[tuple[str, str]]:
    """(maxTrade, maxPosition) pairs a region allows: both OFF, one ON; never both ON."""
    position = str(region).upper() in MAX_POSITION_REGIONS
    return [(t, p) for t in ("OFF", "ON") for p in (("OFF", "ON") if position else ("OFF",)) if not (t == p == "ON")]


INVESTABILITY = {"none": ("OFF", "OFF"), "max_trade": ("ON", "OFF"), "max_position": ("OFF", "ON")}


def investability_of(settings: dict[str, Any] | None) -> str:
    s = settings or {}
    if str(s.get("maxPosition") or "OFF").upper() == "ON":
        return "max_position"
    if str(s.get("maxTrade") or "OFF").upper() == "ON":
        return "max_trade"
    return "none"


def is_power_pool(checks: list[dict[str, Any]] | None) -> bool:
    return any(str(c.get("name")) == POWER_POOL_CHECK for c in checks or [])


def strip_quota_checks(checks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [c for c in checks or [] if str(c.get("name")) not in QUOTA_CHECKS]


def brain_verdict(checks: list[dict[str, Any]] | None, simulation_mode: str | None = None) -> str | None:
    """BRAIN's own reading of an alpha's checks: submittable | pending | refused | None (no checks yet).

    Quick-mode alphas are never submittable (BRAIN needs a Full run). Quota and informational checks are ignored."""
    if str(simulation_mode or "").upper() == QUICK_MODE:
        return "refused"
    gating = {str(c.get("result") or "").upper() for c in strip_quota_checks(checks)
              if str(c.get("name")) not in IGNORED_CHECKS}
    if not gating:
        return None
    if gating & {"FAIL", "ERROR"}:
        return "refused"
    if gating <= {"PASS", "WARNING"}:
        return "submittable"
    return "pending"


def failed_checks(checks: list[dict[str, Any]] | None, *, gating_only: bool = False) -> list[str]:
    return sorted(str(c.get("name")) for c in strip_quota_checks(checks)
                  if str(c.get("result") or "").upper() in ("FAIL", "ERROR")
                  and not (gating_only and str(c.get("name")) in IGNORED_CHECKS))


def is_near_miss(checks: list[dict[str, Any]] | None) -> bool:
    failed = set(failed_checks(checks, gating_only=True))
    return bool(failed) and failed <= NEAR_MISS_CHECKS and len(failed) <= 2


def is_implausible(*sharpes: float | None) -> bool:
    return any(s is not None and abs(float(s)) > IMPLAUSIBLE_SHARPE for s in sharpes)


def check_limits(checks: list[dict[str, Any]] | None) -> dict[str, float]:
    """The cutoffs an alpha's own checks name (they differ by market): LOW_SHARPE, LOW_FITNESS, ..."""
    out: dict[str, float] = {}
    for c in checks or []:
        name = str(c.get("name"))
        if name in ("LOW_SHARPE", "LOW_FITNESS", "LOW_TURNOVER", "HIGH_TURNOVER") and c.get("limit") is not None:
            try:
                out[name] = float(c["limit"])
            except (TypeError, ValueError):
                continue
    return out
