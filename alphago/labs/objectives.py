"""What a lab task maximises, and what the search is told a trial scored (alpha-harness labs/objectives.py).

The runner (`labs.runner.score`) stores each trial's value as its objective less the structural penalty, or FAILURE
for an implausible or unscored one; this module is what the TPE study is told from those rows.
"""

from __future__ import annotations

from typing import Any

from ..brain.platform import QUICK_MODE
from .runner import FAILURE, OBJECTIVES

#: What each structural check failed costs a trial's score in the search: about the distance from a passing figure
#: to a failing one, so a trial failing one ranks below the clean ones near the cutoff without vanishing.
STRUCTURAL_PENALTY = 1.0
#: The objective each lab scores on: train years only where a test period splits them off.
LAB_OBJECTIVES = {"search": "train_sharpe", "template": "train_sharpe", "evolution": "train_fitness"}


def label(objective: str | None) -> str:
    return OBJECTIVES.get(objective or "sharpe", "Sharpe")


def searched(value: float | None, implausible: bool = False) -> float:
    """What the search is told: the stored value (already less its penalty), FAILURE for a straight line."""
    if implausible or value is None:
        return FAILURE
    return float(value)


def search_sharpe(train: float | None, whole: float | None) -> float | None:
    """The Sharpe a search may judge on: the train years', or the whole period's without a split."""
    return whole if train is None else train


def objective_for(lab: str, test_period: str | None) -> str:
    """No test period leaves no train years, so the whole period is scored."""
    own = LAB_OBJECTIVES.get(lab, "sharpe")
    if str(test_period or "") in ("", "P0Y0M0D") and own.startswith("train_"):
        return own.removeprefix("train_")
    return own


def is_quick(settings: dict[str, Any] | None) -> bool:
    return str((settings or {}).get("simulationMode") or "").upper() == QUICK_MODE
