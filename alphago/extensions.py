"""Loads the feature packages ported from alpha-harness, so their labs and job kinds exist before anything runs.

Each package owns one `register`-style module; a failure in one is logged and leaves the rest of the app working.
"""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger(__name__)

MODULES = (
    "alphago.labs.labs",          # Search, Template and Evolution labs
    "alphago.labs.llm_labs",      # LLM Power Pool lab and Super Alpha lab
    "alphago.tools.register",     # Settings / Fields Samplers, Correlation Breaker, Paste Alphas
    "alphago.pool.register",      # tags, portfolio, Power Pool correlation, submission order
    "alphago.engine.catalog_ext", # all-markets catalogue sync, pyramids, competitions, quarter
    "alphago.ai.register",        # chat assistant, key rotation, budget ledger
    "alphago.engine.series",      # BRAIN-exact daily series (job kind fetch_series)
)


def load() -> list[str]:
    """Import every feature module; returns the ones that failed."""
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            log.exception("feature module %s failed to load", name)
            failed.append(name)
    return failed
