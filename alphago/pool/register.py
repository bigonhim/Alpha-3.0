"""Registers this package's labs, job kinds and listeners (imported once at startup by alphago.extensions)."""

from __future__ import annotations

from ..engine import series  # noqa: F401  (job kind fetch_series, which the pool pages queue)
from . import brainops  # noqa: F401  (job kinds: pool_import_tagged, pool_prod_correlation, pool_power_pool_workflow)
