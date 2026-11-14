"""Registers this package's labs, job kinds and listeners (imported once at startup by alphago.extensions):
the LLM Power Pool Lab ("power_pool") and the Super Alpha Lab ("super")."""

from __future__ import annotations

from .power_pool_lab import PowerPoolLab
from .runner import runner
from .super_lab import SuperLab, start_recovery

power_pool = runner.register(PowerPoolLab())
super_alpha = runner.register(SuperLab())

# at startup: SuperAlphas that were out when the app closed are polled again (or closed, never resent); imported
# outside the app's event loop (tests, tools), the first SuperAlpha send does it instead
start_recovery()
