"""Registers this package's labs, job kinds and listeners (imported once at startup by alphago.extensions).

Each tool that simulates is a `Lab` on the lab task runner, so it gets cores, dedup and free answers, restart
recovery and a place in the task list: Settings Sampler, Correlation Breaker, Fields Sampler and Paste Alphas (which
also runs Flip Sign).
"""

from __future__ import annotations

from ..labs.runner import runner
from .correlation_breaker import CorrelationBreakerLab
from .fields_sampler import FieldsSamplerLab
from .paste_alphas import PasteLab
from .settings_sampler import SettingsSamplerLab

LABS = (SettingsSamplerLab(), CorrelationBreakerLab(), FieldsSamplerLab(), PasteLab())


def register(target=runner) -> None:  # noqa: ANN001  (a LabRunner)
    for lab in LABS:
        target.register(lab)


register()
