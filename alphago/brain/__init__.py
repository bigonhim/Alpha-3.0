"""WorldQuant BRAIN API layer."""

from __future__ import annotations

from ..config import settings
from ..credentials import brain_credentials
from .alphas import AlphasAPI, SubmitResult, check_failures, parse_checks
from .client import BrainClient
from .data import DataAPI
from .errors import (AuthError, BrainError, ConcurrencyLimit, PersonaRequired, QuotaExhausted, SimulationRejected)
from .options import SimOptions
from .simulations import (DEFAULT_SETTINGS, SimOutcome, SimulationAPI, group_key, normalize_settings,
                          regular_payload, settings_hash, super_payload)
from .users import UsersAPI


class Brain:
    """Facade bundling the BRAIN endpoint groups around one authenticated client."""

    def __init__(self) -> None:
        self.client = BrainClient(settings.brain, brain_credentials)
        self.sims = SimulationAPI(self.client)
        self.alphas = AlphasAPI(self.client)
        self.data = DataAPI(self.client)
        self.users = UsersAPI(self.client)
        self.options = SimOptions()

    async def load_options(self) -> SimOptions:
        response = await self.client.request("OPTIONS", "/simulations")
        payload = self.client.raise_for(response)
        self.options = SimOptions.from_payload(payload or {})
        return self.options


brain = Brain()

__all__ = [
    "Brain", "brain", "BrainClient", "BrainError", "AuthError", "PersonaRequired", "ConcurrencyLimit",
    "QuotaExhausted", "SimulationRejected", "SimOptions", "SimOutcome", "SubmitResult", "DEFAULT_SETTINGS",
    "normalize_settings", "regular_payload", "super_payload", "settings_hash", "group_key", "parse_checks",
    "check_failures",
]
