"""Import side-effect modules that register background job kinds and engine listeners."""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger(__name__)

_MODULES = [
    "alphago.engine.actions",      # check / submit / properties jobs
    "alphago.engine.autopilot",
    "alphago.engine.programs",     # pyramids, genius, power pool, osmosis, oos
    "alphago.engine.datasets",     # dataset analysis jobs
    "alphago.engine.pnl",          # "Prepare data": daily PnL of submitted alphas
]

for _name in _MODULES:
    try:
        importlib.import_module(_name)
    except ModuleNotFoundError as exc:
        if exc.name != _name:
            raise
        log.debug("optional module %s not present yet", _name)
