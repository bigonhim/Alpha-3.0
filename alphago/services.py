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
    "alphago.engine.spc_auto",     # SPC generation and submission jobs
    "alphago.engine.aqp",          # Alpha Quality Project jobs
    "alphago.engine.themes",       # BRAIN themes (GLB): theme data sync and checks
    "alphago.engine.python_alpha",  # Python alphas: generate, check and simulate
]

for _name in _MODULES:
    try:
        importlib.import_module(_name)
    except ModuleNotFoundError as exc:
        if exc.name != _name:
            raise
        log.debug("optional module %s not present yet", _name)
