"""Registers this package's labs, job kinds and listeners (imported once at startup by alphago.extensions).

Search, Template and Evolution labs run on the shared lab runner; Evolution's Auto Select is a background job
(`evolution_seeds`) that reads the local store and downloads daily PnL where missing, never simulating.
"""

from __future__ import annotations

from ..jobs.queue import jobs
from .ga import EvolutionLab, _job_auto_seeds
from .runner import runner
from .search import SearchLab
from .template import TemplateLab

search_lab = runner.register(SearchLab())
template_lab = runner.register(TemplateLab())
evolution_lab = runner.register(EvolutionLab())
jobs.register("evolution_seeds")(_job_auto_seeds)
