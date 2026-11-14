"""Registers this package's labs, job kinds and listeners (imported once at startup by alphago.extensions).

Job kinds: `catalog_sync_all` (every market's data fields and datasets; params `region_agnostic`) and
`catalog_search_index` (rebuild the Data Explorer's smart-search index).
"""

from __future__ import annotations

from ..catalog import sync as catalog_sync

catalog_sync.register()
