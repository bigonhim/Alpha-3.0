"""Which template families actually produce alphas worth keeping, learned from your own simulation history.

Every template-built candidate records its family in `Candidate.rationale["family"]`. A "hit" is a simulated
candidate whose alpha reached promising or better. Families are ordered by Thompson sampling on a Beta(1+hits,
1+misses) posterior: families with a record of hits come first, untried families still get explored, and a family
that keeps missing sinks without being banned.
"""

from __future__ import annotations

import random
from typing import Any

from sqlalchemy import case, func, select

from ..db import db
from ..db.models import Alpha, Candidate

HIT_VERDICTS = ("promising", "strong", "ready", "submitted")


async def family_stats(region: str | None = None, delay: int | None = None) -> dict[str, tuple[int, int]]:
    """{family: (hits, trials)} over simulated template candidates in this market."""
    fam = func.json_extract(Candidate.rationale, "$.family")
    hit = func.sum(case((Alpha.verdict.in_(HIT_VERDICTS), 1), else_=0))
    q = (select(fam, hit, func.count(Candidate.id))
         .join(Alpha, Alpha.id == Candidate.alpha_id)
         .where(Candidate.alpha_id.is_not(None), fam.is_not(None)))
    if region:
        q = q.where(Candidate.region == region)
    if delay is not None:
        q = q.where(Candidate.delay == int(delay))
    async with db.session() as s:
        rows = (await s.execute(q.group_by(fam))).all()
    return {str(f): (int(h or 0), int(n or 0)) for f, h, n in rows if f}


def thompson_order(families: list[str], stats: dict[str, tuple[int, int]], rng: random.Random | None = None,
                   prior: tuple[float, float] = (1.0, 4.0)) -> list[str]:
    """Families sorted by a draw from each one's Beta posterior (prior ≈ a 20% hit rate before any data)."""
    rng = rng or random.Random()
    a0, b0 = prior

    def draw(f: str) -> float:
        h, n = stats.get(f, (0, 0))
        return rng.betavariate(a0 + h, b0 + max(0, n - h))

    return sorted(dict.fromkeys(families), key=draw, reverse=True)


def weighted_families(families: list[str], stats: dict[str, tuple[int, int]], k: int,
                      rng: random.Random | None = None) -> list[str]:
    """The k families to draw templates from this time."""
    return thompson_order(families, stats, rng)[:max(1, k)]


def summary(stats: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    rows = [{"family": f, "hits": h, "trials": n, "hit_rate": h / n if n else None} for f, (h, n) in stats.items()]
    return sorted(rows, key=lambda r: (-(r["hit_rate"] or 0), -r["trials"]))
