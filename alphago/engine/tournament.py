"""Model tournament and prompt-version results: which model (and which prompt) produces alphas that go somewhere.

When the tournament is on, the ideation calls of every AI batch are shared among the chosen contestants
("provider:model") — the same prompt, no extra calls. Each idea records the model that proposed it, so the
scoreboard can compare models per call and per 100 simulations. Until every contestant has `min_ideas` simulated
ideas the calls rotate; after that each call goes to a Thompson draw on "strong-or-better per simulated idea", so the
better model gets more calls while the others still get some. "Lock the winner" sends every ideation call to it.
"""

from __future__ import annotations

import math
import random
from typing import Any

from sqlalchemy import func, select

from ..config import settings
from ..db import db
from ..db.models import Alpha, Candidate, LLMCall

STRONG = ("strong", "ready", "submitted")
GOOD = ("promising",) + STRONG
PASSED = ("ready", "submitted")


def parse(label: str) -> tuple[str, str] | None:
    if ":" not in (label or ""):
        return None
    provider, model = label.split(":", 1)
    return provider.strip(), model.strip()


def contestants() -> list[tuple[str, str]]:
    from ..llm.router import PROVIDERS
    out = []
    for label in settings.llm.tournament.contestants:
        p = parse(label)
        if p and p[0] in PROVIDERS and PROVIDERS[p[0]].configure():
            out.append(p)
    return out


def wilson(k: int, n: int, z: float = 1.2816) -> tuple[float, float]:
    """80% interval for a rate."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


async def _results(column, keys: list[Any] | None = None) -> dict[Any, dict[str, int]]:  # noqa: ANN001
    async with db.session() as s:
        stmt = (select(column, Candidate.status, Alpha.verdict, func.count())
                .outerjoin(Alpha, Alpha.id == Candidate.alpha_id).where(column.is_not(None))
                .group_by(column, Candidate.status, Alpha.verdict))
        if keys is not None:
            stmt = stmt.where(column.in_(keys))
        rows = (await s.execute(stmt)).all()
    out: dict[Any, dict[str, int]] = {}
    for key, status, verdict, n in rows:
        r = out.setdefault(key, {"ideas": 0, "valid": 0, "simulated": 0, "promising_plus": 0, "strong_plus": 0,
                                 "passed": 0})
        r["ideas"] += n
        r["valid"] += 0 if status == "invalid" else n
        r["simulated"] += n if status in ("done", "reused") else 0
        r["promising_plus"] += n if verdict in GOOD else 0
        r["strong_plus"] += n if verdict in STRONG else 0
        r["passed"] += n if verdict in PASSED else 0
    return out


async def scoreboard() -> list[dict[str, Any]]:
    """Per model: ideas, valid share, simulated, promising+/strong+/passed per call and per 100 simulations."""
    labels = list(dict.fromkeys(settings.llm.tournament.contestants))
    res = await _results(Candidate.llm_model)
    labels += [k for k in res if k not in labels]
    async with db.session() as s:
        calls = {(p or "gemini", m): n for p, m, n in (await s.execute(
            select(LLMCall.provider, LLMCall.model, func.count()).where(LLMCall.purpose == "ideation")
            .group_by(LLMCall.provider, LLMCall.model))).all()}
    out = []
    for label in labels:
        r = res.get(label, {"ideas": 0, "valid": 0, "simulated": 0, "promising_plus": 0, "strong_plus": 0,
                            "passed": 0})
        pm = parse(label) or ("?", label)
        n_calls = calls.get((pm[0], pm[1][:64]), 0)
        lo, hi = wilson(r["strong_plus"], r["simulated"])
        out.append({"label": label, "provider": pm[0], "model": pm[1], "calls": n_calls, **r,
                    "valid_share": round(r["valid"] / r["ideas"], 3) if r["ideas"] else None,
                    "strong_per_call": round(r["strong_plus"] / n_calls, 3) if n_calls else None,
                    "strong_per_100": round(100 * r["strong_plus"] / r["simulated"], 1) if r["simulated"] else None,
                    "strong_per_100_lo": round(100 * lo, 1), "strong_per_100_hi": round(100 * hi, 1),
                    "contestant": label in settings.llm.tournament.contestants,
                    "locked": label == settings.llm.tournament.locked})
    return sorted(out, key=lambda r: (not r["contestant"], -(r["strong_per_100"] or 0)))


async def choose(rng: random.Random | None = None) -> tuple[str, str] | None:
    """The contestant for the next ideation call, or None (tournament off: the normal provider answers)."""
    t = settings.llm.tournament
    if not t.enabled:
        return None
    if t.locked:
        return parse(t.locked)
    pool = contestants()
    if not pool:
        return None
    rng = rng or random.Random()
    labels = [f"{p}:{m}" for p, m in pool]
    res = await _results(Candidate.llm_model, labels)
    stats = [res.get(label, {"simulated": 0, "strong_plus": 0, "ideas": 0}) for label in labels]
    if any(s["simulated"] < t.min_ideas for s in stats):
        i = min(range(len(pool)), key=lambda j: (stats[j]["ideas"], rng.random()))   # rotate until all have data
        return pool[i]
    draws = [rng.betavariate(1 + s["strong_plus"], 1 + s["simulated"] - s["strong_plus"]) for s in stats]
    return pool[max(range(len(pool)), key=lambda j: draws[j])]


async def lock_winner() -> str | None:
    """Lock the contestant with the best strong-or-better rate (each needs `min_ideas` simulated ideas)."""
    t = settings.llm.tournament
    board = [r for r in await scoreboard() if r["contestant"] and r["simulated"] >= t.min_ideas]
    if not board:
        return None
    best = max(board, key=lambda r: (r["strong_plus"] + 1) / (r["simulated"] + 2))
    t.locked = best["label"]
    settings.save()
    return best["label"]


async def prompt_results(name: str) -> list[dict[str, Any]]:
    """Each saved version of a prompt with what its ideas did (ideation only: the other prompts make no ideas)."""
    from ..llm.prompts import versions
    vs = await versions(name)
    res = await _results(Candidate.prompt_version, [v.id for v in vs]) if vs else {}
    out = []
    for v in vs:
        r = res.get(v.id, {"ideas": 0, "valid": 0, "simulated": 0, "promising_plus": 0, "strong_plus": 0,
                           "passed": 0})
        out.append({"id": v.id, "source": v.source, "note": v.note or "",
                    "created": v.created_at.strftime("%Y-%m-%d %H:%M") if v.created_at else "", **r,
                    "strong_per_100": round(100 * r["strong_plus"] / r["simulated"], 1) if r["simulated"] else None})
    return out
