"""Settings Sampler: run one proven expression everywhere BRAIN will accept it (alpha-harness tools/settings_sampler).

The expression, decay and truncation are held exactly as the source Alpha has them, unless the Truncation Agent sets
truncation market by market or a Decay sweep asks for several. What varies is the market — region, delay, universe —
plus neutralization and the maxTrade/maxPosition pair. A market only counts when every data field the expression
reads is downloaded there, so a two-field Alpha is judged on the intersection (coverage is the thinnest field's, and
is shown only: BRAIN reports zero for some markets that simulate fine).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from itertools import product
from typing import Any

from ..brain.platform import (
    DEFAULT_NAN_HANDLING,
    DEFAULT_PASTEURIZATION,
    DEFAULT_TRUNCATION,
    MAX_DECAY,
    MAX_POSITION_REGIONS,
    MULTI_SIMULATION_SIZE,
    RA_CHILDREN,
    REGION_AGNOSTIC,
    TEST_PERIOD,
    TEST_PERIOD_PATTERN,
    investability_pairs,
    truncation_for,
)
from ..db.models_labs import LabTask
from ..labs.runner import Lab, RefillContext, TrialSpec
from .common import ToolError, alpha_body, alpha_id_of, code_of, field_availability, grouping_in, synced_markets

NAME = "settings_sampler"
NO_NEUTRALIZATION = "NONE"
DECAY_STEPS = (0, 2, 5, 10, 20)
NO_NEUTRALIZATION_CHOSEN = "Choose at least one neutralization: nothing chosen means nothing to run."


def decay_ladder(own: int) -> list[int]:
    """Steps above the Alpha's own decay: higher lowers turnover, too much attenuates the signal."""
    return sorted({min(int(own) + s, MAX_DECAY) for s in DECAY_STEPS})


def simulation_cost(region: str) -> int:
    """Simulations of the day's allowance one run in this region uses (All Regions: one per region it reaches)."""
    return RA_CHILDREN if region == REGION_AGNOSTIC else 1


def batch_size(region: str) -> int:
    """How many runs one multi-simulation carries: BRAIN fails a batch of region-agnostic ones."""
    return 1 if region == REGION_AGNOSTIC else MULTI_SIMULATION_SIZE


def market_neutral(neutralization: str, trade: str, position: str) -> bool:
    """Whether this combination holds the book against the market at all: NONE with neither Max Trade nor Max
    Position is the one that does not (nothing projected out, nothing capped)."""
    return neutralization != NO_NEUTRALIZATION or trade == "ON" or position == "ON"


def _source_settings(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "region": source.get("region"), "universe": source.get("universe"), "delay": source.get("delay"),
        "neutralization": source.get("neutralization"), "decay": source.get("decay"),
        "truncation": source.get("truncation"),
        "maxTrade": source.get("maxTrade") or "OFF", "maxPosition": source.get("maxPosition") or "OFF",
        "pasteurization": source.get("pasteurization") or DEFAULT_PASTEURIZATION,
        "nanHandling": source.get("nanHandling") or DEFAULT_NAN_HANDLING,
        "testPeriod": source.get("testPeriod") or TEST_PERIOD,
        "language": source.get("language") or "FASTEXPR",
        "instrumentType": source.get("instrumentType") or "EQUITY",
    }


def _empty(alpha_id: str, expression: str, fields: list[str], source: dict[str, Any], problems: list[str],
           warnings: list[str], grouping: list[str] | None = None) -> dict[str, Any]:
    return {"alpha_id": alpha_id, "expression": expression, "data_fields": fields, "grouping_fields": grouping or [],
            "settings": _source_settings(source), "regions": [], "totals": {"total": 0, "batches": 0},
            "problems": problems, "warnings": warnings}


def regions_of(held: dict[tuple[str, int, str], float]) -> list[dict[str, Any]]:
    """Per region: its delays, universes, neutralizations (from BRAIN's options), legal pairs and markets."""
    from ..brain import brain
    by_region: dict[str, dict[int, dict[str, float]]] = {}
    for (region, delay, universe), coverage in held.items():
        by_region.setdefault(region, {}).setdefault(delay, {})[universe] = coverage
    out: list[dict[str, Any]] = []
    for region, delays in by_region.items():
        neutralizations = [str(n) for n in brain.options.neutralizations(region)]
        pairs = investability_pairs(region)
        markets = [{"region": region, "delay": delay, "universe": universe, "coverage": coverage,
                    "total": len(neutralizations) * len(pairs),
                    "agent_truncation": truncation_for(region, delay, universe)}
                   for delay in sorted(delays) for universe, coverage in sorted(delays[delay].items())]
        out.append({"region": region, "delays": sorted(delays), "universes": sorted({m["universe"] for m in markets}),
                    "neutralizations": neutralizations,
                    "pairs": [{"maxTrade": t, "maxPosition": p} for t, p in pairs],
                    "position_available": region in MAX_POSITION_REGIONS, "cost": simulation_cost(region),
                    "markets": markets, "total": sum(int(m["total"]) for m in markets)})
    out.sort(key=lambda r: (-int(r["total"]), str(r["region"])))       # richest first, ties alphabetical
    return out


def totals_of(regions: list[dict[str, Any]]) -> dict[str, int]:
    batches = 0
    for region in regions:
        per_delay: dict[int, int] = {}
        for market in region["markets"]:
            per_delay[int(market["delay"])] = per_delay.get(int(market["delay"]), 0) + int(market["total"])
        size = batch_size(str(region["region"]))
        batches += sum(math.ceil(n / size) for n in per_delay.values())
    return {"total": sum(int(r["total"]) * int(r["cost"]) for r in regions), "batches": batches}


async def unsynced_count() -> int:
    """Markets BRAIN offers that the catalogue has never downloaded: availability is read from downloaded fields,
    so an unsynced market is invisible rather than empty, and saying so keeps a partial catalogue honest."""
    from ..brain import brain
    opts = brain.options
    offered = {(str(r), int(d), str(u)) for r in opts.regions if r != REGION_AGNOSTIC
               for d in opts.delays(r) for u in opts.universes(r)}
    return len(offered - await synced_markets())


async def plan(alpha_id: str = "", *, expression: str | None = None, decay: int | None = None,
               truncation: float | None = None, pasteurization: str | None = None, nan_handling: str | None = None,
               test_period: str | None = None) -> dict[str, Any]:
    """Everything the screen needs: the expression, its fields and the markets they open up. Simulates nothing.

    From an Alpha, the expression and every setting are its own to begin with; decay, truncation, pasteurization,
    NaN handling and the test period can each be overridden (re-running at a different decay is as much a sweep)."""
    from ..fastexpr import FastExprSyntaxError
    from ..labs.fastexpr import data_fields
    problems: list[str] = []
    warnings: list[str] = []
    if bool((alpha_id or "").strip()) == bool((expression or "").strip()):
        raise ToolError("Give either an Alpha ID or an expression.")
    if expression:
        source: dict[str, Any] = {}
        expression = expression.strip()
    else:
        alpha_id = alpha_id_of(alpha_id)
        body = await alpha_body(alpha_id)
        expression = code_of(body)
        source = dict(body.get("settings") or {})
    source["decay"] = source.get("decay", 0) if decay is None else int(decay)
    source["truncation"] = source.get("truncation", DEFAULT_TRUNCATION) if truncation is None else float(truncation)
    source["pasteurization"] = pasteurization or source.get("pasteurization") or DEFAULT_PASTEURIZATION
    source["nanHandling"] = nan_handling or source.get("nanHandling") or DEFAULT_NAN_HANDLING
    if test_period and not TEST_PERIOD_PATTERN.fullmatch(test_period.upper()):
        raise ToolError(f"Test Period {test_period!r} is not like P2Y0M0D (from P0Y0M0D to P6Y0M0D).")
    source["testPeriod"] = (test_period or "").upper() or source.get("testPeriod") or TEST_PERIOD
    if not 0 <= int(source["decay"] or 0) <= MAX_DECAY:
        raise ToolError(f"Decay must be a whole number from 0 to {MAX_DECAY}.")
    if not expression:
        problems.append(f"{alpha_id or 'The expression'} has no expression to re-run.")
        return _empty(alpha_id, "", [], source, problems, warnings)
    try:
        from .common import fields_in_order
        fields = fields_in_order(expression)
        # grouping fields do not count as data but must exist where it runs, so markets are placed on all it reads
        placed = fields_in_order(expression, grouping=True)
        data_fields(expression)
    except (FastExprSyntaxError, ValueError):
        problems.append("Its expression could not be parsed, so its data fields are unknown.")
        return _empty(alpha_id, expression, [], source, problems, warnings)
    if not fields:
        problems.append("Its expression reads no data field, so there is nothing to place.")
        return _empty(alpha_id, expression, fields, source, problems, warnings, grouping_in(placed))
    availability = await field_availability(placed)
    held: dict[tuple[str, int, str], float] = {}
    for index, f in enumerate(placed):
        here = availability.get(f) or {}
        if not here:
            problems.append(f"{f} is not downloaded in any market. Sync its dataset on the Datasets page first.")
            return _empty(alpha_id, expression, fields, source, problems, warnings, grouping_in(placed))
        # every field must be present; the Alpha is only as covered as its thinnest field
        held = dict(here) if index == 0 else {k: min(v, here[k]) for k, v in held.items() if k in here}
    if not held:
        problems.append(f"No downloaded market holds all of {', '.join(fields)} together.")
        return _empty(alpha_id, expression, fields, source, problems, warnings, grouping_in(placed))
    from ..brain import brain
    if not brain.options.live:
        warnings.append("BRAIN's settings list has not been read since sign-in, so neutralizations are the usual "
                        "list rather than this account's. Sign in to read it.")
    missing = await unsynced_count()
    if missing:
        warnings.append(f"{missing} market{'s' if missing > 1 else ''} BRAIN offers "
                        f"{'are' if missing > 1 else 'is'} not downloaded, so this is what your catalogue can see "
                        "rather than everything BRAIN offers.")
    regions = regions_of(held)
    return {"alpha_id": alpha_id, "expression": expression, "data_fields": fields,
            "grouping_fields": grouping_in(placed), "settings": _source_settings(source), "regions": regions,
            "totals": totals_of(regions), "problems": problems, "warnings": warnings}


def expand(plan_rows: list[dict[str, Any]], chosen: set[tuple[str, int, str]], neutralizations: set[str],
           pairs: set[tuple[str, str]], source: dict[str, Any], *, market_neutral_only: bool = True,
           truncation_agent: bool = False, decays: Sequence[int] = (),
           rng: random.Random | None = None) -> list[dict[str, Any]]:
    """Every simulation the selection asks for, as {expression, settings}, ordered for packing and for watching.

    A multi-simulation's children share region and delay, so the shuffling is a batch at a time: each ten stay in
    one market and the order the tens go out is random. Only full tens are shuffled among themselves; part-full
    tails go last, where a short batch costs nothing the remainder was not already going to cost."""
    rng = rng or random.Random()
    expression = str(source.get("expression") or "")
    swept = sorted(set(int(d) for d in decays)) or [int(source.get("decay") or 0)]
    held = source.get("truncation")
    truncation = DEFAULT_TRUNCATION if held is None else float(held)
    base = {"instrumentType": source.get("instrumentType") or "EQUITY", "language": source.get("language") or "FASTEXPR",
            "pasteurization": str(source.get("pasteurization") or DEFAULT_PASTEURIZATION),
            "nanHandling": str(source.get("nanHandling") or DEFAULT_NAN_HANDLING),
            "testPeriod": str(source.get("testPeriod") or TEST_PERIOD),
            "visualization": bool(source.get("visualization"))}
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for region in plan_rows:
        legal_pairs = [(str(p["maxTrade"]), str(p["maxPosition"])) for p in region["pairs"]
                       if not pairs or (p["maxTrade"], p["maxPosition"]) in pairs]
        legal_neutral = [str(n) for n in region["neutralizations"] if not neutralizations or n in neutralizations]
        markets = [m for m in region["markets"]
                   if not chosen or (str(m["region"]), int(m["delay"]), str(m["universe"])) in chosen]
        for market, neutralization, (trade, position), decay in product(markets, legal_neutral, legal_pairs, swept):
            if market_neutral_only and not market_neutral(neutralization, trade, position):
                continue
            key = (str(market["region"]), int(market["delay"]))
            settings = {**base, "region": key[0], "universe": str(market["universe"]), "delay": key[1],
                        "neutralization": neutralization, "decay": int(decay),
                        "truncation": truncation_for(*key, str(market["universe"])) if truncation_agent
                        else truncation, "maxTrade": trade, "maxPosition": position}
            groups.setdefault(key, []).append({"expression": expression, "settings": settings})
    full: list[list[dict[str, Any]]] = []
    tails: list[list[dict[str, Any]]] = []
    for (region, _delay), members in groups.items():
        size = batch_size(region)
        for i in range(0, len(members), size):
            chunk = members[i:i + size]
            (full if len(chunk) == size else tails).append(chunk)
    rng.shuffle(full)
    rng.shuffle(tails)
    return [request for chunk in (*full, *tails) for request in chunk]


def defaults_choice(found: dict[str, Any]) -> dict[str, Any]:
    """Preset "defaults": every market but All Regions (four simulations a run, so only on purpose), every
    neutralization and every legal pair."""
    return {"markets": sorted((m["region"], m["delay"], m["universe"]) for r in found["regions"]
                              if r["region"] != REGION_AGNOSTIC for m in r["markets"]),
            "neutralizations": sorted({n for r in found["regions"] for n in r["neutralizations"]}),
            "pairs": sorted({(p["maxTrade"], p["maxPosition"]) for r in found["regions"] for p in r["pairs"]}),
            "decays": []}


def own_choice(found: dict[str, Any]) -> dict[str, Any]:
    """Preset "own settings": the Alpha's own market, neutralization and investability alone."""
    own = found["settings"]
    return {"markets": [(own.get("region"), int(own.get("delay") or 0), own.get("universe"))]
            if own.get("region") else [],
            "neutralizations": [own["neutralization"]] if own.get("neutralization") else [],
            "pairs": [(own.get("maxTrade") or "OFF", own.get("maxPosition") or "OFF")], "decays": []}


def ladder_choice(found: dict[str, Any]) -> dict[str, Any]:
    """Preset "decay ladder": the Alpha's own settings at its decay and the steps above it."""
    return {**own_choice(found), "decays": decay_ladder(int(found["settings"].get("decay") or 0))}


def requests_for(found: dict[str, Any], params: dict[str, Any], rng: random.Random | None = None) -> list[dict]:
    """The simulations a form asks for (refuses an empty neutralization list: nobody chose what to run)."""
    neutralizations = list(params.get("neutralizations") or [])
    if not neutralizations:
        raise ToolError(NO_NEUTRALIZATION_CHOSEN)
    chosen = {(str(m[0]), int(m[1]), str(m[2])) for m in params.get("markets") or []} or {
        (str(m["region"]), int(m["delay"]), str(m["universe"])) for r in found["regions"]
        if r["region"] != REGION_AGNOSTIC for m in r["markets"]}
    pairs = {(str(p[0]), str(p[1])) for p in params.get("pairs") or []}
    if ("ON", "ON") in pairs:
        raise ToolError("Max Trade and Max Position cannot both be ON.")
    source = {**found["settings"], "expression": found["expression"], "visualization": bool(params.get("visualization"))}
    return expand(found["regions"], chosen, set(neutralizations), pairs, source,
                  market_neutral_only=bool(params.get("market_neutral_only", True)),
                  truncation_agent=bool(params.get("truncation_agent")), decays=params.get("decays") or (), rng=rng)


async def preview(params: dict[str, Any]) -> dict[str, Any]:
    """The plan, plus what the current choice would queue: count (All Regions at its own cost), batches, sample."""
    found = await plan_of(params)
    out = {**found, "simulations": 0, "cost": 0, "batches": 0, "sample": []}
    if found["problems"]:
        return out
    try:
        reqs = requests_for(found, params, random.Random(0))
    except ToolError as exc:
        out["problems"] = [str(exc)]
        return out
    per_market: dict[tuple[str, int], int] = {}
    for r in reqs:
        key = (r["settings"]["region"], r["settings"]["delay"])
        per_market[key] = per_market.get(key, 0) + 1
    out.update(simulations=len(reqs), cost=sum(simulation_cost(r["settings"]["region"]) for r in reqs),
               batches=sum(math.ceil(n / batch_size(k[0])) for k, n in per_market.items()), sample=reqs[:8])
    if not reqs:
        out["problems"] = ["Nothing to run: choose a market, a neutralization and a Max Trade / Max Position pair."]
    return out


async def plan_of(params: dict[str, Any]) -> dict[str, Any]:
    return await plan(params.get("alpha_id") or "", expression=params.get("expression") or None,
                      decay=params.get("decay"), truncation=params.get("truncation"),
                      pasteurization=params.get("pasteurization"), nan_handling=params.get("nan_handling"),
                      test_period=params.get("test_period"))


class SettingsSamplerLab(Lab):
    """Every simulation is written up front (in batch-shuffled order); the task drains them on its cores."""

    name = NAME
    title = "Settings Sampler"

    async def plan(self, task: LabTask) -> dict[str, Any]:
        found = await plan_of(task.params or {})
        if found["problems"]:
            raise ToolError(found["problems"][0])
        reqs = requests_for(found, task.params or {})
        if not reqs:
            raise ToolError("Nothing to run: choose a market, a neutralization and a Max Trade / Max Position pair.")
        return {"expression": found["expression"], "settings": found["settings"], "markets": len(
            {(r["settings"]["region"], r["settings"]["delay"], r["settings"]["universe"]) for r in reqs}),
            "requests": reqs}

    async def initial(self, task: LabTask) -> list[TrialSpec]:
        return await written_up_front(task)

    def done_early(self, task: LabTask, ctx: RefillContext) -> str | None:
        return all_run(ctx)

    async def preview(self, params: dict[str, Any]) -> dict[str, Any]:
        return await preview(params)


async def written_up_front(task: LabTask, *, budget: int | None = None) -> list[TrialSpec]:
    """The planned requests as parked trials, in order. The task's budget becomes their number (or `budget`) and the
    stored plan keeps only its summary: the trials are the record."""
    from sqlalchemy import update

    from ..db import db
    space = dict(task.space or {})
    reqs = space.pop("requests", []) or []
    space["simulations"] = len(reqs)
    async with db.write() as s:
        await s.execute(update(LabTask).where(LabTask.id == task.id).values(
            space=space, max_trials=max(1, budget if budget is not None else len(reqs))))
    return [TrialSpec(r["expression"], r["settings"], params=dict(r.get("params") or {}),
                      generation=int(r.get("generation") or 1)) for r in reqs]


def all_run(ctx: RefillContext) -> str | None:
    """Finish quietly once nothing is parked or out: a task written up front has nothing more to add."""
    if ctx.seen and not any(v is True for v in ctx.seen.values()):
        return "Every simulation has run."
    return None
