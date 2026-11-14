"""Paste Alphas: an LLM's output read back into simulations (alpha-harness tools/paste_alphas), and Flip Sign.

The output is expected in the shape Copy Alphas writes: `Alpha Expression:`, the expression, then
`Simulation Settings:` and one `Title: Value` line per setting. It is read loosely, since a model rarely keeps a
format to the letter: Markdown bold, headings, bullets and code fences are ignored, and a setting matches BRAIN's
label or its value in any case. Settings are resolved against the choices BRAIN offers this account
(`brain.options`): region, delay and universe are required (there is no market to run in without them); a setting
left out takes BRAIN's default only where that default is offered; Max Trade and Max Position both ON is refused;
an exact repeat runs once.
"""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import select

from ..brain.platform import (
    DEFAULT_DECAY,
    DEFAULT_NAN_HANDLING,
    DEFAULT_NEUTRALIZATION,
    DEFAULT_PASTEURIZATION,
    DEFAULT_TRUNCATION,
    FULL_MODE,
    INSTRUMENT_TYPE,
    LANGUAGE,
    MAX_DECAY,
    MAX_POSITION_REGIONS,
    MAX_SIMULATIONS,
    MULTI_SIMULATION_SIZE,
    RA_CHILDREN,
    REGION_AGNOSTIC,
    TEST_PERIOD_PATTERN,
)
from ..db import db
from ..db.models_labs import LabTask, LabTrial
from ..labs.runner import Lab, RefillContext, TrialSpec
from .common import ToolError, alpha_body, alpha_id_of, code_of, described
from .settings_sampler import all_run, written_up_front

NAME = "paste"

SETTINGS = (
    ("region", "Region"), ("delay", "Delay"), ("universe", "Universe"), ("language", "Language"),
    ("decay", "Decay"), ("truncation", "Truncation"), ("neutralization", "Neutralization"),
    ("pasteurization", "Pasteurization"), ("nanHandling", "NaN Handling"), ("maxTrade", "Max Trade"),
    ("maxPosition", "Max Position"), ("testPeriod", "Test Period"),
)
TITLES = dict(SETTINGS)
REQUIRED = ("region", "delay", "universe")
DEFAULTS: dict[str, Any] = {
    "language": LANGUAGE, "decay": DEFAULT_DECAY, "truncation": DEFAULT_TRUNCATION,
    "neutralization": DEFAULT_NEUTRALIZATION, "pasteurization": DEFAULT_PASTEURIZATION,
    "nanHandling": DEFAULT_NAN_HANDLING, "maxTrade": "OFF", "maxPosition": "OFF",
}
# BRAIN's labels for the values whose label differs from the value (what its settings panel and Copy Alphas show)
_ON_OFF = {"ON": "On", "OFF": "Off"}
LABELS: dict[str, dict[str, str]] = {
    "language": {"FASTEXPR": "Fast Expression"},
    "neutralization": {"NONE": "None", "MARKET": "Market", "SECTOR": "Sector", "INDUSTRY": "Industry",
                       "SUBINDUSTRY": "Subindustry", "COUNTRY": "Country", "STATISTICAL": "Statistical",
                       "CROWDING": "Crowding Factors", "FAST": "Fast Factors", "SLOW": "Slow Factors",
                       "SLOW_AND_FAST": "Slow + Fast Factors", "REVERSION_AND_MOMENTUM": "Reversion and Momentum"},
    "pasteurization": _ON_OFF, "nanHandling": _ON_OFF, "maxTrade": _ON_OFF, "maxPosition": _ON_OFF,
}


def label(name: str, value: Any) -> str:
    """BRAIN's label for a chosen setting value (the value itself where BRAIN shows it as is)."""
    if value is None:
        return "—"
    return LABELS.get(name, {}).get(str(value).upper() if isinstance(value, str) else str(value), str(value))


def labels_of(settings: dict[str, Any]) -> dict[str, str]:
    return {name: label(name, settings.get(name)) for name, _ in SETTINGS if settings.get(name) is not None}


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


_SETTING_KEYS = {_key(t): n for n, t in SETTINGS} | {_key(n): n for n, _ in SETTINGS}
_DECORATION = re.compile(r"^(?:[#>]+\s*|[-*+]\s+|\d+[.)]\s+)*")
_LABEL = re.compile(r"^[A-Za-z][A-Za-z /]*$")       # words only, so `a ? region : b` is never the Region setting


def _label_line(raw: str) -> tuple[str, str] | None:
    """(label, value) when the line is a heading or a `Title: Value` line, known or not. An unknown one, such as
    `Description:`, still ends the expression above it."""
    line = _DECORATION.sub("", raw.strip()).replace("**", "").replace("__", "").strip()
    head, colon, value = line.partition(":")
    if not colon or not _LABEL.match(head.strip()):
        return None
    return _key(head), value.strip().strip("`").strip()


_PAIR = re.compile(
    r"(?<![A-Za-z])("
    + "|".join(sorted({r"\s*".join(t.split()) for _, t in SETTINGS} | {n for n, _ in SETTINGS}, key=len)[::-1])
    + r")\s*\**\s*[:=]\s*\**\s*`?\s*([^|`*,;]+?)\s*`?\s*(?=[|`*,;]|$)",
    re.IGNORECASE,
)


def _pairs(raw: str) -> list[tuple[str, str]]:
    """Every setting written on a line, as (BRAIN name, value)."""
    return [(_SETTING_KEYS[_key(m[1])], m[2].strip()) for m in _PAIR.finditer(raw)]


def _settings_line(raw: str, pairs: list[tuple[str, str]], after_heading: bool) -> bool:
    """Whether a line sets settings rather than mentions one: it starts with a setting or the Simulation Settings
    heading, follows that heading, or holds several. A rationale saying `Decay: 10` in passing is none of those."""
    found = _label_line(raw)
    if len(pairs) > 1:
        return True
    if pairs and after_heading:
        # under the settings heading, unless the line is labelled as something else ("Rationale: ... Decay: 10")
        return found is None or found[0] in _SETTING_KEYS or found[0] == "simulationsettings"
    return bool(pairs) and found is not None and (found[0] in _SETTING_KEYS or found[0] == "simulationsettings")


def _table_row(raw: str) -> tuple[str, str] | None:
    """`| Region | USA |`: a two-cell table row naming a setting."""
    cells = [c.strip().replace("**", "").strip("`").strip() for c in raw.split("|") if c.strip()]
    if raw.strip().startswith("|") and len(cells) == 2 and _key(cells[0]) in _SETTING_KEYS:
        return _SETTING_KEYS[_key(cells[0])], cells[1]
    return None


def parse(text: str) -> list[dict[str, Any]]:
    """Each Alpha in an LLM's output, with the settings written for it, however it is laid out.

    An Alpha is an `Alpha Expression:` label with the expression on it or under it, or a fenced code block: one
    waiting under a label fills it, any other starts the next Alpha, and one holding whole Alphas is read as its own
    output. Settings are read wherever a line sets them, the first value of each kept, for the Alpha in hand until a
    heading or a rule closes it. Settings with no open Alpha wait for the next one and stand only if it writes none of
    its own. A code block nothing sets settings for is an example rather than an Alpha, and is left out."""
    drafts: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    reading = False
    fence: list[str] | None = None
    after_heading = False
    pending: dict[str, str] = {}

    def start(lines: list[str], *, fenced: bool) -> dict[str, Any]:
        draft = {"lines": lines, "given": {}, "held": dict(pending), "fenced": fenced}
        pending.clear()
        drafts.append(draft)
        return draft

    for raw in text.splitlines():
        if raw.strip().startswith("```"):
            if fence is None:
                fence = []
                continue
            labels = (_label_line(line) for line in fence)
            if any(found and found[0] == "alphaexpression" for found in labels):
                for whole in parse("\n".join(fence)):
                    current = start(whole["expression"].splitlines(), fenced=False)
                    current["given"] = whole["given"]
                current = None
            elif current is not None and reading and not current["lines"]:
                current["lines"] = fence
            else:
                current = start(fence, fenced=True)
            fence, reading = None, False
            continue
        if fence is not None:
            fence.append(raw.rstrip())
            continue
        found_label = _label_line(raw)
        if found_label is not None and found_label[0] == "alphaexpression":
            current = start([found_label[1]] if found_label[1] else [], fenced=False)
            reading, after_heading = True, False
            continue
        if raw.lstrip().startswith("#") and found_label is None:
            after_heading = "setting" in _key(raw)        # a heading opens the next Alpha, unless it heads settings
            if not after_heading:
                current = None
            reading = False
            continue
        pairs = _pairs(raw)
        row = _table_row(raw)
        if row is not None:
            pairs = [row]
        if row or _settings_line(raw, pairs, after_heading):
            given = current["given"] if current is not None else None
            fresh = given is not None and any(name not in given for name, _ in pairs)
            target = given if fresh and given is not None else pending
            for name, value in pairs:
                target.setdefault(name, value)
            reading = False
            continue
        if found_label is not None:
            reading = False
            after_heading = found_label[0] == "simulationsettings" and not found_label[1]
            continue
        if raw.strip() in {"---", "***", "___"}:
            current, reading, after_heading = None, False, False
        elif current is not None and reading:
            current["lines"].append(raw.rstrip().strip("`"))
    for d in drafts:
        d["given"] = d["given"] or d["held"]
    return [{"expression": "\n".join(d["lines"]).strip(), "given": d["given"]}
            for d in drafts if d["given"] or not d["fenced"]]


# ---------------------------------------------------------------- resolving against BRAIN's choices

def offered(name: str, settings: dict[str, Any]) -> list[Any] | None:
    """The values BRAIN offers for a setting where these settings run (None: no list for it here)."""
    from ..brain import brain
    opts = brain.options
    region = settings.get("region")
    if name == "region":
        return list(opts.regions) or None
    if region is None:
        return None
    if name == "delay":
        return opts.delays(region) or None
    if name == "universe":
        return opts.universes(region) or None
    if name == "neutralization":
        return opts.neutralizations(region) or None
    values = opts.choices(name, region)
    if not values and name == "maxPosition" and not opts.live:
        return ["OFF", "ON"] if str(region).upper() in MAX_POSITION_REGIONS else None
    return list(values) or None


def _choice(name: str, settings: dict[str, Any], value: str) -> tuple[Any, list[Any]] | None:
    """BRAIN's value for `value` by value or label, and every value offered; None when BRAIN offers no choices."""
    values = offered(name, settings)
    if not values:
        return None
    wanted = value.casefold()
    for v in values:
        if wanted in {str(v).casefold(), label(name, v).casefold()}:
            return v, []
    return None, [label(name, v) if label(name, v) == str(v) else f"{label(name, v)} ({v})" for v in values]


def resolve(given: dict[str, str]) -> dict[str, Any]:
    """BRAIN settings for one pasted Alpha: what it gave, BRAIN's defaults for what it left out, and what stops it."""
    settings: dict[str, Any] = {"instrumentType": INSTRUMENT_TYPE}
    defaulted: list[str] = []
    problems: list[str] = []
    for name, title in SETTINGS:
        value = str(given.get(name, "")).strip()
        if not value:
            if name in REQUIRED:
                problems.append(f"No {title} given.")
            elif name in ("decay", "truncation"):
                settings[name] = DEFAULTS[name]
                defaulted.append(title)
            elif name in DEFAULTS:
                values = offered(name, settings)
                # left out where BRAIN does not offer the setting at all: Max Position is not everywhere
                if values and DEFAULTS[name] in values:
                    settings[name] = DEFAULTS[name]
                    defaulted.append(title)
                elif values:
                    problems.append(f"No {title} given, and BRAIN's default is not offered here.")
            continue
        if name == "decay":
            if not value.isdigit() or int(value) > MAX_DECAY:
                problems.append(f"Decay {value!r} is not a whole number from 0 to {MAX_DECAY}.")
            else:
                settings[name] = int(value)
            continue
        if name == "truncation":
            try:
                number = float(value.rstrip("%")) / (100 if value.endswith("%") else 1)
            except ValueError:
                number = -1.0
            if not 0 <= number <= 1:
                problems.append(f"Truncation {value!r} is not a number from 0 to 1.")
            else:
                settings[name] = number
            continue
        if name == "testPeriod":
            if TEST_PERIOD_PATTERN.fullmatch(value.upper()):
                settings[name] = value.upper()
            else:
                problems.append(f"Test Period {value!r} is not like P2Y0M0D.")
            continue
        found = _choice(name, settings, value)
        if found is None:
            # its parent was refused, which is already said; or BRAIN has no list for it
            if name in REQUIRED and not problems:
                problems.append(f"BRAIN offers no {title} choices here.")
            continue
        chosen, choices = found
        if chosen is None:
            where = f" in {settings.get('region')} delay {settings.get('delay')}"
            problems.append(f"{title} {value!r} is not offered" + (where if name not in ("region", "delay") else "")
                            + f". BRAIN offers {', '.join(map(str, choices))}.")
        else:
            settings[name] = chosen
    if settings.get("maxTrade") == "ON" and settings.get("maxPosition") == "ON":
        problems.append("Max Trade and Max Position cannot both be On.")
    return {"settings": settings, "defaulted": defaulted, "problems": problems}


def fields_of(expression: str) -> list[str]:
    """Data fields read, grouping fields aside; none if it can't be read (BRAIN will say when it runs)."""
    from .common import fields_in_order
    try:
        return fields_in_order(expression)
    except ValueError:
        return []


def batch_key(settings: dict[str, Any]) -> tuple:
    """What every child of one multi-simulation shares, its largest batch and how slow it is."""
    region = str(settings.get("region") or "")
    return (str(settings.get("instrumentType") or INSTRUMENT_TYPE), region, int(settings.get("delay") or 0),
            str(settings.get("language") or LANGUAGE))


def max_batch(key: tuple) -> int:
    return 1 if key[1] == REGION_AGNOSTIC else MULTI_SIMULATION_SIZE


def slowness(key: tuple) -> int:
    return RA_CHILDREN if key[1] == REGION_AGNOSTIC else 2 if key[1] == "GLB" else 1


def chunks(requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in requests:
        groups.setdefault(batch_key(r["settings"]), []).append(r)
    return [members[i:i + max_batch(key)] for key, members in groups.items()
            for i in range(0, len(members), max_batch(key))]


def ordered(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The order to send them in for the soonest finish: batched by BRAIN's batch key so every multi-simulation is as
    full as it can be, then the slowest first (a region-agnostic or GLB batch started last would leave every other
    core idle while it finishes); part-full batches last, as the Settings Sampler sends them."""
    def full(chunk: list[dict[str, Any]]) -> bool:
        return len(chunk) == max_batch(batch_key(chunk[0]["settings"]))
    batched = sorted(chunks(requests), key=lambda c: (not full(c), -slowness(batch_key(c[0]["settings"])), -len(c)))
    return [r for chunk in batched for r in chunk]


async def plan(text: str) -> dict[str, Any]:
    """Every Alpha in the pasted text, resolved, with the simulations it would queue. Simulates nothing."""
    alphas: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    requests: list[dict[str, Any]] = []
    for number, draft in enumerate(parse(text or ""), start=1):
        found = resolve(draft["given"])
        problems = found["problems"]
        if not draft["expression"]:
            problems.insert(0, "No expression under Alpha Expression.")
        duplicate = None
        if not problems:
            identity = json.dumps([draft["expression"], found["settings"]], sort_keys=True, default=str)
            duplicate = seen.setdefault(identity, number)
            if duplicate == number:
                duplicate = None
                requests.append({"expression": draft["expression"], "settings": found["settings"],
                                 "params": {"number": number}})
        alphas.append({"number": number, "expression": draft["expression"], "settings": found["settings"],
                       "labels": labels_of(found["settings"]), "defaulted": found["defaulted"],
                       "problems": problems, "duplicate_of": duplicate, "categories": []})
    problems: list[str] = []
    if not alphas:
        problems.append('No Alpha found. Each one starts with a line reading "Alpha Expression:".')
    elif not requests:
        problems.append("None of the pasted Alphas can run as written.")
    elif len(requests) > MAX_SIMULATIONS:
        problems.append(f"At most {MAX_SIMULATIONS:,} Alphas can run in one task.")
    # the Dataset Categories each reads (over 2 counts toward no Genius pyramid), read where each Alpha runs
    for region in {str(a["settings"].get("region")) for a in alphas}:
        mine = [a for a in alphas if str(a["settings"].get("region")) == region]
        known = await described({f for a in mine for f in fields_of(a["expression"])}, region)
        for a in mine:
            a["categories"] = sorted({str(known[f].category) for f in fields_of(a["expression"])
                                      if f in known and known[f].category})
    queued = ordered(requests)
    return {"alphas": alphas, "requests": queued, "simulations": len(queued),
            "cost": sum(RA_CHILDREN if r["settings"].get("region") == REGION_AGNOSTIC else 1 for r in queued),
            "batches": len(chunks(queued)), "problems": problems}


class PasteLab(Lab):
    """Pasted Alphas (or a Flip Sign), written up front in the order that finishes soonest."""

    name = NAME
    title = "Paste Alphas"

    async def plan(self, task: LabTask) -> dict[str, Any]:
        params = task.params or {}
        if params.get("flip_of"):
            return await flip_plan(params["flip_of"])
        found = await plan(params.get("text") or "")
        if found["problems"]:
            raise ToolError(found["problems"][0])
        return {"alphas": len(found["alphas"]), "requests": found["requests"]}

    async def initial(self, task: LabTask) -> list[TrialSpec]:
        return await written_up_front(task)

    def done_early(self, task: LabTask, ctx: RefillContext) -> str | None:
        return all_run(ctx)

    async def preview(self, params: dict[str, Any]) -> dict[str, Any]:
        found = await plan(params.get("text") or "")
        return {**found, "sample": found["requests"][:8]}


async def flip_plan(alpha_id: str) -> dict[str, Any]:
    from ..fastexpr import FastExprSyntaxError
    from ..labs.fastexpr import negated
    alpha_id = alpha_id_of(alpha_id)
    body = await alpha_body(alpha_id)
    code, settings = code_of(body), body.get("settings")
    if not code or not isinstance(settings, dict) or not settings:
        raise ToolError(f"{alpha_id} has no regular expression to flip.")
    try:
        flipped = negated(code)
    except (FastExprSyntaxError, ValueError) as exc:
        raise ToolError(f"{alpha_id}'s expression could not be read, so it was not flipped: {exc}") from exc
    return {"alphas": 1, "flip_of": alpha_id, "requests": [
        {"expression": flipped, "settings": {**settings, "simulationMode": FULL_MODE}, "params": {"flip_of": alpha_id}}]}


async def flip_sign(alpha_id: str) -> int:
    """One task simulating the Alpha's expression with its sign flipped, at its own settings, in Full mode (so a
    winner needs no second run before it can be submitted). Only the last statement is negated. The task is started
    at once (one simulation); returns its id."""
    from ..labs import runner as runner_mod
    alpha_id = alpha_id_of(alpha_id)
    return await runner_mod.runner.create(NAME, {"flip_of": alpha_id}, cores=1, max_trials=1,
                                          name=f"Flip Sign · {alpha_id}", run=True)


async def add_paste_task(text: str, *, cores: int = 8, run: bool = False) -> int:
    from ..labs import runner as runner_mod
    found = await plan(text)
    if found["problems"]:
        raise ToolError(found["problems"][0])
    return await runner_mod.runner.create(NAME, {"text": text}, cores=cores, max_trials=len(found["requests"]),
                                          name=f"Paste Alphas · {len(found['requests'])} Alphas", run=run)


async def copy_results(task_id: int) -> dict[str, Any]:
    """The task's results as Copy Alphas writes them, each with its checks, then the simulations that failed with
    BRAIN's reason. No glossaries: the LLM already has them. Alphas whose PnL is not stored are queued for download
    (their K-Ratio and correlations read "—" until it is)."""
    from .copy_alphas import markdown
    async with db.session() as s:
        task = await s.get(LabTask, task_id)
        if task is None or task.lab != NAME:
            raise ToolError("No Paste Alphas task with that id.")
        trials = list(await s.scalars(select(LabTrial).where(LabTrial.task_id == task_id).order_by(LabTrial.number)))
        from ..db.models import Alpha
        ids = [t.alpha_id for t in trials if t.alpha_id]
        alphas = {a.id: a for a in await s.scalars(select(Alpha).where(Alpha.id.in_(ids)))} if ids else {}
    found, failures, waiting = [], [], 0
    for t in trials:
        if t.alpha_id and t.alpha_id in alphas:
            found.append(alphas[t.alpha_id])
        elif t.state == "FAIL":
            failures.append({"expression": t.expression, "settings": t.settings, "error": t.message})
        elif t.state in ("PARKED", "RUNNING"):
            waiting += 1
    from ..engine import series
    lacking = [a.id for a in found if a.region != REGION_AGNOSTIC and a.simulation_mode != "QUICK"]
    lacking = await series.missing(lacking) if lacking else []
    if lacking:
        try:
            from ..jobs.queue import jobs
            await jobs.submit("fetch_series", f"Daily PnL for {len(lacking)} pasted alpha(s)",
                              {"alpha_ids": lacking, "turnover": True})
        except Exception:          # a download that cannot be queued leaves those figures as "—"
            pass
    text = await markdown(found, glossary=False, failures=failures)
    return {"text": text, "alphas": len(found), "failed": len(failures), "waiting": waiting,
            "downloading": len(lacking)}
