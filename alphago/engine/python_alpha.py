"""Python alphas: alphas written in Python instead of FASTEXPR.

BRAIN publishes no public specification of Python alphas (code shape, data access, libraries). So the app:
  - detects support from OPTIONS /simulations: `settings.language` must offer a value containing "PYTHON" (or the
    value you set in `python_alpha.language`). Without it nothing is simulated, and the page says so;
  - checks code locally before it costs a simulation: it must parse as Python, stay under the size limit, and use no
    file, network, process or dynamic-code access (BRAIN would reject those anyway);
  - writes code with the AI against the reference you paste (BRAIN's docs or an example from the platform), and fixes
    code that fails the local check, up to `python_alpha.ai_rounds` times. Without a reference the AI is told to
    follow your starter template exactly, and the result is marked "unverified";
  - simulates through the normal BRAIN simulation endpoint with settings.language set to the Python value, honouring
    the daily quota reserve, and stores the result like any other alpha (Alphas page: check, repair, submit).
Nothing here submits: Python alphas are submitted from the alpha details window like any other alpha.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ..brain import brain
from ..brain.simulations import regular_payload
from ..config import settings
from ..db import db
from ..events import bus
from ..jobs.queue import JobContext, jobs
from ..llm import llm
from .ingest import upsert_alpha

log = logging.getLogger(__name__)

HISTORY_KEY = "python_alpha:history"
BLOCKED_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib", "http", "ctypes",
                   "multiprocessing", "threading", "asyncio", "pickle", "importlib", "builtins", "io", "tempfile"}
BLOCKED_CALLS = {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint", "globals", "locals", "vars"}


# ---------------------------------------------------------------- support

def languages() -> list[str]:
    return [str(v) for v in brain.options.choices("language")]


def python_language() -> str | None:
    """The settings.language value BRAIN accepts for Python alphas, or None when the account does not offer one."""
    forced = (settings.python_alpha.language or "").strip()
    offered = languages()
    if forced:
        return forced if (forced in offered or not brain.options.live) else None
    return next((v for v in offered if "PYTHON" in v.upper()), None)


def support() -> dict[str, Any]:
    lang = python_language()
    live = bool(brain.options.live)
    if lang:
        why = f"BRAIN accepts language {lang} for your account."
    elif not live:
        why = ("The allowed settings have not been read from BRAIN yet: sign in, then Sync with BRAIN → Refresh from "
               "BRAIN.")
    else:
        why = (f"BRAIN offers only {', '.join(languages()) or 'no'} language(s) to your account, so Python alphas "
               "cannot be simulated yet. Check BRAIN's announcements; when it is enabled, Refresh from BRAIN.")
    return {"supported": bool(lang) and live, "language": lang, "languages": languages(), "live": live, "why": why}


# ---------------------------------------------------------------- local check

@dataclass
class CodeCheck:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_code(code: str) -> CodeCheck:
    text = code or ""
    res = CodeCheck(True)
    if not text.strip():
        return CodeCheck(False, ["The code is empty."])
    if len(text) > settings.python_alpha.max_chars:
        res.errors.append(f"{len(text):,} characters; the limit is {settings.python_alpha.max_chars:,}.")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return CodeCheck(False, [f"Syntax error on line {exc.lineno}: {exc.msg}"])
    defs = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                if n.split(".")[0] in BLOCKED_MODULES:
                    res.errors.append(f"Line {node.lineno}: import of {n} is not allowed in an alpha.")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
            res.errors.append(f"Line {node.lineno}: {node.func.id}() is not allowed in an alpha.")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            res.errors.append(f"Line {node.lineno}: dunder attribute {node.attr} is not allowed.")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs += 1
        elif isinstance(node, (ast.While,)):
            res.warnings.append(f"Line {node.lineno}: a while loop may run slowly on BRAIN; prefer vectorised code.")
    if not defs:
        res.warnings.append("No function is defined; BRAIN's Python alphas usually provide one entry function.")
    res.ok = not res.errors
    return res


# ---------------------------------------------------------------- AI writing

class PythonAlphaDraft(BaseModel):
    name: str = Field(description="Short name, 3 to 6 words")
    hypothesis: str = Field(description="The economic idea in 2 to 3 sentences")
    code: str = Field(description="The complete Python alpha code, no markdown fences")


WRITE_TEMPLATE = """\
You write a WorldQuant BRAIN alpha in Python (not FASTEXPR). The alpha produces a cross-sectional signal: for each day,
one weight per stock; BRAIN neutralizes and scales it.

{reference_block}
Idea: {idea}
{fastexpr_block}Region {region}, delay {delay}, universe {universe}.

Rules:
- Use ONLY the entry point, data access and libraries shown in the reference / template. Do not invent APIs.
- No file, network, process or dynamic-code access (no os, sys, subprocess, open, eval, exec, __import__).
- Vectorised, deterministic, no look-ahead (only data available on or before each day).
- Plain Python source in "code", no markdown fences.
{revision}Return JSON matching the schema."""


def _reference_block() -> str:
    cfg = settings.python_alpha
    parts = []
    if cfg.reference.strip():
        parts.append("REFERENCE (BRAIN's Python Alpha docs or an example, follow it exactly):\n" + cfg.reference.strip())
    if cfg.starter.strip():
        parts.append("TEMPLATE (keep its structure and entry point):\n" + cfg.starter.strip())
    if not parts:
        parts.append("No reference was provided: keep the code minimal, define one function `alpha(data)` that returns "
                     "the signal, and say in the hypothesis that the data access must be checked against BRAIN's docs.")
    return "\n\n".join(parts) + "\n"


async def write_code(idea: str, *, fastexpr: str = "", region: str = "USA", delay: int = 1,
                     universe: str = "TOP3000") -> dict[str, Any]:
    """AI draft, local check, AI fix on failure. Returns {name, hypothesis, code, check, rounds, verified}."""
    if not llm.configure():
        raise RuntimeError("Set up an AI provider on the LLM Integration page first.")
    cfg = settings.python_alpha
    revision, draft, chk = "", None, CodeCheck(False)
    rounds = 0
    for _ in range(max(0, cfg.ai_rounds) + 1):
        rounds += 1
        prompt = WRITE_TEMPLATE.format(
            reference_block=_reference_block(), idea=idea.strip() or "a robust cross-sectional signal",
            fastexpr_block=(f"Translate this FASTEXPR alpha faithfully: {fastexpr}\n" if fastexpr.strip() else ""),
            region=region, delay=delay, universe=universe, revision=revision)
        draft = await llm.generate_json(prompt, PythonAlphaDraft, role="refine", purpose="python_alpha",
                                        thinking="medium")
        code = _strip_fences(draft.code)
        chk = check_code(code)
        draft.code = code
        if chk.ok:
            break
        revision = ("YOUR PREVIOUS CODE FAILED THE CHECK. Fix exactly these problems:\n"
                    + "\n".join(f"- {e}" for e in chk.errors) + f"\n\nPrevious code:\n{code}\n\n")
    assert draft is not None
    return {"name": draft.name.strip()[:120], "hypothesis": draft.hypothesis.strip(), "code": draft.code,
            "check": {"ok": chk.ok, "errors": chk.errors, "warnings": chk.warnings}, "rounds": rounds,
            "verified": bool(cfg.reference.strip() or cfg.starter.strip())}


def _strip_fences(code: str) -> str:
    t = (code or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip() + "\n"


# ---------------------------------------------------------------- simulation

def sim_settings(base: dict[str, Any] | None = None) -> dict[str, Any]:
    sup = support()
    if not sup["supported"]:
        raise RuntimeError(sup["why"])
    return {**(base or {}), "language": sup["language"]}


async def simulate(code: str, base_settings: dict[str, Any], *, name: str = "", hypothesis: str = "") -> dict[str, Any]:
    """One simulation of a Python alpha. Refuses past the quota reserve and when BRAIN does not offer Python."""
    chk = check_code(code)
    if not chk.ok:
        raise ValueError("Not simulated: " + "; ".join(chk.errors[:3]))
    s = sim_settings(base_settings)
    if not brain.client.quota.can_spend(1, settings.brain.quota_reserve):
        raise RuntimeError("Daily simulation quota reserve reached: not simulated.")
    outcomes = await brain.sims.run([regular_payload(code, s)])
    out = outcomes[0]
    entry = {"name": name, "hypothesis": hypothesis, "status": out.status, "alpha_id": out.alpha_id,
             "message": out.message, "settings": {k: s.get(k) for k in ("region", "delay", "universe",
                                                                         "neutralization", "decay", "language")}}
    if out.alpha_id:
        data = await brain.alphas.get(out.alpha_id)
        alpha = await upsert_alpha(data, origin="app")
        entry.update(verdict=alpha.verdict, sharpe=alpha.sharpe, fitness=alpha.fitness, turnover=alpha.turnover)
        bus.publish("simulate", f"Python alpha simulated: {out.alpha_id} ({alpha.verdict}, Sharpe {alpha.sharpe}).",
                    "success")
    else:
        bus.publish("simulate", f"Python alpha simulation {out.status}: {out.message or ''}", "warning")
    history = await db.kv_get(HISTORY_KEY, []) or []
    await db.kv_set(HISTORY_KEY, [*history[-199:], entry])
    return entry


# ---------------------------------------------------------------- jobs (automation)

@jobs.register("python_alpha_simulate")
async def _job_simulate(ctx: JobContext, params: dict) -> str:
    e = await simulate(str(params["code"]), dict(params.get("settings") or {}), name=str(params.get("name") or ""),
                       hypothesis=str(params.get("hypothesis") or ""))
    return f"{e['status']}: {e.get('alpha_id') or e.get('message') or ''}"


@jobs.register("python_alpha_batch")
async def _job_batch(ctx: JobContext, params: dict) -> str:
    """Write `n` Python alphas with the AI (one per idea), check each locally and simulate the ones that pass."""
    ideas = [str(i) for i in (params.get("ideas") or []) if str(i).strip()] or ["a robust cross-sectional signal"]
    n = max(1, min(20, int(params.get("n") or len(ideas))))
    base = dict(params.get("settings") or {})
    sim_settings(base)                                   # fail fast when BRAIN does not offer Python
    done, good = 0, 0
    for i in range(n):
        idea = ideas[i % len(ideas)]
        await ctx.update(i / n, f"Writing Python alpha {i + 1}/{n}: {idea[:60]}")
        try:
            d = await write_code(idea, region=base.get("region", "USA"), delay=int(base.get("delay", 1)),
                                 universe=base.get("universe", "TOP3000"))
            if not d["check"]["ok"]:
                continue
            e = await simulate(d["code"], base, name=d["name"], hypothesis=d["hypothesis"])
            done += 1
            good += e.get("verdict") in ("promising", "strong", "ready")
        except Exception as exc:
            log.warning("Python alpha %d failed: %s", i + 1, exc)
            bus.publish("simulate", f"Python alpha {i + 1}: {getattr(exc, 'message', None) or exc}", "warning")
            if "quota" in str(exc).lower():
                break
    await ctx.update(1.0)
    return f"Simulated {done}/{n} Python alpha(s); {good} promising or better."
