"""Systematic Predictions Challenge (SPC) on BRAIN: submit a prompt, and read-only discovery of the endpoint.

BRAIN publishes no SPC endpoint, so nothing here is hard-coded. The path, method and body come from `settings.spc`
(copy them from the browser's network tab when you submit the form by hand once, or paste the request copied with
"Copy as cURL" / "Copy as fetch" into SPC Lab's Import box). Until a path is set, `build_request` refuses and nothing
is sent.

Safety: the path must be relative to the BRAIN API (the session cookie never goes to another host), a POST is never
retried automatically (a retry could submit twice), and a response that is not clearly 2xx is never reported as
accepted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from ..config import SPCSettings
from .client import BrainClient
from .errors import BrainError, describe_error_body

PLACEHOLDERS = ("prompt", "name", "model", "version", "frequency", "weight", "competition")
METHODS = ("POST", "PUT", "PATCH")
_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_WHOLE = re.compile(r"\{(\w+)(:num)?\}")


@dataclass
class SPCRequest:
    method: str
    path: str
    body: Any


@dataclass
class SPCSubmitResult:
    status: str                       # accepted | rejected | throttled | error | unknown
    message: str
    http_status: int | None = None
    body: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def _fill(node: Any, values: dict[str, str]) -> Any:
    """Substitute {placeholders} in every string value, in one pass (a value is never parsed as JSON, and text inside
    a substituted value, such as "{model}" written in the prompt, is never substituted again). A string that is
    exactly "{key:num}" becomes a number."""
    if isinstance(node, str):
        whole = _WHOLE.fullmatch(node)
        if whole and whole.group(2) and whole.group(1) in values:
            try:
                f = float(values[whole.group(1)])
                return int(f) if f.is_integer() else f
            except (TypeError, ValueError):
                return values[whole.group(1)]
        return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), node)
    if isinstance(node, list):
        return [_fill(x, values) for x in node]
    if isinstance(node, dict):
        return {k: _fill(v, values) for k, v in node.items()}
    return node


def validate_endpoint(rules: SPCSettings) -> list[str]:
    """Why the submit endpoint cannot be used (empty list = usable)."""
    problems: list[str] = []
    path = (rules.submit_path or "").strip()
    if not path:
        problems.append("No submit path is set. Copy it from the browser's network tab when you submit the SPC form "
                        "by hand (Submit settings below).")
    elif not path.startswith("/") or path.startswith("//") or "://" in path:
        problems.append("The submit path must start with a single '/' and be relative to the BRAIN API "
                        "(no https://… address).")
    elif "{competition}" in path and not (rules.competition_id or "").strip():
        problems.append("The submit path uses {competition}, but no competition id is set.")
    if (rules.submit_method or "").upper() not in METHODS:
        problems.append(f"The submit method must be one of {', '.join(METHODS)}.")
    try:
        body = json.loads(rules.submit_body or "")
        if not isinstance(body, dict):
            problems.append("The submit body must be a JSON object.")
        elif "{prompt}" not in json.dumps(body):
            problems.append("The submit body has no {prompt} placeholder, so the prompt would not be sent.")
    except json.JSONDecodeError as exc:
        problems.append(f"The submit body is not valid JSON: {exc}")
    return problems


def resolved_path(rules: SPCSettings) -> str:
    return (rules.submit_path or "").strip().replace("{competition}", (rules.competition_id or "").strip())


def build_request(rules: SPCSettings, *, name: str, prompt: str, form: dict[str, str] | None = None) -> SPCRequest:
    problems = validate_endpoint(rules)
    if problems:
        raise ValueError(problems[0])
    form = form or {}
    values = {"prompt": prompt, "name": name,
              "model": str(form.get("model", rules.model)), "version": str(form.get("version", rules.version)),
              "frequency": str(form.get("frequency", rules.frequency)), "weight": str(form.get("weight", rules.weight)),
              "competition": (rules.competition_id or "").strip()}
    return SPCRequest(rules.submit_method.upper(), resolved_path(rules), _fill(json.loads(rules.submit_body), values))


# ---------------------------------------------------------------- importing a request copied from the browser

# key names in a captured request body -> the placeholder that fills them
_CAPTURE_KEYS = (("prompt", "prompt"), ("instruction", "prompt"), ("name", "name"), ("title", "name"),
                 ("version", "version"), ("model", "model"), ("frequency", "frequency"), ("weight", "weight"))


def _shell_args(text: str) -> list[str]:
    import shlex
    cleaned = text.replace("\\\r\n", " ").replace("\\\n", " ").replace("^\r\n", " ").replace("^\n", " ")
    if '^"' in cleaned:                          # Chrome's "Copy as cURL (cmd)": ^ escapes everything
        cleaned = re.sub(r"\^(.)", r"\1", cleaned)
    try:
        return shlex.split(cleaned, posix=True)
    except ValueError:
        return cleaned.split()


def parse_captured_request(text: str) -> dict[str, Any]:
    """Turn a request copied from the browser (DevTools → Network → right-click → Copy as cURL (bash or cmd), or Copy
    as fetch) into SPC endpoint settings: {path, method, competition_id, submit_body, mapped, unmapped, host}.

    String values whose key names a form field (prompt, name, model, version, frequency, weight) become placeholders;
    numbers become numeric placeholders; everything else is kept as captured. Cookies and auth headers are dropped:
    the app signs in itself."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Paste the request you copied from the browser.")
    url, method, body = "", "", None
    if raw.lower().startswith(("fetch(", "await fetch(")):
        m = re.search(r"fetch\(\s*[\"']([^\"']+)[\"']", raw)
        url = m.group(1) if m else ""
        mm = re.search(r'"method"\s*:\s*"(\w+)"', raw)
        method = mm.group(1) if mm else ""
        mb = re.search(r'"body"\s*:\s*("(?:[^"\\]|\\.)*")', raw, re.S)
        if mb:
            body = json.loads(mb.group(1))
    else:
        args = _shell_args(raw)
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("-X", "--request") and i + 1 < len(args):
                method = args[i + 1]
                i += 1
            elif a in ("--data-raw", "--data", "-d", "--data-binary", "--data-ascii") and i + 1 < len(args):
                body = args[i + 1]
                i += 1
            elif a in ("-H", "--header", "-b", "--cookie", "-u", "--user") and i + 1 < len(args):
                i += 1                           # headers and cookies are never kept
            elif a.startswith("http") and not url:
                url = a
            i += 1
    if not url:
        raise ValueError("No URL found. Copy the request with 'Copy as cURL' or 'Copy as fetch'.")
    parsed = urlparse(url)
    path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    comp = ""
    mc = re.search(r"/competitions/([^/?]+)", path)
    if mc:
        comp = mc.group(1)
        path = path.replace(f"/competitions/{comp}", "/competitions/{competition}", 1)
    try:
        data = json.loads(body) if isinstance(body, str) else (body or {})
    except json.JSONDecodeError as exc:
        raise ValueError(f"The request body is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("The request body must be a JSON object.")
    mapped: dict[str, str] = {}
    unmapped: list[str] = []

    def walk(node: Any, key: str = "") -> Any:
        if isinstance(node, dict):
            return {k: walk(v, k) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, key) for v in node]
        low = key.lower()
        target = next((ph for nm, ph in _CAPTURE_KEYS if nm in low), None)
        if target and isinstance(node, (str, int, float)) and not isinstance(node, bool):
            mapped[key] = target
            return f"{{{target}}}" if isinstance(node, str) else f"{{{target}:num}}"
        if comp and node == comp:
            mapped[key] = "competition"
            return "{competition}"
        if key:
            unmapped.append(key)
        return node

    template = walk(data)
    if "prompt" not in mapped.values():
        raise ValueError("The captured body has no prompt field. Copy the request sent when you pressed Submit on "
                         "BRAIN's SPC page.")
    return {"path": path, "method": (method or ("POST" if body is not None else "GET")).upper(),
            "competition_id": comp, "submit_body": json.dumps(template, ensure_ascii=False),
            "mapped": mapped, "unmapped": unmapped, "host": parsed.netloc}


class SPCAPI:
    def __init__(self, client: BrainClient) -> None:
        self.client = client

    async def submit(self, req: SPCRequest) -> SPCSubmitResult:
        try:
            # no automatic retries: a repeated POST could submit the prompt twice
            response = await self.client.request(req.method, req.path, json=req.body, retries=0, retry_statuses=())
        except BrainError as exc:
            if exc.status is None:
                return SPCSubmitResult("unknown", f"{exc.message} The request may or may not have reached BRAIN: "
                                                  "check the SPC page on BRAIN before trying again.")
            return SPCSubmitResult("error", exc.message, exc.status, exc.body)
        status = response.status_code
        body = self.client._json(response)
        text = describe_error_body(body) if status >= 400 else ""
        if 200 <= status < 300:
            return SPCSubmitResult("accepted", f"BRAIN accepted the submission (HTTP {status}).", status, body)
        if status in (301, 302, 303, 307, 308):
            return SPCSubmitResult("unknown", f"BRAIN answered HTTP {status} (a redirect to "
                                              f"{response.headers.get('Location', '?')}); check BRAIN before retrying.",
                                   status, body)
        if status == 429:
            return SPCSubmitResult("throttled", text or "Throttled by BRAIN: try again in a few minutes.", status, body)
        if status >= 500:
            return SPCSubmitResult("error", text or f"BRAIN error HTTP {status}; check BRAIN before retrying.",
                                   status, body)
        return SPCSubmitResult("rejected", text or f"BRAIN rejected the submission (HTTP {status}).", status, body)

    async def discover(self) -> dict[str, Any]:
        """Read-only GETs that may show where the SPC lives (competitions the account is in). Never writes."""
        out: dict[str, Any] = {}
        for label, path in (("competitions", "/users/self/competitions"),):
            try:
                out[label] = await self.client.get_json(path)
            except BrainError as exc:
                out[label] = {"error": exc.message}
        return out

    async def competitions(self) -> list[dict[str, Any]]:
        """The competitions your account is enrolled in, flattened (to find the SPC id for {competition})."""
        body = await self.client.get_json("/users/self/competitions")
        items = body if isinstance(body, list) else (body or {}).get("results") or []
        return [{"id": c.get("id"), "name": c.get("name") or c.get("title"), "status": c.get("status"),
                 "start": c.get("startDate") or c.get("dateStart"), "end": c.get("endDate") or c.get("dateEnd")}
                for c in items if isinstance(c, dict)]
