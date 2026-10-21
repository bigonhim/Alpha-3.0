"""Record BRAIN's real responses as test fixtures (opt-in): set ALPHAGO_RECORD_BRAIN=<folder> before starting.

Only read-only GET responses are saved, a few per endpoint shape, with personal data removed (user ids, emails,
names, tokens and anything that looks like a credential); alpha ids are kept so a fixture set stays consistent.
The tests in tests/fakebrain replay these files through the app's parsers, so when BRAIN changes a response shape,
re-recording plus `pytest` shows what broke.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ENV = "ALPHAGO_RECORD_BRAIN"
PER_SHAPE = 3
SECRET_KEYS = {"token", "password", "email", "firstname", "lastname", "name", "username", "user", "phone",
               "inquiry", "authorization", "cookie"}
_counts: dict[str, int] = {}


def shape(path: str) -> str:
    """Endpoint shape: ids replaced by '{id}' (e.g. /alphas/{id}/recordsets/pnl)."""
    p = path.split("?")[0].rstrip("/") or "/"
    parts = []
    for seg in p.split("/"):
        parts.append("{id}" if seg and (any(ch.isdigit() for ch in seg) and len(seg) >= 5) else seg)
    return "/".join(parts)


def sanitize(value: Any, key: str = "") -> Any:
    if key.lower() in SECRET_KEYS and not isinstance(value, (list, dict)):
        return "REDACTED"
    if isinstance(value, dict):
        return {k: ("REDACTED" if k.lower() in SECRET_KEYS and isinstance(v, (str, int)) else sanitize(v, k))
                for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, key) for v in value]
    if isinstance(value, str) and "@" in value and "." in value.split("@")[-1]:
        return "REDACTED@example.com"
    return value


def observe(method: str, url: str, status: int, body: Any, final: bool, response: Any = None) -> None:
    folder = os.environ.get(ENV)
    if not folder or method.upper() != "GET" or response is None or not final:
        return
    path = url if url.startswith("/") else "/" + url.split("://", 1)[-1].split("/", 1)[-1]
    if path.split("?")[0].rstrip("/").endswith("/authentication"):
        return
    key = shape(path)
    if _counts.get(key, 0) >= PER_SHAPE:
        return
    try:
        payload = response.json() if response.content else None
    except ValueError:
        return
    _counts[key] = _counts.get(key, 0) + 1
    record = {"method": "GET", "shape": key, "path": path.split("?")[0], "status": status,
              "retry_after": response.headers.get("Retry-After"), "json": sanitize(payload)}
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_") or "root"
    (out / f"{name}__{_counts[key]}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def install() -> None:
    if not os.environ.get(ENV):
        return
    from . import brain
    if observe not in brain.client.observers:
        brain.client.observers.append(observe)
        log.warning("Recording sanitised BRAIN GET responses to %s", os.environ[ENV])
