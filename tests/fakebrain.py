"""A stateful fake of the WorldQuant BRAIN API for tests (respx routes on the real HTTP client).

It answers like BRAIN does in the shapes the app relies on — the same shapes `alphago.brain.recorder` saves from the
real API into tests/fixtures/brain — so the whole pipeline (sign-in, multi-simulations with Retry-After progress,
alpha details, submission check, PnL record sets, correlations, listings, quota headers, throttling) runs without
touching a real account. `score(expression, settings) -> (sharpe, fitness, turnover)` decides each alpha's result.
"""

from __future__ import annotations

import itertools
import json
import zlib
from typing import Any, Callable

import httpx

API = "https://api.worldquantbrain.com"
AUTH = {"user": {"id": "U-FAKE"}, "token": {"expiry": 14400.0}, "permissions": ["CONSULTANT"]}
LIMITS = {"LOW_SHARPE": 1.25, "LOW_FITNESS": 1.0, "LOW_TURNOVER": 0.01, "HIGH_TURNOVER": 0.7}

Score = Callable[[str, dict], tuple[float, float, float]]


def is_checks(sharpe: float, fitness: float, turnover: float) -> list[dict[str, Any]]:
    return [
        {"name": "LOW_SHARPE", "result": "PASS" if sharpe >= 1.25 else "FAIL", "limit": 1.25, "value": sharpe},
        {"name": "LOW_FITNESS", "result": "PASS" if fitness >= 1.0 else "FAIL", "limit": 1.0, "value": fitness},
        {"name": "LOW_TURNOVER", "result": "PASS" if turnover >= 0.01 else "FAIL", "limit": 0.01,
         "value": turnover},
        {"name": "HIGH_TURNOVER", "result": "PASS" if turnover <= 0.7 else "FAIL", "limit": 0.7, "value": turnover},
        {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
        {"name": "SELF_CORRELATION", "result": "PENDING"},
        {"name": "MATCHES_COMPETITION", "result": "PASS"},
    ]


class FakeBrain:
    def __init__(self, score: Score, *, self_corr: float = 0.3, quota: int = 5000, polls: int = 1,
                 throttle_posts: int = 0) -> None:
        self.score = score
        self.self_corr = self_corr
        self.remaining = quota
        self.limit = quota
        self.polls = polls                       # progress answers before a simulation completes
        self.throttle_posts = throttle_posts     # first N POST /simulations answer 429 (concurrency)
        self.ids = itertools.count(1)
        self.sims: dict[str, dict[str, Any]] = {}
        self.alphas: dict[str, dict[str, Any]] = {}
        self.checked: list[str] = []
        self.submitted: list[str] = []
        self.posted = 0

    # ---------------------------------------------------------- helpers
    def _quota_headers(self) -> dict[str, str]:
        return {"X-Ratelimit-Limit": str(self.limit), "X-Ratelimit-Remaining": str(self.remaining)}

    def _new_alpha(self, item: dict[str, Any]) -> str:
        aid = f"FB{next(self.ids):05d}"
        code = item["regular"] if isinstance(item.get("regular"), str) else (item.get("regular") or {}).get("code")
        sharpe, fitness, turnover = self.score(code, item.get("settings") or {})
        self.alphas[aid] = {
            "id": aid, "type": item.get("type", "REGULAR"), "settings": item.get("settings") or {},
            "regular": {"code": code, "description": None}, "status": "UNSUBMITTED", "stage": "IS",
            "dateCreated": "2026-10-20T10:00:00-04:00", "name": None, "tags": [], "color": None,
            "is": {"sharpe": sharpe, "fitness": fitness, "turnover": turnover, "returns": 0.08, "drawdown": 0.06,
                   "margin": 0.0007, "longCount": 1400, "shortCount": 1380,
                   "checks": is_checks(sharpe, fitness, turnover)},
        }
        return aid

    # ---------------------------------------------------------- endpoints
    def post_sim(self, request: httpx.Request) -> httpx.Response:
        if self.throttle_posts > 0:
            self.throttle_posts -= 1
            return httpx.Response(429, json={"detail": "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"})
        body = json.loads(request.content)
        items = body if isinstance(body, list) else [body]
        self.posted += len(items)
        self.remaining = max(0, self.remaining - len(items))
        sid = f"S{next(self.ids)}"
        self.sims[sid] = {"items": items, "left": self.polls, "children": []}
        return httpx.Response(201, headers={"Location": f"{API}/simulations/{sid}", **self._quota_headers()})

    def get_sim(self, request: httpx.Request) -> httpx.Response:
        sid = request.url.path.rsplit("/", 1)[-1]
        sim = self.sims.get(sid)
        if sim is None:                          # a child of a multi-simulation
            for parent in self.sims.values():
                for child_id, aid in parent["children"]:
                    if child_id == sid:
                        return httpx.Response(200, json={"id": sid, "status": "COMPLETE", "alpha": aid})
            return httpx.Response(404, json={"detail": "Not found."})
        if sim["left"] > 0:
            sim["left"] -= 1
            return httpx.Response(200, headers={"Retry-After": "1"}, json={"progress": 0.5})
        if len(sim["items"]) == 1:
            if not sim["children"]:
                sim["children"] = [(sid, self._new_alpha(sim["items"][0]))]
            return httpx.Response(200, json={"id": sid, "status": "COMPLETE", "alpha": sim["children"][0][1]})
        if not sim["children"]:
            sim["children"] = [(f"{sid}C{i}", self._new_alpha(item)) for i, item in enumerate(sim["items"])]
        return httpx.Response(200, json={"id": sid, "status": "COMPLETE", "children": [c for c, _ in sim["children"]]})

    def get_alpha(self, request: httpx.Request) -> httpx.Response:
        aid = request.url.path.rstrip("/").rsplit("/", 1)[-1]
        a = self.alphas.get(aid)
        return httpx.Response(200, json=a) if a else httpx.Response(404, json={"detail": "Not found."})

    def check(self, request: httpx.Request) -> httpx.Response:
        aid = request.url.path.split("/")[-2]
        a = self.alphas[aid]
        if aid not in self.checked:
            self.checked.append(aid)
            return httpx.Response(200, headers={"Retry-After": "1"})
        checks = [c for c in a["is"]["checks"] if c["name"] != "SELF_CORRELATION"] + [
            {"name": "SELF_CORRELATION", "result": "PASS" if self.self_corr < 0.7 else "FAIL", "limit": 0.7,
             "value": self.self_corr},
            {"name": "PROD_CORRELATION", "result": "PASS", "limit": 0.7, "value": 0.4}]
        return httpx.Response(200, json={"is": {"checks": checks}})

    def pnl(self, request: httpx.Request) -> httpx.Response:
        aid = request.url.path.split("/")[-3]
        sharpe = self.alphas[aid]["is"]["sharpe"]
        import numpy as np
        rng = np.random.default_rng(zlib.crc32(aid.encode()))     # not hash(): that changes per process
        daily = rng.normal(sharpe / 15.87 * 1000, 1000, 1260)
        cum = np.cumsum(daily)
        dates = [d.strftime("%Y-%m-%d") for d in __import__("pandas").bdate_range(end="2026-06-30", periods=1260)]
        return httpx.Response(200, json={"schema": {"properties": [{"name": "date"}, {"name": "pnl"}]},
                                         "records": [[d, round(float(v), 2)] for d, v in zip(dates, cum)]})

    def correlations(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"schema": {"properties": [{"name": "id"}, {"name": "correlation"}]},
                                         "records": [], "max": self.self_corr, "min": 0.0})

    def submit(self, request: httpx.Request) -> httpx.Response:
        aid = request.url.path.split("/")[-2]
        self.submitted.append(aid)
        self.alphas[aid]["status"] = "ACTIVE"
        return httpx.Response(200, json={})

    def install(self, router) -> "FakeBrain":  # noqa: ANN001  (a respx router / respx.mock)
        router.post(f"{API}/authentication").mock(return_value=httpx.Response(201, json=AUTH))
        router.get(f"{API}/authentication").mock(return_value=httpx.Response(200, json=AUTH))
        router.post(f"{API}/simulations").mock(side_effect=self.post_sim)
        router.get(url__regex=rf"{API}/simulations/\w+$").mock(side_effect=self.get_sim)
        router.get(url__regex=rf"{API}/alphas/\w+/check$").mock(side_effect=self.check)
        router.get(url__regex=rf"{API}/alphas/\w+/recordsets/pnl$").mock(side_effect=self.pnl)
        router.get(url__regex=rf"{API}/alphas/\w+/correlations/\w+$").mock(side_effect=self.correlations)
        router.post(url__regex=rf"{API}/alphas/\w+/submit$").mock(side_effect=self.submit)
        router.get(url__regex=rf"{API}/alphas/\w+$").mock(side_effect=self.get_alpha)
        return self
