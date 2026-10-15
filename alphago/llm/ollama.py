"""Ollama (local models) via its REST API: /api/tags, /api/show, /api/chat with a JSON-schema `format`.

Ollama's docs state no default context length and a longer prompt is cut silently, so every request sends
`options.num_ctx` (the LLM page setting, capped at the model's own context length from /api/show), plus
`options.num_predict` so a small model stuck in a loop cannot generate until the context is full."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel

from ..events import bus
from .base import ChatProvider, LLMError, Reply, inline_schema

log = logging.getLogger(__name__)


class OllamaProvider(ChatProvider):
    name = "ollama"
    label = "Ollama (local)"
    local = True

    def __init__(self) -> None:
        super().__init__()
        self._http: httpx.AsyncClient | None = None
        self._http_key: tuple[str, float] | None = None
        self._info: dict[str, dict[str, Any]] = {}       # model -> {"ctx": int | None, "thinking": bool}
        self._no_think: set[str] = set()                   # models that rejected the `think` parameter
        self._warned: set[str] = set()

    @property
    def base_url(self) -> str:
        return (self.cfg.base_url or "http://localhost:11434").rstrip("/")

    def configure(self, api_key: str | None = None) -> bool:
        if not self.cfg.base_url:
            self.status = "unconfigured"
            return False
        if self.status == "unconfigured":
            self.status = "configured"
        return True

    @property
    def http(self) -> httpx.AsyncClient:
        key = (self.base_url, float(self.cfg.timeout_s or 600))
        if self._http is None or self._http_key != key:
            self._http = httpx.AsyncClient(base_url=key[0], timeout=httpx.Timeout(key[1], connect=5.0))
            self._http_key = key
        return self._http

    async def _request(self, method: str, path: str, json: Any = None) -> Any:
        try:
            response = await self.http.request(method, path, json=json)
        except httpx.ConnectError as exc:
            raise LLMError(f"Ollama is not reachable at {self.base_url} — is it installed and running? "
                           "(ollama.com)") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama did not answer within {self._http_key[1]:.0f} s — try a smaller model or raise "
                           "the timeout.") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}", transient=True) from exc
        body: Any
        try:
            body = response.json() if response.content else {}
        except ValueError:
            body = {"error": response.text}
        if response.status_code >= 400:
            msg = (body or {}).get("error") if isinstance(body, dict) else str(body)
            raise LLMError(f"Ollama error {response.status_code}: {msg}", response.status_code,
                           transient=response.status_code == 503)
        return body

    async def list_models(self) -> list[dict[str, Any]]:
        body = await self._request("GET", "/api/tags")
        out = []
        for m in (body or {}).get("models") or []:
            name = m.get("model") or m.get("name")
            if not name:
                continue
            d = m.get("details") or {}
            out.append({"name": name, "display": name, "size": m.get("size"), "family": d.get("family"),
                        "parameter_size": d.get("parameter_size"), "preview": False})
        out.sort(key=lambda d: d["name"])
        return await self._remember_models(out)

    def pick(self, role: str) -> str:
        return self.explicit_model(role) or (self.models[0]["name"] if self.models else "")

    async def _model_info(self, model: str) -> dict[str, Any]:
        if model not in self._info:
            body = await self._request("POST", "/api/show", {"model": model})
            info = (body or {}).get("model_info") or {}
            ctx = next((v for k, v in info.items() if k.endswith(".context_length") and isinstance(v, int)), None)
            self._info[model] = {"ctx": ctx, "thinking": "thinking" in ((body or {}).get("capabilities") or [])}
        return self._info[model]

    def _warn_once(self, model: str, message: str) -> None:
        if model not in self._warned:
            self._warned.add(model)
            bus.publish("llm", message, "warning")

    async def _complete(self, model: str, prompt: str, system: str | None, schema: type[BaseModel] | None,
                        thinking: str) -> Reply:
        info = await self._model_info(model)
        num_ctx = int(self.cfg.num_ctx or 32_768)
        if info.get("ctx"):
            num_ctx = min(num_ctx, int(info["ctx"]))
        estimate = (len(system or "") + len(prompt)) // 4          # rough: ~4 characters per token
        if estimate > num_ctx:
            self._warn_once(model, f"Ollama: this prompt is roughly {estimate:,} tokens but {model} runs with a "
                                   f"{num_ctx:,}-token context, so part of it is cut. Raise 'Context window' on "
                                   "the LLM Integration page or use a model with a longer context.")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        limit = int(self.cfg.num_predict or 8_192)
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False, "keep_alive": "30m",
                                "options": {"num_ctx": num_ctx, "num_predict": limit}}
        if schema is not None:
            body["format"] = inline_schema(schema)
        if info.get("thinking") and model not in self._no_think:
            body["think"] = thinking in ("low", "medium", "high")
        try:
            data = await self._request("POST", "/api/chat", body)
        except LLMError as exc:
            if exc.code == 400 and "think" in body and "think" in exc.message.lower():
                self._no_think.add(model)         # e.g. a model that only accepts thinking levels or none at all
                body.pop("think")
                data = await self._request("POST", "/api/chat", body)
            else:
                raise
        seen = (data or {}).get("prompt_eval_count")
        if isinstance(seen, int) and seen >= num_ctx * 0.98:
            self._warn_once(model, f"Ollama: the prompt filled {model}'s whole {num_ctx:,}-token context and was "
                                   "cut. Raise 'Context window' on the LLM Integration page.")
        if (data or {}).get("done_reason") == "length":
            # a cut-off reply is unusable JSON, and a repair round would cost another slow local call
            raise LLMError(f"Ollama: {model} hit the {limit:,}-token reply limit before finishing (small models can "
                           "loop) — try a larger model.")
        message = (data or {}).get("message") or {}
        return Reply(text=message.get("content") or "", prompt_tokens=seen, output_tokens=(data or {}).get("eval_count"))


ollama = OllamaProvider()
