"""Key health checks, the models a provider offers, and which set-up models can answer right now.

A key check never spends a generation request: it asks the provider's model list (`models.list`, Anthropic with
limit=1000; OpenRouter's /key, since its model list is public), which no provider bills against generation quota.
"""

from __future__ import annotations

from typing import Any

from ..llm.base import LLMError
from . import providers
from .keys import keystore
from .registry import registry


def _provider_for_family(family: str) -> Any:
    """The provider object whose client checks a key of `family` (the native one where there is one)."""
    from ..llm.router import PROVIDERS
    return PROVIDERS.get(family) or next((PROVIDERS[p] for p in providers.users_of(family) if p in PROVIDERS), None)


async def check_key(key_id: int) -> dict[str, Any]:
    """Confirm a key works, without spending a generation request. The outcome is stored against the key."""
    row = keystore.get(key_id)
    if row is None:
        return {"key_id": key_id, "ok": False, "error": f"No key {key_id}."}
    provider = _provider_for_family(row.provider)
    if provider is None or not hasattr(provider, "list_model_ids"):
        return {"key_id": key_id, "ok": False, "error": f"{providers.get(row.provider).label} cannot be checked."}
    try:
        names = await provider.list_model_ids(keystore.secret(key_id))
    except LLMError as exc:
        await keystore.mark(key_id, error=exc.message[:300])
        return {"key_id": key_id, "ok": False, "error": exc.message[:300]}
    except Exception as exc:  # any provider failure is the check's answer, stored and shown
        await keystore.mark(key_id, error=str(exc)[:300])
        return {"key_id": key_id, "ok": False, "error": str(exc)[:300]}
    await keystore.mark(key_id)
    return {"key_id": key_id, "ok": True, "models": len(names), "names": names}


async def check_all() -> list[dict[str, Any]]:
    return [await check_key(r.id) for r in keystore.rows()]


async def offered(provider_id: str) -> dict[str, Any]:
    """Model ids a provider's keys can reach, to pick from when setting a model up. Asked live of the first enabled
    key (or the single saved key); a failure says why, and the id can still be typed."""
    from ..llm.router import PROVIDERS
    provider = PROVIDERS.get(provider_id)
    if provider is None:
        return {"models": [], "error": f"{providers.get(provider_id).label} is not available."}
    secret = keystore.first_secret(provider_id)
    try:
        if secret and hasattr(provider, "list_model_ids") and provider_id != "openrouter":
            return {"models": await provider.list_model_ids(secret), "error": None}
        if not provider.configure():
            return {"models": [], "error": "Add a key for this provider first."}
        return {"models": [m["name"] for m in await provider.list_models()], "error": None}
    except LLMError as exc:
        return {"models": [], "error": exc.message[:300]}
    except Exception as exc:
        return {"models": [], "error": str(exc)[:300]}


async def runnable() -> list[dict[str, Any]]:
    """Set-up models whose provider can answer now (a key in the pool, or the single saved key), most requests left
    today first. A model without such a key is listed under Models but offered nowhere a request is sent."""
    from ..llm.router import PROVIDERS
    status = await keystore.status()
    left = {b["ref"]: b["remaining_today"] for b in status["budget"]}
    out = []
    for m in registry.all():
        provider = PROVIDERS.get(m.provider)
        if provider is None or not provider.configure():
            continue
        out.append({"ref": m.ref, "id": m.id, "provider": m.provider, "label": f"{m.id} · {provider.label}",
                    "remaining_today": left.get(m.ref)})
    return sorted(out, key=lambda m: -(m["remaining_today"] or 0))
