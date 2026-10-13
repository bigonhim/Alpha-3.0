"""Parse `OPTIONS /simulations` into the settings choices available to this account."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Fallback used before the first successful OPTIONS call (community-reported values; the live call is authoritative).
_STD_NEUT = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY", "REVERSION_AND_MOMENTUM", "STATISTICAL",
             "CROWDING", "FAST", "SLOW", "SLOW_AND_FAST"]
FALLBACK: dict[str, dict[str, list[Any]]] = {
    "USA": {"universe": ["TOP3000", "TOP1000", "TOP500", "TOP200", "TOPSP500", "ILLIQUID_MINVOL1M"],
            "delay": [1, 0], "neutralization": _STD_NEUT},
    "GLB": {"universe": ["TOP3000", "MINVOL1M", "TOPDIV3000"], "delay": [1], "neutralization": _STD_NEUT + ["COUNTRY"]},
    "EUR": {"universe": ["TOP2500", "TOP1200", "TOP800", "TOP400", "ILLIQUID_MINVOL1M"], "delay": [1, 0],
            "neutralization": _STD_NEUT + ["COUNTRY"]},
    "ASI": {"universe": ["MINVOL1M", "ILLIQUID_MINVOL1M"], "delay": [1], "neutralization": _STD_NEUT + ["COUNTRY"]},
    "CHN": {"universe": ["TOP2000U"], "delay": [1, 0], "neutralization": [n for n in _STD_NEUT if n != "STATISTICAL"]},
    "IND": {"universe": ["TOP500"], "delay": [1], "neutralization": [n for n in _STD_NEUT if n != "STATISTICAL"]},
}
FALLBACK_GLOBAL: dict[str, list[Any]] = {
    "pasteurization": ["ON", "OFF"],
    "unitHandling": ["VERIFY"],
    "nanHandling": ["OFF", "ON"],
    "language": ["FASTEXPR"],
    "maxTrade": ["OFF", "ON"],
    "selectionHandling": ["POSITIVE", "NON_ZERO", "NON_NAN"],
}


def _values(choices: Any) -> list[Any]:
    out = []
    for item in choices or []:
        out.append(item.get("value") if isinstance(item, dict) else item)
    return out


@dataclass
class SimOptions:
    regions: list[str] = field(default_factory=lambda: list(FALLBACK))
    per_region: dict[str, dict[str, list[Any]]] = field(default_factory=lambda: {k: dict(v) for k, v in FALLBACK.items()})
    global_choices: dict[str, list[Any]] = field(default_factory=lambda: dict(FALLBACK_GLOBAL))
    ranges: dict[str, tuple[float | None, float | None]] = field(default_factory=lambda: {
        "decay": (0, 512), "truncation": (0.0, 1.0)})
    types: list[str] = field(default_factory=lambda: ["REGULAR", "SUPER"])
    live: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SimOptions":
        opts = cls(regions=[], per_region={}, global_choices={}, ranges={}, live=True)
        post = ((payload or {}).get("actions") or {}).get("POST") or {}
        opts.types = _values((post.get("type") or {}).get("choices")) or ["REGULAR", "SUPER"]
        children = (post.get("settings") or {}).get("children") or {}
        region_spec = (children.get("region") or {}).get("choices")
        if isinstance(region_spec, dict):
            eq = (region_spec.get("instrumentType") or {}).get("EQUITY")
            opts.regions = _values(eq if isinstance(eq, list) else [])
        elif isinstance(region_spec, list):
            opts.regions = _values(region_spec)
        for name, spec in children.items():
            if name == "region" or not isinstance(spec, dict):
                continue
            choices = spec.get("choices")
            if isinstance(choices, list):
                opts.global_choices[name] = _values(choices)
            elif isinstance(choices, dict):
                eq = (choices.get("instrumentType") or {}).get("EQUITY")
                if isinstance(eq, list):
                    opts.global_choices[name] = _values(eq)
                elif isinstance(eq, dict):
                    for region, vals in (eq.get("region") or {}).items():
                        opts.per_region.setdefault(region, {})[name] = _values(vals)
            if "minValue" in spec or "maxValue" in spec:
                opts.ranges[name] = (spec.get("minValue"), spec.get("maxValue"))
        if not opts.regions:
            opts.regions = sorted(opts.per_region)
        return opts

    # -- accessors with fallback
    def universes(self, region: str) -> list[str]:
        return self.per_region.get(region, {}).get("universe") or FALLBACK.get(region, {}).get("universe", ["TOP3000"])

    def delays(self, region: str) -> list[int]:
        vals = self.per_region.get(region, {}).get("delay") or FALLBACK.get(region, {}).get("delay", [1])
        return [int(v) for v in vals]

    def neutralizations(self, region: str) -> list[str]:
        return self.per_region.get(region, {}).get("neutralization") or FALLBACK.get(region, {}).get(
            "neutralization", _STD_NEUT)

    def choices(self, name: str, region: str | None = None) -> list[Any]:
        if region and name in self.per_region.get(region, {}):
            return self.per_region[region][name]
        return self.global_choices.get(name) or FALLBACK_GLOBAL.get(name, [])

    def has_setting(self, name: str) -> bool:
        if not self.live:
            return name in FALLBACK_GLOBAL
        return name in self.global_choices or any(name in v for v in self.per_region.values()) or name in self.ranges
