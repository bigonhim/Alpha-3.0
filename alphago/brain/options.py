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
    def from_payload(cls, payload: dict[str, Any]) -> SimOptions:
        """Parse OPTIONS /simulations. The settings tree of each simulation type (REGULAR, REGION_AGNOSTIC) is
        folded into the general one, choices merged per region by value (alpha-harness endpoints._fold/_union), so
        region ALL and its universes appear when the account may run region-agnostic alphas."""
        opts = cls(regions=[], per_region={}, global_choices={}, ranges={}, live=True)
        post = ((payload or {}).get("actions") or {}).get("POST") or {}
        type_choices = (post.get("type") or {}).get("choices") or []
        opts.types = _values(type_choices) or ["REGULAR", "SUPER"]
        trees = [(post.get("settings") or {}).get("children") or {}]
        for choice in type_choices:
            if isinstance(choice, dict) and choice.get("value") in ("REGULAR", "REGION_AGNOSTIC"):
                extra = ((choice.get("settings") or {}).get("children")) or {}
                if extra:
                    trees.append(extra)
        for children in trees:
            opts._fold(children)
        if not opts.regions:
            opts.regions = sorted(opts.per_region)
        return opts

    @staticmethod
    def _union(have: list[Any], new: list[Any]) -> list[Any]:
        return have + [v for v in new if v not in have]

    def _fold(self, children: dict[str, Any]) -> None:
        region_spec = (children.get("region") or {}).get("choices")
        if isinstance(region_spec, dict):
            eq = (region_spec.get("instrumentType") or {}).get("EQUITY")
            self.regions = self._union(self.regions, _values(eq if isinstance(eq, list) else []))
        elif isinstance(region_spec, list):
            self.regions = self._union(self.regions, _values(region_spec))
        for name, spec in children.items():
            if name == "region" or not isinstance(spec, dict):
                continue
            choices = spec.get("choices")
            if isinstance(choices, list):
                self.global_choices[name] = self._union(self.global_choices.get(name, []), _values(choices))
            elif isinstance(choices, dict):
                eq = (choices.get("instrumentType") or {}).get("EQUITY")
                if isinstance(eq, list):
                    self.global_choices[name] = self._union(self.global_choices.get(name, []), _values(eq))
                elif isinstance(eq, dict):
                    for region, vals in (eq.get("region") or {}).items():
                        cur = self.per_region.setdefault(region, {})
                        cur[name] = self._union(cur.get(name, []), _values(vals))
            if ("minValue" in spec or "maxValue" in spec) and name not in self.ranges:
                self.ranges[name] = (spec.get("minValue"), spec.get("maxValue"))

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

    def problems(self, settings: dict[str, Any]) -> list[str]:
        """Why BRAIN would refuse these settings for this account (empty when they are fine).

        Checked locally against the account's OPTIONS /simulations before anything is queued, so a refused value
        never costs a simulation. Only values BRAIN lists as choices are checked; free numbers are range-checked."""
        out: list[str] = []
        region = str(settings.get("region") or "")
        if self.live and self.regions and region not in self.regions and region != "ALL":
            return [f"Region {region!r} is not available for your account. Valid: {', '.join(self.regions)}."]
        checks = {"universe": self.universes(region), "delay": self.delays(region),
                  "neutralization": self.neutralizations(region)}
        for name, valid in checks.items():
            value = settings.get(name)
            if value is None or not valid:
                continue
            if name == "delay":
                ok = int(value) in [int(v) for v in valid]
            else:
                ok = str(value).upper() in {str(v).upper() for v in valid}
            if not ok:
                out.append(f"{name} {value!r} is not available for region={region!r} delay={settings.get('delay')}. "
                           f"Valid: {', '.join(str(v) for v in valid)}.")
        for name in ("pasteurization", "nanHandling", "unitHandling", "maxTrade", "maxPosition", "language",
                     "selectionHandling"):
            value = settings.get(name)
            valid = self.choices(name, region)
            if value is None or not valid or not self.has_setting(name):
                continue
            if str(value).upper() not in {str(v).upper() for v in valid}:
                out.append(f"{name} {value!r} is not offered in {region}. Valid: {', '.join(str(v) for v in valid)}.")
        for name, (lo, hi) in self.ranges.items():
            value = settings.get(name)
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                out.append(f"{name} {value!r} is not a number.")
                continue
            if (lo is not None and v < float(lo)) or (hi is not None and v > float(hi)):
                out.append(f"{name} {value} is outside BRAIN's range {lo}..{hi}.")
        if str(settings.get("maxTrade") or "OFF").upper() == "ON" == str(settings.get("maxPosition") or "OFF").upper():
            out.append("Max Trade and Max Position cannot both be ON.")
        return out

