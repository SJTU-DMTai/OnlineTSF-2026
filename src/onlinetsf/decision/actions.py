"""Update actions depend on declared capabilities, never on model class names."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations

from .config import DecisionConfig


@dataclass(frozen=True)
class Policy:
    name: str
    mode: str = "native"
    period: int = 1
    lr_scale: float = 1.0
    groups: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.name or self.mode not in {"native", "no_grad", "frozen"}:
            raise ValueError("action requires a name and a valid mode")
        if not isinstance(self.period, int) or isinstance(self.period, bool) or self.period <= 0:
            raise ValueError("period must be a positive integer")
        if not math.isfinite(self.lr_scale) or self.lr_scale <= 0:
            raise ValueError("lr_scale must be finite and positive")
        if self.mode != "native" and self.groups:
            raise ValueError("no_grad/frozen cannot select gradient update groups")

    def to_dict(self) -> dict:
        record = asdict(self)
        record["groups"] = sorted(self.groups) if self.groups is not None else "all"
        return record

    @classmethod
    def from_dict(cls, record: dict) -> Policy:
        return cls(**{**record, "groups": None if record["groups"] == "all" else frozenset(record["groups"])})


def policies(groups: tuple[str, ...], config: DecisionConfig, suite: str = "both") -> list[Policy]:
    result = [Policy("native")]
    if suite in {"when", "both"}:
        result.extend(Policy(mode, mode=mode) for mode in config.modes)
        for period in config.periods:
            for scale in config.lr_scales:
                if period != 1 or scale != 1.0:
                    result.append(Policy(f"every{period}_lr{scale:g}", period=period, lr_scale=scale))
    if suite in {"what", "both"}:
        if "no_grad" in config.modes and not any(item.name == "no_grad" for item in result):
            result.append(Policy("no_grad", mode="no_grad"))
        subsets = config.group_subsets
        if subsets is None:
            subsets = tuple(subset for size in range(1, len(groups)) for subset in combinations(groups, size))
        for subset in subsets:
            if not subset or len(subset) != len(set(subset)) or set(subset) - set(groups):
                raise ValueError(f"invalid action update groups: {subset}; available: {groups}")
            result.append(Policy("only_" + "+".join(subset), groups=frozenset(subset)))
    if len({item.name for item in result}) != len(result):
        raise ValueError("action configuration produces duplicate names")
    return result
