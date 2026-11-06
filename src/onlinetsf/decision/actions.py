"""Online update actions: selected units, learning rates, periods and duration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping


def positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class UpdateAction:
    """Learning rate multipliers; omitted scales and periods default to 1.

    A period collects that many arrived samples for one batch update.
    Units with native scheduling do not accept a configurable batch period.

    An empty set of units freezes parameters and adaptation state while
    preserving the method's necessary prediction and feedback records.
    """

    units: frozenset[str] = frozenset()
    learning_rate_scales: Mapping[str, float] = field(default_factory=dict)
    periods: Mapping[str, int] = field(default_factory=dict)
    duration: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.units, str):
            raise ValueError("units must be a collection of unit names")
        units = frozenset(self.units)
        if any(not isinstance(name, str) or not name for name in units):
            raise ValueError("unit names must be nonempty strings")
        scales, periods = dict(self.learning_rate_scales), dict(self.periods)
        positive_integer(self.duration, "duration")
        for name, scale in scales.items():
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"learning rate scale for {name} must be finite and positive")
        for name, period in periods.items():
            positive_integer(period, f"period for {name}")
        if scales.keys() - units or periods.keys() - units:
            raise ValueError("learning rate scales and periods must refer to selected units")
        periods = {name: period for name, period in periods.items() if period != 1}
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "learning_rate_scales", MappingProxyType(scales))
        object.__setattr__(self, "periods", MappingProxyType(periods))

    def to_dict(self) -> dict:
        return {"units": sorted(self.units),
                "learning_rate_scales": dict(self.learning_rate_scales), "periods": dict(self.periods),
                "duration": self.duration}

    @classmethod
    def from_dict(cls, record: dict) -> UpdateAction:
        return cls(**record)

    def __reduce__(self):
        # MappingProxyType protects actions from mutation by a policy/caller.
        return (type(self).from_dict, (self.to_dict(),))
