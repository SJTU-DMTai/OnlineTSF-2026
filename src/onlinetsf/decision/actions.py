"""Online update actions: selected units and learning-rate multipliers."""

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
    """Learning rate multipliers; omitted scales default to 1.

    An empty set of units freezes parameters and adaptation state while
    preserving the method's necessary prediction and feedback records.
    """

    units: frozenset[str] = frozenset()
    learning_rate_scales: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.units, str):
            raise ValueError("units must be a collection of unit names")
        units = frozenset(self.units)
        if any(not isinstance(name, str) or not name for name in units):
            raise ValueError("unit names must be nonempty strings")
        scales = dict(self.learning_rate_scales)
        for name, scale in scales.items():
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"learning rate scale for {name} must be finite and positive")
        if scales.keys() - units:
            raise ValueError("learning rate scales must refer to selected units")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "learning_rate_scales", MappingProxyType(scales))

    def to_dict(self) -> dict:
        return {"units": sorted(self.units), "learning_rate_scales": dict(self.learning_rate_scales)}

    @classmethod
    def from_dict(cls, record: dict) -> UpdateAction:
        return cls(**record)

    def __reduce__(self):
        # MappingProxyType protects actions from mutation by a policy/caller.
        return (type(self).from_dict, (self.to_dict(),))
