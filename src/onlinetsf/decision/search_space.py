"""Configurable action dimensions, constrained by the method's capabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .actions import UpdateAction, positive_integer
from .capabilities import UpdateCapabilities
from .catalog import ActionCatalog


def validate_dimension(spec, name: str, *, integer: bool) -> None:
    if spec is None:
        return
    if isinstance(spec, list):
        if not spec or len(set(spec)) != len(spec):
            raise ValueError(f"{name} choices must be nonempty and distinct")
        values = spec
    elif isinstance(spec, dict):
        allowed = {"low", "high", "log", "step"} if integer else {"low", "high", "log"}
        if set(spec) - allowed or not {"low", "high"} <= spec.keys():
            raise ValueError(f"{name} range requires low/high and optional log" + ("/step" if integer else ""))
        values = [spec["low"], spec["high"]]
        if not isinstance(spec.get("log", False), bool):
            raise ValueError(f"{name}.log must be boolean")
        if integer:
            positive_integer(spec.get("step", 1), f"{name}.step")
            if spec.get("log", False) and spec.get("step", 1) != 1:
                raise ValueError(f"{name} log range requires step 1")
    else:
        raise ValueError(f"{name} must be null, a list or a low/high range")
    for value in values:
        if integer:
            positive_integer(value, name)
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} values must be finite and positive")
    if isinstance(spec, dict):
        if spec["low"] > spec["high"]:
            raise ValueError(f"{name}.low must not exceed high")
        if integer and (spec["high"] - spec["low"]) % spec.get("step", 1):
            raise ValueError(f"{name} high must lie on the low/step grid")


def suggest_dimension(trial, name: str, spec, *, integer: bool):
    if spec is None:
        return 1
    if isinstance(spec, list):
        return spec[0] if len(spec) == 1 else trial.suggest_categorical(name, spec)
    if spec["low"] == spec["high"]:
        return spec["low"]
    if integer:
        return trial.suggest_int(name, spec["low"], spec["high"],
                                 step=spec.get("step", 1), log=spec.get("log", False))
    return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))


def includes_native(spec) -> bool:
    """Whether a positive dimension allows the native value, 1."""
    return spec is None or (1 in spec if isinstance(spec, list) else spec["low"] <= 1 <= spec["high"])


@dataclass
class ActionSearchSpace:
    capabilities: UpdateCapabilities
    durations: list[int]
    units: list[str] | None = None
    learning_rate_scales: dict | list | None = None
    periods: dict | list | None = None

    def __post_init__(self) -> None:
        validate_dimension(self.durations, "durations", integer=True)
        if not isinstance(self.durations, list):
            raise ValueError("durations must be a nonempty list")
        self.durations = sorted(self.durations)
        if self.units is not None:
            if not isinstance(self.units, list) or any(not isinstance(name, str) for name in self.units):
                raise ValueError("units must be null (search all subsets) or a fixed list")
            if len(set(self.units)) != len(self.units) or set(self.units) - self.capabilities.unit_names:
                raise ValueError("units must be distinct names declared by the method")
        validate_dimension(self.learning_rate_scales, "learning_rate_scales", integer=False)
        validate_dimension(self.periods, "periods", integer=True)

    def suggest(self, trial) -> UpdateAction:
        selected, scales, periods = set(), {}, {}
        for unit in self.capabilities.units:
            enabled = (unit.name in self.units if self.units is not None else
                       trial.suggest_categorical(f"enable_{unit.name}", [False, True]))
            if not enabled:
                continue
            selected.add(unit.name)
            if unit.has_learning_rate:
                scale = suggest_dimension(trial, f"lr_scale_{unit.name}", self.learning_rate_scales, integer=False)
                if scale != 1:
                    scales[unit.name] = scale
            if unit.has_batch_period:
                period = suggest_dimension(trial, f"period_{unit.name}", self.periods, integer=True)
                if period != 1:
                    periods[unit.name] = period
        action = UpdateAction(units=selected, learning_rate_scales=scales, periods=periods,
                              duration=suggest_dimension(trial, "duration", self.durations, integer=True))
        self.capabilities.validate(action)
        return action

    def to_dict(self) -> dict:
        return {"units": self.units, "learning_rate_scales": self.learning_rate_scales,
                "periods": self.periods, "durations": self.durations}

    def write_catalog(self, method: str, candidates: list[dict], path: str | Path) -> ActionCatalog:
        """Export selected configurations, adding only missing Native/Frozen references.

        Preserve searched IDs, order and durations. Added references use the
        smallest configured H, even if their update scope lies outside the
        sampled search dimensions.
        """
        ids = [row["id"] for row in candidates]
        actions = [UpdateAction.from_dict(row["action"]) for row in candidates]
        if not any(action.units == self.capabilities.unit_names and not action.periods
                   and all(scale == 1 for scale in action.learning_rate_scales.values()) for action in actions):
            ids.append("native")
            actions.append(self.capabilities.native_action(duration=self.durations[0]))
        if not any(not action.units for action in actions):
            ids.append("frozen")
            actions.append(UpdateAction(duration=self.durations[0]))
        catalog = ActionCatalog(method, ids, actions, self.capabilities)
        Path(path).write_text(yaml.safe_dump(catalog.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
        return catalog
