"""Configurable action dimensions, constrained by the method's capabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .actions import UpdateAction
from .capabilities import UpdateCapabilities
from .catalog import ActionCatalog


def validate_dimension(spec, name: str) -> None:
    if spec is None:
        return
    if isinstance(spec, list):
        if not spec or len(set(spec)) != len(spec):
            raise ValueError(f"{name} choices must be nonempty and distinct")
        values = spec
    elif isinstance(spec, dict):
        if set(spec) - {"low", "high", "log"} or not {"low", "high"} <= spec.keys():
            raise ValueError(f"{name} range requires low/high and optional log")
        values = [spec["low"], spec["high"]]
        if not isinstance(spec.get("log", False), bool):
            raise ValueError(f"{name}.log must be boolean")
    else:
        raise ValueError(f"{name} must be null, a list or a low/high range")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} values must be finite and positive")
    if isinstance(spec, dict):
        if spec["low"] > spec["high"]:
            raise ValueError(f"{name}.low must not exceed high")


def suggest_dimension(trial, name: str, spec):
    if spec is None:
        return 1
    if isinstance(spec, list):
        return spec[0] if len(spec) == 1 else trial.suggest_categorical(name, spec)
    if spec["low"] == spec["high"]:
        return spec["low"]
    return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))


def includes_native(spec) -> bool:
    """Whether a positive dimension allows the native value, 1."""
    return spec is None or (1 in spec if isinstance(spec, list) else spec["low"] <= 1 <= spec["high"])


@dataclass
class ActionSearchSpace:
    capabilities: UpdateCapabilities
    units: list[str] | None = None
    learning_rate_scales: dict | list | None = None

    def __post_init__(self) -> None:
        if self.units is not None:
            if not isinstance(self.units, list) or any(not isinstance(name, str) for name in self.units):
                raise ValueError("units must be null (search all subsets) or a fixed list")
            if len(set(self.units)) != len(self.units) or set(self.units) - self.capabilities.unit_names:
                raise ValueError("units must be distinct names declared by the method")
        validate_dimension(self.learning_rate_scales, "learning_rate_scales")

    def suggest(self, trial) -> UpdateAction:
        selected, scales = set(), {}
        for unit in self.capabilities.units:
            enabled = (unit.name in self.units if self.units is not None else
                       trial.suggest_categorical(f"enable_{unit.name}", [False, True]))
            if not enabled:
                continue
            selected.add(unit.name)
            if unit.has_learning_rate:
                scale = suggest_dimension(trial, f"lr_scale_{unit.name}", self.learning_rate_scales)
                if scale != 1:
                    scales[unit.name] = scale
        action = UpdateAction(units=selected, learning_rate_scales=scales)
        self.capabilities.validate(action)
        return action

    def to_dict(self) -> dict:
        return {"units": self.units, "learning_rate_scales": self.learning_rate_scales}

    def write_catalog(self, method: str, candidates: list[dict], path: str | Path) -> ActionCatalog:
        """Preserve searched IDs and order; add missing Native/Frozen references."""
        ids = [row["id"] for row in candidates]
        actions = [UpdateAction.from_dict(row["action"]) for row in candidates]
        if not any(action.units == self.capabilities.unit_names
                   and all(scale == 1 for scale in action.learning_rate_scales.values()) for action in actions):
            ids.append("native")
            actions.append(self.capabilities.native_action())
        if not any(not action.units for action in actions):
            ids.append("frozen")
            actions.append(UpdateAction())
        catalog = ActionCatalog(method, ids, actions, self.capabilities)
        Path(path).write_text(yaml.safe_dump(catalog.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
        return catalog
