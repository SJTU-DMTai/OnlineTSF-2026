"""One configuration shared by collection, fitting and online inference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


FEATURE_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class DecisionConfig:
    interval: int = 128
    data_features: tuple[str, ...] = ("channel_shift_fraction", "volatility_ratio", "autocorrelation_shift")
    feedback_features: tuple[str, ...] = ("mae_32", "error_ratio", "residual_bias_ratio")
    # None asks the method for its inexpensive, cached diagnostics.
    method_features: tuple[str, ...] | None = None
    feature_window: int = 32
    error_window: int = 128
    periods: tuple[int, ...] = (1, 4, 16)
    lr_scales: tuple[float, ...] = (0.3, 1.0, 3.0)
    modes: tuple[str, ...] = ("frozen", "no_grad")
    # None enumerates proper, nonempty subsets of the method's update groups.
    group_subsets: tuple[tuple[str, ...], ...] | None = None
    data_max_depth: int = 2
    data_min_samples_leaf: int = 4
    full_max_depth: int = 2
    full_min_samples_leaf: int = 8

    def __post_init__(self) -> None:
        for name in ("interval", "feature_window", "error_window", "data_max_depth",
                     "data_min_samples_leaf", "full_max_depth", "full_min_samples_leaf"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"decision.{name} must be a positive integer")
        if self.error_window < self.feature_window:
            raise ValueError("decision.error_window must be >= feature_window")
        if any(not isinstance(p, int) or isinstance(p, bool) or p <= 0 for p in self.periods):
            raise ValueError("decision.periods must contain positive integers")
        if any(not math.isfinite(s) or s <= 0 for s in self.lr_scales):
            raise ValueError("decision.lr_scales must contain finite positive numbers")
        if set(self.modes) - {"frozen", "no_grad"}:
            raise ValueError("decision.modes supports frozen and no_grad")
        for name in ("data_features", "feedback_features", "method_features", "periods", "lr_scales", "modes"):
            values = getattr(self, name)
            if values is not None and len(values) != len(set(values)):
                raise ValueError(f"decision.{name} contains duplicates")

    @classmethod
    def from_dict(cls, record: dict) -> DecisionConfig:
        values = dict(record)
        # Preserve the training settings of existing collected data and saved selectors.
        for name in ("max_depth", "min_samples_leaf"):
            if name in values:
                shared = values.pop(name)
                for policy in ("data", "full"):
                    values.setdefault(f"{policy}_{name}", shared)
        for name in ("data_features", "feedback_features", "method_features", "periods", "lr_scales", "modes"):
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        if values.get("group_subsets") is not None:
            values["group_subsets"] = tuple(tuple(group) for group in values["group_subsets"])
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> DecisionConfig:
        record = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("decision configuration must be a mapping")
        return cls.from_dict(record)

    def to_dict(self) -> dict:
        # JSON-compatible lists make manifest equality independent of YAML/JSON.
        import json

        return json.loads(json.dumps(asdict(self)))

    def feature_sets(self, method_features: tuple[str, ...] | list[str]) -> dict[str, list[str]]:
        internal = method_features if self.method_features is None else self.method_features
        full = list(dict.fromkeys((*self.data_features, *self.feedback_features, *internal)))
        if not full or not self.data_features:
            raise ValueError("policy_full and policy_data must each have at least one feature")
        return {"policy_full": full, "policy_data": list(self.data_features)}

    def tree_parameters(self, controller: str) -> dict[str, int]:
        policy = {"policy_full": "full", "policy_data": "data"}[controller]
        return {name: getattr(self, f"{policy}_{name}") for name in ("max_depth", "min_samples_leaf")}
