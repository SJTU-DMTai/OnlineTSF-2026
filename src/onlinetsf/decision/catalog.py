"""Named finite candidate actions and their structural model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .actions import UpdateAction
from .capabilities import UpdateCapabilities


@dataclass
class ActionCatalog:
    method: str
    ids: list[str]
    actions: list[UpdateAction]
    capabilities: UpdateCapabilities

    def __post_init__(self) -> None:
        if not self.ids or len(self.ids) != len(self.actions) or len(set(self.ids)) != len(self.ids):
            raise ValueError("candidate action IDs must be nonempty and unique")
        if any(not isinstance(name, str) or not name for name in self.ids):
            raise ValueError("candidate action IDs must be nonempty strings")
        keys = []
        for action in self.actions:
            self.capabilities.validate(action)
            keys.append((action.units,
                         tuple(sorted((k, v) for k, v in action.learning_rate_scales.items() if v != 1)),
                         tuple(sorted(action.periods.items())), action.duration))
        if len(set(keys)) != len(keys):
            raise ValueError("candidate catalog contains duplicate configurations")
        if not self.native_indices:
            raise ValueError("candidate catalog must include a Native configuration")

    @property
    def native_indices(self) -> list[int]:
        return [i for i, action in enumerate(self.actions)
                if action.units == self.capabilities.unit_names and not action.periods
                and all(scale == 1 for scale in action.learning_rate_scales.values())]

    @property
    def encoding_names(self) -> list[str]:
        return ([f"enabled_{unit.name}" for unit in self.capabilities.units]
                + [f"log_lr_scale_{unit.name}" for unit in self.capabilities.units if unit.has_learning_rate]
                + [f"log_period_{unit.name}" for unit in self.capabilities.units if unit.has_batch_period]
                + ["log_duration"])

    def encode(self) -> np.ndarray:
        rows = []
        for action in self.actions:
            rows.append([float(unit.name in action.units) for unit in self.capabilities.units]
                        + [np.log(action.learning_rate_scales.get(unit.name, 1)) if unit.name in action.units else 0.
                           for unit in self.capabilities.units if unit.has_learning_rate]
                        + [np.log(action.periods.get(unit.name, 1)) if unit.name in action.units else 0.
                           for unit in self.capabilities.units if unit.has_batch_period]
                        + [np.log(action.duration)])
        return np.asarray(rows, dtype=np.float64)

    def best(self, scores: np.ndarray, legal: np.ndarray | None = None) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2 or scores.shape[1] != len(self.ids) or not np.isfinite(scores).all():
            raise ValueError("scores must be finite and have one column per candidate")
        mask = np.ones_like(scores, dtype=bool) if legal is None else np.broadcast_to(legal, scores.shape)
        if not mask.any(axis=1).all():
            raise ValueError("each state must have at least one legal candidate")
        # Native wins exact ties, followed by the persisted catalog order.
        order = self.native_indices + [i for i in range(len(self.ids)) if i not in self.native_indices]
        ranked = np.where(mask, scores, -np.inf)[:, order]
        return np.asarray(order)[ranked.argmax(axis=1)]

    def to_dict(self) -> dict:
        return {"method": self.method,
                "actions": [{"id": name, **action.to_dict()} for name, action in zip(self.ids, self.actions)]}

    @classmethod
    def from_dict(cls, record: dict, capabilities: UpdateCapabilities) -> ActionCatalog:
        if set(record) != {"method", "actions"} or not isinstance(record["actions"], list):
            raise ValueError("action catalog requires method and actions")
        ids, actions = [], []
        for row in record["actions"]:
            values = dict(row)
            ids.append(values.pop("id"))
            actions.append(UpdateAction.from_dict(values))
        return cls(record["method"], ids, actions, capabilities)

    @classmethod
    def load(cls, path: str | Path, method: str, capabilities: UpdateCapabilities) -> ActionCatalog:
        record = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        catalog = cls.from_dict(record, capabilities)
        if catalog.method != method:
            raise ValueError("action catalog method differs from the forecasting method")
        return catalog
