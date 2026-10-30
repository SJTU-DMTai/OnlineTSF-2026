"""Load a fitted decision policy and control any capable online method."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from time import perf_counter

import torch

from .control import ControlledMethod
from ..online import MethodFeedback
from .actions import Policy
from .config import DecisionConfig, FEATURE_SCHEMA_VERSION
from .features import extract_features, history_size
from .selector import select_action


class DecisionEngine:
    """Shared action selection for production and branch experiments.

    Construction only loads trees and checks capabilities. No model is trained,
    copied or evaluated here. The caller controls the decision clock.
    """

    def __init__(self, bundle: dict, controller: str, method):
        if bundle.get("schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported selector artifact; refit with the current decision module")
        self.config = DecisionConfig.from_dict(bundle["decision_config"])
        self.actions = {record["name"]: Policy.from_dict(record) for record in bundle["actions"]}
        if self.actions.get("native") != Policy("native"):
            raise ValueError("selector artifact requires the unchanged native action")
        if set(bundle["update_groups"]) != set(method.update_group_names):
            raise ValueError("selector artifact and online method have different update capabilities")
        self.controller = controller
        self.selector = bundle["selectors"].get(controller)
        if self.selector is None and controller not in {"native", "best_fixed", "frozen"}:
            raise ValueError(f"unknown decision controller: {controller}")
        self.fixed = controller if controller in {"native", "frozen"} else bundle["best_fixed"]
        if controller == "frozen" and self.actions.get("frozen") != Policy("frozen", mode="frozen"):
            raise ValueError("frozen controller requires the frozen action in the catalog")
        self.feature_names = self.selector["features"] if self.selector is not None else []
        if self.selector is not None and self.selector["actions"] != list(self.actions):
            raise ValueError("selector and action catalog ordering differ")
        for policy in self.actions.values():
            if policy.groups is not None and policy.groups - set(method.update_group_names):
                raise ValueError(f"unsupported action groups: {policy.groups}")

    def decide(self, features: dict) -> tuple[Policy, dict | None]:
        if self.selector is None:
            return self.actions[self.fixed], None
        action, scores = select_action(self.selector, features)
        return self.actions[action], scores


class AdaptiveMethod(ControlledMethod):
    """An online method with causal decisions and bounded feature history.

    Used directly by OnlineExecutor. Decisions occur before the first feedback
    update at a clock, matching the pre-feedback snapshots used for training.
    Partial labels contribute to rolling errors only when a forecast is complete.
    """

    def __init__(self, method, bundle: dict, controller: str = "policy_full"):
        self.engine = DecisionEngine(bundle, controller, method)
        super().__init__(method, self.engine.actions["native"])
        self.config = self.engine.config
        self.configure_feature_history(self.config.feature_window, self.config.error_window)
        self.feature_names = self.engine.feature_names
        capacity = history_size(self.config, self.feature_names)
        self.observations: deque = deque(maxlen=capacity)
        self.errors: deque = deque(maxlen=capacity)
        self.signed_errors: deque = deque(maxlen=capacity)
        self.channel_signed_errors: deque = deque(maxlen=capacity)
        self.recent_update_flags = deque(maxlen=capacity)
        self._partial_errors: dict[int, tuple[torch.Tensor, int]] = {}
        self._index: int | None = None
        self._next_decision: int | None = None
        self.last_decision: dict | None = None
        self.decision_count = 0
        self.decision_seconds = 0.0
        self.update_seconds = 0.0
        self.gradient_update_clocks = 0
        self._gradient_clock: int | None = None

    @classmethod
    def from_file(cls, method, path: str | Path, controller: str = "policy_full") -> AdaptiveMethod:
        return cls(method, json.loads(Path(path).read_text(encoding="utf-8")), controller)

    def on_observation(self, index: int, context: torch.Tensor) -> None:
        if self._index is not None and index != self._index + 1:
            raise ValueError("adaptive decisions require consecutive observation indices")
        context = torch.as_tensor(context)
        if self._index is None:
            self.observations.extend(context.detach().cpu().clone().unbind())
            self._next_decision = index
        else:
            self.observations.append(context[-1].detach().cpu().clone())
        self._index = index
        self.last_decision = None

    def _decide(self, feedback: MethodFeedback | None = None, *, current_context=None) -> None:
        if self._index is None:
            raise RuntimeError("call on_observation before prediction or feedback")
        if self._index < self._next_decision:
            return
        began = perf_counter()
        complete = feedback is not None and bool(feedback.observed_mask.all())
        current_mae, age = None, None
        if feedback is not None:
            previous = self._partial_errors.get(feedback.index)
            count = previous[1] if previous is not None else 0
            if count + int(feedback.observed_mask.sum()) == feedback.target.numel():
                error = feedback.prediction - feedback.target
                if previous is not None:
                    error = torch.where(feedback.observed_mask, error, previous[0])
                current_mae = error.abs().mean().item()
                age = self._index - feedback.index
        features = extract_features(
            torch.stack(tuple(self.observations)), names=self.feature_names, config=self.config,
            errors=self.errors, signed_errors=self.signed_errors, controlled=self, index=self._index,
            channel_signed_errors=self.channel_signed_errors,
            current_context=feedback.current_context if feedback is not None else current_context,
            feedback_target=feedback.target if complete else None,
            current_feedback_mae=current_mae, feedback_age=age,
        ) if self.feature_names else {}
        policy, scores = self.engine.decide(features)
        self.switch_policy(policy)
        self.last_decision = {"index": self._index, "selected_action": policy.name,
                              "steps_since_gradient": self.gradient_age(self._index),
                              "features": features, "predicted_gains": scores}
        self._next_decision = self._index + self.config.interval
        self.decision_count += 1
        self.decision_seconds += perf_counter() - began

    def predict(self, context):
        self._decide(current_context=context)
        return super().predict(context)

    def on_feedback(self, feedback: MethodFeedback, *, training: bool = True) -> float | None:
        live = training and feedback.current_context is not None
        if live:
            self._decide(feedback)
        # Assemble observed errors in forecast order so complete/partial delivery
        # uses exactly the same reductions as historical collection.
        error, count = self._partial_errors.pop(feedback.index, (torch.zeros_like(feedback.prediction), 0))
        error[feedback.observed_mask] = (feedback.prediction - feedback.target)[feedback.observed_mask]
        count += int(feedback.observed_mask.sum())
        if count == feedback.target.numel():
            self.errors.append(error.abs().mean().item())
            self.signed_errors.append(error.mean().item())
            self.channel_signed_errors.append(error.mean(dim=0).detach().cpu().clone())
        else:
            self._partial_errors[feedback.index] = error, count
        before = self.gradient_steps
        loss = super().on_feedback(feedback, training=training)
        if self.last_update is not None:
            self.update_seconds += self.last_update["seconds"]
        if self.gradient_steps > before and self._gradient_clock != feedback.available_at:
            self.gradient_update_clocks += 1
            self._gradient_clock = feedback.available_at
        return loss
