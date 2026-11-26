"""Causal action execution with a decision at every forecast origin."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter

import torch
from torch import Tensor

from ..online import MethodFeedback
from .actions import UpdateAction
from .adapters import PreparedFeedback, build_adapter
from .capabilities import UnitSchedule, UpdateCapabilities


@dataclass(frozen=True)
class DecisionState:
    index: int
    context: Tensor
    feedback: tuple[MethodFeedback, ...]
    capabilities: UpdateCapabilities
    previous_action: UpdateAction | None
    # Read-only snapshot; counters count native opportunities, not optimizer steps.
    opportunity_counts: dict[str, int]
    features: dict[str, float | None] = field(default_factory=dict)


class UpdateController:
    """Wrap a native method for OnlineExecutor, without owning its scheduler.

    ``policy(state)`` is called once on every data clock.
    """

    measure_updates = True
    return_loss = True

    def __init__(self, method, policy: UpdateAction | Callable[[DecisionState], UpdateAction], *, features=None) -> None:
        self.method = method
        self.adapter = build_adapter(method)
        self.capabilities = self.adapter.capabilities
        self.policy = policy
        self.features = features
        self.schedule = UnitSchedule()
        self.action: UpdateAction | None = None
        self.last_decided_index: int | None = None
        self.index: int | None = None
        self.context: Tensor | None = None
        self.last_decision: dict | None = None
        self.prepared: deque[PreparedFeedback] = deque()
        self.decision_count = self.gradient_steps = self.egd_steps = 0
        self.gradient_update_clocks = 0
        self.decision_seconds = self.update_seconds = 0.0
        self._gradient_clock: int | None = None

    def enable_search_mode(self, executor) -> None:
        """Strip diagnostics, timing and unused loss syncs for isolated rollouts."""
        self.features = None
        self.measure_updates = False
        self.return_loss = False
        if hasattr(self.adapter, "enable_search_mode"):
            self.adapter.enable_search_mode(executor)

    def validate_feedback_delay(self, delay: int | tuple[int, ...]) -> None:
        horizon = self.method.model.horizon
        if self.method.incremental_feedback:
            if delay != tuple(range(1, horizon + 1)):
                raise ValueError("DSOF requires first-step through complete feedback at delays 1..horizon")
        elif not isinstance(delay, int) or delay < horizon:
            raise ValueError("complete feedback must wait for the whole forecast horizon")

    def on_observation(self, index: int, context: Tensor) -> None:
        if self.index is not None and index != self.index + 1:
            raise ValueError("update actions require consecutive data clocks")
        self.index = index
        if self.features is not None:
            self.features.on_observation(context)
        if self.measure_updates or not isinstance(self.policy, UpdateAction):
            self.context = torch.as_tensor(context).detach().clone()
        else:
            self.context = context
        self.last_decision = None

    def prepare_feedback(self, feedback: tuple[MethodFeedback, ...]) -> None:
        """First ingest all currently arrived labels, then decide before updates."""
        if self.prepared:
            raise RuntimeError("previous feedback batch has not been consumed")
        for item in feedback:
            prepared = self.adapter.prepare(item, training=item.current_context is not None)
            self.prepared.append(prepared)
            if self.features is not None:
                self.features.observe(prepared)
        if feedback and feedback[0].current_context is not None:
            self._decide(feedback)

    def _decide(self, feedback: tuple[MethodFeedback, ...] = ()) -> None:
        if self.index is None:
            raise RuntimeError("call on_observation before executing an action")
        if self.last_decided_index == self.index:
            return
        if not self.measure_updates and isinstance(self.policy, UpdateAction) and self.action == self.policy:
            self.last_decided_index = self.index
            self.decision_count += 1
            return
        began = perf_counter() if self.measure_updates else 0.0
        if isinstance(self.policy, UpdateAction):
            action = self.policy
        else:
            needs_features = getattr(self.policy, "needs_features", True)
            if callable(needs_features):
                needs_features = needs_features(self.index)
            features = self.features.snapshot(self.method) if self.features is not None and needs_features else {}
            state = DecisionState(self.index, self.context.clone() if self.measure_updates else self.context, feedback,
                                  self.capabilities, self.action, dict(self.schedule.counts), features)
            action = self.policy(state)
        if not isinstance(action, UpdateAction):
            raise TypeError("policy must return an UpdateAction")
        same_action = self.action == action
        if not same_action:
            self.capabilities.validate(action)
        if self.measure_updates:
            self.decision_seconds += perf_counter() - began
        if not same_action:
            if self.action is not None:
                changed = frozenset(name for name in self.action.units
                                    if name not in action.units
                                    or self.action.learning_rate_scales.get(name, 1.0) != action.learning_rate_scales.get(name, 1.0))
                self._finish_batches(changed)
            began = perf_counter() if self.measure_updates else 0.0
            self.adapter.configure(action.learning_rate_scales)
            self.schedule.activate(action)
            self.action = action
        elif self.measure_updates:
            began = perf_counter()
        self.last_decided_index = self.index
        self.decision_count += 1
        if self.measure_updates:
            self.last_decision = {"index": self.index, "action": action.to_dict()}
            selection = getattr(self.policy, "last_selection", None)
            if isinstance(selection, dict):
                self.last_decision.update(selection)
            self.decision_seconds += perf_counter() - began

    def on_feedback(self, feedback: MethodFeedback) -> float | None:
        if not self.prepared or self.prepared[0].feedback is not feedback:
            raise RuntimeError("feedback must be prepared as a batch before execution")
        prepared = self.prepared.popleft()
        if feedback.current_context is None:
            return None
        due = frozenset(name for name in prepared.opportunities if self.schedule.due(name))
        if not due:
            self.method.last_gradient_steps = self.method.last_egd_steps = 0
            return None
        began = None
        if self.measure_updates:
            began = perf_counter()
        loss = self.adapter.apply(prepared, due, self.schedule, return_loss=self.return_loss)
        self._record_update(began)
        return loss

    def _record_update(self, began: float | None) -> None:
        if began is not None:
            self.update_seconds += perf_counter() - began
        self.gradient_steps += self.method.last_gradient_steps
        self.egd_steps += self.method.last_egd_steps
        if self.method.last_gradient_steps and self._gradient_clock != self.index:
            self.gradient_update_clocks += 1
            self._gradient_clock = self.index

    def _finish_batches(self, units: frozenset[str]) -> None:
        pending = self.adapter.batches.pending
        names = [unit.name for unit in self.capabilities.units if unit.name in units and pending.get(unit.name)]
        if not names:
            return
        began = None
        if self.measure_updates:
            began = perf_counter()
        self.adapter.apply_batches(self.adapter.batches.take(names), self.schedule, return_loss=self.return_loss)
        for name in names:
            self.schedule.counts[name] = 0
        self._record_update(began)

    def finish_updates(self) -> None:
        """Use all buffered, already arrived labels before the final prediction."""
        self._decide()
        self._finish_batches(self.capabilities.unit_names)

    def predict(self, context: Tensor) -> Tensor:
        self._decide()
        return self.adapter.predict(context, self.schedule)
