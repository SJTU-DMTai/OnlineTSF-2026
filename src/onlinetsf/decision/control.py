"""Execute update actions through method capabilities, without model inspection."""

from __future__ import annotations

from collections import deque
from time import perf_counter

import torch

from .actions import Policy
from ..online import MethodFeedback


class ControlledMethod:
    """Control neural updates while retaining formula-only native adaptation."""

    def __init__(self, method, policy: Policy):
        self.method = method
        self.groups = frozenset(method.update_group_names)
        self.incremental = method.incremental_feedback
        self.last_update: dict | None = None
        self.last_gradient_at: int | None = None
        self.recent_update_flags: deque[int] = deque(maxlen=128)
        self._last_opportunity: int | None = None
        self.base_rates = [(optimizer, [group["lr"] for group in optimizer.param_groups])
                           for optimizer in method.update_optimizers]
        self.set_policy(policy)

    def set_policy(self, policy: Policy) -> None:
        if policy.groups is not None and policy.groups - self.groups:
            raise ValueError(f"unknown update groups: {policy.groups - self.groups}")
        self.policy = policy
        self.opportunities = self.gradient_steps = self.egd_steps = 0
        self._last_opportunity = None
        for optimizer, rates in self.base_rates:
            for group, rate in zip(optimizer.param_groups, rates, strict=True):
                group["lr"] = rate * policy.lr_scale

    def switch_policy(self, policy: Policy) -> None:
        if self.policy == policy:
            return
        gradient_steps, egd_steps = self.gradient_steps, self.egd_steps
        self.set_policy(policy)
        self.gradient_steps, self.egd_steps = gradient_steps, egd_steps

    def gradient_age(self, index: int) -> int | None:
        return index - self.last_gradient_at if self.last_gradient_at is not None else None

    def configure_feature_history(self, feature_window: int, error_window: int) -> None:
        """Configure optional method diagnostics without inspecting model structure."""
        configure = getattr(self.method, "configure_feature_history", None)
        if configure is not None:
            configure(feature_window, error_window)

    def predict(self, context):
        return self.method.predict(context, advance_state=self.policy.mode != "frozen")

    def on_feedback(self, feedback: MethodFeedback, *, training: bool = True) -> float | None:
        self.last_update = None
        if not training or feedback.current_context is None:
            self.method.discard_feedback(feedback)
            return None
        # Multiple partial labels may arrive at one raw clock. Period controls
        # clocks with feedback, rather than the number of partial label events.
        new_clock = self._last_opportunity != feedback.available_at
        if new_clock:
            self._selected = self.policy.mode == "native" and self.opportunities % self.policy.period == 0
            self.opportunities += 1
            self._last_opportunity = feedback.available_at
            self.recent_update_flags.append(0)
        if self.policy.mode == "frozen":
            self.method.discard_feedback(feedback)
            return None
        groups = (self.groups if self.policy.groups is None else self.policy.groups) if self._selected else frozenset()
        if self.method.device.type == "cuda":
            torch.cuda.synchronize(self.method.device)
        began = perf_counter()
        loss = self.method.on_feedback(feedback, update_groups=groups)
        if self.method.device.type == "cuda":
            torch.cuda.synchronize(self.method.device)
        steps, egd = self.method.last_gradient_steps, self.method.last_egd_steps
        self.gradient_steps += steps
        self.egd_steps += egd
        self.recent_update_flags[-1] |= int(steps > 0)
        if steps:
            self.last_gradient_at = feedback.available_at
        self.last_update = {"forecast_index": feedback.index, "available_at": feedback.available_at,
                            "groups": "+".join(sorted(groups)), "gradient_steps": self.gradient_steps,
                            "egd_steps": self.egd_steps, "adaptation_loss": loss,
                            "seconds": perf_counter() - began}
        return loss
