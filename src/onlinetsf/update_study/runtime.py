"""Bounded-memory causal replay and explicit gradient-update controls."""

from __future__ import annotations

import copy
from collections import deque

import torch

from ..decision.control import ControlledMethod
from ..decision.actions import Policy
from ..online import MethodFeedback
from ..decision.config import DecisionConfig
from ..decision.features import extract_features, history_size


def rng_state() -> tuple:
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def restore_rng(state: tuple) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def move_method(method, device: str) -> None:
    """Move trainable state while leaving delayed forecasts and replay on CPU."""
    method.device = torch.device(device)
    method.model.to(method.device)
    for optimizer in method.update_optimizers:
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[name] = value.to(method.device)


def load_replay(path, device: str) -> tuple:
    """Restore trainable state on device while keeping delayed feedback on CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    replay = checkpoint["replay"]
    move_method(replay.method.method, device)
    saved_rng = checkpoint["rng"]
    if saved_rng[1] is not None and torch.cuda.is_available() and len(saved_rng[1]) != torch.cuda.device_count():
        raise ValueError("use the same visible GPU count as the source checkpoint, normally one")
    return replay, saved_rng


class Replay:
    """Snapshots contain no future labels; only emitted forecasts await delayed scoring."""

    def __init__(self, method: ControlledMethod, horizon: int, config: DecisionConfig | None = None):
        self.method = method
        self.horizon = horizon
        self.config = config or DecisionConfig()
        self.method.configure_feature_history(self.config.feature_window, self.config.error_window)
        self.feature_names = self.config.feature_sets(method.method.adaptation_feature_names)["policy_full"]
        capacity = history_size(self.config, self.feature_names)
        self.method.recent_update_flags = deque(maxlen=capacity)
        self.pending: deque[tuple[int, torch.Tensor]] = deque()
        self.recent_errors: deque[float] = deque(maxlen=capacity)
        self.recent_signed_errors: deque[float] = deque(maxlen=capacity)
        self.recent_channel_signed_errors: deque[torch.Tensor] = deque(maxlen=capacity)
        self.last_index: int | None = None

    def fork(self, policy: Policy) -> Replay:
        branch = copy.deepcopy(self)
        branch.method.set_policy(policy)
        return branch

    def receive(self, dataset, index: int, *, training: bool) -> list[dict]:
        records = []
        if self.method.incremental and training:
            # Deliver exactly the newly observed coordinate of every outstanding
            # forecast, in emission order (the same order as OnlineExecutor).
            for origin, prediction in self.pending:
                step = index - origin - 1
                if 0 <= step < self.horizon:
                    context, target = dataset[origin]
                    mask = torch.zeros_like(target, dtype=torch.bool)
                    mask[step] = True
                    feedback = MethodFeedback(origin, index, context, target.masked_fill(~mask, float("nan")),
                                              mask, prediction, dataset[index][0])
                    self.method.on_feedback(feedback)
                    if self.method.last_update is not None:
                        records.append({"update": self.method.last_update})
        while self.pending and self.pending[0][0] + self.horizon <= index:
            origin, prediction = self.pending.popleft()
            context, target = dataset[origin]
            error = prediction - target
            self.recent_errors.append(error.abs().mean().item())
            self.recent_signed_errors.append(error.mean().item())
            self.recent_channel_signed_errors.append(error.mean(dim=0).detach().cpu().clone())
            feedback = MethodFeedback(origin, origin + self.horizon, context, target,
                                      torch.ones_like(target, dtype=torch.bool), prediction,
                                      dataset[index][0] if training else None)
            expert_errors = self.method.method.expert_errors(target)
            if not self.method.incremental:
                self.method.on_feedback(feedback, training=training)
            elif not training:
                # Partial labels seen during the live stream must not be counted
                # again when the remaining horizon is scored at termination.
                mask = torch.zeros_like(target, dtype=torch.bool)
                mask[max(0, self.last_index - origin):] = True
                feedback = MethodFeedback(origin, origin + self.horizon, context,
                                          target.masked_fill(~mask, float("nan")), mask, prediction)
                self.method.method.discard_feedback(feedback)
            records.append({"index": origin, "available_at": origin + self.horizon,
                            "mae": error.abs().mean().item(), "mse": error.square().mean().item(),
                            "error": error, "prediction": prediction, "target": target,
                            "expert_errors": expert_errors,
                            "update": None if self.method.incremental else self.method.last_update})
        return records

    def step(self, dataset, index: int) -> list[dict]:
        if self.last_index is not None and index != self.last_index + 1:
            raise ValueError("replay must advance one raw observation at a time")
        records = self.receive(dataset, index, training=True)
        context, _ = dataset[index]
        prediction = self.method.predict(context).detach().cpu().clone()
        if not prediction.isfinite().all():
            raise FloatingPointError(f"non-finite forecast at window {index}")
        self.pending.append((index, prediction))
        self.last_index = index
        return records

    def finish(self, dataset) -> list[dict]:
        records = []
        while self.pending:
            records.extend(self.receive(dataset, self.pending[0][0] + self.horizon, training=False))
        return records


def state_features(stream, replay: Replay, index: int, *, names: list[str] | None = None) -> dict:
    """Read only observed rows and matured labels, using the production extractor."""
    clock = index + stream.dataset.context_length - 1
    names = replay.feature_names if names is None else names
    size = history_size(replay.config, names)
    inputs = stream.dataset.decision_values
    if inputs is None:
        inputs = stream.dataset.values
    values = inputs[max(0, clock - size + 1):clock + 1]
    target, current_mae, age = None, None, None
    if replay.pending and replay.pending[0][0] + replay.horizon <= index:
        origin, prediction = replay.pending[0]
        _, target = stream.dataset[origin]
        current_mae = (prediction - target).abs().mean().item()
        age = index - origin
    return extract_features(values, names=names, config=replay.config,
                            errors=replay.recent_errors, signed_errors=replay.recent_signed_errors,
                            channel_signed_errors=replay.recent_channel_signed_errors,
                            current_context=stream.dataset[index][0],
                            controlled=replay.method, index=index, feedback_target=target,
                            current_feedback_mae=current_mae, feedback_age=age)
