"""Bounded-memory causal replay and explicit gradient-update controls."""

from __future__ import annotations

import copy
from collections import deque

import torch

from ..adaptation import ControlledMethod, Policy, policies
from ..online import MethodFeedback


def rng_state() -> tuple:
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def restore_rng(state: tuple) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


class Replay:
    """Snapshots contain no future labels; only emitted forecasts await delayed scoring."""

    def __init__(self, method: ControlledMethod, horizon: int):
        self.method = method
        self.horizon = horizon
        self.pending: deque[tuple[int, torch.Tensor]] = deque()
        self.recent_errors: deque[float] = deque(maxlen=128)
        self.recent_signed_errors: deque[float] = deque(maxlen=128)
        self.last_index: int | None = None

    def fork(self, policy: Policy) -> Replay:
        branch = copy.deepcopy(self)
        branch.method.set_policy(policy)
        return branch

    def receive(self, dataset, index: int, *, training: bool) -> list[dict]:
        records = []
        if self.method.adapter.incremental and training:
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
            feedback = MethodFeedback(origin, origin + self.horizon, context, target,
                                      torch.ones_like(target, dtype=torch.bool), prediction,
                                      dataset[index][0] if training else None)
            expert_errors = self.method.adapter.expert_errors(target)
            if not self.method.adapter.incremental:
                self.method.on_feedback(feedback, training=training)
            elif not training:
                self.method.adapter.discard(feedback)
            records.append({"index": origin, "available_at": origin + self.horizon,
                            "mae": error.abs().mean().item(), "mse": error.square().mean().item(),
                            "error": error, "prediction": prediction, "target": target,
                            "expert_errors": expert_errors,
                            "update": None if self.method.adapter.incremental else self.method.last_update})
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


def state_features(stream, replay: Replay, index: int) -> dict:
    """Read current inputs and matured forecast errors without executing extra model forwards."""
    dataset = stream.dataset
    clock = index + dataset.context_length - 1
    values = dataset.values
    features = {}
    for width in (32, 128):
        recent = values[max(0, clock - width + 1):clock + 1]
        previous = values[max(0, clock - 2 * width + 1):max(0, clock - width + 1)]
        features[f"input_mean_{width}"] = recent.mean().item()
        features[f"input_std_{width}"] = recent.std(dim=0, unbiased=False).mean().item()
        features[f"input_trend_{width}"] = (recent[-1] - recent[0]).abs().mean().item() / max(1, len(recent) - 1)
        centered = recent - recent.mean(dim=0)
        variance = centered.square().mean(dim=0)
        features[f"autocorrelation_{width}"] = ((centered[1:] * centered[:-1]).mean(dim=0) / variance.clamp_min(1e-8)).mean().item()
        features[f"mean_shift_{width}"] = (recent.mean(dim=0) - previous.mean(dim=0)).abs().mean().item() if len(previous) else None
        errors = list(replay.recent_errors)[-width:]
        features[f"mae_{width}"] = sum(errors) / len(errors) if errors else None
    # Correlation on at most 16 fixed channels bounds the cost for Electricity.
    channel_indices = torch.linspace(0, values.shape[1] - 1, min(16, values.shape[1])).long()
    recent = values[max(0, clock - 127):clock + 1, channel_indices]
    centered = recent - recent.mean(dim=0)
    normalized = centered / centered.square().mean(dim=0).sqrt().clamp_min(1e-8)
    correlation = normalized.T @ normalized / len(recent)
    mask = ~torch.eye(len(channel_indices), dtype=torch.bool)
    features["cross_correlation"] = correlation[mask].abs().mean().item() if mask.any() else 0.0
    features["residual_bias"] = sum(replay.recent_signed_errors) / len(replay.recent_signed_errors) if replay.recent_signed_errors else None
    features["steps_since_gradient"] = index - replay.method.last_gradient_at if replay.method.last_gradient_at is not None else None
    flags = replay.method.recent_update_flags
    features["recent_update_fraction"] = sum(flags) / len(flags) if flags else None
    if replay.pending and replay.pending[0][0] + replay.horizon <= index:
        origin, prediction = replay.pending[0]
        _, target = dataset[origin]
        features["current_feedback_mae"] = (prediction - target).abs().mean().item()
        features["feedback_age"] = replay.horizon
    else:
        features["current_feedback_mae"] = None
        features["feedback_age"] = None
    target = None
    if replay.pending and replay.pending[0][0] + replay.horizon <= index:
        _, target = dataset[replay.pending[0][0]]
    features.update(replay.method.adapter.features(target))
    return features
