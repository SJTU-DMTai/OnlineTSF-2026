"""Bounded-memory causal replay and explicit gradient-update controls."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import torch

from ..methods import FSNetMethod, OneNetMethod
from ..online import MethodFeedback


@dataclass(frozen=True)
class Policy:
    name: str
    mode: str = "native"
    period: int = 1
    lr_scale: float = 1.0
    groups: frozenset[str] | None = None


def policies(method: str, lr_scales: list[float], suite: str) -> list[Policy]:
    result = [Policy("native")]
    if suite in {"when", "both"}:
        result.extend([Policy("frozen", mode="frozen"), Policy("no_grad", mode="no_grad")])
        for period in (1, 4, 16):
            for scale in lr_scales:
                if period == 1 and scale == 1.0:
                    continue
                result.append(Policy(f"every{period}_lr{scale:g}", period=period, lr_scale=scale))
    if suite in {"what", "both"}:
        names = ("base", "adapter", "head") if method == "fsnet" else ("time", "variable", "decision")
        for size in range(3):
            for subset in combinations(names, size):
                if not subset:
                    if not any(item.name == "no_grad" for item in result):
                        result.append(Policy("no_grad", mode="no_grad"))
                else:
                    result.append(Policy("only_" + "+".join(subset), groups=frozenset(subset)))
    return result


def rng_state() -> tuple:
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def restore_rng(state: tuple) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


class ControlledMethod:
    """Select optimizer updates while preserving the method's other state transitions."""

    def __init__(self, method: FSNetMethod | OneNetMethod, policy: Policy):
        self.method = method
        self.opportunities = 0
        self.gradient_steps = 0
        self.egd_steps = 0
        self.last_update: dict | None = None
        self.last_gradient_at: int | None = None
        self.recent_update_flags: deque[int] = deque(maxlen=128)
        self.base_rates = {name: [group["lr"] for group in optimizer.param_groups]
                           for name, optimizer in vars(method).items() if isinstance(optimizer, torch.optim.Optimizer)}
        self.base_weight_rate = method.weight_learning_rate if isinstance(method, OneNetMethod) else None
        self.set_policy(policy)

    def set_policy(self, policy: Policy) -> None:
        self.policy = policy
        self.opportunities = self.gradient_steps = self.egd_steps = 0
        for name, rates in self.base_rates.items():
            for group, rate in zip(getattr(self.method, name).param_groups, rates, strict=True):
                group["lr"] = rate * policy.lr_scale
        if isinstance(self.method, OneNetMethod):
            # Timing/LR ablations change neural training; retain EGD at the same rate.
            self.method.weight_learning_rate = self.base_weight_rate

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        if self.policy.mode == "frozen" and isinstance(self.method, FSNetMethod):
            self.method.model.eval()
            with torch.no_grad():
                return self.method.model(context.unsqueeze(0).to(self.method.device), advance_state=False).squeeze(0)
        return self.method.predict(context)

    def on_feedback(self, feedback: MethodFeedback, *, training: bool = True) -> None:
        self.last_update = None
        if not training or self.policy.mode == "frozen":
            if isinstance(self.method, OneNetMethod):
                self.method._pending_predictions.popleft()
            return
        selected = self.policy.mode != "no_grad" and self.opportunities % self.policy.period == 0
        self.opportunities += 1
        groups = self.policy.groups if selected else frozenset()
        if groups is None:
            groups = frozenset({"base", "adapter", "head"} if isinstance(self.method, FSNetMethod)
                               else {"time", "variable", "decision"})
        self.recent_update_flags.append(int(bool(groups)))
        if groups:
            self.last_gradient_at = feedback.available_at
        if self.method.device.type == "cuda":
            torch.cuda.synchronize(self.method.device)
        began = perf_counter()
        loss = self.method.on_feedback(feedback, update_groups=groups)
        if self.method.device.type == "cuda":
            torch.cuda.synchronize(self.method.device)
        self.gradient_steps += self.method.n_inner * (int(bool(groups)) if isinstance(self.method, FSNetMethod)
                               else int(bool(groups & {"time", "variable"})) + int("decision" in groups))
        if isinstance(self.method, OneNetMethod):
            self.egd_steps += self.method.n_inner
        self.last_update = {"forecast_index": feedback.index, "available_at": feedback.available_at,
                            "groups": "+".join(sorted(groups)), "gradient_steps": self.gradient_steps,
                            "egd_steps": self.egd_steps, "adaptation_loss": loss,
                            "seconds": perf_counter() - began}


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
        while self.pending and self.pending[0][0] + self.horizon <= index:
            origin, prediction = self.pending.popleft()
            context, target = dataset[origin]
            error = prediction - target
            self.recent_errors.append(error.abs().mean().item())
            self.recent_signed_errors.append(error.mean().item())
            feedback = MethodFeedback(origin, origin + self.horizon, context, target,
                                      torch.ones_like(target, dtype=torch.bool), prediction,
                                      dataset[index][0] if training else None)
            expert_errors = None
            if isinstance(self.method.method, OneNetMethod):
                time, variable = self.method.method._pending_predictions[0]
                expert_errors = ((time - target).abs().mean().item(), (variable - target).abs().mean().item())
            self.method.on_feedback(feedback, training=training)
            records.append({"index": origin, "available_at": origin + self.horizon,
                            "mae": error.abs().mean().item(), "mse": error.square().mean().item(),
                            "error": error, "prediction": prediction, "target": target,
                            "expert_errors": expert_errors, "update": self.method.last_update})
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
    method = replay.method.method
    if isinstance(method, OneNetMethod):
        features["long_time_weight"] = method.model.weights(include_short_term=False).mean().item()
        features["combined_time_weight"] = method.model.weights().mean().item()
        if method._pending_predictions:
            time, variable = method._pending_predictions[0]
            features["expert_disagreement"] = (time - variable).abs().mean().item()
            if replay.pending[0][0] + replay.horizon <= index:
                _, target = dataset[replay.pending[0][0]]
                features["time_expert_mae"] = (time - target).abs().mean().item()
                features["variable_expert_mae"] = (variable - target).abs().mean().item()
    else:
        layers = [module for module in method.model.modules() if hasattr(module, "gradient_ema")]
        features["gradient_memory_norm"] = sum(module.gradient_ema.norm().item() for module in layers) / len(layers)
        features["memory_interactions"] = sum(module.memory_interactions.item() for module in layers)
    return features
