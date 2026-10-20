"""Method-specific capabilities used by the shared online decision runtime."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import torch

from .methods import DSOFMethod, FSNetMethod, OneNetMethod
from .online import MethodFeedback


UPDATE_GROUPS = {
    "onenet": ("time", "variable", "decision"),
    "fsnet": ("base", "adapter", "head"),
    # student = supervised replay; online = fast TD. These share parameters but
    # have independent optimizers and independently selectable training paths.
    "dsof": ("teacher", "student", "online"),
}


@dataclass(frozen=True)
class Policy:
    name: str
    mode: str = "native"
    period: int = 1
    lr_scale: float = 1.0
    groups: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"native", "no_grad", "frozen"}:
            raise ValueError("unknown update mode")
        if not isinstance(self.period, int) or isinstance(self.period, bool) or self.period <= 0:
            raise ValueError("period must be a positive integer")
        if not math.isfinite(self.lr_scale) or self.lr_scale <= 0:
            raise ValueError("lr_scale must be finite and positive")
        if self.mode != "native" and self.groups:
            raise ValueError("no_grad/frozen cannot select gradient update groups")


def policies(method: str, lr_scales: list[float], suite: str) -> list[Policy]:
    names = UPDATE_GROUPS[method]
    result = [Policy("native")]
    if suite in {"when", "both"}:
        result.extend([Policy("frozen", mode="frozen"), Policy("no_grad", mode="no_grad")])
        for period in (1, 4, 16):
            for scale in lr_scales:
                if period != 1 or scale != 1.0:
                    result.append(Policy(f"every{period}_lr{scale:g}", period=period, lr_scale=scale))
    if suite in {"what", "both"}:
        for size in range(len(names)):
            for subset in combinations(names, size):
                if subset:
                    result.append(Policy("only_" + "+".join(subset), groups=frozenset(subset)))
                elif not any(item.name == "no_grad" for item in result):
                    result.append(Policy("no_grad", mode="no_grad"))
    return result


class MethodAdapter:
    """Keep model structure and method-specific state out of scheduling."""

    def __init__(self, method):
        self.method = method
        if isinstance(method, OneNetMethod):
            self.name = "onenet"
        elif isinstance(method, FSNetMethod):
            self.name = "fsnet"
        elif isinstance(method, DSOFMethod):
            self.name = "dsof"
        else:
            raise TypeError(f"unsupported online method: {type(method).__name__}")
        self.groups = frozenset(UPDATE_GROUPS[self.name])
        self.incremental = self.name == "dsof"

    def predict(self, context, *, frozen: bool = False):
        if frozen and self.name == "fsnet":
            self.method.model.eval()
            with torch.no_grad():
                return self.method.model(context.unsqueeze(0).to(self.method.device), advance_state=False).squeeze(0)
        return self.method.predict(context)

    def discard(self, feedback: MethodFeedback) -> None:
        """Retire emission bookkeeping without adapting model or optimizer state."""
        if self.name == "onenet":
            self.method._pending_predictions.popleft()
        elif self.name == "dsof":
            self.method._partial.pop(feedback.index, None)

    def update_counts(self, groups: frozenset[str]) -> tuple[int, int]:
        if self.name == "dsof":
            return self.method.last_gradient_steps, 0
        if self.name == "fsnet":
            return self.method.n_inner * int(bool(groups)), 0
        return self.method.n_inner * (int(bool(groups & {"time", "variable"})) + int("decision" in groups)), self.method.n_inner

    def expert_errors(self, target) -> tuple[float, float] | None:
        if self.name == "onenet":
            time, variable = self.method._pending_predictions[0]
            return (time - target).abs().mean().item(), (variable - target).abs().mean().item()
        return None

    def features(self, target=None) -> dict:
        method = self.method
        if self.name == "onenet":
            result = {"long_time_weight": method.model.weights(include_short_term=False).mean().item(),
                      "combined_time_weight": method.model.weights().mean().item()}
            if method._pending_predictions:
                time, variable = method._pending_predictions[0]
                result["expert_disagreement"] = (time - variable).abs().mean().item()
                if target is not None:
                    result["time_expert_mae"], result["variable_expert_mae"] = self.expert_errors(target)
            return result
        if self.name == "fsnet":
            layers = [module for module in method.model.modules() if hasattr(module, "gradient_ema")]
            return {"gradient_memory_norm": sum(module.gradient_ema.norm().item() for module in layers) / len(layers),
                    "memory_interactions": sum(module.memory_interactions.item() for module in layers)}
        return {"replay_fill_fraction": len(method.replay) / method.replay.maxlen,
                "replay_updates_due": method._complete_count % method.replay_frequency}


class ControlledMethod:
    """Control neural updates while retaining formula-only native adaptation."""

    def __init__(self, method, policy: Policy):
        self.method = method
        self.adapter = MethodAdapter(method)
        self.last_update: dict | None = None
        self.last_gradient_at: int | None = None
        self.recent_update_flags: deque[int] = deque(maxlen=128)
        self._last_opportunity: int | None = None
        self.base_rates = {name: [group["lr"] for group in optimizer.param_groups]
                           for name, optimizer in vars(method).items() if isinstance(optimizer, torch.optim.Optimizer)}
        self.set_policy(policy)

    def set_policy(self, policy: Policy) -> None:
        if policy.groups is not None and policy.groups - self.adapter.groups:
            raise ValueError(f"unknown {self.adapter.name} update groups: {policy.groups - self.adapter.groups}")
        self.policy = policy
        self.opportunities = self.gradient_steps = self.egd_steps = 0
        self._last_opportunity = None
        for name, rates in self.base_rates.items():
            for group, rate in zip(getattr(self.method, name).param_groups, rates, strict=True):
                group["lr"] = rate * policy.lr_scale

    def switch_policy(self, policy: Policy) -> None:
        if self.policy == policy:
            return
        gradient_steps, egd_steps = self.gradient_steps, self.egd_steps
        self.set_policy(policy)
        self.gradient_steps, self.egd_steps = gradient_steps, egd_steps

    def predict(self, context):
        return self.adapter.predict(context, frozen=self.policy.mode == "frozen")

    def on_feedback(self, feedback: MethodFeedback, *, training: bool = True) -> float | None:
        self.last_update = None
        if not training or feedback.current_context is None or self.policy.mode == "frozen":
            self.adapter.discard(feedback)
            return None
        # Multiple partial labels may arrive at one raw clock. Period controls
        # clocks with feedback, rather than the number of partial label events.
        new_clock = self._last_opportunity != feedback.available_at
        if new_clock:
            self._selected = self.policy.mode != "no_grad" and self.opportunities % self.policy.period == 0
            self.opportunities += 1
            self._last_opportunity = feedback.available_at
        groups = (self.adapter.groups if self.policy.groups is None else self.policy.groups) if self._selected else frozenset()
        if self.method.device.type == "cuda":
            torch.cuda.synchronize(self.method.device)
        began = perf_counter()
        loss = self.method.on_feedback(feedback, update_groups=groups)
        if self.method.device.type == "cuda":
            torch.cuda.synchronize(self.method.device)
        steps, egd = self.adapter.update_counts(groups)
        self.gradient_steps += steps
        self.egd_steps += egd
        if new_clock:
            self.recent_update_flags.append(int(steps > 0))
        else:
            self.recent_update_flags[-1] |= int(steps > 0)
        if steps:
            self.last_gradient_at = feedback.available_at
        self.last_update = {"forecast_index": feedback.index, "available_at": feedback.available_at,
                            "groups": "+".join(sorted(groups)), "gradient_steps": self.gradient_steps,
                            "egd_steps": self.egd_steps, "adaptation_loss": loss,
                            "seconds": perf_counter() - began}
        return loss
