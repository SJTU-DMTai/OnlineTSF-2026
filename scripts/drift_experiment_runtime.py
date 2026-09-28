# -*- coding: utf-8 -*-
"""Causal replay controls. Original methods remain unchanged outside experiments."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from time import perf_counter

import torch
from torch.nn import functional as F

from onlinetsf.methods import DSOFMethod, FSNetMethod, OneNetMethod, UnderCaliMethod
from onlinetsf.online import MethodFeedback


def rng_state() -> tuple:
    return random.getstate(), torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None


def restore_rng(state: tuple) -> None:
    random.setstate(state[0])
    torch.set_rng_state(state[1])
    if state[2] is not None:
        torch.cuda.set_rng_state_all(state[2])


def optimizer_counts(method) -> dict[str, int]:
    """Adam/AdamW step clocks count real parameter updates, not feedback calls."""
    counts = {}
    for name, value in vars(method).items():
        if isinstance(value, torch.optim.Optimizer):
            if not isinstance(value, (torch.optim.Adam, torch.optim.AdamW)):
                raise TypeError("experiment step accounting supports the project's Adam/AdamW optimizers")
            counts[name] = max((int(state.get("step", 0)) for state in value.state.values()), default=0)
    return counts


def method_parameters(method) -> list:
    modules = [method.model]
    if isinstance(method, UnderCaliMethod):
        modules.extend((method.reliable, method.unreliable, method.uncertainty))
    return [parameter for module in modules for parameter in module.parameters()]


def method_state(method) -> dict:
    """Read-only internal signals; no extra prediction calls or random draws."""
    if isinstance(method, DSOFMethod):
        return {"replay_size": len(method.replay), "partial_windows": len(method._partial),
                "replay_clock": method._complete_count}
    if isinstance(method, UnderCaliMethod):
        return {"allocation_mean": method.allocation_mean, "allocation_variance": method.allocation_variance,
                "trigger_mean": method.trigger_mean, "trigger_variance": method.trigger_variance,
                "last_uncertainty": method._last_route[0] if method._last_route else None,
                "last_reliable": method._last_route[1] if method._last_route else None}
    if isinstance(method, OneNetMethod):
        return {"long_weight": method.model.weights(include_short_term=False).detach().cpu().tolist(),
                "combined_weight": method.model.weights().detach().cpu().tolist(),
                "short_bias": method.model.short_bias.detach().cpu().tolist()}
    if isinstance(method, FSNetMethod):
        layers = [module for module in method.model.modules() if hasattr(module, "memory_interactions")]
        return {"memory_interactions": [int(layer.memory_interactions) for layer in layers],
                "memory_trigger": [bool(layer.memory_trigger) for layer in layers],
                "gradient_cosine": [F.cosine_similarity(layer.fast_gradient_ema, layer.gradient_ema, dim=0).item() for layer in layers]}
    return {}


def under_cali_feedback(method, feedback: MethodFeedback, action: str, rounds: int | None) -> float | None:
    """Consume each route and advance its trigger once, even when optimization is skipped.

    Forced updates change only the gate decision. UE still updates only on reliable
    routes, and the source model stays frozen, as in the native implementation.
    """
    score, reliable = method._pending_routes.pop(feedback.index)
    if method.trigger_mean is None:
        should_adapt = False
        method.trigger_mean = score
    else:
        threshold = method.trigger_mean + method.trigger_std_k * math.sqrt(method.trigger_variance)
        should_adapt = score > threshold
        delta = score - method.trigger_mean
        method.trigger_mean += method.trigger_alpha * delta
        method.trigger_variance = (1 - method.trigger_alpha) * method.trigger_variance + method.trigger_alpha * delta * delta
    if action == "skip" or (action == "native" and not should_adapt):
        return None
    context = feedback.context.unsqueeze(0).to(method.device)
    target = feedback.target.unsqueeze(0).to(method.device)
    mask = feedback.observed_mask.unsqueeze(0).to(method.device)
    expert = method.reliable if reliable else method.unreliable
    optimizer = method.reliable_optimizer if reliable else method.unreliable_optimizer
    steps = method.update_steps if rounds is None else rounds
    losses = []
    expert.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        forecast = expert(context, method.model)
        loss = F.mse_loss(forecast[mask], target[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.detach().item())
    expert.eval()
    if reliable:
        method.uncertainty.train()
        with torch.no_grad():
            forecast = method.reliable(context, method.model)
            errors = (forecast - target).masked_fill(~mask, 0.0).square().mean(dim=(1, 2))
            normalized = ((errors - method.uncertainty.error_min) /
                          (method.uncertainty.error_max - method.uncertainty.error_min).clamp_min(1e-8)).clamp(0.0, 1.0)
        for _ in range(steps):
            method.uncertainty_optimizer.zero_grad(set_to_none=True)
            loss = F.l1_loss(method.uncertainty(context, forecast), normalized)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(method.uncertainty.parameters(), 1.0)
            method.uncertainty_optimizer.step()
        method.uncertainty.eval()
    return sum(losses) / len(losses)


def dsof_feedback(method, feedback: MethodFeedback, action: str, rounds: int | None) -> float | None:
    """Always ingest causal labels; suppress training without losing replay history."""
    if feedback.current_context is None:
        return None
    if feedback.index not in method._partial:
        method._partial[feedback.index] = (torch.zeros_like(feedback.target), torch.zeros_like(feedback.observed_mask))
    values, observed = method._partial[feedback.index]
    values[feedback.observed_mask] = feedback.target[feedback.observed_mask]
    observed |= feedback.observed_mask
    losses = []
    if observed.all():
        method.replay.append((feedback.context.detach().clone(), values.detach().clone()))
        del method._partial[feedback.index]
        if action == "skip":
            if len(method.replay) >= method.replay_batch_size:
                method._complete_count += 1
        else:
            old_epochs = method.replay_epochs
            if rounds is not None:
                method.replay_epochs = rounds
            try:
                replay_loss = method._replay_update()
            finally:
                method.replay_epochs = old_epochs
            if replay_loss is not None:
                losses.append(replay_loss)
    if action != "skip" and feedback.observed_mask[0].all():
        for _ in range(rounds or 1):
            losses.append(method._fast_update(feedback))
    return sum(losses) / len(losses) if losses else None


def apply_feedback(method, feedback: MethodFeedback, action: str, rounds: int | None = None, frozen: bool = False):
    if action == "native" and rounds is None:
        return method.on_feedback(feedback)
    if frozen:
        if isinstance(method, OneNetMethod):
            method._pending_predictions.popleft()
        elif isinstance(method, UnderCaliMethod):
            method._pending_routes.pop(feedback.index)
        return None
    if isinstance(method, DSOFMethod):
        return dsof_feedback(method, feedback, action, rounds)
    if isinstance(method, UnderCaliMethod):
        return under_cali_feedback(method, feedback, action, rounds)
    if action == "skip":
        if isinstance(method, OneNetMethod):
            method._pending_predictions.popleft()
        return None
    option = "n_inner" if isinstance(method, (OneNetMethod, FSNetMethod)) else "update_steps"
    old_steps = getattr(method, option)
    if rounds is not None:
        setattr(method, option, rounds)
    try:
        return method.on_feedback(feedback)
    finally:
        setattr(method, option, old_steps)


def is_opportunity(method, feedback: MethodFeedback) -> bool:
    if feedback.current_context is None:
        return False
    if not isinstance(method, DSOFMethod):
        return True
    fast = bool(feedback.observed_mask[0].all())
    complete = bool(feedback.observed_mask[-1].all())
    replay_ready = len(method.replay) + 1 >= method.replay_batch_size
    replay_due = method._complete_count % method.replay_frequency == 0
    return fast or (complete and replay_ready and replay_due)


def update_opportunities(method, start: int, stop: int, horizon: int) -> list[tuple[int, int]]:
    """(arrival index, forecast origin), using timing only; excludes terminal flush."""
    events = set()
    for origin in range(start, stop):
        if isinstance(method, DSOFMethod):
            if origin + 1 < stop:
                events.add((origin + 1, origin))
            complete_count = origin - start + 1
            if origin + horizon < stop and complete_count >= method.replay_batch_size:
                if (complete_count - method.replay_batch_size) % method.replay_frequency == 0:
                    events.add((origin + horizon, origin))
        elif origin + horizon < stop:
            events.add((origin + horizon, origin))
    return sorted(events)


@dataclass
class UpdatePolicy:
    """Small action description, with no access to future observations."""

    name: str = "native"
    period: int = 1
    rounds: int | None = None
    selected: set[tuple[int, int]] = field(default_factory=set)
    force: bool = False
    budget: int = 0
    total_opportunities: int = 0
    residual_threshold: float = 1.0
    round_budget: int | None = None


class ControlledMethod:
    """Apply an experimental update policy and account for real optimizer steps."""

    def __init__(self, method, policy: UpdatePolicy, context_length: int, parameter_metrics: bool = False):
        self.method = method
        self.policy = policy
        self.context_length = context_length
        self.parameter_metrics = parameter_metrics
        self.training_enabled = True
        self.opportunity_count = 0
        self.selected_count = 0
        self.rounds_used = 0
        self.first_opportunity_at: int | None = None
        self.records = []
        self.override_next: str | None = None
        self.last_components = {}
        self.collect_components = False

    def predict(self, context):
        method = self.method
        if self.policy.name == "frozen" and isinstance(method, FSNetMethod):
            method.model.eval()
            with torch.no_grad():
                prediction = method.model(context.unsqueeze(0).to(method.device), advance_state=False).squeeze(0)
        else:
            old_allocation = None
            if self.policy.name == "frozen" and isinstance(method, UnderCaliMethod):
                old_allocation = (method.allocation_mean, method.allocation_variance)
            prediction = method.predict(context)
            if old_allocation is not None:
                method.allocation_mean, method.allocation_variance = old_allocation
        self.last_components = {}
        if isinstance(method, OneNetMethod):
            first, second = method._pending_predictions[-1]
            self.last_components = {"time_expert": first.clone(), "variable_expert": second.clone()}
        elif self.collect_components and isinstance(method, DSOFMethod):
            with torch.no_grad():
                teacher = method.model.teacher(context.unsqueeze(0).to(method.device)).squeeze(0).cpu()
            self.last_components = {"teacher": teacher}
        elif self.collect_components and isinstance(method, UnderCaliMethod):
            with torch.no_grad():
                batched = context.unsqueeze(0).to(method.device)
                self.last_components = {"source": method.model(batched).squeeze(0).cpu(),
                                        "reliable": method.reliable(batched, method.model).squeeze(0).cpu(),
                                        "unreliable": method.unreliable(batched, method.model).squeeze(0).cpu()}
        return prediction

    def on_emission(self, index):
        if isinstance(self.method, UnderCaliMethod):
            self.method.on_emission(index)

    def on_feedback(self, feedback):
        method, policy = self.method, self.policy
        eligible = is_opportunity(method, feedback) and self.training_enabled
        ordinal = self.opportunity_count
        budget_fill = False
        selected = False
        if eligible:
            if self.first_opportunity_at is None:
                self.first_opportunity_at = feedback.available_at
            self.opportunity_count += 1
            if policy.name == "native":
                selected = True
            elif policy.name == "periodic":
                # A period means raw observation steps, not DSOF's multiple feedback events.
                selected = (feedback.available_at - self.first_opportunity_at) % policy.period == 0
                budget_fill = policy.round_budget is not None and ordinal == policy.total_opportunities - 1 and self.rounds_used < policy.round_budget
                selected = selected or budget_fill
            elif policy.name == "scheduled":
                selected = (feedback.available_at, feedback.index) in policy.selected
            elif policy.name == "residual":
                remaining = policy.budget - self.selected_count
                budget_fill = remaining > 0 and policy.total_opportunities - ordinal <= remaining
                error = (feedback.prediction[feedback.observed_mask] - feedback.target[feedback.observed_mask]).abs().mean().item()
                selected = remaining > 0 and (error > policy.residual_threshold or budget_fill)
        action = ("force" if policy.force else "native") if selected else "skip"
        if self.override_next is not None:
            action = self.override_next
            selected = action != "skip"
            self.override_next = None
        if not self.training_enabled or feedback.current_context is None:
            action, selected = "skip", False
        rounds = policy.rounds
        if selected and policy.round_budget is not None:
            remaining_rounds = policy.round_budget - self.rounds_used
            rounds = remaining_rounds if budget_fill else min(rounds, remaining_rounds)
            if rounds == 0:
                action, selected = "skip", False
        if selected:
            self.selected_count += 1
            if rounds is not None:
                self.rounds_used += rounds
        before = optimizer_counts(method)
        previous_state = method_state(method)
        parameters = method_parameters(method) if self.parameter_metrics else []
        saved = [parameter.detach().clone() for parameter in parameters]
        if method.device.type == "cuda":
            torch.cuda.synchronize(method.device)
        began = perf_counter()
        # Terminal feedback is scored, but cannot change parameters or adaptation state.
        loss = apply_feedback(method, feedback, action, rounds,
                              frozen=policy.name == "frozen" or not self.training_enabled)
        if method.device.type == "cuda":
            torch.cuda.synchronize(method.device)
        elapsed = perf_counter() - began
        after = optimizer_counts(method)
        changes = {name: after[name] - before[name] for name in before}
        norm = math.sqrt(sum((new.detach() - old).square().sum().item() for old, new in zip(saved, parameters))) if saved else None
        parameter_norm = math.sqrt(sum(old.square().sum().item() for old in saved)) if saved else None
        indices = feedback.observed_mask.any(dim=1).nonzero().flatten()
        sample_uses = sum(count * (method.replay_batch_size if isinstance(method, DSOFMethod) and name in {"teacher_optimizer", "student_optimizer"} else 1)
                          for name, count in changes.items())
        self.records.append({"forecast_index": feedback.index, "available_at": feedback.available_at,
                             "raw_available_index": feedback.available_at + self.context_length - 1,
                             "raw_target_start": feedback.index + self.context_length,
                             "raw_observed_target_end": feedback.index + self.context_length + int(indices[-1]),
                             "observed_values": int(feedback.observed_mask.sum()),
                             "sample_age": feedback.available_at - feedback.index,
                             "opportunity_index": ordinal if eligible else None,
                             "eligible": eligible, "selected": selected, "action": action,
                             "control_rounds": rounds if selected else 0,
                             "budget_fill": budget_fill, "terminal": not self.training_enabled,
                             "optimizer_steps": sum(changes.values()), "module_steps": json.dumps(changes),
                             "optimizer_sample_uses": sample_uses, "parameter_delta_norm": norm,
                             "parameter_relative_delta": norm / parameter_norm if parameter_norm else None,
                             "adaptation_loss": loss, "feedback_seconds": elapsed,
                             "state_before": json.dumps(previous_state), "state_after": json.dumps(method_state(method))})
        return loss


def clone_with_policy(method, policy: UpdatePolicy, context_length: int, parameter_metrics: bool = False, lr_scale: float = 1.0):
    cloned = copy.deepcopy(method)
    for optimizer in vars(cloned).values():
        if isinstance(optimizer, torch.optim.Optimizer):
            for group in optimizer.param_groups:
                group["lr"] *= lr_scale
    return ControlledMethod(cloned, policy, context_length, parameter_metrics)
