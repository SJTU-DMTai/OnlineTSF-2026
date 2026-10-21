# -*- coding: utf-8 -*-
"""OneNet online ensembling for complementary forecasting experts."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.optim import Adam, AdamW, Optimizer

from ..online import MethodFeedback


LossFunction = Callable[[Tensor, Tensor], Tensor]


class OneNetEnsemble(nn.Module):
    """Combine a variable-independent expert and a cross-variable expert."""

    def __init__(
        self,
        cross_time: nn.Module,
        cross_variable: nn.Module,
        *,
        horizon: int,
        num_targets: int,
        decision_hidden: int = 32,
        decision_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if horizon <= 0 or num_targets <= 0 or decision_hidden <= 0:
            raise ValueError("horizon, num_targets, and decision_hidden must be positive")
        if not 0.0 <= decision_dropout < 1.0:
            raise ValueError("decision_dropout must be in [0, 1)")

        self.cross_time = cross_time
        self.cross_variable = cross_variable
        self.horizon = horizon
        self.num_targets = num_targets
        # One logit per target represents the two normalized expert weights as p and 1-p.
        self.register_buffer("long_logits", torch.zeros(num_targets))
        self.decision = nn.Sequential(
            nn.Linear(3 * horizon, decision_hidden),
            nn.Dropout(decision_dropout),
            nn.Tanh(),
            nn.Linear(decision_hidden, decision_hidden),
            nn.Dropout(decision_dropout),
            nn.Tanh(),
            nn.Linear(decision_hidden, 1),
        )
        self.register_buffer("short_bias", torch.zeros(num_targets))

    def expert_predictions(self, context: Tensor) -> tuple[Tensor, Tensor]:
        cross_time = self.cross_time(context)
        cross_variable = self.cross_variable(context)
        expected = (context.shape[0], self.horizon, self.num_targets)
        if cross_time.shape != expected or cross_variable.shape != expected:
            raise ValueError(f"OneNet experts must both return shape {expected}")
        return cross_time, cross_variable

    def weights(self, *, include_short_term: bool = True) -> Tensor:
        logits = self.long_logits
        if include_short_term:
            logits = logits + self.short_bias
        return torch.sigmoid(logits)

    def combine(
        self,
        cross_time: Tensor,
        cross_variable: Tensor,
        *,
        include_short_term: bool = True,
    ) -> Tensor:
        weight = self.weights(include_short_term=include_short_term).view(1, 1, -1)
        return weight * cross_time + (1.0 - weight) * cross_variable

    def decision_bias(self, cross_time: Tensor, cross_variable: Tensor, target: Tensor) -> Tensor:
        """Predict the short-term logit adjustment from one labeled forecast."""

        long_weight = self.weights(include_short_term=False).detach().view(1, 1, -1)
        decision_input = torch.cat(
            (long_weight * cross_time, (1.0 - long_weight) * cross_variable, target),
            dim=1,
        )
        return self.decision(decision_input.transpose(1, 2)).squeeze(-1)

    def forward(self, context: Tensor) -> Tensor:
        return self.combine(*self.expert_predictions(context))


class OneNetMethod:
    """Train both OneNet experts and its long- and short-term ensemble weights."""

    update_group_names = ("time", "variable", "decision")
    incremental_feedback = False
    adaptation_feature_names = ("expert_relative_gap", "combined_time_weight")

    def __init__(
        self,
        model: OneNetEnsemble,
        *,
        learning_rate: float = 1e-3,
        weight_learning_rate: float = 1e-3,
        decision_learning_rate: float = 1e-3,
        expert_optimizer: Optimizer | None = None,
        decision_optimizer: Optimizer | None = None,
        loss_fn: LossFunction | None = None,
        n_inner: int = 1,
        device: torch.device | str | None = None,
    ) -> None:
        if (
            learning_rate <= 0.0
            or weight_learning_rate <= 0.0
            or decision_learning_rate <= 0.0
        ):
            raise ValueError("OneNet learning rates must be positive")
        if not isinstance(n_inner, int) or isinstance(n_inner, bool) or n_inner <= 0:
            raise ValueError("n_inner must be positive")

        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.n_inner = n_inner
        self.last_gradient_steps = self.last_egd_steps = 0
        expert_parameters = list(model.cross_time.parameters()) + list(
            model.cross_variable.parameters()
        )
        self.expert_optimizer = expert_optimizer or AdamW(expert_parameters, lr=learning_rate)
        self.weight_learning_rate = weight_learning_rate
        self.decision_optimizer = decision_optimizer or Adam(
            model.decision.parameters(), lr=decision_learning_rate
        )
        self.loss_fn = loss_fn or nn.MSELoss()
        # Scalar feedback delays preserve emission order, so a FIFO retains exact expert forecasts.
        self._pending_predictions: deque[tuple[Tensor, Tensor]] = deque()

    @property
    def update_optimizers(self) -> tuple[Optimizer, ...]:
        return self.expert_optimizer, self.decision_optimizer

    def predict(self, context: Tensor, *, advance_state: bool = True) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            cross_time, cross_variable = self.model.expert_predictions(
                context.unsqueeze(0).to(self.device)
            )
            prediction = self.model.combine(cross_time, cross_variable)
        self._pending_predictions.append(
            (
                cross_time.squeeze(0).detach().cpu(),
                cross_variable.squeeze(0).detach().cpu(),
            )
        )
        return prediction.squeeze(0)

    def discard_feedback(self, feedback: MethodFeedback) -> None:
        self._pending_predictions.popleft()

    def expert_errors(self, target) -> tuple[float, float] | None:
        if not self._pending_predictions:
            return None
        time, variable = self._pending_predictions[0]
        return (time - target).abs().mean().item(), (variable - target).abs().mean().item()

    def adaptation_features(self, target=None, *, current_context=None) -> dict:
        result = {"long_time_weight": self.model.weights(include_short_term=False).mean().item(),
                  "combined_time_weight": self.model.weights().mean().item(),
                  "expert_disagreement": None, "time_expert_mae": None, "variable_expert_mae": None}
        if self._pending_predictions:
            time, variable = self._pending_predictions[0]
            result["expert_disagreement"] = (time - variable).abs().mean().item()
            if target is not None:
                result["time_expert_mae"], result["variable_expert_mae"] = self.expert_errors(target)
        return result

    def on_feedback(self, feedback: MethodFeedback, *, update_groups: frozenset[str] | None = None) -> float | None:
        """EGD and bias refresh remain active even when neural parameters stay fixed."""
        groups = frozenset({"time", "variable", "decision"}) if update_groups is None else update_groups
        if groups - {"time", "variable", "decision"}:
            raise ValueError("unknown OneNet update group")
        if not feedback.observed_mask.all():
            raise ValueError("OneNetMethod requires complete-horizon feedback")
        if not self._pending_predictions:
            raise RuntimeError("OneNet feedback has no matching emitted expert predictions")

        cross_time, cross_variable = self._pending_predictions.popleft()
        cross_time = cross_time.unsqueeze(0).to(self.device)
        cross_variable = cross_variable.unsqueeze(0).to(self.device)
        target = feedback.target.unsqueeze(0).to(self.device)
        context = feedback.context.unsqueeze(0).to(self.device)
        adaptation_loss = self.loss_fn(feedback.prediction, feedback.target).detach().item()
        self.last_gradient_steps = self.last_egd_steps = 0
        for _ in range(self.n_inner):
            self._update(context, target, cross_time, cross_variable, groups)
        return adaptation_loss if groups else None

    @torch.no_grad()
    def reset_online_state(self) -> None:
        """Start online ensembling from equal weights, as in the official test loop."""

        self.model.long_logits.zero_()
        self.model.short_bias.zero_()
        self._pending_predictions.clear()

    def train_batch(self, context: Tensor, target: Tensor) -> float:
        """Train all OneNet components on one offline mini-batch."""

        context = context.to(self.device)
        target = target.to(self.device)
        self.model.train()
        cross_time, cross_variable = self.model.expert_predictions(context)
        loss = self.loss_fn(cross_time, target) + self.loss_fn(cross_variable, target)
        self.expert_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.expert_optimizer.step()
        self._update_ensemble(cross_time.detach(), cross_variable.detach(), target)
        return loss.detach().item()

    def _update(
        self,
        context: Tensor,
        target: Tensor,
        emitted_cross_time: Tensor,
        emitted_cross_variable: Tensor,
        groups: frozenset[str],
    ) -> None:
        self.model.train()
        self.expert_optimizer.zero_grad(set_to_none=True)
        if groups & {"time", "variable"}:
            losses = []
            if "time" in groups:
                losses.append(self.loss_fn(self.model.cross_time(context), target))
            if "variable" in groups:
                losses.append(self.loss_fn(self.model.cross_variable(context), target))
            sum(losses).backward()
            self.expert_optimizer.step()
            self.last_gradient_steps += 1
        self._update_ensemble(emitted_cross_time, emitted_cross_variable, target, train_decision="decision" in groups)

    def _update_ensemble(
        self,
        cross_time: Tensor,
        cross_variable: Tensor,
        target: Tensor,
        *,
        train_decision: bool = True,
    ) -> None:
        # The decision network learns a recent correction conditioned on the labeled forecast.
        self.decision_optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_decision):
            bias = self.model.decision_bias(cross_time, cross_variable, target)
            if train_decision:
                short_weight = torch.sigmoid(self.model.long_logits.view(1, -1) + bias)
                short_prediction = short_weight.unsqueeze(1) * cross_time + (1.0 - short_weight.unsqueeze(1)) * cross_variable
                self.loss_fn(short_prediction, target).backward()
                self.decision_optimizer.step()
                self.last_gradient_steps += 1
        self.model.short_bias.copy_(bias.detach().mean(dim=0))

        # Paper Eq. (2), with per-target horizon-mean squared error as expert loss.
        # For two experts, log(w_time / w_variable) has this exact EGD update.
        with torch.no_grad():
            time_loss = (cross_time - target).square().mean(dim=(0, 1))
            variable_loss = (cross_variable - target).square().mean(dim=(0, 1))
            self.model.long_logits.add_(self.weight_learning_rate * (variable_loss - time_loss))
            self.last_egd_steps += 1
