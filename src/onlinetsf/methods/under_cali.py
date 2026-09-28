# -*- coding: utf-8 -*-
"""Under-Cali calibration for a frozen forecasting backbone.

Based on Wen et al., KDD 2026, and the authors' public implementation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from ..online import MethodFeedback


class GatedTemporalCalibrator(nn.Module):
    """Variable-wise temporal correction with a small residual gate."""

    def __init__(self, length: int, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(length, length, channels))
        self.bias = nn.Parameter(torch.zeros(length, channels))
        self.gate = nn.Parameter(torch.full((channels,), 0.01))
        self.mlp = nn.Sequential(
            nn.Linear(length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, length),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, values: Tensor) -> Tensor:
        temporal = values + torch.tanh(self.gate) * (
            torch.einsum("bic,ijc->bjc", values, self.weight) + self.bias
        )
        correction = self.mlp(temporal.transpose(1, 2)).transpose(1, 2)
        return values + torch.tanh(self.gate) * correction


class CalibrationExpert(nn.Module):
    def __init__(
        self, context_length: int, horizon: int, num_features: int,
        num_targets: int, hidden_dim: int,
    ) -> None:
        super().__init__()
        self.input = GatedTemporalCalibrator(context_length, num_features, hidden_dim)
        self.output = GatedTemporalCalibrator(horizon, num_targets, hidden_dim)

    def forward(self, context: Tensor, source: nn.Module) -> Tensor:
        return self.output(source(self.input(context)))


class UncertaintyEstimator(nn.Module):
    """Predict normalized forecast error from the context and reliable forecast."""

    def __init__(
        self, context_length: int, horizon: int, num_features: int,
        num_targets: int, hidden_dim: int,
    ) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_length * num_features, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.forecast_encoder = nn.Sequential(
            nn.Linear(horizon * num_targets, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(nn.Linear(2 * hidden_dim, 2 * hidden_dim), nn.ReLU())
        self.error_head = nn.Sequential(nn.Linear(2 * hidden_dim, num_targets), nn.Sigmoid())
        self.weight_head = nn.Sequential(nn.Linear(2 * hidden_dim, num_targets), nn.Sigmoid())
        self.register_buffer("error_min", torch.tensor(0.0))
        self.register_buffer("error_max", torch.tensor(1.0))

    def forward(self, context: Tensor, forecast: Tensor) -> Tensor:
        features = torch.cat(
            (
                self.context_encoder(context.flatten(start_dim=1)),
                self.forecast_encoder(forecast.flatten(start_dim=1)),
            ),
            dim=1,
        )
        fused = self.fusion(features)
        error = self.error_head(fused)
        weight = self.weight_head(fused)
        return (error * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-8)


class UnderCaliMethod:
    """Offline uncertainty pretraining and delayed-feedback dual-expert adaptation."""

    def __init__(
        self,
        model: nn.Module,
        *,
        context_length: int,
        horizon: int,
        num_features: int,
        num_targets: int,
        learning_rate: float,
        adapt_lr: float = 1e-3,
        uncertainty_lr: float = 1e-4,
        uncertainty_adapt_lr: float = 1e-4,
        calibrator_hidden_dim: int = 64,
        uncertainty_hidden_dim: int = 128,
        uncertainty_epochs: int = 20,
        update_steps: int = 5,
        allocation_alpha: float = 0.75,
        allocation_std_k: float = 0.25,
        trigger_alpha: float = 0.25,
        trigger_std_k: float = 0.75,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.reliable = CalibrationExpert(
            context_length, horizon, num_features, num_targets, calibrator_hidden_dim
        ).to(self.device)
        self.unreliable = CalibrationExpert(
            context_length, horizon, num_features, num_targets, calibrator_hidden_dim
        ).to(self.device)
        self.uncertainty = UncertaintyEstimator(
            context_length, horizon, num_features, num_targets, uncertainty_hidden_dim
        ).to(self.device)
        self.reliable_optimizer = torch.optim.Adam(self.reliable.parameters(), lr=adapt_lr)
        self.unreliable_optimizer = torch.optim.Adam(self.unreliable.parameters(), lr=adapt_lr * 0.5)
        self.uncertainty_optimizer = torch.optim.Adam(
            self.uncertainty.parameters(), lr=uncertainty_lr
        )
        self.uncertainty_adapt_lr = uncertainty_adapt_lr
        self.uncertainty_epochs = uncertainty_epochs
        self.update_steps = update_steps
        self.allocation_alpha = allocation_alpha
        self.allocation_std_k = allocation_std_k
        self.trigger_alpha = trigger_alpha
        self.trigger_std_k = trigger_std_k
        self.allocation_mean: float | None = None
        self.allocation_variance = 0.0
        self.trigger_mean: float | None = None
        self.trigger_variance = 0.0
        self._last_route: tuple[float, bool] | None = None
        self._pending_routes: dict[int, tuple[float, bool]] = {}

    def pretrain_uncertainty(
        self, dataset: Sequence[tuple[Tensor, Tensor]], train_size: int, batch_size: int
    ) -> None:
        """Fit error normalization and UE using only the offline prefix."""

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        loader = DataLoader(Subset(dataset, range(train_size)), batch_size=batch_size)
        error_min = math.inf
        error_max = -math.inf
        with torch.no_grad():
            for context, target in loader:
                prediction = self.model(context.to(self.device))
                errors = (prediction - target.to(self.device)).square().mean(dim=(1, 2))
                error_min = min(error_min, errors.min().item())
                error_max = max(error_max, errors.max().item())
        self.uncertainty.error_min.fill_(error_min)
        self.uncertainty.error_max.fill_(error_max)

        for _ in range(self.uncertainty_epochs):
            self.uncertainty.train()
            for context, target in loader:
                context = context.to(self.device)
                target = target.to(self.device)
                with torch.no_grad():
                    prediction = self.model(context)
                    errors = (prediction - target).square().mean(dim=(1, 2))
                    normalized = ((errors - self.uncertainty.error_min) /
                                  (self.uncertainty.error_max - self.uncertainty.error_min).clamp_min(1e-8))
                    normalized = normalized.clamp(0.0, 1.0)
                self.uncertainty_optimizer.zero_grad(set_to_none=True)
                loss = F.l1_loss(self.uncertainty(context, prediction), normalized)
                loss.backward()
                self.uncertainty_optimizer.step()
        self.uncertainty.eval()
        self.uncertainty_optimizer = torch.optim.Adam(
            self.uncertainty.parameters(), lr=self.uncertainty_adapt_lr
        )

    def predict(self, context: Tensor) -> Tensor:
        self.model.eval()
        self.reliable.eval()
        self.unreliable.eval()
        self.uncertainty.eval()
        context = context.unsqueeze(0).to(self.device)
        with torch.no_grad():
            reliable_forecast = self.reliable(context, self.model)
            score = self.uncertainty(context, reliable_forecast).item()
            if self.allocation_mean is None:
                self.allocation_mean = score
            else:
                delta = score - self.allocation_mean
                self.allocation_mean += self.allocation_alpha * delta
                self.allocation_variance = (
                    (1 - self.allocation_alpha) * self.allocation_variance
                    + self.allocation_alpha * delta * delta
                )
            threshold = self.allocation_mean + self.allocation_std_k * math.sqrt(
                self.allocation_variance
            )
            reliable = score < threshold
            prediction = (
                reliable_forecast if reliable else self.unreliable(context, self.model)
            )
        self._last_route = (score, reliable)
        return prediction.squeeze(0)

    def on_emission(self, index: int) -> None:
        self._pending_routes[index] = self._last_route

    def on_feedback(self, feedback: MethodFeedback) -> float | None:
        score, reliable = self._pending_routes.pop(feedback.index)
        if self.trigger_mean is None:
            should_adapt = False
            self.trigger_mean = score
        else:
            threshold = self.trigger_mean + self.trigger_std_k * math.sqrt(
                self.trigger_variance
            )
            should_adapt = score > threshold
            delta = score - self.trigger_mean
            self.trigger_mean += self.trigger_alpha * delta
            self.trigger_variance = (
                (1 - self.trigger_alpha) * self.trigger_variance
                + self.trigger_alpha * delta * delta
            )
        if not should_adapt:
            return None

        context = feedback.context.unsqueeze(0).to(self.device)
        target = feedback.target.unsqueeze(0).to(self.device)
        mask = feedback.observed_mask.unsqueeze(0).to(self.device)
        expert = self.reliable if reliable else self.unreliable
        optimizer = self.reliable_optimizer if reliable else self.unreliable_optimizer
        expert.train()
        losses: list[float] = []
        for _ in range(self.update_steps):
            optimizer.zero_grad(set_to_none=True)
            forecast = expert(context, self.model)
            loss = F.mse_loss(forecast[mask], target[mask])
            loss.backward()
            nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.detach().item())
        expert.eval()

        if reliable:
            self.uncertainty.train()
            with torch.no_grad():
                forecast = self.reliable(context, self.model)
                errors = ((forecast - target).masked_fill(~mask, 0.0).square()).mean(
                    dim=(1, 2)
                )
                normalized = ((errors - self.uncertainty.error_min) /
                              (self.uncertainty.error_max - self.uncertainty.error_min).clamp_min(1e-8))
                normalized = normalized.clamp(0.0, 1.0)
            for _ in range(self.update_steps):
                self.uncertainty_optimizer.zero_grad(set_to_none=True)
                loss = F.l1_loss(self.uncertainty(context, forecast), normalized)
                loss.backward()
                nn.utils.clip_grad_norm_(self.uncertainty.parameters(), 1.0)
                self.uncertainty_optimizer.step()
            self.uncertainty.eval()
        return sum(losses) / len(losses)
