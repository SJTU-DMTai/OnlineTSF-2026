# -*- coding: utf-8 -*-
"""Ordinary online gradient descent for forecasting backbones."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from ..online import MethodFeedback


LossFunction = Callable[[Tensor, Tensor], Tensor]


class OGDMethod:
    """Adapt a standard PyTorch forecaster only when feedback is available."""

    update_group_names = ("model",)
    incremental_feedback = True
    adaptation_feature_names = ()
    last_egd_steps = 0

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: Optimizer | None = None,
        loss_fn: LossFunction | None = None,
        update_steps: int = 1,
        device: torch.device | str | None = None,
    ) -> None:
        if update_steps <= 0:
            raise ValueError("update_steps must be positive")
        if (optimizer is None) != (loss_fn is None):
            raise ValueError("optimizer and loss_fn must be provided together")

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.update_steps = update_steps
        self.last_gradient_steps = 0
        self.device = torch.device(device) if device is not None else self._model_device()
        self.model.to(self.device)

    def _model_device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.model.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

    @property
    def update_optimizers(self) -> tuple[Optimizer, ...]:
        return (self.optimizer,) if self.optimizer is not None else ()

    def predict(self, context: Tensor, *, advance_state: bool = True) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(context.unsqueeze(0).to(self.device))
        return prediction.squeeze(0)

    def discard_feedback(self, feedback: MethodFeedback) -> None:
        pass

    def expert_errors(self, target) -> None:
        return None

    def adaptation_features(self, target=None, *, current_context=None) -> dict:
        return {}

    def on_feedback(self, feedback: MethodFeedback, *, update_groups: frozenset[str] | None = None) -> float | None:
        self.last_gradient_steps = 0
        if update_groups is not None:
            if update_groups - {"model"}:
                raise ValueError("unknown OGD update group")
            if not update_groups:
                return None
        if self.optimizer is None or self.loss_fn is None:
            return None

        self.model.train()
        context = feedback.context.unsqueeze(0).to(self.device)
        target = feedback.target.unsqueeze(0).to(self.device)
        observed_mask = feedback.observed_mask.unsqueeze(0).to(self.device)
        losses: list[float] = []
        for _ in range(self.update_steps):
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(context)
            loss = self.loss_fn(prediction[observed_mask], target[observed_mask])
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            loss.backward()
            self.optimizer.step()
            self.last_gradient_steps += 1
            losses.append(loss.detach().item())
        return sum(losses) / len(losses)
