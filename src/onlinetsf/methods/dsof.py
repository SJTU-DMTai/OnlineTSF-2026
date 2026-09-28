# -*- coding: utf-8 -*-
"""DSOF residual model and causal fast/slow online updates."""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor, nn

from ..online import MethodFeedback


class DSOFModel(nn.Module):
    """A forecasting teacher plus a channel-wise MLP residual student."""

    def __init__(
        self,
        teacher: nn.Module,
        *,
        context_length: int,
        horizon: int,
        target_indices: tuple[int, ...],
        student_hidden: int = 16,
        student_depth: int = 3,
        student_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if student_hidden <= 0 or student_depth < 2 or not 0.0 <= student_dropout < 1.0:
            raise ValueError("invalid student MLP dimensions or dropout")
        self.teacher = teacher
        self.horizon = horizon
        self.target_indices = target_indices
        layers: list[nn.Module] = [nn.Linear(context_length + horizon, student_hidden)]
        for _ in range(student_depth - 2):
            layers.extend((nn.Identity(), nn.Dropout(student_dropout), nn.Linear(student_hidden, student_hidden)))
        layers.extend((nn.Identity(), nn.Dropout(student_dropout), nn.Linear(student_hidden, horizon)))
        self.student = nn.Sequential(*layers)

    def residual(self, context: Tensor, teacher_prediction: Tensor) -> Tensor:
        history = context[:, :, self.target_indices]
        student_input = torch.cat((history, teacher_prediction.detach()), dim=1).transpose(1, 2)
        return self.student(student_input).transpose(1, 2)

    def forward(self, context: Tensor) -> Tensor:
        teacher_prediction = self.teacher(context)
        return teacher_prediction + self.residual(context, teacher_prediction)


class DSOFMethod:
    """Replay complete windows, then adapt the student from one-step TD labels."""

    def __init__(
        self,
        model: DSOFModel,
        *,
        learning_rate: float = 1e-3,
        student_learning_rate: float = 1e-3,
        online_learning_rate: float = 1e-3,
        replay_buffer_size: int = 300,
        replay_batch_size: int = 32,
        replay_epochs: int = 1,
        replay_frequency: int = 1,
        discount: float = 0.9,
        device: torch.device | str | None = None,
    ) -> None:
        if min(learning_rate, student_learning_rate, online_learning_rate) <= 0:
            raise ValueError("DSOF learning rates must be positive")
        if min(replay_buffer_size, replay_batch_size, replay_epochs, replay_frequency) <= 0:
            raise ValueError("DSOF replay sizes, epochs, and frequency must be positive")
        if replay_batch_size > replay_buffer_size:
            raise ValueError("replay_batch_size cannot exceed replay_buffer_size")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.teacher_optimizer = torch.optim.AdamW(model.teacher.parameters(), lr=learning_rate)
        self.student_optimizer = torch.optim.AdamW(model.student.parameters(), lr=student_learning_rate)
        self.online_optimizer = torch.optim.AdamW(model.student.parameters(), lr=online_learning_rate)
        self.replay: deque[tuple[Tensor, Tensor]] = deque(maxlen=replay_buffer_size)
        self.replay_batch_size = replay_batch_size
        self.replay_epochs = replay_epochs
        self.replay_frequency = replay_frequency
        self.discount = discount
        self._complete_count = 0
        self._partial: dict[int, tuple[Tensor, Tensor]] = {}

    def predict(self, context: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(context.unsqueeze(0).to(self.device)).squeeze(0)

    def train_batch(self, context: Tensor, target: Tensor) -> float:
        """Supervise the teacher and residual student during initial training."""
        return self._supervised_update(context.to(self.device), target.to(self.device))

    def reset_online_state(self) -> None:
        """Start the online phase with empty replay and fresh optimizer moments."""
        self.replay.clear()
        self._partial.clear()
        self._complete_count = 0
        self.teacher_optimizer.state.clear()
        self.student_optimizer.state.clear()
        self.online_optimizer.state.clear()

    def _supervised_update(self, context: Tensor, target: Tensor) -> float:
        self.model.train()
        teacher_prediction = self.model.teacher(context)
        final_prediction = teacher_prediction.detach() + self.model.residual(context, teacher_prediction)
        teacher_loss = (teacher_prediction - target).square().mean()
        student_loss = (final_prediction - target).square().mean()
        self.teacher_optimizer.zero_grad(set_to_none=True)
        self.student_optimizer.zero_grad(set_to_none=True)
        teacher_loss.backward()
        student_loss.backward()
        self.teacher_optimizer.step()
        self.student_optimizer.step()
        return student_loss.detach().item()

    def _replay_update(self) -> float | None:
        if len(self.replay) < self.replay_batch_size:
            return None
        self._complete_count += 1
        if (self._complete_count - 1) % self.replay_frequency:
            return None
        losses: list[float] = []
        for _ in range(self.replay_epochs):
            indices = torch.randperm(len(self.replay))[:self.replay_batch_size].tolist()
            context = torch.stack([self.replay[index][0] for index in indices]).to(self.device)
            target = torch.stack([self.replay[index][1] for index in indices]).to(self.device)
            losses.append(self._supervised_update(context, target))
        return sum(losses) / len(losses)

    def _fast_update(self, feedback: MethodFeedback) -> float:
        self.model.teacher.eval()
        self.model.student.train()
        previous_context = feedback.context.unsqueeze(0).to(self.device)
        current_context = feedback.current_context.unsqueeze(0).to(self.device)
        with torch.no_grad():
            previous_teacher = self.model.teacher(previous_context)
            current_teacher = self.model.teacher(current_context)
            pseudo_target = torch.cat((feedback.target[:1].unsqueeze(0).to(self.device), current_teacher[:, :-1]), dim=1)
        prediction = previous_teacher + self.model.residual(previous_context, previous_teacher)
        weights = self.discount ** torch.arange(self.model.horizon, device=self.device, dtype=prediction.dtype)
        loss = ((prediction - pseudo_target).square() * weights.view(1, -1, 1)).mean()
        self.online_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.online_optimizer.step()
        return loss.detach().item()

    def on_feedback(self, feedback: MethodFeedback) -> float | None:
        # Flush scores final forecasts after replay; it has no new observed context.
        if feedback.current_context is None:
            return None
        partial = self._partial.get(feedback.index)
        if partial is None:
            partial = (torch.zeros_like(feedback.target), torch.zeros_like(feedback.observed_mask))
            self._partial[feedback.index] = partial
        values, observed = partial
        values[feedback.observed_mask] = feedback.target[feedback.observed_mask]
        observed |= feedback.observed_mask

        losses: list[float] = []
        if observed.all():
            self.replay.append((feedback.context.detach().clone(), values.detach().clone()))
            del self._partial[feedback.index]
            replay_loss = self._replay_update()
            if replay_loss is not None:
                losses.append(replay_loss)
        if feedback.observed_mask[0].all():
            losses.append(self._fast_update(feedback))
        return sum(losses) / len(losses) if losses else None
