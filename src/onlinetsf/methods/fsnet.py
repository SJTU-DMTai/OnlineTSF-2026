# -*- coding: utf-8 -*-
"""FSNet model and online update logic based on the official implementation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer

from ..forecasting import ForecastBackbone
from ..online import MethodFeedback


LossFunction = Callable[[Tensor, Tensor], Tensor]


class AdaptiveConv1d(nn.Module):
    """FSNet's same-padded convolution, gradient adapter, and memory."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        *,
        controller_hidden: int = 64,
        gradient_decay: float = 0.9,
        fast_decay: float = 0.3,
        memory_size: int = 32,
        memory_top_k: int = 2,
        memory_threshold: float = 0.75,
        memory_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, kernel_size, dilation, controller_hidden) <= 0:
            raise ValueError("channel counts, kernel_size, dilation, and controller_hidden must be positive")
        if out_channels % in_channels != 0:
            raise ValueError("out_channels must be divisible by in_channels")
        if not 0.0 <= fast_decay < gradient_decay < 1.0:
            raise ValueError("EMA decays must satisfy 0 <= fast_decay < gradient_decay < 1")
        if memory_size <= 0 or not 0 < memory_top_k <= memory_size:
            raise ValueError("memory_top_k must be in [1, memory_size]")
        if not 0.0 < memory_threshold < 1.0 or memory_temperature <= 0.0:
            raise ValueError("memory_threshold must be in (0, 1) and temperature must be positive")

        receptive_field = (kernel_size - 1) * dilation + 1
        self.padding = receptive_field // 2
        self.remove = 1 if receptive_field % 2 == 0 else 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.gradient_decay = gradient_decay
        self.fast_decay = fast_decay
        self.memory_top_k = memory_top_k
        self.memory_threshold = memory_threshold
        self.memory_temperature = memory_temperature

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.n_chunks = in_channels
        chunk_input_size = self.conv.weight.numel() // self.n_chunks
        output_ratio = out_channels // in_channels
        self.controller = nn.Sequential(
            nn.Linear(chunk_input_size, controller_hidden),
            nn.SiLU(),
        )
        self.weight_scale = nn.Linear(controller_hidden, kernel_size)
        self.bias_scale = nn.Linear(controller_hidden, output_ratio)
        self.feature_scale = nn.Linear(controller_hidden, output_ratio)

        adaptation_size = in_channels * kernel_size + 2 * out_channels
        memory = torch.empty(adaptation_size, memory_size)
        nn.init.xavier_uniform_(memory)
        memory.div_(max(1.0, torch.linalg.vector_norm(memory).item()))
        self.register_buffer("memory", memory)
        self.register_buffer("gradient_ema", torch.zeros(self.conv.weight.numel()))
        self.register_buffer("fast_gradient_ema", torch.zeros(self.conv.weight.numel()))
        self.register_buffer("adaptation_ema", torch.zeros(adaptation_size))
        self.register_buffer("adaptation_ema_initialized", torch.tensor(False))
        self.register_buffer("memory_trigger", torch.tensor(False))
        self.register_buffer("memory_interactions", torch.tensor(0, dtype=torch.long))

    def _adaptation(self, advance_state: bool) -> tuple[Tensor, Tensor, Tensor]:
        chunks = self.gradient_ema.view(self.n_chunks, -1)
        representation = self.controller(chunks)
        weight_scale = self.weight_scale(representation)
        bias_scale = self.bias_scale(representation)
        feature_scale = self.feature_scale(representation)
        adaptation = torch.cat(
            (weight_scale.flatten(), bias_scale.flatten(), feature_scale.flatten())
        )

        if advance_state:
            if self.adaptation_ema_initialized.item():
                self.adaptation_ema.mul_(self.fast_decay).add_(
                    adaptation.detach(), alpha=1.0 - self.fast_decay
                )
            else:
                self.adaptation_ema_initialized.fill_(True)
            query = self.adaptation_ema

            if self.memory_trigger.item():
                attention = F.softmax(
                    query @ self.memory / self.memory_temperature,
                    dim=0,
                )
                top_values, top_indices = torch.topk(attention, self.memory_top_k)
                selected_memory = self.memory.index_select(1, top_indices)
                recalled = selected_memory @ top_indices.to(selected_memory.dtype)

                sparse_attention = torch.zeros_like(attention)
                sparse_attention[top_indices] = top_values
                memory_update = recalled.unsqueeze(1) * sparse_attention.unsqueeze(0)
                self.memory[:, top_indices] = (
                    self.memory_threshold * selected_memory
                    + (1.0 - self.memory_threshold)
                    * memory_update.index_select(1, top_indices)
                )
                self.memory.div_(max(1.0, torch.linalg.vector_norm(self.memory).item()))
                adaptation = (
                    self.memory_threshold * adaptation
                    + (1.0 - self.memory_threshold) * recalled
                )
                self.memory_trigger.fill_(False)
                self.memory_interactions.add_(1)

        weight_end = self.in_channels * self.kernel_size
        bias_end = weight_end + self.out_channels
        return (
            adaptation[:weight_end].view(1, self.in_channels, self.kernel_size),
            adaptation[weight_end:bias_end],
            adaptation[bias_end:],
        )

    def forward(self, values: Tensor, *, advance_state: bool = True) -> Tensor:
        weight_scale, bias_scale, feature_scale = self._adaptation(advance_state)
        output = F.conv1d(
            values,
            self.conv.weight * weight_scale,
            self.bias * bias_scale,
            padding=self.padding,
            dilation=self.conv.dilation,
        )
        if self.remove:
            output = output[..., :-self.remove]
        return output * feature_scale.view(1, -1, 1)

    @torch.no_grad()
    def record_gradient(self) -> None:
        """Update official fast/slow gradient EMAs after backpropagation."""

        gradient = F.normalize(self.conv.weight.grad.detach()).flatten()
        self.fast_gradient_ema.mul_(self.fast_decay).add_(
            gradient, alpha=1.0 - self.fast_decay
        )
        if not self.training:
            similarity = F.cosine_similarity(
                self.fast_gradient_ema,
                self.gradient_ema,
                dim=0,
                eps=1e-6,
            )
            if similarity.item() < -self.memory_threshold:
                self.memory_trigger.fill_(True)
        self.gradient_ema.mul_(self.gradient_decay).add_(
            gradient, alpha=1.0 - self.gradient_decay
        )


class FSNetBlock(nn.Module):
    """Official two-convolution residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        *,
        final: bool = False,
        **adapter_options: float | int,
    ) -> None:
        super().__init__()
        self.conv1 = AdaptiveConv1d(
            in_channels, out_channels, kernel_size, dilation, **adapter_options
        )
        self.conv2 = AdaptiveConv1d(
            out_channels, out_channels, kernel_size, dilation, **adapter_options
        )
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, values: Tensor, *, advance_state: bool = True) -> Tensor:
        residual = values if self.projector is None else self.projector(values)
        output = self.conv1(F.gelu(values), advance_state=advance_state)
        output = self.conv2(F.gelu(output), advance_state=advance_state)
        return output + residual


class FSNetTCN(ForecastBackbone):
    """Official FSNet TCN encoder with a direct multi-horizon regressor."""

    def __init__(
        self,
        context_length: int,
        num_features: int,
        horizon: int,
        num_targets: int | None = None,
        *,
        hidden_channels: int = 64,
        depth: int = 10,
        representation_channels: int = 320,
        kernel_size: int = 3,
        representation_dropout: float = 0.1,
        controller_hidden: int = 64,
        gradient_decay: float = 0.9,
        fast_decay: float = 0.3,
        memory_size: int = 32,
        memory_top_k: int = 2,
        memory_threshold: float = 0.75,
        memory_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if context_length <= 0 or num_features <= 0 or horizon <= 0:
            raise ValueError("context_length, num_features, and horizon must be positive")
        if hidden_channels <= 0 or depth <= 0 or representation_channels <= 0 or kernel_size <= 0:
            raise ValueError("channel counts, depth, and kernel_size must be positive")
        if representation_channels % hidden_channels != 0:
            raise ValueError("representation_channels must be divisible by hidden_channels")
        if not 0.0 <= representation_dropout < 1.0:
            raise ValueError("representation_dropout must be in [0, 1)")

        self.context_length = context_length
        self.num_features = num_features
        self.horizon = horizon
        self.num_targets = num_targets if num_targets is not None else num_features
        if self.num_targets <= 0:
            raise ValueError("num_targets must be positive")

        self.input_projection = nn.Linear(num_features, hidden_channels)
        adapter_options: dict[str, float | int] = {
            "controller_hidden": controller_hidden,
            "gradient_decay": gradient_decay,
            "fast_decay": fast_decay,
            "memory_size": memory_size,
            "memory_top_k": memory_top_k,
            "memory_threshold": memory_threshold,
            "memory_temperature": memory_temperature,
        }
        channels = [hidden_channels] * depth + [representation_channels]
        self.blocks = nn.ModuleList(
            FSNetBlock(
                channels[index - 1] if index else hidden_channels,
                output_channels,
                kernel_size,
                2**index,
                final=index == len(channels) - 1,
                **adapter_options,
            )
            for index, output_channels in enumerate(channels)
        )
        self.representation_dropout = nn.Dropout(representation_dropout)
        self.head = nn.Linear(representation_channels, horizon * self.num_targets)

    def forward(self, context: Tensor, *, advance_state: bool = True) -> Tensor:
        if context.ndim != 3 or context.shape[1:] != (self.context_length, self.num_features):
            raise ValueError(
                "context shape does not match the model configuration: "
                f"expected [batch, {self.context_length}, {self.num_features}]"
            )
        valid_steps = ~context.isnan().any(dim=-1)
        values = context.masked_fill(~valid_steps.unsqueeze(-1), 0.0)
        encoded = self.input_projection(values).transpose(1, 2)
        for block in self.blocks:
            encoded = block(encoded, advance_state=advance_state)
        encoded = self.representation_dropout(encoded)
        forecast = self.head(encoded[..., -1])
        return forecast.reshape(context.shape[0], self.horizon, self.num_targets)

    def record_gradients(self) -> None:
        """Store gradients for every adapted convolution."""

        for module in self.modules():
            if isinstance(module, AdaptiveConv1d):
                module.record_gradient()


class FSNetMethod:
    """Own FSNet prediction and official-style online feedback updates."""

    def __init__(
        self,
        model: FSNetTCN,
        *,
        learning_rate: float = 1e-3,
        optimizer: Optimizer | None = None,
        loss_fn: LossFunction | None = None,
        n_inner: int = 1,
        device: torch.device | str | None = None,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if n_inner <= 0:
            raise ValueError("n_inner must be positive")
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.optimizer = optimizer or AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn or nn.MSELoss()
        self.n_inner = n_inner

    def predict(self, context: Tensor) -> Tensor:
        """Forecast one context using the current adapter and memory state."""

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(context.unsqueeze(0).to(self.device), advance_state=True)
        return prediction.squeeze(0)

    def on_feedback(self, feedback: MethodFeedback) -> float:
        """Apply official-style inner updates after complete feedback arrives."""

        if not feedback.observed_mask.all():
            raise ValueError("FSNetMethod requires complete-horizon feedback")
        self.model.eval()
        context = feedback.context.unsqueeze(0).to(self.device)
        target = feedback.target.unsqueeze(0).to(self.device)
        losses: list[float] = []
        for inner_step in range(self.n_inner):
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(context, advance_state=inner_step > 0)
            loss = self.loss_fn(prediction, target)
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            loss.backward()
            self.optimizer.step()
            self.model.record_gradients()
            losses.append(loss.detach().item())
        return sum(losses) / len(losses)
