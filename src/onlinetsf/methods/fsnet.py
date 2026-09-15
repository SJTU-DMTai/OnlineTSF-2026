# -*- coding: utf-8 -*-
"""Fast and Slow Network (FSNet) for online time-series forecasting.

The implementation follows the paper's two essential mechanisms: each TCN
convolution is modulated from a gradient EMA, and a second, faster EMA triggers
sparse reads and writes to a per-layer associative memory.
"""

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
    """A causal convolution with FSNet's gradient adapter and memory."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
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
        if channels <= 0 or kernel_size <= 0 or dilation <= 0 or controller_hidden <= 0:
            raise ValueError("channels, kernel_size, dilation, and controller_hidden must be positive")
        if not 0.0 <= fast_decay < gradient_decay < 1.0:
            raise ValueError("EMA decays must satisfy 0 <= fast_decay < gradient_decay < 1")
        if memory_size <= 0 or not 0 < memory_top_k <= memory_size:
            raise ValueError("memory_top_k must be in [1, memory_size]")
        if not 0.0 < memory_threshold < 1.0 or memory_temperature <= 0.0:
            raise ValueError("memory_threshold must be in (0, 1) and temperature must be positive")

        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.gradient_decay = gradient_decay
        self.fast_decay = fast_decay
        self.memory_top_k = memory_top_k
        self.memory_threshold = memory_threshold
        self.memory_temperature = memory_temperature

        self.weight = nn.Parameter(torch.empty(channels, channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        gradient_chunk_size = channels * kernel_size
        self.controller = nn.Sequential(
            nn.Linear(gradient_chunk_size, controller_hidden),
            nn.SiLU(),
        )
        self.weight_scale = nn.Linear(controller_hidden, kernel_size)
        self.bias_scale = nn.Linear(controller_hidden, 1)
        self.feature_scale = nn.Linear(controller_hidden, 1)
        for head in (self.weight_scale, self.bias_scale, self.feature_scale):
            nn.init.zeros_(head.weight)
            nn.init.ones_(head.bias)

        adaptation_size = channels * (kernel_size + 2)
        memory = torch.empty(memory_size, adaptation_size)
        nn.init.xavier_uniform_(memory)
        memory.div_(max(1.0, torch.linalg.vector_norm(memory).item()))
        self.register_buffer("memory", memory)
        self.register_buffer("gradient_ema", torch.zeros_like(self.weight).flatten())
        self.register_buffer("fast_gradient_ema", torch.zeros_like(self.weight).flatten())
        self.register_buffer("adaptation_ema", torch.zeros(adaptation_size))
        self.register_buffer("memory_trigger", torch.tensor(False))
        self.register_buffer("memory_interactions", torch.tensor(0, dtype=torch.long))

    def _adaptation(self, *, advance_state: bool) -> tuple[Tensor, Tensor, Tensor]:
        chunks = self.gradient_ema.view(self.channels, self.channels, self.kernel_size)
        chunks = chunks.permute(1, 0, 2).reshape(self.channels, -1)
        representation = self.controller(chunks)
        weight_scale = self.weight_scale(representation)
        bias_scale = self.bias_scale(representation).flatten()
        feature_scale = self.feature_scale(representation).flatten()
        adaptation = torch.cat((weight_scale.flatten(), bias_scale, feature_scale))

        if advance_state:
            self.adaptation_ema.mul_(self.fast_decay).add_(
                adaptation.detach(), alpha=1.0 - self.fast_decay
            )
            if self.memory_trigger.item():
                scores = self.memory @ self.adaptation_ema
                attention = F.softmax(scores / self.memory_temperature, dim=0)
                top_values, top_indices = torch.topk(attention, self.memory_top_k)
                recalled = (self.memory.index_select(0, top_indices) * top_values.unsqueeze(1)).sum(dim=0)

                sparse_attention = torch.zeros_like(attention)
                sparse_attention[top_indices] = top_values
                self.memory.mul_(self.memory_threshold).add_(
                    sparse_attention.unsqueeze(1) * self.adaptation_ema.unsqueeze(0),
                    alpha=1.0 - self.memory_threshold,
                )
                self.memory.div_(max(1.0, torch.linalg.vector_norm(self.memory).item()))
                adaptation = (
                    self.memory_threshold * adaptation
                    + (1.0 - self.memory_threshold) * recalled
                )
                self.memory_trigger.fill_(False)
                self.memory_interactions.add_(1)

        weight_end = self.channels * self.kernel_size
        bias_end = weight_end + self.channels
        return (
            adaptation[:weight_end].view(1, self.channels, self.kernel_size),
            adaptation[weight_end:bias_end],
            adaptation[bias_end:],
        )

    def forward(self, values: Tensor, *, advance_state: bool = False) -> Tensor:
        weight_scale, bias_scale, feature_scale = self._adaptation(advance_state=advance_state)
        output = F.conv1d(
            values,
            self.weight * weight_scale,
            self.bias * bias_scale,
            padding=self.padding,
            dilation=self.dilation,
        )
        if self.padding:
            output = output[..., :-self.padding]
        return output * feature_scale.view(1, -1, 1)

    @torch.no_grad()
    def record_gradient(self) -> None:
        """Update both gradient EMAs and arm the next memory interaction."""

        gradient = F.normalize(self.weight.grad.detach().flatten(), dim=0)
        self.fast_gradient_ema.mul_(self.fast_decay).add_(gradient, alpha=1.0 - self.fast_decay)
        similarity = F.cosine_similarity(
            self.fast_gradient_ema,
            self.gradient_ema,
            dim=0,
            eps=1e-6,
        )
        if similarity.item() < -self.memory_threshold:
            self.memory_trigger.fill_(True)
        self.gradient_ema.mul_(self.gradient_decay).add_(gradient, alpha=1.0 - self.gradient_decay)


class FSNetBlock(nn.Module):
    """Residual pair of FSNet adaptive convolutions."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, **adapter_options: float | int) -> None:
        super().__init__()
        self.conv1 = AdaptiveConv1d(channels, kernel_size, dilation, **adapter_options)
        self.conv2 = AdaptiveConv1d(channels, kernel_size, dilation, **adapter_options)

    def forward(self, values: Tensor, *, advance_state: bool = False) -> Tensor:
        output = self.conv1(F.gelu(values), advance_state=advance_state)
        output = self.conv2(F.gelu(output), advance_state=advance_state)
        return output + values


class FSNetTCN(ForecastBackbone):
    """TCN forecaster equipped with FSNet adapters on every convolution."""

    def __init__(
        self,
        context_length: int,
        num_features: int,
        horizon: int,
        num_targets: int | None = None,
        *,
        hidden_channels: int = 64,
        depth: int = 10,
        kernel_size: int = 3,
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
        if hidden_channels <= 0 or depth <= 0 or kernel_size <= 0:
            raise ValueError("hidden_channels, depth, and kernel_size must be positive")

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
        self.blocks = nn.ModuleList(
            FSNetBlock(hidden_channels, kernel_size, 2**index, **adapter_options)
            for index in range(depth)
        )
        self.head = nn.Linear(hidden_channels, horizon * self.num_targets)

    def forward(self, context: Tensor, *, advance_state: bool = False) -> Tensor:
        if context.ndim != 3 or context.shape[1:] != (self.context_length, self.num_features):
            raise ValueError(
                "context shape does not match the model configuration: "
                f"expected [batch, {self.context_length}, {self.num_features}]"
            )
        encoded = self.input_projection(context).transpose(1, 2)
        for block in self.blocks:
            encoded = block(encoded, advance_state=advance_state)
        forecast = self.head(encoded[..., -1])
        return forecast.reshape(context.shape[0], self.horizon, self.num_targets)

    def record_gradients(self) -> None:
        """Update adapter state after loss backpropagation."""

        for module in self.modules():
            if isinstance(module, AdaptiveConv1d):
                module.record_gradient()


class FSNetMethod:
    """Own FSNet prediction and feedback updates behind the online lifecycle."""

    def __init__(
        self,
        model: FSNetTCN,
        *,
        learning_rate: float = 1e-3,
        optimizer: Optimizer | None = None,
        loss_fn: LossFunction | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.optimizer = optimizer or AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn or nn.MSELoss()

    def predict(self, context: Tensor) -> Tensor:
        """Forecast one context and advance FSNet's prediction-time state."""

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(context.unsqueeze(0).to(self.device), advance_state=True)
        return prediction.squeeze(0)

    def on_feedback(self, feedback: MethodFeedback) -> float:
        """Learn from one newly available target and update FSNet gradient state."""

        self.model.train()
        context = feedback.context.unsqueeze(0).to(self.device)
        target = feedback.target.unsqueeze(0).to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(context, advance_state=False)
        loss = self.loss_fn(prediction, target)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar tensor")
        loss.backward()
        self.model.record_gradients()
        self.optimizer.step()
        return loss.detach().item()
