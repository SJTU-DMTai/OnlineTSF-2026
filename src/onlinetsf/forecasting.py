# -*- coding: utf-8 -*-
"""Minimal PyTorch forecasting backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# Every backbone shares this contract: batch dimension first, then time, then features.
def _validate_context(context: Tensor, context_length: int, num_features: int) -> None:
    if context.ndim != 3:
        raise ValueError("context must have shape [batch, context_length, num_features]")
    if context.shape[1:] != (context_length, num_features):
        raise ValueError(
            "context shape does not match the model configuration: "
            f"expected [batch, {context_length}, {num_features}]"
        )


class ForecastBackbone(nn.Module, ABC):
    """A forecasting model maps a context window to a multi-step forecast."""

    @abstractmethod
    def forward(self, context: Tensor) -> Tensor:
        """Return a tensor with shape ``[batch, horizon, num_targets]``."""


class LinearForecastBackbone(ForecastBackbone):
    """Direct linear multi-horizon baseline."""

    def __init__(
        self,
        context_length: int,
        num_features: int,
        horizon: int,
        num_targets: int | None = None,
    ) -> None:
        super().__init__()
        if context_length <= 0 or num_features <= 0 or horizon <= 0:
            raise ValueError("context_length, num_features, and horizon must be positive")

        self.context_length = context_length
        self.num_features = num_features
        self.horizon = horizon
        self.num_targets = num_targets if num_targets is not None else num_features
        if self.num_targets <= 0:
            raise ValueError("num_targets must be positive")

        # The linear baseline flattens all history and predicts every future step at once.
        self.projection = nn.Linear(
            context_length * num_features,
            horizon * self.num_targets,
        )

    def forward(self, context: Tensor) -> Tensor:
        _validate_context(context, self.context_length, self.num_features)
        forecast = self.projection(context.reshape(context.shape[0], -1))
        return forecast.reshape(context.shape[0], self.horizon, self.num_targets)


class TemporalBlock(nn.Module):
    """A residual pair of dilated causal convolutions."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # Right padding creates artificial future positions and is removed for causality.
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else nn.Identity()
        )

    # Conv1d pads both sides; removing the right tail prevents information from future positions.
    def _remove_right_padding(self, values: Tensor) -> Tensor:
        if self.padding == 0:
            return values
        return values[..., :-self.padding]

    def forward(self, values: Tensor) -> Tensor:
        output = self._remove_right_padding(self.conv1(values))
        output = self.dropout(self.activation(output))
        output = self._remove_right_padding(self.conv2(output))
        output = self.dropout(self.activation(output))
        return self.activation(output + self.residual(values))


class TCNForecastBackbone(ForecastBackbone):
    """Dilated causal-convolution backbone for multi-step forecasting."""

    def __init__(
        self,
        context_length: int,
        num_features: int,
        horizon: int,
        num_targets: int | None = None,
        channels: Sequence[int] = (32, 32, 32, 32, 32),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if context_length <= 0 or num_features <= 0 or horizon <= 0:
            raise ValueError("context_length, num_features, and horizon must be positive")
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive channel counts")
        if kernel_size <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("kernel_size must be positive and dropout must be in [0, 1)")

        self.context_length = context_length
        self.num_features = num_features
        self.horizon = horizon
        self.num_targets = num_targets if num_targets is not None else num_features
        if self.num_targets <= 0:
            raise ValueError("num_targets must be positive")

        # Dilation grows as 1, 2, 4, ... to expand the receptive field efficiently.
        blocks: list[nn.Module] = []
        input_channels = num_features
        for block_index, output_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    kernel_size=kernel_size,
                    dilation=2**block_index,
                    dropout=dropout,
                )
            )
            input_channels = output_channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(channels[-1], horizon * self.num_targets)

    def forward(self, context: Tensor) -> Tensor:
        _validate_context(context, self.context_length, self.num_features)
        # Conv1d consumes [batch, channels, time]; its last position summarizes visible history.
        encoded = self.network(context.transpose(1, 2))
        forecast = self.head(encoded[..., -1])
        return forecast.reshape(context.shape[0], self.horizon, self.num_targets)


class PatchTSTForecastBackbone(ForecastBackbone):
    """Channel-independent patch Transformer for multivariate forecasting."""

    def __init__(
        self,
        context_length: int,
        num_features: int,
        horizon: int,
        num_targets: int | None = None,
        target_indices: Sequence[int] | None = None,
        patch_length: int = 16,
        patch_stride: int = 8,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if context_length <= 0 or num_features <= 0 or horizon <= 0:
            raise ValueError("context_length, num_features, and horizon must be positive")
        if patch_length <= 0 or patch_length > context_length or patch_stride <= 0:
            raise ValueError("patch_length must be in [1, context_length] and patch_stride must be positive")
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0 or num_layers <= 0:
            raise ValueError("d_model must be divisible by num_heads and num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.context_length = context_length
        self.num_features = num_features
        self.horizon = horizon
        self.num_targets = num_targets if num_targets is not None else num_features
        if self.num_targets <= 0:
            raise ValueError("num_targets must be positive")

        selected_targets = tuple(target_indices) if target_indices is not None else tuple(range(self.num_targets))
        if len(selected_targets) != self.num_targets:
            raise ValueError("target_indices must contain exactly num_targets entries")
        if not selected_targets or min(selected_targets) < 0 or max(selected_targets) >= num_features:
            raise ValueError("target_indices contain an out-of-range feature index")
        # This is fixed configuration rather than learned state, so checkpoints need not store it.
        self.register_buffer("target_indices", torch.tensor(selected_targets, dtype=torch.long), persistent=False)

        self.patch_length = patch_length
        self.patch_stride = patch_stride
        # PatchTST splits every feature channel into its own sequence of time patches.
        self.num_patches = (context_length - patch_length + patch_stride - 1) // patch_stride + 1
        padded_length = (self.num_patches - 1) * patch_stride + patch_length
        self.right_padding = padded_length - context_length

        self.patch_embedding = nn.Linear(patch_length, d_model)
        self.position_embedding = nn.Parameter(torch.empty(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.normalization = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.num_patches * d_model, horizon)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, context: Tensor) -> Tensor:
        _validate_context(context, self.context_length, self.num_features)
        values = context.transpose(1, 2)
        # Replicating the last value completes the final patch without observing the future.
        if self.right_padding:
            values = F.pad(values, (0, self.right_padding), mode="replicate")
        # unfold produces overlapping local temporal segments.
        patches = values.unfold(dimension=2, size=self.patch_length, step=self.patch_stride)
        # Merge batch and feature dimensions to run one shared Transformer per feature channel.
        tokens = patches.reshape(
            context.shape[0] * self.num_features,
            self.num_patches,
            self.patch_length,
        )
        encoded = self.patch_embedding(tokens) + self.position_embedding
        encoded = self.normalization(self.encoder(encoded))
        per_channel_forecast = self.head(encoded.flatten(start_dim=1))
        forecast = per_channel_forecast.reshape(
            context.shape[0],
            self.num_features,
            self.horizon,
        ).transpose(1, 2)
        # Retain only the target channels requested by the caller.
        return forecast.index_select(dim=2, index=self.target_indices)
