# -*- coding: utf-8 -*-
"""Statistics and interval operations used by stable-interval selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class DistributionMetrics:
    """Normalized change between the first and second half of a data window."""

    mean_change: float
    std_change: float
    quantile_change: float


@dataclass(frozen=True)
class LossMetrics:
    """Level and variability of one model's loss inside a candidate window."""

    mean: float
    coefficient_of_variation: float


@dataclass(frozen=True)
class CandidateWindow:
    """One candidate window evaluated by all conditions."""

    sample_start: int
    sample_end: int
    raw_start: int
    raw_end: int
    detector_ok: bool
    distribution_ok: bool
    loss_ok: bool

    @property
    def stable(self) -> bool:
        return self.detector_ok and self.distribution_ok and self.loss_ok


@dataclass(frozen=True)
class StableInterval:
    """Consecutive stable candidate windows merged into one raw interval."""

    sample_start: int
    sample_end: int
    raw_start: int
    raw_end: int
    window_count: int

    @property
    def raw_length(self) -> int:
        return self.raw_end - self.raw_start


def distribution_metrics(
    values: Tensor,
    *,
    feature_quantile: float = 0.95,
    eps: float = 1e-6,
) -> DistributionMetrics:
    """Compare means, standard deviations and quartiles across two half-windows."""

    if values.ndim != 2 or values.shape[0] < 4:
        raise ValueError("values must have shape [time, features] with at least four rows")
    if not 0.0 < feature_quantile <= 1.0:
        raise ValueError("feature_quantile must be in (0, 1]")

    values = values.to(dtype=torch.float64)
    middle = values.shape[0] // 2
    left = values[:middle]
    right = values[middle:]
    scale = values.std(dim=0, unbiased=False).clamp_min(eps)

    mean_change = (left.mean(dim=0) - right.mean(dim=0)).abs() / scale
    std_change = (
        left.std(dim=0, unbiased=False) - right.std(dim=0, unbiased=False)
    ).abs() / scale
    probabilities = torch.tensor((0.25, 0.5, 0.75), dtype=values.dtype)
    left_quantiles = torch.quantile(left, probabilities, dim=0)
    right_quantiles = torch.quantile(right, probabilities, dim=0)
    quantile_change = ((left_quantiles - right_quantiles).abs() / scale).amax(dim=0)

    return DistributionMetrics(
        mean_change=torch.quantile(mean_change, feature_quantile).item(),
        std_change=torch.quantile(std_change, feature_quantile).item(),
        quantile_change=torch.quantile(quantile_change, feature_quantile).item(),
    )


def loss_metrics(
    losses: Sequence[float], *, block_size: int = 1, eps: float = 1e-12
) -> LossMetrics:
    """Return mean loss and CV of consecutive block-mean losses."""

    if not losses:
        raise ValueError("losses must not be empty")
    if block_size <= 0 or len(losses) < 2 * block_size:
        raise ValueError("losses must contain at least two blocks")
    tensor = torch.tensor(losses, dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError("losses must be finite")
    mean = tensor.mean()
    blocks = list(tensor.split(block_size))
    if blocks[-1].numel() < block_size:
        blocks[-2] = torch.cat((blocks[-2], blocks[-1]))
        blocks.pop()
    block_means = torch.tensor([block.mean().item() for block in blocks], dtype=torch.float64)
    coefficient = block_means.std(unbiased=False) / mean.abs().clamp_min(eps)
    return LossMetrics(mean=mean.item(), coefficient_of_variation=coefficient.item())


def quantile_threshold(values: Sequence[float], quantile: float) -> float:
    """Compute a deterministic empirical threshold for low-loss windows."""

    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    return torch.quantile(torch.tensor(values, dtype=torch.float64), quantile).item()


def merge_stable_windows(
    windows: Sequence[CandidateWindow],
    *,
    minimum_raw_length: int,
) -> list[StableInterval]:
    """Merge consecutive overlapping AND-passing windows, then filter length."""

    if minimum_raw_length < 0:
        raise ValueError("minimum_raw_length must be non-negative")

    intervals: list[StableInterval] = []
    current: StableInterval | None = None
    for window in windows:
        if not window.stable:
            if current is not None and current.raw_length > minimum_raw_length:
                intervals.append(current)
            current = None
            continue

        if current is not None and window.sample_start <= current.sample_end:
            current = StableInterval(
                sample_start=current.sample_start,
                sample_end=max(current.sample_end, window.sample_end),
                raw_start=current.raw_start,
                raw_end=max(current.raw_end, window.raw_end),
                window_count=current.window_count + 1,
            )
        else:
            if current is not None and current.raw_length > minimum_raw_length:
                intervals.append(current)
            current = StableInterval(
                sample_start=window.sample_start,
                sample_end=window.sample_end,
                raw_start=window.raw_start,
                raw_end=window.raw_end,
                window_count=1,
            )

    if current is not None and current.raw_length > minimum_raw_length:
        intervals.append(current)
    return intervals


def condition_counts(windows: Sequence[CandidateWindow]) -> Mapping[str, int]:
    """Count individual condition passes and their final AND intersection."""

    return {
        "total": len(windows),
        "detector": sum(window.detector_ok for window in windows),
        "distribution": sum(window.distribution_ok for window in windows),
        "loss": sum(window.loss_ok for window in windows),
        "all": sum(window.stable for window in windows),
    }


def covered_raw_length(intervals: Sequence[StableInterval]) -> int:
    """Return union length even if raw intervals overlap because of lookback."""

    if not intervals:
        return 0
    ordered = sorted(intervals, key=lambda interval: interval.raw_start)
    total = 0
    start = ordered[0].raw_start
    end = ordered[0].raw_end
    for interval in ordered[1:]:
        if interval.raw_start <= end:
            end = max(end, interval.raw_end)
        else:
            total += end - start
            start = interval.raw_start
            end = interval.raw_end
    return total + end - start
