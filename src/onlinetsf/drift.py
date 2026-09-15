# -*- coding: utf-8 -*-
"""Residual-based concept drift detection."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DriftUpdate:
    """The detector state after consuming one scalar signal."""

    detected: bool
    value: float
    mean: float
    score: float


class PageHinkleyDetector:
    """Detect persistent upward shifts in a scalar error stream.

    Feed this detector a scalar residual statistic, such as per-forecast MAE..
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        min_instances: int = 30,
    ) -> None:
        if delta < 0 or threshold <= 0 or min_instances <= 0:
            raise ValueError("delta must be non-negative; threshold and min_instances must be positive")
        self.delta = delta
        self.threshold = threshold
        self.min_instances = min_instances
        self.reset()

    def reset(self) -> None:
        """Clear detector statistics after a caller handles a drift event."""

        self.num_observations = 0
        self.mean = 0.0
        self.cumulative_sum = 0.0
        self.minimum_cumulative_sum = 0.0

    def update(self, value: float) -> DriftUpdate:
        """Consume one residual statistic and return the current drift status."""

        value = float(value)
        if not math.isfinite(value):
            raise ValueError("drift signal must be finite")

        self.num_observations += 1
        self.mean += (value - self.mean) / self.num_observations
        self.cumulative_sum += value - self.mean - self.delta
        self.minimum_cumulative_sum = min(self.minimum_cumulative_sum, self.cumulative_sum)
        score = self.cumulative_sum - self.minimum_cumulative_sum
        detected = self.num_observations >= self.min_instances and score > self.threshold
        return DriftUpdate(detected=detected, value=value, mean=self.mean, score=score)


class ADWINDetector:
    """Adaptive windowing detector for bounded scalar signals in ``[0, 1]``.

    The detector compares all valid cuts in its current window using an
    ADWIN-style Hoeffding bound.  After detection it retains only the recent
    segment, so subsequent updates adapt to the new distribution.
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_window_length: int = 5,
        max_window_length: int = 512,
    ) -> None:
        if not 0 < delta < 1:
            raise ValueError("delta must be between zero and one")
        if min_window_length <= 0 or max_window_length < 2 * min_window_length:
            raise ValueError("max_window_length must hold two minimum-length windows")
        self.delta = delta
        self.min_window_length = min_window_length
        self.max_window_length = max_window_length
        self.reset()

    def reset(self) -> None:
        """Clear the adaptive window."""

        self.window: list[float] = []
        self.num_observations = 0

    def update(self, value: float) -> DriftUpdate:
        """Consume one normalized signal and return the current drift status."""

        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("ADWIN requires a finite signal in the range [0, 1]")

        self.window.append(value)
        self.num_observations += 1
        if len(self.window) > self.max_window_length:
            self.window.pop(0)

        detected = False
        score = 0.0
        if len(self.window) >= 2 * self.min_window_length:
            total = sum(self.window)
            left_sum = 0.0
            window_length = len(self.window)
            for cut in range(1, window_length):
                left_sum += self.window[cut - 1]
                right_length = window_length - cut
                if cut < self.min_window_length or right_length < self.min_window_length:
                    continue

                left_mean = left_sum / cut
                right_mean = (total - left_sum) / right_length
                epsilon = math.sqrt(
                    0.5 * (1.0 / cut + 1.0 / right_length) * math.log(4.0 / self.delta)
                )
                excess = abs(right_mean - left_mean) - epsilon
                if excess > score:
                    score = excess
                if excess > 0.0:
                    self.window = self.window[cut:]
                    detected = True
                    break

        mean = sum(self.window) / len(self.window)
        return DriftUpdate(detected=detected, value=value, mean=mean, score=score)


class KSWINDetector:
    """Kolmogorov-Smirnov windowing detector for scalar residuals."""

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int = 0,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between zero and one")
        if stat_size <= 0 or window_size < 2 * stat_size:
            raise ValueError("window_size must hold a reference and recent sample")
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        """Clear the current reference and recent windows."""

        import random

        self.window: list[float] = []
        self.num_observations = 0
        self._random = random.Random(self.seed)

    @staticmethod
    def _ks_statistic(reference: list[float], recent: list[float]) -> float:
        reference = sorted(reference)
        recent = sorted(recent)
        reference_index = 0
        recent_index = 0
        maximum_difference = 0.0

        while reference_index < len(reference) or recent_index < len(recent):
            next_reference = (
                reference[reference_index] if reference_index < len(reference) else math.inf
            )
            next_recent = recent[recent_index] if recent_index < len(recent) else math.inf
            value = min(next_reference, next_recent)
            while reference_index < len(reference) and reference[reference_index] <= value:
                reference_index += 1
            while recent_index < len(recent) and recent[recent_index] <= value:
                recent_index += 1
            difference = abs(
                reference_index / len(reference) - recent_index / len(recent)
            )
            maximum_difference = max(maximum_difference, difference)

        return maximum_difference

    def update(self, value: float) -> DriftUpdate:
        """Compare recent values with a random reference sample from the window."""

        value = float(value)
        if not math.isfinite(value):
            raise ValueError("drift signal must be finite")

        self.window.append(value)
        self.num_observations += 1
        if len(self.window) > self.window_size:
            self.window.pop(0)

        mean = sum(self.window) / len(self.window)
        if len(self.window) < self.window_size:
            return DriftUpdate(detected=False, value=value, mean=mean, score=0.0)

        reference_pool = self.window[:-self.stat_size]
        reference = self._random.sample(reference_pool, self.stat_size)
        recent = self.window[-self.stat_size:]
        statistic = self._ks_statistic(reference, recent)
        threshold = math.sqrt(
            -0.5
            * math.log(self.alpha / 2.0)
            * (len(reference) + len(recent))
            / (len(reference) * len(recent))
        )
        return DriftUpdate(
            detected=statistic > threshold,
            value=value,
            mean=mean,
            score=statistic,
        )
