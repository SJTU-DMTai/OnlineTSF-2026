# -*- coding: utf-8 -*-
"""Standard streaming drift detectors for scalar forecast residuals."""

from __future__ import annotations

from dataclasses import dataclass
import math

from river import drift


@dataclass(frozen=True)
class DriftUpdate:
    """The result of one detector update."""

    detected: bool
    value: float


@dataclass(frozen=True)
class DriftObservation:
    """One source value routed to its variable-specific detector."""

    variable_name: str
    variable_index: int
    horizon_step: int | None
    value: float
    available_at: int


@dataclass(frozen=True)
class DriftRecord:
    """One scalar residual consumed by a drift detector."""

    detector: str
    source: str
    index: int
    available_at: int
    variable_name: str
    variable_index: int
    horizon_step: int | None
    value: float
    detected: bool


class PageHinkleyDetector:
    """River's standard Page-Hinkley detector."""

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        mode: str = "both",
    ) -> None:
        self.options = {
            "min_instances": min_instances,
            "delta": delta,
            "threshold": threshold,
            "alpha": alpha,
            "mode": mode,
        }
        self.reset()

    def reset(self) -> None:
        self.detector = drift.PageHinkley(**self.options)

    def update(self, value: float) -> DriftUpdate:
        value = _finite_value(value)
        self.detector.update(value)
        return DriftUpdate(detected=self.detector.drift_detected, value=value)


class ADWINDetector:
    """River's ADWIN2 detector."""

    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
    ) -> None:
        self.options = {
            "delta": delta,
            "clock": clock,
            "max_buckets": max_buckets,
            "min_window_length": min_window_length,
            "grace_period": grace_period,
        }
        self.reset()

    def reset(self) -> None:
        self.detector = drift.ADWIN(**self.options)

    def update(self, value: float) -> DriftUpdate:
        value = _finite_value(value)
        self.detector.update(value)
        return DriftUpdate(detected=self.detector.drift_detected, value=value)


class KSWINDetector:
    """River's standard KSWIN detector."""

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int | None = 0,
    ) -> None:
        self.options = {
            "alpha": alpha,
            "window_size": window_size,
            "stat_size": stat_size,
            "seed": seed,
        }
        self.reset()

    def reset(self) -> None:
        self.detector = drift.KSWIN(**self.options)

    def update(self, value: float) -> DriftUpdate:
        value = _finite_value(value)
        self.detector.update(value)
        return DriftUpdate(detected=self.detector.drift_detected, value=value)


def _finite_value(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("drift signal must be finite")
    return value
