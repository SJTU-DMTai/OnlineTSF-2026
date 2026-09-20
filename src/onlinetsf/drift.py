# -*- coding: utf-8 -*-
"""Streaming detectors for scalar forecast signals and feature vectors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from river import drift


@dataclass(frozen=True)
class DriftUpdate:
    """The result of one detector update."""

    detected: bool
    value: float | None


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
    """One detector update and its source location."""

    detector: str
    source: str
    index: int
    available_at: int
    raw_signal_index: int
    variable_name: str
    variable_index: int
    horizon_step: int | None
    value: float | None
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


class HDDMWDetector:
    """River's weighted Hoeffding detector for binary forecast errors."""

    def __init__(
        self,
        drift_confidence: float = 0.001,
        warning_confidence: float = 0.005,
        lambda_val: float = 0.05,
        two_sided_test: bool = False,
    ) -> None:
        self.options = {
            "drift_confidence": drift_confidence,
            "warning_confidence": warning_confidence,
            "lambda_val": lambda_val,
            "two_sided_test": two_sided_test,
        }
        self.reset()

    def reset(self) -> None:
        self.detector = drift.binary.HDDMW(**self.options)

    def update(self, value: float) -> DriftUpdate:
        value = _binary_value(value)
        self.detector.update(value)
        return DriftUpdate(detected=self.detector.drift_detected, value=value)


class CapyMOADetector:
    """Adapter for CapyMOA's SEED, STEPD, and multivariate ABCD detectors."""

    def __init__(self, name: str, **options: object) -> None:
        self.name = name
        self.options = options
        self.reset()

    def reset(self) -> None:
        try:
            from capymoa.drift import detectors
        except ImportError as error:
            raise ImportError(
                f"{self.name} requires CapyMOA; install onlinetsf[capymoa] and configure Java"
            ) from error
        detector_class = {"seed": detectors.SEED, "stepd": detectors.STEPD, "abcd": detectors.ABCD}[self.name]
        self.detector = detector_class(**self.options)

    def update(self, value: float | Sequence[float]) -> DriftUpdate:
        if self.name == "abcd":
            import numpy as np

            vector = np.asarray(value, dtype=float)
            if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all():
                raise ValueError("ABCD requires a finite multivariate feature vector")
            self.detector.add_element(vector)
            return DriftUpdate(detected=self.detector.detected_change(), value=None)

        scalar = _finite_value(value) if self.name == "seed" else _binary_value(value)
        self.detector.add_element(scalar)
        return DriftUpdate(detected=self.detector.detected_change(), value=scalar)


def _finite_value(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("drift signal must be finite")
    return value


def _binary_value(value: float) -> float:
    value = _finite_value(value)
    if value not in (0.0, 1.0):
        raise ValueError("this detector requires a binary 0/1 signal")
    return value
