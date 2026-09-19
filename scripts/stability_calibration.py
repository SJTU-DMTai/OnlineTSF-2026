# -*- coding: utf-8 -*-
"""Data-driven, per-stream detector calibration for stability selection."""

from __future__ import annotations

import statistics
from typing import Any, Sequence

from onlinetsf.__main__ import _build_detector, _drift_values
from onlinetsf.online import OnlineRun


def calibration_streams(
    config: dict[str, Any], run: OnlineRun, start: int, end: int,
) -> dict[str, list[float]]:
    """Use the same observation routing as the production detector pass."""
    streams: dict[str, list[float]] = {}
    seen_features: set[int] = set()
    for event in run.events:
        if not start <= event.index < end:
            continue
        if config["drift"]["source"] == "features":
            if event.index in seen_features:
                continue
            seen_features.add(event.index)
        for observation in _drift_values(config, event):
            streams.setdefault(observation.variable_name, []).append(observation.value)
    return streams


def alarm_positions(config: dict[str, Any], values: Sequence[float], parameters: dict[str, Any]) -> list[int]:
    detector = _build_detector(config, parameters)
    alarms: list[int] = []
    for index, value in enumerate(values):
        if detector.update(value).detected:
            alarms.append(index)
    return alarms


def calibrate_stream(
    config: dict[str, Any], values: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select sensitivity using original alarms and three injected changes.

    Original-stream alarms are only a proxy for false alarms: the source data
    has no drift labels. Injection sensitivity prevents choosing a deaf detector.
    """
    name = config["drift"]["name"]
    if len(values) < 64:
        raise ValueError(f"automatic {name} calibration needs at least 64 observations per stream")
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = statistics.pstdev(values)
    constant_stream = scale <= 1e-12
    if scale <= 1e-12:
        scale = 1.0

    base = config["drift"]["parameters"]
    if name == "page_hinkley":
        candidates = [
            {"threshold": scale * multiplier, "delta": scale * 0.01}
            for multiplier in (4.0, 8.0, 16.0, 32.0)
        ]
    elif name == "adwin":
        candidates = [{"delta": delta} for delta in (0.0001, 0.001, 0.01)]
    elif name == "kswin":
        candidates = [{"alpha": alpha} for alpha in (1e-6, 1e-4, 0.001, 0.005)]
        if len(values) < 2 * base["window_size"]:
            raise ValueError("automatic KSWIN calibration needs at least twice window_size observations")
    else:
        raise ValueError(f"unsupported detector for automatic calibration: {name}")

    onset = len(values) // 2
    detection_span = max(32, len(values) // 8)
    original = list(values)
    mean_shift = original[:onset] + [value + 2.0 * scale for value in original[onset:]]
    gradual = original[:onset] + [
        value + 2.0 * scale * (index - onset + 1) / (len(values) - onset)
        for index, value in enumerate(original[onset:], start=onset)
    ]
    variance = original[:onset] + [median + 2.0 * (value - median) for value in original[onset:]]
    injections = {"mean": mean_shift, "gradual_mean": gradual, "variance": variance}

    evaluations: list[dict[str, Any]] = []
    for candidate in candidates:
        parameters = {**base, **candidate}
        baseline_alarms = alarm_positions(config, original, parameters)
        baseline_first = next(
            (index for index in baseline_alarms if onset <= index < onset + detection_span), None
        )
        delays: dict[str, int | None] = {}
        for injection_name, injected in injections.items():
            alarms = alarm_positions(config, injected, parameters)
            first = next(
                (index for index in alarms
                 if onset <= index < onset + detection_span
                 and (baseline_first is None or index < baseline_first)),
                None,
            )
            delays[injection_name] = None if first is None else first - onset
        hits = sum(delay is not None for delay in delays.values())
        evaluations.append({
            "parameters": candidate,
            "baseline_alarm_count": len(baseline_alarms),
            "baseline_first_alarm_after_onset": baseline_first,
            "injection_delays": delays,
            "injection_hits": hits,
            "score": hits - min(len(baseline_alarms), 3),
        })
    selected = max(
        evaluations,
        key=lambda item: (
            item["score"], item["injection_hits"], -item["baseline_alarm_count"],
            -sum(delay for delay in item["injection_delays"].values() if delay is not None),
        ),
    )
    return selected["parameters"], {
        "observations": len(values),
        "robust_scale": scale,
        "constant_stream_scale_fallback": constant_stream,
        "injection_onset": onset,
        "detection_span": detection_span,
        "candidates": evaluations,
        "selected_parameters": selected["parameters"],
        "warning": "Original-stream alarms are not known false positives; injected drifts are proxy tests.",
    }
