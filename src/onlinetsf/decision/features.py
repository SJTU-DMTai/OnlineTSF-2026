"""Causal features shared by historical probes and deployed decisions.

Only requested statistics are evaluated. Inputs are observed rows and cached
predictions/errors; feature extraction never invokes the forecasting model.
"""

from __future__ import annotations

import re

import numpy as np
import torch

from .config import DecisionConfig


def extract_features(values: torch.Tensor | np.ndarray, *, names: list[str], config: DecisionConfig,
                     errors, signed_errors, controlled, index: int,
                     channel_signed_errors=(), feedback_target=None, current_context=None,
                     current_feedback_mae=None, feedback_age=None) -> dict:
    values = torch.as_tensor(values)
    requested = set(names)
    result = {}
    width = config.feature_window
    data_names = {"channel_shift_fraction", "volatility_ratio", "autocorrelation_shift"}
    if requested & data_names:
        if len(values) < 2 * width:
            result.update(dict.fromkeys(data_names))
        else:
            # Original NumPy float64 population statistics, on inputs retained
            # before model casting and fitted with double-precision scaling.
            observed = np.asfortranarray(values.detach().cpu().numpy(), dtype=np.float64)
            recent, previous = observed[-width:], observed[-2 * width:-width]
            previous_std = previous.std(axis=0)
            result["channel_shift_fraction"] = float(np.mean(
                np.abs(recent.mean(axis=0) - previous.mean(axis=0)) > previous_std + 1e-6))
            result["volatility_ratio"] = float(recent.std(axis=0).mean() / max(previous_std.mean(), 1e-10))
            if "autocorrelation_shift" in requested and width < 3:
                result["autocorrelation_shift"] = None
            elif "autocorrelation_shift" in requested:
                correlations, valid_channels = [], []
                for segment in (recent, previous):
                    left, right = segment[:-1], segment[1:]
                    left, right = left - left.mean(axis=0), right - right.mean(axis=0)
                    denominator = np.sqrt((left * left).sum(axis=0) * (right * right).sum(axis=0))
                    valid = denominator > 1e-12
                    correlations.append(np.divide((left * right).sum(axis=0), denominator,
                                                   out=np.zeros_like(denominator), where=valid))
                    valid_channels.append(valid)
                valid = valid_channels[0] & valid_channels[1]
                result["autocorrelation_shift"] = (float(np.abs(correlations[0][valid] - correlations[1][valid]).mean())
                                                    if valid.any() else None)
    error_values = list(errors)
    for name in names:
        match = re.fullmatch(r"(input_mean|input_std|input_trend|autocorrelation|mean_shift|mae)_(\d+)", name)
        if match is None:
            continue
        statistic, length = match[1], int(match[2])
        if length <= 0:
            raise ValueError(f"invalid feature window: {name}")
        recent = values[-length:]
        if statistic == "mae":
            selected = error_values[-length:]
            result[name] = sum(selected) / len(selected) if selected else None
        elif statistic == "input_mean":
            result[name] = recent.mean().item()
        elif statistic == "input_std":
            result[name] = recent.std(dim=0, unbiased=False).mean().item()
        elif statistic == "input_trend":
            result[name] = (recent[-1] - recent[0]).abs().mean().item() / max(1, len(recent) - 1)
        elif statistic == "autocorrelation":
            centered = recent - recent.mean(dim=0)
            result[name] = ((centered[1:] * centered[:-1]).mean(dim=0)
                            / centered.square().mean(dim=0).clamp_min(1e-8)).mean().item() if len(recent) > 1 else None
        else:
            previous = values[-2 * length:-length]
            result[name] = (recent.mean(dim=0) - previous.mean(dim=0)).abs().mean().item() if len(previous) else None
    if "error_ratio" in requested:
        short, long = error_values[-width:], error_values[-config.error_window:]
        result["error_ratio"] = ((sum(short) / len(short)) / max(sum(long) / len(long), 1e-10)
                                 if short and long else None)
    if "cross_correlation" in requested:
        channels = torch.linspace(0, values.shape[1] - 1, min(16, values.shape[1])).long()
        recent = values[-config.error_window:, channels]
        centered = recent - recent.mean(dim=0)
        normalized = centered / centered.square().mean(dim=0).sqrt().clamp_min(1e-8)
        correlation = normalized.T @ normalized / len(recent)
        mask = ~torch.eye(len(channels), dtype=torch.bool)
        result["cross_correlation"] = correlation[mask].abs().mean().item() if mask.any() else 0.0
    if "residual_bias" in requested:
        selected = list(signed_errors)[-config.error_window:]
        result["residual_bias"] = sum(selected) / len(selected) if selected else None
    if "residual_bias_ratio" in requested:
        selected = list(channel_signed_errors)[-width:]
        if selected:
            bias = torch.stack(selected).double().mean(dim=0).abs().mean().item()
            mae = sum(error_values[-len(selected):]) / len(selected)
            result["residual_bias_ratio"] = min(1.0, bias / (mae + 1e-10))
        else:
            result["residual_bias_ratio"] = None
    if "steps_since_gradient" in requested:
        result["steps_since_gradient"] = index - controlled.last_gradient_at if controlled.last_gradient_at is not None else None
    if "recent_update_fraction" in requested:
        flags = list(controlled.recent_update_flags)[-config.error_window:]
        result["recent_update_fraction"] = sum(flags) / len(flags) if flags else None
    if "current_feedback_mae" in requested:
        result["current_feedback_mae"] = current_feedback_mae
    if "feedback_age" in requested:
        result["feedback_age"] = feedback_age
    internal = requested - result.keys()
    if internal:
        cached = controlled.method.adaptation_features(feedback_target, current_context=current_context)
        if "expert_relative_gap" in internal:
            time, variable = cached.get("time_expert_mae"), cached.get("variable_expert_mae")
            result["expert_relative_gap"] = ((time - variable) / max(time + variable, 1e-10)
                                              if time is not None and variable is not None else None)
        result.update({name: cached[name] for name in internal if name in cached})
    unknown = requested - result.keys()
    if unknown:
        raise ValueError(f"unknown decision features: {sorted(unknown)}")
    return {name: result[name] for name in names}


def history_size(config: DecisionConfig, names: list[str]) -> int:
    """Bound storage while allowing configurable numbered rolling statistics."""
    windows = [2 * config.feature_window, config.error_window]
    for name in names:
        match = re.search(r"_(\d+)$", name)
        if match:
            windows.append(2 * int(match[1]))
    return max(windows)
