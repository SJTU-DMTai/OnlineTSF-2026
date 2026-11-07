"""Bounded causal state statistics, maintained on every arrived feedback."""

from collections import deque

import numpy as np
import torch

from .actions import positive_integer


DATA_FEATURES = ("channel_shift_fraction", "volatility_ratio", "autocorrelation_shift",
                 "normalized_mean_shift", "trend_shift", "cross_correlation_shift")
FEEDBACK_FEATURES = ("recent_mae", "error_ratio", "residual_bias_ratio",
                     "error_volatility", "tail_error_ratio", "first_step_mae", "residual_autocorrelation")


def channel_correlation(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlations per channel, with constant channels marked invalid."""
    left, right = left - left.mean(0), right - right.mean(0)
    denominator = np.sqrt((left * left).sum(0) * (right * right).sum(0))
    valid = denominator > 1e-12
    return np.divide((left * right).sum(0), denominator,
                     out=np.zeros_like(denominator), where=valid), valid


class StateFeatures:
    def __init__(self, feature_window: int = 32, error_window: int = 128) -> None:
        positive_integer(feature_window, "feature_window")
        positive_integer(error_window, "error_window")
        if feature_window < 3 or error_window < feature_window:
            raise ValueError("require 3 <= feature_window <= error_window")
        self.feature_window, self.error_window = feature_window, error_window
        self.values = deque(maxlen=2 * feature_window)
        self.errors = deque(maxlen=error_window)
        self.channel_biases = deque(maxlen=feature_window)
        self.first_step_residuals = deque(maxlen=feature_window)
        self.expert_errors = deque(maxlen=feature_window)
        self.partial: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}

    def on_observation(self, context) -> None:
        values = np.asarray(torch.as_tensor(context).detach().cpu(), dtype=np.float64)
        if not self.values:
            self.values.extend(row.copy() for row in values[-2 * self.feature_window:])
        else:
            self.values.append(values[-1].copy())

    def observe(self, prepared) -> None:
        feedback = prepared.feedback
        mask = feedback.observed_mask.cpu()
        error = (feedback.prediction.cpu().double() - feedback.target.cpu().double()).masked_fill(~mask, 0)
        if mask[0].all():
            self.first_step_residuals.append(error[0].numpy().copy())
        absolute, signed, count = self.partial.pop(
            feedback.index, (torch.zeros(error.shape[1], dtype=torch.float64),
                             torch.zeros(error.shape[1], dtype=torch.float64), 0))
        absolute += error.abs().sum(dim=0)
        signed += error.sum(dim=0)
        count += int(mask.sum())
        if count == error.numel():
            self.errors.append(absolute.sum().item() / count)
            self.channel_biases.append((signed / error.shape[0]).numpy())
        else:
            self.partial[feedback.index] = absolute, signed, count
        if prepared.expert_predictions is not None:
            time, variable = prepared.expert_predictions
            self.expert_errors.append(((time.double() - feedback.target.double()).abs().mean().item(),
                                       (variable.double() - feedback.target.double()).abs().mean().item()))

    def snapshot(self, method) -> dict[str, float | None]:
        result = dict.fromkeys((*DATA_FEATURES, *FEEDBACK_FEATURES))
        width = self.feature_window
        if len(self.values) >= 2 * width:
            values = np.asarray(self.values)
            previous, recent = values[:width], values[width:]
            std = previous.std(axis=0)
            result["channel_shift_fraction"] = float(np.mean(np.abs(recent.mean(0) - previous.mean(0)) > std + 1e-6))
            result["volatility_ratio"] = float(recent.std(0).mean() / max(std.mean(), 1e-10))
            result["normalized_mean_shift"] = float((np.abs(recent.mean(0) - previous.mean(0))
                                                      / np.maximum(std, 1e-6)).mean())
            time = np.arange(width, dtype=float) - (width - 1) / 2
            slope_change = time @ (recent - previous) / (time @ time)
            result["trend_shift"] = float((np.abs(slope_change) * (width - 1) / np.maximum(std, 1e-6)).mean())
            correlations, valid_channels = [], []
            for segment in (previous, recent):
                correlation, valid = channel_correlation(segment[:-1], segment[1:])
                correlations.append(correlation)
                valid_channels.append(valid)
            valid = valid_channels[0] & valid_channels[1]
            result["autocorrelation_shift"] = float(np.abs(correlations[0][valid] - correlations[1][valid]).mean()) if valid.any() else None
            # Compare each off-diagonal pair only when both windows have variance.
            valid = (previous.std(0) > 1e-12) & (recent.std(0) > 1e-12)
            channels = np.flatnonzero(valid)
            if len(channels) > 1:
                previous_z = (previous[:, channels] - previous[:, channels].mean(0)) / previous[:, channels].std(0)
                recent_z = (recent[:, channels] - recent[:, channels].mean(0)) / recent[:, channels].std(0)
                difference = (recent_z.T @ recent_z - previous_z.T @ previous_z) / width
                result["cross_correlation_shift"] = float(np.abs(difference[np.triu_indices(len(channels), 1)]).mean())
        if self.errors:
            short = list(self.errors)[-width:]
            result["recent_mae"] = float(np.mean(short))
            result["error_ratio"] = float(np.mean(short) / max(np.mean(self.errors), 1e-10))
            result["residual_bias_ratio"] = min(1., float(np.abs(np.mean(self.channel_biases, axis=0)).mean()
                                                         / (np.mean(short) + 1e-10)))
            result["error_volatility"] = float(np.std(short) / max(np.mean(short), 1e-10))
            result["tail_error_ratio"] = float(np.quantile(short, .9) / max(np.mean(short), 1e-10))
        if self.first_step_residuals:
            residuals = np.asarray(self.first_step_residuals)
            result["first_step_mae"] = float(np.abs(residuals).mean())
            if len(residuals) >= 3:
                correlation, valid = channel_correlation(residuals[:-1], residuals[1:])
                result["residual_autocorrelation"] = float(np.abs(correlation[valid]).mean()) if valid.any() else None
        result.update(method.adaptation_features())
        if "expert_relative_gap" in method.adaptation_feature_names:
            gap = None
            if self.expert_errors:
                time, variable = np.mean(self.expert_errors, axis=0)
                gap = float((time - variable) / max(time + variable, 1e-10))
            result["expert_relative_gap"] = gap
        return result
