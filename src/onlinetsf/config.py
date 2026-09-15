# -*- coding: utf-8 -*-
"""Loading and validation for the YAML experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


BACKBONES = frozenset(("linear", "tcn", "patchtst", "fsnet_tcn"))
METHODS = frozenset(("ogd", "fsnet"))
DRIFT_DETECTORS = frozenset(("none", "page_hinkley", "adwin", "kswin"))


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"config.{name} must be a mapping")
    return dict(value)


def _parameters(section: dict[str, Any], name: str) -> dict[str, Any]:
    value = section.get("parameters", {})
    if not isinstance(value, dict):
        raise ValueError(f"config.{name}.parameters must be a mapping")
    return dict(value)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read an experiment YAML file and validate its component selections."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")

    config = dict(loaded)
    data = _section(config, "data")
    forecasting = _section(config, "forecasting")
    method = _section(config, "method")
    online = _section(config, "online")
    drift = _section(config, "drift")

    for key in ("name", "path", "context_length", "horizon"):
        if key not in data:
            raise ValueError(f"config.data.{key} is required")
    data_path = Path(data["path"])
    if not data_path.is_absolute():
        data["path"] = str((source.parent / data_path).resolve())

    backbone = forecasting.get("backbone")
    if backbone not in BACKBONES:
        raise ValueError(f"config.forecasting.backbone must be one of: {', '.join(sorted(BACKBONES))}")
    forecasting["parameters"] = _parameters(forecasting, "forecasting")
    reserved_model_options = {"context_length", "num_features", "horizon", "num_targets", "target_indices"}
    supplied_reserved_options = reserved_model_options.intersection(forecasting["parameters"])
    if supplied_reserved_options:
        names = ", ".join(sorted(supplied_reserved_options))
        raise ValueError(f"model dimensions are derived from data and cannot be configured: {names}")

    method_name = method.get("name")
    if method_name not in METHODS:
        raise ValueError(f"config.method.name must be one of: {', '.join(sorted(METHODS))}")
    if (method_name == "fsnet") != (backbone == "fsnet_tcn"):
        raise ValueError("method fsnet must be paired with backbone fsnet_tcn")
    if method_name == "fsnet" and method.get("learning_rate") is None:
        raise ValueError("config.method.learning_rate is required for fsnet")

    if "feedback_delay" not in online:
        raise ValueError("config.online.feedback_delay is required")
    feedback_delay = online["feedback_delay"]
    if isinstance(feedback_delay, list) and any(not isinstance(delay, int) or delay < 0 for delay in feedback_delay):
        raise ValueError("config.online.feedback_delay must contain non-negative integers")
    if not isinstance(feedback_delay, (int, list)) or isinstance(feedback_delay, bool):
        raise ValueError("config.online.feedback_delay must be an integer or integer list")
    if isinstance(feedback_delay, int) and feedback_delay < 0:
        raise ValueError("config.online.feedback_delay must be non-negative")
    if method_name == "fsnet" and isinstance(feedback_delay, list):
        raise ValueError("fsnet requires a scalar complete-feedback delay")

    detector_name = drift.get("name")
    if detector_name not in DRIFT_DETECTORS:
        valid_names = ", ".join(sorted(DRIFT_DETECTORS))
        raise ValueError(f"config.drift.name must be one of: {valid_names}")
    if drift.get("signal", "mae") not in {"mae", "mse"}:
        raise ValueError("config.drift.signal must be mae or mse")
    drift["parameters"] = _parameters(drift, "drift")
    if detector_name == "adwin":
        scale = drift.get("scale")
        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError("config.drift.scale must be positive for adwin")

    output_value = config.get("output", {})
    if not isinstance(output_value, dict):
        raise ValueError("config.output must be a mapping")
    output = dict(output_value)
    output_directory = Path(output.get("directory", "runs"))
    if not output_directory.is_absolute():
        output["directory"] = str((source.parent / output_directory).resolve())
    run_name = output.get("run_name")
    if run_name is not None and not isinstance(run_name, str):
        raise ValueError("config.output.run_name must be a string or null")

    config["data"] = data
    config["forecasting"] = forecasting
    config["method"] = method
    config["online"] = online
    config["drift"] = drift
    config["output"] = output
    return config
