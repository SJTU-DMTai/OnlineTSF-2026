# -*- coding: utf-8 -*-
"""Loading and validation for the YAML experiment configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


BACKBONES = frozenset(("linear", "dlinear", "lstm", "tcn", "patchtst", "fsnet_tcn", "onenet_tcn"))
METHODS = frozenset(("ogd", "fsnet", "onenet", "dsof", "under_cali"))
DRIFT_DETECTORS = frozenset(("none", "page_hinkley", "adwin", "kswin", "seed", "stepd", "hddmw", "abcd"))
DRIFT_SOURCES = frozenset(("features", "target", "residual"))
BINARY_RESIDUAL_DETECTORS = frozenset(("stepd", "hddmw"))
DETECTOR_SOURCES = {
    "seed": frozenset(("features", "residual")),
    "stepd": frozenset(("residual",)),
    "hddmw": frozenset(("residual",)),
    "abcd": frozenset(("features",)),
}


def _load_mapping(source: Path, label: str) -> dict[str, Any]:
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{label} root must be a mapping")
    return dict(loaded)


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


def _profile(profiles: dict[str, Any], group: str, name: Any) -> dict[str, Any]:
    selector = {"strategies": "strategy", "detectors": "detector"}[group]
    if not isinstance(name, str):
        raise ValueError(f"config.selection.{selector} must be a string")
    entries = profiles.get(group)
    if not isinstance(entries, dict):
        raise ValueError(f"profiles.{group} must be a mapping")
    value = entries.get(name)
    if not isinstance(value, dict):
        available = ", ".join(sorted(entries))
        raise ValueError(f"unknown {selector} profile {name!r}; choose one of: {available}")
    return dict(value)


def load_config(
    path: str | Path,
    *,
    strategy_name: str | None = None,
    detector_name: str | None = None,
) -> dict[str, Any]:
    """Read an experiment YAML file and validate its component selections."""

    source = Path(path)
    config = _load_mapping(source, "config")
    data = _section(config, "data")
    online = _section(config, "online")
    offline_value = config.get("offline", {})
    if not isinstance(offline_value, dict):
        raise ValueError("config.offline must be a mapping")
    offline = dict(offline_value)

    selection = _section(config, "selection")
    if strategy_name is not None:
        selection["strategy"] = strategy_name
    if detector_name is not None:
        selection["detector"] = detector_name
    profile_path_value = config.get("profiles")
    if not isinstance(profile_path_value, str):
        raise ValueError("config.profiles must be a path string")
    profile_path = Path(profile_path_value)
    if not profile_path.is_absolute():
        profile_path = source.parent / profile_path
    profiles = _load_mapping(profile_path, "profiles")
    strategy = _profile(profiles, "strategies", selection.get("strategy"))
    forecasting = _section(strategy, "forecasting")
    method = _section(strategy, "method")
    drift = _profile(profiles, "detectors", selection.get("detector"))
    config["profiles"] = str(profile_path.resolve())
    config["selection"] = selection

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
    paired_backbones = {"fsnet": "fsnet_tcn", "onenet": "onenet_tcn"}
    expected_backbone = paired_backbones.get(method_name)
    if expected_backbone is not None and backbone != expected_backbone:
        raise ValueError(f"method {method_name} must be paired with backbone {expected_backbone}")
    if backbone in paired_backbones.values() and expected_backbone != backbone:
        raise ValueError(f"backbone {backbone} requires its matching online method")
    if method_name in paired_backbones and method.get("learning_rate") is None:
        raise ValueError(f"config.method.learning_rate is required for {method_name}")
    if method_name == "dsof":
        if backbone in {"fsnet_tcn", "onenet_tcn"}:
            raise ValueError("dsof requires a standard forecasting backbone")
        for option in ("learning_rate", "student_learning_rate", "online_learning_rate"):
            value = method.get(option, 1e-3)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"config.method.{option} must be positive for dsof")
            method[option] = float(value)
        for option, default in (("student_hidden", 16), ("student_depth", 3),
                                ("replay_buffer_size", 300), ("replay_batch_size", 32),
                                ("replay_epochs", 1), ("replay_frequency", 1)):
            value = method.get(option, default)
            minimum = 2 if option == "student_depth" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"config.method.{option} must be an integer >= {minimum} for dsof")
            method[option] = value
        if method["replay_batch_size"] > method["replay_buffer_size"]:
            raise ValueError("config.method.replay_batch_size cannot exceed replay_buffer_size")
        dropout = method.get("student_dropout", 0.0)
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("config.method.student_dropout must be in [0, 1) for dsof")
        method["student_dropout"] = float(dropout)
        discount = method.get("discount", 0.9)
        if not isinstance(discount, (int, float)) or isinstance(discount, bool) or not math.isfinite(discount) or not 0 < discount <= 1:
            raise ValueError("config.method.discount must be in (0, 1] for dsof")
        method["discount"] = float(discount)
        if data.get("stride", 1) != 1:
            raise ValueError("dsof requires data.stride: 1 for one observation per step")
    if method_name == "fsnet":
        n_inner = method.get("n_inner", 1)
        if not isinstance(n_inner, int) or isinstance(n_inner, bool) or n_inner <= 0:
            raise ValueError("config.method.n_inner must be a positive integer for fsnet")
        method["n_inner"] = n_inner
    if method_name == "onenet":
        for option in ("learning_rate", "weight_learning_rate", "decision_learning_rate"):
            value = method.get(option, 1e-3)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0.0:
                raise ValueError(f"config.method.{option} must be positive for onenet")
            method[option] = float(value)
        decision_hidden = method.get("decision_hidden", 32)
        if not isinstance(decision_hidden, int) or isinstance(decision_hidden, bool) or decision_hidden <= 0:
            raise ValueError("config.method.decision_hidden must be a positive integer for onenet")
        method["decision_hidden"] = decision_hidden
        decision_dropout = method.get("decision_dropout", 0.1)
        if (
            not isinstance(decision_dropout, (int, float))
            or isinstance(decision_dropout, bool)
            or not 0.0 <= decision_dropout < 1.0
        ):
            raise ValueError("config.method.decision_dropout must be in [0, 1) for onenet")
        method["decision_dropout"] = float(decision_dropout)
        n_inner = method.get("n_inner", 1)
        if not isinstance(n_inner, int) or isinstance(n_inner, bool) or n_inner <= 0:
            raise ValueError("config.method.n_inner must be a positive integer for onenet")
        method["n_inner"] = n_inner

    if method_name == "under_cali":
        if backbone in {"fsnet_tcn", "onenet_tcn"}:
            raise ValueError("under_cali requires a standard forecasting backbone")
        for option, default in (("learning_rate", None), ("adapt_lr", 1e-3),
                                ("uncertainty_lr", 1e-4), ("uncertainty_adapt_lr", 1e-4)):
            value = method.get(option, default)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"config.method.{option} must be positive for under_cali")
            method[option] = float(value)
        for option, default in (("calibrator_hidden_dim", 64), ("uncertainty_hidden_dim", 128),
                                ("uncertainty_epochs", 20), ("update_steps", 5)):
            value = method.get(option, default)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"config.method.{option} must be a positive integer for under_cali")
            method[option] = value
        for option, default in (("allocation_alpha", 0.75), ("trigger_alpha", 0.25)):
            value = method.get(option, default)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 1:
                raise ValueError(f"config.method.{option} must be in (0, 1] for under_cali")
            method[option] = float(value)
        for option, default in (("allocation_std_k", 0.25), ("trigger_std_k", 0.75)):
            value = method.get(option, default)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"config.method.{option} must be non-negative for under_cali")
            method[option] = float(value)

    offline_train_ratio = offline.get("train_ratio", 0.0)
    if (
        not isinstance(offline_train_ratio, (int, float))
        or isinstance(offline_train_ratio, bool)
        or not 0.0 <= offline_train_ratio < 1.0
    ):
        raise ValueError("config.offline.train_ratio must be in [0, 1)")
    offline["train_ratio"] = float(offline_train_ratio)
    offline_epochs = offline.get("epochs", 1)
    if not isinstance(offline_epochs, int) or isinstance(offline_epochs, bool) or offline_epochs <= 0:
        raise ValueError("config.offline.epochs must be a positive integer")
    offline["epochs"] = offline_epochs
    offline_batch_size = offline.get("batch_size", 32)
    if not isinstance(offline_batch_size, int) or isinstance(offline_batch_size, bool) or offline_batch_size <= 0:
        raise ValueError("config.offline.batch_size must be a positive integer")
    offline["batch_size"] = offline_batch_size
    if method_name == "under_cali" and offline["train_ratio"] == 0.0:
        raise ValueError("under_cali requires offline.train_ratio > 0")
    if offline["train_ratio"] > 0.0 and method.get("learning_rate") is None:
        raise ValueError("config.method.learning_rate is required when offline training is enabled")

    if "feedback_delay" not in online:
        raise ValueError("config.online.feedback_delay is required")
    feedback_delay = online["feedback_delay"]
    if isinstance(feedback_delay, list) and any(not isinstance(delay, int) or delay < 0 for delay in feedback_delay):
        raise ValueError("config.online.feedback_delay must contain non-negative integers")
    if not isinstance(feedback_delay, (int, list)) or isinstance(feedback_delay, bool):
        raise ValueError("config.online.feedback_delay must be an integer or integer list")
    if isinstance(feedback_delay, int) and feedback_delay < 0:
        raise ValueError("config.online.feedback_delay must be non-negative")
    if method_name == "dsof" and feedback_delay != list(range(1, data["horizon"] + 1)):
        raise ValueError("dsof requires online.feedback_delay: [1, ..., horizon]")
    if method_name in {"fsnet", "onenet", "under_cali"} and isinstance(feedback_delay, list):
        raise ValueError(f"{method_name} requires a scalar complete-feedback delay")

    detector_name = drift.get("name")
    if detector_name not in DRIFT_DETECTORS:
        valid_names = ", ".join(sorted(DRIFT_DETECTORS))
        raise ValueError(f"config.drift.name must be one of: {valid_names}")
    if "signal" in drift or "scale" in drift:
        raise ValueError("drift.signal and drift.scale are not supported; use drift.source")
    drift_source = drift.get("source", "features")
    if drift_source not in DRIFT_SOURCES:
        valid_sources = ", ".join(sorted(DRIFT_SOURCES))
        raise ValueError(f"config.drift.source must be one of: {valid_sources}")
    if drift_source not in DETECTOR_SOURCES.get(detector_name, DRIFT_SOURCES):
        allowed = ", ".join(sorted(DETECTOR_SOURCES[detector_name]))
        raise ValueError(f"{detector_name} requires drift.source: {allowed}")
    if detector_name in BINARY_RESIDUAL_DETECTORS:
        threshold = drift.get("error_threshold")
        if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
                or not math.isfinite(threshold) or threshold <= 0):
            raise ValueError(f"{detector_name} requires a positive drift.error_threshold")
        drift["error_threshold"] = float(threshold)
    drift["source"] = drift_source
    drift["parameters"] = _parameters(drift, "drift")

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
    write_per_value_errors = output.get("write_per_value_errors", True)
    if not isinstance(write_per_value_errors, bool):
        raise ValueError("config.output.write_per_value_errors must be a boolean")
    output["write_per_value_errors"] = write_per_value_errors

    config["data"] = data
    config["forecasting"] = forecasting
    config["method"] = method
    config["online"] = online
    config["offline"] = offline
    config["drift"] = drift
    config["output"] = output
    return config
