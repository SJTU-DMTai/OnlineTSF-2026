"""Offline fitting and reusable Native snapshots, independent of oracle search."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ..data import load_benchmark_dataset
from ..factory import build_backbone, build_method
from ..online import OnlineExecutor
from .actions import UpdateAction, positive_integer
from .catalog import ActionCatalog
from .features import DATA_FEATURES, FEEDBACK_FEATURES, StateFeatures
from .rollout import advance, rng_state
from .runtime import UpdateController


def save_json(path: Path, record) -> None:
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare_experiment(config: dict, base: Path, output: Path) -> None:
    threads, seed = config.get("threads", 1), config.get("seed", 0)
    positive_integer(threads, "threads")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    forecast, collection = config["forecast"], config.get("collection", {})
    if {"tau", "offline_fraction", "train_end_fraction", "workers"} & collection.keys():
        raise ValueError("use split for data boundaries and collection.h_act for action length; collection.workers is no longer supported")
    device = torch.device(forecast.get("device", "cpu"))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    profiles = yaml.safe_load((base / forecast["profiles"]).read_text(encoding="utf-8"))["strategies"]
    profile = profiles[forecast["strategy"]]
    data = dict(config["data"])
    data["path"] = str((base / data["path"]).resolve())
    if data.get("stride", 1) != 1:
        raise ValueError("decision experiments require data.stride: 1")
    dataset = load_benchmark_dataset(**data, keep_decision_inputs=True)
    split = config.get("split", {})
    if set(split) - {"offline_end", "train_end", "test_end", "offline_fraction", "train_end_fraction", "test_end_fraction"}:
        raise ValueError("unknown split option")
    bounds = []
    for name, fraction, default in (("offline_end", "offline_fraction", .2),
                                     ("train_end", "train_end_fraction", .6),
                                     ("test_end", "test_end_fraction", 1.)):
        if name in split and fraction in split:
            raise ValueError(f"specify either split.{name} or split.{fraction}")
        if name in split:
            positive_integer(split[name], f"split.{name}")
            bounds.append(split[name])
        else:
            value = split.get(fraction, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError(f"split.{fraction} must satisfy 0 < value <= 1")
            bounds.append(int(len(dataset.values) * value))
    offline_end, train_end, test_end = bounds
    if not 0 < offline_end < train_end < test_end <= len(dataset.values):
        raise ValueError("require 0 < offline_end < train_end < test_end <= data rows")
    train_windows = offline_end - dataset.context_length - dataset.horizon + 1
    if train_windows <= 0 or test_end - train_end < dataset.horizon:
        raise ValueError("offline/test intervals are too short for complete forecast windows")
    for name in ("epochs", "batch_size"):
        positive_integer(forecast[name], name)
    mean = dataset.values[:offline_end].mean(0).numpy()
    std = dataset.values[:offline_end].std(0, unbiased=False).numpy()
    std[std < 1e-6] = 1.
    dataset.standardize(train_windows)
    model_config = {"data": data, "forecasting": profile["forecasting"], "method": profile["method"],
                    "online": {"device": str(device)}}
    method = build_method(model_config, build_backbone(model_config, dataset.num_features,
                                                      dataset.num_targets, dataset.target_indices))
    features = StateFeatures(collection.get("feature_window", 32), collection.get("error_window", 128))
    if hasattr(method, "configure_feature_history"):
        method.configure_feature_history(features.feature_window, features.error_window)
    controller = UpdateController(method, UpdateAction(), features=features)
    catalog = ActionCatalog.load(base / config["actions"], profile["method"]["name"], controller.capabilities)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "preparing", "schema_version": 5, "kind": "prepared", "config": config,
                "forecast_config": model_config, "catalog": catalog.to_dict(),
                "data_sha256": hashlib.sha256(Path(data["path"]).read_bytes()).hexdigest(),
                "capabilities": [asdict(unit) for unit in catalog.capabilities.units],
                "feature_names": list((*DATA_FEATURES, *FEEDBACK_FEATURES, *method.adaptation_feature_names)),
                "offline_end": offline_end, "train_end": train_end, "test_end": test_end,
                "online_start": offline_end - dataset.context_length,
                "test_start": train_end - dataset.context_length,
                "test_stop": test_end - dataset.context_length - dataset.horizon + 1,
                "target_mean": mean[list(dataset.target_indices)].tolist(),
                "target_std": std[list(dataset.target_indices)].tolist()}
    save_json(output / "manifest.json", manifest)
    try:
        loader = DataLoader(Subset(dataset, range(train_windows)), batch_size=forecast["batch_size"], shuffle=True)
        for epoch in range(forecast["epochs"]):
            for context, target in loader:
                if not np.isfinite(method.train_batch(context, target)):
                    raise FloatingPointError("nonfinite offline training loss")
            if hasattr(method, "end_offline_epoch"):
                method.end_offline_epoch(epoch + 1)
            print(f"offline epoch {epoch + 1}/{forecast['epochs']}", flush=True)
        if hasattr(method, "reset_online_state"):
            method.reset_online_state()
        controller.policy = controller.capabilities.native_action()
        delay = tuple(range(1, dataset.horizon + 1)) if method.incremental_feedback else dataset.horizon
        executor = OnlineExecutor(controller, feedback_delay=delay)
        initial_rng = rng_state()
        torch.save({"executor": executor, "dataset": dataset, "rng": initial_rng}, output / "online_start.pt")
        _, state = advance(executor, dataset, manifest["online_start"], manifest["test_start"],
                           controller.policy, initial_rng)
        torch.save({"executor": executor, "dataset": dataset, "rng": state}, output / "evaluation_start.pt")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", manifest)


def load_interval(directory: Path, *, split: str = "test", start_row: int | None = None,
                  end_row: int | None = None):
    """Bounds refer to raw target rows [start_row, end_row), never window indices."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest.get("kind") != "prepared":
        raise ValueError("prepare the forecasting snapshots before searching")
    if split not in {"history", "test", "online"}:
        raise ValueError("split must be history, test or online")
    first = manifest["train_end"] if split == "test" else manifest["offline_end"]
    last = manifest["train_end"] if split == "history" else manifest["test_end"]
    first = first if start_row is None else start_row
    last = last if end_row is None else end_row
    if not manifest["offline_end"] <= first < last <= manifest["test_end"]:
        raise ValueError("search target rows must lie after offline training and within test_end")
    if split == "history" and last > manifest["train_end"]:
        raise ValueError("history search cannot access labels past train_end")
    saved = torch.load(directory / "online_start.pt", weights_only=False)
    dataset, executor = saved["dataset"], saved["executor"]
    start, stop = first - dataset.context_length, last - dataset.context_length - dataset.horizon + 1
    if stop <= start:
        raise ValueError("interval needs at least one complete forecast window")
    _, state = advance(executor, dataset, manifest["online_start"], start,
                       executor.method.capabilities.native_action(), saved["rng"])
    return manifest, executor, dataset, start, stop, state
