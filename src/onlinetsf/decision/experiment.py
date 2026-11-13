"""Collect causal counterfactuals, fit selectors, and evaluate continuous streams."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ..data import load_benchmark_dataset
from ..factory import build_backbone, build_method
from ..online import OnlineExecutor
from .actions import UpdateAction, positive_integer
from .catalog import ActionCatalog
from .capabilities import UpdateCapabilities, UpdateUnit
from .features import DATA_FEATURES, FEEDBACK_FEATURES, StateFeatures
from .feature_selection import select_state_features
from .runtime import UpdateController
from .selector import ActionSelector, ESTIMATORS, FORMULATIONS, LOG1P_FEATURES, SelectorPolicy
from .training import select_temperature


def save_json(path: Path, record) -> None:
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def rng_state() -> tuple:
    return torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []


def restore_rng(state: tuple) -> None:
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


def step_dataset(executor: OnlineExecutor, dataset, index: int, *, final: bool = False):
    context, target = dataset[index]
    values = dataset.decision_values
    decision_context = values[index:index + dataset.context_length] if values is not None else context
    return executor.step(index, context, target, decision_context=decision_context, final=final)


class ProbePolicy:
    """Capture the same pre-update state seen by deployed selectors."""

    def __init__(self, action: UpdateAction) -> None:
        self.action = action
        self.features: dict | None = None

    def __call__(self, state):
        if self.features is None:
            self.features = state.features
        return self.action


def score_action(source: OnlineExecutor, dataset, origin: int, action: UpdateAction, *,
                 steps: int = 64, stop: int | None = None) -> tuple[dict, dict]:
    """Apply one action for `steps` origins when collecting training labels."""
    positive_integer(steps, "steps")
    saved_rng = rng_state()
    try:
        branch = copy.deepcopy(source)
        policy = ProbePolicy(action)
        branch.method.policy = policy
        absolute = squared = 0.
        count = 0
        end = origin + steps if stop is None else min(origin + steps, stop)
        for index in range(origin, end):
            prediction = step_dataset(branch, dataset, index, final=stop is not None and index == stop - 1).emission.prediction.double()
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"nonfinite forecast at origin {origin}")
            error = prediction - dataset[index][1].double()
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
            count += error.numel()
            branch._events.clear()
        return policy.features, {"mae": absolute / count, "mse": squared / count}
    finally:
        restore_rng(saved_rng)


def initialize_collection_worker(dataset, threads: int, device: torch.device) -> None:
    global _collection_dataset
    _collection_dataset = dataset
    torch.set_num_threads(threads)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def score_action_worker(snapshot: bytes, origin: int, action: UpdateAction, steps: int) -> tuple[dict, dict]:
    # Bytes avoid sharing mutable model/optimizer CUDA storage between processes.
    saved = torch.load(io.BytesIO(snapshot), weights_only=False)
    restore_rng(saved["rng"])
    return score_action(saved["executor"], _collection_dataset, origin, action, steps=steps)


def sample_origins(first_target: int, end_row: int, context: int, horizon: int,
                   steps: int, count: int) -> list[int]:
    positive_integer(count, "state count")
    first, last = first_target - context, end_row - context - steps - horizon + 1
    if first < 0 or last - first + 1 < count:
        raise ValueError("historical split is too short for the state count, tau and full forecast horizon")
    return np.linspace(first, last, count, dtype=np.int64).tolist()


def collect_experiment(config: dict, base: Path, output: Path) -> None:
    """One data set and forecasting method per independently fitted experiment."""
    print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] collection started", flush=True)
    seed, threads = config.get("seed", 0), config.get("threads", 1)
    positive_integer(threads, "threads")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    forecast, collection = config["forecast"], config["collection"]
    workers = collection.get("workers", 1)
    positive_integer(workers, "collection.workers")
    profiles = yaml.safe_load((base / forecast["profiles"]).read_text(encoding="utf-8"))["strategies"]
    profile = profiles[forecast["strategy"]]
    method_name = profile["method"]["name"]
    if method_name not in {"onenet", "fsnet", "dsof"}:
        raise ValueError("selector experiments support onenet, fsnet and dsof")
    device = torch.device(forecast.get("device", "cpu"))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    data = dict(config["data"])
    data["path"] = str((base / data["path"]).resolve())
    if data.get("stride", 1) != 1:
        raise ValueError("selector experiments require data.stride: 1")
    dataset = load_benchmark_dataset(**data, keep_decision_inputs=True)
    fractions = [collection["offline_fraction"], collection["train_end_fraction"]]
    if not 0 < fractions[0] < fractions[1] < 1:
        raise ValueError("require 0 < offline < train-end < 1")
    offline_end, train_end = [int(len(dataset.values) * fraction) for fraction in fractions]
    train_windows = offline_end - dataset.context_length - dataset.horizon + 1
    if train_windows <= 0:
        raise ValueError("offline split is shorter than context plus horizon")
    for name in ("epochs", "batch_size"):
        positive_integer(forecast[name], name)
    # These statistics and the retained high-precision decision inputs fit only the offline prefix.
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
    catalog = ActionCatalog.load(base / config["actions"], method_name, controller.capabilities)
    tau = collection.get("tau", 64)
    positive_integer(tau, "collection.tau")
    train_origins = sample_origins(offline_end + dataset.horizon, train_end, dataset.context_length,
                                   dataset.horizon, tau, collection["train_states"])
    test_start = train_end - dataset.context_length
    if test_start >= len(dataset):
        raise ValueError("test split has no complete forecast windows")
    feature_names = list((*DATA_FEATURES, *FEEDBACK_FEATURES, *method.adaptation_feature_names))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "collecting", "schema_version": 4, "config": config, "tau": tau,
                "forecast_config": model_config, "data_sha256": hashlib.sha256(Path(data["path"]).read_bytes()).hexdigest(),
                "catalog": catalog.to_dict(), "feature_names": feature_names,
                "capabilities": [asdict(unit) for unit in catalog.capabilities.units],
                "offline_end": offline_end, "train_end": train_end,
                "train_origins": train_origins, "test_start": test_start,
                "test_stop": len(dataset), "metric_scale": "offline-standardized numeric channels",
                "target_mean": mean[list(dataset.target_indices)].tolist(),
                "target_std": std[list(dataset.target_indices)].tolist(), "target_names": list(dataset.target_names),
                "gain": "(same-origin same-tau Native MAE - action MAE) / max(Native MAE, 1e-12)",
                "regression_target": "same-origin same-tau action MAE",
                "branch_boundary": "nonterminal; no artificial tail-batch completion at tau",
                "versions": {"torch": torch.__version__, "numpy": np.__version__}}
    save_json(output / "manifest.json", manifest)
    began = perf_counter()
    try:
        loader = DataLoader(Subset(dataset, range(train_windows)), batch_size=forecast["batch_size"], shuffle=True)
        for epoch in range(forecast["epochs"]):
            for context, target in loader:
                if not np.isfinite(method.train_batch(context, target)):
                    raise FloatingPointError("nonfinite offline training loss")
            if hasattr(method, "end_offline_epoch"):
                method.end_offline_epoch(epoch + 1)
            print(f"offline epoch {epoch + 1}/{forecast['epochs']}", flush=True)
        print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] offline training finished", flush=True)
        if hasattr(method, "reset_online_state"):
            method.reset_online_state()
        controller.policy = controller.capabilities.native_action()
        delay = tuple(range(1, dataset.horizon + 1)) if method.incremental_feedback else dataset.horizon
        native = OnlineExecutor(controller, feedback_delay=delay)
        requested = set(train_origins)
        state_values, gains, maes, origins = [], [], [], []
        with (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                  initializer=initialize_collection_worker, initargs=(dataset, threads, device))
              if workers > 1 else nullcontext()) as pool, \
                (output / "states.csv").open("w", encoding="utf-8", newline="") as state_file, \
                (output / "outcomes.csv").open("w", encoding="utf-8", newline="") as outcome_file:
            state_writer = csv.DictWriter(state_file, fieldnames=["state_id", "origin", *feature_names])
            outcome_writer = csv.DictWriter(outcome_file, fieldnames=["state_id", "origin", "action_id",
                                                                     "tau", "target_end_exclusive", "mae", "mse",
                                                                     "native_mae", "relative_mae_gain"])
            state_writer.writeheader()
            outcome_writer.writeheader()
            for index in range(offline_end - dataset.context_length, test_start):
                if index in requested:
                    action_indices = [number for number in range(len(catalog.actions))
                                      if number not in catalog.native_indices]
                    jobs = [catalog.capabilities.native_action()]
                    jobs.extend(catalog.actions[number] for number in action_indices)
                    if pool is None:
                        results = [score_action(native, dataset, index, action, steps=tau) for action in jobs]
                    else:
                        snapshot = io.BytesIO()
                        torch.save({"executor": native, "rng": rng_state()}, snapshot)
                        payload = snapshot.getvalue()
                        futures = [pool.submit(score_action_worker, payload, index, action, tau) for action in jobs]
                        results = [future.result() for future in futures]
                    baseline = results[0][1]
                    scores = {number: result[1] for number, result in zip(action_indices, results[1:])}
                    state = results[0][0]
                    state_id = len(origins)
                    state_writer.writerow({"state_id": state_id, "origin": index,
                                           **{name: state[name] for name in feature_names}})
                    row, mae_row = [], []
                    for number, (name, action) in enumerate(zip(catalog.ids, catalog.actions)):
                        score = baseline if number in catalog.native_indices else scores[number]
                        reference = baseline["mae"]
                        gain = 0. if number in catalog.native_indices else (reference - score["mae"]) / max(reference, 1e-12)
                        row.append(gain)
                        mae_row.append(score["mae"])
                        outcome_writer.writerow({"state_id": state_id, "origin": index,
                                                 "action_id": name, "tau": tau,
                                                 "target_end_exclusive": index + dataset.context_length + tau + dataset.horizon - 1,
                                                 **score, "native_mae": reference, "relative_mae_gain": gain})
                    state_values.append([state[name] for name in feature_names])
                    gains.append(row)
                    maes.append(mae_row)
                    origins.append(index)
                    state_file.flush()
                    outcome_file.flush()
                    print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] collected train origin={index}, actions={len(catalog.ids)}", flush=True)
                step_dataset(native, dataset, index)
                native._events.clear()
        np.savez(output / "training.npz", states=np.asarray(state_values, dtype=float),
                 gains=np.asarray(gains), maes=np.asarray(maes),
                 origins=np.asarray(origins), label_ends=np.asarray(origins) + tau + dataset.horizon - 1)
        torch.save({"executor": native, "dataset": dataset, "rng": rng_state()}, output / "evaluation_start.pt")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["seconds"] = perf_counter() - began
        save_json(output / "manifest.json", manifest)


def train_experiment(data_directory: Path, output: Path, options: dict | None = None) -> None:
    manifest = json.loads((data_directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("collection must complete before selector training")
    if manifest["schema_version"] != 4:
        raise ValueError("recollect training data with the current feature schema")
    options = manifest["config"]["selectors"] if options is None else options
    formulations, estimators = options["formulations"], options["estimators"]
    if "feature_sets" in options:
        raise ValueError("feature_sets is obsolete; ElasticNet selects from selectors.features or all collected features")
    for values, allowed, name in ((formulations, FORMULATIONS, "formulations"),
                                   (estimators, ESTIMATORS, "estimators")):
        if not values or len(set(values)) != len(values) or set(values) - set(allowed):
            raise ValueError(f"invalid or duplicate selector {name}")
    parameters = options.get("parameters", {})
    if set(parameters) - set(ESTIMATORS):
        raise ValueError("unknown estimator parameter section")
    formulation_parameters = options.get("formulation_parameters", {})
    if set(formulation_parameters) - set(FORMULATIONS):
        raise ValueError("unknown formulation parameter section")
    if any(set(values) - set(ESTIMATORS) for values in formulation_parameters.values()):
        raise ValueError("unknown estimator in formulation parameters")
    classification = options.get("classification", {})
    if set(classification) - {"temperature", "temperature_candidates", "cv_splits"}:
        raise ValueError("unknown classification temperature option")
    torch.set_num_threads(manifest["config"].get("threads", 1))
    capabilities = UpdateCapabilities(tuple(UpdateUnit(**row) for row in manifest["capabilities"]))
    catalog = ActionCatalog.from_dict(manifest["catalog"], capabilities)
    with np.load(data_directory / "training.npz") as values:
        states, gains, maes = values["states"], values["gains"], values["maes"]
        origins, label_ends = values["origins"], values["label_ends"]
    expected_ends = origins + manifest["tau"] + manifest["config"]["data"]["horizon"] - 1
    if not np.array_equal(label_ends, expected_ends) or np.any(label_ends + manifest["config"]["data"]["context_length"] > manifest["train_end"]):
        raise ValueError("historical action labels must end before the test split")
    collection_digest = hashlib.sha256((data_directory / "training.npz").read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    save_json(output / "manifest.json", {"status": "training", "data": str(data_directory.resolve()), "options": options})
    try:
        names = options.get("features", manifest["feature_names"])
        if not names or len(set(names)) != len(names) or set(names) - set(manifest["feature_names"]):
            raise ValueError("selectors.features must contain unique collected feature names")
        candidate_names = list(names)
        candidate_states = states[:, [manifest["feature_names"].index(name) for name in candidate_names]]
        selection_options = options.get("feature_selection", {})
        began = perf_counter()
        selection = select_state_features(candidate_states, gains, candidate_names, catalog.ids,
                                          catalog.native_indices, **selection_options)
        selection.update(seconds=perf_counter() - began, collection_sha256=collection_digest)
        save_json(output / "feature_selection.json", selection)
        names = selection["selected_features"]
        states = candidate_states[:, [candidate_names.index(name) for name in names]]
        print(f"ElasticNet selected {len(names)}/{len(candidate_names)} state features: {names}", flush=True)
        for formulation in formulations:
            for estimator in estimators:
                began = perf_counter()
                selector = ActionSelector(catalog, names, formulation, estimator,
                                          parameters={**parameters.get(estimator, {}),
                                                      **formulation_parameters.get(formulation, {}).get(estimator, {})},
                                          seed=manifest["config"].get("seed", 0),
                                          feature_window=manifest["config"]["collection"].get("feature_window", 32),
                                          error_window=manifest["config"]["collection"].get("error_window", 128))
                selector.tau = manifest["tau"]
                temperature_report = None
                if formulation == "classification":
                    temperature = classification.get("temperature", "auto")
                    if temperature == "auto":
                        temperature_selector = copy.copy(selector)
                        temperature_selector.features = candidate_names
                        temperature_report = select_temperature(
                            temperature_selector, candidate_states, maes, gains, origins, label_ends,
                            temperatures=classification.get("temperature_candidates", [.01, .02, .05, .1, .2]),
                            cv_splits=classification.get("cv_splits", 3), feature_selection=selection_options)
                        temperature = temperature_report["temperature"]
                    else:
                        temperature_report = {"method": "fixed", "temperature": temperature}
                    selector.temperature = temperature
                    print(f"classification/{estimator}: temperature={temperature}", flush=True)
                selector.fit(states, maes)
                fit_seconds = perf_counter() - began
                directory = output / formulation / estimator
                directory.mkdir(parents=True)
                selector.save(directory / "selector.pkl")
                if temperature_report is not None:
                    save_json(directory / "temperature_selection.json", temperature_report)
                if estimator == "mlp":
                    save_json(directory / "learning_rates.json", {
                        "schedule": "cosine", "unit": "epoch",
                        "models": [{"action_id": catalog.ids[i] if formulation == "independent_regression" else None,
                                    "rates": model.learning_rates_}
                                   for i, model in enumerate(selector.models) if model is not None]})
                row = {"formulation": formulation, "estimator": estimator, "feature_count": len(names),
                       "fit_seconds": fit_seconds, "model_bytes": (directory / "selector.pkl").stat().st_size}
                if formulation == "classification":
                    row.update(target="softmax_negative_mae", temperature=selector.temperature)
                else:
                    row.update(target="action_mae")
                results.append(row)
                save_json(directory / "metadata.json", {**row, "features": names, "catalog": catalog.to_dict(),
                                                        "action_encoding": catalog.encoding_names,
                                                        "parameters": selector.parameters,
                                                        "feature_selection": "elastic_net",
                                                        "feature_selection_parameters": selection["parameters"],
                                                        "feature_transforms": {name: "log1p" for name in names
                                                                               if name in LOG1P_FEATURES},
                                                        "collection_sha256": collection_digest})
                print(f"trained {formulation}/{estimator}", flush=True)
    except BaseException as error:
        save_json(output / "manifest.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "results": results})
        raise
    save_json(output / "manifest.json", {"status": "complete", "data": str(data_directory.resolve()),
                                         "options": options, "results": results})


def evaluate_policy(executor: OnlineExecutor, dataset, start: int, stop: int, output: Path,
                    *, name: str, target_mean: np.ndarray, target_std: np.ndarray) -> dict:
    """Persist issued forecasts separately from the later label-arrival clock."""
    if not 0 <= start < stop <= len(dataset):
        raise ValueError("invalid test forecast interval")
    output.mkdir(parents=True, exist_ok=False)
    controller = executor.method
    controller.last_decided_index = None
    controller.gradient_steps = controller.egd_steps = controller.gradient_update_clocks = controller.decision_count = 0
    controller.decision_seconds = controller.update_seconds = 0.
    controller._gradient_clock = None
    shape = (stop - start, dataset.horizon, dataset.num_targets)
    predictions = np.lib.format.open_memmap(output / "predictions.npy", mode="w+", dtype="float64", shape=shape)
    targets = np.lib.format.open_memmap(output / "targets.npy", mode="w+", dtype="float64", shape=shape)
    absolute = squared = raw_absolute = raw_squared = 0.
    decision_index, action_id = -1, name
    began = perf_counter()
    with (output / "forecasts.csv").open("w", encoding="utf-8", newline="") as forecast_file, \
            (output / "decisions.jsonl").open("w", encoding="utf-8") as decision_file:
        writer = csv.writer(forecast_file)
        writer.writerow(["forecast_index", "raw_context_end_index", "raw_target_start_index", "decision_index",
                         "action_id", "mae", "mse", "original_mae", "original_mse"])
        for offset, index in enumerate(range(start, stop)):
            prediction = step_dataset(executor, dataset, index, final=index == stop - 1).emission.prediction.double().numpy()
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"nonfinite test prediction at {index}")
            target = dataset[index][1].double().numpy()
            if controller.last_decision is not None:
                decision_index += 1
                record = {"decision_index": decision_index, **controller.last_decision}
                action_id = record.get("action_id", name)
                record["action_id"] = action_id
                decision_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            predictions[offset] = prediction * target_std + target_mean
            targets[offset] = target * target_std + target_mean
            error = prediction - target
            raw_error = error * target_std
            mae, mse = float(np.abs(error).mean()), float(np.square(error).mean())
            raw_mae, raw_mse = float(np.abs(raw_error).mean()), float(np.square(raw_error).mean())
            absolute += mae
            squared += mse
            raw_absolute += raw_mae
            raw_squared += raw_mse
            writer.writerow([index, index + dataset.context_length - 1, index + dataset.context_length,
                             decision_index, action_id, mae, mse, raw_mae, raw_mse])
            executor._events.clear()
        executor.flush()  # Scores outstanding labels only; no decisions or training.
        executor._events.clear()
    predictions.flush()
    targets.flush()
    if controller.method.device.type == "cuda":
        torch.cuda.synchronize(controller.method.device)
    summary = {"name": name, "forecasts": stop - start, "mae": absolute / (stop - start),
               "mse": squared / (stop - start), "original_mae": raw_absolute / (stop - start),
               "original_mse": raw_squared / (stop - start), "online_seconds": perf_counter() - began,
               "decision_seconds": controller.decision_seconds, "update_seconds": controller.update_seconds,
               "gradient_steps": controller.gradient_steps, "formula_steps": controller.egd_steps,
               "gradient_update_clocks": controller.gradient_update_clocks, "decisions": controller.decision_count,
               "test_start": start, "test_stop": stop, "target_names": list(dataset.target_names),
               "prediction_scale": "original numeric channel units", "metric_scale": "offline-standardized numeric channels",
               "array_axes": ["forecast (forecasts.csv order)", "horizon step (zero based)", "target (target_names order)"],
               "target_mean": target_mean.tolist(), "target_std": target_std.tolist()}
    save_json(output / "summary.json", summary)
    return summary


def initialize_evaluation_worker(device: str, threads: int) -> None:
    """Give each evaluation process one physical GPU and a private CUDA context."""
    global _evaluation_device, _evaluation_device_name
    requested = torch.device(device)
    _evaluation_device_name = device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(requested.index) if requested.type == "cuda" else ""
    _evaluation_device = torch.device("cuda:0" if requested.type == "cuda" else "cpu")
    torch.set_num_threads(threads)
    if requested.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def evaluate_policy_worker(data_directory: Path, manifest: dict, output: Path, name: str, policy) -> dict:
    """Restore a fresh OneNet trajectory, including optimizer state, for one policy."""
    saved = torch.load(data_directory / "evaluation_start.pt", weights_only=False, map_location="cpu")
    executor, dataset = saved["executor"], saved["dataset"]
    method = executor.method.method
    original_device = method.device
    original_parameters = tuple(method.model.parameters())
    method.model.to(_evaluation_device)
    parameters = dict(zip(original_parameters, method.model.parameters()))
    method.device = _evaluation_device
    # Keep optimizer groups intact: the adapter holds references to these groups.
    # OneNet's Adam/AdamW moment tensors follow parameters; step counters stay on CPU.
    for optimizer in method.update_optimizers:
        for group in optimizer.param_groups:
            group["params"] = [parameters[parameter] for parameter in group["params"]]
        for parameter in list(optimizer.state):
            moved = parameters[parameter]
            if moved is not parameter:
                optimizer.state[moved] = optimizer.state.pop(parameter)
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(_evaluation_device)
    torch.set_rng_state(saved["rng"][0])
    if _evaluation_device.type == "cuda":
        if saved["rng"][1]:
            torch.cuda.set_rng_state(saved["rng"][1][original_device.index or 0], device=0)
        else:
            torch.cuda.manual_seed(manifest["config"].get("seed", 0))
    executor.method.policy = policy
    summary = evaluate_policy(executor, dataset, manifest["test_start"], manifest["test_stop"],
                              output / name, name=name,
                              target_mean=np.asarray(manifest["target_mean"]),
                              target_std=np.asarray(manifest["target_std"]))
    summary["device"] = _evaluation_device_name
    save_json(output / name / "summary.json", summary)
    return summary


def evaluate_experiment(data_directory: Path, model_directory: Path, output: Path, *,
                        devices: list[str] | None = None, workers_per_device: list[int] | None = None) -> None:
    manifest = json.loads((data_directory / "manifest.json").read_text(encoding="utf-8"))
    trained = json.loads((model_directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or trained["status"] != "complete":
        raise ValueError("collection and model training must both complete")
    torch.set_num_threads(manifest["config"].get("threads", 1))
    capabilities = UpdateCapabilities(tuple(UpdateUnit(**row) for row in manifest["capabilities"]))
    catalog = ActionCatalog.from_dict(manifest["catalog"], capabilities)
    if devices is None and workers_per_device is not None:
        raise ValueError("workers_per_device requires devices")
    if devices is not None:
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("evaluation devices must be nonempty and unique")
        for name in devices:
            device = torch.device(name)
            if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and device.index is None):
                raise ValueError("evaluation devices must be cpu or indexed CUDA devices such as cuda:2")
        workers_per_device = [1] * len(devices) if workers_per_device is None else workers_per_device
        if len(workers_per_device) != len(devices):
            raise ValueError("workers_per_device must provide one count per device")
        for count in workers_per_device:
            positive_integer(count, "workers_per_device")
        if catalog.method != "onenet":
            raise ValueError("parallel evaluation currently supports OneNet only")
    digest = hashlib.sha256((data_directory / "training.npz").read_bytes()).hexdigest()
    jobs = [("native", catalog.actions[catalog.native_indices[0]]),
            ("frozen", UpdateAction())]
    for row in trained["results"]:
        directory = model_directory / row["formulation"] / row["estimator"]
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["collection_sha256"] != digest or metadata["catalog"] != catalog.to_dict():
            raise ValueError("selector was trained on a different collection or candidate catalog")
        selector = ActionSelector.load(directory / "selector.pkl")
        jobs.append((f"policy/{row['formulation']}/{row['estimator']}", SelectorPolicy(selector)))
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    save_json(output / "manifest.json", {"status": "evaluating"})
    try:
        if devices is None:
            saved = torch.load(data_directory / "evaluation_start.pt", weights_only=False)
            source, dataset = saved["executor"], saved["dataset"]
            for name, policy in jobs:
                executor = copy.deepcopy(source)
                executor.method.policy = policy
                restore_rng(saved["rng"])
                summaries.append(evaluate_policy(executor, dataset, manifest["test_start"], manifest["test_stop"],
                                                 output / name, name=name,
                                                 target_mean=np.asarray(manifest["target_mean"]),
                                                 target_std=np.asarray(manifest["target_std"])))
                print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] tested {name}: MAE={summaries[-1]['mae']:.6f}", flush=True)
                del executor
        else:
            pools, pending = [], {}
            remaining = iter(jobs)
            try:
                for device, count in zip(devices, workers_per_device):
                    pool = ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn"),
                                               initializer=initialize_evaluation_worker,
                                               initargs=(device, manifest["config"].get("threads", 1)))
                    pools.append(pool)
                    for _ in range(count):
                        job = next(remaining, None)
                        if job is not None:
                            name, policy = job
                            pending[pool.submit(evaluate_policy_worker, data_directory, manifest, output, name, policy)] = pool
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        pool = pending.pop(future)
                        summary = future.result()
                        summaries.append(summary)
                        print(f"[{datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}] tested {summary['name']} on {summary['device']}: MAE={summary['mae']:.6f}", flush=True)
                        job = next(remaining, None)
                        if job is not None:
                            name, policy = job
                            pending[pool.submit(evaluate_policy_worker, data_directory, manifest, output, name, policy)] = pool
            finally:
                for future in pending:
                    future.cancel()
                for pool in pools:
                    pool.shutdown(wait=True, cancel_futures=True)
            order = {name: index for index, (name, _) in enumerate(jobs)}
            summaries.sort(key=lambda row: order[row["name"]])
    except BaseException as error:
        save_json(output / "manifest.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "results": summaries})
        raise
    save_json(output / "manifest.json", {"status": "complete", "results": summaries})
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [key for key, value in summaries[0].items() if not isinstance(value, list)]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
