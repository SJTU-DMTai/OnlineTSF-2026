"""Collect causal counterfactuals, fit selectors, and evaluate continuous streams."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..online import OnlineExecutor
from .actions import UpdateAction, positive_integer
from .catalog import ActionCatalog
from .capabilities import UpdateCapabilities, UpdateUnit
from .feature_selection import select_state_features
from .collection import collect_experiment
from .selector import ActionSelector, ESTIMATORS, FORMULATIONS, LOG1P_FEATURES, SelectorPolicy, resolve_ema_beta
from .training import select_temperature
from .ranking import RANKING_PAIRS, RankingSelector
from .rollout import RecordingPolicy, rng_state, restore_rng, step_dataset
from .preparation import save_json


def score_action(source: OnlineExecutor, dataset, origin: int, action: UpdateAction, *,
                 steps: int = 1, stop: int | None = None) -> tuple[dict, dict]:
    """Fixed-action diagnostics for the separate greedy hindsight baseline."""
    positive_integer(steps, "steps")
    saved_rng = rng_state()
    try:
        branch = copy.deepcopy(source)
        policy = RecordingPolicy(action)
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


def train_experiment(data_directory: Path, output: Path, options: dict | None = None) -> None:
    manifest = json.loads((data_directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("collection must complete before selector training")
    if manifest["schema_version"] != 5:
        raise ValueError("recollect training data with the current feature schema")
    options = manifest["config"]["selectors"] if options is None else options
    if "formulations" in options or "estimators" in options:
        raise ValueError("replace selectors.formulations/estimators with an explicit selectors.models list")
    models = options.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("selectors.models must be a nonempty list of formulation/estimator pairs")
    pairs = []
    for model in models:
        if (not isinstance(model, dict) or not {"formulation", "estimator"} <= set(model)
                or set(model) - {"formulation", "estimator", "loss"}):
            raise ValueError("each selectors.models entry requires formulation and estimator")
        pair = model["formulation"], model["estimator"], model.get("loss", "default")
        if pair[2] not in {"default", "margin_ranking"}:
            raise ValueError(f"unknown selector loss: {pair[2]}")
        if pair[2] == "margin_ranking" and pair[:2] not in RANKING_PAIRS:
            raise ValueError(f"unsupported margin ranking selector: {pair[:2]}")
        if pair[0] not in FORMULATIONS or pair[1] not in ESTIMATORS:
            raise ValueError(f"unknown selector formulation/estimator: {pair}")
        if pair in pairs:
            raise ValueError(f"duplicate selector formulation/estimator: {pair}")
        pairs.append(pair)
    ranking = options.get("ranking", {})
    if set(ranking) - {"margin"}:
        raise ValueError("unknown selector ranking option")
    if "feature_sets" in options:
        raise ValueError("feature_sets is obsolete; ElasticNet selects from selectors.features or all collected features")
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
        states, gains, maes = values["states"], values["advantages"], values["maes"]
        origins, label_ends = values["origins"], values["label_ends"]
    expected_ends = origins + manifest["h_act"] + manifest["h_tail"] + manifest["config"]["data"]["horizon"] - 1
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
        if selection_options is None:
            selection = {"method": "none", "candidate_features": candidate_names,
                         "selected_features": list(candidate_names), "parameters": {}}
        else:
            selection = select_state_features(candidate_states, gains, candidate_names, catalog.ids,
                                              catalog.native_indices, **selection_options)
        selection.update(seconds=perf_counter() - began, collection_sha256=collection_digest)
        save_json(output / "feature_selection.json", selection)
        names = selection["selected_features"]
        states = candidate_states[:, [candidate_names.index(name) for name in names]]
        print(f"Feature selection ({selection['method']}): {len(names)}/{len(candidate_names)} state features: {names}", flush=True)
        for formulation, estimator, loss in pairs:
            began = perf_counter()
            selector_type = RankingSelector if loss == "margin_ranking" else ActionSelector
            selector = selector_type(catalog, names, formulation, estimator,
                                      **({"margin": ranking.get("margin", 1.)} if loss == "margin_ranking" else {}),
                                      parameters={**parameters.get(estimator, {}),
                                                  **formulation_parameters.get(formulation, {}).get(estimator, {})},
                                      seed=manifest["config"].get("seed", 0),
                                      feature_window=manifest["config"]["collection"].get("feature_window", 32),
                                      error_window=manifest["config"]["collection"].get("error_window", 128))
            selector.tau = manifest["tau"]
            temperature_report = None
            if formulation == "classification" and loss == "default":
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
            model_name = estimator + ("_margin_ranking" if loss == "margin_ranking" else "")
            directory = output / formulation / model_name
            directory.mkdir(parents=True)
            selector.save(directory / "selector.pkl")
            if temperature_report is not None:
                save_json(directory / "temperature_selection.json", temperature_report)
            if estimator == "mlp":
                save_json(directory / "learning_rates.json", {
                    "schedule": "cosine", "unit": "epoch",
                    "models": [{"action_id": catalog.ids[i] if formulation == "independent_regression" and loss == "default" else None,
                                "rates": model.learning_rates_}
                               for i, model in enumerate(selector.models) if model is not None]})
            row = {"formulation": formulation, "estimator": estimator, "feature_count": len(names),
                   "fit_seconds": fit_seconds, "model_bytes": (directory / "selector.pkl").stat().st_size}
            if loss == "margin_ranking":
                row.update(target="margin_ranking", loss=loss, margin=selector.margin,
                           pair_reduction="mean_per_state_then_mean_states", ties="skip_exact",
                           score_kind=selector.score_kind)
            elif formulation == "classification":
                row.update(target="softmax_negative_regret", temperature=selector.temperature)
            else:
                row.update(target="relative_advantage")
            results.append(row)
            save_json(directory / "metadata.json", {**row, "features": names, "catalog": catalog.to_dict(),
                                                    "action_encoding": catalog.encoding_names,
                                                    "parameters": selector.parameters,
                                                    "feature_selection": selection["method"],
                                                    "feature_selection_parameters": selection["parameters"],
                                                    "feature_transforms": {name: "log1p" for name in names
                                                                           if name in LOG1P_FEATURES},
                                                    "collection_sha256": collection_digest})
            print(f"trained {formulation}/{model_name}", flush=True)
    except BaseException as error:
        save_json(output / "manifest.json", {"status": "failed", "error": f"{type(error).__name__}: {error}", "results": results})
        raise
    save_json(output / "manifest.json", {"status": "complete", "data": str(data_directory.resolve()),
                                         "options": options, "results": results})


def evaluate_policy(executor: OnlineExecutor, dataset, start: int, stop: int, output: Path,
                    *, name: str, target_mean: np.ndarray, target_std: np.ndarray, terminal: bool = True) -> dict:
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
            prediction = step_dataset(executor, dataset, index, final=terminal and index == stop - 1).emission.prediction.double().numpy()
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
    """Restore a fresh trajectory, including optimizer state, for one policy."""
    saved = torch.load(data_directory / manifest["prepared_data"] / "evaluation_start.pt", weights_only=False, map_location="cpu")
    executor, dataset = saved["executor"], saved["dataset"]
    method = executor.method.method
    original_device = method.device
    original_parameters = tuple(method.model.parameters())
    method.model.to(_evaluation_device)
    parameters = dict(zip(original_parameters, method.model.parameters()))
    method.device = _evaluation_device
    # Keep optimizer groups intact: the adapter holds references to these groups.
    # Adam/AdamW moment tensors follow parameters; step counters stay on CPU.
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
                        devices: list[str] | None = None, workers_per_device: list[int] | None = None,
                        ema_beta: float | None = None) -> None:
    manifest = json.loads((data_directory / "manifest.json").read_text(encoding="utf-8"))
    trained = json.loads((model_directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or trained["status"] != "complete":
        raise ValueError("collection and model training must both complete")
    beta = resolve_ema_beta(ema_beta if ema_beta is not None else
                            manifest["config"].get("evaluation", {}).get("ema_beta"), manifest["tau"])
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
    digest = hashlib.sha256((data_directory / "training.npz").read_bytes()).hexdigest()
    jobs = [("native", catalog.actions[catalog.native_indices[0]]),
            ("frozen", UpdateAction())]
    for row in trained["results"]:
        model_name = row["estimator"] + ("_margin_ranking" if row.get("loss") == "margin_ranking" else "")
        directory = model_directory / row["formulation"] / model_name
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["collection_sha256"] != digest or metadata["catalog"] != catalog.to_dict():
            raise ValueError("selector was trained on a different collection or candidate catalog")
        selector = ActionSelector.load(directory / "selector.pkl")
        jobs.append((f"policy/{row['formulation']}/{model_name}", SelectorPolicy(selector, ema_beta=beta)))
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    save_json(output / "manifest.json", {"status": "evaluating", "ema_beta": beta})
    try:
        if devices is None:
            saved = torch.load(data_directory / manifest["prepared_data"] / "evaluation_start.pt", weights_only=False)
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
        save_json(output / "manifest.json", {"status": "failed", "ema_beta": beta, "error": f"{type(error).__name__}: {error}", "results": summaries})
        raise
    save_json(output / "manifest.json", {"status": "complete", "ema_beta": beta, "results": summaries})
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [key for key, value in summaries[0].items() if not isinstance(value, list)]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
