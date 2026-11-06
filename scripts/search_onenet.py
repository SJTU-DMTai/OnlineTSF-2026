"""Search OneNet update units, independent rate scales and holding durations.

Only forecasts issued in [start, start + duration) are scored, with full-horizon
MAE. No updates, predictions or continuation actions run after that interval.
Run ``python scripts/search_onenet.py --help`` for the experiment controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import optuna
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onlinetsf.data import SlidingWindowDataset, load_benchmark_dataset
from onlinetsf.decision import UpdateAction, UpdateController
from onlinetsf.decision.adapters import OneNetAdapter
from onlinetsf.decision.capabilities import UnitSchedule
from onlinetsf.factory import build_backbone, build_method
from onlinetsf.online import OnlineExecutor
from scripts.experiment_onenet import restore_rng, rng_state, synchronize


DEFAULT_DURATIONS = (32, 64, 128, 256, 512)


def suggest_action(trial, durations: list[int], scale_low: float, scale_high: float) -> UpdateAction:
    units, scales = set(), {}
    for unit in OneNetAdapter.capabilities.units:
        if trial.suggest_categorical(f"enable_{unit.name}", [False, True]):
            units.add(unit.name)
            if unit.has_learning_rate:
                scales[unit.name] = trial.suggest_float(
                    f"lr_scale_{unit.name}", scale_low, scale_high, log=True)
    return UpdateAction(units=units, learning_rate_scales=scales,
                        duration=trial.suggest_categorical("duration", durations))


def action_key(action: UpdateAction, *, include_duration: bool = True) -> str:
    record = {"units": sorted(action.units),
              "learning_rate_scales": {name: value for name, value in action.learning_rate_scales.items()
                                       if value != 1.0}}
    if include_duration:
        record["duration"] = action.duration
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def select_origins(first_target: int, end_row: int, context: int, horizon: int,
                   max_duration: int, count: int) -> list[int]:
    """Keep every target, including the last horizon, inside this raw-row split."""
    first = first_target - context
    last = end_row - context - max_duration - horizon + 1
    if first < 0 or count <= 0 or last - first + 1 < count:
        raise ValueError("interval is too short for the requested states and largest duration")
    return np.linspace(first, last, count, dtype=np.int64).tolist()


def score_prefixes(executor: OnlineExecutor, dataset: SlidingWindowDataset, start: int,
                   durations: list[int]) -> dict[int, dict]:
    """Reuse identical prediction prefixes across durations; never flush or continue."""
    results = {}
    absolute_sum = squared_sum = 0.0
    values = 0
    requested = set(durations)
    for offset in range(max(durations)):
        context, target = dataset[start + offset]
        prediction = executor.step(start + offset, context, target).emission.prediction
        if not torch.isfinite(prediction).all():
            for duration in durations:
                if duration not in results:
                    results[duration] = {"mae": None, "mse": None, "status": "nonfinite"}
            break
        error = prediction.double() - target.double()
        absolute_sum += error.abs().sum().item()
        squared_sum += error.square().sum().item()
        values += error.numel()
        if offset + 1 in requested:
            results[offset + 1] = {"mae": absolute_sum / values, "mse": squared_sum / values,
                                   "status": "complete"}
    return results


def collect_snapshots(native: OnlineExecutor, dataset: SlidingWindowDataset, start: int,
                      origins: list[int], directory: Path) -> dict[int, Path]:
    """Save the native state before the origin's feedback, including RNG and pending labels."""
    paths = {}
    requested = set(origins)
    device = native.method.method.device
    for index in range(start, max(origins) + 1):
        if index in requested:
            scheduler = native.state_dict()
            scheduler["events"] = []
            paths[index] = directory / f"state-{index}.pt"
            torch.save({"controller": native.method, "scheduler": scheduler,
                        "rng": rng_state(device)}, paths[index])
            print(f"snapshot origin={index}", flush=True)
        if index < max(origins):
            native.step(index, *dataset[index])
            # Aggregate metrics remain valid; individual historical events are not needed.
            native._events.clear()
    return paths


class ActionEvaluator:
    """Independent causal branches, exact prefix caching and same-window baselines."""

    def __init__(self, dataset: SlidingWindowDataset, snapshots: dict[int, Path],
                 durations: list[int], log_path: Path) -> None:
        self.dataset = dataset
        self.snapshots = snapshots
        self.durations = durations
        self.log_path = log_path
        self.cache: dict[tuple[int, str], dict[int, dict]] = {}
        self.baselines: dict[int, dict[int, dict]] = {}
        self.branch_count = 0

    def evaluate(self, origin: int, action: UpdateAction) -> dict:
        key = (origin, action_key(action, include_duration=False))
        cached = self.cache.setdefault(key, {})
        if action.duration not in cached:
            saved = torch.load(self.snapshots[origin], weights_only=False)
            controller = saved["controller"]
            # Loading owns fresh storage, so a second deepcopy is unnecessary.
            controller.policy = action
            controller.expires_at = None
            controller.schedule = UnitSchedule()
            executor = OnlineExecutor(controller, feedback_delay=self.dataset.horizon)
            executor.load_state_dict(saved["scheduler"])
            restore_rng(saved["rng"], controller.method.device)
            prefixes = [duration for duration in self.durations if duration <= action.duration]
            scores = score_prefixes(executor, self.dataset, origin, prefixes)
            self.branch_count += 1
            with self.log_path.open("a", encoding="utf-8") as handle:
                for duration, metrics in scores.items():
                    if duration in cached:
                        continue
                    record = action.to_dict()
                    record["duration"] = duration
                    handle.write(json.dumps({"origin": origin, "action": record, **metrics},
                                            allow_nan=False) + "\n")
            cached.update(scores)
        return cached[action.duration]

    def prepare_baselines(self, origins: list[int]) -> None:
        action = OneNetAdapter.capabilities.native_action(duration=max(self.durations))
        for origin in origins:
            self.evaluate(origin, action)
            self.baselines[origin] = dict(self.cache[(origin, action_key(action, include_duration=False))])
            if any(row["status"] != "complete" for row in self.baselines[origin].values()):
                raise FloatingPointError(f"nonfinite Native forecast at origin {origin}")

    def ratios(self, action: UpdateAction, origins: list[int]) -> list[float]:
        ratios = []
        for origin in origins:
            score = self.evaluate(origin, action)
            ratios.append(score["mae"] / max(self.baselines[origin][action.duration]["mae"], 1e-12)
                          if score["status"] == "complete" else math.inf)
        return ratios


def run_study(evaluator: ActionEvaluator, origins: list[int], args, name: str,
              trials: int, seed: int, *, enqueue_baselines: bool = False,
              deadline: float | None = None) -> optuna.Study:
    study = optuna.create_study(
        study_name=name, storage=f"sqlite:///{(args.output / 'optuna.db').as_posix()}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(20, max(1, trials // 3)),
                                          multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20,
                                          n_warmup_steps=max(1, len(origins) // 2), n_min_trials=5))
    if enqueue_baselines:
        # Matched references at every H, not a hand-picked restriction of the search space.
        for duration in args.durations:
            for units in (OneNetAdapter.capabilities.unit_names, frozenset()):
                params = {f"enable_{unit.name}": unit.name in units for unit in OneNetAdapter.capabilities.units}
                params.update({f"lr_scale_{unit.name}": 1.0 for unit in OneNetAdapter.capabilities.units
                               if unit.name in units and unit.has_learning_rate})
                study.enqueue_trial({**params, "duration": duration})
    for _ in range(trials):
        if deadline is not None and perf_counter() >= deadline:
            break
        trial = study.ask()
        action = suggest_action(trial, args.durations, args.lr_scale_low, args.lr_scale_high)
        trial.set_user_attr("action", action.to_dict())
        ratios = []
        try:
            for step, origin in enumerate(origins, start=1):
                if deadline is not None and perf_counter() >= deadline:
                    raise optuna.TrialPruned("search time budget exhausted")
                ratio = evaluator.ratios(action, [origin])[0]
                if not math.isfinite(ratio):
                    raise optuna.TrialPruned("nonfinite predictions")
                ratios.append(ratio)
                trial.report(float(np.mean(ratios)), step)
                if len(origins) > 1 and step < len(origins) and trial.should_prune():
                    raise optuna.TrialPruned()
            study.tell(trial, float(np.mean(ratios)))
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        except BaseException:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            raise
        print(f"{name} trial={trial.number} H={action.duration} units={','.join(sorted(action.units)) or 'frozen'} "
              f"states={len(ratios)}/{len(origins)} ratio={np.mean(ratios) if ratios else 'invalid'}", flush=True)
    return study


def select_portfolio(actions: list[UpdateAction], ratios: np.ndarray, count: int,
                     anchor: UpdateAction) -> list[int]:
    """Greedy state coverage, starting with Native and using prediction loss only."""
    selected = [next(index for index, action in enumerate(actions) if action_key(action) == action_key(anchor))]
    best = ratios[selected[0]].copy()
    while len(selected) < min(count, len(actions)):
        scores = np.minimum(ratios, best).mean(axis=1)
        scores[selected] = math.inf
        index = int(np.argmin(scores))
        if scores[index] >= best.mean() - 1e-12:
            break
        selected.append(index)
        best = np.minimum(best, ratios[index])
    return selected


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/ETTm1.csv")
    parser.add_argument("--profiles", type=Path, default=root / "profiles.yaml")
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing runs are never overwritten")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--durations", nargs="+", type=int, default=list(DEFAULT_DURATIONS))
    parser.add_argument("--lr-scale-low", type=float, default=.01)
    parser.add_argument("--lr-scale-high", type=float, default=10.)
    parser.add_argument("--offline-ratio", type=float, default=.2)
    parser.add_argument("--search-end", type=float, default=.4)
    parser.add_argument("--validation-end", type=float, default=.6)
    parser.add_argument("--states", type=int, default=12, help="uniformly spaced search snapshots")
    parser.add_argument("--validation-states", type=int, default=6)
    parser.add_argument("--trials", type=int, default=100, help="global trials including 2 references per duration")
    parser.add_argument("--timeout", type=float, default=1800,
                        help="seconds for all Optuna trials combined; excludes preparation and final validation")
    parser.add_argument("--local-trials", type=int, default=8, help="extra trials per search state; 0 disables")
    parser.add_argument("--pool-size", type=int, default=24, help="maximum actions evaluated on validation states")
    parser.add_argument("--candidates", type=int, default=8, help="maximum final portfolio size, including Native")
    args = parser.parse_args(argv)
    if min(args.context_length, args.horizon, args.epochs, args.batch_size, args.threads,
           args.states, args.validation_states, args.trials, args.pool_size, args.candidates, *args.durations) <= 0:
        parser.error("lengths and counts must be positive")
    if args.local_trials < 0 or args.candidates > args.pool_size:
        parser.error("local-trials must be nonnegative and candidates cannot exceed pool-size")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if (not math.isfinite(args.lr_scale_low) or not math.isfinite(args.lr_scale_high)
            or not 0 < args.lr_scale_low <= 1 <= args.lr_scale_high):
        parser.error("finite learning rate bounds must satisfy 0 < low <= 1 <= high")
    if not 0 < args.offline_ratio < args.search_end < args.validation_end <= 1:
        parser.error("require 0 < offline-ratio < search-end < validation-end <= 1")
    if len(set(args.durations)) != len(args.durations):
        parser.error("durations must be distinct")
    args.durations.sort()
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("device must be cpu or cuda[:index]")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dataset = load_benchmark_dataset("ettm1", args.data, args.context_length, args.horizon,
                                     include_time_features=True)
    offline_end, search_end, validation_end = [int(len(dataset.values) * ratio) for ratio in
                                              (args.offline_ratio, args.search_end, args.validation_end)]
    train_windows = offline_end - args.context_length - args.horizon + 1
    if train_windows <= 0:
        parser.error("offline interval is shorter than context plus horizon")
    # First search snapshot has matured feedback, while all its labels remain historical.
    search_origins = select_origins(offline_end + args.horizon, search_end, args.context_length,
                                    args.horizon, max(args.durations), args.states)
    validation_origins = select_origins(search_end, validation_end, args.context_length,
                                        args.horizon, max(args.durations), args.validation_states)
    prefix = dataset.values[:offline_end]
    std = prefix.std(dim=0, unbiased=False)
    scaler = {"mean": prefix.mean(dim=0).tolist(),
              "std": torch.where(std < 1e-6, torch.ones_like(std), std).tolist()}
    dataset.standardize(train_windows)
    with args.profiles.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)["strategies"]["onenet"]
    if profile["method"]["name"] != "onenet":
        raise ValueError("profiles.strategies.onenet must specify method.name: onenet")
    config = {"data": {"context_length": args.context_length, "horizon": args.horizon},
              "forecasting": profile["forecasting"], "method": profile["method"],
              "online": {"device": str(device)}}
    method = build_method(config, build_backbone(config, dataset.num_features,
                                                dataset.num_targets, dataset.target_indices))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "training", "arguments": {key: str(value) if isinstance(value, Path) else value
                                                    for key, value in vars(args).items()},
                "config": config, "optuna_version": optuna.__version__, "scaler": scaler,
                "feature_names": list(dataset.feature_names), "target_names": list(dataset.target_names),
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "raw_rows": len(dataset.values), "offline_end": offline_end,
                "search_end": search_end, "validation_end": validation_end,
                "search_origins": search_origins, "validation_origins": validation_origins,
                "objective": "mean of per-state MAE / Native MAE over identical duration windows",
                "scoring": "full horizons of forecasts issued in [origin, origin + H); no continuation or flush",
                "periods": "not searched; all selected units retain native period 1",
                "snapshot_timing": "before origin feedback; native trajectory; optimizer and pending forecasts retained"}
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    began = perf_counter()
    try:
        loader = DataLoader(Subset(dataset, range(train_windows)), batch_size=args.batch_size, shuffle=True)
        for epoch in range(args.epochs):
            for context, target in loader:
                if not math.isfinite(method.train_batch(context, target)):
                    raise FloatingPointError("nonfinite offline loss")
            method.end_offline_epoch(epoch + 1)
            print(f"offline epoch={epoch + 1}/{args.epochs}", flush=True)
        method.reset_online_state()
        native = OnlineExecutor(UpdateController(method, OneNetAdapter.capabilities.native_action(
            duration=len(dataset))), feedback_delay=args.horizon)
        with TemporaryDirectory(prefix=".snapshots-", dir=args.output) as temporary:
            paths = collect_snapshots(native, dataset, offline_end - args.context_length,
                                      search_origins + validation_origins, Path(temporary))
            del native, method
            evaluator = ActionEvaluator(dataset, paths, args.durations, args.output / "evaluations.jsonl")
            evaluator.prepare_baselines(search_origins)
            manifest["status"] = "searching"
            manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
            # A fixed permutation prevents pruning from considering only the earliest states.
            order = np.random.default_rng(args.seed).permutation(search_origins).tolist()
            deadline = perf_counter() + args.timeout
            studies = [run_study(evaluator, order, args, "global", args.trials, args.seed,
                                 enqueue_baselines=True, deadline=deadline)]
            for number, origin in enumerate(search_origins):
                if args.local_trials and perf_counter() < deadline:
                    studies.append(run_study(evaluator, [origin], args, f"state-{origin}",
                                              args.local_trials, args.seed + number + 1, deadline=deadline))
            manifest["search_budget_exhausted"] = perf_counter() >= deadline
            anchor = OneNetAdapter.capabilities.native_action(duration=args.durations[len(args.durations) // 2])
            unique = {action_key(anchor): anchor}
            for study in studies:
                completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
                # Local specialists are fully rescored before entering the shared pool.
                retained = completed if study.study_name == "global" else sorted(completed, key=lambda t: t.value)[:2]
                for trial in retained:
                    action = UpdateAction.from_dict(trial.user_attrs["action"])
                    unique.setdefault(action_key(action), action)
            actions = list(unique.values())
            search_ratios = np.asarray([evaluator.ratios(action, search_origins) for action in actions])
            valid = np.isfinite(search_ratios).all(axis=1)
            actions = [action for action, keep in zip(actions, valid) if keep]
            search_ratios = search_ratios[valid]
            pool = select_portfolio(actions, search_ratios, args.pool_size, anchor)
            actions = [actions[index] for index in pool]
            search_ratios = search_ratios[pool]
            evaluator.prepare_baselines(validation_origins)
            validation_ratios = np.asarray([evaluator.ratios(action, validation_origins) for action in actions])
            # Validation failures exclude a candidate, never silently average fewer states.
            valid = np.isfinite(validation_ratios).all(axis=1)
            actions = [action for action, keep in zip(actions, valid) if keep]
            search_ratios, validation_ratios = search_ratios[valid], validation_ratios[valid]
            selected = select_portfolio(actions, validation_ratios, args.candidates, anchor)
            records = [{"id": f"a{index:03d}", "action": action.to_dict(),
                        "search_ratio": float(search_ratios[index].mean()),
                        "validation_ratio": float(validation_ratios[index].mean()),
                        "search_state_ratios": search_ratios[index].tolist(),
                        "validation_state_ratios": validation_ratios[index].tolist()}
                       for index, action in enumerate(actions)]
            (args.output / "pool.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
            (args.output / "candidates.json").write_text(json.dumps(
                [records[index] for index in selected], indent=2, allow_nan=False), encoding="utf-8")
            manifest.update(status="complete", branch_evaluations=evaluator.branch_count,
                            pool_size=len(actions), candidate_count=len(selected),
                            validation_portfolio_ratio=float(validation_ratios[selected].min(axis=0).mean()))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        synchronize(device)
        manifest["seconds"] = perf_counter() - began
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
