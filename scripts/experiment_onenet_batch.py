"""Search OneNet feedback batch sizes in full/branches experiments.

All five update units stay enabled at learning rate scale 1. Period p collects
p complete labeled windows, then takes one mean-loss step per optimizer. An
incomplete final batch is updated before the segment's last prediction.
The long-term combination step is scaled by the actual batch size so its
logit increment sums, rather than averages, the samples' expert loss gaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_onenet import (
    fork_executor, restore_rng, rng_state, run_segment,
    stream_partition, synchronize,
)
from onlinetsf.data import SlidingWindowDataset, load_benchmark_dataset
from onlinetsf.decision import UpdateAction, UpdateController
from onlinetsf.decision.adapters import OneNetAdapter, PreparedFeedback
from onlinetsf.factory import build_backbone, build_method
from onlinetsf.online import OnlineExecutor


DEFAULT_PERIODS = (1, 2, 4, 8, 16, 32)
_WORKER_DATASET: SlidingWindowDataset


def build_batch_actions(periods, duration: int) -> list[dict]:
    """Search batch size only, with all units enabled and native learning rates."""
    units = OneNetAdapter.capabilities.unit_names
    return [{"action_id": f"a{index:04d}", "sweep": "period", "mask": (1 << len(units)) - 1,
             "units": "+".join(unit.name for unit in OneNetAdapter.capabilities.units),
             "lr_scale": 1.0, "period": period,
             "action": UpdateAction(units=units, periods={name: period for name in units}, duration=duration)}
            for index, period in enumerate(periods)]


class BatchOneNetAdapter(OneNetAdapter):
    """Record sample and batch counts for the shared action executor."""

    def __init__(self, method, action: UpdateAction) -> None:
        if method.n_inner != 1:
            raise ValueError("batch experiments require OneNet n_inner=1 for one step per optimizer")
        super().__init__(method)
        self.batch_size = max(action.periods.values(), default=1)
        self.feedback_samples = self.samples_used = self.batch_updates = self.tail_batch_updates = 0

    def apply(self, prepared: PreparedFeedback, due, schedule) -> float | None:
        self.feedback_samples += 1
        return super().apply(prepared, due, schedule)

    def apply_batches(self, batches, schedule) -> float | None:
        loss = super().apply_batches(batches, schedule)
        for _, items in batches:
            self.samples_used += len(items)
            self.batch_updates += 1
            self.tail_batch_updates += int(len(items) < self.batch_size)
        return loss


def run_batch_configuration(item, source, scheduler_state, saved_rng, dataset, start, stop,
                            forecasts_path) -> dict:
    """One-factor OneNet sweeps: collect p complete feedback samples, update once.

    All selected optimizers take one step on the batch mean loss. At the final
    prediction clock, any smaller remaining batch is also updated. The original
    native online method supplies every branch's initial state. n_inner must be 1.
    Online batch sizes come from --periods; --batch-size controls offline training.
    """
    restore_rng(saved_rng, source.method.device)
    synchronize(source.method.device)
    began = perf_counter()
    branch = fork_executor(source, scheduler_state, item["action"])
    adapter = BatchOneNetAdapter(branch.method.method, item["action"])
    branch.method.adapter = adapter
    synchronize(source.method.device)
    fork_seconds = perf_counter() - began
    result = run_segment(branch, dataset, start, stop, forecasts_path)
    return {"fork_seconds": fork_seconds, **result, "update_rule": "batch_all_feedback",
            "batch_size": adapter.batch_size, "feedback_samples": adapter.feedback_samples,
            "samples_used": adapter.samples_used, "batch_updates": adapter.batch_updates,
            "tail_batch_updates": adapter.tail_batch_updates,
            "buffered_samples": max((len(items) for items in adapter.batches.pending.values()), default=0)}


def initialize_worker(dataset: SlidingWindowDataset, threads: int, device: torch.device) -> None:
    global _WORKER_DATASET
    _WORKER_DATASET = dataset
    torch.set_num_threads(threads)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_batch_worker(task: tuple) -> dict:
    item, checkpoint_path, start, stop, forecasts_path = task
    saved = torch.load(checkpoint_path, weights_only=False)
    return run_batch_configuration(item, saved["controller"], saved["scheduler"], saved["rng"],
                                   _WORKER_DATASET, start, stop, forecasts_path)


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("full", "branches"), help="continuous trajectories or native-state branches")
    parser.add_argument("--data", type=Path, default=root / "data/ETTm1.csv")
    parser.add_argument("--profiles", type=Path, default=root / "configs/profiles.yaml")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1,
                        help="independent configuration processes sharing the selected device (default: 1)")
    parser.add_argument("--periods", type=int, nargs="+", default=list(DEFAULT_PERIODS))
    parser.add_argument("--save-forecasts", action=argparse.BooleanOptionalAction, default=True,
                        help="save per-origin errors/costs (CSV) and predictions/targets (NPZ); enabled by default")
    parser.add_argument("--windows", type=int, default=32, help="number of windows in branches mode (default: 32)")
    args = parser.parse_args(argv)
    mode = args.mode
    if min(args.context_length, args.horizon, args.epochs, args.batch_size, args.threads, args.workers) <= 0:
        parser.error("context length, horizon, epochs, batch size, threads and workers must be positive")
    if min(args.periods) <= 0 or len(set(args.periods)) != len(args.periods):
        parser.error("periods must be distinct positive integers")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("device must be cpu or cuda[:index]")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    with args.profiles.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)["strategies"]["onenet"]
    config = {"data": {"context_length": args.context_length, "horizon": args.horizon},
              "forecasting": profile["forecasting"], "method": profile["method"],
              "online": {"device": str(device)}}
    if config["method"]["name"] != "onenet":
        raise ValueError("profiles.strategies.onenet must declare method.name: onenet")
    if config["method"].get("n_inner", 1) != 1:
        raise ValueError("batch experiments require OneNet n_inner=1")
    setup_began = perf_counter()
    dataset = load_benchmark_dataset("ettm1", args.data, args.context_length, args.horizon,
                                     include_time_features=True)
    split, train_windows, bounds = stream_partition(dataset, args.windows if mode == "branches" else 1)
    training = dataset.values[:split]
    training_std = training.std(dim=0, unbiased=False)
    scaler = {"feature_names": list(dataset.feature_names), "mean": training.mean(dim=0).tolist(),
              "std": torch.where(training_std < 1e-6, torch.ones_like(training_std), training_std).tolist()}
    dataset.standardize(train_windows)
    actions = build_batch_actions(args.periods, bounds[-1] - bounds[0])
    model = build_backbone(config, dataset.num_features, dataset.num_targets, dataset.target_indices)
    method = build_method(config, model)
    synchronize(device)
    setup_seconds = perf_counter() - setup_began
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / "manifest.json"
    manifest = {"status": "training", "mode": mode, "dataset": "ETTm1",
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "config": config, "raw_rows": len(dataset.values), "offline_end_row_exclusive": split,
                "train_windows": train_windows, "window_bounds": bounds,
                "update_rule": "batch_all_feedback",
                "long_step_compensation": "multiply by actual batch size; sum expert loss gaps",
                "tail_policy": "update incomplete batch before final prediction",
                "unit_bits": [unit.name for unit in OneNetAdapter.capabilities.units],
                "actions_per_window": len(actions), "planned_runs": len(actions) * (len(bounds) - 1),
                "setup_seconds": setup_seconds, "completed_runs": 0,
                "scaler": scaler, "target_names": list(dataset.target_names),
                "forecast_records": {"csv_costs": "per clock and cumulative within segment",
                                     "npz_shape": "[forecast_origin, horizon_step, target_variable]",
                                     "value_space": "standardized; original = standardized * std + mean",
                                     "row_indexing": "zero-based data rows excluding CSV header",
                                     "horizon_indexing": "zero-based; target row = target_start_row + horizon_step"},
                "scoring": "full horizons of forecasts issued inside each window; standardized targets",
                "timing_note": ("parallel workers share the device; measured times include resource contention"
                                if args.workers > 1 else "configurations execute serially"),
                "branch_timing": "before boundary feedback; optimizer and pending forecasts inherited; new action phase"}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    try:
        training_began = perf_counter()
        loader = DataLoader(Subset(dataset, range(train_windows)), batch_size=args.batch_size, shuffle=True)
        for epoch in range(args.epochs):
            total_loss = 0.0
            for inputs, target in loader:
                loss = method.train_batch(inputs, target)
                if not math.isfinite(loss):
                    raise FloatingPointError("nonfinite offline training loss")
                total_loss += loss * len(inputs)
            method.end_offline_epoch(epoch + 1)
            print(f"offline epoch={epoch + 1}/{args.epochs} loss={total_loss / train_windows:.6g}", flush=True)
        method.reset_online_state()
        synchronize(device)
        manifest["offline_training_seconds"] = perf_counter() - training_began
        native_action = OneNetAdapter.capabilities.native_action(duration=bounds[-1] - bounds[0])
        native = OnlineExecutor(UpdateController(method, native_action), feedback_delay=args.horizon)
        with (args.output / "actions.json").open("w", encoding="utf-8") as handle:
            json.dump([{**item, "action": item["action"].to_dict()} for item in actions], handle, indent=2)
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        with (TemporaryDirectory(prefix=".workers-", dir=args.output) if args.workers > 1
              else nullcontext(None)) as worker_directory, \
                (ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"),
                                     initializer=initialize_worker, initargs=(dataset, args.threads, device))
                 if args.workers > 1 else nullcontext(None)) as pool, \
                (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as summary_file, \
                (args.output / "native.csv").open("w", encoding="utf-8", newline="") as native_file:
            summary_writer = native_writer = None
            for window, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
                saved_rng = rng_state(device)
                scheduler_state = native.state_dict()
                # Historical scores are not required to resume causal forecasting.
                scheduler_state["events"] = []
                details = args.output / "forecasts" / f"window-{window:02d}"
                if args.save_forecasts:
                    details.mkdir(parents=True)
                print(f"window={window + 1}/{len(bounds) - 1} origins=[{start},{stop}) "
                      f"actions={len(actions)} workers={args.workers}", flush=True)
                if pool is None:
                    results = (run_batch_configuration(item, native.method, scheduler_state, saved_rng,
                                                 dataset, start, stop,
                                                 details / f"{item['action_id']}.csv" if args.save_forecasts else None)
                               for item in actions)
                else:
                    checkpoint_path = Path(worker_directory) / "source.pt"
                    torch.save({"controller": native.method, "scheduler": scheduler_state, "rng": saved_rng},
                               checkpoint_path)
                    tasks = [(item, checkpoint_path, start, stop,
                              details / f"{item['action_id']}.csv" if args.save_forecasts else None)
                             for item in actions]
                    results = pool.map(run_batch_worker, tasks)
                for item, result in zip(actions, results, strict=True):
                    row = {"window": window, "start": start, "stop": stop,
                           "target_start_row": start + args.context_length,
                           "last_target_end_row_exclusive": stop - 1 + args.context_length + args.horizon,
                           **{key: value for key, value in item.items() if key != "action"},
                           **result}
                    if summary_writer is None:
                        summary_writer = csv.DictWriter(summary_file, fieldnames=list(row))
                        summary_writer.writeheader()
                    summary_writer.writerow(row)
                    summary_file.flush()
                    manifest["completed_runs"] += 1
                    print(f"  {item['action_id']} sweep={item['sweep']} M={item['units']} scale={item['lr_scale']:g} "
                          f"p={item['period']} {result['status']} mae={result['mae']} "
                          f"grad={result['gradient_steps']}", flush=True)
                # Candidate branches never become the source of later windows.
                restore_rng(saved_rng, device)
                result = run_segment(native, dataset, start, stop,
                                     details / "native.csv" if args.save_forecasts else None)
                native_row = {"window": window, "start": start, "stop": stop, **result}
                if native_writer is None:
                    native_writer = csv.DictWriter(native_file, fieldnames=list(native_row))
                    native_writer.writeheader()
                native_writer.writerow(native_row)
                native_file.flush()
                if result["status"] != "complete":
                    raise FloatingPointError("native trajectory produced a nonfinite prediction")
                manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
