"""OneNet action grids on a continuous stream or branches of its native trajectory."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onlinetsf.data import SlidingWindowDataset, load_benchmark_dataset
from onlinetsf.decision import UpdateAction, UpdateController
from onlinetsf.decision.adapters import OneNetAdapter
from onlinetsf.decision.capabilities import UnitSchedule
from onlinetsf.factory import build_backbone, build_method
from onlinetsf.online import OnlineExecutor


DEFAULT_SCALES = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0)
DEFAULT_PERIODS = (1, 2, 4, 8, 16)
COUNTERS = ("gradient_steps", "egd_steps", "gradient_update_clocks", "decision_count",
            "decision_seconds", "update_seconds")


def build_actions(scales, periods, duration: int) -> list[dict]:
    """One common multiplier and period for all applicable selected units.

    Preserve all requested grid points, including equivalent frozen/short-only
    settings. A state unit without a learning rate never receives a multiplier.
    """
    capabilities = OneNetAdapter.capabilities
    actions = []
    for mask, scale, period in itertools.product(range(1 << len(capabilities.units)), scales, periods):
        selected = [unit for bit, unit in enumerate(capabilities.units) if mask & (1 << bit)]
        action = UpdateAction(
            units={unit.name for unit in selected},
            learning_rate_scales={unit.name: scale for unit in selected if unit.has_learning_rate},
            periods={unit.name: period for unit in selected},
            duration=duration,
        )
        capabilities.validate(action)
        actions.append({"action_id": f"a{len(actions):04d}", "mask": mask,
                        "units": "+".join(unit.name for unit in selected) or "frozen",
                        "lr_scale": scale, "period": period, "action": action})
    return actions


def stream_partition(dataset: SlidingWindowDataset, windows: int) -> tuple[int, int, list[int]]:
    """Split raw rows at 20%; no offline target or scaler uses a test row."""
    split = int(len(dataset.values) * 0.2)
    train_windows = split - dataset.context_length - dataset.horizon + 1
    start = split - dataset.context_length
    if train_windows <= 0:
        raise ValueError("first 20% of rows must contain a context-plus-horizon training window")
    if windows <= 0 or len(dataset) - start < windows:
        raise ValueError("each online window must contain at least one complete forecast")
    # Partition forecast origins, not feedback arrival clocks. Horizon targets
    # may overlap the following window, but never extend past the dataset end.
    bounds = [start + (len(dataset) - start) * part // windows for part in range(windows + 1)]
    return split, train_windows, bounds


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rng_state(device: torch.device) -> tuple:
    return (torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None)


def restore_rng(state: tuple, device: torch.device) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def fork_executor(source: UpdateController, scheduler_state: dict, action: UpdateAction) -> OnlineExecutor:
    """Keep weights, optimizer moments and pending forecasts; start a new action phase.

    The snapshot is taken before the boundary clock's feedback and prediction.
    A new schedule makes every selected unit due on its first legal opportunity.
    """
    controller = copy.deepcopy(source)
    controller.policy = action
    controller.expires_at = None
    controller.schedule = UnitSchedule()
    executor = OnlineExecutor(controller, feedback_delay=scheduler_state["feedback_delay"])
    executor.load_state_dict(scheduler_state)
    return executor


def run_segment(executor: OnlineExecutor, dataset: SlidingWindowDataset, start: int, stop: int,
                forecasts_path: Path | None = None) -> dict:
    """Score only predictions issued here, while training only on arrived labels.

    Full-horizon targets are read after emission for evaluation only. We neither
    train past stop nor flush inherited feedback into this segment's scores.
    Counters measure work performed at clocks [start, stop), including updates
    on pending forecasts from before start. Copy/training/file I/O are excluded
    from online_seconds; elapsed_seconds includes evaluation and CSV writing.
    """
    controller = executor.method
    before = {name: getattr(controller, name) for name in COUNTERS}
    absolute_sum = squared_sum = online_seconds = 0.0
    values_scored = scored_forecasts = 0
    status, invalid_index = "complete", None
    synchronize(controller.method.device)
    began = perf_counter()
    context = (forecasts_path.open("w", encoding="utf-8", newline="")
               if forecasts_path is not None else nullcontext(None))
    with context as handle:
        writer = None
        if handle is not None:
            writer = csv.DictWriter(handle, fieldnames=(
                "index", "target_start_row", "mae", "mse", "gradient_steps", "egd_steps",
                "update_seconds", "online_seconds"))
            writer.writeheader()
        for index in range(start, stop):
            inputs, target = dataset[index]
            step_began = perf_counter()
            step = executor.step(index, inputs, target)
            online_seconds += perf_counter() - step_began
            prediction = step.emission.prediction
            if not torch.isfinite(prediction).all():
                status, invalid_index = "nonfinite", index
                break
            error = prediction.double() - target.double()
            absolute_sum += error.abs().sum().item()
            squared_sum += error.square().sum().item()
            values_scored += error.numel()
            scored_forecasts += 1
            if writer is not None:
                writer.writerow({"index": index, "target_start_row": index + dataset.context_length,
                                 "mae": error.abs().mean().item(), "mse": error.square().mean().item(),
                                 "gradient_steps": controller.gradient_steps - before["gradient_steps"],
                                 "egd_steps": controller.egd_steps - before["egd_steps"],
                                 "update_seconds": controller.update_seconds - before["update_seconds"],
                                 "online_seconds": online_seconds})
    synchronize(controller.method.device)
    result = {name: getattr(controller, name) - before[name] for name in COUNTERS}
    result.update(status=status, invalid_index=invalid_index, scored_forecasts=scored_forecasts,
                  values_scored=values_scored,
                  mae=absolute_sum / values_scored if status == "complete" else None,
                  mse=squared_sum / values_scored if status == "complete" else None,
                  online_seconds=online_seconds, elapsed_seconds=perf_counter() - began)
    return result


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("full", "branches"), help="continuous trajectories or native-state branches")
    parser.add_argument("--data", type=Path, default=root / "data/ETTm1.csv")
    parser.add_argument("--profiles", type=Path, default=root / "profiles.yaml")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--lr-scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--periods", type=int, nargs="+", default=list(DEFAULT_PERIODS))
    parser.add_argument("--save-forecasts", action="store_true", help="also write per-origin errors and cumulative costs")
    parser.add_argument("--windows", type=int, default=32, help="number of windows in branches mode (default: 32)")
    args = parser.parse_args(argv)
    mode = args.mode
    if min(args.context_length, args.horizon, args.epochs, args.batch_size, args.threads) <= 0:
        parser.error("context length, horizon, epochs, batch size and threads must be positive")
    if (not all(math.isfinite(scale) and scale > 0 for scale in args.lr_scales)
            or len(set(args.lr_scales)) != len(args.lr_scales) or 1.0 not in args.lr_scales):
        parser.error("learning rate scales must be distinct, finite, positive and include 1")
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
    setup_began = perf_counter()
    dataset = load_benchmark_dataset("ettm1", args.data, args.context_length, args.horizon,
                                     include_time_features=True)
    split, train_windows, bounds = stream_partition(dataset, args.windows if mode == "branches" else 1)
    dataset.standardize(train_windows)
    actions = build_actions(args.lr_scales, args.periods, bounds[-1] - bounds[0])
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
                "unit_bits": [unit.name for unit in OneNetAdapter.capabilities.units],
                "actions_per_window": len(actions), "planned_runs": len(actions) * (len(bounds) - 1),
                "setup_seconds": setup_seconds, "completed_runs": 0,
                "scoring": "full horizons of forecasts issued inside each window; standardized targets",
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
        with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as summary_file, \
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
                print(f"window={window + 1}/{len(bounds) - 1} origins=[{start},{stop}) actions={len(actions)}", flush=True)
                for item in actions:
                    restore_rng(saved_rng, device)
                    synchronize(device)
                    fork_began = perf_counter()
                    branch = fork_executor(native.method, scheduler_state, item["action"])
                    synchronize(device)
                    fork_seconds = perf_counter() - fork_began
                    result = run_segment(branch, dataset, start, stop,
                                         details / f"{item['action_id']}.csv" if args.save_forecasts else None)
                    row = {"window": window, "start": start, "stop": stop,
                           "target_start_row": start + args.context_length,
                           "last_target_end_row_exclusive": stop - 1 + args.context_length + args.horizon,
                           **{key: value for key, value in item.items() if key != "action"},
                           "fork_seconds": fork_seconds, **result}
                    if summary_writer is None:
                        summary_writer = csv.DictWriter(summary_file, fieldnames=list(row))
                        summary_writer.writeheader()
                    summary_writer.writerow(row)
                    summary_file.flush()
                    manifest["completed_runs"] += 1
                    print(f"  {item['action_id']} M={item['units']} scale={item['lr_scale']:g} "
                          f"p={item['period']} {result['status']} mae={result['mae']} "
                          f"grad={result['gradient_steps']}", flush=True)
                    del branch
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

