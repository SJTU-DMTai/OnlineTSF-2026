# -*- coding: utf-8 -*-
"""Measure local update effects and full-stream FSNet/OneNet update policies.

Example: python scripts/experiment3_adaptation.py --output runs/experiment3
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch

from onlinetsf.__main__ import _build_backbone, _build_method, _run_offline_training
from onlinetsf.config import load_config
from onlinetsf.data import load_benchmark_dataset
from onlinetsf.methods import OneNetMethod
from onlinetsf.online import MethodFeedback


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def phase_at(raw_index: int, labels: list[dict], recovery_rows: int) -> str:
    for label in labels:
        start = int(label["start_index"])
        end = int(label["end_index_exclusive"])
        if start <= raw_index < end:
            return "transition"
    for label in labels:
        end = int(label["end_index_exclusive"])
        if end <= raw_index < end + recovery_rows:
            return "recovery"
    return "stable"


def consume_feedback(method, feedback: MethodFeedback, update: bool) -> float | None:
    if update:
        return method.on_feedback(feedback)
    if isinstance(method, OneNetMethod):
        # A skipped optimization must still consume the emitted experts' queue entry.
        method._pending_predictions.popleft()
    return None


def replay(
    method, dataset, start: int, stop: int, policy: str, labels: list[dict],
    recovery_rows: int, local_effects: bool = False,
) -> tuple[list[dict], list[dict], float]:
    """One-step prequential replay; local probes branch before each feedback update."""

    steps: list[dict] = []
    effects: list[dict] = []
    pending_prediction = None
    began = perf_counter()
    for index in range(start, stop):
        context, target = dataset[index]
        prediction = pending_prediction
        if prediction is None:
            prediction = method.predict(context).detach().cpu()
        pending_prediction = None
        raw_index = index + dataset.context_length
        error = prediction - target
        mae = error.abs().mean().item()
        mse = error.square().mean().item()
        if policy == "always":
            update = True
        elif policy == "never":
            update = False
        elif policy == "oracle":
            update = phase_at(raw_index, labels, recovery_rows) != "stable"
        else:
            period = int(policy.removeprefix("every_"))
            update = (index - start + 1) % period == 0
        feedback = MethodFeedback(
            index=index, available_at=index, context=context, target=target,
            observed_mask=torch.ones_like(target, dtype=torch.bool), prediction=prediction,
        )
        skipped = copy.deepcopy(method) if local_effects and index + 1 < stop else None
        update_loss = consume_feedback(method, feedback, update)
        steps.append({
            "forecast_index": index, "target_raw_index": raw_index,
            "phase": phase_at(raw_index, labels, recovery_rows),
            "mae": mae, "mse": mse, "updated": update,
            "adaptation_loss": update_loss,
        })
        if skipped is not None:
            consume_feedback(skipped, feedback, False)
            next_context, next_target = dataset[index + 1]
            pending_prediction = method.predict(next_context).detach().cpu()
            skipped_prediction = skipped.predict(next_context).detach().cpu()
            updated_error = (pending_prediction - next_target).abs().mean().item()
            skipped_error = (skipped_prediction - next_target).abs().mean().item()
            effects.append({
                "feedback_raw_index": raw_index,
                "next_target_raw_index": raw_index + 1,
                "phase": phase_at(raw_index, labels, recovery_rows),
                "next_phase": phase_at(raw_index + 1, labels, recovery_rows),
                "updated_next_mae": updated_error,
                "skipped_next_mae": skipped_error,
                "update_minus_skip_mae": updated_error - skipped_error,
            })
    return steps, effects, perf_counter() - began


def event_metrics(steps: list[dict], labels: list[dict], pre_rows: int, recovery_rows: int) -> list[dict]:
    """Compare each event with the same policy's immediately preceding error."""

    events: list[dict] = []
    first_raw = steps[0]["target_raw_index"]
    last_raw = steps[-1]["target_raw_index"]
    for position, label in enumerate(labels):
        start = int(label["start_index"])
        end = int(label["end_index_exclusive"])
        previous_end = int(labels[position - 1]["end_index_exclusive"]) if position else first_raw
        next_start = int(labels[position + 1]["start_index"]) if position + 1 < len(labels) else last_raw + 1
        pre_start = max(first_raw, previous_end, start - pre_rows)
        post_end = min(next_start, last_raw + 1, end + recovery_rows)
        before = [row for row in steps if pre_start <= row["target_raw_index"] < start]
        after = [row for row in steps if start <= row["target_raw_index"] < post_end]
        recovery = [row for row in after if row["target_raw_index"] >= end]
        pre_mae = mean(row["mae"] for row in before) if before else None
        post_mae = mean(row["mae"] for row in after) if after else None
        early_mae = mean(row["mae"] for row in recovery[:8]) if len(recovery) >= 8 else None
        initially_degraded = early_mae > 1.2 * pre_mae if early_mae is not None and pre_mae is not None else None
        recovery_delay = None
        if initially_degraded:
            for offset in range(len(recovery) - 7):
                if mean(row["mae"] for row in recovery[offset:offset + 8]) <= 1.1 * pre_mae:
                    recovery_delay = recovery[offset]["target_raw_index"] - end
                    break
        events.append({
            "event_id": label["event_id"], "method": label["method"],
            "transition": label["transition"], "start_index": start,
            "end_index_exclusive": end, "pre_count": len(before), "post_count": len(after),
            "pre_mae": pre_mae, "post_mae": post_mae,
            "mae_increase": post_mae - pre_mae if pre_mae is not None and post_mae is not None else None,
            "early_recovery_mae": early_mae, "initially_degraded": initially_degraded,
            "recovered": recovery_delay is not None if initially_degraded else None,
            "recovery_censored": post_end < end + recovery_rows,
            "recovery_delay": recovery_delay,
            "updates_in_post_window": sum(row["updated"] for row in after),
        })
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, default=Path("data/generated"))
    parser.add_argument("--config-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+", default=["fsnet", "onenet"])
    parser.add_argument("--periods", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pre-rows", type=int, default=32)
    parser.add_argument("--recovery-rows", type=int, default=32)
    parser.add_argument("--no-local-effects", action="store_true", help="skip the costly one-step branch test")
    parser.add_argument("--max-streams", type=int, help="limit streams for a quick check")
    args = parser.parse_args()
    if any(period < 2 for period in args.periods) or min(args.pre_rows, args.recovery_rows) < 1:
        parser.error("periods must be >= 2; pre-rows and recovery-rows must be positive")
    if any(name not in {"fsnet", "onenet"} for name in args.strategies):
        parser.error("experiment 3 supports fsnet and onenet strategies")
    if args.max_streams is not None and args.max_streams < 1:
        parser.error("max-streams must be positive")
    manifest_paths = sorted(args.generated.rglob("manifest.json"))
    if args.max_streams is not None:
        manifest_paths = manifest_paths[:args.max_streams]
    if not manifest_paths:
        parser.error(f"no generated manifests found under {args.generated}")
    args.output.mkdir(parents=True, exist_ok=False)

    policy_steps: list[dict] = []
    policy_summaries: list[dict] = []
    event_rows: list[dict] = []
    effect_rows: list[dict] = []
    policies = ["always", "never", *(f"every_{period}" for period in args.periods), "oracle"]
    for manifest_path in manifest_paths:
        dataset_name = manifest_path.parent.parent.name.removesuffix("-batch")
        stream_id = f"{dataset_name}-batch/{manifest_path.parent.name}"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with (manifest_path.parent / manifest["files"]["labels"]).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            labels = sorted(csv.DictReader(handle), key=lambda row: int(row["start_index"]))
        for strategy in args.strategies:
            config = load_config(args.config_dir / f"config_{dataset_name}.yaml", strategy_name=strategy)
            config["data"]["path"] = str(manifest_path.parent / manifest["files"]["data"])
            config["online"]["device"] = args.device
            if config["data"]["horizon"] != 1 or config["data"].get("stride", 1) != 1:
                raise ValueError("experiment 3 requires horizon=1 and stride=1")
            if config["online"]["feedback_delay"] != 0:
                raise ValueError("experiment 3 requires feedback_delay=0")
            data = config["data"]
            dataset = load_benchmark_dataset(
                data["name"], data["path"], data["context_length"], data["horizon"], stride=1,
            )
            torch.manual_seed(config["seed"])
            model = _build_backbone(
                config, dataset.num_features, dataset.num_targets, dataset.target_indices,
            )
            initial_method = _build_method(config, model)
            offline_stop = _run_offline_training(dataset, initial_method, config["offline"])
            start = max(offline_stop, config["online"].get("start", 0))
            stop = config["online"].get("stop") or len(dataset)
            if start >= stop:
                raise ValueError(f"no online samples for {stream_id}")
            for policy in policies:
                torch.manual_seed(config["seed"])
                steps, _, seconds = replay(
                    copy.deepcopy(initial_method), dataset, start, stop, policy, labels,
                    args.recovery_rows,
                )
                for step in steps:
                    policy_steps.append({
                        "stream_id": stream_id, "dataset": dataset_name,
                        "strategy": strategy, "policy": policy, **step,
                    })
                stable_steps = [row for row in steps if row["phase"] == "stable"]
                policy_summaries.append({
                    "stream_id": stream_id, "dataset": dataset_name,
                    "strategy": strategy, "policy": policy,
                    "forecasts": len(steps), "mae": mean(row["mae"] for row in steps),
                    "mse": mean(row["mse"] for row in steps),
                    "stable_mae": mean(row["mae"] for row in stable_steps) if stable_steps else None,
                    "updates": sum(row["updated"] for row in steps),
                    "online_seconds": seconds,
                })
                for event in event_metrics(steps, labels, args.pre_rows, args.recovery_rows):
                    event_rows.append({
                        "stream_id": stream_id, "dataset": dataset_name,
                        "strategy": strategy, "policy": policy, **event,
                    })
            if not args.no_local_effects:
                torch.manual_seed(config["seed"])
                _, effects, _ = replay(
                    copy.deepcopy(initial_method), dataset, start, stop, "always", labels,
                    args.recovery_rows, local_effects=True,
                )
                for effect in effects:
                    effect_rows.append({
                        "stream_id": stream_id, "dataset": dataset_name,
                        "strategy": strategy, **effect,
                    })
        print(f"completed {stream_id}", flush=True)

    effect_groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in effect_rows:
        if row["phase"] != row["next_phase"]:
            continue
        effect_groups[(row["stream_id"], row["dataset"], row["strategy"], row["phase"])].append(
            row["update_minus_skip_mae"]
        )
    effect_summary = []
    for (stream_id, dataset_name, strategy, phase), values in sorted(effect_groups.items()):
        effect_summary.append({
            "stream_id": stream_id, "dataset": dataset_name, "strategy": strategy,
            "phase": phase, "steps": len(values), "mean_update_minus_skip_mae": mean(values),
            "median_update_minus_skip_mae": median(values),
            "fraction_update_harmful": sum(value > 0 for value in values) / len(values),
        })

    write_rows(args.output / "policy_steps.csv", [
        "stream_id", "dataset", "strategy", "policy", "forecast_index", "target_raw_index",
        "phase", "mae", "mse", "updated", "adaptation_loss",
    ], policy_steps)
    write_rows(args.output / "policy_summary.csv", [
        "stream_id", "dataset", "strategy", "policy", "forecasts", "mae", "mse",
        "stable_mae", "updates", "online_seconds",
    ], policy_summaries)
    write_rows(args.output / "event_recovery.csv", [
        "stream_id", "dataset", "strategy", "policy", "event_id", "method", "transition",
        "start_index", "end_index_exclusive", "pre_count", "post_count", "pre_mae", "post_mae",
        "mae_increase", "early_recovery_mae", "initially_degraded", "recovered",
        "recovery_censored", "recovery_delay", "updates_in_post_window",
    ], event_rows)
    write_rows(args.output / "local_update_effects.csv", [
        "stream_id", "dataset", "strategy", "feedback_raw_index", "next_target_raw_index",
        "phase", "next_phase", "updated_next_mae", "skipped_next_mae", "update_minus_skip_mae",
    ], effect_rows)
    write_rows(args.output / "local_effect_summary.csv", [
        "stream_id", "dataset", "strategy", "phase", "steps",
        "mean_update_minus_skip_mae", "median_update_minus_skip_mae", "fraction_update_harmful",
    ], effect_summary)
    (args.output / "settings.json").write_text(json.dumps({
        "generated": str(args.generated.resolve()), "strategies": args.strategies,
        "policies": policies, "pre_rows": args.pre_rows, "recovery_rows": args.recovery_rows,
        "local_effects": not args.no_local_effects, "device": args.device,
        "streams": len(manifest_paths),
        "note": "Never skips feedback optimization; FSNet prediction-time state remains active. Oracle uses drift labels.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
