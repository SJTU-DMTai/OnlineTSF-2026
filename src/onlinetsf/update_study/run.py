"""Run independent full-stream controls and capped same-state diagnostic forks."""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from ..factory import build_backbone, build_method as build_online_method
from ..methods import DSOFMethod, FSNetMethod, OneNetMethod
from .data import load_stream, select_probes
from .reporting import Reporter, write_csv
from .runtime import ControlledMethod, Policy, Replay, policies, restore_rng, rng_state, state_features


def build_method(name: str, dataset, profile: dict, device: str):
    if profile["method"]["name"] != name:
        raise ValueError("profile method differs from requested method")
    forecasting = dict(profile["forecasting"])
    forecasting.setdefault("backbone", {"fsnet": "fsnet_tcn", "onenet": "onenet_tcn", "dsof": "dlinear"}[name])
    config = {"data": {"context_length": dataset.context_length, "horizon": dataset.horizon},
              "forecasting": forecasting, "method": profile["method"], "online": {"device": device}}
    model = build_backbone(config, dataset.num_features, dataset.num_targets, dataset.target_indices)
    return build_online_method(config, model)


def train_prefix(method, dataset, prefix_rows: int, epochs: int, batch_size: int) -> None:
    """Train only windows whose complete targets lie strictly inside the prefix."""
    windows = prefix_rows - dataset.context_length - dataset.horizon + 1
    loader = DataLoader(Subset(dataset, range(windows)), batch_size=batch_size, shuffle=True)
    method.model.train()
    for _ in range(epochs):
        for context, target in loader:
            if isinstance(method, (OneNetMethod, DSOFMethod)):
                method.train_batch(context, target)
            else:
                method.optimizer.zero_grad(set_to_none=True)
                loss = method.loss_fn(method.model(context.to(method.device)), target.to(method.device))
                loss.backward()
                method.model.record_gradients()
                method.optimizer.step()
    if isinstance(method, (OneNetMethod, DSOFMethod)):
        method.reset_online_state()


def policy_record(policy: Policy) -> dict:
    record = asdict(policy)
    record["groups"] = sorted(policy.groups) if policy.groups is not None else "all"
    return record


def run_branch(replay: Replay, stream, probe: dict, policy: Policy, directory: Path,
               steps: int, lookaheads: list[int], save_predictions: bool = False) -> list[dict]:
    began = perf_counter()
    branch = replay.fork(policy)
    start = probe["index"]
    reporter = Reporter(directory, stream, start, save_predictions)
    scored = []
    for index in range(start, start + steps):
        records = branch.step(stream.dataset, index)
        reporter.add(records)
        scored.extend(record for record in records if record.get("index", -1) >= start)
    tail = branch.finish(stream.dataset)
    reporter.add(tail)
    scored.extend(record for record in tail if record["index"] >= start)
    summary = reporter.close(branch.method, perf_counter() - began)
    write_csv(directory / "summary.csv", [summary])
    result = []
    for lookahead in lookaheads:
        selected = scored[:lookahead]
        if len(selected) != lookahead:
            raise RuntimeError("probe did not receive its complete evaluation horizon")
        result.append({"policy": policy.name, "lookahead": lookahead,
                       "mae": sum(record["mae"] for record in selected) / lookahead,
                       "mse": sum(record["mse"] for record in selected) / lookahead,
                       "branch_gradient_steps": summary["gradient_steps"], "branch_egd_steps": summary["egd_steps"],
                       "branch_seconds": summary["seconds"]})
    return result


def run_job(args, stream, name: str, seed: int, profile: dict, directory: Path, probes: list[dict]) -> None:
    dataset = stream.dataset
    start = stream.prefix_rows - dataset.context_length
    stop = len(dataset) if args.max_steps is None else min(len(dataset), start + args.max_steps)
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    method = build_method(name, dataset, profile, args.device)
    began = perf_counter()
    train_prefix(method, dataset, stream.prefix_rows, args.epochs, args.batch_size)
    if method.device.type == "cuda":
        torch.cuda.synchronize(method.device)
    offline_seconds = perf_counter() - began
    initial = Replay(ControlledMethod(method, Policy("native")), dataset.horizon)
    initial_rng = rng_state()
    torch.save({"method": method, "rng": initial_rng, "prefix_rows": stream.prefix_rows}, directory / "initial.pt")
    variants = policies(name, args.lr_scales, args.suite)
    (directory / "policies.json").write_text(json.dumps([policy_record(item) for item in variants], indent=2), encoding="utf-8")
    metadata = {"dataset": stream.manifest["dataset"], "method": name, "horizon": dataset.horizon, "seed": seed}
    summaries, outcomes, states = [], [], []
    # Native history is replayed once; every fork shares its exact pre-feedback state.
    replay = initial.fork(Policy("native"))
    restore_rng(initial_rng)
    reporter = Reporter(directory / "native", stream, start, args.save_predictions)
    by_index = {probe["index"]: probe for probe in probes}
    began = perf_counter()
    branch_seconds = 0.0
    for index in range(start, stop):
        if index in by_index and args.scope in {"branches", "both"}:
            probe = by_index[index]
            probe_id = f"{probe['kind']}_{probe['clock']}"
            features = state_features(stream, replay, index)
            states.append(dict(metadata, probe_id=probe_id, **probe, **features))
            saved_rng = rng_state()
            branch_began = perf_counter()
            for policy in variants:
                restore_rng(saved_rng)
                results = run_branch(replay, stream, probe, policy, directory / "probes" / probe_id / policy.name,
                                     args.branch_steps, args.lookaheads, args.save_predictions)
                outcomes.extend(dict(metadata, probe_id=probe_id, kind=probe["kind"], **result) for result in results)
            branch_seconds += perf_counter() - branch_began
            restore_rng(saved_rng)
            # Persist completed probes so an interrupted long replay remains inspectable.
            write_csv(directory / "probe_states.csv", states)
            write_csv(directory / "probe_outcomes.csv", outcomes)
            print(f"  {name} H={dataset.horizon} seed={seed}: probe {probe_id} complete", flush=True)
        reporter.add(replay.step(dataset, index))
    reporter.add(replay.finish(dataset))
    summary = reporter.close(replay.method, perf_counter() - began - branch_seconds)
    summaries.append(dict(metadata, policy="native", offline_seconds=offline_seconds, **summary))
    write_csv(directory / "summary.csv", summaries)
    if args.scope in {"full", "both"} and args.suite in {"when", "both"}:
        for policy in policies(name, args.lr_scales, "when")[1:]:
            restore_rng(initial_rng)
            replay = initial.fork(policy)
            reporter = Reporter(directory / policy.name, stream, start, args.save_predictions)
            began = perf_counter()
            for index in range(start, stop):
                reporter.add(replay.step(dataset, index))
            reporter.add(replay.finish(dataset))
            summary = reporter.close(replay.method, perf_counter() - began)
            summaries.append(dict(metadata, policy=policy.name, offline_seconds=offline_seconds, **summary))
            print(f"  {name} H={dataset.horizon} seed={seed}: {policy.name} complete", flush=True)
            write_csv(directory / "summary.csv", summaries)
    write_csv(directory / "summary.csv", summaries)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("runs/real-drift-electricity-5pct-20261010"))
    parser.add_argument("--datasets", nargs="+", default=["ETTh2", "ETTm1", "Electricity"])
    parser.add_argument("--methods", nargs="+", choices=["fsnet", "onenet", "dsof"], default=["fsnet", "onenet", "dsof"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 24])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--profiles", type=Path, default=Path("profiles.yaml"))
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--prefix-rows", type=int, help="default/minimum: max(calibration rows, ceil(20%% of raw rows))")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-scales", nargs="+", type=float, default=[0.3, 1.0, 3.0])
    parser.add_argument("--scope", choices=["branches", "full", "both"], default="both")
    parser.add_argument("--suite", choices=["when", "what", "both"], default="both")
    parser.add_argument("--max-probes", type=int, default=12)
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--probe-separation", type=int, default=64)
    parser.add_argument("--probe-offset", type=int, default=16)
    parser.add_argument("--probe-seed", type=int, default=2026)
    parser.add_argument("--branch-steps", type=int, default=128)
    parser.add_argument("--lookaheads", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--max-steps", type=int, help="limit prediction origins for smoke testing")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--save-predictions", action="store_true", help="also persist exact forecast/target tensors in bounded chunks")
    args = parser.parse_args(argv)
    positive = [args.context_length, args.epochs, args.batch_size, args.probe_separation, args.branch_steps,
                args.threads, *args.horizons, *args.lookaheads]
    if any(value <= 0 for value in positive) or args.max_probes < 0 or args.max_events < 0 or args.probe_offset < 0:
        parser.error("lengths/counts must be positive; probe counts and offsets may be zero")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("max-steps must be positive")
    if any(not math.isfinite(scale) or scale <= 0 for scale in args.lr_scales):
        parser.error("learning-rate scales must be finite and positive")
    if max(args.lookaheads) > args.branch_steps:
        parser.error("lookaheads must not exceed branch-steps")
    if args.scope == "full" and args.suite == "what":
        parser.error("module ablations require --scope branches or both")
    for values in (args.datasets, args.methods, args.horizons, args.seeds, args.lr_scales, args.lookaheads):
        if len(values) != len(set(values)):
            parser.error("duplicate values would repeat or overwrite experiment cells")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    profiles = yaml.safe_load(args.profiles.read_text(encoding="utf-8"))["strategies"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("runs") / f"update-study-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "python": platform.python_version(), "torch": torch.__version__, "profiles": profiles,
                "label_use": "posthoc_sampling_and_analysis_only", "status": "planning" if args.plan_only else "running", "jobs": []}
    manifest_path = output / "manifest.json"
    try:
        # Load one dataset/horizon at a time, including in plan mode (Electricity is large).
        for dataset_name in args.datasets:
            for horizon in args.horizons:
                stream = load_stream(args.annotations / dataset_name, args.context_length, horizon, args.prefix_rows)
                start = stream.prefix_rows - args.context_length
                stop = len(stream.dataset) if args.max_steps is None else min(len(stream.dataset), start + args.max_steps)
                probes = select_probes(stream, start, stop, max_probes=args.max_probes, max_events=args.max_events,
                                       separation=args.probe_separation, branch_steps=args.branch_steps,
                                       offset=args.probe_offset, seed=args.probe_seed)
                if args.scope == "full":
                    probes = []
                for name in args.methods:
                    for seed in args.seeds:
                        full_count = len(policies(name, args.lr_scales, "when")) if args.scope in {"full", "both"} and args.suite in {"when", "both"} else 1
                        branch_count = len(policies(name, args.lr_scales, args.suite))
                        job = {"dataset": dataset_name, "method": name, "horizon": horizon, "seed": seed,
                               "online_origins": stop - start, "prefix_rows": stream.prefix_rows,
                               "prefix_fraction": stream.prefix_rows / len(stream.phases),
                               "calibration_rows": stream.manifest["calibration_rows"], "probes": probes,
                               "source": str(stream.source.resolve()), "feature_columns": stream.dataset.feature_names,
                               "input_sha256": stream.manifest["labeled_data_sha256"],
                               "branch_policy_count": branch_count, "full_policy_count": full_count,
                               "expected_prediction_origins": (stop - start) * full_count + len(probes) * branch_count * args.branch_steps,
                               "status": "planned"}
                        manifest["jobs"].append(job)
                        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                        print(f"{dataset_name} {name} H={horizon} seed={seed}: {stop-start} origins, {len(probes)} probes, {job['expected_prediction_origins']} total forecasts", flush=True)
                        if not args.plan_only:
                            job["status"] = "running"
                            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                            job_began = perf_counter()
                            profile = profiles["dlinear_dsof" if name == "dsof" else name]
                            run_job(args, stream, name, seed, profile, output / dataset_name / f"{name}_h{horizon}_seed{seed}", probes)
                            job["seconds"] = perf_counter() - job_began
                            job["status"] = "complete"
        if not args.plan_only:
            from .analyze import analyze
            analysis_began = perf_counter()
            analyze(output)
            manifest["analysis_seconds"] = perf_counter() - analysis_began
        manifest["status"] = "planned" if args.plan_only else "complete"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
