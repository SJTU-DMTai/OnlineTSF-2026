"""One experiment entry point for preparation, collection, testing and comparison."""

from __future__ import annotations

import argparse
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter

import yaml

from .config import DecisionConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="To add diagnostics later: python -m onlinetsf.decision diagnose --help")
    parser.add_argument("--method", choices=["onenet", "fsnet", "dsof"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="parent of the labeled dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory")
    parser.add_argument("--decision-config", type=Path, default=Path("decision.yaml"))
    parser.add_argument("--profiles", type=Path, default=Path("profiles.yaml"))
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--positions", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    boundary = parser.add_mutually_exclusive_group()
    boundary.add_argument("--test-start", type=int)
    boundary.add_argument("--test-start-fraction", type=float, default=.6)
    parser.add_argument("--steps", type=int, help="limit test origins; default: all remaining origins")
    parser.add_argument("--oracle-diagnostics", choices=["none", "policy_full", "policy_data", "both"], default="none")
    parser.add_argument("--oracle-metric", choices=["mae", "mse"], default="mae")
    parser.add_argument("--controller", choices=["policy_full", "policy_data"],
                        help="which diagnostic to summarize; defaults to the enabled controller")
    args = parser.parse_args(argv)
    if min(args.horizon, args.context_length, args.positions, args.epochs, args.batch_size,
           args.workers, args.threads) <= 0 or args.workers > args.positions:
        parser.error("sizes must be positive and workers must not exceed positions")
    if not 0 < args.test_start_fraction < 1 or args.steps is not None and args.steps <= 0:
        parser.error("test-start-fraction must be between zero and one; steps must be positive")
    decision = DecisionConfig.load(args.decision_config)
    args.controller = args.controller or ("policy_data" if args.oracle_diagnostics == "policy_data" else "policy_full")
    args.output.mkdir(parents=True, exist_ok=False)
    # Freeze the effective configuration once, then let each stage inherit it.
    snapshot = args.output / "decision.yaml"
    snapshot.write_text(yaml.safe_dump(decision.to_dict(), sort_keys=False), encoding="utf-8")
    common = ["--annotations", str(args.annotations), "--device", args.device, "--threads", str(args.threads)]
    identity = ["--method", args.method, "--dataset", args.dataset, "--horizon", str(args.horizon)]
    boundary_args = (["--test-start", str(args.test_start)] if args.test_start is not None else
                     ["--test-start-fraction", str(args.test_start_fraction)])
    source, training, experiment, comparison = (args.output / name for name in ("source", "training", "experiment", "comparison"))
    manifest = {"status": "running", "decision_config": decision.to_dict(),
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "stages": []}
    manifest_path = args.output / "pipeline.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    began = perf_counter()
    try:
        # Call the package implementations directly, without launching scripts
        # or separate Python interpreters for the four stages.
        from ..update_study import collect, comparison as compare_stage, policy_experiment, run

        stages = [
            ("source", run.main, [*common, "--methods", args.method, "--datasets", args.dataset,
                                 "--horizons", str(args.horizon), "--seeds", str(args.seed),
                                 "--profiles", str(args.profiles), "--context-length", str(args.context_length),
                                 "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                                 "--decision-config", str(snapshot), "--scope", "prepare", "--output", str(source)]),
            ("collection", collect.main, [*common, *identity, *boundary_args, "--runs", str(source),
                                          "--seed", str(args.seed), "--positions", str(args.positions),
                                          "--workers", str(args.workers), "--output", str(training)]),
            ("experiment", policy_experiment.main, [*common, *identity, *boundary_args, "--runs", str(source),
                                                    "--train-seeds", str(args.seed), "--test-seed", str(args.seed),
                                                    "--training-data", str(training),
                                                    "--oracle-diagnostics", args.oracle_diagnostics,
                                                    "--oracle-metric", args.oracle_metric,
                                                    *(["--steps", str(args.steps)] if args.steps is not None else []),
                                                    "--output", str(experiment)]),
            ("comparison", compare_stage.main, [*common, "--runs", str(source), "--experiment", str(experiment),
                                                "--controller", args.controller, "--metric", args.oracle_metric,
                                                "--output", str(comparison)]),
        ]
        for number, (name, execute, arguments) in enumerate(stages, 1):
            print(f"[{args.dataset}/{args.method}] {number}/4: {name}", flush=True)
            record = {"name": name, "status": "running", "log": f"{number:02d}-{name}.log"}
            manifest["stages"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            stage_began = perf_counter()
            with (args.output / record["log"]).open("w", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    try:
                        execute(arguments)
                    except (Exception, SystemExit, KeyboardInterrupt):
                        traceback.print_exc()
                        raise
            record.update(status="complete", seconds=perf_counter() - stage_began)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["status"] = "complete"
        print(f"[{args.dataset}/{args.method}] completed: {comparison / 'comparison.csv'}", flush=True)
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        if manifest["stages"]:
            manifest["stages"][-1]["status"] = "failed"
        raise
    finally:
        manifest["wall_seconds"] = perf_counter() - began
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
