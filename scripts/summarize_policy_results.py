"""Summarize completed policy pipelines, including nested fsnet-now results.

Run on the machine storing the results:
python scripts/summarize_policy_results.py --runs runs/policy-RUN
Only saved JSON/CSV files are read; no models are loaded or executed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: RUNS/summary")
    args = parser.parse_args()
    if not args.runs.is_dir():
        parser.error(f"run directory does not exist: {args.runs}")
    pipelines = sorted(args.runs.rglob("pipeline.json"))
    if not pipelines:
        parser.error(f"no pipeline.json found under {args.runs}")

    jobs, frames = [], []
    for path in pipelines:
        # Active pipelines rewrite their manifest; an incomplete write is not a failed task.
        try:
            pipeline = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            jobs.append({"job": str(path.parent), "status": "manifest_being_written"})
            continue
        arguments = pipeline["arguments"]
        identity = {name: arguments[name] for name in ("dataset", "method", "horizon", "seed")}
        stages = pipeline["stages"]
        job = {**identity, "status": pipeline["status"],
               "last_stage": stages[-1]["name"] if stages else "",
               "last_stage_status": stages[-1]["status"] if stages else "",
               "pipeline_wall_seconds": pipeline.get("wall_seconds"),
               "error": pipeline.get("error", ""), "job": str(path.parent)}
        for stage in stages:
            job[f"{stage['name']}_seconds"] = stage.get("seconds")
        jobs.append(job)
        if pipeline["status"] != "complete":
            continue
        # Preserve all measured costs and gains from the official comparison.
        comparison = pd.read_csv(path.parent / "comparison" / "comparison.csv")
        for name, value in identity.items():
            comparison[name] = value
        comparison["job"] = str(path.parent)
        frames.append(comparison)

    tasks = pd.DataFrame(jobs)
    leading = ["dataset", "method", "horizon", "seed", "controller"]
    metrics = (pd.concat(frames, ignore_index=True) if frames else
               pd.DataFrame(columns=[*leading, "mae", "mse", "gradient_steps", "online_seconds"]))
    metrics = metrics[[*leading, *(name for name in metrics.columns if name not in leading)]]
    metrics = metrics.sort_values(leading, ignore_index=True)
    output = args.output or args.runs / "summary"
    output.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(output / "task_status.csv", index=False)
    metrics.to_csv(output / "policy_metrics.csv", index=False)

    status_columns = [name for name in ("dataset", "method", "status", "last_stage", "last_stage_status")
                      if name in tasks.columns]
    print("Task status (tasks not yet started have no pipeline.json and are not listed):")
    print(tasks[status_columns].to_string(index=False))
    print(f"\nCompleted pipelines: {len(frames)}/{len(pipelines)}; policy rows: {len(metrics)}")
    if not metrics.empty:
        columns = [name for name in (
            "dataset", "method", "controller", "mae", "mse",
            "mae_gain_vs_native_pct", "mse_gain_vs_native_pct",
            "online_seconds", "gradient_steps", "egd_steps",
            "candidate_search_seconds", "candidate_gradient_steps",
        ) if name in metrics.columns]
        print(metrics[columns].to_string(index=False, float_format="%.6g"))
    print(f"\nSaved: {output / 'policy_metrics.csv'}")
    print(f"Saved: {output / 'task_status.csv'}")


if __name__ == "__main__":
    main()
