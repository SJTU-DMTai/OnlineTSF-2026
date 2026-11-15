"""One-off recovery of completed-policy outputs, using existing evaluation workers."""

import argparse
import csv
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from onlinetsf.decision.actions import UpdateAction
from onlinetsf.decision.experiment import (
    evaluate_policy_worker, initialize_evaluation_worker, save_json,
)
from onlinetsf.decision.selector import ActionSelector, SelectorPolicy, resolve_ema_beta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="FSNet experiment directory with data and models")
    parser.add_argument("--device", required=True, help="physical GPU, for example cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    root = args.root.resolve()
    data, models, output = root / "data", root / "models", root / "test_ema_default"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    trained = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
    previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or trained["status"] != "complete":
        raise ValueError("collection and model training must be complete")
    if manifest["catalog"]["method"] != "fsnet":
        raise ValueError("root must point to the FSNet experiment")
    beta = resolve_ema_beta(None, manifest["tau"])
    if previous["ema_beta"] != beta:
        raise ValueError("existing evaluation does not use the default EMA coefficient")
    digest = hashlib.sha256((data / "training.npz").read_bytes()).hexdigest()
    native = next(row for row in manifest["catalog"]["actions"] if row["id"] == "native")
    jobs = [("native", UpdateAction.from_dict({k: v for k, v in native.items() if k != "id"})),
            ("frozen", UpdateAction())]
    for row in trained["results"]:
        directory = models / row["formulation"] / row["estimator"]
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["collection_sha256"] != digest or metadata["catalog"] != manifest["catalog"]:
            raise ValueError("model does not match collection")
        selector = ActionSelector.load(directory / "selector.pkl")
        jobs.append((f"policy/{row['formulation']}/{row['estimator']}", SelectorPolicy(selector, ema_beta=beta)))
    summaries, pending_jobs = {}, []
    for name, policy in jobs:
        summary_path = output / name / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (summary["name"] != name or summary["test_start"] != manifest["test_start"]
                    or summary["test_stop"] != manifest["test_stop"]
                    or summary["forecasts"] != manifest["test_stop"] - manifest["test_start"]):
                raise ValueError(f"completed output does not match test interval: {name}")
            summaries[name] = summary
            print(f"Keeping completed {name}", flush=True)
        else:
            pending_jobs.append((name, policy))
    # Preserve interrupted files in a separate directory before restarting a policy.
    backup = root / ("interrupted_ema_default_" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    for name, _ in pending_jobs:
        directory = output / name
        if directory.exists():
            destination = backup / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(destination)
            print(f"Archived incomplete output: {destination}", flush=True)
    save_json(output / "manifest.json", {"status": "evaluating", "ema_beta": beta,
                                         "results": list(summaries.values())})
    if pending_jobs:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_evaluation_worker,
                                 initargs=(args.device, manifest["config"].get("threads", 1))) as pool:
            futures = [pool.submit(evaluate_policy_worker, data, manifest, output, name, policy)
                       for name, policy in pending_jobs]
            for future in as_completed(futures):
                summary = future.result()
                summaries[summary["name"]] = summary
                save_json(output / "manifest.json", {"status": "evaluating", "ema_beta": beta,
                                                     "results": list(summaries.values())})
                print(f"Tested {summary['name']}: MAE={summary['mae']:.6f}", flush=True)
    results = [summaries[name] for name, _ in jobs]
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[k for k, v in results[0].items() if not isinstance(v, list)],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    save_json(output / "manifest.json", {"status": "complete", "ema_beta": beta, "results": results})
    print("Default EMA evaluation complete.", flush=True)
