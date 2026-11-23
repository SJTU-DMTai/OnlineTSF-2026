"""Evaluate a fixed-stride greedy Oracle from prepared forecasting snapshots."""

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from onlinetsf.decision.catalog import ActionCatalog
from onlinetsf.decision.experiment import evaluate_policy
from onlinetsf.decision.oracle_greedy import GreedyOracleExecutor
from onlinetsf.decision.preparation import save_json
from onlinetsf.decision.rollout import restore_rng


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", "--prepared", dest="run_root", type=Path, required=True,
                        help="prepared snapshot directory, or experiment root containing it; beam results are not required")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1, help="origins per greedy segment; score each candidate over the whole segment")
    parser.add_argument("--check-only", action="store_true", help="check alignment without loading snapshots or running forecasts")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0",
                        help="use CUDA_VISIBLE_DEVICES=2 with cuda:0 to select physical GPU 2")
    parser.add_argument("--max-origins", type=int, help="short smoke run only; omit for the aligned full interval")
    args = parser.parse_args()
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"run-root directory does not exist: {run_root} (working directory: {Path.cwd()})")
    if (run_root / "online_start.pt").is_file():
        matches = [run_root / "manifest.json"]
    elif run_root.name == "oracle_online_beam4":
        matches = [run_root.parent / "prepared/manifest.json"]
    else:
        matches = sorted(run_root.glob("**/prepared/manifest.json"))
    if len(matches) != 1:
        candidates = "\n".join(str(path.parent) for path in matches) or "(none)"
        raise ValueError(f"Found {len(matches)} prepared snapshot directories under {run_root}.\n"
                         f"Matching directories:\n{candidates}\n"
                         "Locate online_start.pt and pass its directory via --prepared.")
    prepared_dir = matches[0].parent
    prepared_path = prepared_dir / "manifest.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared["status"] != "complete" or prepared.get("kind") != "prepared":
        raise ValueError("forecasting preparation must be complete")
    if not (prepared_dir / "online_start.pt").is_file():
        raise FileNotFoundError(prepared_dir / "online_start.pt")
    start, stop = prepared["online_start"], prepared["test_stop"]
    if args.max_origins is not None:
        if args.max_origins <= 0:
            raise ValueError("max-origins must be positive")
        stop = min(stop, start + args.max_origins)
    full_interval = stop == prepared["test_stop"]
    plan = {"status": "checked", "kind": "aligned_greedy_oracle", "stride": args.stride,
            "prepared_data": str(prepared_dir),
            "prepared_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(), "catalog": prepared["catalog"],
            "split": "online", "start": start, "stop": stop,
            "target_start_row": prepared["offline_end"],
            "target_end_row": stop + prepared["config"]["data"]["context_length"]
                              + prepared["config"]["data"]["horizon"] - 1,
            "full_reference_interval": full_interval,
            "terminal": full_interval,
            "selection": "minimum segment mean MAE; hold configuration for the whole segment; no EMA or tail lookahead",
            "uses_future_labels_for_selection": True, "device": args.device}
    print(json.dumps({**plan, "catalog": {"action_count": len(plan["catalog"]["actions"])}},
                     ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return

    torch.set_num_threads(prepared["config"].get("threads", 1))
    if args.device == "cuda:0":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run in the GPU environment with CUDA_VISIBLE_DEVICES=2")
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # Preserve CPU-resident queues and RNG tensors when loading CUDA snapshots.
    load_options = {"weights_only": False}
    if args.device == "cpu":
        load_options["map_location"] = "cpu"
    saved = torch.load(prepared_dir / "online_start.pt", **load_options)
    source, state, dataset = saved["executor"], saved["rng"], saved["dataset"]
    if args.device == "cpu":
        source.method.method.device = torch.device("cpu")
        state = (state[0], [])
    elif source.method.method.device != torch.device("cuda:0"):
        raise ValueError("expected the prepared snapshot on logical cuda:0")
    del saved
    if source._last_index not in (None, start - 1):
        raise ValueError("prepared snapshot does not precede the aligned interval")
    catalog = ActionCatalog.from_dict(prepared["catalog"], source.method.capabilities)
    args.output.mkdir(parents=True, exist_ok=False)
    plan["status"] = "evaluating"
    save_json(args.output / "manifest.json", plan)
    summaries = []
    try:
        print("Running Native from the prepared snapshot...", flush=True)
        native = copy.deepcopy(source)
        native.method.policy = catalog.actions[catalog.native_indices[0]]
        restore_rng(state)
        summary = evaluate_policy(native, dataset, start, stop, args.output / "native", name="native",
                                  target_mean=np.asarray(prepared["target_mean"]),
                                  target_std=np.asarray(prepared["target_std"]), terminal=plan["terminal"])
        del native
        summaries.append(summary)
        save_json(args.output / "manifest.json", plan)
        print(f"Native MAE={summary['mae']:.9f}. Starting greedy Oracle ({stop - start} origins)...", flush=True)
        restore_rng(state)
        oracle = GreedyOracleExecutor(copy.deepcopy(source), dataset, catalog, stop,
                                      ema_beta=None, terminal=plan["terminal"], stride=args.stride)
        summary = evaluate_policy(oracle, dataset, start, stop, args.output / "oracle_greedy",
                                  name=f"oracle_greedy_stride{args.stride}", target_mean=np.asarray(prepared["target_mean"]),
                                  target_std=np.asarray(prepared["target_std"]), terminal=plan["terminal"])
        summary.update(candidate_search_seconds=oracle.candidate_search_seconds,
                       candidate_evaluations=oracle.candidate_evaluations, stride=args.stride)
        save_json(args.output / "oracle_greedy/summary.json", summary)
        summaries.append(summary)
        with (args.output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ["name", "forecasts", "test_start", "test_stop", "mae", "mse",
                      "original_mae", "original_mse", "gradient_steps", "update_seconds", "online_seconds"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        plan.update(status="complete", results=summaries)
        print(f"Complete: greedy MAE={summary['mae']:.9f}; results={args.output}", flush=True)
    except BaseException as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(args.output / "manifest.json", plan)


if __name__ == "__main__":
    main()
