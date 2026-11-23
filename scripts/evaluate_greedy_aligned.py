"""Evaluate the one-origin greedy Oracle from a saved beam experiment's snapshot."""

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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="check alignment without loading snapshots or running forecasts")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0",
                        help="use CUDA_VISIBLE_DEVICES=2 with cuda:0 to select physical GPU 2")
    parser.add_argument("--max-origins", type=int, help="short smoke run only; omit for the aligned full interval")
    args = parser.parse_args()
    matches = list(args.run_root.resolve().glob("**/dsof/oracle_online_beam4/manifest.json"))
    if len(matches) != 1:
        raise ValueError("run-root must contain exactly one dsof/oracle_online_beam4/manifest.json")
    reference_dir = matches[0].parent
    prepared_dir = reference_dir.parent / "prepared"
    reference = json.loads(matches[0].read_text(encoding="utf-8"))
    prepared_path = prepared_dir / "manifest.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if reference["status"] != "complete" or prepared["status"] != "complete":
        raise ValueError("beam search and preparation must both be complete")
    if reference["prepared_sha256"] != hashlib.sha256(prepared_path.read_bytes()).hexdigest():
        raise ValueError("prepared manifest differs from the beam experiment")
    if reference["catalog"] != prepared["catalog"]:
        raise ValueError("beam and prepared candidate catalogs differ")
    for filename in (reference_dir / "start.pt", prepared_dir / "online_start.pt"):
        if not filename.is_file():
            raise FileNotFoundError(filename)
    # Check both beam results before including them in the comparison table.
    beam_summaries = []
    for width in (1, 4):
        directory = reference_dir.parent / f"oracle_online_beam{width}"
        other = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for key in ("prepared_sha256", "catalog", "split", "start", "stop", "terminal",
                    "target_start_row", "target_end_row"):
            if other[key] != reference[key]:
                raise ValueError(f"beam={width} differs in {key}")
        if other["status"] != "complete":
            raise ValueError(f"beam={width} is incomplete")
        row = json.loads((directory / "oracle/summary.json").read_text(encoding="utf-8"))
        beam_summaries.append({**row, "name": f"beam{width}"})
    start, stop = reference["start"], reference["stop"]
    if args.max_origins is not None:
        if args.max_origins <= 0:
            raise ValueError("max-origins must be positive")
        stop = min(stop, start + args.max_origins)
    full_interval = stop == reference["stop"]
    plan = {"status": "checked", "kind": "aligned_greedy_oracle",
            "reference": str(reference_dir), "prepared_data": str(prepared_dir),
            "prepared_sha256": reference["prepared_sha256"], "catalog": reference["catalog"],
            "split": reference["split"], "start": start, "stop": stop,
            "target_start_row": reference["target_start_row"],
            "target_end_row": stop + prepared["config"]["data"]["context_length"]
                              + prepared["config"]["data"]["horizon"] - 1,
            "full_reference_interval": full_interval,
            "terminal": reference["terminal"] and full_interval,
            "selection": "minimum current-window MAE at every forecast origin; no EMA",
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
    saved = torch.load(reference_dir / "start.pt", **load_options)
    source, state = saved["executor"], saved["rng"]
    if args.device == "cpu":
        source.method.method.device = torch.device("cpu")
        state = (state[0], [])
    elif source.method.method.device != torch.device("cuda:0"):
        raise ValueError("expected the reference snapshot on logical cuda:0")
    dataset_snapshot = torch.load(prepared_dir / "online_start.pt", **load_options)
    dataset = dataset_snapshot["dataset"]
    del saved, dataset_snapshot
    if source._last_index not in (None, start - 1):
        raise ValueError("reference snapshot does not precede the aligned interval")
    catalog = ActionCatalog.from_dict(reference["catalog"], source.method.capabilities)
    args.output.mkdir(parents=True, exist_ok=False)
    plan["status"] = "evaluating"
    save_json(args.output / "manifest.json", plan)
    summaries = []
    try:
        print("Replaying Native to check the saved experiment...", flush=True)
        native = copy.deepcopy(source)
        native.method.policy = catalog.actions[catalog.native_indices[0]]
        restore_rng(state)
        summary = evaluate_policy(native, dataset, start, stop, args.output / "native", name="native",
                                  target_mean=np.asarray(prepared["target_mean"]),
                                  target_std=np.asarray(prepared["target_std"]), terminal=plan["terminal"])
        del native
        with (reference_dir / "native/forecasts.csv").open(encoding="utf-8", newline="") as handle:
            expected = list(csv.DictReader(handle))[:stop - start]
        with (args.output / "native/forecasts.csv").open(encoding="utf-8", newline="") as handle:
            actual = list(csv.DictReader(handle))
        if [row["forecast_index"] for row in actual] != [row["forecast_index"] for row in expected]:
            raise RuntimeError("Native forecast origins differ from the saved experiment")
        differences = {}
        for metric in ("mae", "mse"):
            left = np.asarray([float(row[metric]) for row in actual])
            right = np.asarray([float(row[metric]) for row in expected])
            differences[metric] = float(np.abs(left - right).max())
            if not np.allclose(left, right, rtol=1e-5, atol=1e-7):
                raise RuntimeError(f"Native {metric} differs from reference: max absolute difference={differences[metric]}; check code/CUDA environment")
        plan["native_max_absolute_difference"] = differences
        summaries.append(summary)
        save_json(args.output / "manifest.json", plan)
        print(f"Native aligned: {differences}. Starting greedy Oracle ({stop - start} origins)...", flush=True)
        restore_rng(state)
        oracle = GreedyOracleExecutor(copy.deepcopy(source), dataset, catalog, stop,
                                      ema_beta=None, terminal=plan["terminal"])
        summary = evaluate_policy(oracle, dataset, start, stop, args.output / "oracle_greedy",
                                  name="oracle_greedy", target_mean=np.asarray(prepared["target_mean"]),
                                  target_std=np.asarray(prepared["target_std"]), terminal=plan["terminal"])
        summary.update(candidate_search_seconds=oracle.candidate_search_seconds,
                       candidate_evaluations=oracle.candidate_evaluations)
        save_json(args.output / "oracle_greedy/summary.json", summary)
        summaries.append(summary)
        if full_interval:
            summaries.extend(beam_summaries)
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
