"""Persist and replay lookahead search sequences, independently of collection."""

import copy
import csv
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .beam_search import CoarseToFineBeamSearch
from .catalog import ActionCatalog
from .experiment import evaluate_policy
from .preparation import load_interval, save_json
from .profiling import SearchProfile
from .rollout import SequencePolicy, restore_rng


def evaluate_oracle(data_directory: Path, output: Path, *, split: str = "test",
                    start_row: int | None = None, end_row: int | None = None,
                    search: dict | None = None, profile: bool = False) -> dict:
    began = perf_counter()
    manifest, source, dataset, start, stop, state = load_interval(
        data_directory, split=split, start_row=start_row, end_row=end_row)
    load_seconds = perf_counter() - began
    torch.set_num_threads(manifest["config"].get("threads", 1))
    catalog = ActionCatalog.from_dict(manifest["catalog"], source.method.capabilities)
    options = dict(manifest["config"].get("search", {}))
    if search is not None:
        if "k" in search:
            options.pop("k_max", None)
            options.pop("k_min", None)
        elif "k_max" in search or "k_min" in search:
            options.pop("k", None)
        options.update(search)
    # tail_policy controls label collection, not beam pruning.
    options.pop("tail_policy", None)
    engine = CoarseToFineBeamSearch(catalog, **options)
    options = {"k": [k for k, _ in engine.schedule],
               "h_tail": [tail for _, tail in engine.schedule],
               "beam_width": engine.beam_width, "workers": engine.workers}
    output.mkdir(parents=True, exist_ok=False)
    plan = {"status": "searching", "kind": "beam_oracle", "prepared_data": str(data_directory.resolve()),
            "prepared_sha256": hashlib.sha256((data_directory / "manifest.json").read_bytes()).hexdigest(),
            "catalog": catalog.to_dict(), "search": options, "split": split,
            "start": start, "stop": stop, "target_start_row": start + dataset.context_length,
            "target_end_row": stop + dataset.context_length + dataset.horizon - 1,
            "uses_future_labels_for_selection": True,
            "terminal": split != "history"}
    save_json(output / "manifest.json", plan)
    recorder = SearchProfile(output / "profile.jsonl") if profile else None
    try:
        if recorder is not None:
            recorder.record("metadata", search=options,
                            device=str(source.method.method.device),
                            start=start, stop=stop, action_count=len(catalog.ids),
                            load_interval_seconds=load_seconds,
                            timing="host wall clock; no added CUDA synchronization; worker times overlap")
            recorder.record("phase_start", phase="save_start_snapshot")
        began = perf_counter()
        torch.save({"executor": source, "rng": state}, output / "start.pt")
        if recorder is not None:
            recorder.record("phase_end", phase="save_start_snapshot", wall_seconds=perf_counter() - began)
            recorder.record("phase_start", phase="native_evaluation")
        began = perf_counter()
        native_executor = copy.deepcopy(source)
        native_executor.method.policy = SequencePolicy(catalog, start, (catalog.native_indices[0],) * (stop - start))
        restore_rng(state)
        native_summary = evaluate_policy(
            native_executor, dataset, start, stop, output / "native", name="native",
            target_mean=np.asarray(manifest["target_mean"]),
            target_std=np.asarray(manifest["target_std"]), terminal=plan["terminal"],
        )
        del native_executor
        with (output / "native/forecasts.csv").open(encoding="utf-8", newline="") as handle:
            native_errors = np.asarray([float(row["mae"]) for row in csv.DictReader(handle)], dtype=np.float64)
        if recorder is not None:
            recorder.record("phase_end", phase="native_evaluation", wall_seconds=perf_counter() - began)
            recorder.record("phase_start", phase="beam_search")
        began = perf_counter()
        result = engine.search(source, dataset, start, stop, state,
                               terminal=stop if plan["terminal"] else None,
                               native_errors=native_errors, profile=recorder)
        search_seconds = perf_counter() - began
        if recorder is not None:
            recorder.record("phase_end", phase="beam_search", wall_seconds=search_seconds,
                            candidate_evaluations=result.candidate_evaluations,
                            winning_round=result.winning_round, winning_k=result.winning_k,
                            winning_h_tail=result.winning_h_tail,
                            winning_reference_round=result.winning_reference_round)
            recorder.record("phase_start", phase="sequence_replay")
        began = perf_counter()
        with (output / "sequence.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["origin", "action_id"])
            writer.writerows((start + offset, catalog.ids[index]) for offset, index in enumerate(result.actions))
        winning_origins = tuple(range(start, stop, result.winning_k))
        replay_policy = SequencePolicy(catalog, start, result.actions, record_origins=frozenset(winning_origins))
        oracle_executor = copy.deepcopy(source)
        oracle_executor.method.policy = replay_policy
        restore_rng(state)
        oracle_summary = evaluate_policy(
            oracle_executor, dataset, start, stop, output / "oracle", name="oracle",
            target_mean=np.asarray(manifest["target_mean"]),
            target_std=np.asarray(manifest["target_std"]), terminal=plan["terminal"],
        )
        del oracle_executor
        with (output / "oracle/forecasts.csv").open(encoding="utf-8", newline="") as handle:
            actual = np.asarray([float(row["mae"]) for row in csv.DictReader(handle)], dtype=np.float64)
        if not np.allclose(actual, result.errors, rtol=1e-10, atol=1e-12):
            raise RuntimeError("persisted sequence differs from searched trajectory")
        replay_seconds = perf_counter() - began
        expansions_began = perf_counter()
        if len(result.winning_expansions) != len(winning_origins):
            raise RuntimeError("winning round expansion count differs from winning segment origins")
        feature_names = manifest["feature_names"]
        num_segments, num_actions = len(winning_origins), len(catalog.ids)
        origins_arr = np.asarray([seg.origin for seg in result.winning_expansions], dtype=np.int64)
        ends_arr = np.asarray([seg.end for seg in result.winning_expansions], dtype=np.int64)
        tail_ends_arr = np.asarray([seg.tail_end for seg in result.winning_expansions], dtype=np.int64)
        teacher_arr = np.asarray([seg.action_index for seg in result.winning_expansions], dtype=np.int64)
        states_arr = np.full((num_segments, len(feature_names)), np.nan, dtype=np.float64)
        mean_maes = np.full((num_segments, num_actions), np.nan, dtype=np.float64)
        act_maes = np.full((num_segments, num_actions), np.nan, dtype=np.float64)
        tail_maes = np.full((num_segments, num_actions), np.nan, dtype=np.float64)
        evaluated = np.zeros((num_segments, num_actions), dtype=bool)
        native_index = catalog.native_indices[0]
        for seg_idx, seg in enumerate(result.winning_expansions):
            recorded = replay_policy.recorded_features.get(seg.origin, {})
            for feat_idx, name in enumerate(feature_names):
                value = recorded.get(name)
                if value is not None:
                    states_arr[seg_idx, feat_idx] = float(value)
            for action_idx, (m_mae, a_mae, t_mae) in seg.scores.items():
                mean_maes[seg_idx, action_idx] = m_mae
                act_maes[seg_idx, action_idx] = a_mae
                tail_maes[seg_idx, action_idx] = t_mae
                evaluated[seg_idx, action_idx] = True
            if result.winning_round == 0 and not evaluated[seg_idx, native_index]:
                act_slice = actual[seg.origin - start:seg.end - start]
                tail_slice = actual[seg.end - start:seg.tail_end - start]
                act_sum = float(act_slice.sum())
                tail_sum = float(tail_slice.sum())
                mean_maes[seg_idx, native_index] = (act_sum + tail_sum) / (seg.tail_end - seg.origin)
                act_maes[seg_idx, native_index] = act_sum / (seg.end - seg.origin)
                tail_maes[seg_idx, native_index] = (tail_sum / len(tail_slice)) if len(tail_slice) else np.nan
                evaluated[seg_idx, native_index] = True
        expansions_path = output / "expansions.npz"
        np.savez(expansions_path, origins=origins_arr, ends=ends_arr, tail_ends=tail_ends_arr,
                 teacher_actions=teacher_arr, states=states_arr, mean_maes=mean_maes,
                 act_maes=act_maes, tail_maes=tail_maes, evaluated=evaluated,
                 reference_actions=np.asarray(result.winning_reference_actions, dtype=np.int64))
        expansions_save_seconds = perf_counter() - expansions_began
        if recorder is not None:
            recorder.record("phase_end", phase="sequence_replay", wall_seconds=perf_counter() - began,
                            replay_seconds=replay_seconds,
                            decision_seconds=oracle_summary["decision_seconds"],
                            update_seconds=oracle_summary["update_seconds"],
                            expansions_save_seconds=expansions_save_seconds,
                            recorded_states=num_segments,
                            expansions_evaluated=int(evaluated.sum()),
                            expansions_total=int(evaluated.size),
                            expansions_bytes=expansions_path.stat().st_size)
        summaries = [native_summary, oracle_summary]
        with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = [key for key, value in summaries[0].items() if not isinstance(value, list)]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summaries)
        plan.update(status="complete", rounds=result.rounds, winning_round=result.winning_round,
                    winning_k=result.winning_k, winning_h_tail=result.winning_h_tail,
                    winning_reference_round=result.winning_reference_round,
                    expansions_file="expansions.npz",
                    expansions_sha256=hashlib.sha256(expansions_path.read_bytes()).hexdigest(),
                    expansions_evaluated=int(evaluated.sum()),
                    expansions_total=int(evaluated.size),
                    candidate_evaluations=result.candidate_evaluations,
                    search_seconds=search_seconds, results=summaries,
                    sequence_sha256=hashlib.sha256((output / "sequence.csv").read_bytes()).hexdigest())
    except BaseException as error:
        plan.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", plan)
        if recorder is not None:
            recorder.record("run_end", status=plan["status"], error=plan.get("error"))
            recorder.close()
    return plan
