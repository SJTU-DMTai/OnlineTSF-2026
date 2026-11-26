"""Collect full-window greedy teachers without training a decision model."""

import copy
import csv
import hashlib
import io
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .actions import positive_integer
from .beam_search import _executor_device
from .catalog import ActionCatalog
from .collection import _init_collection_worker, _score_collection_job, score_candidate
from .preparation import load_interval, save_json
from .profiling import SearchProfile
from .rollout import advance


def collect_fixed_oracle(data_directory: Path, output: Path, *, window: int = 32,
                         workers: int | None = None, profile: bool = False) -> None:
    """Evaluate each action for window origins, then execute the winner for window.

    Every candidate uses the same complete state and RNG. Only complete windows
    inside history become labels; no EMA, Native safeguard or teacher anchoring
    changes the greedy choice. Warm-up segments still advance the teacher.
    """
    positive_integer(window, "window")
    prepared, executor, dataset, start, stop, state = load_interval(data_directory, split="history")
    options = copy.deepcopy(prepared["config"])
    workers = options.get("search", {}).get("workers", 1) if workers is None else workers
    positive_integer(workers, "workers")
    torch.set_num_threads(options.get("threads", 1))
    catalog = ActionCatalog.from_dict(prepared["catalog"], executor.method.capabilities)
    first = prepared["online_start"] + dataset.horizon
    segment_origins = list(range(start, stop - window + 1, window))
    if not any(origin >= first for origin in segment_origins):
        raise ValueError("history has no complete fixed windows after feedback warm-up")
    options.setdefault("collection", {}).update(mode="fixed_window", h_act=window, h_tail=0,
                                                train_states=None, anchor_epsilon=None)
    manifest = {**prepared, "status": "collecting", "kind": "collection", "config": options,
                "collection_mode": "fixed_window", "h_act": window, "h_tail": 0, "tau": window,
                "start": start, "stop": stop, "executed_stop": segment_origins[-1] + window,
                "decision_stride": window, "workers": workers, "anchor_epsilon": None,
                "prepared_data": Path(os.path.relpath(data_directory.resolve(), output.resolve())).as_posix(),
                "prepared_sha256": hashlib.sha256((data_directory / "manifest.json").read_bytes()).hexdigest(),
                "uses_future_labels_for_selection": True,
                "hard_target": "teacher_actions; exact ties prefer Native",
                "soft_target": "softmax(-maes / temperature)",
                "target": "mean MAE of a fixed action over a complete window"}
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", manifest)
    recorder = SearchProfile(output / "profile.jsonl") if profile else None
    states, maes, teacher_actions, origins = [], [], [], []
    try:
        if recorder is not None:
            recorder.record("metadata", window=window, workers=workers, start=start, stop=stop,
                            action_count=len(catalog.ids), timing="host wall clock; worker times overlap")
        with (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                  initializer=_init_collection_worker,
                                  initargs=(dataset, catalog, catalog.actions[catalog.native_indices[0]],
                                            window, 0, _executor_device(executor)))
              if workers > 1 else nullcontext()) as pool, \
                (output / "states.csv").open("w", encoding="utf-8", newline="") as sf, \
                (output / "outcomes.csv").open("w", encoding="utf-8", newline="") as of, \
                (output / "sequence.csv").open("w", encoding="utf-8", newline="") as seq:
            sw = csv.DictWriter(sf, fieldnames=["origin", *prepared["feature_names"]])
            ow = csv.DictWriter(of, fieldnames=["origin", "action_id", "h_act", "h_tail", "label_end",
                                                "mean_mae", "raw_mean_mae", "teacher_action_id"])
            sequence = csv.writer(seq)
            sw.writeheader()
            ow.writeheader()
            sequence.writerow(["origin", "action_id"])
            for step, origin in enumerate(segment_origins, 1):
                began = perf_counter()
                if pool is None:
                    results = [score_candidate(executor, dataset, origin, action, catalog.actions[0], state,
                                               h_act=window, h_tail=0) for action in catalog.actions]
                else:
                    snapshot = io.BytesIO()
                    torch.save({"executor": executor, "rng": state}, snapshot)
                    payload = snapshot.getvalue()
                    results = list(pool.map(_score_collection_job,
                                            [(payload, origin, i) for i in range(len(catalog.ids))]))
                costs = np.asarray([result[1]["mean_mae"] for result in results], dtype=np.float64)
                chosen = int(catalog.best(-costs[None, :])[0])
                evaluation_seconds = perf_counter() - began
                actual, state = advance(executor, dataset, origin, origin + window, catalog.actions[chosen], state)
                if not np.isclose(actual.mean(), costs[chosen], rtol=1e-10, atol=1e-12):
                    raise RuntimeError("fixed-window teacher differs from scored candidate")
                sequence.writerows((index, catalog.ids[chosen]) for index in range(origin, origin + window))
                if origin >= first:
                    features = results[0][0]
                    row = [np.nan if features[name] is None else features[name] for name in prepared["feature_names"]]
                    states.append(row)
                    maes.append(costs)
                    teacher_actions.append(chosen)
                    origins.append(origin)
                    sw.writerow({"origin": origin, **{name: features[name] for name in prepared["feature_names"]}})
                    for action_id, cost in zip(catalog.ids, costs, strict=True):
                        ow.writerow({"origin": origin, "action_id": action_id, "h_act": window, "h_tail": 0,
                                     "label_end": origin + window + dataset.horizon - 1,
                                     "mean_mae": float(cost), "raw_mean_mae": float(cost),
                                     "teacher_action_id": catalog.ids[chosen]})
                if recorder is not None:
                    recorder.record("collect_origin", step=step, origin=origin,
                                    evaluation_seconds=evaluation_seconds, wall_seconds=perf_counter() - began,
                                    action_id=catalog.ids[chosen], retained=origin >= first)
                print(f"fixed window {step}/{len(segment_origins)} origin={origin} "
                      f"action={catalog.ids[chosen]} mae={costs[chosen]:.6f}", flush=True)
        values = np.asarray(maes, dtype=np.float64)
        origin_values = np.asarray(origins, dtype=np.int64)
        np.savez(output / "training.npz", states=np.asarray(states, dtype=np.float64),
                 maes=values, raw_maes=values, act_maes=values,
                 tail_maes=np.full_like(values, np.nan),
                 advantages=values - values[:, catalog.native_indices[:1]],
                 regrets=values - values.min(1, keepdims=True),
                 teacher_actions=np.asarray(teacher_actions, dtype=np.int64), origins=origin_values,
                 label_ends=origin_values + window + dataset.horizon - 1)
        manifest.update(status="complete", state_count=len(origins),
                        candidate_evaluations=len(segment_origins) * len(catalog.ids))
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", manifest)
        if recorder is not None:
            recorder.record("run_end", status=manifest["status"], error=manifest.get("error"))
            recorder.close()
