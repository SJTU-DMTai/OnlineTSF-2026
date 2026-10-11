"""Experiment 1: learn a configuration selector and score existing same-state forks.

No torch import or forecasting-model execution. Defaults: OneNet/ETTh2/H=24,
seeds 0,1 for training and seed 2 for testing; all 17 profiled actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from update_policy_common import (
    BASIC_ACTIONS, DATA_FEATURES, DEFAULT_ANNOTATIONS, DEFAULT_RUNS, EXPERT_FEATURES,
    FEEDBACK_FEATURES, decision_features, file_sha256, find_job, fit_selector,
    numeric_inputs, plot_forecasts, save_json, select_actions,
)


def split_states(states: pd.DataFrame, train_seeds: list[int], test_seed: int, split: str,
                 train_fraction: float, lookahead: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    train = states[states.seed.isin(train_seeds)].copy()
    test = states[states.seed == test_seed].copy()
    boundary = None
    if split == "seed-time":
        clocks = sorted(states.clock.unique())
        boundary = int(clocks[int(len(clocks) * train_fraction)])
        # Last training target is clock + lookahead + H - 1, inclusive.
        train = train[train.clock + lookahead + horizon - 1 < boundary]
        test = test[test.clock >= boundary]
    if train.probe_id.nunique() < 3 or test.empty:
        raise ValueError("split requires at least three training positions and one test position")
    return train, test, boundary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--dataset", default="ETTh2")
    parser.add_argument("--method", choices=["onenet", "fsnet"], default="onenet")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--test-seed", type=int, default=2)
    parser.add_argument("--split", choices=["seed", "seed-time"], default="seed")
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--actions", nargs="+", default=["all"], help="all, basic, or explicit profiled action names")
    parser.add_argument("--lookahead", type=int, default=128)
    parser.add_argument("--metric", choices=["mae", "mse"], default="mae")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument("--prefix-rows", type=int, help="needed only when source run manifest is absent")
    parser.add_argument("--output", type=Path, default=Path("runs/policy-exp1-onenet-etth2-h24"))
    args = parser.parse_args(argv)
    if args.test_seed in args.train_seeds or len(set(args.train_seeds)) != len(args.train_seeds):
        parser.error("training seeds must be unique and exclude test-seed")
    if not args.train_seeds or not 0 < args.train_fraction < 1 or min(args.horizon, args.lookahead, args.max_depth, args.min_samples_leaf) <= 0:
        parser.error("invalid split or non-positive size")

    jobs = {seed: find_job(args.runs, args.dataset, args.method, args.horizon, seed)
            for seed in [*args.train_seeds, args.test_seed]}
    catalogs = [json.loads((job / "policies.json").read_text(encoding="utf-8")) for job in jobs.values()]
    if any(catalog != catalogs[0] for catalog in catalogs[1:]):
        raise ValueError("source jobs have different action definitions")
    catalog = {item["name"]: item for item in catalogs[0]}
    actions = list(catalog) if args.actions == ["all"] else BASIC_ACTIONS[args.method] if args.actions == ["basic"] else args.actions
    if "native" not in actions or len(actions) != len(set(actions)) or set(actions) - catalog.keys():
        parser.error("actions must be unique, profiled, and include native")
    actions = ["native"] + [action for action in actions if action != "native"]

    annotation_dir = args.annotations / args.dataset
    annotation_manifest = json.loads((annotation_dir / "manifest.json").read_text(encoding="utf-8"))
    prefix = args.prefix_rows or max(annotation_manifest["calibration_rows"], (annotation_manifest["rows"] + 4) // 5)
    source_hash = file_sha256(annotation_dir / "labeled_data.csv")
    # The original run manifests bind the prefix and input hash to these results.
    for seed, job in jobs.items():
        manifest_path = job.parent.parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            item = next(item for item in manifest["jobs"] if item["dataset"] == args.dataset and
                        item["method"] == args.method and item["horizon"] == args.horizon and item["seed"] == seed)
            if item["input_sha256"] != source_hash:
                raise ValueError("annotations differ from the inputs used in the source experiment")
            if args.prefix_rows is not None and item["prefix_rows"] != args.prefix_rows:
                raise ValueError("prefix differs from source run")
            if seed == args.train_seeds[0]:
                prefix = item["prefix_rows"]
            if item["prefix_rows"] != prefix:
                raise ValueError("source jobs used different prefixes")
    values, phases, _ = numeric_inputs(annotation_dir, prefix)
    state_frames, outcome_frames = [], []
    for seed, job in jobs.items():
        states = pd.read_csv(job / "probe_states.csv")
        outcomes = pd.read_csv(job / "probe_outcomes.csv")
        if not (states.seed.eq(seed) & states.dataset.eq(args.dataset) & states.method.eq(args.method) & states.horizon.eq(args.horizon)).all():
            raise ValueError("job identity differs from state rows")
        enriched = [decision_features(row, values, int(row["clock"])) for row in states.to_dict("records")]
        state_frames.append(pd.DataFrame(enriched))
        outcome_frames.append(outcomes[outcomes.policy.isin(actions)])
    states = pd.concat(state_frames, ignore_index=True).sort_values(["seed", "clock"]).reset_index(drop=True)
    outcomes = pd.concat(outcome_frames, ignore_index=True)
    if states.duplicated(["seed", "probe_id"]).any() or outcomes.duplicated(["seed", "probe_id", "policy", "lookahead"]).any():
        raise ValueError("duplicate source states or outcomes")
    contexts = (states.clock - states["index"] + 1).unique()
    if len(contexts) != 1:
        raise ValueError("mixed context lengths")
    context_length = int(contexts[0])
    expected = set(states[states.seed == args.test_seed].probe_id)
    if any(set(states[states.seed == seed].probe_id) != expected for seed in args.train_seeds):
        raise ValueError("seed comparison requires identical sampled positions")
    wide = outcomes[outcomes.lookahead == args.lookahead].pivot(index=["seed", "probe_id"], columns="policy", values=args.metric)
    wide = wide.reindex(pd.MultiIndex.from_frame(states[["seed", "probe_id"]]))[actions]
    if not np.isfinite(wide.to_numpy()).all() or (wide.native <= 0).any():
        raise ValueError("every state requires finite action outcomes and positive native loss")
    gains = (1 - wide.div(wide.native, axis=0)) * 100
    gains.index = states.index
    train, test, boundary = split_states(states, args.train_seeds, args.test_seed, args.split,
                                        args.train_fraction, args.lookahead, args.horizon)
    best_fixed = wide.loc[pd.MultiIndex.from_frame(train[["seed", "probe_id"]])].mean().idxmin()
    selectors, rules = {}, []
    choices = {"native": ["native"] * len(test), "best_fixed": [best_fixed] * len(test)}
    score_rows = []
    for name, features in [("policy_full", DATA_FEATURES + FEEDBACK_FEATURES + EXPERT_FEATURES[args.method]),
                           ("policy_data", DATA_FEATURES)]:
        selector, text = fit_selector(train, gains.loc[train.index], actions, features, args.max_depth, args.min_samples_leaf)
        selectors[name] = selector
        choices[name], scores = select_actions(selector, test)
        rules.append(name + "\n" + text)
        for row_number, (_, state) in enumerate(test.iterrows()):
            for column, action in enumerate(actions):
                score_rows.append({"controller": name, "seed": args.test_seed, "probe_id": state.probe_id,
                                   "action": action, "predicted_gain_pct": scores[row_number, column]})

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    bundle = {"schema": 1, "dataset": args.dataset, "method": args.method, "horizon": args.horizon,
              "context_length": context_length, "prefix_rows": int(prefix), "input_sha256": source_hash,
              "runs": str(args.runs), "train_seeds": args.train_seeds, "test_seed": args.test_seed,
              "split": args.split, "test_clock_boundary": boundary, "lookahead": args.lookahead,
              "metric": args.metric, "best_fixed": best_fixed, "actions": [catalog[action] for action in actions],
              "selectors": selectors, "train_probe_ids": train.probe_id.unique().tolist(),
              "train_target_end_exclusive": int((train.clock + args.lookahead + args.horizon).max()),
              "test_clocks": sorted(int(x) for x in test.clock),
              "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    save_json(output / "selectors.json", bundle)
    (output / "tree_rules.txt").write_text("\n".join(rules), encoding="utf-8")
    membership = states[["seed", "probe_id", "clock"]].copy()
    membership["partition"] = "excluded"
    membership.loc[train.index, "partition"] = "train"
    membership.loc[test.index, "partition"] = "test"
    membership.to_csv(output / "split.csv", index=False)
    pd.DataFrame(score_rows).to_csv(output / "predicted_gains.csv", index=False)
    results = []
    test_job = jobs[args.test_seed]
    for position, (_, state) in enumerate(test.iterrows()):
        reference = outcomes[(outcomes.seed == args.test_seed) & (outcomes.probe_id == state.probe_id)]
        curves = {}
        for name, selected in choices.items():
            action = selected[position]
            for lookahead in sorted(reference.lookahead.unique()):
                row = reference[(reference.policy == action) & (reference.lookahead == lookahead)].iloc[0]
                native = reference[(reference.policy == "native") & (reference.lookahead == lookahead)].iloc[0]
                oracle = reference[reference.lookahead == lookahead][args.metric].min()
                results.append({"controller": name, "seed": args.test_seed, "probe_id": state.probe_id,
                                "clock": int(state.clock), "phase_at_decision": phases[int(state.clock)],
                                "selected_action": action, "lookahead": int(lookahead), "mae": row.mae, "mse": row.mse,
                                "native_mae": native.mae, "native_mse": native.mse,
                                "mae_gain_pct": 100 * (1 - row.mae / native.mae),
                                "mse_gain_pct": 100 * (1 - row.mse / native.mse),
                                "oracle_loss": oracle, "selected_loss": row[args.metric]})
            curves[name] = pd.read_csv(test_job / "probes" / state.probe_id / action / "forecasts.csv")
        plot_forecasts(curves, output / f"{state.probe_id}_errors", f"{args.dataset} {args.method} H={args.horizon} | {state.probe_id}")
    result = pd.DataFrame(results)
    result.to_csv(output / "test_outcomes.csv", index=False)
    summary = result.groupby(["controller", "lookahead"]).agg(
        mae=("mae", "mean"), mse=("mse", "mean"), native_mae=("native_mae", "mean"), native_mse=("native_mse", "mean"),
        mae_win_fraction=("mae_gain_pct", lambda x: (x > 0).mean()), mse_win_fraction=("mse_gain_pct", lambda x: (x > 0).mean()),
        oracle_loss=("oracle_loss", "mean"), selected_loss=("selected_loss", "mean"), probes=("probe_id", "nunique")).reset_index()
    for metric in ["mae", "mse"]:
        summary[f"{metric}_gain_pct"] = 100 * (1 - summary[metric] / summary[f"native_{metric}"])
    summary.to_csv(output / "summary.csv", index=False)
    result.groupby(["controller", "lookahead", "phase_at_decision"])[["mae", "mse", "mae_gain_pct", "mse_gain_pct"]].mean().to_csv(output / "phase_summary.csv")
    print(summary[summary.lookahead == args.lookahead].to_string(index=False), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
