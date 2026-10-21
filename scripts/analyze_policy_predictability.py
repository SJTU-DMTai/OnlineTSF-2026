"""Analyze saved policy branches without importing torch or replaying forecasts.

python scripts/analyze_policy_predictability.py --runs runs/policy-RUN --output reports/predictability-RUN

Input may be a batch root, one dataset/method directory, or its training/ directory.
Only complete, hash-verified schema-5 collections are analyzed. Partial jobs are
listed as pending. All fitting is per action and uses purged forward time folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import sklearn

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def discover_training(runs: Path) -> list[Path]:
    if not runs.is_dir():
        raise ValueError(f"input directory does not exist: {runs}")
    if runs.name == "training" or (runs / "probe_states.csv").exists():
        return [runs]
    if (runs / "pipeline.json").exists() or (runs / "training").is_dir():
        return [runs / "training"]
    jobs = {path.parent / "training" for path in runs.glob("*/*/pipeline.json")}
    jobs.update(path.parent for path in runs.glob("*/*/training/manifest.json"))
    if not jobs:
        raise ValueError("no pipeline jobs found; use a batch root, job directory or training directory")
    return sorted(jobs)


def read_collection(directory: Path) -> tuple[dict, pd.DataFrame, np.ndarray, dict, dict]:
    """Validate the collection before computing targets; never deserialize models."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("collection is not complete")
    if manifest.get("feature_schema_version") != 5 or manifest.get("data_feature_precision") != "float64":
        raise ValueError("requires the current schema-5 float64 collection; do not mix feature semantics")
    for filename, key in [("probe_states.csv", "states_sha256"), ("probe_outcomes.csv", "outcomes_sha256")]:
        if file_sha256(directory / filename) != manifest[key]:
            raise ValueError(f"{filename}: SHA256 differs from the collection manifest")
    config = manifest["decision_config"]
    internal = config["method_features"] if config["method_features"] is not None else manifest["method_features"]
    data = list(config["data_features"])
    feedback = list(dict.fromkeys([*data, *config["feedback_features"]]))
    groups = {"data_tree": data, "data_feedback_tree": feedback,
              "full_tree": list(dict.fromkeys([*feedback, *internal]))}
    # Both forms occur in the user's schema-5 artifacts. Read the saved settings,
    # never the locally edited DecisionConfig defaults.
    depth = config["full_max_depth"] if "full_max_depth" in config else config["max_depth"]
    leaf = config["full_min_samples_leaf"] if "full_min_samples_leaf" in config else config["min_samples_leaf"]
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (depth, leaf)):
        raise ValueError("invalid saved tree parameters")
    states = pd.read_csv(directory / "probe_states.csv").sort_values("clock").reset_index(drop=True)
    outcomes = pd.read_csv(directory / "probe_outcomes.csv")
    actions = [record["name"] for record in manifest["actions"]]
    if len(actions) < 2 or actions[0] != "native" or len(set(actions)) != len(actions):
        raise ValueError("action catalog must be unique, with native first")
    for frame in (states, outcomes):
        for key in ("dataset", "method", "horizon", "seed"):
            if not frame[key].eq(manifest[key]).all():
                raise ValueError(f"row identity differs from manifest: {key}")
    if (states.probe_id.duplicated().any() or states.clock.duplicated().any()
            or len(states) != manifest["positions"] or set(states.clock) != set(manifest["clocks"])):
        raise ValueError("duplicate or incorrect collection positions")
    if not np.isfinite(states.clock).all() or not states.clock.eq(states.clock.astype(np.int64)).all():
        raise ValueError("clocks must be finite integers")
    if not (states.clock - states["index"] + 1).eq(manifest["context_length"]).all():
        raise ValueError("context length differs from state indices")
    if not outcomes.lookahead.eq(manifest["decision_interval"]).all():
        raise ValueError("branch evaluation lengths differ")
    if (outcomes.duplicated(["probe_id", "policy"]).any() or set(outcomes.policy) != set(actions)
            or set(outcomes.probe_id) != set(states.probe_id) or len(outcomes) != len(states) * len(actions)):
        raise ValueError("each state must contain every action exactly once")
    losses = outcomes.pivot(index="probe_id", columns="policy", values="mae").reindex(
        index=states.probe_id, columns=actions).to_numpy(dtype=float)
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("branch MAE must be complete, finite and nonnegative")
    span = manifest["decision_interval"] + manifest["horizon"]
    if (states.clock + span > manifest["test_start"]).any():
        raise ValueError("collection labels cross the original online-test boundary")
    missing = set(groups["full_tree"]) - set(states.columns)
    if missing:
        raise ValueError(f"configured features were not saved: {sorted(missing)}")
    for name in groups["full_tree"]:
        states[name] = pd.to_numeric(states[name], errors="raise")
    valid = losses[:, 0] > 0
    audit = {"positions_collected": len(states), "positions_used": int(valid.sum()),
             "excluded_zero_native_clocks": states.loc[~valid, "clock"].astype(int).tolist(),
             "native_mae_min": float(losses[:, 0].min()), "actions": actions,
             "feature_groups": groups, "tree_parameters": {"max_depth": depth, "min_samples_leaf": leaf},
             "schema": 5, "label_span": span,
             "source_hashes": {key: manifest[key] for key in ("states_sha256", "outcomes_sha256")}}
    states = states.loc[valid].reset_index(drop=True)
    states["state_pos"] = np.arange(len(states))
    return manifest, states, losses[valid], groups, audit


def forward_folds(clocks: np.ndarray, span: int) -> tuple[list, pd.DataFrame]:
    """Group by raw clock and remove labels not available before each held-out block."""
    clocks = np.asarray(clocks)
    if len(clocks) == 0 or np.any(np.diff(clocks) <= 0):
        raise ValueError("forward folds require unique increasing clocks")
    n = len(clocks)
    cuts = [math.floor(n * fraction) for fraction in (.5, .65, .8)] + [n]
    folds, records = [], []
    for number, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:]), 1):
        name = f"F{number}"
        if start == stop:
            continue
        eligible = (np.arange(n) < start) & (clocks + span <= clocks[start])
        train, test = np.flatnonzero(eligible), np.arange(start, stop)
        folds.append((name, train, test))
        for i in range(stop):
            records.append({"fold": name, "state_pos": i, "clock": int(clocks[i]),
                            "target_end_exclusive": int(clocks[i] + span),
                            "evaluation_start": int(clocks[start]),
                            "partition": "evaluation" if i >= start else "train" if eligible[i] else "purged"})
    return folds, pd.DataFrame(records)


def fit_action_trees(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                     parameters: dict) -> tuple[np.ndarray, list[int], list[float]]:
    """One tree per non-native action, with train-only imputation and missing flags."""
    train_x, test_x = np.asarray(train_x, dtype=float), np.asarray(test_x, dtype=float)
    medians = [float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.0
               for column in train_x.T]
    missing_train, missing_test = ~np.isfinite(train_x), ~np.isfinite(test_x)
    design_train = np.column_stack((np.where(missing_train, medians, train_x), missing_train.astype(float)))
    design_test = np.column_stack((np.where(missing_test, medians, test_x), missing_test.astype(float)))
    predictions = np.zeros((len(test_x), train_y.shape[1]))
    leaves = [0]
    for column in range(1, train_y.shape[1]):
        model = DecisionTreeRegressor(**parameters, random_state=0)
        model.fit(design_train, train_y[:, column])
        predictions[:, column] = model.predict(design_test)
        leaves.append(model.get_n_leaves())
    return predictions, leaves, medians


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def prediction_metrics(truth, predicted, baseline) -> dict:
    truth, predicted, baseline = np.asarray(truth), np.asarray(predicted), np.asarray(baseline)
    mae = float(np.abs(truth - predicted).mean())
    mean_mae = float(np.abs(truth - baseline).mean())
    variance = float(np.square(truth - truth.mean()).sum())
    return {"values": len(truth), "gain_mae_pp": mae, "mean_baseline_mae_pp": mean_mae,
            "mae_skill": 1 - mae / mean_mae if mean_mae > 0 else None,
            "r2": 1 - float(np.square(truth - predicted).sum()) / variance if variance > 0 else None,
            "spearman": spearman(truth, predicted)}


def selection_records(states: pd.DataFrame, losses: np.ndarray, choices: np.ndarray,
                      actions: list[str], fold: str, model: str) -> pd.DataFrame:
    selected = losses[np.arange(len(losses)), choices]
    optimum = losses.min(axis=1)
    regret = np.full(len(losses), np.nan)
    np.divide(selected - optimum, optimum, out=regret, where=optimum > 0)
    frame = states[["state_pos", "clock", "probe_id"]].copy()
    frame = frame.assign(fold=fold, model=model, selected_action=[actions[i] for i in choices],
                         selected_loss=selected, native_loss=losses[:, 0], oracle_loss=optimum,
                         absolute_regret=selected - optimum, relative_regret=regret,
                         correct=np.isclose(selected, optimum, rtol=1e-7, atol=1e-10),
                         within_1pct=selected <= optimum * 1.01 + 1e-12)
    return frame


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    # Balanced state/action rows make the all-action MAE an equal action average.
    expanded = pd.concat([predictions, predictions.assign(fold="all")], ignore_index=True)
    expanded = pd.concat([expanded, expanded.assign(action="all")], ignore_index=True)
    rows = []
    for (fold, model, action), group in expanded.groupby(["fold", "model", "action"], sort=False):
        metrics = prediction_metrics(group.true_gain, group.predicted_gain, group.mean_gain)
        # Cross-action offsets can inflate a pooled correlation/R2. Keep these
        # diagnostics per action; the primary aggregate is balanced action MAE.
        if action == "all":
            metrics.update(r2=None, spearman=None)
        rows.append({"fold": fold, "model": model, "action": action, **metrics})
    return pd.DataFrame(rows)


def feature_associations(states: pd.DataFrame, gains: np.ndarray, features: list[str],
                         actions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive associations only; held-out prediction tests establish utility."""
    associations, quality = [], []
    midpoint = len(states) // 2
    for feature in features:
        values = states[feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        quality.append({"feature": feature, "positions": len(values), "finite": len(finite),
                        "missing_fraction": 1 - len(finite) / len(values),
                        "unique_finite": len(np.unique(finite)),
                        "constant_or_all_missing": len(np.unique(finite)) < 2})
        for period, indices in [("early", np.arange(midpoint)), ("late", np.arange(midpoint, len(states))),
                                ("all", np.arange(len(states)))]:
            for column, action in enumerate(actions[1:], 1):
                x, y = values[indices], gains[indices, column]
                associations.append({"period": period, "feature": feature, "action": action,
                                     "valid_pairs": int((np.isfinite(x) & np.isfinite(y)).sum()),
                                     "spearman": spearman(x, y)})
    return pd.DataFrame(associations), pd.DataFrame(quality)


def summarize_selections(selections: pd.DataFrame) -> pd.DataFrame:
    expanded = pd.concat([selections, selections.assign(fold="all")], ignore_index=True)
    rows = []
    for (fold, model), group in expanded.groupby(["fold", "model"], sort=False):
        relative = group.relative_regret.dropna()
        rows.append({"fold": fold, "model": model, "states": len(group),
                     "mae": group.selected_loss.mean(),
                     "mae_gain_pct": 100 * (1 - group.selected_loss.mean() / group.native_loss.mean()),
                     "mean_absolute_regret": group.absolute_regret.mean(),
                     "mean_relative_regret": relative.mean() if len(relative) else None,
                     "median_relative_regret": relative.median() if len(relative) else None,
                     "relative_regret_states": len(relative), "zero_oracle_states": int(group.oracle_loss.eq(0).sum()),
                     "within_1pct": group.within_1pct.mean(), "acc": group.correct.mean()})
    return pd.DataFrame(rows)


def effective_block_size(clocks, requested: int, span: int) -> int:
    gaps = np.diff(np.asarray(clocks))
    return max(requested, math.ceil(span / int(gaps.min()))) if len(gaps) else requested


def permuted_blocks(n: int, block_size: int, rng) -> np.ndarray | None:
    blocks = [np.arange(start, min(n, start + block_size)) for start in range(0, n, block_size)]
    if len(blocks) < 2:
        return None
    order = rng.permutation(len(blocks))
    while np.array_equal(order, np.arange(len(blocks))):
        order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order])


def paired_intervals(predictions: pd.DataFrame, selections: pd.DataFrame, span: int,
                     block_sizes: list[int], repeats: int, seed: int) -> pd.DataFrame:
    """Resample paired positions within each forward fold, retaining action groups."""
    errors = predictions.assign(error=(predictions.true_gain - predictions.predicted_gain).abs())
    errors = errors.groupby(["fold", "state_pos", "clock", "model"], sort=False).error.mean().unstack("model")
    regrets = selections.pivot(index=["fold", "state_pos", "clock"], columns="model", values="relative_regret")
    comparisons = [("gain_mae", errors, a, b) for a, b in [
        ("mean", "full_tree"), ("clock_tree", "full_tree"),
        ("data_tree", "data_feedback_tree"), ("data_feedback_tree", "full_tree")]]
    comparisons += [("relative_regret", regrets, a, "full_tree") for a in ("native", "best_fixed")]
    rows = []
    for metric, table, reference, model in comparisons:
        difference = (table[reference] - table[model]).rename("improvement").reset_index().sort_values("clock")
        values = difference.improvement.to_numpy()
        for requested in block_sizes:
            rng = np.random.default_rng(seed)
            groups = list(difference.groupby("fold", sort=False))
            blocks = {name: min(len(group), effective_block_size(group.clock, requested, span)) for name, group in groups}
            # If a whole fold is fixed by the chosen block size, a pooled CI
            # would silently omit that fold's uncertainty. Report no CI instead.
            flexible = all(blocks[name] < len(group) for name, group in groups)
            distribution = []
            if repeats and flexible and np.isfinite(values).any():
                for _ in range(repeats):
                    sampled = []
                    for name, group in groups:
                        size, block = len(group), blocks[name]
                        starts = rng.integers(0, size - block + 1, size=math.ceil(size / block))
                        indices = np.concatenate([np.arange(start, start + block) for start in starts])[:size]
                        sampled.extend(group.improvement.to_numpy()[indices])
                    finite = np.asarray(sampled)[np.isfinite(sampled)]
                    if len(finite):
                        distribution.append(float(finite.mean()))
            rows.append({"metric": metric, "reference": reference, "model": model,
                         "requested_block": requested, "effective_blocks": json.dumps(blocks),
                         "blocks_per_fold": json.dumps({name: math.ceil(len(group) / blocks[name]) for name, group in groups}),
                         "positions": len(values), "valid_positions": int(np.isfinite(values).sum()),
                         "improvement": float(np.nanmean(values)) if np.isfinite(values).any() else None,
                         "ci_low": float(np.quantile(distribution, .025)) if distribution else None,
                         "ci_high": float(np.quantile(distribution, .975)) if distribution else None,
                         "bootstrap_samples": len(distribution),
                         "status": "estimated" if distribution else "disabled" if not repeats else "insufficient_blocks_or_values"})
    return pd.DataFrame(rows)


def mismatch_control(states: pd.DataFrame, gains: np.ndarray, losses: np.ndarray,
                     folds: list, features: list[str], parameters: dict, span: int,
                     repeats: int, requested_block: int, seed: int) -> pd.DataFrame:
    """Break training X/Y alignment in whole position blocks; keep test data fixed.

    Permute complete feature vectors, including missingness, jointly for every
    action. This is a diagnostic null, not an exact permutation significance test:
    time blocks are not guaranteed to be exchangeable under concept drift.
    """
    rows = []
    x = states[features].to_numpy(dtype=float)
    for number, (fold, train, test) in enumerate(folds):
        block = effective_block_size(states.iloc[train].clock, requested_block, span)
        rng = np.random.default_rng(seed + number)
        seen = set()
        observed, _, _ = fit_action_trees(x[train], gains[train], x[test], parameters)
        for repeat in range(repeats):
            order = permuted_blocks(len(train), block, rng)
            if order is None:
                rows.append({"fold": fold, "repeat": repeat, "status": "insufficient_blocks",
                             "train_positions": len(train), "block_size": block})
                break
            seen.add(tuple(order.tolist()))
            predicted, _, _ = fit_action_trees(x[train[order]], gains[train], x[test], parameters)
            choices = predicted.argmax(axis=1)
            actual_choices = observed.argmax(axis=1)
            optimum = losses[test].min(axis=1)
            valid = optimum > 0
            selected = losses[test, choices]
            actual_selected = losses[test, actual_choices]
            rows.append({"fold": fold, "repeat": repeat, "status": "estimated",
                         "train_positions": len(train), "block_size": block,
                         "distinct_orders_so_far": len(seen), "positions": len(test),
                         "valid_regret_positions": int(valid.sum()),
                         "null_gain_mae_pp": float(np.abs(predicted[:, 1:] - gains[test, 1:]).mean()),
                         "observed_gain_mae_pp": float(np.abs(observed[:, 1:] - gains[test, 1:]).mean()),
                         "null_relative_regret": float(((selected[valid] - optimum[valid]) / optimum[valid]).mean()) if valid.any() else None,
                         "observed_relative_regret": float(((actual_selected[valid] - optimum[valid]) / optimum[valid]).mean()) if valid.any() else None})
    return pd.DataFrame(rows, columns=["fold", "repeat", "status", "train_positions", "block_size",
                                      "distinct_orders_so_far", "positions", "valid_regret_positions",
                                      "null_gain_mae_pp", "observed_gain_mae_pp",
                                      "null_relative_regret", "observed_relative_regret"])


def summarize_mismatch(control: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = control.loc[control.status.eq("estimated")]
    if valid.empty:
        return pd.DataFrame(columns=["fold", "metric", "observed", "null_mean", "fraction_null_le_observed", "repeats"])
    pooled = []
    for repeat, group in valid.groupby("repeat", sort=False):
        row = {"fold": "all", "repeat": repeat}
        for metric, weight in [("gain_mae_pp", "positions"), ("relative_regret", "valid_regret_positions")]:
            mask = group[weight].gt(0)
            for source in ("null", "observed"):
                row[f"{source}_{metric}"] = np.average(group.loc[mask, f"{source}_{metric}"], weights=group.loc[mask, weight]) if mask.any() else np.nan
        pooled.append(row)
    combined = pd.concat([valid, pd.DataFrame(pooled)], ignore_index=True)
    for fold, group in combined.groupby("fold", sort=False):
        for metric in ("gain_mae_pp", "relative_regret"):
            paired = group[[f"null_{metric}", f"observed_{metric}"]].dropna()
            rows.append({"fold": fold, "metric": metric,
                         "observed": paired[f"observed_{metric}"].mean(),
                         "null_mean": paired[f"null_{metric}"].mean(),
                         "fraction_null_le_observed": (paired[f"null_{metric}"] <= paired[f"observed_{metric}"]).mean() if len(paired) else None,
                         "repeats": len(paired)})
    return pd.DataFrame(rows)


def make_figures(output: Path, metrics: pd.DataFrame, choices: pd.DataFrame,
                 associations: pd.DataFrame, features: list[str], actions: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = metrics.loc[metrics.fold.eq("all") & metrics.action.eq("all")]
    local = choices.loc[choices.fold.eq("all")]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    axes[0].bar(aggregate.model, aggregate.gain_mae_pp)
    axes[0].set_ylabel("Held-out gain MAE (percentage points)")
    axes[1].bar(local.model, pd.to_numeric(local.mean_relative_regret))
    axes[1].set_ylabel("Mean local relative regret")
    for axis in axes:
        axis.tick_params(axis="x", rotation=60)
    figure.savefig(output / "predictability.png", dpi=160)
    plt.close(figure)
    by_action = metrics.loc[metrics.fold.eq("all") & metrics.model.eq("full_tree") & metrics.action.ne("all")]
    figure, axis = plt.subplots(figsize=(12, 5), layout="constrained")
    axis.bar(by_action.action, pd.to_numeric(by_action.mae_skill))
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_ylabel("Full tree MAE skill versus historical mean")
    axis.tick_params(axis="x", rotation=70)
    figure.savefig(output / "action_skill.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(16, max(4, len(features) * .5)), layout="constrained")
    for period, axis in zip(("early", "late"), axes, strict=True):
        grid = associations.loc[associations.period.eq(period)].pivot(index="feature", columns="action", values="spearman")
        grid = grid.reindex(index=features, columns=actions[1:])
        graphic = axis.imshow(grid.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        axis.set_xticks(np.arange(len(actions) - 1), actions[1:], rotation=90)
        axis.set_yticks(np.arange(len(features)), features)
        axis.set_title(f"{period}: feature vs. actual gain Spearman")
    figure.colorbar(graphic, ax=axes, label="Spearman (blank = undefined)", shrink=.8)
    figure.savefig(output / "feature_correlations.png", dpi=160)
    plt.close(figure)


def analyze_collection(directory: Path, output: Path, args: argparse.Namespace) -> dict:
    manifest, states, losses, groups, audit = read_collection(directory)
    if len(states) < 4:
        raise ValueError("fewer than four positive-native positions; cannot construct forward evaluation")
    gains = 100 * (1 - losses / losses[:, :1])
    if not np.isfinite(gains).all():
        raise ValueError("relative gains overflow; inspect the native loss scale")
    actions, parameters, span = audit["actions"], audit["tree_parameters"], audit["label_span"]
    folds, split_audit = forward_folds(states.clock.to_numpy(), span)
    split_rows, usable = [], []
    for name, train, test in folds:
        enough = len(train) >= 2 * parameters["min_samples_leaf"]
        split_rows.append({"fold": name, "train_positions": len(train), "evaluation_positions": len(test),
                           "purged_positions": int(test[0] - len(train)),
                           "status": "evaluated" if enough else "insufficient_train_for_two_leaves"})
        if enough:
            usable.append((name, train, test))
    if not usable:
        raise ValueError("no fold has enough purged training positions for two saved-size leaves")
    output.mkdir(parents=True)
    audit.update(input=str(directory.resolve()), dataset=manifest["dataset"], method=manifest["method"],
                 horizon=manifest["horizon"], seed=manifest["seed"],
                 features=len(groups["full_tree"]), evaluated_folds=[fold[0] for fold in usable])
    save_json(output / "audit.json", audit)
    split_audit.to_csv(output / "split_audit.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "split_summary.csv", index=False)
    associations, quality = feature_associations(states, gains, groups["full_tree"], actions)
    associations.to_csv(output / "feature_associations.csv", index=False)
    quality.to_csv(output / "feature_audit.csv", index=False)
    predictions, selections, diagnostics = [], [], []
    for fold, train, test in usable:
        mean = np.tile(gains[train].mean(axis=0), (len(test), 1))
        model_predictions = {"mean": mean}
        feature_sets = {"clock_tree": ["clock"], **groups}
        for model, features in feature_sets.items():
            x = states[features].to_numpy(dtype=float)
            predicted, leaves, medians = fit_action_trees(x[train], gains[train], x[test], parameters)
            model_predictions[model] = predicted
            for action, leaf_count in zip(actions[1:], leaves[1:], strict=True):
                diagnostics.append({"fold": fold, "model": model, "action": action, "leaves": leaf_count,
                                    "features": json.dumps(features), "train_medians": json.dumps(medians)})
        for model, predicted in model_predictions.items():
            for column, action in enumerate(actions[1:], 1):
                frame = states.iloc[test][["state_pos", "clock", "probe_id"]].copy()
                frame = frame.assign(fold=fold, model=model, action=action, true_gain=gains[test, column],
                                     predicted_gain=predicted[:, column], mean_gain=mean[:, column])
                predictions.append(frame)
            selections.append(selection_records(states.iloc[test], losses[test], predicted.argmax(axis=1), actions, fold, model))
        fixed = int(losses[train].mean(axis=0).argmin())
        for model, choices in [("native", np.zeros(len(test), dtype=int)),
                               ("best_fixed", np.full(len(test), fixed)),
                               ("oracle", losses[test].argmin(axis=1))]:
            selections.append(selection_records(states.iloc[test], losses[test], choices, actions, fold, model))
    predictions, selections = pd.concat(predictions, ignore_index=True), pd.concat(selections, ignore_index=True)
    metrics, choices = summarize_predictions(predictions), summarize_selections(selections)
    intervals = paired_intervals(predictions, selections, span, args.block_sizes, args.bootstrap_repeats, args.seed)
    control = mismatch_control(states, gains, losses, usable, groups["full_tree"], parameters, span,
                               args.permutations, args.permutation_block, args.seed)
    control_summary = summarize_mismatch(control)
    for filename, frame in [("predictions", predictions), ("prediction_metrics", metrics), ("selections", selections),
                            ("selection_metrics", choices), ("tree_diagnostics", pd.DataFrame(diagnostics)),
                            ("paired_intervals", intervals), ("mismatch_control", control),
                            ("mismatch_summary", control_summary)]:
        frame.to_csv(output / f"{filename}.csv", index=False)
    if not args.no_plots:
        make_figures(output, metrics, choices, associations, groups["full_tree"], actions)
    full = metrics.loc[metrics.fold.eq("all") & metrics.action.eq("all") & metrics.model.eq("full_tree")].iloc[0]
    selected = choices.loc[choices.fold.eq("all")].set_index("model")
    summary = {key: manifest[key] for key in ("dataset", "method", "horizon", "seed")}
    summary.update(positions=audit["positions_used"], evaluation_positions=int(selected.loc["full_tree", "states"]),
                   features=audit["features"], max_depth=parameters["max_depth"], min_samples_leaf=parameters["min_samples_leaf"],
                   full_gain_mae_pp=full.gain_mae_pp, full_mae_skill=full.mae_skill,
                   full_relative_regret=selected.loc["full_tree", "mean_relative_regret"],
                   native_relative_regret=selected.loc["native", "mean_relative_regret"],
                   fixed_relative_regret=selected.loc["best_fixed", "mean_relative_regret"])
    # Text stays factual: poor/unstable results must not turn into a claim that
    # the underlying problem is unpredictable, or that correlation is causal.
    lines = [f"# {manifest['dataset']} / {manifest['method']}: offline predictability", "",
             f"Input: `{directory.resolve()}`", "",
             f"Verified schema 5; {audit['positions_used']} usable positions; {audit['features']} saved features. "
             f"Trees use saved max_depth={parameters['max_depth']}, min_samples_leaf={parameters['min_samples_leaf']} for every feature group.", "",
             "## How to read the evidence", "",
             "1. `prediction_metrics.csv`: positive held-out MAE skill supports gain predictability versus a per-action historical mean. Check Clock and individual folds/actions too.",
             "2. `feature_associations.csv`: early/late feature-gain Spearman is descriptive. `data_tree` → `data_feedback_tree` → `full_tree` tests incremental group information on identical held-out states.",
             "3. `selection_metrics.csv`: compare Full local regret with Native and training-selected Best Fixed; Oracle quantifies available action headroom.",
             "4. `paired_intervals.csv`: paired moving-block bootstrap within folds, block sizes 2/4/8 by default, expanded when needed to cover label span. Positive differences favor Full or the added feature group. Intervals are approximate with few time positions.",
             "5. `mismatch_summary.csv`: whole-block training X/Y mismatch control. A lower observed error/regret than the mismatch results supports the relevance of alignment. The reported fraction is descriptive, not an exact p-value under drift. Check distinct orders in `mismatch_control.csv`.", "",
             "## Aggregate held-out results", "",
             "| Model | Gain MAE (pp) | MAE skill |", "| --- | ---: | ---: |"]
    for row in metrics.loc[metrics.fold.eq("all") & metrics.action.eq("all")].itertuples():
        skill = f"{row.mae_skill:.6g}" if pd.notna(row.mae_skill) else "NA"
        lines.append(f"| {row.model} | {row.gain_mae_pp:.6g} | {skill} |")
    lines += ["", "| Selection | MAE | Relative regret | Within 1% |", "| --- | ---: | ---: | ---: |"]
    for row in choices.loc[choices.fold.eq("all")].itertuples():
        regret = f"{row.mean_relative_regret:.6g}" if pd.notna(row.mean_relative_regret) else "NA"
        lines.append(f"| {row.model} | {row.mae:.6g} | {regret} | {row.within_1pct:.3f} |")
    lines += ["", "## Scope and checks", "",
              "Gains are 100*(1-action MAE/native MAE); native is fixed at zero and excluded from regression scores. All-action MAE averages balanced non-native actions; R2/Spearman are per-action only. Undefined quantities are blank in CSV, not zero.", "",
              "Forward evaluation uses 50–65%, 65–80%, 80–100% of saved positions. Training labels must end at or before the first evaluation clock (clock+K+H). All preprocessing and Best Fixed choices use only that fold's training positions. `split_audit.csv` records the exclusions.", "",
              "Zero-native states are excluded and listed in audit.json. Zero-oracle states retain absolute regret but relative regret is undefined. Native wins predicted-gain ties according to the saved action order.", "",
              "These are local counterfactual results at saved Native states, not a continuous online trajectory. Correlation does not establish causation; feature-group improvements do not establish the independent value of every feature. One seed is not multi-seed evidence. Missing/weak gains and wide or inconsistent intervals mean insufficient evidence, not proof of unpredictability.", "",
              "No forecast replay, checkpoint deserialization, new feature collection, or parameter search was performed. Current local policy defaults were not used.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="batch root, job directory, or training directory")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing results are never overwritten")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--permutations", type=int, default=100, help="Full tree training-alignment negative controls; 0 disables")
    parser.add_argument("--permutation-block", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026, help="analysis resampling seed, not a forecast-model seed")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if (min(args.bootstrap_repeats, args.permutations, args.seed) < 0
            or min([args.permutation_block, *args.block_sizes]) < 1):
        parser.error("repeat counts/seed must be nonnegative and block sizes must be positive")
    if args.output.exists():
        parser.error("output directory already exists; choose a new output directory")
    try:
        directories = discover_training(args.runs)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True)
    save_json(args.output / "manifest.json", {
        "input": str(args.runs.resolve()), "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "script_sha256": file_sha256(Path(__file__)),
        "scope": "saved-branch offline analysis only; no continuous-online replay"})
    inventory, summaries = [], []
    for directory in directories:
        entry = {"input": str(directory.resolve())}
        try:
            path = directory / "manifest.json"
            if not path.exists():
                entry.update(status="pending", detail="collection manifest not yet available")
            else:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("status") in ("planned", "running"):
                    entry.update(status="pending", detail=f"collection status: {manifest.get('status')}")
                elif manifest.get("status") != "complete":
                    entry.update(status="invalid", detail=f"collection status: {manifest.get('status')}; {manifest.get('error', '')}")
                else:
                    destination = args.output / f"{directory.parent.parent.name}__{directory.parent.name}"
                    summary = analyze_collection(directory, destination, args)
                    summaries.append(summary)
                    entry.update(status="analyzed", detail="complete, hashes verified", output=str(destination.resolve()))
        except (ValueError, KeyError, OSError) as error:
            entry.update(status="invalid", detail=str(error))
        inventory.append(entry)
        print(f"{entry['status']}: {directory} ({entry['detail']})", flush=True)
    pd.DataFrame(inventory).to_csv(args.output / "inventory.csv", index=False)
    pd.DataFrame(summaries, columns=["dataset", "method", "horizon", "seed", "positions", "evaluation_positions",
                                     "features", "max_depth", "min_samples_leaf", "full_gain_mae_pp", "full_mae_skill",
                                     "full_relative_regret", "native_relative_regret", "fixed_relative_regret"]).to_csv(
        args.output / "summary.csv", index=False)
    counts = pd.Series([row["status"] for row in inventory]).value_counts().to_dict()
    (args.output / "README.md").write_text(
        "# Saved-policy predictability analysis\n\n"
        + f"Jobs: {counts}. See inventory.csv for completed, pending and invalid collections.\n\n"
        + "summary.csv compares completed jobs; each job directory contains its evidence tables, figures and reading guide. "
        + "Pending jobs are not treated as failed experiments. Rerun into a new output directory when more collections finish. "
        + "Local counterfactual selection cannot establish continuous-online performance.\n", encoding="utf-8")
    return 1 if counts.get("invalid", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
