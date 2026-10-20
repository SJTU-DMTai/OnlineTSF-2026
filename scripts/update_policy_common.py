"""Shared causal features and small, JSON-serializable configuration selectors.

This module deliberately does not import torch. Training uses scikit-learn;
selection only requires NumPy/pandas and the saved tree arrays.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


DEFAULT_RUNS = Path("runs/update-study-completed/update-study-main-4gpu")
DEFAULT_ANNOTATIONS = Path("runs/real-drift-electricity-5pct-20261010")
DATA_FEATURES = ["channel_shift_fraction", "volatility_ratio"]
FEEDBACK_FEATURES = ["mae_32", "error_ratio"]
EXPERT_FEATURES = {
    "onenet": ["expert_relative_gap", "combined_time_weight"],
    "fsnet": ["gradient_memory_norm", "memory_interactions"],
    "dsof": ["replay_fill_fraction", "replay_updates_due"],
}
BASIC_ACTIONS = {
    "onenet": ["native", "every1_lr0.3", "every4_lr0.3", "only_variable+decision"],
    "fsnet": ["native", "every1_lr0.3", "every4_lr0.3", "only_base+adapter"],
    "dsof": ["native", "every1_lr0.3", "every4_lr0.3", "only_student+online"],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_job(runs: Path, dataset: str, method: str, horizon: int, seed: int) -> Path:
    matches = list(runs.glob(f"**/{dataset}/{method}_h{horizon}_seed{seed}/policies.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one job for {dataset}/{method}/H={horizon}/seed={seed}, found {len(matches)}")
    return matches[0].parent


def numeric_inputs(directory: Path, prefix_rows: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use the forecasting prefix z-score, independently of detector normalization."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(directory / "labeled_data.csv", usecols=manifest["feature_columns"] + ["annotation_phase"])
    raw = frame[manifest["feature_columns"]].to_numpy(dtype=np.float64)
    scale = raw[:prefix_rows].std(axis=0)
    scale[scale < 1e-6] = 1.0
    values = (raw - raw[:prefix_rows].mean(axis=0)) / scale
    if not np.isfinite(values).all():
        raise ValueError("non-finite forecasting input")
    return values, frame.annotation_phase.to_numpy(), manifest


def decision_features(state: dict, values: np.ndarray, clock: int) -> dict:
    """Same feature calculation for saved states and live, pre-feedback decisions."""
    if clock < 63:
        raise ValueError("features require 64 observed rows")
    current, previous = values[clock - 31:clock + 1], values[clock - 63:clock - 31]
    previous_std = previous.std(axis=0)
    result = dict(state)
    result["channel_shift_fraction"] = float(np.mean(
        np.abs(current.mean(axis=0) - previous.mean(axis=0)) > previous_std + 1e-6))
    result["volatility_ratio"] = float(current.std(axis=0).mean() / max(previous_std.mean(), 1e-10))
    short, long = state.get("mae_32"), state.get("mae_128")
    result["error_ratio"] = float(short / max(long, 1e-10)) if pd.notna(short) and pd.notna(long) else np.nan
    time, variable = state.get("time_expert_mae"), state.get("variable_expert_mae")
    result["expert_relative_gap"] = (float((time - variable) / max(time + variable, 1e-10))
                                    if pd.notna(time) and pd.notna(variable) else np.nan)
    return result


def historical_training_data(states: pd.DataFrame, outcomes: pd.DataFrame, train_seeds: list[int],
                             test_start: int, interval: int, horizon: int, actions: list[str],
                             metric: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Purge every branch with a target at or after the first test origin.

    A branch at raw origin c, scored over K forecasts of length H, uses target
    rows [c + 1, c + K + H). Its entire label must be available before test_start.
    """
    selected = states[states.seed.isin(train_seeds)].copy()
    if selected.duplicated(["seed", "probe_id"]).any():
        raise ValueError("duplicate source states")
    audit = selected[["seed", "probe_id", "clock"]].copy()
    audit["target_end_exclusive"] = audit.clock + interval + horizon
    audit["partition"] = np.where(audit.target_end_exclusive <= test_start, "train", "excluded_future_targets")
    selected = selected.loc[audit.partition == "train"].reset_index(drop=True)
    if selected.probe_id.nunique() < 3:
        raise ValueError("at least three historical training positions are required; choose a later test-start")
    # Join eligible states first. Excluded future losses never enter validation,
    # preprocessing, best-fixed selection or fitting.
    eligible = outcomes[(outcomes.lookahead == interval) & outcomes.policy.isin(actions)].merge(
        selected[["seed", "probe_id"]], on=["seed", "probe_id"], validate="many_to_one")
    if eligible.duplicated(["seed", "probe_id", "policy"]).any():
        raise ValueError("duplicate historical outcomes")
    losses = eligible.pivot(index=["seed", "probe_id"], columns="policy", values=metric)
    losses = losses.reindex(index=pd.MultiIndex.from_frame(selected[["seed", "probe_id"]]), columns=actions)
    if not np.isfinite(losses.to_numpy()).all() or (losses.native <= 0).any():
        raise ValueError("every historical state requires all action outcomes and positive native loss")
    losses.index = selected.index
    return selected, losses, audit


def fit_selector(states: pd.DataFrame, gains: pd.DataFrame, actions: list[str], features: list[str],
                 max_depth: int, min_samples_leaf: int) -> tuple[dict, str]:
    """One small regression tree per action predicts relative MAE/MSE improvement."""
    from sklearn.tree import DecisionTreeRegressor, export_text

    raw = states[features].to_numpy(dtype=float)
    medians = [float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.0
               for column in raw.T]
    missing = ~np.isfinite(raw)
    design = np.column_stack([np.where(missing, medians, raw), missing.astype(float)])
    weights = 1 / states.groupby("probe_id")["probe_id"].transform("size").to_numpy()
    selector = {"features": features, "medians": medians, "trees": {}, "actions": actions}
    rules = []
    for action in actions:
        if action == "native":
            continue
        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
        tree.fit(design, gains[action].to_numpy(), sample_weight=weights)
        selector["trees"][action] = {
            "left": tree.tree_.children_left.tolist(), "right": tree.tree_.children_right.tolist(),
            "feature": tree.tree_.feature.tolist(), "threshold": tree.tree_.threshold.tolist(),
            "value": tree.tree_.value[:, 0, 0].tolist(),
        }
        rules.append(action + "\n" + export_text(tree, feature_names=features + [f"missing_{x}" for x in features]))
    return selector, "\n".join(rules)


def select_actions(selector: dict, states: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    raw = states[selector["features"]].to_numpy(dtype=float)
    missing = ~np.isfinite(raw)
    # sklearn's tree prediction casts inputs to float32 before threshold tests.
    design = np.column_stack([np.where(missing, selector["medians"], raw), missing.astype(float)]).astype(np.float32)
    scores = np.zeros((len(states), len(selector["actions"])))
    for column, action in enumerate(selector["actions"]):
        if action == "native":
            continue
        tree = selector["trees"][action]
        for row, features in enumerate(design):
            node = 0
            while tree["left"][node] != -1:
                node = tree["left"][node] if features[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
            scores[row, column] = tree["value"][node]
    # Native is first and has a known relative improvement of zero, including ties.
    selected = [selector["actions"][index] for index in scores.argmax(axis=1)]
    return selected, scores


def save_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_timings(directory: Path, timings: dict) -> None:
    """Export measured durations in seconds; overlapping totals are not additive."""
    save_json(directory / "timings.json", timings)
    pd.DataFrame(timings.items(), columns=["stage", "seconds"]).to_csv(directory / "timings.csv", index=False)


def plot_forecasts(frames: dict[str, pd.DataFrame], destination: Path, title: str) -> None:
    """Actual emitted forecast errors, with a short trailing average for readability."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    colors = ["#285B86", "#D27735", "#399A80", "#9666A4", "#9B7045", "#BD455C"]
    for ax, metric in zip(axes, ["mae", "mse"]):
        for (name, frame), color in zip(frames.items(), colors):
            ax.plot(np.arange(1, len(frame) + 1), frame[metric].rolling(32, min_periods=1).mean(),
                    label=name, color=color, linewidth=1.5)
        ax.set(xlabel="Steps", ylabel=metric.upper() + " (trailing 32)")
        ax.grid(axis="y", color="#DDE2E6", linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=170)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def oracle_decision(outcomes: list[dict], selected: str, metric: str) -> dict:
    """Score a choice against the best fixed action over this local window.

    ACC accepts all numerically tied optima (rtol=1e-7, atol=1e-10).
    Relative regret is (selected - optimum) / optimum, not a percentage.
    """
    losses = np.array([row[metric] for row in outcomes], dtype=float)
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("oracle losses must be finite and nonnegative")
    best = float(losses.min())
    if best == 0:
        raise ValueError("relative regret is undefined for a zero optimal loss")
    names = [row["action"] for row in outcomes]
    chosen = float(losses[names.index(selected)])
    optimal = np.isclose(losses, best, rtol=1e-7, atol=1e-10)
    relative = (chosen - best) / best
    return {"metric": metric, "selected_action": selected,
            "oracle_action": names[int(losses.argmin())],
            "optimal_actions": json.dumps([name for name, match in zip(names, optimal) if match]),
            "selected_loss": chosen, "oracle_loss": best,
            "native_loss_same_state": float(losses[names.index("native")]),
            "correct": bool(optimal[names.index(selected)]),
            "relative_regret": relative, "relative_regret_pct": relative * 100,
            "within_1pct": bool(chosen <= best * 1.01 + 1e-12)}


def summarize_decisions(decisions: pd.DataFrame, metric: str) -> dict:
    """Each full decision interval has equal weight in these metrics."""
    return {"metric": metric, "decisions": len(decisions),
            "correct_decisions": int(decisions.correct.sum()), "acc": float(decisions.correct.mean()),
            "acc_pct": float(decisions.correct.mean() * 100),
            "mean_relative_regret": float(decisions.relative_regret.mean()),
            "mean_relative_regret_pct": float(decisions.relative_regret.mean() * 100),
            "median_relative_regret_pct": float(decisions.relative_regret.median() * 100),
            "within_1pct_decisions": int(decisions.within_1pct.sum()),
            "within_1pct_fraction": float(decisions.within_1pct.mean()),
            "within_1pct_pct": float(decisions.within_1pct.mean() * 100),
            "mean_selected_loss": float(decisions.selected_loss.mean()),
            "mean_oracle_loss_same_states": float(decisions.oracle_loss.mean()),
            "mean_native_loss_same_states": float(decisions.native_loss_same_state.mean())}


def score_action(replay, stream, index: int, steps: int, policy) -> dict:
    """Preserve unchanged-policy cadence and score only newly emitted forecasts."""
    began = perf_counter()
    branch = copy.deepcopy(replay)
    before_gradient, before_egd = branch.method.gradient_steps, branch.method.egd_steps
    branch.method.switch_policy(policy)
    count, absolute, squared = 0, 0., 0.
    for current in range(index, index + steps):
        for record in branch.step(stream.dataset, current):
            if record.get("index", -1) >= index:
                count += 1
                absolute += record["mae"]
                squared += record["mse"]
    # Mature terminal predictions without applying future gradient updates.
    for record in branch.finish(stream.dataset):
        if record["index"] >= index:
            count += 1
            absolute += record["mae"]
            squared += record["mse"]
    if count != steps:
        raise RuntimeError(f"{policy.name}: scored {count} forecasts, expected {steps}")
    return {"action": policy.name, "mae": absolute / steps, "mse": squared / steps,
            "forecasts": count, "gradient_steps": branch.method.gradient_steps - before_gradient,
            "egd_steps": branch.method.egd_steps - before_egd, "seconds": perf_counter() - began}
