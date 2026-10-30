"""Offline fitting and causal training partitions for decision selectors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .selector import select_action


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
    results = [select_action(selector, row) for row in states.to_dict("records")]
    return [action for action, _ in results], np.array([[scores[name] for name in selector["actions"]]
                                                      for _, scores in results])
