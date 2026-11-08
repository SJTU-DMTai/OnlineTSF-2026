"""Historical-only temperature selection for soft action classification."""

import numpy as np

from .feature_selection import purged_time_splits
from .selector import ActionSelector, soft_targets


def select_temperature(selector: ActionSelector, states: np.ndarray, gains: np.ndarray,
                       origins: np.ndarray, label_ends: np.ndarray, *,
                       temperatures=(.01, .02, .05, .1, .2), cv_splits: int = 3) -> dict:
    """Choose by realized held-out gain, then refit externally on all historical states.

    Cross-entropies at different temperatures are not comparable selection scores.
    Every fold independently fits preprocessing and purges labels not yet arrived.
    """
    if selector.formulation != "classification":
        raise ValueError("temperature selection requires a classification selector")
    if not temperatures or len(set(temperatures)) != len(temperatures):
        raise ValueError("temperature candidates must be nonempty and unique")
    for temperature in temperatures:
        soft_targets(np.zeros((1, 1)), temperature)
    folds = purged_time_splits(origins, label_ends, cv_splits)
    report = {"method": "purged_forward_gain", "selection_metric": "mean_selected_relative_mae_gain",
              "tie_break": "larger_temperature", "folds": [], "candidates": []}
    for train, validation in folds:
        report["folds"].append({"train_origins": origins[train].tolist(),
                                "validation_origins": origins[validation].tolist(),
                                "train_label_end_max": int(label_ends[train].max())})
    for temperature in sorted(temperatures):
        total_gain = total_regret = 0.
        total_states = 0
        fold_scores = []
        for train, validation in folds:
            candidate = ActionSelector(selector.catalog, selector.features, "classification", selector.estimator,
                                       parameters=selector.parameters, seed=selector.seed,
                                       feature_window=selector.feature_window, error_window=selector.error_window,
                                       temperature=temperature)
            candidate.fit(states[train], gains[train])
            selected, _ = candidate.select(states[validation])
            actual = gains[validation, selected]
            regret = gains[validation].max(axis=1) - actual
            total_gain += actual.sum()
            total_regret += regret.sum()
            total_states += len(validation)
            fold_scores.append({"states": len(validation), "mean_selected_gain": float(actual.mean()),
                                "mean_regret": float(regret.mean())})
        report["candidates"].append({"temperature": temperature,
                                     "mean_selected_gain": float(total_gain / total_states),
                                     "mean_regret": float(total_regret / total_states), "fold_scores": fold_scores})
    best = max(report["candidates"], key=lambda row: (row["mean_selected_gain"], row["temperature"]))
    report["temperature"] = best["temperature"]
    return report
