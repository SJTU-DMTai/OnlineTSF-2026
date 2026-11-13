"""Historical-only temperature selection for soft action classification."""

import numpy as np

from .feature_selection import purged_time_splits, select_state_features
from .selector import ActionSelector, soft_targets


def select_temperature(selector: ActionSelector, states: np.ndarray, maes: np.ndarray,
                       gains: np.ndarray, origins: np.ndarray, label_ends: np.ndarray, *,
                       temperatures=(.01, .02, .05, .1, .2), cv_splits: int = 3,
                       feature_selection: dict | None = None) -> dict:
    """Choose by held-out action MAE, then refit on all historical states.

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
    report = {"method": "purged_forward_mae", "selection_metric": "mean_selected_mae",
              "tie_break": "larger_temperature", "folds": [], "candidates": []}
    fold_features = []
    for train, validation in folds:
        names = selector.features
        if feature_selection is not None:
            selection = select_state_features(states[train], gains[train], names, selector.catalog.ids,
                                              selector.catalog.native_indices, **feature_selection)
            names = selection["selected_features"]
        indices = np.asarray([selector.features.index(name) for name in names], dtype=int)
        fold_features.append((names, indices))
        report["folds"].append({"train_origins": origins[train].tolist(),
                                "validation_origins": origins[validation].tolist(),
                                "train_label_end_max": int(label_ends[train].max()), "features": names})
    for temperature in sorted(temperatures):
        total_mae = total_regret = 0.
        total_states = 0
        fold_scores = []
        for (train, validation), (names, indices) in zip(folds, fold_features):
            candidate = ActionSelector(selector.catalog, names, "classification", selector.estimator,
                                       parameters=selector.parameters, seed=selector.seed,
                                       feature_window=selector.feature_window, error_window=selector.error_window,
                                       temperature=temperature)
            candidate.fit(states[train][:, indices], maes[train])
            selected, _ = candidate.select(states[validation][:, indices])
            actual = maes[validation, selected]
            regret = actual - maes[validation].min(axis=1)
            total_mae += actual.sum()
            total_regret += regret.sum()
            total_states += len(validation)
            fold_scores.append({"states": len(validation), "mean_selected_mae": float(actual.mean()),
                                "mean_regret": float(regret.mean())})
        report["candidates"].append({"temperature": temperature,
                                     "mean_selected_mae": float(total_mae / total_states),
                                     "mean_regret": float(total_regret / total_states), "fold_scores": fold_scores})
    best = min(report["candidates"], key=lambda row: (row["mean_selected_mae"], -row["temperature"]))
    report["temperature"] = best["temperature"]
    return report
