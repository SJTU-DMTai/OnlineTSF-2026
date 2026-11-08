"""Fixed ElasticNet state screening and causal historical validation splits."""

import numpy as np


def purged_time_splits(origins: np.ndarray, label_ends: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Both clocks use window origins; label_ends are exclusive target bounds."""
    from sklearn.model_selection import TimeSeriesSplit

    origins, label_ends = np.asarray(origins), np.asarray(label_ends)
    if (origins.ndim != 1 or label_ends.shape != origins.shape or not np.isfinite(origins).all()
            or not np.isfinite(label_ends).all() or np.any(np.diff(origins) <= 0)
            or np.any(label_ends <= origins)):
        raise ValueError("time splits require increasing origins and later finite label ends")
    folds = []
    for train, validation in TimeSeriesSplit(n_splits=n_splits).split(origins):
        train = train[label_ends[train] <= origins[validation[0]]]
        if len(train) < 2:
            raise ValueError("too few historical states after label purging; collect more widely spaced states or reduce cv_splits")
        folds.append((train, validation))
    return folds


def select_state_features(states: np.ndarray, gains: np.ndarray, names: list[str],
                          action_ids: list[str], native_indices: list[int], *,
                          alpha: float = .06, l1_ratio: float = .5) -> dict:
    """Select the union of nonzero ElasticNet coefficients across action gains.

    A feature survives if its value or missing indicator has a nonzero weight.
    Parameters are fixed. Imputation, scaling and screening fit only the supplied
    historical training rows; no test samples enter this function.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    states, gains = np.asarray(states, dtype=float), np.asarray(gains, dtype=float)
    if (states.ndim != 2 or not len(states) or states.shape[1] != len(names) or not names
            or len(set(names)) != len(names)):
        raise ValueError("feature selection requires nonempty states matching unique candidate features")
    if gains.shape != (len(states), len(action_ids)) or not np.isfinite(gains).all():
        raise ValueError("feature selection requires finite gains for every state and action")
    if not np.all(gains[:, native_indices] == 0):
        raise ValueError("Native training gains must be zero")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not np.isfinite(alpha) or alpha <= 0
            or isinstance(l1_ratio, bool) or not isinstance(l1_ratio, (int, float))
            or not np.isfinite(l1_ratio) or not 0 < l1_ratio <= 1):
        raise ValueError("ElasticNet requires finite alpha > 0 and 0 < l1_ratio <= 1")
    targets = [i for i in range(len(action_ids)) if i not in native_indices and np.ptp(gains[:, i]) > 0]
    report = {"method": "elastic_net", "candidate_features": list(names), "selected_features": [],
              "target_actions": [action_ids[i] for i in targets],
              "parameters": {"alpha": alpha, "l1_ratio": l1_ratio},
              "state_count": len(states), "coefficients": []}
    if not targets:
        report["reason"] = "all non-Native gains are constant; use a state-independent decision model"
        return report
    model = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                      ("scale", StandardScaler()),
                      ("elastic_net", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000))])
    model.fit(np.where(np.isfinite(states), states, np.nan), gains[:, targets])
    coefficients = np.atleast_2d(model["elastic_net"].coef_)
    value_weights = coefficients[:, :len(names)]
    missing_weights = np.zeros_like(value_weights)
    missing_weights[:, model["imputer"].indicator_.features_] = coefficients[:, len(names):]
    selected = np.any((np.abs(value_weights) > 1e-8) | (np.abs(missing_weights) > 1e-8), axis=0)
    report.update(selected_features=[name for name, keep in zip(names, selected) if keep],
                  coefficients=[{"feature": name, "selected": bool(selected[i]),
                                 "value": value_weights[:, i].tolist(), "missing": missing_weights[:, i].tolist()}
                                for i, name in enumerate(names)])
    return report
