"""ElasticNet state screening on purged, forward-only historical folds."""

import numpy as np


def purged_time_splits(origins: np.ndarray, label_ends: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Both clocks use window origins; label_ends are exclusive target bounds."""
    from sklearn.model_selection import TimeSeriesSplit

    origins, label_ends = np.asarray(origins), np.asarray(label_ends)
    if (origins.ndim != 1 or label_ends.shape != origins.shape or not np.isfinite(origins).all()
            or not np.isfinite(label_ends).all() or np.any(np.diff(origins) <= 0)
            or np.any(label_ends <= origins)):
        raise ValueError("feature selection requires increasing origins and later finite label ends")
    folds = []
    for train, validation in TimeSeriesSplit(n_splits=n_splits).split(origins):
        train = train[label_ends[train] <= origins[validation[0]]]
        if len(train) < 2:
            raise ValueError("too few historical states after label purging; collect more widely spaced states or reduce cv_splits")
        folds.append((train, validation))
    return folds


def select_state_features(states: np.ndarray, gains: np.ndarray, names: list[str],
                          action_ids: list[str], native_indices: list[int],
                          origins: np.ndarray, label_ends: np.ndarray, *,
                          alphas=(.0001, .001, .01, .1), l1_ratios=(.5, .9, 1.),
                          cv_splits: int = 3) -> dict:
    """Select the union of nonzero ElasticNet coefficients across action gains.

    A feature survives if its value or missing indicator has a nonzero weight.
    Each CV fold fits its own imputation and scaling. The winning model is then
    refitted on all historical states; no test samples enter this function.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    states, gains = np.asarray(states, dtype=float), np.asarray(gains, dtype=float)
    if (states.ndim != 2 or states.shape != (len(origins), len(names)) or not names
            or len(set(names)) != len(names)):
        raise ValueError("feature selection states must match origins and unique candidate features")
    if gains.shape != (len(states), len(action_ids)) or not np.isfinite(gains).all():
        raise ValueError("feature selection requires finite gains for every state and action")
    alphas, l1_ratios = np.asarray(alphas, dtype=float), np.asarray(l1_ratios, dtype=float)
    if (alphas.ndim != 1 or not len(alphas) or not np.isfinite(alphas).all() or np.any(alphas <= 0)
            or l1_ratios.ndim != 1 or not len(l1_ratios) or not np.isfinite(l1_ratios).all()
            or np.any((l1_ratios <= 0) | (l1_ratios > 1))):
        raise ValueError("ElasticNet requires positive finite alphas and 0 < l1_ratios <= 1")
    targets = [i for i in range(len(action_ids)) if i not in native_indices and np.ptp(gains[:, i]) > 0]
    report = {"method": "elastic_net", "candidate_features": list(names), "selected_features": [],
              "target_actions": [action_ids[i] for i in targets], "best_parameters": None,
              "cv_results": [], "folds": [], "coefficients": [], "seconds": 0.}
    if not targets:
        report["reason"] = "all non-Native gains are constant; use a state-independent decision model"
        return report
    folds = purged_time_splits(origins, label_ends, cv_splits)
    pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                         ("scale", StandardScaler()), ("elastic_net", ElasticNet(max_iter=20000))])
    search = GridSearchCV(pipeline, {"elastic_net__alpha": alphas.tolist(), "elastic_net__l1_ratio": l1_ratios.tolist()},
                          scoring="neg_mean_squared_error", cv=folds, n_jobs=1, error_score="raise")
    search.fit(np.where(np.isfinite(states), states, np.nan), gains[:, targets])
    model = search.best_estimator_
    coefficients = np.atleast_2d(model["elastic_net"].coef_)
    value_weights = coefficients[:, :len(names)]
    missing_weights = np.zeros_like(value_weights)
    missing_weights[:, model["imputer"].indicator_.features_] = coefficients[:, len(names):]
    selected = np.any((np.abs(value_weights) > 1e-8) | (np.abs(missing_weights) > 1e-8), axis=0)
    report.update(selected_features=[name for name, keep in zip(names, selected) if keep],
                  best_parameters={"alpha": search.best_params_["elastic_net__alpha"],
                                   "l1_ratio": search.best_params_["elastic_net__l1_ratio"]},
                  cv_results=[{"alpha": params["elastic_net__alpha"], "l1_ratio": params["elastic_net__l1_ratio"],
                               "mean_mse": float(-score)}
                              for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])],
                  folds=[{"train_origins": np.asarray(origins)[train].tolist(),
                          "validation_origins": np.asarray(origins)[validation].tolist(),
                          "train_label_end_max": int(np.max(np.asarray(label_ends)[train]))}
                         for train, validation in folds],
                  coefficients=[{"feature": name, "selected": bool(selected[i]),
                                 "value": value_weights[:, i].tolist(), "missing": missing_weights[:, i].tolist()}
                                for i, name in enumerate(names)])
    return report
