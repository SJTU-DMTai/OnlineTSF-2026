"""Three selection formulations with RF, LightGBM or a small PyTorch MLP."""

from __future__ import annotations

import pickle
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .catalog import ActionCatalog


FORMULATIONS = ("independent_regression", "shared_regression", "classification")
ESTIMATORS = ("random_forest", "lightgbm", "mlp")
LOG1P_FEATURES = ("volatility_ratio", "normalized_mean_shift", "trend_shift")
FORMULATION_PARAMETERS = {
    "independent_regression": {
        "lightgbm": {"n_estimators": 100, "learning_rate": .03, "num_leaves": 3,
                     "min_child_samples": 12, "reg_lambda": 1.}},
    "shared_regression": {
        "random_forest": {"max_depth": 12, "min_samples_leaf": 2},
        "lightgbm": {"n_estimators": 600, "num_leaves": 31, "min_child_samples": 2}},
}


def soft_targets(maes: np.ndarray, temperature: float) -> np.ndarray:
    """Lower action MAE receives a higher soft classification target."""
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("classification temperature must be finite and positive")
    logits = (maes.min(axis=1, keepdims=True) - maes) / temperature
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


class MLPEstimator:
    """CPU MLP trained on all supplied samples for the configured epoch count."""

    def __init__(self, outputs: int, *, classification: bool, seed: int,
                 hidden_sizes=(64, 32), learning_rate: float = .001, epochs: int = 100,
                 batch_size: int = 64, weight_decay: float = .0001,
                 min_learning_rate: float | None = None) -> None:
        if (outputs <= 0 or not hidden_sizes or any(size <= 0 for size in hidden_sizes)
                or min(learning_rate, epochs, batch_size) <= 0 or weight_decay < 0):
            raise ValueError("invalid MLP dimensions or training parameters")
        self.outputs, self.classification, self.seed = outputs, classification, seed
        self.hidden_sizes, self.learning_rate = tuple(hidden_sizes), learning_rate
        self.min_learning_rate = learning_rate * .1 if min_learning_rate is None else min_learning_rate
        if (not np.isfinite([learning_rate, self.min_learning_rate]).all()
                or not 0 <= self.min_learning_rate <= learning_rate):
            raise ValueError("MLP requires a finite 0 <= min_learning_rate <= learning_rate")
        self.epochs, self.batch_size = epochs, batch_size
        self.weight_decay = weight_decay
        self.classes_ = np.arange(outputs)

    def fit(self, x: np.ndarray, y: np.ndarray) -> MLPEstimator:
        self.target_mean = 0. if self.classification else float(np.mean(y))
        self.target_scale = 1. if self.classification else max(float(np.std(y)), 1e-6)
        values = torch.as_tensor(x, dtype=torch.float32)
        targets = torch.as_tensor(y, dtype=torch.float32) if self.classification else torch.as_tensor(
            (y - self.target_mean) / self.target_scale, dtype=torch.float32).reshape(-1, 1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            layers = []
            sizes = (x.shape[1], *self.hidden_sizes)
            for left, right in zip(sizes[:-1], sizes[1:]):
                layers.extend((nn.Linear(left, right), nn.ReLU()))
            layers.append(nn.Linear(sizes[-1], self.outputs))
            self.network = nn.Sequential(*layers)
            optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            # First and last training epochs use the requested endpoints; a
            # single epoch uses the initial rate. Never step back up afterwards.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(self.epochs - 1, 1), eta_min=self.min_learning_rate)
            loss_fn = nn.CrossEntropyLoss() if self.classification else nn.MSELoss()
            self.epochs_trained = 0
            self.learning_rates_ = []
            for epoch in range(self.epochs):
                self.learning_rates_.append(float(optimizer.param_groups[0]["lr"]))
                self.network.train()
                for batch in torch.randperm(len(values)).split(self.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(self.network(values[batch]), targets[batch])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite MLP training loss")
                    loss.backward()
                    optimizer.step()
                self.epochs_trained = epoch + 1
                if epoch + 1 < self.epochs:
                    scheduler.step()
        self.network.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            result = self.network(torch.as_tensor(x, dtype=torch.float32)).numpy()
        return result[:, 0] * self.target_scale + self.target_mean

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.network(torch.as_tensor(x, dtype=torch.float32)).softmax(dim=1).numpy()


def fit_estimator(kind: str, x: np.ndarray, y: np.ndarray, *, classification: bool,
                  class_count: int, parameters: dict, seed: int):
    """Library-specific fitting stays outside the formulation and runtime code."""
    if kind == "mlp":
        model = MLPEstimator(class_count if classification else 1, classification=classification,
                             seed=seed, **parameters)
        return model.fit(x, y)
    weights = None
    if classification:
        if y.shape != (len(x), class_count):
            raise ValueError("classification requires one soft target probability per state and action")
        # Weighted copies have exactly the same cross-entropy as soft targets:
        # sum_a -q[state,a] * log p(a|state). All copies use identical inputs.
        weights = y.reshape(-1)
        y = np.tile(np.arange(class_count), len(x))
        x = np.repeat(x, class_count, axis=0)
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        model_type = RandomForestClassifier if classification else RandomForestRegressor
        options = {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 4, "n_jobs": 1, **parameters}
        if classification:
            options["criterion"] = "log_loss"
            # Convert integer leaf thresholds to the expanded sample scale.
            if isinstance(options["min_samples_leaf"], int):
                options["min_samples_leaf"] *= class_count
    elif kind == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        model_type = LGBMClassifier if classification else LGBMRegressor
        options = {"n_estimators": 200, "learning_rate": .05, "num_leaves": 7, "min_child_samples": 4,
                   "verbosity": -1, "n_jobs": 1, **parameters}
        if classification:
            options["min_child_samples"] *= class_count
    else:
        raise ValueError(f"unknown estimator: {kind}")
    if classification and class_count == 1:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="constant", constant=y[0]).fit(x, y)
    return model_type(random_state=seed, **options).fit(x, y, sample_weight=weights)


class ActionSelector:
    def __init__(self, catalog: ActionCatalog, features: list[str], formulation: str,
                 estimator: str, *, parameters: dict | None = None, seed: int = 0,
                 feature_window: int = 32, error_window: int = 128, temperature: float = .05) -> None:
        if formulation not in FORMULATIONS or estimator not in ESTIMATORS:
            raise ValueError("unknown selector formulation or estimator")
        if len(features) != len(set(features)):
            raise ValueError("state features must be unique")
        self.catalog, self.features = catalog, list(features)
        self.formulation, self.estimator = formulation, estimator
        self.parameters = {**FORMULATION_PARAMETERS.get(formulation, {}).get(estimator, {}), **(parameters or {})}
        self.seed = seed
        soft_targets(np.zeros((1, 1)), temperature)
        self.temperature = temperature
        self.feature_window, self.error_window = feature_window, error_window
        self.tau = 64
        self.models = []

    def transform(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != len(self.features):
            raise ValueError("state matrix differs from the saved feature schema")
        if not self.features:
            # A constant input represents a state-independent action preference.
            return np.zeros((len(states), 1))
        missing = ~np.isfinite(states)
        values = np.where(missing, self.medians, states)
        for index, name in enumerate(self.features):
            if name in LOG1P_FEATURES:
                values[:, index] = np.log1p(values[:, index])
        return (values - self.mean) / self.scale

    def design(self, states: np.ndarray) -> np.ndarray:
        states = self.transform(states)
        if self.formulation != "shared_regression":
            return states
        actions = self.catalog.encode()
        if self.estimator == "mlp":
            actions = (actions - self.action_mean) / self.action_scale
        return np.column_stack((np.repeat(states, len(actions), axis=0), np.tile(actions, (len(states), 1))))

    def fit(self, states: np.ndarray, targets: np.ndarray) -> ActionSelector:
        states, targets = np.asarray(states, dtype=float), np.asarray(targets, dtype=float)
        if states.ndim != 2 or states.shape[1] != len(self.features) or not len(states):
            raise ValueError("training requires a nonempty state matrix matching the feature schema")
        if targets.shape != (len(states), len(self.catalog.ids)) or not np.isfinite(targets).all():
            raise ValueError("training requires a finite target for every state and action")
        if np.any(targets < 0):
            raise ValueError("action MAE targets must be nonnegative")
        self.medians = np.asarray([float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.
                                   for column in states.T])
        self.mean, self.scale = np.zeros(states.shape[1]), np.ones(states.shape[1])
        if self.estimator == "mlp":
            filled = self.transform(states)
            self.mean, self.scale = filled.mean(0), filled.std(0)
            self.scale[self.scale < 1e-6] = 1.
        action_values = self.catalog.encode()
        self.action_mean, self.action_scale = action_values.mean(0), action_values.std(0)
        self.action_scale[self.action_scale < 1e-6] = 1.
        x = self.design(states)
        classification = self.formulation == "classification"
        self.models = []
        fit_targets = ([targets[:, i] for i in range(len(self.catalog.ids))] if self.formulation == "independent_regression"
                       else [soft_targets(targets, self.temperature) if classification else targets.reshape(-1)])
        for index, target in enumerate(fit_targets):
            self.models.append(fit_estimator(self.estimator, x, target, classification=classification,
                                            class_count=len(self.catalog.ids), parameters=self.parameters,
                                            seed=self.seed + index))
        return self

    def scores(self, states: np.ndarray) -> np.ndarray:
        x = self.design(states)
        if self.formulation == "independent_regression":
            scores = np.column_stack([model.predict(x) for model in self.models])
        elif self.formulation == "shared_regression":
            scores = self.models[0].predict(x).reshape(-1, len(self.catalog.ids))
        else:
            scores = np.zeros((len(x), len(self.catalog.ids)))
            scores[:, np.asarray(self.models[0].classes_, dtype=int)] = self.models[0].predict_proba(x)
        return scores

    def select(self, states: np.ndarray, legal: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        scores = self.scores(states)
        ranked = scores if self.formulation == "classification" else -scores
        return self.catalog.best(ranked, legal), scores

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: str | Path) -> ActionSelector:
        model = pickle.loads(Path(path).read_bytes())
        if not isinstance(model, cls):
            raise ValueError("artifact is not an ActionSelector")
        return model


class SelectorPolicy:
    """The runtime consumes state features and returns the persisted action."""

    def __init__(self, selector: ActionSelector) -> None:
        self.selector = selector
        self.last_selection: dict | None = None
        self.smoothed_scores: np.ndarray | None = None
        self.rho = math.exp(-1 / selector.tau)

    def __call__(self, state):
        for action in self.selector.catalog.actions:
            state.capabilities.validate(action)
        values = np.asarray([[state.features[name] for name in self.selector.features]], dtype=float)
        scores = self.selector.scores(values)[0]
        self.smoothed_scores = (scores.copy() if self.smoothed_scores is None else
                                self.rho * self.smoothed_scores + (1 - self.rho) * scores)
        ranked = (self.smoothed_scores if self.selector.formulation == "classification" else
                  -self.smoothed_scores)
        index = int(self.selector.catalog.best(ranked[None, :])[0])
        self.last_selection = {"action_id": self.selector.catalog.ids[index],
                               "scores": dict(zip(self.selector.catalog.ids, self.smoothed_scores.tolist())),
                               "raw_scores": dict(zip(self.selector.catalog.ids, scores.tolist())),
                               "ema_rho": self.rho,
                               "score_kind": "probability" if self.selector.formulation == "classification" else "action_mae",
                               "features": {name: state.features[name] for name in self.selector.features}}
        return self.selector.catalog.actions[index]
