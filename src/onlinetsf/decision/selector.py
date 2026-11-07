"""Three selection formulations with RF, LightGBM or a small PyTorch MLP."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .catalog import ActionCatalog


FORMULATIONS = ("independent_regression", "shared_regression", "classification")
ESTIMATORS = ("random_forest", "lightgbm", "mlp")


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
        targets = torch.as_tensor(y, dtype=torch.long) if self.classification else torch.as_tensor(
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
    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        model_type = RandomForestClassifier if classification else RandomForestRegressor
        options = {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 4, "n_jobs": 1, **parameters}
    elif kind == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        model_type = LGBMClassifier if classification else LGBMRegressor
        options = {"n_estimators": 200, "learning_rate": .05, "num_leaves": 7, "min_child_samples": 4,
                   "verbosity": -1, "n_jobs": 1, **parameters}
    else:
        raise ValueError(f"unknown estimator: {kind}")
    if classification and len(np.unique(y)) == 1:
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="constant", constant=y[0]).fit(x, y)
    return model_type(random_state=seed, **options).fit(x, y)


class ActionSelector:
    def __init__(self, catalog: ActionCatalog, features: list[str], formulation: str,
                 estimator: str, *, parameters: dict | None = None, seed: int = 0,
                 feature_window: int = 32, error_window: int = 128) -> None:
        if formulation not in FORMULATIONS or estimator not in ESTIMATORS:
            raise ValueError("unknown selector formulation or estimator")
        if len(features) != len(set(features)):
            raise ValueError("state features must be unique")
        self.catalog, self.features = catalog, list(features)
        self.formulation, self.estimator = formulation, estimator
        self.parameters, self.seed = dict(parameters or {}), seed
        self.feature_window, self.error_window = feature_window, error_window
        self.models = []

    def transform(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != len(self.features):
            raise ValueError("state matrix differs from the saved feature schema")
        if not self.features:
            # ElasticNet may reject every state feature. A constant input lets
            # the existing models learn a fixed action preference honestly.
            return np.zeros((len(states), 1))
        missing = ~np.isfinite(states)
        result = np.column_stack((np.where(missing, self.medians, states), missing.astype(float)))
        return (result - self.mean) / self.scale

    def design(self, states: np.ndarray) -> np.ndarray:
        states = self.transform(states)
        if self.formulation != "shared_regression":
            return states
        actions = self.catalog.encode()
        if self.estimator == "mlp":
            actions = (actions - self.action_mean) / self.action_scale
        return np.column_stack((np.repeat(states, len(actions), axis=0), np.tile(actions, (len(states), 1))))

    def fit(self, states: np.ndarray, gains: np.ndarray) -> ActionSelector:
        states, gains = np.asarray(states, dtype=float), np.asarray(gains, dtype=float)
        if states.ndim != 2 or states.shape[1] != len(self.features) or not len(states):
            raise ValueError("training requires a nonempty state matrix matching the feature schema")
        if gains.shape != (len(states), len(self.catalog.ids)) or not np.isfinite(gains).all():
            raise ValueError("training requires a finite gain for every state and action")
        if not np.all(gains[:, self.catalog.native_indices] == 0):
            raise ValueError("Native training gains must be zero")
        self.medians = np.asarray([float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.
                                   for column in states.T])
        self.mean, self.scale = np.zeros(2 * states.shape[1]), np.ones(2 * states.shape[1])
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
        targets = ([gains[:, i] for i in range(len(self.catalog.ids))] if self.formulation == "independent_regression"
                   else [self.catalog.best(gains) if classification else gains.reshape(-1)])
        for index, target in enumerate(targets):
            if self.formulation == "independent_regression" and index in self.catalog.native_indices:
                self.models.append(None)
                continue
            self.models.append(fit_estimator(self.estimator, x, target, classification=classification,
                                            class_count=len(self.catalog.ids), parameters=self.parameters,
                                            seed=self.seed + index))
        return self

    def scores(self, states: np.ndarray) -> np.ndarray:
        x = self.design(states)
        if self.formulation == "independent_regression":
            scores = np.column_stack([np.zeros(len(x)) if model is None else model.predict(x) for model in self.models])
        elif self.formulation == "shared_regression":
            scores = self.models[0].predict(x).reshape(-1, len(self.catalog.ids))
        else:
            scores = np.zeros((len(x), len(self.catalog.ids)))
            scores[:, np.asarray(self.models[0].classes_, dtype=int)] = self.models[0].predict_proba(x)
        if self.formulation != "classification":
            scores[:, self.catalog.native_indices] = 0.
        return scores

    def select(self, states: np.ndarray, legal: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        scores = self.scores(states)
        return self.catalog.best(scores, legal), scores

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

    def __call__(self, state):
        for action in self.selector.catalog.actions:
            state.capabilities.validate(action)
        values = np.asarray([[state.features[name] for name in self.selector.features]], dtype=float)
        selected, scores = self.selector.select(values)
        index = int(selected[0])
        self.last_selection = {"action_id": self.selector.catalog.ids[index],
                               "scores": dict(zip(self.selector.catalog.ids, scores[0].tolist())),
                               "score_kind": "probability" if self.selector.formulation == "classification" else "relative_mae_gain",
                               "features": {name: state.features[name] for name in self.selector.features}}
        return self.selector.catalog.actions[index]
