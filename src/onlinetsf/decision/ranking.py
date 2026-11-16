"""Pure all-pairs hinge ranking, isolated from the existing MAE/CE training."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .selector import ActionSelector, MLPEstimator


RANKING_PAIRS = {("independent_regression", "mlp"), ("shared_regression", "mlp"),
                 ("classification", "mlp")}


def margin_ranking_loss(scores: torch.Tensor, maes: torch.Tensor, margin: float) -> torch.Tensor:
    """Mean over unequal unordered pairs per state, then over states.

    Scores are higher-is-better. Exact MAE ties carry no constraint; comparisons
    retain the original float64 labels, even when the networks use float32.
    """
    left, right = torch.triu_indices(scores.shape[1], scores.shape[1], offset=1, device=scores.device)
    signs = torch.sign(maes[:, right] - maes[:, left])
    valid = signs != 0
    losses = F.margin_ranking_loss(scores[:, left], scores[:, right], signs.to(scores.dtype),
                                   margin=margin, reduction="none")
    return ((losses * valid).sum(1) / valid.sum(1).clamp_min(1)).mean()


class RankingMLP(MLPEstimator):
    """Same MLP sizes and optimizer schedule, with state-grouped ranking batches."""

    def __init__(self, action_count: int, formulation: str, margin: float, *, seed: int, **parameters) -> None:
        super().__init__(action_count if formulation == "classification" else 1,
                         classification=formulation == "classification", seed=seed, **parameters)
        self.action_count, self.formulation, self.margin = action_count, formulation, margin

    def fit(self, x: np.ndarray, y: np.ndarray):
        values = torch.as_tensor(x, dtype=torch.float32)
        targets = torch.as_tensor(y, dtype=torch.float64)
        if self.formulation == "shared_regression":
            values = values.reshape(len(y), self.action_count, -1)
        with torch.random.fork_rng(devices=[]):
            networks = []
            for index in range(self.action_count if self.formulation == "independent_regression" else 1):
                torch.random.default_generator.manual_seed(self.seed + index)
                layers = []
                sizes = (values.shape[-1], *self.hidden_sizes)
                for left, right in zip(sizes[:-1], sizes[1:]):
                    layers.extend((nn.Linear(left, right), nn.ReLU()))
                layers.append(nn.Linear(sizes[-1], self.outputs))
                networks.append(nn.Sequential(*layers))
            self.networks = nn.ModuleList(networks)
            torch.random.default_generator.manual_seed(self.seed)
            optimizer = torch.optim.Adam(self.networks.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(self.epochs - 1, 1), eta_min=self.min_learning_rate)
            self.learning_rates_ = []
            self.epochs_trained = 0
            for epoch in range(self.epochs):
                self.learning_rates_.append(float(optimizer.param_groups[0]["lr"]))
                self.networks.train()
                for batch in torch.randperm(len(targets)).split(self.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    loss = margin_ranking_loss(self.forward(values[batch]), targets[batch], self.margin)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite margin ranking loss")
                    loss.backward()
                    optimizer.step()
                self.epochs_trained = epoch + 1
                if epoch + 1 < self.epochs:
                    scheduler.step()
        self.networks.eval()
        return self

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.formulation == "independent_regression":
            return torch.cat([network(values) for network in self.networks], dim=1)
        if self.formulation == "shared_regression":
            return self.networks[0](values).squeeze(-1)
        return self.networks[0](values)

    def predict_scores(self, x: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(x, dtype=torch.float32)
        if self.formulation == "shared_regression":
            values = values.reshape(-1, self.action_count, values.shape[-1])
        with torch.no_grad():
            scores = self.forward(values)
            # Preserve D's probability output and its existing probability EMA.
            if self.classification:
                scores = scores.softmax(dim=1)
        return scores.numpy()


class RankingSelector(ActionSelector):
    """Same state/action preprocessing and persistence, pure ranking training."""

    def __init__(self, *args, margin: float = 1., **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (self.formulation, self.estimator) not in RANKING_PAIRS:
            raise ValueError("margin ranking supports only A/MLP, B/MLP and D/MLP")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
            raise ValueError("ranking margin must be finite and nonnegative")
        self.margin = margin

    def fit(self, states: np.ndarray, targets: np.ndarray):
        targets = np.asarray(targets, dtype=float)
        x = self.prepare_training(states, targets)
        model = RankingMLP(len(self.catalog.ids), self.formulation, self.margin,
                           seed=self.seed, **self.parameters).fit(x, targets)
        self.models = [model]
        return self

    def scores(self, states: np.ndarray) -> np.ndarray:
        x = self.design(states)
        return self.models[0].predict_scores(x)

    def rank_scores(self, scores: np.ndarray) -> np.ndarray:
        return scores

    @property
    def score_kind(self) -> str:
        return "ranking_probability" if self.formulation == "classification" else "ranking_score"
