"""Teacher top-1 cost-weighted BPR, isolated from ordinary regression and CE."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .selector import ActionSelector, MLPEstimator, resolve_teacher_actions


RANKING_PAIRS = {("independent_regression", "mlp"), ("shared_regression", "mlp"),
                 ("classification", "mlp")}


def top1_bpr_loss(scores: torch.Tensor, costs: torch.Tensor,
                   teacher_actions: torch.Tensor, r_max: float) -> torch.Tensor:
    """Sum capped-regret BPR over teacher-versus-rest pairs; mean over states.

    Equal-cost alternatives have zero weight. Regrets are computed in the
    label precision before conversion to the network's dtype.
    """
    teacher_cost = costs.gather(1, teacher_actions[:, None])
    weights = (costs - teacher_cost).clamp(min=0, max=r_max).to(scores.dtype)
    teacher_score = scores.gather(1, teacher_actions[:, None])
    return (weights * F.softplus(scores - teacher_score)).sum(1).mean()


class RankingMLP(MLPEstimator):
    """Same MLP sizes and optimizer schedule, with state-grouped ranking batches."""

    def __init__(self, action_count: int, formulation: str, r_max: float, *, seed: int, **parameters) -> None:
        super().__init__(action_count if formulation == "classification" else 1,
                         classification=formulation == "classification", seed=seed, **parameters)
        self.action_count, self.formulation, self.r_max = action_count, formulation, r_max

    def fit(self, x: np.ndarray, y: np.ndarray, teacher_actions: np.ndarray):
        values = torch.as_tensor(x, dtype=torch.float32)
        targets = torch.as_tensor(y, dtype=torch.float64)
        teacher = torch.as_tensor(teacher_actions, dtype=torch.long)
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
                    loss = top1_bpr_loss(self.forward(values[batch]), targets[batch], teacher[batch], self.r_max)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite top-1 BPR loss")
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
            # Apply classification EMA to probabilities, as in ordinary classification.
            if self.classification:
                scores = scores.softmax(dim=1)
        return scores.numpy()


class RankingSelector(ActionSelector):
    """Same state/action preprocessing and persistence, pure ranking training."""

    def __init__(self, *args, r_max: float = 1., **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (self.formulation, self.estimator) not in RANKING_PAIRS:
            raise ValueError("top1_bpr supports only MLP selectors")
        if self.label_mode != "soft":
            raise ValueError("pure top1_bpr has no soft/hard label distinction")
        if isinstance(r_max, bool) or not isinstance(r_max, (int, float)) or not math.isfinite(r_max) or r_max <= 0:
            raise ValueError("ranking r_max must be finite and positive")
        self.r_max = r_max

    def fit(self, states: np.ndarray, targets: np.ndarray, *,
            teacher_actions: np.ndarray | None = None):
        targets = np.asarray(targets, dtype=float)
        x = self.prepare_training(states, targets)
        teacher = resolve_teacher_actions(self.catalog, targets, teacher_actions)
        teacher_costs = targets[np.arange(len(targets)), teacher]
        if np.any(targets < teacher_costs[:, None]):
            raise ValueError("BPR requires teacher-minimal costs; enable teacher anchoring during collection")
        model = RankingMLP(len(self.catalog.ids), self.formulation, self.r_max,
                           seed=self.seed, **self.parameters).fit(x, targets, teacher)
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
