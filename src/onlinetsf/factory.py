"""Shared construction of forecasting backbones and their native online methods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .forecasting import (
    DLinearForecastBackbone, LinearForecastBackbone, LSTMForecastBackbone,
    PatchTSTForecastBackbone, TCNForecastBackbone, TimeTCNForecastBackbone,
)
from .methods import (
    DSOFMethod, DSOFModel, FSNetMethod, FSNetTCN, OGDMethod,
    OneNetEnsemble, OneNetMethod, UnderCaliMethod,
)


def build_backbone(config: dict[str, Any], num_features: int, num_targets: int, target_indices: Sequence[int]):
    data = config["data"]
    forecasting = config["forecasting"]
    options = forecasting["parameters"]
    dimensions = {
        "context_length": data["context_length"],
        "num_features": num_features,
        "horizon": data["horizon"],
        "num_targets": num_targets,
    }
    backbone = forecasting["backbone"]
    if backbone == "linear":
        model = LinearForecastBackbone(**dimensions, **options)
    elif backbone == "dlinear":
        model = DLinearForecastBackbone(**dimensions, target_indices=target_indices, **options)
    elif backbone == "lstm":
        model = LSTMForecastBackbone(**dimensions, **options)
    elif backbone == "tcn":
        model = TCNForecastBackbone(**dimensions, **options)
    elif backbone == "patchtst":
        model = PatchTSTForecastBackbone(**dimensions, target_indices=target_indices, **options)
    elif backbone == "onenet_tcn":
        cross_time = TimeTCNForecastBackbone(
            **dimensions, target_indices=target_indices, **options
        )
        cross_variable = TCNForecastBackbone(**dimensions, **options)
        return OneNetEnsemble(
            cross_time,
            cross_variable,
            horizon=data["horizon"],
            num_targets=num_targets,
            decision_hidden=config["method"].get("decision_hidden", 32),
            decision_dropout=config["method"].get("decision_dropout", 0.1),
        )
    else:
        model = FSNetTCN(**dimensions, **options)
    if config["method"]["name"] == "dsof":
        method = config["method"]
        return DSOFModel(
            model,
            context_length=data["context_length"],
            horizon=data["horizon"],
            target_indices=tuple(target_indices),
            student_hidden=method.get("student_hidden", 16),
            student_depth=method.get("student_depth", 3),
            student_dropout=method.get("student_dropout", 0.0),
        )
    return model

def build_method(config: dict[str, Any], model: nn.Module):
    method = config["method"]
    device = config["online"].get("device")
    learning_rate = method.get("learning_rate")
    if method["name"] == "dsof":
        assert isinstance(model, DSOFModel)
        return DSOFMethod(
            model,
            learning_rate=learning_rate,
            student_learning_rate=method.get("student_learning_rate", 1e-3),
            online_learning_rate=method.get("online_learning_rate", 1e-3),
            replay_buffer_size=method.get("replay_buffer_size", 300),
            replay_batch_size=method.get("replay_batch_size", 32),
            replay_epochs=method.get("replay_epochs", 1),
            replay_frequency=method.get("replay_frequency", 1),
            discount=method.get("discount", 0.9),
            device=device,
        )
    if method["name"] == "onenet":
        assert isinstance(model, OneNetEnsemble)
        return OneNetMethod(
            model,
            learning_rate=learning_rate,
            weight_learning_rate=method.get("weight_learning_rate", 1e-3),
            decision_learning_rate=method.get("decision_learning_rate", 1e-3),
            n_inner=method.get("n_inner", 1),
            device=device,
        )
    if method["name"] == "fsnet":
        assert isinstance(model, FSNetTCN)
        return FSNetMethod(
            model,
            learning_rate=learning_rate,
            n_inner=method.get("n_inner", 1),
            device=device,
        )

    if method["name"] == "under_cali":
        return UnderCaliMethod(
            model,
            context_length=config["data"]["context_length"],
            horizon=config["data"]["horizon"],
            num_features=config["data"]["num_features"],
            num_targets=len(config["data"]["target_names"]),
            learning_rate=learning_rate,
            adapt_lr=method["adapt_lr"],
            uncertainty_lr=method["uncertainty_lr"],
            uncertainty_adapt_lr=method["uncertainty_adapt_lr"],
            calibrator_hidden_dim=method["calibrator_hidden_dim"],
            uncertainty_hidden_dim=method["uncertainty_hidden_dim"],
            uncertainty_epochs=method["uncertainty_epochs"],
            update_steps=method["update_steps"],
            allocation_alpha=method["allocation_alpha"],
            allocation_std_k=method["allocation_std_k"],
            trigger_alpha=method["trigger_alpha"],
            trigger_std_k=method["trigger_std_k"],
            device=device,
        )

    if learning_rate is None:
        return OGDMethod(model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return OGDMethod(
        model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        update_steps=method.get("update_steps", 1),
        device=device,
    )


