# -*- coding: utf-8 -*-
"""Train a forecasting baseline and inspect residual drift."""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from onlinetsf import (
    LinearForecastBackbone,
    PageHinkleyDetector,
    PatchTSTForecastBackbone,
    TCNForecastBackbone,
    load_benchmark_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("etth1", "etth2", "traffic"), required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model", choices=("linear", "tcn", "patchtst"), default="linear")
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    dataset = load_benchmark_dataset(
        args.dataset,
        args.data_path,
        context_length=args.context_length,
        horizon=args.horizon,
    )
    train_end = int(dataset.values.shape[0] * 0.8)
    mean = dataset.values[:train_end].mean(dim=0, keepdim=True)
    std = dataset.values[:train_end].std(dim=0, keepdim=True).clamp_min(1e-6)
    dataset.values = (dataset.values - mean) / std

    train_indices = [
        index
        for index in range(len(dataset))
        if index + args.context_length + args.horizon <= train_end
    ]
    eval_indices = list(range(max(0, train_end - args.context_length), len(dataset)))
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(Subset(dataset, eval_indices), batch_size=args.batch_size)

    if args.model == "linear":
        model = LinearForecastBackbone(
            context_length=args.context_length,
            num_features=dataset.num_features,
            horizon=args.horizon,
            num_targets=dataset.num_targets,
        )
    elif args.model == "tcn":
        model = TCNForecastBackbone(
            context_length=args.context_length,
            num_features=dataset.num_features,
            horizon=args.horizon,
            num_targets=dataset.num_targets,
        )
    else:
        model = PatchTSTForecastBackbone(
            context_length=args.context_length,
            num_features=dataset.num_features,
            horizon=args.horizon,
            num_targets=dataset.num_targets,
            target_indices=dataset.target_indices,
            patch_length=args.patch_length,
            patch_stride=args.patch_stride,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for context, target in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(context), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * context.shape[0]
        print(f"epoch={epoch + 1} train_mse={total_loss / len(train_indices):.6f}")

    detector = PageHinkleyDetector(delta=0.001, threshold=0.5, min_instances=10)
    model.eval()
    with torch.no_grad():
        for batch_index, (context, target) in enumerate(eval_loader):
            mae = (model(context) - target).abs().mean().item()
            update = detector.update(mae)
            if update.detected:
                print(f"drift detected at evaluation batch {batch_index}: score={update.score:.4f}")
                detector.reset()


if __name__ == "__main__":
    main()
