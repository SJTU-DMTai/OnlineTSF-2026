# -*- coding: utf-8 -*-
"""CSV loading and rolling-window datasets for benchmark time series."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSpec:
    """Column conventions used by supported benchmark CSV files."""

    time_column: str
    target_columns: tuple[str, ...] | None


BENCHMARK_SPECS = {
    "etth1": DatasetSpec(time_column="date", target_columns=None),
    "etth2": DatasetSpec(time_column="date", target_columns=None),
    "ettm1": DatasetSpec(time_column="date", target_columns=None),
    "ettm2": DatasetSpec(time_column="date", target_columns=None),
    "traffic": DatasetSpec(time_column="date", target_columns=None),
    "weather": DatasetSpec(time_column="date", target_columns=None),
    "labeled": DatasetSpec(time_column="index", target_columns=None),
}


class SlidingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Creates ``[context, target]`` pairs from one uniformly sampled series.

    ``values`` has shape ``[time, features]``.  The target may select a subset
    of the input features.
    """

    def __init__(
        self,
        values: Tensor,
        context_length: int,
        horizon: int,
        target_indices: Sequence[int] | None = None,
        target_names: Sequence[str] | None = None,
        feature_names: Sequence[str] | None = None,
        time_features: Tensor | None = None,
        stride: int = 1,
    ) -> None:
        if values.ndim != 2:
            raise ValueError("values must have shape [time, features]")
        if context_length <= 0 or horizon <= 0 or stride <= 0:
            raise ValueError("context_length, horizon, and stride must be positive")
        if values.shape[0] < context_length + horizon:
            raise ValueError("series is shorter than one context-plus-horizon window")

        # A sliding sample contains visible history (context) followed by its future target.
        self.values = values.to(dtype=torch.float32)
        self.time_features = time_features
        self.context_length = context_length
        self.horizon = horizon
        # stride controls how far the next forecasting origin moves along the time axis.
        self.stride = stride
        self.target_indices = tuple(target_indices or range(values.shape[1]))
        self.feature_names = tuple(
            feature_names or (f"feature_{index}" for index in range(values.shape[1]))
        )
        if len(self.feature_names) != values.shape[1]:
            raise ValueError("feature_names must have one name per input feature")

        if not self.target_indices:
            raise ValueError("target_indices must not be empty")
        if min(self.target_indices) < 0 or max(self.target_indices) >= values.shape[1]:
            raise ValueError("target_indices contain an out-of-range feature index")
        if target_names is None:
            self.target_names = tuple(f"target_{index}" for index in self.target_indices)
        else:
            self.target_names = tuple(target_names)
            if len(self.target_names) != len(self.target_indices):
                raise ValueError("target_names must have the same length as target_indices")

    @property
    def num_features(self) -> int:
        time_channels = 0 if self.time_features is None else self.time_features.shape[1]
        return self.values.shape[1] + time_channels

    @property
    def num_targets(self) -> int:
        return len(self.target_indices)

    def __len__(self) -> int:
        # The final sample must still contain a complete future horizon.
        return (self.values.shape[0] - self.context_length - self.horizon) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        # context is visible history; target is the ground truth revealed later.
        start = index * self.stride
        split = start + self.context_length
        stop = split + self.horizon
        context = self.values[start:split]
        if self.time_features is not None:
            context = torch.cat((context, self.time_features[start:split]), dim=1)
        target = self.values[split:stop, list(self.target_indices)]
        return context, target

    def standardize(self, train_windows: int) -> None:
        """Fit numeric channel statistics on the offline training rows only."""

        if train_windows == 0:
            return
        train_end = (train_windows - 1) * self.stride + self.context_length + self.horizon
        training = self.values[:train_end]
        mean = training.mean(dim=0)
        std = training.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.values = (self.values - mean) / std

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        context_length: int,
        horizon: int,
        *,
        time_column: str = "date",
        target_columns: Sequence[str] | None = None,
        feature_columns: Sequence[str] | None = None,
        include_time_features: bool = False,
        stride: int = 1,
    ) -> "SlidingWindowDataset":
        """Load numeric columns from a benchmark-style CSV file.

        By default all columns other than ``time_column`` are input features.
        If ``target_columns`` is omitted, all selected features are forecast.
        """

        source = Path(path)
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"CSV file has no header: {source}")

            # The time column is metadata; selected numeric columns are model features.
            columns = list(feature_columns or [name for name in reader.fieldnames if name != time_column])
            if not columns:
                raise ValueError("no numeric feature columns were selected")
            missing = set(columns).difference(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV columns not found: {sorted(missing)}")

            rows: list[list[float]] = []
            dates: list[datetime] = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    rows.append([float(row[column]) for column in columns])
                    if include_time_features:
                        dates.append(datetime.fromisoformat(row[time_column]))
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid value at CSV row {row_number}") from error

        if not rows:
            raise ValueError(f"CSV file has no data rows: {source}")

        # With no explicit target columns, every input feature is forecast.
        selected_targets = tuple(target_columns or columns)
        missing_targets = set(selected_targets).difference(columns)
        if missing_targets:
            raise ValueError(f"target columns are not selected features: {sorted(missing_targets)}")
        target_indices = [columns.index(column) for column in selected_targets]
        time_features = None
        if include_time_features:
            include_minute = any(stamp.minute for stamp in dates)
            time_features = torch.tensor(
                [
                    [
                        (stamp.month - 1) / 11 - 0.5,
                        (stamp.day - 1) / 30 - 0.5,
                        stamp.weekday() / 6 - 0.5,
                        stamp.hour / 23 - 0.5,
                    ] + ([stamp.minute / 59 - 0.5] if include_minute else [])
                    for stamp in dates
                ],
                dtype=torch.float32,
            )
        return cls(
            torch.tensor(rows, dtype=torch.float32),
            context_length=context_length,
            horizon=horizon,
            target_indices=target_indices,
            target_names=selected_targets,
            feature_names=columns,
            time_features=time_features,
            stride=stride,
        )


def load_benchmark_dataset(
    name: str,
    path: str | Path,
    context_length: int,
    horizon: int,
    *,
    stride: int = 1,
    time_column: str | None = None,
    target_columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    include_time_features: bool = False,
) -> SlidingWindowDataset:
    """Load datasets."""

    normalized_name = name.lower()
    if normalized_name not in BENCHMARK_SPECS:
        supported = ", ".join(sorted(BENCHMARK_SPECS))
        raise ValueError(f"unsupported dataset {name!r}; choose one of: {supported}")

    spec = BENCHMARK_SPECS[normalized_name]
    return SlidingWindowDataset.from_csv(
        path,
        context_length=context_length,
        horizon=horizon,
        time_column=spec.time_column if time_column is None else time_column,
        target_columns=spec.target_columns if target_columns is None else target_columns,
        feature_columns=feature_columns,
        include_time_features=(
            include_time_features
            and (spec.time_column if time_column is None else time_column) == "date"
        ),
        stride=stride,
    )
