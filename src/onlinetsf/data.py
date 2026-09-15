# -*- coding: utf-8 -*-
"""CSV loading and rolling-window datasets for benchmark time series."""

from __future__ import annotations

import csv
from dataclasses import dataclass
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
    "etth1": DatasetSpec(time_column="date", target_columns=("OT",)),
    "etth2": DatasetSpec(time_column="date", target_columns=("OT",)),
    "traffic": DatasetSpec(time_column="date", target_columns=None),
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
        self.context_length = context_length
        self.horizon = horizon
        # stride controls how far the next forecasting origin moves along the time axis.
        self.stride = stride
        self.target_indices = tuple(target_indices or range(values.shape[1]))

        if not self.target_indices:
            raise ValueError("target_indices must not be empty")
        if min(self.target_indices) < 0 or max(self.target_indices) >= values.shape[1]:
            raise ValueError("target_indices contain an out-of-range feature index")

    @property
    def num_features(self) -> int:
        return self.values.shape[1]

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
        target = self.values[split:stop, list(self.target_indices)]
        return context, target

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
            for row_number, row in enumerate(reader, start=2):
                try:
                    rows.append([float(row[column]) for column in columns])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"non-numeric value at CSV row {row_number}") from error

        if not rows:
            raise ValueError(f"CSV file has no data rows: {source}")

        # With no explicit target columns, every input feature is forecast.
        selected_targets = tuple(target_columns or columns)
        missing_targets = set(selected_targets).difference(columns)
        if missing_targets:
            raise ValueError(f"target columns are not selected features: {sorted(missing_targets)}")
        target_indices = [columns.index(column) for column in selected_targets]
        return cls(
            torch.tensor(rows, dtype=torch.float32),
            context_length=context_length,
            horizon=horizon,
            target_indices=target_indices,
            stride=stride,
        )


def load_benchmark_dataset(
    name: str,
    path: str | Path,
    context_length: int,
    horizon: int,
    *,
    stride: int = 1,
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
        time_column=spec.time_column,
        target_columns=spec.target_columns,
        stride=stride,
    )
