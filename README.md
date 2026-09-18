<!-- -*- coding: utf-8 -*- -->
# OnlineTSF-2026

## Included

- Backbones: linear, LSTM, TCN, PatchTST, FSNet-TCN, and OneNet-TCN.
- Methods: OGD, FSNet, and OneNet online ensembling.
- Drift detectors: Page-Hinkley, ADWIN, and KSWIN.
- Prequential evaluation with delayed feedback.

## Data

Supported datasets are ETTh1, ETTh2, and Traffic. ETTh files need `date` and
`OT` columns. Traffic files need `date` and numeric sensor columns.

## Run

Install the project:

```powershell
python -m pip install -e .[dev]
```

Set `data.path` and select a strategy and detector in `config.yaml`, then run:

```powershell
python -m onlinetsf --config config.yaml
```

`config.yaml` selects complete profiles from `profiles.yaml`. Choose one of
`linear_ogd`, `lstm_ogd`, `tcn_ogd`, `patchtst_ogd`, `fsnet`, or `onenet` as the
strategy, and one of `none`, `page_hinkley`, `adwin`, or `kswin` as the detector.
Edit the selected profile when its parameters need to change. FSNet is paired with
`fsnet_tcn`, and its default profile uses the official 10x64 + 320 same-padded
encoder. OneNet is paired with `onenet_tcn` by its strategy profile. OneNet-TCN
combines a variable-independent Time-TCN with a cross-variable TCN. It learns
per-target long-term weights plus a short-term decision-network correction; both
experts continue adapting when delayed labels arrive. `n_inner` controls updates per
feedback event, and `decision_dropout` configures the decision MLP. Combination
weights start from an equal mixture when online evaluation begins, matching the
official test loop. The `offline` section trains on an initial prefix of sliding
windows before online evaluation. For example, `train_ratio: 0.25` trains on the
first 25% of windows and starts prequential forecasting from the remaining 75%;
`epochs` and `batch_size` control that phase. Set `train_ratio: 0` for an
online-only experiment.

Each run creates a new timestamped directory under `output.directory` with:

```text
config.txt
online_training.log
results.txt
```

Each run also writes `forecast_values.csv`, with one row per observed
horizon/target value, and `drift_trace.csv`, with every detector update. Drift
detectors use `drift.source` (`features`, `target`, or `residual`) and maintain
one independent detector for each input feature or target. The default source is
`features`; MAE/MSE are evaluation metrics only and are not detector inputs.
`target` consumes each target time point once, while `residual` keeps every
forecast error because forecasts from different origins are distinct observations.

## Batch run

Run multiple strategies, seeds, and detectors without repeating a forecasting
replay for every detector:

```powershell
python -m onlinetsf.batch --config config.yaml --strategies linear_ogd tcn_ogd fsnet --detectors page_hinkley adwin kswin --seeds 0 1 2
```

The timestamped batch directory contains `forecast_steps.csv`,
`forecast_values.csv`, `drift_trace.csv`, `drift_events.csv`, and `summary.csv`.
It also retains one complete run-document directory for every strategy and seed.

## Test

```powershell
python -m pytest
```
