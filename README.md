<!-- -*- coding: utf-8 -*- -->
# OnlineTSF-2026

Minimal research baseline for online time-series forecasting.

## Current scope

- Rolling-window CSV loading for **ETTh1**, **ETTh2**, and **Traffic**.
- Three PyTorch forecasting backbones: linear, TCN, and PatchTST.
  - TCN uses causal dilated convolutions.
  - PatchTST uses channel-independent patches and a Transformer encoder.
- Three drift detectors: Page-Hinkley, ADWIN, and KSWIN.
- FSNet with a causal TCN, per-convolution gradient adapters, dual gradient EMAs,
  and sparse associative memory.

The package includes a prequential online executor with configurable delayed feedback.
Plain PyTorch models can use its optional OGD update path; methods such as FSNet own
their method-specific prediction and feedback updates behind a small lifecycle protocol.

Page-Hinkley and KSWIN accept any finite scalar residual statistic. ADWIN uses a Hoeffding bound and therefore requires that statistic to be normalized to `[0, 1]` first.

## Data layout

Download the benchmark CSV files separately. ETTh files are expected to contain a `date` column and `OT` target column. Traffic is expected to contain a `date` column plus numeric traffic-sensor columns.

## Run the baseline

```powershell
python -m pip install -e .[dev]
python examples/run_baseline.py --dataset etth1 --data-path path\to\ETTh1.csv --model tcn
python examples/run_baseline.py --dataset etth1 --data-path path\to\ETTh1.csv --model patchtst
```

The example uses the first 80% of timestamps for fitting, normalizes using only that partition, and reports Page-Hinkley events over held-out forecast MAE.

## Online execution

`OnlineExecutor` implements prequential forecasting: it records a forecast before the matching target is used for scoring or adaptation. Pass an optimizer and scalar loss to enable online gradient updates; omit both for a frozen forecaster.

```python
from torch import nn
from onlinetsf import OnlineExecutor

executor = OnlineExecutor(
    model,
    feedback_delay=horizon,
    optimizer=optimizer,
    loss_fn=nn.MSELoss(),
)
result = executor.run_dataset(dataset, start=online_start)
print(result.metrics.mae, result.metrics.mse)
```

`feedback_delay=0` gives immediate feedback, matching the protocol used in the FSNet paper. For a full `horizon`-step target from a stride-one sliding-window dataset, use `feedback_delay=horizon`; the entire target is then available before the forecast at `index + horizon`. The executor reports per-feedback MAE/MSE and cumulative prequential MAE/MSE, while adaptation is performed only after those metrics have been recorded. Its `state_dict()` preserves pending delayed feedback; save the model and optimizer state separately in the same checkpoint.

By default, feedback history stores scalar indicators rather than all forecasts and targets. Set `keep_predictions=True` when those tensors are required for analysis.

### FSNet

FSNet is kept in `onlinetsf.methods.fsnet`; the executor only calls `predict` and
`on_feedback` and does not depend on its adapter, EMA, trigger, or memory details.
The paper defaults are used for the two gradient EMA decays (`0.9`, `0.3`), memory
size (`32`), top-k retrieval (`2`), and interaction threshold (`0.75`).

```python
from onlinetsf import FSNetMethod, FSNetTCN, OnlineExecutor

model = FSNetTCN(
    context_length=60,
    num_features=dataset.num_features,
    horizon=1,
    num_targets=dataset.num_targets,
)
method = FSNetMethod(model, learning_rate=1e-3)
executor = OnlineExecutor(method=method, feedback_delay=0)
result = executor.run_dataset(dataset, start=online_start)
```

For delayed labels, change `feedback_delay`; FSNet updates only when the executor
delivers the corresponding feedback. Save `model.state_dict()` and
`method.optimizer.state_dict()` alongside `executor.state_dict()` to preserve the
model, FSNet EMA/memory buffers, optimizer, and pending feedback queue.

## Test

```powershell
python -m pytest
```
