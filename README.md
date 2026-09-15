<!-- -*- coding: utf-8 -*- -->
# OnlineTSF-2026

## Included

- Backbones: linear, TCN, PatchTST, and FSNet-TCN.
- Methods: OGD and FSNet.
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

Set `data.path` in `config.yaml`, choose the backbone, method, and detector,
then run:

```powershell
python -m onlinetsf --config config.yaml
```

`config.yaml` supports `linear`, `tcn`, `patchtst`, and `fsnet_tcn`; `ogd` and
`fsnet`; and `none`, `page_hinkley`, `adwin`, and `kswin`. FSNet must be paired
with `fsnet_tcn`.

Each run creates a new timestamped directory under `output.directory` with:

```text
config.txt
online_training.log
results.txt
```

## Test

```powershell
python -m pytest
```
