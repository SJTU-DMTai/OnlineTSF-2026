#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

export RUN_ROOT="runs/policy-20261021-182721"
export OUT_ROOT="$RUN_ROOT/diagnostics-$(date +%Y%m%d-%H%M%S)"
export CONTROLLER="${CONTROLLER:-policy_full}"

mkdir -p "$OUT_ROOT/logs"
echo "Results: $OUT_ROOT"

for batch in main fsnet; do
  if [[ "$batch" == main ]]; then
    jobs=(
      "ETTm1 onenet"
      "Weather onenet"
      "ETTm1 dsof"
      "Weather dsof"
      "ETTh2 onenet"
      "ETTh2 dsof"
    )
  else
    jobs=(
      "ETTm1 fsnet"
      "Weather fsnet"
      "ETTh2 fsnet"
    )
  fi

  echo "BATCH: $batch"

  printf '%s\n' "${jobs[@]}" |
    xargs -P 4 -n 2 --process-slot-var=GPU_SLOT bash -c '
      set -euo pipefail

      dataset="$1"
      method="$2"

      case "$GPU_SLOT" in
        0|2) gpu=1 ;;
        1|3) gpu=3 ;;
      esac

      case "$dataset" in
        ETTh2)   annotations="data/drift-etth" ;;
        ETTm1)   annotations="data/drift-ettm" ;;
        Weather) annotations="data/drift-weather" ;;
      esac

      echo "START GPU=$gpu $dataset/$method"

      CUDA_VISIBLE_DEVICES="$gpu" python -u -m onlinetsf.decision diagnose \
        --experiment "$RUN_ROOT/$dataset/$method/experiment" \
        --annotations "$annotations" \
        --controller "$CONTROLLER" \
        --metric mae \
        --device cuda:0 \
        --threads 1 \
        --output "$OUT_ROOT/$dataset/$method" \
        > "$OUT_ROOT/logs/$dataset-$method.log" 2>&1

      echo "DONE  GPU=$gpu $dataset/$method"
    ' _
done

python - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/*/*/oracle_summary.json")):
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "dataset": path.parents[2].name,
        "method": path.parents[1].name,
        "controller": path.parent.name,
        "decisions": summary["decisions"],
        "acc_pct": summary["acc_pct"],
        "within_1pct_pct": summary["within_1pct_pct"],
        "mean_relative_regret_pct": summary["mean_relative_regret_pct"],
        "median_relative_regret_pct": summary["median_relative_regret_pct"],
    })

frame = pd.DataFrame(rows)
frame.to_csv(root / "summary.csv", index=False)
print(frame.round(3).to_string(index=False))
print(f"\nSaved: {root / 'summary.csv'}")
PY