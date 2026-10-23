#!/usr/bin/env bash
# Usage: bash scripts/run_short_policy.sh OLD_RUN_ROOT NEW_RUN_ROOT
# OLD_RUN_ROOT has DATASET/METHOD/source/{manifest.json,DATASET/.../initial.pt}.
# Reuse the original offline models; collect 10%/20%, then test to the end.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SOURCE_ROOT="${1:?Usage: bash scripts/run_short_policy.sh OLD_RUN_ROOT NEW_RUN_ROOT}"
RUN_ROOT="${2:?Usage: bash scripts/run_short_policy.sh OLD_RUN_ROOT NEW_RUN_ROOT}"
WORKERS="${WORKERS:-2}"
MAX_JOBS="${MAX_JOBS:-3}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
DECISION_CONFIG="${DECISION_CONFIG:-decision.yaml}"
ETTH2_ANNOTATIONS="${ETTH2_ANNOTATIONS:-data/drift-etth}"
ETTM1_ANNOTATIONS="${ETTM1_ANNOTATIONS:-data/drift-ettm}"
ETTH2_POSITIONS="${ETTH2_POSITIONS:-128}"
ETTM1_POSITIONS="${ETTM1_POSITIONS:-256}"
mkdir -p "$RUN_ROOT/logs"

# Check all source directories before launching expensive collection jobs.
for dataset in ETTh2 ETTm1; do
    for method in onenet dsof; do
        if [[ ! -f "$SOURCE_ROOT/$dataset/$method/source/manifest.json" ]]; then
            echo "Missing source manifest: $SOURCE_ROOT/$dataset/$method/source/manifest.json" >&2
            exit 1
        fi
    done
done

run_job() {
    local fraction="$1" dataset="$2" method="$3" positions annotations source job log
    if [[ "$dataset" == ETTh2 ]]; then
        positions="$ETTH2_POSITIONS"
        annotations="$ETTH2_ANNOTATIONS"
    else
        positions="$ETTM1_POSITIONS"
        annotations="$ETTM1_ANNOTATIONS"
    fi
    source="$SOURCE_ROOT/$dataset/$method/source"
    job="$RUN_ROOT/test-start-$fraction/$dataset/$method"
    log="$RUN_ROOT/logs/$dataset-$method-$fraction.log"
    mkdir -p "$(dirname "$job")"
    # Existing outputs are never overwritten or silently mixed.
    mkdir "$job"
    echo "START GPU=$GPU $dataset/$method test_start=$fraction positions=$positions log=$log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/collect_policy_training.py \
        --runs "$source" --annotations "$annotations" --dataset "$dataset" --method "$method" \
        --horizon 24 --seed "$SEED" --positions "$positions" --workers "$WORKERS" \
        --decision-config "$DECISION_CONFIG" --test-start-fraction "$fraction" \
        --device cuda:0 --threads 1 --output "$job/training" > "$log" 2>&1
    CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/experiment_policy_2.py \
        --runs "$source" --annotations "$annotations" --dataset "$dataset" --method "$method" \
        --horizon 24 --train-seeds "$SEED" --test-seed "$SEED" \
        --training-data "$job/training" --test-start-fraction "$fraction" \
        --controllers policy_data policy_full frozen --oracle-diagnostics none \
        --device cuda:0 --threads 1 --output "$job/experiment" >> "$log" 2>&1
    echo "DONE GPU=$GPU $dataset/$method test_start=$fraction"
}

# xargs immediately fills each free slot; a slot covers collection and all three controllers.
export -f run_job
export SOURCE_ROOT RUN_ROOT WORKERS SEED DECISION_CONFIG GPU
export ETTH2_ANNOTATIONS ETTM1_ANNOTATIONS ETTH2_POSITIONS ETTM1_POSITIONS
if [[ ! "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer" >&2
    exit 1
fi
if ! (
    for fraction in 0.3 0.4; do
        for dataset in ETTh2 ETTm1; do
            for method in onenet dsof; do
                printf '%s %s %s\n' "$fraction" "$dataset" "$method"
            done
        done
    done
) | xargs -P "$MAX_JOBS" -n 3 bash -c 'set -euo pipefail; run_job "$@"' _; then
    echo "A task failed; inspect $RUN_ROOT/logs. No aggregate summary was produced." >&2
    exit 1
fi

python - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
frames = []
for path in sorted(root.glob("test-start-*/*/*/experiment/summary.csv")):
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise RuntimeError(f"Incomplete experiment: {path.parent}")
    frame = pd.read_csv(path)
    for name in ("dataset", "method", "test_start", "steps", "training_positions"):
        frame[name] = manifest[name]
    frame["test_start_fraction"] = manifest["arguments"]["test_start_fraction"]
    frame["experiment"] = str(path.parent)
    frames.append(frame)
summary = pd.concat(frames, ignore_index=True)
summary.to_csv(root / "summary.csv", index=False)
print(summary[["test_start_fraction", "dataset", "method", "controller", "training_positions",
               "mae", "mse", "gradient_steps", "online_seconds"]].to_string(index=False))
print(f"Saved: {root / 'summary.csv'}")
PY
