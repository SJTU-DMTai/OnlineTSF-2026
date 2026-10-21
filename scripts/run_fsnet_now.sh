#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_ROOT="${1:?Usage: bash scripts/run_fsnet_now.sh runs/existing-policy-directory}"
WORKERS="${WORKERS:-2}"
ORACLE_DIAGNOSTICS="${ORACLE_DIAGNOSTICS:-none}"
DECISION_CONFIG="${DECISION_CONFIG:-decision.yaml}"
ETTH2_ANNOTATIONS="${ETTH2_ANNOTATIONS:-data/drift-etth}"
ETTM1_ANNOTATIONS="${ETTM1_ANNOTATIONS:-data/drift-ettm}"
WEATHER_ANNOTATIONS="${WEATHER_ANNOTATIONS:-data/drift-weather}"

[[ -d "$RUN_ROOT" ]] || { echo "Existing run directory required: $RUN_ROOT" >&2; exit 1; }
for dataset in ETTh2 ETTm1 Weather; do
    if [[ -e "$RUN_ROOT/$dataset/fsnet" || -L "$RUN_ROOT/$dataset/fsnet" ]]; then
        echo "FSNet output already reserved or started: $RUN_ROOT/$dataset/fsnet" >&2
        exit 1
    fi
done

mkdir "$RUN_ROOT/fsnet-now"
FSNET_ROOT="$(cd "$RUN_ROOT/fsnet-now" && pwd)"
mkdir "$FSNET_ROOT/logs"
# Reserve every old output before launching anything, including the queued Weather job.
# The old Python entry rejects these directories at mkdir(exist_ok=False), before training.
for dataset in ETTh2 ETTm1 Weather; do
    mkdir -p "$RUN_ROOT/$dataset"
    mkdir "$RUN_ROOT/$dataset/fsnet"
    printf 'Reserved by run_fsnet_now.sh to prevent duplicate training.\nActual results: %s/%s/fsnet\n' \
        "$FSNET_ROOT" "$dataset" > "$RUN_ROOT/$dataset/fsnet/RESULT_LOCATION.txt"
done
echo "Reserved old FSNet outputs. The old scheduler will report FileExistsError instead of retraining."
printf 'dataset\tmethod\tgpu\tpid\n' > "$FSNET_ROOT/pids.tsv"

run_fsnet() {
    local gpu="$1" dataset="$2" positions="$3" annotations="$4"
    local log="$FSNET_ROOT/logs/$dataset-fsnet.log"
    echo "START GPU=$gpu dataset=$dataset method=fsnet log=$log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m onlinetsf.decision \
        --annotations "$annotations" --dataset "$dataset" --method fsnet \
        --horizon 24 --seed 0 --profiles profiles.yaml --context-length 96 \
        --epochs 6 --batch-size 32 --positions "$positions" --workers "$WORKERS" \
        --decision-config "$DECISION_CONFIG" --test-start-fraction 0.6 \
        --oracle-diagnostics "$ORACLE_DIAGNOSTICS" --oracle-metric mae \
        --device cuda:0 --threads 1 --output "$FSNET_ROOT/$dataset/fsnet" > "$log" 2>&1 &
    local pid=$!
    printf '%s\tfsnet\t%s\t%s\n' "$dataset" "$gpu" "$pid" >> "$FSNET_ROOT/pids.tsv"
    if wait "$pid"; then
        echo "DONE GPU=$gpu dataset=$dataset method=fsnet"
    else
        echo "FAILED GPU=$gpu dataset=$dataset method=fsnet; inspect $log" >&2
        return 1
    fi
}

run_fsnet 1 ETTh2 48 "$ETTH2_ANNOTATIONS" &
gpu1_pid=$!
(
    run_fsnet 3 ETTm1 64 "$ETTM1_ANNOTATIONS"
    run_fsnet 3 Weather 48 "$WEATHER_ANNOTATIONS"
) &
gpu3_pid=$!

failed=0
for pid in "$gpu1_pid" "$gpu3_pid"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
echo "FSNet queues finished. Exit status=$failed; results=$FSNET_ROOT"
exit "$failed"
