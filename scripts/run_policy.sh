#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_ROOT="${1:?Usage: bash scripts/run_policy.sh runs/unique-output-directory}"
WORKERS="${WORKERS:-2}"
# Optional ACC diagnostics: none, policy_full, policy_data, or both.
ORACLE_DIAGNOSTICS="${ORACLE_DIAGNOSTICS:-none}"
DECISION_CONFIG="${DECISION_CONFIG:-decision.yaml}"
ETTH2_ANNOTATIONS="${ETTH2_ANNOTATIONS:-data/drift-etth}"
ETTM1_ANNOTATIONS="${ETTM1_ANNOTATIONS:-data/drift-ettm}"
WEATHER_ANNOTATIONS="${WEATHER_ANNOTATIONS:-data/drift-weather}"
mkdir -p "$RUN_ROOT/logs"

run_pipeline() {
    local gpu="$1" dataset="$2" method="$3" positions="$4" annotations="$5"
    local job="$RUN_ROOT/$dataset/$method"
    local log="$RUN_ROOT/logs/$dataset-$method.log"
    echo "START GPU=$gpu dataset=$dataset method=$method log=$log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m onlinetsf.decision \
        --annotations "$annotations" --dataset "$dataset" --method "$method" \
        --horizon 24 --seed 0 --profiles profiles.yaml --context-length 96 \
        --epochs 6 --batch-size 32 --positions "$positions" --workers "$WORKERS" \
        --decision-config "$DECISION_CONFIG" --test-start-fraction 0.6 \
        --oracle-diagnostics "$ORACLE_DIAGNOSTICS" --oracle-metric mae \
        --device cuda:0 --threads 1 --output "$job" > "$log" 2>&1 &
    local pid=$!
    printf '%s\t%s\t%s\t%s\n' "$dataset" "$method" "$gpu" "$pid" >> "$RUN_ROOT/pids.tsv"
    if wait "$pid"; then
        echo "DONE GPU=$gpu dataset=$dataset method=$method"
    else
        echo "FAILED GPU=$gpu dataset=$dataset method=$method; inspect $log" >&2
        return 1
    fi
}

wait_batch() {
    local failed=0 pid
    for pid in "$@"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if (( failed )); then
        echo "Batch failed; remaining batches will not start. Inspect pipeline logs." >&2
        return 1
    fi
}

printf 'dataset\tmethod\tgpu\tpid\n' > "$RUN_ROOT/pids.tsv"

# In the first batch, GPU 1 runs two queues and GPU 3 runs one queue.
echo "Batch 1/2: OneNet and DSOF; two pipelines on GPU 1, one on GPU 3."
(
    run_pipeline 1 ETTh2 onenet 48 "$ETTH2_ANNOTATIONS"
    run_pipeline 1 ETTm1 onenet 64 "$ETTM1_ANNOTATIONS"
) &
gpu1_onenet_pid=$!
(
    run_pipeline 1 ETTh2 dsof 48 "$ETTH2_ANNOTATIONS"
    run_pipeline 1 ETTm1 dsof 64 "$ETTM1_ANNOTATIONS"
) &
gpu1_dsof_pid=$!
(
    run_pipeline 3 Weather dsof 48 "$WEATHER_ANNOTATIONS"
    run_pipeline 3 Weather onenet 48 "$WEATHER_ANNOTATIONS"
) &
gpu3_pid=$!
wait_batch "$gpu1_onenet_pid" "$gpu1_dsof_pid" "$gpu3_pid"

# Run the slower FSNet experiments only after OneNet and DSOF finish.
echo "Batch 2/2: FSNet; ETTh2 on GPU 1, ETTm1 then Weather on GPU 3."
run_pipeline 1 ETTh2 fsnet 48 "$ETTH2_ANNOTATIONS" &
gpu1_fsnet_pid=$!
(
    run_pipeline 3 ETTm1 fsnet 64 "$ETTM1_ANNOTATIONS"
    run_pipeline 3 Weather fsnet 48 "$WEATHER_ANNOTATIONS"
) &
gpu3_fsnet_pid=$!
wait_batch "$gpu1_fsnet_pid" "$gpu3_fsnet_pid"

echo "All nine pipelines finished successfully; results=$RUN_ROOT"
