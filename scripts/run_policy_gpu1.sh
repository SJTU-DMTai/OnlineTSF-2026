#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_ROOT="${1:?Usage: bash scripts/run_policy_gpu1.sh runs/unique-output-directory}"
WORKERS="${WORKERS:-2}"
ETTM1_ANNOTATIONS="${ETTM1_ANNOTATIONS:-data/drift-ettm}"
WEATHER_ANNOTATIONS="${WEATHER_ANNOTATIONS:-data/drift-weather}"
mkdir -p "$RUN_ROOT/logs"

run_pipeline() {
    local dataset="$1" method="$2" positions="$3" annotations="$4"
    local job="$RUN_ROOT/$dataset/$method"
    mkdir -p "$job"

    echo "[$dataset/$method] 1/4: offline training and initial replay"
    # One initial diagnostic probe suffices; stage 2 collects all training positions.
    python -u -m onlinetsf.update_study.run \
        --annotations "$annotations" --datasets "$dataset" --methods "$method" \
        --horizons 24 --seeds 0 --profiles profiles.yaml --context-length 96 \
        --epochs 6 --batch-size 32 --scope branches --suite both \
        --max-probes 1 --branch-steps 128 --lookaheads 128 \
        --lr-scales 0.3 1 3 --device cuda:0 --threads 1 \
        --output "$job/source" > "$job/01-source.log" 2>&1

    echo "[$dataset/$method] 2/4: collect $positions positions x 17 actions x 128 origins"
    python -u scripts/collect_policy_training.py \
        --runs "$job/source" --annotations "$annotations" \
        --dataset "$dataset" --method "$method" --horizon 24 --seed 0 \
        --test-start-fraction 0.6 --positions "$positions" --decision-interval 128 \
        --workers "$WORKERS" --device cuda:0 --threads 1 \
        --output "$job/training" > "$job/02-collection.log" 2>&1

    echo "[$dataset/$method] 3/4: selector fitting and full test interval"
    python -u scripts/experiment_policy_2.py \
        --runs "$job/source" --annotations "$annotations" \
        --dataset "$dataset" --method "$method" --horizon 24 \
        --train-seeds 0 --test-seed 0 --training-data "$job/training" \
        --test-start-fraction 0.6 --decision-interval 128 \
        --max-depth 2 --min-samples-leaf 4 --oracle-diagnostics policy_full --oracle-metric mae \
        --device cuda:0 --threads 1 \
        --output "$job/experiment" > "$job/03-experiment.log" 2>&1

    echo "[$dataset/$method] 4/4: frozen/native/automatic/oracle comparison and costs"
    python -u scripts/evaluate_policy_comparison.py \
        --runs "$job/source" --annotations "$annotations" \
        --experiment "$job/experiment" --controller policy_full --metric mae \
        --device cuda:0 --threads 1 \
        --output "$job/comparison" > "$job/04-comparison.log" 2>&1
    echo "[$dataset/$method] completed: $job/comparison/comparison.csv"
}

pids=()
printf 'dataset\tmethod\tpid\n' > "$RUN_ROOT/pids.tsv"
for method in onenet fsnet dsof; do
    run_pipeline ETTm1 "$method" 64 "$ETTM1_ANNOTATIONS" \
        > "$RUN_ROOT/logs/ETTm1-$method.log" 2>&1 &
    pids+=("$!")
    printf 'ETTm1\t%s\t%s\n' "$method" "$!" >> "$RUN_ROOT/pids.tsv"
done

for method in onenet fsnet dsof; do
    run_pipeline Weather "$method" 48 "$WEATHER_ANNOTATIONS" \
        > "$RUN_ROOT/logs/Weather-$method.log" 2>&1 &
    pids+=("$!")
    printf 'Weather\t%s\t%s\n' "$method" "$!" >> "$RUN_ROOT/pids.tsv"
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "Pipeline PID=$pid failed; inspect its pipeline and stage logs." >&2
        failed=1
    fi
done
echo "All launched pipelines finished. Exit status=$failed; results=$RUN_ROOT"
exit "$failed"
