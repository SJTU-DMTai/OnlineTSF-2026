#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_ROOT="${1:-runs/ettm1_dsof_beam_compare_20261121_155045}"
OUTPUT="${2:-runs/ettm1_dsof_greedy_aligned_$(date +%Y%m%d_%H%M%S)}"

exec python -u scripts/evaluate_greedy_aligned.py \
    --run-root "$RUN_ROOT" --output "$OUTPUT" --device cuda:0
