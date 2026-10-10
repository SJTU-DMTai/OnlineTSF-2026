#!/usr/bin/env bash
# Launch independent method/horizon shards on a remote Linux server.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_ROOT"
RUN_ROOT="${1:-runs/update-study-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p -- "$(dirname -- "$RUN_ROOT")"
mkdir -- "$RUN_ROOT"
mkdir -- "$RUN_ROOT/logs"

COMMON=(--annotations runs/real-drift-electricity-5pct-20261010
        --datasets ETTh2 ETTm1 Electricity --scope both --suite both
        --seeds 0 1 2 --device cuda --threads 1)

nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" -u -m onlinetsf.update_study.run "${COMMON[@]}" \
    --methods fsnet --horizons 1 --output "$RUN_ROOT/gpu0" \
    > "$RUN_ROOT/logs/gpu0.log" 2>&1 &
echo "$!" > "$RUN_ROOT/gpu0.pid"

nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" -u -m onlinetsf.update_study.run "${COMMON[@]}" \
    --methods fsnet --horizons 24 --output "$RUN_ROOT/gpu1" \
    > "$RUN_ROOT/logs/gpu1.log" 2>&1 &
echo "$!" > "$RUN_ROOT/gpu1.pid"

nohup env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" -u -m onlinetsf.update_study.run "${COMMON[@]}" \
    --methods onenet --horizons 1 --output "$RUN_ROOT/gpu2" \
    > "$RUN_ROOT/logs/gpu2.log" 2>&1 &
echo "$!" > "$RUN_ROOT/gpu2.pid"

nohup env CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON_BIN" -u -m onlinetsf.update_study.run "${COMMON[@]}" \
    --methods onenet --horizons 24 --output "$RUN_ROOT/gpu3" \
    > "$RUN_ROOT/logs/gpu3.log" 2>&1 &
echo "$!" > "$RUN_ROOT/gpu3.pid"

printf 'Started four background jobs. Output: %s\n' "$RUN_ROOT"
printf 'After all jobs complete, merge with:\nPYTHONPATH=src %q -m onlinetsf.update_study.analyze %q\n' "$PYTHON_BIN" "$RUN_ROOT"
