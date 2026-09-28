#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f onlinetsf/bin/activate ]]; then
  echo "Missing virtual environment: $ROOT_DIR/onlinetsf/bin/activate" >&2
  exit 1
fi
source onlinetsf/bin/activate

if [[ ! -d data/labeled_drift ]]; then
  echo "Missing labeled data: $ROOT_DIR/data/labeled_drift" >&2
  exit 1
fi

if [[ ! -d data/labeled_drift_sisc_sampled ]]; then
  python scripts/sample_sisc_labeled_drift.py \
    --per-type 2 --seed 2026 \
    --output data/labeled_drift_sisc_sampled
fi

for group in mendeley_liu sisc; do
  for suffix in a b c d_frequency_lr d_rounds; do
    if [[ -e "runs/${group}_${suffix}" ]]; then
      echo "Output already exists: runs/${group}_${suffix}" >&2
      exit 1
    fi
  done
done

mkdir -p runs/nohup_logs

launch_experiment() {
  local gpu="$1"
  local experiment="$2"
  local pid

  CUDA_VISIBLE_DEVICES="$gpu" nohup bash -c '
    cd "$1"
    source onlinetsf/bin/activate
    set -euo pipefail
    experiment="$2"

    for group in mendeley_liu sisc; do
      if [[ "$group" == mendeley_liu ]]; then
        common=(
          --data-root data/labeled_drift
          --families "mendeley/*" "liu2023/*"
          --max-per-family 2
          --selection-seed 2026
          --seeds 0
          --horizons 1 24
          --device cuda
        )
      else
        common=(
          --data-root data/labeled_drift_sisc_sampled
          --max-per-family 0
          --seeds 0
          --horizons 1 24
          --device cuda
        )
      fi

      case "$experiment" in
        A)
          python scripts/experiment_a_profile.py \
            "${common[@]}" --output "runs/${group}_a"
          ;;
        B)
          python scripts/experiment_b_update_benefit.py \
            "${common[@]}" --output "runs/${group}_b" \
            --probes-per-phase 4 --lookaheads 1 4 8 16 32
          ;;
        C)
          python scripts/experiment_c_update_timing.py \
            "${common[@]}" --output "runs/${group}_c" \
            --budget-fraction 0.25 --oracle-delays 0 4 8 32 64
          ;;
        D)
          python scripts/experiment_d_update_frequency.py \
            "${common[@]}" --output "runs/${group}_d_frequency_lr" \
            --periods 1 2 4 8 16 --rounds 1 --lr-scales 0.3 1 3

          python scripts/experiment_d_update_frequency.py \
            "${common[@]}" --output "runs/${group}_d_rounds" \
            --periods 4 --rounds 1 2 4 --lr-scales 1
          ;;
      esac
    done
  ' bash "$ROOT_DIR" "$experiment" \
    > "runs/nohup_logs/${experiment}.log" 2>&1 < /dev/null &

  pid=$!
  printf '%s\n' "$pid" > "runs/nohup_logs/${experiment}.pid"
  echo "Experiment $experiment: GPU $gpu, PID $pid, log runs/nohup_logs/${experiment}.log"
}

launch_experiment 0 A
launch_experiment 1 B
launch_experiment 2 C
launch_experiment 3 D
