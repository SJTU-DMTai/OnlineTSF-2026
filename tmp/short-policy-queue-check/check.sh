#!/usr/bin/env bash
set -euo pipefail
python() {
    if [[ "$*" == *collect_policy_training.py* ]]; then
        if [[ "$*" == *ETTm1* ]]; then sleep 1; else sleep 0.1; fi
    fi
    return 0
}
export -f python
GPU=0 MAX_JOBS=3 bash scripts/run_short_policy.sh tmp/short-policy-queue-check/source tmp/short-policy-queue-check/result-varied > tmp/short-policy-queue-check/events-varied.log
