#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
dataset="${1:?Usage: bash scripts/evaluate.sh spinnerf|usid RESULTS_ROOT [metric options]}"
results="${2:?Provide the results directory}"
shift 2
python "${REPO_ROOT}/metrics/evaluate.py" \
    --dataset "$dataset" --results_root "$results" "$@"
