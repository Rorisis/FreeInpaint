#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

categories=(bottle cup bench)
sequences=(575_84616_167245 14_202_1205 106_12654_23221)

for i in "${!categories[@]}"; do
    echo "[Info] Running scene: ${categories[i]}/${sequences[i]}"
    python "${REPO_ROOT}/test.py" --dataset co3d \
        --category "${categories[i]}" --sequence_name "${sequences[i]}" \
        --negative_prompt "${categories[i]}" --save_gs "$@"
done
