#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

scenes=(bagpack_laptop_cup colema_statue bear_and_girl_statue)
prompts=(cup statue statue)

for i in "${!scenes[@]}"; do
    echo "[Info] Running scene: ${scenes[i]}"
    python "${REPO_ROOT}/test.py" --dataset gs25 \
        --scene_name "${scenes[i]}" --mask_prompt "${prompts[i]}" \
        --negative_prompt "${prompts[i]}" --save_gs "$@"
done
