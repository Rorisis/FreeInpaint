#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
scenes=(carton cone cookie newcone plant skateboard sunflower)
for scene in "${scenes[@]}"; do
    echo "[Info] Running scene: ${scene}"
    python "${REPO_ROOT}/test.py" --dataset usid \
        --scene_name "$scene" --negative_prompt "$scene" "$@"
done
