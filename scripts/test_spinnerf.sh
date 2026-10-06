#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
scenes=(1 2 3 4 7 9 10 12 book trash)
for scene in "${scenes[@]}"; do
    echo "[Info] Running scene: ${scene}"
    prompt=""
    [[ "$scene" == "book" || "$scene" == "trash" ]] && prompt="$scene"
    python "${REPO_ROOT}/test.py" --dataset spinnerf \
        --scene_name "$scene" --negative_prompt "$prompt" "$@"
done
