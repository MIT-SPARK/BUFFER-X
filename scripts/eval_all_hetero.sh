#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <experiment_id> [cuda_visible_devices] [additional test.py args...]"
  echo "Example: $0 threedmatch 0 --pose_estimator ransac --verbose"
  exit 1
fi

EXPERIMENT_ID="$1"
shift || true

CUDA_VISIBLE_DEVICES_VALUE="0"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  CUDA_VISIBLE_DEVICES_VALUE="$1"
  shift || true
fi
EXTRA_ARGS=("$@")

HETERO_PAIRS=(
  "KAIST_hetero:Aeva->Avia"
  "KAIST_hetero:Avia->Ouster"
  "KAIST_hetero:Ouster->Aeva"
  "TIERS_hetero:os0_128->os1_64"
  "TIERS_hetero:os1_64->vel16"
  "TIERS_hetero:vel16->os0_128"
)

echo "============================================================"
echo "Starting hetero evaluation across configured sensor directions"
echo "============================================================"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE:-0}" python test.py \
  --dataset KAIST_hetero TIERS_hetero \
  --experiment_id "${EXPERIMENT_ID}" \
  --hetero_pairs "${HETERO_PAIRS[@]}" \
  --verbose \
  "${EXTRA_ARGS[@]}"
