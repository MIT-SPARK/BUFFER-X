#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <experiment_id>"
  exit 1
fi

EXPERIMENT_ID="$1"

DATASETS=(
  "3DMatch"
  "3DLoMatch"
  "Scannetpp_iphone"
  "Scannetpp_faro"
  "Tiers"
  "KITTI"
  "WOD"
  "MIT"
  "KAIST"
  "ETH"
  "Oxford"
)

for DATASET in "${DATASETS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python test.py --dataset "$DATASET" --experiment_id "$EXPERIMENT_ID"
done