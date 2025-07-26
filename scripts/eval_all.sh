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
  "TIERS"
  "KITTI"
  "WOD"
  "MIT"
  "KAIST"
  "ETH"
  "Oxford"
)

DATASET_ARGS="${DATASETS[@]}"

CUDA_VISIBLE_DEVICES=0 python test.py --dataset $DATASET_ARGS --experiment_id "$EXPERIMENT_ID" --verbose
