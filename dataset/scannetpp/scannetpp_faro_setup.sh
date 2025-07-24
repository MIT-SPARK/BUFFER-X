#!/bin/bash

python decompose_aligned_pointcloud.py
if [ $? -ne 0 ]; then
  echo "Error: decompose_aligned_pointcloud.py failed"
  exit 1
fi

python pair_gen_faro.py
if [ $? -ne 0 ]; then
  echo "Error: pair_gen_faro.py failed"
  exit 1
fi

echo "Both scripts ran successfully."
