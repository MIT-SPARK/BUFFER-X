#!/bin/bash
set -e  # If any command fails, exit immediately

echo "Step 1: Running decompose_aligned_pointcloud.py..."
python faro/decompose_aligned_pointcloud.py
if [ $? -ne 0 ]; then
  echo "❌ Error: decompose_aligned_pointcloud.py failed"
  exit 1
fi
echo "✅ decompose_aligned_pointcloud.py completed successfully."


echo "Step 2: Running pair_gen_faro.py..."
python faro/pair_gen_faro.py
if [ $? -ne 0 ]; then
  echo "❌ Error: pair_gen_faro.py failed"
  exit 1
fi
echo "✅ pair_gen_faro.py completed successfully."


echo "🎉 All FARO preprocessing steps completed successfully."
