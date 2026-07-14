#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

# Compile cpp subsampling
cd "$SCRIPT_DIR/cpp_subsampling"
"$PYTHON_BIN" setup.py build_ext --inplace

# Compile cpp neighbors
cd "$SCRIPT_DIR/cpp_neighbors"
"$PYTHON_BIN" setup.py build_ext --inplace
