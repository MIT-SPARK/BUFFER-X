#!/usr/bin/env bash
set -euo pipefail

CUDA="cu124"
WITH_APT="auto"
SKIP_TORCH=0
SKIP_CUDA_EXTENSIONS=0
WITH_HUB=0
WITH_KISS=0
WITH_SCANNETPP=0
WITH_DEV=0
PYTHON_BIN="${PYTHON:-python3}"
TMP_DIR=""

cleanup_tmp_dir() {
  if [[ -n "${TMP_DIR:-}" && -d "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
}

trap cleanup_tmp_dir EXIT

usage() {
  cat <<'USAGE'
Usage: ./scripts/install.sh [options]

Options:
  --cuda cu124|cu118|cu111|cpu  PyTorch wheel channel to install (default: cu124)
  --python PATH                 Python executable to use (default: $PYTHON or python3)
  --no-apt                      Skip apt package installation
  --skip-torch                  Do not install torch/torchvision/torchaudio
  --skip-cuda-extensions        Skip pointnet2_ops, KNN_CUDA, cpp wrappers, torch-batch-svd
  --with-hub                    Install Hugging Face upload/download dependencies
  --with-kiss                   Install optional KISS-Matcher Python package
  --with-scannetpp              Install ScanNet++ preprocessing dependencies
  --dev                         Install developer tooling
  -h, --help                    Show this help message

Examples:
  ./scripts/install.sh --cuda cu124 --with-hub
  ./scripts/install.sh --cuda cu118 --with-kiss
  ./scripts/install.sh --cuda cpu --skip-cuda-extensions
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      CUDA="${2:?Missing value for --cuda}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?Missing value for --python}"
      shift 2
      ;;
    --no-apt)
      WITH_APT=0
      shift
      ;;
    --skip-torch)
      SKIP_TORCH=1
      shift
      ;;
    --skip-cuda-extensions)
      SKIP_CUDA_EXTENSIONS=1
      shift
      ;;
    --with-hub)
      WITH_HUB=1
      shift
      ;;
    --with-kiss)
      WITH_KISS=1
      shift
      ;;
    --with-scannetpp)
      WITH_SCANNETPP=1
      shift
      ;;
    --dev)
      WITH_DEV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$CUDA" in
  cu124|cu118|cu111|cpu) ;;
  *)
    echo "Unsupported --cuda value: $CUDA" >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

command_exists() {
  command -v "$@" >/dev/null 2>&1
}

run_with_sudo_if_available() {
  if command_exists sudo && sudo -n true >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

pip_install() {
  "$PYTHON_BIN" -m pip install "$@"
}

install_system_packages() {
  if [[ "$WITH_APT" == "0" ]]; then
    return
  fi
  if ! command_exists apt-get; then
    echo "apt-get not found; skipping system package installation."
    return
  fi

  run_with_sudo_if_available apt-get update -y
  run_with_sudo_if_available apt-get install -y \
    build-essential \
    cmake \
    g++ \
    gcc \
    git \
    libc++-dev \
    libc++1 \
    libc++abi-dev \
    libeigen3-dev \
    libgl1 \
    libtbb-dev \
    ninja-build \
    python3-dev \
    python3-pip \
    unzip
}

install_torch() {
  if [[ "$SKIP_TORCH" == "1" ]]; then
    return
  fi

  case "$CUDA" in
    cu124)
      pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
      ;;
    cu118)
      pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      ;;
    cu111)
      pip_install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
        --extra-index-url https://download.pytorch.org/whl/cu111
      ;;
    cpu)
      pip_install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
      ;;
  esac
}

install_python_package() {
  local extras=("runtime")
  [[ "$WITH_HUB" == "1" ]] && extras+=("hub")
  [[ "$WITH_KISS" == "1" ]] && extras+=("kiss")
  [[ "$WITH_SCANNETPP" == "1" ]] && extras+=("scannetpp")
  [[ "$WITH_DEV" == "1" ]] && extras+=("dev")

  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel ninja

  if [[ "${#extras[@]}" -gt 0 ]]; then
    local extra_spec
    extra_spec="$(IFS=,; echo "${extras[*]}")"
    pip_install -e ".[${extra_spec}]"
  else
    pip_install -e .
  fi
}

install_cuda_extensions() {
  if [[ "$SKIP_CUDA_EXTENSIONS" == "1" || "$CUDA" == "cpu" ]]; then
    echo "Skipping CUDA extensions."
    return
  fi
  if ! command_exists git; then
    echo "git is required to install CUDA extensions." >&2
    exit 1
  fi
  if ! command_exists nvcc; then
    echo "nvcc is required to build pointnet2_ops. Install CUDA Toolkit or pass --skip-cuda-extensions." >&2
    exit 1
  fi

  if [[ -z "${CUDA_HOME:-}" ]]; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi

  TMP_DIR="$(mktemp -d)"

  git clone --depth 1 https://github.com/LucasColas/Pointnet2_PyTorch.git "$TMP_DIR/Pointnet2_PyTorch"
  pip_install "$TMP_DIR/Pointnet2_PyTorch/pointnet2_ops_lib/." --verbose

  pip_install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

  sh cpp_wrappers/compile_wrappers.sh

  git clone --depth 1 https://github.com/KinglittleQ/torch-batch-svd.git "$TMP_DIR/torch-batch-svd"
  pip_install "$TMP_DIR/torch-batch-svd"
}

install_system_packages
install_torch
install_python_package
install_cuda_extensions

echo "BUFFER-X installation completed."
