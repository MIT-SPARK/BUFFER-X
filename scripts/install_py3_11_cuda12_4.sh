#!/bin/bash
set -e

# NOTE(hlim): To support Docker env
command_exists() {
  command -v "$@" >/dev/null 2>&1
}

user_can_sudo() {
  command_exists sudo || return 1
  ! LANG= sudo -n -v 2>&1 | grep -q "may not run sudo"
}

if user_can_sudo; then
SUDO="sudo"
else
SUDO="" # To support docker environment
fi

# -----------------------
# Install basic packages
# -----------------------
$SUDO apt-get update -y
$SUDO apt-get install -y gcc g++ build-essential python3-pip python3-dev cmake git ninja-build unzip libgl1 libtbb-dev libeigen3-dev libc++1 libc++-dev libc++abi-dev

# ------------------------
# Install Python packages
# ------------------------
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install open3d==0.18.0

# NOTE(hlim): In numpy 2.x version, PyObject* {aka _object*}-relevant error happens
# in running `cd cpp_wrappers && sh compile_wrappers.sh && cd ..` line
# So, we need to downgrade the numpy version.
pip3 install numpy==1.26.3

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
# Install `pointnet2-ops`
# NOTE(hlim): With Newer GPU architecture, CUDA compatibilities and architecture versions should be updated
# So we clone modified version of `Pointnet2_PyTorch`
git clone https://github.com/LucasColas/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/ && pip3 install pointnet2_ops_lib/. --verbose
cd ..
rm -rf Pointnet2_PyTorch/

pip3 install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip3 install ninja kornia einops easydict tensorboard tensorboardX tabulate pathlib

# In my desktop, below command did not work
# pip install nibabel -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip3 install nibabel -i https://pypi.tuna.tsinghua.edu.cn/simple

cd cpp_wrappers && sh compile_wrappers.sh && cd ..

git clone https://github.com/KinglittleQ/torch-batch-svd.git && cd torch-batch-svd && python3 setup.py install && cd .. && rm -rf torch-batch-svd/
