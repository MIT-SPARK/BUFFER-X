[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
## BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes (ICCV 2025)

This is the official repository of **BUFFER-X**, a zero-shot point cloud registration method designed for robust performance across diverse scenes without retraining or manual tuning. For technical details, please refer to:

**[BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](https://arxiv.org/abs/2503.07940)**  <br />
[Minkyun Seo*](https://scholar.google.com/citations?user=esoiHnYAAAAJ&hl=en), [Hyungtae Lim*](https://scholar.google.com/citations?user=S1A3nbIAAAAJ&hl=en), [Kanghee Lee](https://scholar.google.com/citations?user=s-haNkwAAAAJ&hl=en), [Luca Carlone](https://scholar.google.com/citations?user=U4kKRdMAAAAJ&hl=it), [Jaesik Park](https://scholar.google.com/citations?user=_3q6KBIAAAAJ&hl=en). <br />

<!-- **[[Paper](https://arxiv.org/abs/2503.07940)] [Video] [Project page]** <br /> -->


### (1) Overview
![fig1](fig/BUFFER-X_Overview.png)


### (2) Setup
This code has been tested with Python 3.8, Pytorch 1.9.1, CUDA 11.1 on Ubuntu 20.04.
 
- Clone the repository 
```
git clone https://github.com/MIT-SPARK/BUFFER-X && cd BUFFERX
```
- Setup conda virtual environment
```
conda create -n bufferx python=3.8
source activate bufferx
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install open3d==0.13.0

export CUDA_HOME=/your/cuda/home/directory/
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
pip install ninja kornia einops easydict tensorboard tensorboardX
pip install nibabel -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
cd cpp_wrappers && sh compile_wrappers.sh && cd ..
git clone https://github.com/KinglittleQ/torch-batch-svd.git && cd torch-batch-svd && python setup.py install && cd .. && sudo rm -rf torch-batch-svd/
```

### (3) Datasets

Due to the large number and variety of datasets used in our experiments, we provide detailed download instructions and folder structures in a separate document:

[DATASETS.md](dataset/DATASETS.md)

### (4) Training and Testing

**Training**

BUFFER-X supports training on either the **3DMatch** or **KITTI** dataset.  

#### Example
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3DMatch
```

**Testing**

To evaluate **BUFFER-X** on a specific dataset, use the `test.py` script with the following arguments:

- `--dataset`: The name of the dataset to test on. Must be one of:
    - `3DMatch`
    - `3DLoMatch`
    - `Scannetpp_iphone`
    - `Scannetpp_faro`
    - `Tiers`
    - `KITTI`
    - `WOD`
    - `MIT`
    - `KAIST`
    - `ETH`
    - `Oxford`

- `--experiment_id`: The ID of the experiment to use for testing.

#### Example  
Evaluate a model with the experiment ID `threedmatch` on the `KITTI` dataset:
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataset KITTI --experiment_id threedmatch
```

You can also run evaluation on all supported datasets with a single script:

```bash
./eval_all.sh EXPERIMENT_ID
```

## Acknowledgement

In this project, we use (parts of) the implementations of the following works:

* [FCGF](https://github.com/chrischoy/FCGF)
* [Vector Neurons](https://github.com/FlyingGiraffe/vnn)
* [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch)
* [PointDSC](https://github.com/XuyangBai/PointDSC)
* [SpinNet](https://github.com/QingyongHu/SpinNet)
* [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
* [RoReg](https://github.com/HpWang-whu/RoReg)
* [BUFFER](https://github.com/SYSU-SAIL/BUFFER)

### Citation
If you find our work useful in your research, please consider citing:

    @article{Seo_BUFFERX_arXiv_2025,
    Title={BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes},
    Author={Minkyun Seo and Hyungtae Lim and Kanghee Lee and Luca Carlone and Jaesik Park},
    Journal={2503.07940 (arXiv)},
    Year={2025}
    }

### Updates
* 25/06/2025: This paper has been accepted by ICCV 2025!