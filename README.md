[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

## Code Clean-up Blueprint

### Directory Re-organization
- `dataset`
    - `dataloader.py`
    - `threedmatch`
    - `kitti`
    - `kaist`
    - ...
- `config`
    - `config.py`
    - `threedmatch_config.py`
    - `kitti_config.py`
    - `kaist_config.py`
    - ...
- `models`
    - `BUFFER.py`
    - ...
- `utils`
    - `common.py`
    - ...
- `snapshot`
    - `threedmatch`
    - `kitti`
- `test.py`
- `train.py`
- `trainer.py`
- `demo.py`



## 민균 코멘트

주요 변경점
- SuperSet_Eval을 통해 한 번에 Eval 가능 (단, 3DMatch의 경우 3DMatch Setting은 따로 해줘야 함)
- utils/tools.py에 있는 find_voxel_size 함수를 이용하여 voxel size를 정하도록 수정함
- models/BUFFER.py에 있는 find_des_r 함수를 이용하여 radius를 정하도록 수정함
- 기존 BUFFER의 point_learner 부분 삭제 (사실 point_learner.py 삭제해도 됨)
- 3가지 scale의 radius로 매칭
- dataset.py에 있는 second downsample도 test 시점에는 완전히 불필요함. + Normal 계산도 마찬가지
 단, 추후 training 과정에서 필요할까봐 남겨둠.
- snapshot에서 pretrained weight는 3DMatch, KITTI 중 고르면 됨.
- 현재 유일하게 자동화하지 않은 부분은 RANSAC param.

## Train & Test 

### Training 관련

- Train Dataset에 대한 Prior knowledge는 있다고 판단
- 따라서, Train Dataset에 대한 Voxel Size, Radius는 지정해준 값을 일단 사용 중 (BUFFER와 동일)
- 하지만, Train 시점에서부터 Voxel Size, Radius를 동적으로 사용하는 것도 테스트 예정
  (일종의 augmentation 기대)
- Training은 3DMatch, KITTI에서만 하는 것으로

### 3DMatch

```
cd ./ThreeDMatch
python train.py
python test.py
```

### Superset Eval

```
cd ./SuperSet_eval
python test.py
```

### Todo
- Voxel size 찾는 부분 속도 개선
- 3 Scale Matching 시간 개선 (병렬화 필요)
- RANSAC param 자동화

테스트 환경
(민균 기준 BUFFER의 환경 그대로 사용함)
- Python 3.10 
- Cuda 11.8 
- Cudnn 8 

## 참고용:

- 알고리즘에서 scale을 쓰는 부분을 모두 제거했음!
- Superset 관리를 `config["KITTI"]`, `config["ThreeDMatch"]`와 같이 dataset의 이름을 key로 받아서 관리하도록 수정함
- `config.data.voxel_size_0`의 값을 reference voxel size로 사용함
- 앞으로는 Docker 내의 주소를 위해 다음과 같이 주소를 통일하시죳!

```angular2html
/opt/project <- BUFFER directory
/opt/datasets
├── ETH
    ├── ...
├── kitti
    ├── ...
├── ThreeDMatch
    ├── ...
```

## 하면 좋은 것 

- 현재는 batch를 1밖에 지원안 함...이를 수정하면 더 좋을 것 같음!

---

###  Setup
This code has been tested with Python 3.8, Pytorch 1.9.1, CUDA 11.1 on Ubuntu 20.04.
 
- Clone the repository 
```
git clone https://github.com/aosheng1996/BUFFER && cd BUFFER
```
- Setup conda virtual environment
```
conda create -n buffer python=3.8
source activate buffer
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

### (3) 3DMatch
Following [Predator](https://github.com/prs-eth/OverlapPredator.git), we provide the processed 3DMatch training set (subsampled fragments with voxel size of 1.5cm and their ground truth transformation matrices). 

Download the processed dataset from [Google Drive](https://drive.google.com/drive/folders/1tWVV4u_YablYmPta8fmHLY-JN4kZWh8R?usp=sharing) and put the folder into `data`. 
Then the structure should be as follows:


**Training**

Training BUFFER on the 3DMatch dataset:
```
cd ./ThreeDMatch
python train.py
```
**Testing**

Evaluate the performance of the trained models on the 3DMatch dataset:

```
cd ./ThreeDMatch
python test.py
```
To evaluate the performance of BUFFER on the 3DLoMatch dataset, you only need to modify the `_C.data.dataset = '3DMatch'` in `config.py` to `_C.data.dataset = '3DLoMatch'` and performs:
```
python test.py
``` 

### (4) KITTI
Download the data from the [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) into `data`. 
Then the structure is as follows:

- `data`
    - `KITTI`
        - `dataset`
            - `pose`
                - `00.txt`
                - ...
            - `sequences`
                - `00`
                - ...

**Training**

Training BUFFER on the KITTI dataset:

```
cd ./KITTI
python train.py
```

**Testing**

Evaluate the performance of the trained models on the KITTI dataset:

```
cd ./KITTI
python test.py
```

### (5) ETH

The test set can be downloaded from [here](https://share.phys.ethz.ch/~gsg/3DSmoothNet/data/ETH.rar), and put the folder into `data`, then the structure is as follows:

- `data`
    - `ETH`
        - `gazebo_summer`
        - `gazebo_winter`
        - `wood_autmn`
        - `wood_summer`


### (6) Generalizing to Unseen Datasets 

**3DMatch to ETH**

Generalization from 3DMatch dataset to ETH dataset:
```
cd ./generalization/ThreeD2ETH
python test.py
```

**3DMatch to KITTI**

Generalization from 3DMatch dataset to KITTI dataset:

```
cd ./generalization/ThreeD2KITTI
python test.py
```

**KITTI to 3DMatch**

Generalization from KITTI dataset to 3DMatch dataset:
```
cd ./generalization/KITTI2ThreeD
python test.py
```

**KITTI to ETH**

Generalization from KITTI dataset to ETH dataset:
```
cd ./generalization/KITTI2ETH
python test.py
```

## Acknowledgement

In this project, we use (parts of) the implementations of the following works:

* [Vector Neurons](https://github.com/FlyingGiraffe/vnn)
* [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch)
* [PointDSC](https://github.com/XuyangBai/PointDSC)
* [SpinNet](https://github.com/QingyongHu/SpinNet)
* [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
* [RoReg](https://github.com/HpWang-whu/RoReg)

### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{ao2023buffer,
      title={BUFFER: Balancing Accuracy, Efficiency, and Generalizability in Point Cloud Registration},
      author={Ao, Sheng and Hu, Qingyong and Wang, Hanyun and Xu, Kai and Guo, Yulan},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={1255--1264},
      year={2023}
    }

### Updates
* 07/06/2023: The code is released!
* 28/02/2023: This paper has been accepted by CVPR 2023!

## Related Repos
1. [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://github.com/QingyongHu/RandLA-Net) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/RandLA-Net.svg?style=flat&label=Star)
2. [SoTA-Point-Cloud: Deep Learning for 3D Point Clouds: A Survey](https://github.com/QingyongHu/SoTA-Point-Cloud) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SoTA-Point-Cloud.svg?style=flat&label=Star)
3. [3D-BoNet: Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds](https://github.com/Yang7879/3D-BoNet) ![GitHub stars](https://img.shields.io/github/stars/Yang7879/3D-BoNet.svg?style=flat&label=Star)
4. [SensatUrban: Learning Semantics from Urban-Scale Photogrammetric Point Clouds](https://github.com/QingyongHu/SensatUrban) ![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SensatUrban.svg?style=flat&label=Star)
5. [SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration](https://github.com/QingyongHu/SpinNet)![GitHub stars](https://img.shields.io/github/stars/QingyongHu/SpinNet.svg?style=flat&label=Star)
