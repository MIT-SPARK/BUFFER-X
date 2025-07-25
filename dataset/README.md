<div align="center">
    <h1>BUFFER-X</h1>
    <p align="center">
      <a href="https://scholar.google.com/citations?user=esoiHnYAAAAJ&hl=en">Minkyun Seo*</a>,
      <a href="https://scholar.google.com/citations?user=S1A3nbIAAAAJ&hl=en">Hyungtae Lim*</a>,
      <a href="https://scholar.google.com/citations?user=s-haNkwAAAAJ&hl=en">Kanghee Lee</a>,
      <a href="https://scholar.google.com/citations?user=U4kKRdMAAAAJ&hl=it">Luca Carlone</a>,
      <a href="https://scholar.google.com/citations?user=_3q6KBIAAAAJ&hl=en">Jaesik Park</a>
      <br />
    </p>
    <a href="https://github.com/MIT-SPARK/BUFFER-X"><img src="https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54" /></a>
    <a href="https://github.com/MIT-SPARK/BUFFER-X"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://arxiv.org/abs/2503.07940"><img src="https://img.shields.io/badge/arXiv-b33737?logo=arXiv" /></a>
    <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode"><img src="https://img.shields.io/badge/license-CC4.0-blue.svg" /></a>
  <br />
  <br />
  <p align="center"><img src="https://github.com/user-attachments/assets/8cbc95e2-7dc8-46af-9691-b136eb36caad" alt="BUFFER-X" width="95%"/></p>
  <p><strong><em>Towards zero-shot and beyond! 🚀 <br>
  Official repository of BUFFER-X, a zero-shot point cloud registration method<br> across diverse scenes without retraining or tuning.</em></strong></p>
</div>

______________________________________________________________________


# Generalization Benchmark in BUFFER-X

This document provides an overview of the datasets used in our experiments. The datasets are categorized into indoor and outdoor datasets. Each entry includes brief instructions and expected folder structures for proper use.

You can click the links below to jump to each dataset section.

## 🚀 Quick Start

Except for the 'ScanNet++ iPhone' and 'ScanNet++ Faro' datasets, all other datasets can be downloaded in a single command. To download the datasets, run the following command:

```bash
bash download_all_data.sh
```

This script will download all datasets and place them in the `datasets` directory. The expected folder structure is as follows:

```
- `BUFFER-X`
- `datasets`
  - `ThreeDMatch`
  - `tiers_indoor`
  - `kitti`
  - `WOD`
  - `helipr_kaist05`
  - `kimera-multi`
  - `ETH`
  - `newer-college`
```

For ScanNet++ iPhone and Faro datasets, please follow the instructions in their respective sections below.

## How to Parse ScanNet++ Datasets

TBU 
---

## Additional Explanations About Datasets 

### 📌 Indoor Datasets

- [1. 3DMatch](#1-3dmatch)
- [2. 3DLoMatch](#2-3dlomatch)
- [3. ScanNet++ iPhone](#3-scannet-iphone)
- [4. ScanNet++ Faro](#4-scannet-faro)
- [5. TIERS](#5-tiers)

### 📌 Outdoor Datasets

- [6. KITTI](#6-kitti)
- [7. Waymo Open Dataset](#7-waymo-open-dataset)
- [8. KAIST](#8-kaist)
- [9. MIT](#9-mit)
- [10. ETH](#10-eth)
- [11. Oxford](#11-oxford)

## Indoor Datasets

### (1) 3DMatch

Following [Predator](https://github.com/prs-eth/OverlapPredator.git), we provide the processed 3DMatch training set (subsampled fragments with voxel size of 1.5cm and their ground truth transformation matrices).

The structure should be as follows:

- `datasets`
  - `ThreeDMatch`
    - `train`
      - `7-scenes-chess`
      - ...
      - `3DMatch_train_overlap.pkl`
      - `train_3dmatch.txt`
      - `val_3dmatch.txt`
    - `test`
      - `3DLoMatch`
      - `3DMatch`

### (2) 3DLoMatch

3DLoMatch shares the same data structure as 3DMatch.

### (3) ScanNet++ iPhone

Due to dataset sharing policies, we are unable to provide the preprocessed ScanNet++ iPhone and FARO data. Please download the raw ScanNet++ dataset directly from the official source and run our preprocessing scripts.

Place the downloaded ScanNet++ dataset into the `datasets` directory. The expected folder structure is as follows:

```
- `BUFFER-X`
- `datasets`
  - `scannetpp`
    - `scannet-plusplus`
      - `data`
        - `0a5c013435`
          - `iphone`
          - `scans`
```

Then run the following commands to set up the environment.
Make sure you Setup your own virtual environment, e.g., run

```
conda create -n scannetpp_process python=3.8
conda activate scannetpp_process
```

```bash
cd dataset/scannetpp
./env_setup.sh
```

Then run the preprocessing script to generate the required data structure:

```bash
./scannetpp_iphone_preprocess.sh
```

### (4) ScanNet++ Faro

Similar to the iPhone dataset, the Faro dataset requires you to download the raw ScanNet++ dataset and run our preprocessing scripts. ScanNet++ Faro shares environment created in the previous step.

Then run the preprocessing script to generate the required data structure:

```bash
cd dataset/scannetpp
./scannetpp_iphone_preprocess.sh
```

### (4) ScanNet++ Faro

Then run the preprocessing script to generate the required data structure:

```bash
cd dataset/scannetpp
./scannetpp_faro_preprocess.sh
```

### (5) TIERS

The original dataset is available at the official [TIERS GitHub repository](https://github.com/TIERS/tiers-lidars-dataset). In our experiments, we only use the indoor sequences from the TIERS dataset.

The structure should be as follows:

- `datasets`
  - `TIERS`
    - `tiers_indoor06`
      - `os0_128`
        - `scans`
          - `000000.pcd`
          - ...
        - `poses_kitti.txt`
      - `os1_64`
      - `vel16`
    - `tiers_indoor08`
    - `tiers_indoor09`
    - `tiers_indoor10`
    - `tiers_indoor11`

## Outdoor Datasets

### (6) KITTI

the structure is as follows:

- `datasets`
  - `KITTI`
    - `dataset`
      - `pose`
        - `00.txt`
        - ...
      - `sequences`
        - `00`
        - ...

### (7) Waymo Open Dataset

Following [EYOC](https://github.com/liuQuan98/EYOC), we provide the processed WOD dataset.

The structure should be as follows:

- `datasets`
  - `WOD`
    - `test`
      - `sequences`
        - `2601205676330128831_4880_000_4900_000`
          - `scans`
            - `000000.bin`
            - ...
          - `poses.txt`
        - ...

### (8) KAIST

This dataset is derived from the **HeliPR dataset**, using only the **KAIST sequence** for our experiments. The original HeliPR dataset can be downloaded from the [official website](https://sites.google.com/view/heliprdataset).

The structure should be as follows:

- `datasets`
  - `KAIST`
    - `Aeva`
      - `velodyne`
        - `000000.bin`
        - ...
      - `calib.txt`
      - `poses.txt`
    - `Avia`
    - `Ouster`

### (9) MIT

This dataset is derived from the Kimera-Multi dataset, using only the **MIT sequence** for our experiments. The original dataset can be downloaded from the [official website](https://github.com/MIT-SPARK/Kimera-Multi).

The structure should be as follows:

- `datasets`
  - `MIT`
    - `acl_jackal`
      - `scans`
        - `000000.pcd`
        - ...
      - `kimera_multi1_map.pcd`
      - `poses_kitti.txt`
      - `poses_tum.txt`

### (10) ETH

The structure should be as follows:

- `datasets`
  - `ETH`
    - `gazebo_summer`
    - `gazebo_winter`
    - `wood_autmn`
    - `wood_summer`

### (11) Oxford

This dataset is based on the [Newer College Dataset](https://ori-drs.github.io/newer-college-dataset/). We use selected sequences from the dataset for evaluation.

The structure should be as follows:

- `datasets`
  - `Oxford`
    - `01_short_experiments`
      - `scans`
        - `000000.pcd`
        - ...
      - `01_short_experiments_map.pcd`
      - `poses_kitti.txt`
      - `poses_tum.txt`
    - `05_quad_with_dynamics`
    - `07_parkland_mound`
