---
library_name: pytorch
license: mit
tags:
- point-cloud-registration
- 3d-vision
- lidar
- robotics
- pytorch
- iccv-2025
---

# BUFFER-X

BUFFER-X is a PyTorch model for zero-shot point cloud registration across indoor,
outdoor, homogeneous, and heterogeneous sensor settings.

This Hugging Face repository is intended to host the pretrained BUFFER-X snapshots.
The code lives in the official GitHub repository:

https://github.com/MIT-SPARK/BUFFER-X

The repository metadata uses the MIT license to match the included `LICENSE` file.
If you release model weights under different terms, update the YAML metadata before
uploading.

## Expected Files

Upload pretrained weights under the same layout used by the GitHub code:

```text
snapshot/
  threedmatch/
    Desc/best.pth
    Pose/best.pth
  kitti/
    Desc/best.pth
    Pose/best.pth
```

The included upload helper preserves this layout automatically when a local
`snapshot/` directory exists.

## Usage

Install BUFFER-X, then download the pretrained snapshots from this model repo:

```bash
git clone https://github.com/MIT-SPARK/BUFFER-X
cd BUFFER-X
./scripts/install.sh --cuda cu124 --with-hub
python scripts/download_pretrained_models.py --source hf --repo-id <this-model-repo>
```

Run evaluation after preparing the datasets:

```bash
python test.py --dataset 3DMatch TIERS Oxford MIT --experiment_id threedmatch --verbose
```

## Requirements

BUFFER-X inference uses CUDA-specific dependencies, including `pointnet2_ops`,
`KNN_CUDA`, custom C++ wrappers, and `torch-batch-svd`. The GitHub installation
script installs these pieces for supported PyTorch/CUDA combinations.

## Limitations

- The hosted pretrained snapshots do not include benchmark datasets.
- ScanNet++ preprocessing must be run from the original dataset because modified
  files cannot be redistributed by this project.
- CPU-only installation is useful for reading utilities and packaging checks, but
  full BUFFER-X inference requires CUDA extensions.

## Citation

```bibtex
@article{Seo_BUFFERX_arXiv_2025,
  title={BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes},
  author={Minkyun Seo and Hyungtae Lim and Kanghee Lee and Luca Carlone and Jaesik Park},
  journal={2503.07940 (arXiv)},
  year={2025}
}
```
