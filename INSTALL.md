# Installing BUFFER-X

BUFFER-X requires Linux, an NVIDIA GPU, and CUDA extensions for full inference.
The unified installer handles PyTorch, Python dependencies, and the required
extensions.

## Recommended Installation

```bash
git clone https://github.com/MIT-SPARK/BUFFER-X
cd BUFFER-X
conda create -n bufferx python=3.11 -y
conda activate bufferx
./scripts/install.sh --cuda cu124 --with-hub
```

Supported CUDA targets:

- `--cuda cu124`: PyTorch CUDA 12.4 wheels (recommended)
- `--cuda cu118`: PyTorch CUDA 11.8 wheels
- `--cuda cu111`: legacy PyTorch 1.9.1 and CUDA 11.1
- `--cuda cpu --skip-cuda-extensions`: utilities only; inference is unavailable

Optional installer flags:

- `--with-hub`: Hugging Face download and upload support
- `--with-kiss`: KISS-Matcher pose estimation
- `--with-scannetpp`: ScanNet++ preprocessing dependencies
- `--dev`: development tools
- `--no-apt`: skip system package installation
- `--skip-torch`: use an existing PyTorch installation
- `--skip-cuda-extensions`: skip CUDA extension installation

Run `./scripts/install.sh --help` for the complete option list.

## Pretrained Models

Download the public checkpoints from Hugging Face:

```bash
python scripts/download_pretrained_models.py \
  --source hf \
  --repo-id Hyungtae-Lim/BUFFER-X
```

The files are downloaded to:

```text
snapshot/
  threedmatch/Desc/best.pth
  threedmatch/Pose/best.pth
  kitti/Desc/best.pth
  kitti/Pose/best.pth
```

The original Dropbox release remains available through:

```bash
./scripts/download_pretrained_models.sh
```

## Other Installation Options

The repository also includes environment-specific legacy installers:

```bash
./scripts/install_py3_8_cuda11_1.sh
./scripts/install_py3_10_cuda11_8.sh
./scripts/install_py3_11_cuda12_4.sh
```

For pure-Python dependencies only:

```bash
pip install -e '.[runtime]'
```

This does not install PyTorch or the CUDA extensions required for inference.

## KISS-Matcher

Install with:

```bash
./scripts/install.sh --cuda cu124 --with-kiss
```

You can also follow the
[official KISS-Matcher installation instructions](https://github.com/MIT-SPARK/KISS-Matcher?tab=readme-ov-file#package-installation).
If KISS-Matcher is unavailable, BUFFER-X falls back to RANSAC.

## Publishing to Hugging Face

This section is for maintainers publishing checkpoints or the Space helper.

```bash
pip install -e '.[hub]'
hf auth login
python scripts/upload_to_huggingface.py \
  --model-repo-id Hyungtae-Lim/BUFFER-X \
  --space-repo-id Hyungtae-Lim/buffer-x-hub-helper \
  --dry-run
```

Remove `--dry-run` to upload. The helper publishes the model card, local
`snapshot/` directory, and CPU-only Space helper.

Do not include Hugging Face access tokens in issues, pull requests, notebooks,
or command history.
