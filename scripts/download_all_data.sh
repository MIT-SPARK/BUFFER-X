#!/bin/bash
set -e

# NOTE(hlim): `ROOT_DIR` should be equal to `args.root_dir` in `test.py`
ROOT_DIR="../datasets"

mkdir -p "$ROOT_DIR"

# NOTE(hlim): `DATASET_NAME` should be equal to the directory name in
# the member variable `self._C.data.root` in `config/*_config.py

# 3DMatch
DATASET_NAME="ThreeDMatch"
wget -O "$ROOT_DIR/$DATASET_NAME.zip" "https://www.dropbox.com/scl/fi/4xq1a7nfoz9a7juh5aa70/ThreeDMatch.zip?rlkey=2dck3huy0eno2od7u4jwvni4n&st=035hiakw&dl=1"
unzip "$ROOT_DIR/$DATASET_NAME.zip" -d "$ROOT_DIR"
rm "$ROOT_DIR/$DATASET_NAME.zip"

# TIERS
DATASET_NAME="tiers_indoor"
wget -O "$ROOT_DIR/$DATASET_NAME.zip" "https://www.dropbox.com/scl/fi/5nwj9322c1udqjksmxnw7/tiers_indoor.zip?rlkey=3ye4k5jensmutvnvdb7ogrzx1&st=n30vbr0q&dl=1"
unzip "$ROOT_DIR/$DATASET_NAME.zip" -d "$ROOT_DIR"
rm "$ROOT_DIR/$DATASET_NAME.zip"

# KITTI
DATASET_NAME="kitti"
wget -O "$ROOT_DIR/$DATASET_NAME.zip" "https://www.dropbox.com/scl/fi/w4odsfngu5q961k5m5mg5/kitti.zip?rlkey=snhfdwapxcsc252m9me1kzppl&st=jeohoe7g&dl=1"
unzip "$ROOT_DIR/$DATASET_NAME.zip" -d "$ROOT_DIR"
rm "$ROOT_DIR/$DATASET_NAME.zip"

# WOD
DATASET_NAME="WOD"
wget -O "$ROOT_DIR/$DATASET_NAME.zip" "https://www.dropbox.com/scl/fi/45lzbcb5xndt93ot7pn05/WOD.zip?rlkey=rntndreq6eif0rokegyh25vmm&st=6onhudk8&dl=1"
unzip "$ROOT_DIR/$DATASET_NAME.zip" -d "$ROOT_DIR"
rm "$ROOT_DIR/$DATASET_NAME.zip"

# KAIST
DATASET_NAME="helipr_kaist05"
wget -O "$ROOT_DIR/$DATASET_NAME.zip" "https://www.dropbox.com/scl/fi/80h04267u0uo96gvyl6r9/helipr_kaist05.zip?rlkey=75lkovmdx6n2ofeymkvr0zq8q&st=4irxe4dw&dl=1"
unzip "$ROOT_DIR/$DATASET_NAME.zip" -d "$ROOT_DIR"
rm "$ROOT_DIR/$DATASET_NAME.zip"

# MIT
DATASET_NAME="kimera-multi"
wget -O "$ROOT_DIR/$DATASET_NAME.zip" "https://www.dropbox.com/scl/fi/89cnbk3txywm06pg81tn5/kimera-multi.zip?rlkey=fanne42csa69lf96bpmvc0chi&st=30z8hdn3&dl=1"
unzip "$ROOT_DIR/$DATASET_NAME.zip" -d "$ROOT_DIR"
rm "$ROOT_DIR/$DATASET_NAME.zip"

# ETH
DATASET_NAME="ETH"
wget -O "$ROOT_DIR/$DATASET_NAME.zip" "https://www.dropbox.com/scl/fi/k3uj3h8hf9b76ymp7nebi/ETH.zip?rlkey=11hg1owj5veaymh5wpu9wfvtj&st=wbyaae3i&dl=1"
unzip "$ROOT_DIR/$DATASET_NAME.zip" -d "$ROOT_DIR"
rm "$ROOT_DIR/$DATASET_NAME.zip"

# Oxford
DATASET_NAME="newer-college"
wget -O "$ROOT_DIR/$DATASET_NAME.zip" "https://www.dropbox.com/scl/fi/lrzmlpsu6hlt834w1lgl9/newer-college.zip?rlkey=sn24v0oxi29xzbhfajb13wxad&st=mnttgb31&dl=1"
unzip "$ROOT_DIR/$DATASET_NAME.zip" -d "$ROOT_DIR"
rm "$ROOT_DIR/$DATASET_NAME.zip"
