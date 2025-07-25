"""
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import json
import zlib
import numpy as np
import imageio as iio
import lz4.block

from .scene_release import ScannetppScene_Release
from .utils import run_command, load_yaml_munch


def extract_rgb(scene):
    scene.iphone_rgb_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {scene.iphone_video_path} -start_number 0 -q:v 1 {scene.iphone_rgb_dir}/frame_%06d.color.jpg"
    run_command(cmd, verbose=True)


def extract_masks(scene):
    scene.iphone_video_mask_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"ffmpeg -i {str(scene.iphone_video_mask_path)} -pix_fmt gray -start_number 0 {scene.iphone_video_mask_dir}/frame_%06d.png"
    run_command(cmd, verbose=True)


def extract_depth(scene):
    # global compression with zlib
    height, width = 192, 256
    sample_rate = 1
    scene.iphone_depth_dir.mkdir(parents=True, exist_ok=True)

    if not scene.iphone_depth_path.exists():
        return None

    try:
        with open(scene.iphone_depth_path, "rb") as infile:
            data = infile.read()
            data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
            depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

        for frame_id in tqdm(range(0, depth.shape[0], sample_rate), desc="decode_depth"):
            iio.imwrite(
                f"{scene.iphone_depth_dir}/frame_{frame_id:06}.depth.png",
                (depth * 1000).astype(np.uint16),
            )
    # per frame compression with lz4/zlib
    except:
        frame_id = 0
        with open(scene.iphone_depth_path, "rb") as infile:
            while True:
                size = infile.read(4)  # 32-bit integer
                if len(size) == 0:
                    break
                size = int.from_bytes(size, byteorder="little")
                if frame_id % sample_rate != 0:
                    infile.seek(size, 1)
                    frame_id += 1
                    continue

                # read the whole file
                data = infile.read(size)
                try:
                    # try using lz4
                    data = lz4.block.decompress(
                        data, uncompressed_size=height * width * 2
                    )  # UInt16 = 2bytes
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
                except:
                    # try using zlib
                    data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                    depth = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                    depth = (depth * 1000).astype(np.uint16)

                # 6 digit frame id = 277 minute video at 60 fps
                iio.imwrite(f"{scene.iphone_depth_dir}/frame_{frame_id:06}.depth.png", depth)
                frame_id += 1


def extract_poses(scene):
    """
    Extracts pose data from pose_intrinsic_imu.json and saves each frame's pose as a separate .txt file.
    """
    scene.pose_dir.mkdir(parents=True, exist_ok=True)
    json_path = scene.pose_intrinsic_imu_path

    with open(json_path, "r") as f:
        data = json.load(f)

    for frame_name, frame_data in tqdm(data.items(), desc="extract_poses"):
        pose = frame_data.get("aligned_pose", [])  # 'aligned_pose'
        if pose:
            pose_filename = scene.pose_dir / f"{frame_name}.pose.txt"
            with open(pose_filename, "w") as pose_file:
                for row in pose:
                    pose_file.write(" ".join(map(str, row)) + "\n")


def extract_intrinsic(scene):
    """
    Extracts intrinsic data from pose_intrinsic_imu.json and saves each frame's intrinsic as a separate .txt file.
    """
    scene.intrinsic_dir.mkdir(parents=True, exist_ok=True)
    json_path = scene.pose_intrinsic_imu_path

    ratio = 7.5  # 7.5 = 1920 / 256

    with open(json_path, "r") as f:
        data = json.load(f)

    for frame_name, frame_data in tqdm(data.items(), desc="extract_intrinsic"):
        intrinsic = frame_data.get("intrinsic", [])
        if intrinsic:
            intrinsic_filename = scene.intrinsic_dir / f"{frame_name}.intrinsic.txt"
            with open(intrinsic_filename, "w") as intrinsic_file:
                # Divide all values in the intrinsic matrix by ratio
                intrinsic_scaled = [[element / ratio for element in row] for row in intrinsic]
                for row in intrinsic_scaled:
                    intrinsic_file.write(" ".join(map(str, row)) + "\n")


def main(args):
    cfg = load_yaml_munch(args.config_file)

    # get the scenes to process
    if cfg.get("scene_ids"):
        scene_ids = cfg.scene_ids
    # elif cfg.get('splits'):
    #     scene_ids = []
    #     for split in cfg.splits:
    #         split_path = Path(cfg.data_root) / 'splits' / f'{split}.txt'
    #         scene_ids += read_txt_list(split_path)

    # get all scene directories from the data root
    # data_root = Path(cfg.data_root)
    # scene_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    # scene_dirs = sorted([d.name for d in scene_dirs])

    # get the options to process
    # go through each scene
    for scene_id in tqdm(scene_ids, desc="scene"):
        # scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / 'data')
        scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root))
        if cfg.extract_rgb:
            extract_rgb(scene)

        if cfg.extract_masks:
            extract_masks(scene)

        if cfg.extract_depth:
            extract_depth(scene)

        if cfg.extract_poses:
            extract_poses(scene)

        if cfg.extract_intrinsic:
            extract_intrinsic(scene)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
