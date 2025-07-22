import os
import open3d as o3d
import numpy as np
import re
from tqdm import tqdm
from pointcloud import compute_overlap_ratio
import json

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define ray sampling resolution
azimuth_steps = 1800
elevation_steps = 900
ply_prefix = f"faro_{azimuth_steps}x{elevation_steps}_scanner"


# Helper function to sort file names based on numeric values
def numeric_sort(file_name):
    numbers = re.findall(r"\d+", file_name)
    return int(numbers[0]) if numbers else 0


def trans_scene(scene_path):
    pose_path = os.path.join(scene_path, "scanner_poses.json")
    poses = json.load(open(pose_path))

    for i, pose in enumerate(poses):
        trans_path = os.path.join(scene_path, f"trans_{ply_prefix}_{i}.ply")

        # Skip if the transformed file already exists
        if os.path.exists(trans_path):
            print(f"[SKIP] Transformed point cloud already exists: trans_{ply_prefix}_{i}.ply")
            continue

        input_path = os.path.join(scene_path, f"{ply_prefix}_{i}.ply")
        if not os.path.exists(input_path):
            print(f"Missing input: {input_path}. Skipping.")
            continue

        pcd = o3d.io.read_point_cloud(input_path)
        pcd.transform(np.linalg.inv(pose))
        o3d.io.write_point_cloud(trans_path, pcd)


# Compute overlap ratio and save valid scanner pairs with transformations
def process_scene(scene_path, voxel_size=0.05):
    pose_path = os.path.join(scene_path, "scanner_poses.json")
    poses = json.load(open(pose_path))

    total_cloud_bins = len(poses)
    valid_pairs = []

    gt_log_path = os.path.join(scene_path, "gt.log")
    with open(gt_log_path, "w") as gt_log_file:
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                src_pcd_path = os.path.join(scene_path, f"trans_{ply_prefix}_{i}.ply")
                tgt_pcd_path = os.path.join(scene_path, f"trans_{ply_prefix}_{j}.ply")

                trans = np.linalg.inv(poses[j]) @ poses[i]

                pcd0 = o3d.io.read_point_cloud(src_pcd_path)
                pcd1 = o3d.io.read_point_cloud(tgt_pcd_path)

                ratio = compute_overlap_ratio(pcd0, pcd1, trans, voxel_size)

                if ratio >= 0.3:
                    valid_pairs.append(f"{i} {j} {ratio:.6f}")
                    gt_log_file.write(f"{i}\t{j}\t{total_cloud_bins}\n")
                    for row in trans:
                        gt_log_file.write(
                            f" {row[0]: .8e}\t{row[1]: .8e}\t{row[2]: .8e}\t{row[3]: .8e}\n"
                        )

    overlap_file = os.path.join(scene_path, "overlap_ratio.txt")
    with open(overlap_file, "w") as f:
        f.write("\n".join(valid_pairs))

    print(f"Processed all pairs: {len(valid_pairs)} valid pairs saved to {overlap_file}")


# Main entry point: process only scenes listed in the split file
if __name__ == "__main__":
    base_path = "/root/dataset/scannetpp/scannet-plusplus/data"
    split_path = "../../config/splits/test_scannetpp.txt"

    with open(split_path, "w") as f:
        for scene_id in tqdm(sorted(os.listdir(base_path))):
            scene_dir = os.path.join(base_path, scene_id)
            if not os.path.isdir(scene_dir):
                continue

            scene_path = os.path.join(scene_dir, "scans")
            print(f"Processing scene {scene_id} at {scene_path}")
            if not os.path.exists(os.path.join(scene_path, f"{ply_prefix}_0.ply")):
                break

            trans_scene(scene_path)
            process_scene(scene_path)
            f.write(f"{scene_id}\n")
