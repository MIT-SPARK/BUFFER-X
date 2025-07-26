import os
import open3d as o3d
import numpy as np
import re
from tqdm import tqdm
from pointcloud import compute_overlap_ratio
import random
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Function to sort files based on the numeric part of the file name
def numeric_sort(file_name):
    numbers = re.findall(r'\d+', file_name)
    return int(numbers[0]) if numbers else 0

# Function to load pose based on the corresponding PCD file
def load_pose_from_pcd_path(pcd_path, pose_root):
    pose_index = pcd_path.split('_')[-1].split('.')[0]  # Extract the number part of cloud_bin_x.ply
    pose_filename = f"frame_{int(pose_index) * 50:06d}.pose.txt"  # Match with frame number
    pose_path = os.path.join(pose_root, pose_filename)
    pose = np.loadtxt(pose_path)
    return pose

# Function to filter valid point clouds based on point count
def filter_point_clouds(base_path, point_count_threshold_ratio=0.6):
    tsdf_path = os.path.join(base_path, 'tsdf')
    # Get the list of all point cloud files in the directory
    pcd_files = sorted([f for f in os.listdir(tsdf_path) if f.endswith('.ply')], key=numeric_sort)

    # Calculate the point count for each point cloud
    point_counts = []
    for pcd_file in pcd_files:
        pcd_path = os.path.join(tsdf_path, pcd_file)
        pcd = o3d.io.read_point_cloud(pcd_path)
        point_count = len(pcd.points)
        point_counts.append(point_count)

    # Calculate the median point count
    median_point_count = np.median(point_counts)

    # Define the point count threshold
    point_count_threshold = median_point_count * point_count_threshold_ratio
    print(f"Median point count: {median_point_count}, Threshold: {point_count_threshold}")

    # Filter out point clouds with less than the threshold number of points
    valid_pcd_files = [pcd_files[i] for i in range(len(pcd_files)) if point_counts[i] >= point_count_threshold]

    # Save the valid point cloud file names to a text file
    valid_file_path = os.path.join(base_path, 'valid_pcd_files.txt')
    with open(valid_file_path, 'w') as f:
        for valid_file in valid_pcd_files:
            f.write(f"{valid_file}\n")

    print(f"\nValid point clouds saved to {valid_file_path}")

    return valid_pcd_files

# Load pairs and compute overlap ratio for the valid point clouds
def process_scene(base_path, valid_files, voxel_size=0.05):

    # Prepare lists for storing valid pairs and overlap ratios
    valid_pairs = []
    overlap_ratios = []

    tsdf_path = os.path.join(base_path, 'tsdf')
    pose_path = os.path.join(base_path, 'pose')

    # Get the total number of cloud bins (all .ply files in tsdf path)
    total_cloud_bins = len([f for f in os.listdir(tsdf_path) if f.endswith('.ply')])

    # Prepare the file to store valid poses in gt.log format
    gt_log_path = os.path.join(base_path, 'gt.log')
    with open(gt_log_path, 'w') as gt_log_file:
        # Iterate over valid point cloud pairs
        for i in range(len(valid_files)):
            for j in range(i + 1, len(valid_files)):
                if j - i > 60:
                    break
                # Skip 75% of pairs
                if random.random() < 0.75:
                    continue
                src_pcd_path = os.path.join(tsdf_path, valid_files[i])
                tgt_pcd_path = os.path.join(tsdf_path, valid_files[j])
                src_idx = int(valid_files[i].split('_')[2].split('.')[0])
                tgt_idx = int(valid_files[j].split('_')[2].split('.')[0])

                # Load and transform source and target point clouds using corresponding pose files
                src_pose = load_pose_from_pcd_path(src_pcd_path, pose_path)
                tgt_pose = load_pose_from_pcd_path(tgt_pcd_path, pose_path)
                trans = np.linalg.inv(tgt_pose) @ src_pose

                # Load the point clouds
                pcd0 = o3d.io.read_point_cloud(src_pcd_path)
                pcd1 = o3d.io.read_point_cloud(tgt_pcd_path)

                # Compute the overlap ratio with the actual transformations
                ratio = compute_overlap_ratio(pcd0, pcd1, trans, voxel_size)
                overlap_ratios.append(f"{src_idx}\t{tgt_idx}\t{ratio:.6f}")

                # If overlap ratio is above threshold, save the pose to the gt.log file
                if ratio >= 0.5:
                    valid_pairs.append(f"{src_idx} {tgt_idx} {ratio:.6f}")
                    # Write the pair and corresponding transformation matrix to the gt.log file
                    gt_log_file.write(f"{src_idx}\t{tgt_idx}\t{total_cloud_bins}\n")
                    for row in trans:
                        # Ensure each number has uniform formatting and is tab-separated
                        gt_log_file.write(f" {row[0]: .8e}\t{row[1]: .8e}\t{row[2]: .8e}\t{row[3]: .8e}\n")

    # Save valid pairs to a file (for overlap ratio log)
    valid_pairs_path = os.path.join(base_path, 'overlap_ratio.txt')
    with open(valid_pairs_path, 'w') as f:
        f.write("\n".join(valid_pairs))

    print(f"Processed all pairs: Saved {len(valid_pairs)} valid pairs to {valid_pairs_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process all iPhone scenes in a base directory.")
    parser.add_argument(
        "--base_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../datasets/Scannetpp_iphone/test"),
        help="Base directory containing scene folders with 'iphone' subdirectories"
    )
    args = parser.parse_args()

    # use tqdm
    for scene_id in tqdm(os.listdir(args.base_path)):
        if not os.path.isdir(os.path.join(args.base_path, scene_id)):
            continue
        scene_path = os.path.join(args.base_path, scene_id, 'iphone')
        print(f"Processing scene {scene_id} at {scene_path}")
        valid_files = filter_point_clouds(scene_path)
        process_scene(scene_path, valid_files)
