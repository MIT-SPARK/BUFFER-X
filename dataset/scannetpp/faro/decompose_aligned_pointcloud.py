import torch
import open3d as o3d
import numpy as np
import os
import json
from tqdm import tqdm
import time
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def simulate_faro_scans(data_path, split_file_path):
    # Load scene IDs from the test split file
    with open(split_file_path, "r") as f:
        target_scene_ids = [line.strip() for line in f if line.strip()]

    # Process only scenes listed in the split file
    for scene_id in sorted(target_scene_ids):
        print(f"Processing Scene: {scene_id}")
        scene_path = os.path.join(data_path, scene_id)

        # Skip if the directory does not exist (robustness)
        if not os.path.exists(scene_path):
            print(f"Scene directory not found: {scene_path}. Skipping.")
            continue

        scan_path = os.path.join(scene_path, "scans")
        ply_path = os.path.join(scan_path, "pc_aligned.ply")
        pose_path = os.path.join(scan_path, "scanner_poses.json")
        output_path = scan_path

        pcd = o3d.io.read_point_cloud(ply_path)
        pcd_points = torch.tensor(np.asarray(pcd.points), device="cuda", dtype=torch.float32)

        with open(pose_path, "r") as f:
            poses = json.load(f)

        scanner_positions_np = np.array(poses)[:, :3, 3]  # shape: (N, 3)
        scanner_positions = torch.tensor(scanner_positions_np, device="cuda", dtype=torch.float32)

        azimuth_steps = 1800
        elevation_steps = 900

        if all(
            [
                f"faro_{azimuth_steps}x{elevation_steps}_scanner_{i}.ply" in os.listdir(output_path)
                for i in range(len(scanner_positions))
            ]
        ):
            print("All PLY Files Already Exist. Skipping...")
            continue
        else:
            print("Creating PLY Files...")

        # Define raycasting parameters (range of azimuth and elevation angles)
        azimuths = torch.linspace(
            0, 2 * torch.pi, steps=azimuth_steps, device="cuda", dtype=torch.float32
        )
        elevations = torch.linspace(
            -torch.pi / 2, torch.pi / 2, steps=elevation_steps, device="cuda", dtype=torch.float32
        )

        # Define ray directions as vectors
        azimuth_grid, elevation_grid = torch.meshgrid(azimuths, elevations, indexing="ij")
        ray_directions = torch.stack(
            [
                torch.cos(elevation_grid) * torch.cos(azimuth_grid),
                torch.cos(elevation_grid) * torch.sin(azimuth_grid),
                torch.sin(elevation_grid),
            ],
            dim=-1,
        ).reshape(-1, 3)  # (num_rays, 3)

        num_rays = ray_directions.shape[0]
        num_points = pcd_points.shape[0]

        # Generate PLY file for each scanner position
        for i, scanner_position in enumerate(scanner_positions):
            tic = time.time()
            ray_hit_points = torch.empty((0, 3), device="cuda")

            # 1. Compute direction vectors from scanner to each point
            directions = pcd_points - scanner_position.unsqueeze(0)
            distances = torch.norm(directions, dim=1)

            # Filter: keep points that are at least 0.6m away from the scanner
            valid_mask = distances >= 0.6
            valid_points = pcd_points[valid_mask]
            valid_directions = directions[valid_mask]
            distances = distances[valid_mask]

            # Normalize direction vectors
            directions_normalized = valid_directions / distances.unsqueeze(1)

            # Compute azimuth (in radians, range 0 to 2π)
            azimuths_points = torch.atan2(directions_normalized[:, 1], directions_normalized[:, 0])
            azimuths_points = torch.where(
                azimuths_points < 0, azimuths_points + 2 * torch.pi, azimuths_points
            )

            # Compute elevation
            elevations_points = torch.asin(directions_normalized[:, 2])

            # Assign each point to a ray
            azimuth_indices = torch.bucketize(azimuths_points, azimuths)
            elevation_indices = torch.bucketize(elevations_points, elevations)

            ray_indices = azimuth_indices * elevations.shape[0] + elevation_indices

            # Map from ray index to point indices (with progress bar)
            ray_to_point_map = [[] for _ in range(num_rays)]
            for point_idx, ray_idx, distance in zip(
                tqdm(range(num_points), desc=f"Assigning Points to Rays for Scanner {i+1}"),
                ray_indices,
                distances,
            ):
                ray_to_point_map[ray_idx].append(point_idx)

            # 2. Process each ray to find the closest point
            ray_hit_points = torch.empty((0, 3), device="cuda")

            print(f"Processing Scanner {i+1}...")

            for point_indices in tqdm(ray_to_point_map, desc=f"Processing Rays for Scanner {i+1}"):
                if len(point_indices) == 0:
                    continue

                # Get coordinates of points in the ray
                points_in_ray = valid_points[point_indices]

                # Select the closest point to the scanner
                hit_index = torch.argmin(
                    torch.norm(points_in_ray - scanner_position.unsqueeze(0), dim=1)
                )
                closest_point = points_in_ray[hit_index].unsqueeze(0)

                # Add to result tensor
                ray_hit_points = torch.cat((ray_hit_points, closest_point), dim=0)

            # Convert result to Open3D point cloud and save
            ray_hit_pcd = o3d.geometry.PointCloud()
            ray_hit_pcd.points = o3d.utility.Vector3dVector(ray_hit_points.cpu().numpy())
            ply_file_name = f"faro_{azimuth_steps}x{elevation_steps}_scanner_{i}.ply"
            output_file_name = os.path.join(output_path, ply_file_name)
            o3d.io.write_point_cloud(output_file_name, ray_hit_pcd)
            print(f"Saving PLY File Time for Scanner {i+1}: {time.time() - tic:.4f} seconds")
            print(f"Saved: {output_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FARO-style scanner simulation for ScanNet++ scenes")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join("..", "..", "..", "datasets", "scannetpp", "scannet-plusplus", "data"),
        help="Path to the ScanNet++ dataset",
    )
    parser.add_argument(
        "--split_file_path",
        type=str,
        default=os.path.join("..", "..", "config", "splits", "test_scannetpp_faro.txt"),
        help="Path to the split file listing scene IDs to process",
    )
    args = parser.parse_args()

    simulate_faro_scans(args.data_path, args.split_file_path)
