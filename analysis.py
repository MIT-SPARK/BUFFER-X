# Voxelization 후 point 갯수 / max range txxt로 save하기
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
 
# Define the base directory (modify this if needed)
BASE_DIR = "./"  # 데이터셋이 있는 최상위 경로
 
# Define the sensors
sensors = ["os0_128", "os1_64", "vel16"]
 
# Define voxel sizes
voxel_sizes = [0.05, 0.1, 0.25, 0.5]
 
# Iterate through each sensor type
for sensor in sensors:
    print(f"Processing sensor: {sensor}")
 
    # Output file for the sensor
    output_file = f"tiers_{sensor}_voxel_results.txt"
 
    # Find all scan directories for the sensor
    scan_dirs = []
    for root, dirs, files in os.walk(BASE_DIR):
        if root.endswith(f"/{sensor}/scans"):  # 찾고자 하는 scans 폴더
            scan_dirs.append(root)
 
    # Process each scan directory
    for scan_dir in scan_dirs:
        print(f"  Processing directory: {scan_dir}")
        # List all PCD files in the scan directory
        pcd_files = sorted([f for f in os.listdir(scan_dir) if f.endswith(".pcd")])
 
        for pcd_file in tqdm(pcd_files, desc=f"    Processing {sensor} scans"):
            # Load PCD file
            pcd_path = os.path.join(scan_dir, pcd_file)
            pcd = o3d.io.read_point_cloud(pcd_path)
 
            if len(pcd.points) == 0:
                print(f"Warning: Empty PCD file {pcd_path}")
                continue
 
            # Convert to numpy array
            pts = np.asarray(pcd.points)
 
            # Lists to store results
            results = []
 
            for v_size in voxel_sizes:
                # Apply voxel downsampling
                voxelized_pcd = pcd.voxel_down_sample(v_size)
 
                # Get number of points after voxelization
                num_points = np.asarray(voxelized_pcd.points).shape[0]
                results.append(num_points)
 
            # Compute max L2 norm
            max_range = np.max(np.linalg.norm(pts, axis=1))
            results.append(max_range)
 
            # Convert to numpy array (1x5 shape)
            results_array = np.array([results])  # Shape: (1, 5)
 
            # Save to TXT file (append mode)
            with open(output_file, "a") as f:  # "a" mode for appending
                np.savetxt(f, results_array, fmt="%.6f")
 
    print(f"Results saved to {output_file}")
 
print("Processing complete!")