import open3d as o3d
import numpy as np
import json
import os

# Function to load and visualize two PCD files with different colors
def visualize_transformed_pcd(file_path1, file_path2, pose1, pose2):
    # Read the first point cloud file
    xyz1 = np.fromfile(file_path1, dtype=np.float32).reshape(-1, 3)
    xyz2 = np.fromfile(file_path2, dtype=np.float32).reshape(-1, 3)
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(xyz1)
    pcd1.paint_uniform_color([1, 0.706, 0]) # orange

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(xyz2)
    pcd2.paint_uniform_color([0, 0.651, 0.929]) # blue

    raw_pcd = pcd1 + pcd2
    o3d.io.write_point_cloud("raw_pcd.ply", raw_pcd)
    
    trans_1 = pose2 @ np.linalg.inv(pose1)
    trans_2 = np.linalg.inv(pose2) @ pose1
    
    pcd1.transform(trans_2)
    transformed_pcd = pcd1 + pcd2
    o3d.io.write_point_cloud("transformed_pcd.ply", transformed_pcd)
    
    # Print details of the point clouds
    print("PCD 1 (transformed):", pcd1)
    print(np.asarray(pcd1.points))
    print("PCD 2:", pcd2)
    print(np.asarray(pcd2.points))
    bounding_box = pcd1.get_axis_aligned_bounding_box()
    bbox_min = np.asarray(bounding_box.min_bound)
    bbox_max = np.asarray(bounding_box.max_bound)
    print("Bounding box min:", bbox_min)
    print("Bounding box max:", bbox_max)
    max_distance = np.linalg.norm(bbox_max - bbox_min)
    print("Max distance:", max_distance)

# Example usage
pcd_file1 = '/root/dataset/WOD/test/sequences/2601205676330128831_4880_000_4900_000/scans/000000.bin'
pcd_file2 = '/root/dataset/WOD/test/sequences/2601205676330128831_4880_000_4900_000/scans/000062.bin'

pose_root = '/Users/minkyun/Downloads/pose/'
# #load pose from txt file
# pose1 = load_pose_from_pcd_path(pcd_file1, pose_root)
# pose2 = load_pose_from_pcd_path(pcd_file2, pose_root)


pose1 = [[-9.977478517327e-01, 6.555497281845e-02, -1.420457326365e-02, 2.498940875916e+03],
        [-6.561996278420e-02, -9.978360241170e-01, 4.158059474901e-03, 1.435311584571e+04],
        [-1.390125343382e-02, 5.080798477385e-03, 9.998904643209e-01, 6.098900000000e+01],
        [0, 0, 0, 1]]

pose2 = [[-9.671917650374e-01, 2.540465634627e-01, 6.582071264276e-04, 2.489280959018e+03],
        [-2.540300304369e-01, -9.671526399760e-01, 9.193183546701e-03, 1.435190728836e+04],
        [2.972083447296e-03, 8.724367044486e-03, 9.999575251678e-01, 6.102600000000e+01],
        [0, 0, 0, 1]]

visualize_transformed_pcd(pcd_file1, pcd_file2, pose1, pose2)

## 2601205676330128831_4880_000_4900_000_0_62
# [[0.9809806409889579, 0.19343348041735353, 0.016139103402491334, -9.7520879244894],
#  [-0.193494499181391, 0.9810986777784008, 0.0022941769684937142, 1.2770330642874097],
#  [-0.015390282372999706, -0.005373370923196297, 0.9998671242187191, 0.03636210223600637],
#  [0, 0, 0, 1]]
