import os
import open3d
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import time

def get_pcd(pcdpath, filename):
    return open3d.io.read_point_cloud(os.path.join(pcdpath, filename + '.ply'))


def get_keypts(keyptspath, filename):
    keypts = np.fromfile(os.path.join(keyptspath, filename + '.keypts.bin'), dtype=np.float32)
    num_keypts = int(keypts[0])
    keypts = keypts[1:].reshape([num_keypts, 3])
    return keypts
    
    
def get_ETH_keypts(pcd, keyptspath, filename):
    pts = np.array(pcd.points)
    key_ind = np.loadtxt(os.path.join(keyptspath, filename + '_Keypoints.txt'), dtype=np.int)
    keypts = pts[key_ind]
    return keypts


def get_keypts_(keyptspath, filename):
    keypts = np.load(os.path.join(keyptspath, filename + f'.keypts.bin.npy'))
    return keypts


def get_desc(descpath, filename, desc_name):
    if desc_name == '3dmatch':
        desc = np.fromfile(os.path.join(descpath, filename + '.desc.3dmatch.bin'), dtype=np.float32)
        num_desc = int(desc[0])
        desc_size = int(desc[1])
        desc = desc[2:].reshape([num_desc, desc_size])
    elif desc_name == 'LSD':
        desc = np.load(os.path.join(descpath, filename + '.desc.LSDNet.bin.npy'))
    elif desc_name == 'RIDE':
        desc = np.load(os.path.join(descpath, filename + '.desc.bin.npy'))
    else:
        print("No such descriptor")
        exit(-1)
    return desc


def loadlog(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans

    return result

def analyze_pointcloud_statistics(pcd, num_sample_points=1000, voxel_size=0.05):
    """
    Analyze the statistics of a given point cloud using a subset of points.

    Args:
        pcd (open3d.geometry.PointCloud): Open3D PointCloud object.
        num_sample_points (int): Number of points to sample for analysis.
        voxel_size (float): Voxel size for analysis.

    Returns:
        dict: A dictionary containing the overall statistics of the point cloud.
    """
    # Extract points as a NumPy array
    points = np.array(pcd.points)
    total_points = len(points)
    
    # Compute overall bounding box
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bounding_box = bbox_max - bbox_min
    scale = np.linalg.norm(bounding_box)

    # Voxelize point cloud
    voxel_indices = np.floor(points / voxel_size).astype(int)

    # Sample a subset of points
    sampled_indices = np.random.choice(len(points), size=min(num_sample_points, len(points)), replace=False)
    sampled_points = points[sampled_indices]

    # Create a dictionary mapping voxel indices to points
    voxel_to_points = {}
    for i, voxel_idx in enumerate(voxel_indices):
        voxel_key = tuple(voxel_idx)
        if voxel_key not in voxel_to_points:
            voxel_to_points[voxel_key] = []
        voxel_to_points[voxel_key].append(points[i])

    # Analyze each sampled point
    avg_distances_from_samples = []
    num_points_within_voxels = []

    for sample_point in sampled_points:
        # Compute voxel index for the sample point
        sample_voxel_index = tuple(np.floor(sample_point / voxel_size).astype(int))

        # Retrieve points in the same voxel
        same_voxel_points = np.array(voxel_to_points.get(sample_voxel_index, []))

        if len(same_voxel_points) > 1:  # Avoid division by zero
            # Compute distances only for points within the same voxel
            distances_from_sample = np.linalg.norm(same_voxel_points - sample_point, axis=1)
            # Find the 10 smallest distances (excluding the point itself)
            nearest_10_distances = np.partition(distances_from_sample, min(10, len(distances_from_sample) - 1))[:10]
            avg_sample_distance = np.mean(nearest_10_distances)
        else:
            avg_sample_distance = 0.0  # No other points in the voxel

        # Store the number of points in the voxel
        num_points_in_voxel = len(same_voxel_points)

        avg_distances_from_samples.append(avg_sample_distance)
        num_points_within_voxels.append(num_points_in_voxel)

    # Compute overall averages
    avg_distance_from_sample_within_voxel = np.mean(avg_distances_from_samples)
    avg_num_points_within_voxels = np.mean(num_points_within_voxels)

    return {
        "scale": scale,
        "avg_distance_from_sample_within_voxel": avg_distance_from_sample_within_voxel,
        "avg_num_points_within_voxels": avg_num_points_within_voxels,
        "total_points": total_points,
    }

def find_voxel_size(src_pcd, tgt_pcd):
    """
    Finds the voxel_size corresponding to the given target percentages of points within the radius.

    Args:
        src_pts (torch.Tensor): Source points of shape (N, 3).
        src_kpts (torch.Tensor): Keypoints of shape (num_keypts, 3).
        
    Returns:
        list: The calculated voxel_size for the given percentages.
    """
    src_pts = np.asarray(src_pcd.points)
    tgt_pts = np.asarray(tgt_pcd.points)
    
    if src_pts.shape[0] > tgt_pts.shape[0]:
        points = src_pts
    else:
        points = tgt_pts
        
    points_num = points.shape[0]
    sample_size = int(points_num / 10)
    sampled_indices = np.random.choice(points_num, size=sample_size, replace=False)
    
    sampled_points = points[sampled_indices]
    
    pca = PCA(n_components=3)
    pca.fit(sampled_points)
      
    transformed_points = pca.transform(points)
    x_range = transformed_points[:, 0].max() - transformed_points[:, 0].min()
    y_range = transformed_points[:, 1].max() - transformed_points[:, 1].min()
    z_range = transformed_points[:, 2].max() - transformed_points[:, 2].min()
    
    eigenvalues = pca.explained_variance_
    lambda1, lambda2, lambda3 = sorted(eigenvalues, reverse=True)
    # linearity = (lambda1 - lambda2) / lambda1
    planarity = (lambda2 - lambda3) / lambda1
    sphericity = lambda3 / lambda1
    
    if (sphericity < 0.05):
        alpha = 1.0
    else:
        alpha = 1.5

    voxel_size = np.sqrt(z_range) / 100 * alpha
    
    return round(voxel_size, 4), sphericity