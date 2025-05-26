import os
import open3d
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import time
import nibabel.quaternions as nq
import logging

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

def read_trajectory(filename, dim=4):
    with open(filename) as f:
        lines = f.readlines()

    keys = lines[0::(dim + 1)]
    final_keys = [[k.split('\t')[0].strip(), k.split('\t')[1].strip(), k.split('\t')[2].strip()] for k in keys]
    traj = [lines[i].split('\t')[0:dim] for i in range(len(lines)) if i % (dim + 1) != 0]
    return np.asarray(final_keys), np.asarray(traj, dtype=np.float32).reshape(-1, dim, dim)

def read_trajectory_info(filename, dim=6):
    with open(filename) as fid:
        contents = fid.readlines()
    n_pairs = len(contents) // 7
    info_list = []
    for i in range(n_pairs):
        info_matrix = np.vstack([np.fromstring(contents[j], sep='\t').reshape(1, -1) for j in range(i * 7 + 1, i * 7 + 7)])
        info_list.append(info_matrix)
    return int(contents[0].strip().split()[2]), np.asarray(info_list, dtype=np.float32).reshape(-1, dim, dim)

def computeTransformationErr(trans, info):
    t, r = trans[:3, 3], trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    return (er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]).item()

def evaluate_registration(num_fragment, result, result_pairs, gt_pairs, gt, gt_info, err2=0.2):
    err2 = err2 ** 2
    gt_mask = np.zeros((num_fragment, num_fragment), dtype=np.int64)
    flags, transformation_errors = [], np.full(result_pairs.shape[0], np.nan)

    for idx in range(gt_pairs.shape[0]):
        i, j = int(gt_pairs[idx, 0]), int(gt_pairs[idx, 1])
        if j - i > 1:
            gt_mask[i, j] = idx

    good, n_res, n_gt = 0, 0, np.sum(gt_mask > 0)
    for idx in range(result_pairs.shape[0]):
        i, j, pose = int(result_pairs[idx, 0]), int(result_pairs[idx, 1]), result[idx, :, :]
        if gt_mask[i, j] > 0:
            n_res += 1
            gt_idx = gt_mask[i, j]
            p = computeTransformationErr(np.linalg.inv(gt[gt_idx]) @ pose, gt_info[gt_idx])
            transformation_errors[idx] = p
            if p <= err2:
                good += 1
                flags.append(0)
            else:
                flags.append(1)
        else:
            flags.append(2)
    return good / max(n_res, 1e-6), good / n_gt, flags, transformation_errors

def compute_pca_alignment(pcd):
        pts = np.asarray(pcd.points)
        num_points = pts.shape[0]
        sample_size = int(num_points / 10)
        sampled_pts = pts[np.random.choice(num_points, size=sample_size, replace=False)]

        pca = PCA(n_components=3)
        pca.fit(sampled_pts)
        eigenvalues = pca.explained_variance_
        lambda1, lambda2, lambda3 = sorted(eigenvalues, reverse=True)
        sphericity = lambda3 / lambda1

        z_axis_candidate = pca.components_[-1]
        z_axis_candidate = z_axis_candidate / np.linalg.norm(z_axis_candidate)
        z_alignment_score = np.abs(np.dot(z_axis_candidate, np.array([0, 0, 1])))
        is_aligned = z_alignment_score > 0.98

        return sphericity, is_aligned, pca

def sphericity_based_voxel_analysis(src_pcd, tgt_pcd):
    """
    Estimates voxel size and sphericity, and checks if both source and target point clouds
    are aligned with the global z-axis using PCA.

    Args:
        src_pcd (open3d.geometry.PointCloud): Source point cloud.
        tgt_pcd (open3d.geometry.PointCloud): Target point cloud.

    Returns:
        voxel_size (float): Estimated voxel size.
        sphericity (float): Sphericity of the denser point cloud.
        is_aligned_to_global_z (bool): True if both clouds are aligned to global z-axis.
    """

    # PCA for both src and tgt
    sphericity_src, is_src_aligned, pca_src = compute_pca_alignment(src_pcd)
    sphericity_tgt, is_tgt_aligned, pca_tgt = compute_pca_alignment(tgt_pcd)

    # Use denser point cloud for voxel size estimation
    if len(src_pcd.points) > len(tgt_pcd.points):
        ref_pcd = src_pcd
        sphericity = sphericity_src
        pca = pca_src
    else:
        ref_pcd = tgt_pcd
        sphericity = sphericity_tgt
        pca = pca_tgt

    transformed_pts = pca.transform(np.asarray(ref_pcd.points))
    z_range = transformed_pts[:, 2].max() - transformed_pts[:, 2].min()
    alpha = 1.0 if sphericity < 0.05 else 1.5
    voxel_size = np.sqrt(z_range) / 100 * alpha
    voxel_size = max(voxel_size, 0.001)
    
    z_src = pca_src.components_[-1]
    z_tgt = pca_tgt.components_[-1]

    z_src /= np.linalg.norm(z_src)
    z_tgt /= np.linalg.norm(z_tgt)

    dot_z = np.dot(z_src, z_tgt)
    same_direction = dot_z > 0.96

    is_aligned_to_global_z = is_src_aligned and is_tgt_aligned and same_direction

    return round(voxel_size, 4), sphericity, is_aligned_to_global_z

def setup_logger(log_path=None):
    logger = logging.getLogger("BUFFER-X")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_path and not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path) for h in logger.handlers):
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
