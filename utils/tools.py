import os
import open3d
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import time
import nibabel.quaternions as nq

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