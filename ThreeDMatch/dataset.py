import torch.utils.data as Data
import os
from os.path import join, exists
import pickle
import open3d as o3d
import glob
from utils.SE3 import *
from utils.tools import get_pcd, get_keypts, loadlog, analyze_pointcloud_statistics, find_voxel_size
from utils.common import make_open3d_point_cloud
import copy
import gc
import pointnet2_ops.pointnet2_utils as pnt2

def get_matching_indices(source, target, relt_pose, search_voxel_size):
    source = transform(source, relt_pose)
    diffs = source[:, None] - target[None]
    dist = np.sqrt(np.sum(diffs ** 2, axis=-1) + 1e-12)
    min_ind = np.concatenate([np.arange(source.shape[0])[:, None], np.argmin(dist, axis=1)[:, None]], axis=-1)
    min_val = np.min(dist, axis=1)
    match_inds = min_ind[min_val < search_voxel_size]

    return match_inds

def calculate_scale_with_percentile(pcd, low_percentile=2, high_percentile=98):
    """
    Calculate the scale of a point cloud using percentiles to exclude outliers.
    
    Args:
        pcd (numpy.ndarray): Point cloud array of shape (N, 3).
        low_percentile (float): Lower percentile for excluding outliers.
        high_percentile (float): Upper percentile for excluding outliers.
        
    Returns:
        float: The calculated scale of the point cloud.
    """
    # Compute the minimum and maximum coordinates within the percentile range
    min_coords = np.percentile(pcd, low_percentile, axis=0)
    max_coords = np.percentile(pcd, high_percentile, axis=0)
    
    # Compute the bounding box size
    bounding_box_size = max_coords - min_coords
    
    # Calculate the scale as the norm of the bounding box size
    scale = np.linalg.norm(bounding_box_size)
    return scale

class ThreeDMatchDataset(Data.Dataset):
    def __init__(self,
                 split,
                 config=None
                 ):
        self.config = config
        self.root = config.data.root
        self.split = split
        self.files = []
        self.length = 0

        if split != 'test':
            self.root = join(self.root, 'train')
            overlap_filename = join(self.root, f'3DMatch_train_overlap.pkl')
            with open(overlap_filename, 'rb') as file:
                self.overlap = pickle.load(file)
            print(f"Load PKL file from {overlap_filename}")

            self.scene_list = open(join(self.root, f'{self.split}_3dmatch.txt')).read().split()
            for key in self.overlap.keys():
                src_id, tgt_id = key.split('@')[0], key.split('@')[1]
                if src_id.split("/")[0] in self.scene_list:
                    self.files.append([src_id, tgt_id])
                    self.length += 1
        else:
            self.root = join(self.root, f'{split}')
            scene_list = [
                '7-scenes-redkitchen',
                'sun3d-home_at-home_at_scan1_2013_jan_1',
                'sun3d-home_md-home_md_scan9_2012_sep_30',
                'sun3d-hotel_uc-scan3',
                'sun3d-hotel_umd-maryland_hotel1',
                'sun3d-hotel_umd-maryland_hotel3',
                'sun3d-mit_76_studyroom-76-1studyroom2',
                'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
            ]
            self.poses = []
            for scene in scene_list:
                if config.data.dataset == '3DMatch':
                    gtpath = self.root + f'/{config.data.dataset}/gt_result/{scene}'
                elif config.data.dataset == '3DLoMatch':
                    gtpath = self.root + f'/{config.data.dataset}/{scene}'

                gtLog = loadlog(gtpath)
                pcdpath = f'3DMatch/fragments/{scene}'
                for key in gtLog.keys():
                    id1 = key.split('_')[0]
                    id2 = key.split('_')[1]
                    src_id = join(pcdpath, f'cloud_bin_{id1}')
                    tgt_id = join(pcdpath, f'cloud_bin_{id2}')
                    self.files.append([src_id, tgt_id])
                    self.length += 1
                    self.poses.append(gtLog[key])


    def __getitem__(self, index):

        # load meta data
        src_id, tgt_id = self.files[index][0], self.files[index][1]

        if self.split != 'test':
            if random.random() > 0.5:
                src_id, tgt_id = tgt_id, src_id
        
        # load src fragment
        src_path = os.path.join(self.root, src_id)
        src_pcd = o3d.io.read_point_cloud(src_path + '.ply')
        src_pcd.paint_uniform_color([1, 0.706, 0])
        # results = analyze_pointcloud_statistics(src_pcd, num_sample_points=1000, voxel_size=0.015)
        results = None
        
        self.config.data.downsample = find_voxel_size(src_pcd)
        
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=self.config.data.downsample)
        src_pts = np.array(src_pcd.points)
        np.random.shuffle(src_pts)

        # load tgt fragment
        tgt_path = os.path.join(self.root, tgt_id)
        tgt_pcd = o3d.io.read_point_cloud(tgt_path + '.ply')
        tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=self.config.data.downsample)
        
        ## Uniform downsample version
        # load src fragment
        # src_path = os.path.join(self.root, src_id)
        # src_pcd = o3d.io.read_point_cloud(src_path + '.ply')
    
        # src_pts = np.asarray(src_pcd.points)
        # src_indices = np.arange(src_pts.shape[0])
        # np.random.shuffle(src_indices)
        # src_pts = src_pts[src_indices]
        # src_pcd = make_open3d_point_cloud(src_pts)
        # src_pcd.paint_uniform_color([1, 0.706, 0])
        # src_pcd = o3d.geometry.PointCloud.uniform_down_sample(src_pcd, 10)
        # src_pts = np.array(src_pcd.points)

        # # load tgt fragment
        # tgt_path = os.path.join(self.root, tgt_id)
        # tgt_pcd = o3d.io.read_point_cloud(tgt_path + '.ply')

        # tgt_pts = np.asarray(tgt_pcd.points)
        # tgt_indices = np.arange(tgt_pts.shape[0])
        # np.random.shuffle(tgt_indices)
        # tgt_pts = tgt_pts[tgt_indices]
        # tgt_pcd = make_open3d_point_cloud(tgt_pts)
        # tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
        # tgt_pcd = o3d.geometry.PointCloud.uniform_down_sample(tgt_pcd, 10)
        # tgt_pts = np.array(tgt_pcd.points)

        if self.split != 'test':
            # SO(3) augmentation
            R = rotation_matrix(3, 1)
            t = np.zeros([3, 1])#np.random.random([3, 1]) * np.random.randint(100)#
            aug_trans = integrate_trans(R, t)
            tgt_pcd.transform(aug_trans)
        tgt_pts = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_pts)
        
        # Calculate scale for source and target point clouds
        src_scale = calculate_scale_with_percentile(src_pts)
        tgt_scale = calculate_scale_with_percentile(tgt_pts)

        # relative pose
        if self.split != 'test':
            src_pose = np.load(f'{src_path}.pose.npy')
            tgt_pose = np.load(f'{tgt_path}.pose.npy')
            relt_pose = aug_trans @ np.matmul(np.linalg.inv(tgt_pose), src_pose)

            src_pts += (np.random.rand(src_pts.shape[0], 3) - 0.5) * self.config.train.augmentation_noise
            tgt_pts += (np.random.rand(tgt_pts.shape[0], 3) - 0.5) * self.config.train.augmentation_noise
        else:
            relt_pose = np.linalg.inv(self.poses[index])

        # second sample
        ds_size = self.config.data.voxel_size_0
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=ds_size)
        src_kpt = np.array(src_pcd.points)
        np.random.shuffle(src_kpt) 

        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=ds_size)
        tgt_kpt = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_kpt) 

        # if we get too many points, we do some downsampling
        if (src_kpt.shape[0] > self.config.data.max_numPts):
            idx = np.random.choice(range(src_kpt.shape[0]), self.config.data.max_numPts, replace=False)
            src_kpt = src_kpt[idx]

        if (tgt_kpt.shape[0] > self.config.data.max_numPts):
            idx = np.random.choice(range(tgt_kpt.shape[0]), self.config.data.max_numPts, replace=False)
            tgt_kpt = tgt_kpt[idx]

        if self.split == 'test':
            src_pcd = make_open3d_point_cloud(src_kpt)
            src_pcd.estimate_normals()
            src_pcd.orient_normals_towards_camera_location()
            src_noms = np.array(src_pcd.normals)
            src_kpt = np.concatenate([src_kpt, src_noms], axis=-1)

            tgt_pcd = make_open3d_point_cloud(tgt_kpt)
            tgt_pcd.estimate_normals()
            tgt_pcd.orient_normals_towards_camera_location()
            tgt_noms = np.array(tgt_pcd.normals)
            tgt_kpt = np.concatenate([tgt_kpt, tgt_noms], axis=-1)


        return {'src_fds_pts': src_pts, # first downsampling
                'tgt_fds_pts': tgt_pts,
                'relt_pose': relt_pose,
                'src_sds_pts': src_kpt, # second downsampling
                'tgt_sds_pts': tgt_kpt,
                'src_scale': src_scale,
                'tgt_scale': tgt_scale,
                'src_id': src_id,
                'tgt_id': tgt_id,
                'voxel_size': ds_size,
                'dataset_name': self.config.data.dataset,
                'results': results
                }

    def __len__(self):
        return self.length
