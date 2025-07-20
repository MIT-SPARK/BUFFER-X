import torch.utils.data as Data
import os
from os.path import join, exists
import pickle
import open3d as o3d
import glob
from utils.SE3 import *
from utils.tools import loadlog, sphericity_based_voxel_analysis

class ETHDataset(Data.Dataset):
    def __init__(self,
                 split,
                 config=None
                 ):
        self.config = config
        self.root = config.data.root
        self.split = split
        self.files = []
        self.length = 0
        self.prepare_matching_pairs(split=self.split)
        
    def prepare_matching_pairs(self, split='train'):

        scene_list = [
            'gazebo_summer',
            'gazebo_winter',
            'wood_autmn',
            'wood_summer',
        ]
        self.poses = []
        for scene in scene_list:
            pcdpath = f'{scene}/'
            gtpath = self.root + f'/{scene}/'
            gtLog = loadlog(gtpath)
            for key in gtLog.keys():
                id1 = key.split('_')[0]
                id2 = key.split('_')[1]
                src_id = join(pcdpath, f'Hokuyo_{id1}')
                tgt_id = join(pcdpath, f'Hokuyo_{id2}')
                self.files.append([src_id, tgt_id])
                self.length += 1
                self.poses.append(gtLog[key])

    def __getitem__(self, index):

        # load meta data
        src_id, tgt_id = self.files[index][0], self.files[index][1]
        # load src fragment
        src_path = os.path.join(self.root, src_id)
        src_pcd = o3d.io.read_point_cloud(src_path + '.ply')
        src_pcd.paint_uniform_color([1, 0.706, 0])
        
        # load tgt fragment
        tgt_path = os.path.join(self.root, tgt_id)
        tgt_pcd = o3d.io.read_point_cloud(tgt_path + '.ply')
        tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
        
        self.config.data.downsample, sphericity, _ = sphericity_based_voxel_analysis(src_pcd, tgt_pcd)
        is_aligned_to_global_z = self.config.patch.is_aligned_to_global_z
        
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=self.config.data.downsample)
        src_pts = np.array(src_pcd.points)
        np.random.shuffle(src_pts)
        
        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=self.config.data.downsample)
        tgt_pts = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_pts)
        
        # relative pose
        relt_pose = np.linalg.inv(self.poses[index])

        # voxel sampling
        ds_size = self.config.data.voxel_size_0
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=ds_size)
        src_kpt = np.array(src_pcd.points)
        np.random.shuffle(src_kpt)
        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=ds_size)
        tgt_kpt = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_kpt)

        # if we get too many points, we do random downsampling
        if (src_kpt.shape[0] > self.config.data.max_numPts):
            idx = np.random.choice(range(src_kpt.shape[0]), self.config.data.max_numPts, replace=False)
            src_kpt = src_kpt[idx]

        if (tgt_kpt.shape[0] > self.config.data.max_numPts):
            idx = np.random.choice(range(tgt_kpt.shape[0]), self.config.data.max_numPts, replace=False)
            tgt_kpt = tgt_kpt[idx]
            
        return {'src_fds_pts': src_pts, # first downsampling
                'tgt_fds_pts': tgt_pts,
                'relt_pose': relt_pose,
                'src_sds_pts': src_kpt, # second downsampling
                'tgt_sds_pts': tgt_kpt,
                'src_id': src_id,
                'tgt_id': tgt_id,
                'voxel_size': ds_size,
                'dataset_name': self.config.data.dataset,
                'sphericity': sphericity,
                'is_aligned_to_global_z': is_aligned_to_global_z,
                }


    def __len__(self):
        return self.length