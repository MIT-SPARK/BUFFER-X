import torch.utils.data as Data
import os
from os.path import join, exists
import pickle
import open3d as o3d
import glob
from utils.SE3 import *
from utils.tools import get_pcd, get_keypts, loadlog
from utils.common import make_open3d_point_cloud
import copy
import gc
import numpy as np
import json


def get_matching_indices(source, target, relt_pose, search_voxel_size):
    source = transform(source, relt_pose)
    diffs = source[:, None] - target[None]
    dist = np.sqrt(np.sum(diffs ** 2, axis=-1) + 1e-12)
    min_ind = np.concatenate([np.arange(source.shape[0])[:, None], np.argmin(dist, axis=1)[:, None]], axis=-1)
    min_val = np.min(dist, axis=1)
    match_inds = min_ind[min_val < search_voxel_size]

    return match_inds


class NSSDataset(Data.Dataset):
    def __init__(self,
                 split,
                 config=None
                 ):
        self.config = config
        self.root = config.data.root
        self.split = split
        self.files = []
        self.poses = []
        self.length = 0
        
        self.graph_root = os.path.join(self.root, 'pose_graphs/original')
        self.pcd_root = os.path.join(self.root, 'point_cloud')
        pose_graph_path = os.path.join(self.graph_root, f'{split}.pose_graph.json')
        
         # Load pose graph JSON file
        with open(pose_graph_path, 'r') as file:
            self.pose_json = json.load(file)
        print(f"Loaded pose graph from {pose_graph_path}")

        # Extract files and poses from the JSON
        for pose in self.pose_json:
            nodes = pose['nodes']
            edges = pose['edges']
            node_id_to_index = {node['id']: index for index, node in enumerate(nodes)}

            for edge in edges:
                src_id = edge['source']
                tgt_id = edge['target']
                relt_pose = np.array(edge['tsfm'])
                if np.isnan(relt_pose).any(): 
                    print(f"Error: relt_pose is nan")
                    breakpoint()
                    continue  # Skip this pair if pose is nan
               # Check if the source and target IDs exist in the node mapping
                if src_id not in node_id_to_index or tgt_id not in node_id_to_index:
                    print(f"Error: src_id {src_id} or tgt_id {tgt_id} is out of range for nodes with available IDs: {list(node_id_to_index.keys())}")
                    breakpoint()
                    continue  # Skip this pair if IDs are out of range

                src_index = node_id_to_index[src_id]
                tgt_index = node_id_to_index[tgt_id]
                src_pcd_name = nodes[src_index]['name']
                tgt_pcd_name = nodes[tgt_index]['name']

                self.files.append([src_pcd_name, tgt_pcd_name])
                self.poses.append(relt_pose)
                self.length += 1
                
    def __getitem__(self, index):

        src_name, tgt_name = self.files[index]
        src_path, tgt_path= os.path.join(self.pcd_root, src_name), os.path.join(self.pcd_root, tgt_name)
        relt_pose = self.poses[index]
        
        # Load source point cloud
        src_pcd = o3d.io.read_point_cloud(src_path)
        src_pcd.paint_uniform_color([1, 0.706, 0])
        src_pcd = src_pcd.voxel_down_sample(voxel_size=self.config.data.downsample)
        src_pts = np.array(src_pcd.points)

        # Load target point cloud
        tgt_pcd = o3d.io.read_point_cloud(tgt_path)
        tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
        tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size=self.config.data.downsample)
        tgt_pts = np.array(tgt_pcd.points)


        if self.split != 'test':
            R = rotation_matrix(3, 1)  # Random SO(3) rotation
            t = np.zeros([3, 1])  # No translation augmentation applied here
            aug_trans = integrate_trans(R, t)
            tgt_pcd.transform(aug_trans)
            tgt_pts = np.array(tgt_pcd.points)
            np.random.shuffle(tgt_pts)

            # Noise augmentation
            src_pts += (np.random.rand(src_pts.shape[0], 3) - 0.5) * self.config.train.augmentation_noise
            tgt_pts += (np.random.rand(tgt_pts.shape[0], 3) - 0.5) * self.config.train.augmentation_noise

            # Relative pose considering augmentation
            relt_pose = aug_trans @ relt_pose
            
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
                'src_id': src_name,
                'tgt_id': tgt_name,
                'voxel_size': ds_size,
                'dataset_name': self.config.data.dataset}

    def __len__(self):
        return self.length
