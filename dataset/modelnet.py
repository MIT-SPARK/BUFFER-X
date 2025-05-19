import torch.utils.data as Data
import os
from os.path import join, exists
import pickle
import open3d as o3d
import glob
from utils.SE3 import *
from utils.tools import loadlog, sphericity_based_voxel_analysis
import copy

class RandomTransformSE3_euler:
    def __init__(self, rot_mag=45.0, trans_mag=0.5, random_mag=False):
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self):
        if self._random_mag:
            attenuation = np.random.random()
            rot_mag = attenuation * self._rot_mag
            trans_mag = attenuation * self._trans_mag
        else:
            rot_mag = self._rot_mag
            trans_mag = self._trans_mag

        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx, cosy, cosz = np.cos([anglex, angley, anglez])
        sinx, siny, sinz = np.sin([anglex, angley, anglez])
        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
        R = Rx @ Ry @ Rz
        t = np.random.uniform(-trans_mag, trans_mag, 3)
        return R, t

class ModelNet40Dataset(Data.Dataset):
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
        self.mode = 'clean'
        self.pose_generator = RandomTransformSE3_euler(rot_mag=45, trans_mag=0.5, random_mag=False)
        self.allowed_categories = self.load_category_list("/root/code/BUFFER-v2/config/splits/modelnet40_half2.txt")
        self.prepare_matching_pairs(split=self.split)
        
    def load_category_list(self, file_path):
        with open(file_path, 'r') as f:
            categories = [line.strip() for line in f if line.strip()]
        return set(categories)

    def prepare_matching_pairs(self, split='test'):
        for category in os.listdir(self.root):
            if category not in self.allowed_categories:
                continue
            category_path = os.path.join(self.root, category)
            if not os.path.isdir(category_path):
                continue

            for fname in os.listdir(category_path):
                if fname.endswith('.ply'):
                    relative_path = os.path.join(category, fname)
                    self.files.append(relative_path)

                    R, t = self.pose_generator.generate_transform()
                    T = np.eye(4, dtype=np.float32)
                    T[:3, :3] = R
                    T[:3, 3] = t
                    self.poses.append(T)

        self.length = len(self.files)
                
    def __getitem__(self, index):
        
        relative_path = self.files[index] 
        relt_pose = self.poses[index]     
        
        # load src fragment
        src_path = os.path.join(self.root, relative_path)
        src_pcd = o3d.io.read_point_cloud(src_path)
        src_pcd.paint_uniform_color([1, 0.706, 0])
        
        # load tgt fragment
        tgt_pcd = copy.deepcopy(src_pcd)
        tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
        tgt_pcd.transform(np.linalg.inv(relt_pose))
        
        self.mode == 'jitter'
        
        if self.mode == 'jitter':
            src_pcd = self.jitter(src_pcd)
            tgt_pcd = self.jitter(tgt_pcd)

        elif self.mode == 'partial':
            src_pcd = self.random_crop_halfspace(src_pcd, keep_ratio=0.7)
            tgt_pcd = self.random_crop_halfspace(tgt_pcd, keep_ratio=0.7)
            
            src_pcd = self.jitter(src_pcd)
            tgt_pcd = self.jitter(tgt_pcd)

        
        self.config.data.downsample, sphericity, is_aligned_to_global_z = sphericity_based_voxel_analysis(src_pcd, tgt_pcd)
        
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

        # # if we get too many points, we do random downsampling
        # if (src_kpt.shape[0] > 717):
        #     idx = np.random.choice(range(src_kpt.shape[0]), 717, replace=False)
        #     src_kpt = src_kpt[idx]

        # if (tgt_kpt.shape[0] > 717):
        #     idx = np.random.choice(range(tgt_kpt.shape[0]), 717, replace=False)
        #     tgt_kpt = tgt_kpt[idx]
            
        return {'src_fds_pts': src_pts, # first downsampling
                'tgt_fds_pts': tgt_pts,
                'relt_pose': relt_pose,
                'src_sds_pts': src_kpt, # second downsampling
                'tgt_sds_pts': tgt_kpt,
                'src_id': relative_path,
                'tgt_id': relative_path,
                'voxel_size': ds_size,
                'dataset_name': self.config.data.dataset,
                'sphericity': sphericity,
                'is_aligned_to_global_z': is_aligned_to_global_z,
                }
    def voxel_down_and_shuffle(self, pcd, voxel_size, max_pts=None):
        pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=voxel_size)
        pts = np.array(pcd.points)
        np.random.shuffle(pts)
        if max_pts and pts.shape[0] > max_pts:
            idx = np.random.choice(pts.shape[0], max_pts, replace=False)
            pts = pts[idx]
        return pts

    def jitter(self, pcd, std=0.01, clip=0.05):
        pts = np.array(pcd.points)
        noise = np.clip(np.random.normal(0.0, std, size=pts.shape), -clip, clip)
        pts += noise
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def random_crop_halfspace(pcd, keep_ratio=0.7):
        pts = np.asarray(pcd.points)
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        centered = pts - np.mean(pts, axis=0)
        dist = np.dot(centered, direction)
        threshold = np.percentile(dist, (1.0 - keep_ratio) * 100)
        mask = dist > threshold
        cropped_pts = pts[mask]
        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(cropped_pts)
        return cropped_pcd

    def __len__(self):
        return self.length