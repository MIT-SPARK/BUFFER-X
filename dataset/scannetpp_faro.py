import torch.utils.data as Data
import os
from os.path import join
import open3d as o3d
import numpy as np
import random
from utils.SE3 import rotation_matrix, integrate_trans
from utils.tools import loadlog, sphericity_based_voxel_analysis

curr_path = os.path.dirname(os.path.realpath(__file__))
split_path = curr_path + "/../config/splits"


class ScannetppFaroDataset(Data.Dataset):
    def __init__(self, split, config=None):
        self.config = config
        self.root = config.data.root
        self.split = split
        self.files = []
        self.length = 0
        self.prepare_matching_pairs(split=self.split)

    def prepare_matching_pairs(self, split="train"):
        self.scene_list = open(join(split_path, f"{self.split}_scannetpp_faro.txt")).read().split()

        self.poses = []
        for scene in self.scene_list:
            gtpath = self.root / "data" / f"{scene}" / "scans"
            gtLog = loadlog(gtpath)
            for key in gtLog.keys():
                id1 = key.split("_")[0]
                id2 = key.split("_")[1]
                src_id = f"data/{scene}/scans/trans_faro_1800x900_scanner_{id1}"
                tgt_id = f"data/{scene}/scans/trans_faro_1800x900_scanner_{id2}"
                self.files.append([src_id, tgt_id])
                self.length += 1
                self.poses.append(gtLog[key])

    def __getitem__(self, index):
        # load meta data
        src_id, tgt_id = self.files[index][0], self.files[index][1]
        if self.split != "test":
            if random.random() > 0.5:
                src_id, tgt_id = tgt_id, src_id

        # load src fragment
        src_path = os.path.join(self.root, src_id)
        src_pcd = o3d.io.read_point_cloud(src_path + ".ply")
        src_pcd.paint_uniform_color([1, 0.706, 0])

        # load tgt fragment
        tgt_path = os.path.join(self.root, tgt_id)
        tgt_pcd = o3d.io.read_point_cloud(tgt_path + ".ply")
        tgt_pcd.paint_uniform_color([0, 0.651, 0.929])

        self.config.data.downsample, sphericity, _ = sphericity_based_voxel_analysis(
            src_pcd, tgt_pcd
        )
        is_aligned_to_global_z = self.config.patch.is_aligned_to_global_z
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(
            src_pcd, voxel_size=self.config.data.downsample
        )
        src_pts = np.array(src_pcd.points)
        np.random.shuffle(src_pts)

        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(
            tgt_pcd, voxel_size=self.config.data.downsample
        )

        if self.split != "test":
            # SO(3) augmentation
            R = rotation_matrix(3, 1)
            t = np.zeros([3, 1])  # np.random.random([3, 1]) * np.random.randint(100)#
            aug_trans = integrate_trans(R, t)
            tgt_pcd.transform(aug_trans)
        tgt_pts = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_pts)

        # relative pose
        if self.split != "test":
            relt_pose = aug_trans @ self.poses[index]
            src_pts += (
                np.random.rand(src_pts.shape[0], 3) - 0.5
            ) * self.config.train.augmentation_noise
            tgt_pts += (
                np.random.rand(tgt_pts.shape[0], 3) - 0.5
            ) * self.config.train.augmentation_noise
        else:
            relt_pose = self.poses[index]

        # second sample
        ds_size = self.config.data.voxel_size_0
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=ds_size)
        src_kpt = np.array(src_pcd.points)
        np.random.shuffle(src_kpt)

        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=ds_size)
        tgt_kpt = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_kpt)

        # if we get too many points, we do some downsampling
        if src_kpt.shape[0] > self.config.data.max_numPts:
            idx = np.random.choice(
                range(src_kpt.shape[0]), self.config.data.max_numPts, replace=False
            )
            src_kpt = src_kpt[idx]

        if tgt_kpt.shape[0] > self.config.data.max_numPts:
            idx = np.random.choice(
                range(tgt_kpt.shape[0]), self.config.data.max_numPts, replace=False
            )
            tgt_kpt = tgt_kpt[idx]

        return {
            "src_fds_pts": src_pts,  # first downsampling
            "tgt_fds_pts": tgt_pts,
            "relt_pose": relt_pose,
            "src_sds_pts": src_kpt,  # second downsampling
            "tgt_sds_pts": tgt_kpt,
            "src_id": src_id,
            "tgt_id": tgt_id,
            "voxel_size": ds_size,
            "dataset_name": self.config.data.dataset,
            "sphericity": sphericity,
            "is_aligned_to_global_z": is_aligned_to_global_z,
        }

    def __len__(self):
        return self.length
