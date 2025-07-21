import torch.utils.data as Data
import os
from os.path import join
import pickle
import open3d as o3d
import numpy as np
import random
from utils.SE3 import rotation_matrix, integrate_trans
from utils.tools import loadlog, sphericity_based_voxel_analysis


class ThreeDMatchDataset(Data.Dataset):
    def __init__(self, split, config=None):
        self.config = config
        self.root = config.data.root
        self.split = split
        self.files = []
        self.length = 0
        self.prepare_matching_pairs(split=self.split)

    def prepare_matching_pairs(self, split="train"):
        if split != "test":
            self.root = join(self.root, "train")
            overlap_filename = join(self.root, "3DMatch_train_overlap.pkl")
            with open(overlap_filename, "rb") as file:
                self.overlap = pickle.load(file)
            print(f"Load PKL file from {overlap_filename}")

            self.scene_list = open(join(self.root, f"{split}_3dmatch.txt")).read().split()
            for key in self.overlap.keys():
                src_id, tgt_id = key.split("@")[0], key.split("@")[1]
                if src_id.split("/")[0] in self.scene_list:
                    self.files.append([src_id, tgt_id])
                    self.length += 1
        else:
            self.root = join(self.root, f"{split}")
            scene_list = [
                "7-scenes-redkitchen",
                "sun3d-home_at-home_at_scan1_2013_jan_1",
                "sun3d-home_md-home_md_scan9_2012_sep_30",
                "sun3d-hotel_uc-scan3",
                "sun3d-hotel_umd-maryland_hotel1",
                "sun3d-hotel_umd-maryland_hotel3",
                "sun3d-mit_76_studyroom-76-1studyroom2",
                "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika",
            ]
            self.poses = []
            for scene in scene_list:
                if self.config.data.benchmark == "3DMatch":
                    gtpath = self.root + f"/{self.config.data.benchmark}/gt_result/{scene}"
                elif self.config.data.benchmark == "3DLoMatch":
                    gtpath = self.root + f"/{self.config.data.benchmark}/{scene}"

                gtLog = loadlog(gtpath)
                pcdpath = f"3DMatch/fragments/{scene}"
                for key in gtLog.keys():
                    id1 = key.split("_")[0]
                    id2 = key.split("_")[1]
                    src_id = join(pcdpath, f"cloud_bin_{id1}")
                    tgt_id = join(pcdpath, f"cloud_bin_{id2}")
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

        tgt_path = os.path.join(self.root, tgt_id)
        tgt_pcd = o3d.io.read_point_cloud(tgt_path + ".ply")

        if self.split == "test":
            self.config.data.downsample, sphericity, _ = sphericity_based_voxel_analysis(
                src_pcd, tgt_pcd
            )
            is_aligned_to_global_z = self.config.patch.is_aligned_to_global_z
        else:
            sphericity = 0
            is_aligned_to_global_z = False

        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(
            src_pcd, voxel_size=self.config.data.downsample
        )
        src_pts = np.array(src_pcd.points)
        np.random.shuffle(src_pts)

        # load tgt fragment
        tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
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
            src_pose = np.load(f"{src_path}.pose.npy")
            tgt_pose = np.load(f"{tgt_path}.pose.npy")
            relt_pose = aug_trans @ np.matmul(np.linalg.inv(tgt_pose), src_pose)

            src_pts += (
                np.random.rand(src_pts.shape[0], 3) - 0.5
            ) * self.config.train.augmentation_noise
            tgt_pts += (
                np.random.rand(tgt_pts.shape[0], 3) - 0.5
            ) * self.config.train.augmentation_noise
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
