import torch.utils.data as Data
import os
import open3d as o3d
import numpy as np
from utils.tools import sphericity_based_voxel_analysis


class ModelNet40Dataset(Data.Dataset):
    def __init__(self, split, config=None):
        self.config = config
        self.root = config.data.root
        self.split = split
        self.files = []
        self.poses = []
        self.length = 0
        self.prepare_matching_pairs(split=self.split)

    def load_modelnet_log(self):
        with open(os.path.join(self.root, "gt.log")) as f:
            content = f.readlines()
        logs = {}
        i = 0
        count = 0
        while i < len(content):
            line = content[i].replace("\n", "").split("\t")[0:3]
            src_id, tgt_id, object = line[0], line[1], line[2]
            trans = np.zeros([4, 4])
            trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
            trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
            trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
            trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
            i = i + 5
            logs[count] = (src_id, tgt_id, object, trans)
            count += 1
        return logs

    def prepare_matching_pairs(self, split="test"):
        logs = self.load_modelnet_log()
        for log in logs:
            src_id, tgt_id, object, T = logs[log]
            self.files.append([src_id, tgt_id, object])
            self.poses.append(T)

        self.length = len(self.files)

    def __getitem__(self, index):
        src_id, tgt_id, object = self.files[index]
        relt_pose = self.poses[index]

        # load src fragment
        src_path = os.path.join(self.root, object, src_id + ".ply")
        src_pcd = o3d.io.read_point_cloud(src_path)
        src_pcd.paint_uniform_color([1, 0.706, 0])

        # load tgt fragment
        tgt_path = os.path.join(self.root, object, tgt_id + ".ply")
        tgt_pcd = o3d.io.read_point_cloud(tgt_path)
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
        tgt_pts = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_pts)

        # voxel sampling
        ds_size = self.config.data.voxel_size_0
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=ds_size)
        src_kpt = np.array(src_pcd.points)
        np.random.shuffle(src_kpt)

        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=ds_size)
        tgt_kpt = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_kpt)

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
