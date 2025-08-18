import torch.utils.data as Data
import os
import open3d as o3d
import glob
import numpy as np

from utils.common import make_open3d_point_cloud
from utils.tools import sphericity_based_voxel_analysis

curr_path = os.path.dirname(os.path.realpath(__file__))
split_path = curr_path + "/../config/splits"


class KAISTDataset(Data.Dataset):
    DATA_FILES = {"train": "train_kaist.txt", "val": "val_kaist.txt", "test": "test_kaist.txt"}

    def __init__(self, split, config=None):
        self.config = config
        self.pc_path = config.data.root
        self.split = split
        self.files = {"train": [], "val": [], "test": []}
        self.poses = []
        self.length = 0
        self.pdist = config.test.pdist
        self.kaist_cache = {}
        self.prepare_matching_pairs(split=self.split)

    def prepare_matching_pairs(self, split="train"):
        subset_names = open(os.path.join(split_path, self.DATA_FILES[split])).read().split()
        for dirname in subset_names:
            drive_id = str(dirname)
            fnames = glob.glob(str(self.pc_path) + "/%s/velodyne/*.bin" % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.pc_path} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))
            valid_pairs = pdist > self.pdist
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(valid_pairs[curr_time][curr_time : curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files[split].append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1

        self.length = len(self.files[split])

    def __getitem__(self, index):
        # load meta data
        drive = self.files[self.split][index][0]
        t0, t1 = self.files[self.split][index][1], self.files[self.split][index][2]

        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        relt_pose = np.linalg.inv(positions[1]) @ positions[0]

        # process point clouds
        src_pcd = make_open3d_point_cloud(xyz0, [1, 0.706, 0])
        tgt_pcd = make_open3d_point_cloud(xyz1, [0, 0.651, 0.929])

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
        tgt_pts = np.asarray(tgt_pcd.points)
        np.random.shuffle(tgt_pts)

        # second sample
        ds_size = self.config.data.voxel_size_0
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=ds_size)
        src_kpt = np.array(src_pcd.points)
        np.random.shuffle(src_kpt)

        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=ds_size)
        tgt_kpt = np.array(tgt_pcd.points)
        np.random.shuffle(tgt_kpt)

        # if we get too many points, we do random downsampling
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
            "voxel_size": ds_size,
            "src_id": "%s_%d" % (drive, t0),
            "tgt_id": "%s_%d" % (drive, t1),
            "dataset_name": self.config.data.dataset,
            "sphericity": sphericity,
            "is_aligned_to_global_z": is_aligned_to_global_z,
        }

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def get_video_odometry(self, drive, indices=None, ext=".txt", return_all=False):
        data_path = self.pc_path / f"{drive}" / "poses.txt"
        if data_path not in self.kaist_cache:
            self.kaist_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return self.kaist_cache[data_path]
        else:
            return self.kaist_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        fname = self.pc_path / f"{drive}" / "velodyne" / f"{t:06d}.bin"
        return fname

    def __len__(self):
        return self.length


class KAISTHeteroDataset(KAISTDataset):
    def __init__(self, split, config=None):
        # 'Aeva', 'Avia', 'Ouster'
        self.src_sensor = config.data.src_sensor
        self.tgt_sensor = config.data.tgt_sensor
        super().__init__(split, config)
        self.files = {"train": [], "val": [], "test": []}
        self.prepare_matching_pairs(split=self.split)

    def prepare_matching_pairs(self, split="train"):
        src_dir = os.path.join(self.pc_path, self.src_sensor, "velodyne")
        tgt_dir = os.path.join(self.pc_path, self.tgt_sensor, "velodyne")

        src_fnames = glob.glob(os.path.join(src_dir, "*.bin"))
        tgt_fnames = glob.glob(os.path.join(tgt_dir, "*.bin"))

        assert (
            len(src_fnames) > 0 and len(tgt_fnames) > 0
        ), f"Missing data in {src_dir} or {tgt_dir}"

        src_inames = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in src_fnames])
        tgt_inames = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in tgt_fnames])

        inames = sorted(set(src_inames) & set(tgt_inames))
        src_odo = self.get_video_odometry(self.src_sensor, return_all=True)
        tgt_odo = self.get_video_odometry(self.tgt_sensor, return_all=True)
        frame_to_index = {frame: idx for idx, frame in enumerate(inames)}

        min_len = min(len(inames), len(src_odo), len(tgt_odo))
        inames = inames[:min_len]

        src_positions = np.array(
            [self.odometry_to_positions(src_odo[frame_to_index[f]]) for f in inames]
        )
        tgt_positions = np.array(
            [self.odometry_to_positions(tgt_odo[frame_to_index[f]]) for f in inames]
        )

        Ts_src = src_positions[:, :3, 3]
        Ts_tgt = tgt_positions[:, :3, 3]

        pdist = np.linalg.norm(Ts_src[:, None, :] - Ts_tgt[None, :, :], axis=-1)
        valid_pairs = pdist > self.pdist

        curr_time = inames[0]
        while curr_time in inames:
            curr_idx = frame_to_index[curr_time]
            next_mask = valid_pairs[curr_idx][curr_idx : curr_idx + 100]
            next_offset = np.where(next_mask)[0]

            if len(next_offset) == 0:
                curr_time += 1
                continue

            next_idx = curr_idx + next_offset[0]
            if next_idx >= len(inames):
                break

            next_time = inames[next_idx]
            self.files[split].append((self.src_sensor, self.tgt_sensor, curr_time, next_time))
            curr_time = next_time + 1
        self.length = len(self.files[split])

    def __getitem__(self, index):
        src_sensor, tgt_sensor, t0, t1 = self.files[self.split][index]

        src_pose = self.odometry_to_positions(self.get_video_odometry(src_sensor, [t0])[0])
        tgt_pose = self.odometry_to_positions(self.get_video_odometry(tgt_sensor, [t1])[0])
        relt_pose = np.linalg.inv(tgt_pose) @ src_pose

        fname0 = self._get_velodyne_fn(src_sensor, t0)
        fname1 = self._get_velodyne_fn(tgt_sensor, t1)

        # XYZ and reflectance
        xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
        xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

        xyz0 = xyzr0[:, :3]
        xyz1 = xyzr1[:, :3]

        src_pcd = make_open3d_point_cloud(xyz0, [1, 0.706, 0])
        tgt_pcd = make_open3d_point_cloud(xyz1, [0, 0.651, 0.929])

        self.config.data.downsample, sphericity, _ = sphericity_based_voxel_analysis(
            src_pcd, tgt_pcd
        )
        is_aligned_to_global_z = self.config.patch.is_aligned_to_global_z
        src_pcd = src_pcd.voxel_down_sample(voxel_size=self.config.data.downsample)
        tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size=self.config.data.downsample)
        src_pts = np.array(src_pcd.points)
        tgt_pts = np.array(tgt_pcd.points)
        np.random.shuffle(src_pts)
        np.random.shuffle(tgt_pts)

        # Second downsampling
        ds_size = self.config.data.voxel_size_0
        src_kpt = np.array(src_pcd.voxel_down_sample(ds_size).points)
        tgt_kpt = np.array(tgt_pcd.voxel_down_sample(ds_size).points)
        np.random.shuffle(src_kpt)
        np.random.shuffle(tgt_kpt)

        if src_kpt.shape[0] > self.config.data.max_numPts:
            src_kpt = src_kpt[
                np.random.choice(len(src_kpt), self.config.data.max_numPts, replace=False)
            ]
        if tgt_kpt.shape[0] > self.config.data.max_numPts:
            tgt_kpt = tgt_kpt[
                np.random.choice(len(tgt_kpt), self.config.data.max_numPts, replace=False)
            ]

        return {
            "src_fds_pts": src_pts,
            "tgt_fds_pts": tgt_pts,
            "relt_pose": relt_pose,
            "src_sds_pts": src_kpt,
            "tgt_sds_pts": tgt_kpt,
            "voxel_size": ds_size,
            "src_id": f"{t0}",
            "tgt_id": f"{t1}",
            "dataset_name": self.config.data.dataset,
            "sphericity": sphericity,
            "is_aligned_to_global_z": is_aligned_to_global_z,
            "sensor": f"{src_sensor}->{tgt_sensor}",
        }
