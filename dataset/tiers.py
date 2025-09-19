import torch.utils.data as Data
import os
import open3d as o3d
import glob
import numpy as np
import pickle
from utils.SE3 import rotation_matrix, integrate_trans
from utils.common import make_open3d_point_cloud
from utils.tools import sphericity_based_voxel_analysis, compute_overlap_ratio

curr_path = os.path.dirname(os.path.realpath(__file__))
split_path = curr_path + "/../config/splits"


class TIERSDataset(Data.Dataset):
    DATA_FILES = {"train": "train_tiers.txt", "val": "val_tiers.txt", "test": "test_tiers.txt"}

    def __init__(self, split, config=None):
        self.config = config
        self.pc_path = config.data.root
        self.split = split
        self.files = {"train": [], "val": [], "test": []}
        self.poses = []
        self.length = 0
        self.pdist = config.test.pdist
        self.tiers_cache = {}
        self.prepare_matching_pairs(split=self.split)

    def prepare_matching_pairs(self, split="train"):
        subset_names = open(os.path.join(split_path, self.DATA_FILES[split])).read().split()
        for dirname in subset_names:
            drive_id = str(dirname)
            sensor_types = os.listdir(os.path.join(self.pc_path, dirname))
            for sensor in sensor_types:
                fnames = glob.glob(str(self.pc_path / f"{drive_id}" / f"{sensor}/scans/*.pcd"))
                assert len(fnames) > 0, f"Make sure that the path {self.pc_path} has data {dirname}"
                inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

                all_odo = self.get_video_odometry(drive_id, sensor, return_all=True)
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
                        self.files[split].append((drive_id, sensor, curr_time, next_time))
                        curr_time = next_time + 1

            self.length = len(self.files[split])

    def __getitem__(self, index):
        # load meta data
        drive = self.files[self.split][index][0]
        sensor = self.files[self.split][index][1]
        t0, t1 = self.files[self.split][index][2], self.files[self.split][index][3]

        all_odometry = self.get_video_odometry(drive, sensor, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, sensor, t0)
        fname1 = self._get_velodyne_fn(drive, sensor, t1)

        # XYZ and reflectance
        o3d_cloud0 = o3d.io.read_point_cloud(fname0)
        o3d_cloud1 = o3d.io.read_point_cloud(fname1)
        xyz0 = np.asarray(o3d_cloud0.points, dtype=np.float32)
        xyz1 = np.asarray(o3d_cloud1.points, dtype=np.float32)

        # Note (Minkyun Seo):
        # Above code is commented out because it does not work well for the tiers dataset.
        trans = np.linalg.inv(positions[1]) @ positions[0]
        if self.split != "test":
            xyz0 += (np.random.rand(xyz0.shape[0], 3) - 0.5) * self.config.train.augmentation_noise
            xyz1 += (np.random.rand(xyz1.shape[0], 3) - 0.5) * self.config.train.augmentation_noise

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

        if self.split != "test":
            # SO(2) augmentation
            R = rotation_matrix(1, 1)
            t = np.zeros([3, 1])
            aug_trans = integrate_trans(R, t)
            tgt_pcd.transform(aug_trans)
            relt_pose = aug_trans @ trans
        else:
            relt_pose = trans

        tgt_pts = np.array(tgt_pcd.points)
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
            "sensor": sensor,
        }

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    def get_video_odometry(self, drive, sensor, indices=None, ext=".txt", return_all=False):
        data_path = self.pc_path / f"{drive}" / f"{sensor}" / "poses_kitti.txt"
        if data_path not in self.tiers_cache:
            self.tiers_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return self.tiers_cache[data_path]
        else:
            return self.tiers_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def _get_velodyne_fn(self, drive, sensor, t):
        fname = self.pc_path / f"{drive}" / f"{sensor}" / "scans" / f"{t:06d}.pcd"
        return str(fname)

    def __len__(self):
        return self.length


class TIERSHeteroDataset(TIERSDataset):
    def __init__(self, split, config=None):
        # 'os0_128', 'os1_64', 'vel16'
        self.src_sensor = config.data.src_sensor
        self.tgt_sensor = config.data.tgt_sensor
        self.overlap_voxel_size = config.test.overlap_voxel_size
        self.overlap_thresh = config.test.overlap_thresh
        super().__init__(split, config)
        self.files = {"train": [], "val": [], "test": []}
        self.prepare_matching_pairs(split=self.split)

    def prepare_matching_pairs(self, split="train"):
        # Drop 'tiers_indoor09': it's a long-wall sequence (lack of keypoints).
        # Same-sensor barely works, but hetero setup fails due to distribution shift.
        subset_names = open(os.path.join(split_path, self.DATA_FILES[split])).read().split()
        subset_names.remove("tiers_indoor09")

        cache_dir = os.path.join(self.pc_path, "overlap_pairs")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir, f"{self.src_sensor}_to_{self.tgt_sensor}_overlap_pairs.pkl"
        )
        overlap_voxel_size = self.overlap_voxel_size
        overlap_thresh = self.overlap_thresh

        if os.path.exists(cache_file):
            print(f"[INFO] Loading cached pairs from {cache_file}")
            with open(cache_file, "rb") as f:
                all_pairs = pickle.load(f)

            self.files[split] = [
                pair
                for pair in all_pairs
                if len(pair) == 7 and (max(pair[-2], pair[-1]) > overlap_thresh)
            ]

            self.length = len(self.files[split])
            return

        for dirname in subset_names:
            drive_id = str(dirname)

            src_dir = os.path.join(self.pc_path, drive_id, self.src_sensor, "scans")
            tgt_dir = os.path.join(self.pc_path, drive_id, self.tgt_sensor, "scans")

            src_fnames = glob.glob(os.path.join(src_dir, "*.pcd"))
            tgt_fnames = glob.glob(os.path.join(tgt_dir, "*.pcd"))

            assert (
                len(src_fnames) > 0 and len(tgt_fnames) > 0
            ), f"Missing data in {src_dir} or {tgt_dir}"

            src_inames = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in src_fnames])
            tgt_inames = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in tgt_fnames])

            inames = sorted(set(src_inames) & set(tgt_inames))
            if not inames:
                continue

            src_odo = self.get_video_odometry(drive_id, self.src_sensor, return_all=True)
            tgt_odo = self.get_video_odometry(drive_id, self.tgt_sensor, return_all=True)
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
                # Load point clouds
                src_pcd_path = os.path.join(src_dir, f"{curr_time:06d}.pcd")
                tgt_pcd_path = os.path.join(tgt_dir, f"{next_time:06d}.pcd")
                src_pcd = o3d.io.read_point_cloud(src_pcd_path)
                tgt_pcd = o3d.io.read_point_cloud(tgt_pcd_path)

                # Compute transformation
                src_pose = src_positions[curr_idx]
                tgt_pose = tgt_positions[next_idx]
                relt_pose = np.linalg.inv(tgt_pose) @ src_pose
                overlap0, overlap1 = compute_overlap_ratio(
                    src_pcd, tgt_pcd, relt_pose, overlap_voxel_size
                )
                self.files[split].append(
                    (
                        drive_id,
                        self.src_sensor,
                        self.tgt_sensor,
                        curr_time,
                        next_time,
                        overlap0,
                        overlap1,
                    )
                )
                curr_time = next_time + 1
        self.length = len(self.files[split])
        with open(cache_file, "wb") as f:
            pickle.dump(self.files[split], f)
        print(f"[INFO] Saved computed pairs to {cache_file}")
        self.files[split] = [
            pair
            for pair in self.files[split]
            if len(pair) == 7 and (max(pair[-2], pair[-1]) > overlap_thresh)
        ]

    def __getitem__(self, index):
        drive, src_sensor, tgt_sensor, t0, t1, overlap0, overlap1 = self.files[self.split][index]

        src_pose = self.odometry_to_positions(self.get_video_odometry(drive, src_sensor, [t0])[0])
        tgt_pose = self.odometry_to_positions(self.get_video_odometry(drive, tgt_sensor, [t1])[0])
        trans = np.linalg.inv(tgt_pose) @ src_pose

        fname0 = self._get_velodyne_fn(drive, src_sensor, t0)
        fname1 = self._get_velodyne_fn(drive, tgt_sensor, t1)

        o3d_cloud0 = o3d.io.read_point_cloud(fname0)
        o3d_cloud1 = o3d.io.read_point_cloud(fname1)
        xyz0 = np.asarray(o3d_cloud0.points, dtype=np.float32)
        xyz1 = np.asarray(o3d_cloud1.points, dtype=np.float32)

        if self.split != "test":
            xyz0 += (np.random.rand(*xyz0.shape) - 0.5) * self.config.train.augmentation_noise
            xyz1 += (np.random.rand(*xyz1.shape) - 0.5) * self.config.train.augmentation_noise

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

        if self.split != "test":
            R = rotation_matrix(3, 1) if self.config.stage == "Ref" else rotation_matrix(1, 1)
            aug_trans = integrate_trans(R, np.zeros((3, 1)))
            tgt_pcd.transform(aug_trans)
            relt_pose = aug_trans @ trans
        else:
            relt_pose = trans

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
            "src_id": f"{drive}_{t0}",
            "tgt_id": f"{drive}_{t1}",
            "dataset_name": self.config.data.dataset,
            "sphericity": sphericity,
            "is_aligned_to_global_z": is_aligned_to_global_z,
            "sensor": f"{src_sensor}->{tgt_sensor}",
        }
