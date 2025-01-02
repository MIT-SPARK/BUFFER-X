import torch.utils.data as Data
import os
import open3d as o3d
import glob
from utils.SE3 import *
from utils.common import make_open3d_point_cloud

kitti_icp_cache = {}
kitti_cache = {}
cur_path = os.path.dirname(os.path.realpath(__file__))


def get_matching_indices(source, target, relt_pose, search_voxel_size):
    source = transform(source, relt_pose)
    diffs = source[:, None] - target[None]
    dist = np.sqrt(np.sum(diffs ** 2, axis=-1) + 1e-12)
    min_ind = np.concatenate([np.arange(source.shape[0])[:, None], np.argmin(dist, axis=1)[:, None]], axis=-1)
    min_val = np.min(dist, axis=1)
    match_inds = min_ind[min_val < search_voxel_size]

    return match_inds


class KITTIDataset(Data.Dataset):
    DATA_FILES = {
        'train': 'train_kitti.txt',
        'val': 'val_kitti.txt',
        'test': 'test_kitti.txt'
    }

    def __init__(self,
                 split,
                 config=None
                 ):
        self.config = config
        self.pc_path = config.data.root + '/dataset'
        self.icp_path = config.data.root + '/icp'
        self.split = split
        self.files = {'train': [], 'val': [], 'test': []}
        self.poses = []
        self.length = 0

        self.prepare_kitti_ply(split=self.split)

    def prepare_kitti_ply(self, split='train'):
        subset_names = open(os.path.join(cur_path, self.DATA_FILES[split])).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.pc_path + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.pc_path} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files[split].append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1

        # pair (8, 15, 58) is wrong.
        if self.split == 'test':
            self.files[split].remove((8, 15, 58))

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

        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in kitti_icp_cache:
            if not os.path.exists(filename):
                M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                     @ np.linalg.inv(self.velo2cam)).T
                xyz0_t = self.apply_transform(xyz0, M)
                pcd0 = make_open3d_point_cloud(xyz0_t, [0.5, 0.5, 0.5])
                pcd1 = make_open3d_point_cloud(xyz1, [0, 1, 0])
                reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.20, np.eye(4),
                                                                  o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                                  o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                      max_iteration=200))
                pcd0.transform(reg.transformation)
                M2 = M @ reg.transformation
                # write to a file
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            kitti_icp_cache[key] = M2
        else:
            M2 = kitti_icp_cache[key]
        trans = M2

        if self.split != 'test':
            xyz0 += (np.random.rand(xyz0.shape[0], 3) - 0.5) * self.config.train.augmentation_noise
            xyz1 += (np.random.rand(xyz1.shape[0], 3) - 0.5) * self.config.train.augmentation_noise

        # process point clouds
        src_pcd = make_open3d_point_cloud(xyz0, [1, 0.706, 0])
        src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=self.config.data.downsample)
        src_pts = np.array(src_pcd.points)
        np.random.shuffle(src_pts)
        
        # src_pts = np.asarray(src_pcd.points)
        # src_indices = np.arange(src_pts.shape[0])
        # np.random.shuffle(src_indices)
        # src_pts = src_pts[src_indices]
        # src_pcd = make_open3d_point_cloud(src_pts)
        # src_pcd.paint_uniform_color([1, 0.706, 0])
        # src_pcd = o3d.geometry.PointCloud.uniform_down_sample(src_pcd, 5)
        # src_pts = np.array(src_pcd.points)

        tgt_pcd = make_open3d_point_cloud(xyz1, [0, 0.651, 0.929])
        tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=self.config.data.downsample)
        tgt_pts = np.asarray(tgt_pcd.points)
        
        # tgt_indices = np.arange(tgt_pts.shape[0])
        # np.random.shuffle(tgt_indices)
        # tgt_pts = tgt_pts[tgt_indices]
        # tgt_pcd = make_open3d_point_cloud(tgt_pts)
        # tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
        # tgt_pcd = o3d.geometry.PointCloud.uniform_down_sample(tgt_pcd, 5)
        # tgt_pts = np.array(tgt_pcd.points)
        
        if self.split != 'test':
            if self.config.stage == 'Ref':
                # SO(3) augmentation
                R = rotation_matrix(3, 1)
            else:
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
        if (src_kpt.shape[0] > self.config.data.max_numPts):
            idx = np.random.choice(range(src_kpt.shape[0]), self.config.data.max_numPts, replace=False)
            src_kpt = src_kpt[idx]

        if (tgt_kpt.shape[0] > self.config.data.max_numPts):
            idx = np.random.choice(range(tgt_kpt.shape[0]), self.config.data.max_numPts, replace=False)
            tgt_kpt = tgt_kpt[idx]

        if self.split == 'test':
            src_pcd = make_open3d_point_cloud(src_kpt, [1, 0.706, 0])
            src_pcd.estimate_normals()
            src_pcd.orient_normals_towards_camera_location()
            src_noms = np.array(src_pcd.normals)
            src_kpt = np.concatenate([src_kpt, src_noms], axis=-1)

            tgt_pcd = make_open3d_point_cloud(tgt_kpt, [0, 0.651, 0.929])
            tgt_pcd.estimate_normals()
            tgt_pcd.orient_normals_towards_camera_location()
            tgt_noms = np.array(tgt_pcd.normals)
            tgt_kpt = np.concatenate([tgt_kpt, tgt_noms], axis=-1)

        return {'src_fds_pts': src_pts,  # first downsampling
                'tgt_fds_pts': tgt_pts,
                'relt_pose': relt_pose,
                'src_sds_pts': src_kpt,  # second downsampling
                'tgt_sds_pts': tgt_kpt,
                'voxel_size': ds_size,
                'src_id': '%d_%d' % (drive, t0),
                'tgt_id': '%d_%d' % (drive, t1),
                'dataset_name': self.config.data.dataset}

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        data_path = self.pc_path + '/poses/%02d.txt' % drive
        if data_path not in kitti_cache:
            kitti_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return kitti_cache[data_path]
        else:
            return kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        fname = self.pc_path + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def __len__(self):
        return self.length
