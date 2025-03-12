
from easydict import EasyDict as edict


_C = edict()

# data
_C.data = edict()
_C.data.dataset = 'ETH'
_C.data.root = '../../datasets/ETH'
_C.data.downsample = 0.05
_C.data.voxel_size_0 = 0.15 # ETH
_C.data.voxel_size_1 = 0.03 # 3DMatch
_C.data.max_numPts = 30000
_C.data.manual_seed = 123

# test
_C.test = edict()
_C.test.scale = _C.data.voxel_size_0 / _C.data.voxel_size_1 # 'target size / source size'
_C.test.experiment_id = 'KITTI_Rad_Aug_Uniform'
_C.test.pose_refine = False
_C.test.all_stage = ['Desc', 'Inlier']
_C.test.batch_size = 1

# point-wise learner
_C.point = edict()
_C.point.in_points_dim = 3
_C.point.in_feats_dim = 3
_C.point.first_feats_dim = 32
_C.point.conv_radius = 2.0
_C.point.keypts_th = 0.5
_C.point.num_keypts = 2000

# patch-wise embedder
_C.patch = edict()
_C.patch.des_r = 1.0
_C.patch.num_points_per_patch = 512
_C.patch.rad_n = 3
_C.patch.azi_n = 20
_C.patch.ele_n = 7
_C.patch.delta = 0.8
_C.patch.voxel_sample = 10

# inliers && ransac
_C.match = edict()
_C.match.dist_th = 0.20
_C.match.inlier_th = 1.5
_C.match.similar_th = 0.9
_C.match.confidence = 1.0
_C.match.iter_n = 50000



def make_cfg():
    return _C
