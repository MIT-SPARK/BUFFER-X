
from easydict import EasyDict as edict


_C = edict()
_C.data = edict()
_C.data.dataset = 'SuperSet'
# Note: It should be equal to the `_C.data.dataset` in KITTI and ThreeDMatch config file
_C.data.subsetdatasets = ['KITTI', '3DMatch', 'Scannetpp_faro', 'Scannetpp_iphone', 'WOD']
_C.data.voxel_size_0 = 0.035 # Criteria: ThreeDMatch
_C.data.manual_seed = 123

# training
_C.train = edict()
_C.train.epoch = 30
_C.train.max_iter = 50000
_C.train.batch_size = 1
_C.train.num_workers = 16
_C.train.pos_num = 512
_C.train.augmentation_noise = 0.001 # for 3DMatch
_C.train.pretrain_model = '' # Example '../KITTI/snapshot/06050001'
_C.train.all_stage = ['Ref', 'Desc', 'Keypt', 'Inlier']

# test
_C.test = edict()
_C.test.experiment_id = '06050001'
_C.test.pose_refine = False

# optim
_C.optim = edict()
_C.optim.lr = {'Ref': 0.005, 'Desc':0.001, 'Keypt':0.001, 'Inlier':0.001}
_C.optim.lr_decay = 0.50
_C.optim.weight_decay = 1e-6
# ToDo. Support different interval for different dataset?
# In 3DMatch, _C.optim.scheduler_interval = {'Ref': 1, 'Desc':2, 'Keypt':1, 'Inlier':1}
# In KITTI, _C.optim.scheduler_interval = {'Ref': 5, 'Desc':10, 'Keypt':5, 'Inlier':5}
_C.optim.scheduler_interval = {'Ref': 3, 'Desc':6, 'Keypt':3, 'Inlier':3}
# _C.optim.scheduler_interval = {'Ref': 5, 'Desc':10, 'Keypt':5, 'Inlier':5}

# point-wise learner
_C.point = edict()
_C.point.in_points_dim = 3
_C.point.in_feats_dim = 3
_C.point.first_feats_dim = 32
_C.point.conv_radius = 2.0              # NOTE: this conv_radius is ratio
_C.point.keypts_th = 0.5                # KITTI: 0.5, 3DMatch: 0.1, used at inference
_C.point.num_keypts = 1500

# patch-wise embedder
_C.patch = edict()
_C.patch.des_r = 3.0                    # KITTI: 3.0, 3DMatch: 0.3
_C.patch.num_points_per_patch = 512
_C.patch.rad_n = 3
_C.patch.azi_n = 20
_C.patch.ele_n = 7
_C.patch.delta = 0.8
_C.patch.voxel_sample = 10

# inliers && ransac
_C.match = edict()
_C.match.dist_th = 0.30                 # KITTI: 0.30, 3DMatch: 0.1
_C.match.inlier_th = 2.0                # KITTI: 2.0 , 3DMatch: 0.33
_C.match.similar_th = 0.9               # KITTI: 0.9 , 3DMatch: 0.8
_C.match.confidence = 1.0               # KITTI: 1.0 , 3DMatch: 0.999
_C.match.iter_n = 50000



def make_cfg():
    return _C
