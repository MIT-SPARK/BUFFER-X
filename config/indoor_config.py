from easydict import EasyDict as edict

class IndoorBaseConfig:
    def __init__(self):
        self._C = edict()

        # Data
        self._C.data = edict()
        self._C.data.dataset = ''
        self._C.data.root = ''
        self._C.data.downsample = 0.02
        self._C.data.voxel_size_0 = 0.035
        self._C.data.voxel_size_1 = self._C.data.voxel_size_0
        self._C.data.max_numPts = 30000
        self._C.data.manual_seed = 123

        # Training
        self._C.train = edict()
        self._C.train.epoch = 10
        self._C.train.max_iter = 50000
        self._C.train.batch_size = 1
        self._C.train.num_workers = 0
        self._C.train.pos_num = 512
        self._C.train.augmentation_noise = 0.001
        self._C.train.pretrain_model = ''
        self._C.train.all_stage = ['Desc', 'Inlier']

        # Test
        self._C.test = edict()
        self._C.test.experiment_id = 'threedmatch'
        self._C.test.pose_refine = False

        # Evaluation thresholds
        self._C.test.rte_thresh = 0.3  # RTE threshold for indoor datasets
        self._C.test.rre_thresh = 15.0  # RRE threshold for indoor datasets

        # Optimizer
        self._C.optim = edict()
        self._C.optim.lr = {'Desc': 0.001, 'Inlier': 0.001}
        self._C.optim.lr_decay = 0.50
        self._C.optim.weight_decay = 1e-6
        self._C.optim.scheduler_interval = {'Desc': 2, 'Inlier': 1}

        # Point-wise learner
        self._C.point = edict()
        self._C.point.in_points_dim = 3
        self._C.point.in_feats_dim = 3
        self._C.point.first_feats_dim = 32
        self._C.point.conv_radius = 2.0
        self._C.point.keypts_th = 0.1
        self._C.point.num_fps = 1500

        # Patch-wise embedder
        self._C.patch = edict()
        self._C.patch.des_r = 0.3
        self._C.patch.num_points_per_patch = 512
        self._C.patch.rad_n = 3
        self._C.patch.azi_n = 20
        self._C.patch.ele_n = 7
        self._C.patch.delta = 0.8
        self._C.patch.voxel_sample = 10
        self._C.patch.num_points_radius_estimate = 2000

        # Inliers & RANSAC
        self._C.match = edict()
        self._C.match.dist_th = 0.10
        self._C.match.inlier_th = 1/3
        self._C.match.similar_th = 0.8
        self._C.match.confidence = 0.999
        self._C.match.iter_n = 50000

    def get_cfg(self):
        return self._C