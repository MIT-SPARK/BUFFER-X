from .outdoor_config import OutdoorBaseConfig

class KITTIConfig(OutdoorBaseConfig):
    def __init__(self):
        super().__init__()
        self._C.data.dataset = 'KITTI'
        self._C.data.root = '../datasets/kitti'
        self._C.test.experiment_id = 'threedmatch'

        self._C.train.pretrain_model = ''
        self._C.train.all_stage = ['Desc', 'Inlier']
        
def make_cfg():
    return KITTIConfig().get_cfg()