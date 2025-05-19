from .indoor_config import IndoorBaseConfig

class ModelNet40Config(IndoorBaseConfig):
    def __init__(self):
        super().__init__()
        self._C.data.dataset = 'ModelNet40'
        self._C.data.root = '/root/dataset/modelnet_ply'
        self._C.test.experiment_id = 'threedmatch'
        self._C.test.pose_refine = False
        
        self._C.train.pretrain_model = ''
        self._C.train.all_stage = ['Desc', 'Inlier']

def make_cfg():
    return ModelNet40Config().get_cfg()