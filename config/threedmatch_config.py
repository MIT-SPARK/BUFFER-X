from .indoor_config import IndoorBaseConfig

class ThreeDMatchConfig(IndoorBaseConfig):
    def __init__(self):
        super().__init__()
        self._C.data.dataset = '3DMatch'
        self._C.data.benchmark = '3DMatch'
        self._C.data.root = '../../datasets/ThreeDMatch'
        self._C.test.experiment_id = 'threedmatch'
        self._C.test.pose_refine = True

def make_cfg():
    return ThreeDMatchConfig().get_cfg()