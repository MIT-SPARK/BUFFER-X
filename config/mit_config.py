from .outdoor_config import OutdoorBaseConfig

class MITConfig(OutdoorBaseConfig):
    def __init__(self):
        super().__init__()
        self._C.data.dataset = 'MIT'
        self._C.data.root = '../datasets/kimera-multi'
        self._C.test.pdist = 5
        self._C.test.experiment_id = 'threedmatch'

def make_cfg():
    return MITConfig().get_cfg()