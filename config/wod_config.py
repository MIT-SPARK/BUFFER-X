from .outdoor_config import OutdoorBaseConfig

class WODConfig(OutdoorBaseConfig):
    def __init__(self):
        super().__init__()
        self._C.data.dataset = 'WOD'
        self._C.data.root = '../../datasets/WOD'
        self._C.test.experiment_id = 'threedmatch'

def make_cfg():
    return WODConfig().get_cfg()