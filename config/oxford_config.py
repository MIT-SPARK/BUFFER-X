from .outdoor_config import OutdoorBaseConfig

class OxfordConfig(OutdoorBaseConfig):
    def __init__(self):
        super().__init__()
        self._C.data.dataset = 'Oxford'
        self._C.data.root = '../../datasets/newer-college'
        self._C.test.experiment_id = 'threedmatch'

def make_cfg():
    return OxfordConfig().get_cfg()