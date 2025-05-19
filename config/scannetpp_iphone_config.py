from .indoor_config import IndoorBaseConfig

class ScannetppIphoneConfig(IndoorBaseConfig):
    def __init__(self):
        super().__init__()
        self._C.data.dataset = 'Scannetpp_iphone'
        self._C.data.root = '../datasets/ScanNetpp_Iphone'
        self._C.test.experiment_id = 'threedmatch'

def make_cfg():
    return ScannetppIphoneConfig().get_cfg()