from .indoor_config import IndoorBaseConfig

class ScannetppFaroConfig(IndoorBaseConfig):
    def __init__(self):
        super().__init__()
        self._C.data.dataset = 'Scannetpp_faro'
        self._C.data.root = '../datasets/scannetpp/scannet-plusplus'
        self._C.test.experiment_id = 'threedmatch'

def make_cfg():
    return ScannetppFaroConfig().get_cfg()