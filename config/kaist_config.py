from .outdoor_config import OutdoorBaseConfig


class KAISTConfig(OutdoorBaseConfig):
    def __init__(self):
        super().__init__()
        self._C.data.dataset = "KAIST"
        self._C.data.root = "../datasets/helipr_kaist05"
        self._C.test.pdist = 10
        self._C.test.experiment_id = "threedmatch"


def make_cfg():
    return KAISTConfig().get_cfg()
