from .outdoor_config import OutdoorBaseConfig

# Note
# Although Tiers is an indoor dataset, it's scale is as large as outdoor dataset,
# so we inherit from OutdoorBaseConfig


class TiersConfig(OutdoorBaseConfig):
    def __init__(self):
        super().__init__()
        self._C.data.dataset = "Tiers"
        self._C.data.root = "../datasets/250212_tiers"
        self._C.test.pdist = 2
        self._C.test.experiment_id = "threedmatch"


def make_cfg():
    return TiersConfig().get_cfg()
