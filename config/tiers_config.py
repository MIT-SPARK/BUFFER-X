from .outdoor_config import OutdoorBaseConfig

# Note
# Although Tiers is an indoor dataset, it's scale is as large as outdoor dataset,
# so we inherit from OutdoorBaseConfig


class TIERSConfig(OutdoorBaseConfig):
    def __init__(self):
        super().__init__()
        self._C.data.dataset = "TIERS"
        self._C.data.root = "../datasets/tiers_indoor"
        self._C.test.pdist = 2
        self._C.test.experiment_id = "threedmatch"


def make_cfg():
    return TIERSConfig().get_cfg()
