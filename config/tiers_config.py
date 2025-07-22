from .outdoor_config import OutdoorBaseConfig
from pathlib import Path

# Note
# Although Tiers is an indoor dataset, it's scale is as large as outdoor dataset,
# so we inherit from OutdoorBaseConfig


class TIERSConfig(OutdoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "TIERS"
        self._C.data.root = root_dir / "tiers_indoor"
        self._C.test.pdist = 2
        self._C.test.experiment_id = "threedmatch"


def make_cfg(root_dir):
    return TIERSConfig(root_dir).get_cfg()
