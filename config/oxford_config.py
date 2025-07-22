from .outdoor_config import OutdoorBaseConfig
from pathlib import Path


class OxfordConfig(OutdoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "Oxford"
        self._C.data.root = root_dir / "newer-college"
        self._C.test.pdist = 5
        self._C.test.experiment_id = "threedmatch"


def make_cfg(root_dir):
    return OxfordConfig(root_dir).get_cfg()
