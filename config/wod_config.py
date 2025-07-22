from .outdoor_config import OutdoorBaseConfig
from pathlib import Path


class WODConfig(OutdoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "WOD"
        self._C.data.root = root_dir / "WOD"
        self._C.test.pdist = 10
        self._C.test.experiment_id = "threedmatch"


def make_cfg(root_dir):
    return WODConfig(root_dir).get_cfg()
