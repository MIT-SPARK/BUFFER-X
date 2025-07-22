from .outdoor_config import OutdoorBaseConfig
from pathlib import Path


class MITConfig(OutdoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "MIT"
        self._C.data.root = root_dir / "kimera-multi"
        self._C.test.pdist = 5
        self._C.test.experiment_id = "threedmatch"


def make_cfg(root_dir):
    return MITConfig(root_dir).get_cfg()
