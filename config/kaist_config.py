from .outdoor_config import OutdoorBaseConfig
from pathlib import Path


class KAISTConfig(OutdoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "KAIST"
        self._C.data.root = root_dir / "helipr_kaist05"
        self._C.test.pdist = 10
        self._C.test.experiment_id = "threedmatch"


def make_cfg(root_dir):
    return KAISTConfig(root_dir).get_cfg()
