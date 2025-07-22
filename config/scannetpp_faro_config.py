from .indoor_config import IndoorBaseConfig
from pathlib import Path


class ScannetppFaroConfig(IndoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "Scannetpp_faro"
        self._C.data.root = root_dir / "scannetpp" / "scannet-plusplus"
        self._C.test.experiment_id = "threedmatch"


def make_cfg(root_dir):
    return ScannetppFaroConfig(root_dir).get_cfg()
