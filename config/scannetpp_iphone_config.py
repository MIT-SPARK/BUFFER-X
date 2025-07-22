from .indoor_config import IndoorBaseConfig
from pathlib import Path


class ScannetppIphoneConfig(IndoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "Scannetpp_iphone"
        self._C.data.root = root_dir / "ScanNetpp_Iphone"
        self._C.test.experiment_id = "threedmatch"


def make_cfg(root_dir):
    return ScannetppIphoneConfig(root_dir).get_cfg()
