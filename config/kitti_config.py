from .outdoor_config import OutdoorBaseConfig
from pathlib import Path


class KITTIConfig(OutdoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "KITTI"
        self._C.data.root = root_dir / "kitti"
        self._C.test.pdist = 10

        self._C.train.pretrain_model = ""
        self._C.train.all_stage = ["Desc", "Pose"]

        self._C.test.experiment_id = "threedmatch"


def make_cfg(root_dir):
    return KITTIConfig(root_dir).get_cfg()
