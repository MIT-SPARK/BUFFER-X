from .indoor_config import IndoorBaseConfig
from pathlib import Path


class ThreeDLoMatchConfig(IndoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "3DMatch"
        self._C.data.benchmark = "3DLoMatch"
        self._C.data.root = root_dir / "ThreeDMatch"
        self._C.test.experiment_id = "threedmatch"
        self._C.test.pose_refine = True


def make_cfg(root_dir):
    return ThreeDLoMatchConfig(root_dir).get_cfg()
