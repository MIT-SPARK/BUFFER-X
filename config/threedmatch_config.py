from .indoor_config import IndoorBaseConfig
from pathlib import Path


class ThreeDMatchConfig(IndoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "3DMatch"
        self._C.data.benchmark = "3DMatch"
        self._C.data.root = root_dir / "ThreeDMatch"
        self._C.test.experiment_id = "threedmatch"
        self._C.test.pose_refine = True

        self._C.train.pretrain_model = ""
        self._C.train.all_stage = ["Desc", "Pose"]


def make_cfg(root_dir):
    return ThreeDMatchConfig(root_dir).get_cfg()
