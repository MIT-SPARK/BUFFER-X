from .indoor_config import IndoorBaseConfig
from pathlib import Path


class ModelNet40Config(IndoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "ModelNet40"
        self._C.data.root = root_dir / "processed_modelnet40"
        self._C.test.experiment_id = "threedmatch"
        self._C.test.pose_refine = False
        self._C.test.rte_thresh = 0.1  # RTE threshold for object scale


def make_cfg(root_dir):
    return ModelNet40Config(root_dir).get_cfg()
