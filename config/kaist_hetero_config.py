from .outdoor_config import OutdoorBaseConfig
from pathlib import Path


class KAISTHeteroConfig(OutdoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "KAIST_hetero"
        self._C.data.root = root_dir / "helipr_kaist05"

        # 'Aeva', 'Avia', 'Ouster'
        self._C.data.src_sensor = "Avia"
        self._C.data.tgt_sensor = "Ouster"

        self._C.test.pdist = 10
        self._C.test.experiment_id = "threedmatch"


def make_cfg(root_dir):
    return KAISTHeteroConfig(root_dir).get_cfg()
