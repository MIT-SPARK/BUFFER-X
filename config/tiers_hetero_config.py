from .outdoor_config import OutdoorBaseConfig
from pathlib import Path

# Note
# Although Tiers is an indoor dataset, it's scale is as large as outdoor dataset,
# so we inherit from OutdoorBaseConfig


class TIERSHeteroConfig(OutdoorBaseConfig):
    def __init__(self, root_dir=Path("../datasets")):
        super().__init__()
        self._C.data.dataset = "TIERS_hetero"
        self._C.data.root = root_dir / "tiers_indoor"

        # 'os0_128', 'os1_64', 'vel16'
        self._C.data.src_sensor = "os0_128"
        self._C.data.tgt_sensor = "os1_64"

        # overlap voxel size and threshold are set to avoid lack of GT correspondences
        # when overlap is too low in heterogeneous sensor setups
        self._C.test.overlap_voxel_size = 0.1
        self._C.test.overlap_thresh = 0.3

        self._C.test.pdist = 2
        self._C.test.experiment_id = "threedmatch"


def make_cfg(root_dir):
    return TIERSHeteroConfig(root_dir).get_cfg()
