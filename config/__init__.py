from .threedmatch_config import make_cfg as make_3dmatch_cfg
from .threedlomatch_config import make_cfg as make_3dlomatch_cfg
from .scannetpp_iphone_config import make_cfg as make_scannetpp_iphone_cfg
from .scannetpp_faro_config import make_cfg as make_scannetpp_faro_cfg
from .tiers_config import make_cfg as make_tiers_cfg
from .kitti_config import make_cfg as make_kitti_cfg
from .wod_config import make_cfg as make_wod_cfg
from .mit_config import make_cfg as make_mit_cfg
from .kaist_config import make_cfg as make_kaist_cfg
from .eth_config import make_cfg as make_eth_cfg
from .oxford_config import make_cfg as make_oxford_cfg

def make_cfg(dataset_name):
    """
    Generalized function to return the appropriate configuration based on dataset name.
    """
    if dataset_name == "3DMatch":
        return make_3dmatch_cfg()
    elif dataset_name == "3DLoMatch":
        return make_3dlomatch_cfg()
    elif dataset_name == "Scannetpp_iphone":
        return make_scannetpp_iphone_cfg()
    elif dataset_name == "Scannetpp_faro":
        return make_scannetpp_faro_cfg()
    elif dataset_name == "Tiers":
        return make_tiers_cfg()
    elif dataset_name == "KITTI":
        return make_kitti_cfg()
    elif dataset_name == "WOD":
        return make_wod_cfg()
    elif dataset_name == "MIT":
        return make_mit_cfg()
    elif dataset_name == "KAIST":
        return make_kaist_cfg()
    elif dataset_name == "ETH":
        return make_eth_cfg()
    elif dataset_name == "Oxford":
        return make_oxford_cfg()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")