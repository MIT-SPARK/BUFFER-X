import torch.utils.data as Data
import os
import open3d as o3d
import glob
from utils.SE3 import *
from utils.common import make_open3d_point_cloud
from KITTI.dataset import *
from ThreeDMatch.dataset import *
from Scannetpp_faro.dataset import *
from Scannetpp_iphone.dataset import *
from WOD.dataset import *

THREEDMATCH_SPLITS = 30
KITTI_SPLITS = 3
FARO_SPLITS = 3
IPHONE_SPLITS = 30
WOD_SPLITS = 30

def get_matching_indices(source, target, relt_pose, search_voxel_size):
    source = transform(source, relt_pose)
    diffs = source[:, None] - target[None]
    dist = np.sqrt(np.sum(diffs ** 2, axis=-1) + 1e-12)
    min_ind = np.concatenate([np.arange(source.shape[0])[:, None], np.argmin(dist, axis=1)[:, None]], axis=-1)
    min_val = np.min(dist, axis=1)
    match_inds = min_ind[min_val < search_voxel_size]

    return match_inds

class SupersetDataset(Data.Dataset):
    def __init__(self, split, config=None):
        self.datasets = []
        self.lengths = []
        self.split_indices = []
        current_length = 0
        
        for subsetdataset in config.data.subsetdatasets:
            if subsetdataset == "KITTI":
                DatasetClass = KITTIDataset
            elif subsetdataset == "3DMatch":
                DatasetClass = ThreeDMatchDataset
            elif subsetdataset == "Scannetpp_faro":
                DatasetClass = ScannetppFaroDataset
            elif subsetdataset == "Scannetpp_iphone":
                DatasetClass = ScannetppIphoneDataset
            elif subsetdataset == "WOD":
                DatasetClass = WODDataset
            else:
                raise ValueError("Unsupported dataset class has been given")
            dataset = DatasetClass(split=split,
                                   config=config[subsetdataset])
            self.datasets.append(dataset)
            dataset_length = len(dataset)
            
            if subsetdataset == "KITTI":
                dataset_length = dataset_length // KITTI_SPLITS
            elif subsetdataset == "3DMatch":
                dataset_length = dataset_length // THREEDMATCH_SPLITS
            elif subsetdataset == "Scannetpp_faro":
                dataset_length = dataset_length // FARO_SPLITS
            elif subsetdataset == "Scannetpp_iphone":
                dataset_length = dataset_length // IPHONE_SPLITS
            elif subsetdataset == "WOD":
                dataset_length = dataset_length // WOD_SPLITS
                
            self.lengths.append(dataset_length)
            self.split_indices.append(0)
            current_length += dataset_length
            print(f"{subsetdataset} has {dataset_length} samples")


        self.total_len = current_length
        self.dataset_intervals = np.cumsum([0] + self.lengths)
        self.randg = np.random.RandomState()

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        for i, (start, end) in enumerate(zip(self.dataset_intervals[:-1], self.dataset_intervals[1:])):
            if start <= index < end:
                start_index = self.split_indices[i] * self.lengths[i]
                return self.datasets[i][start_index + index - start]

    
    def next_split(self):
        for i, dataset in enumerate(self.datasets):
            if isinstance(dataset, KITTIDataset):
                split = KITTI_SPLITS
            elif isinstance(dataset, ThreeDMatchDataset):
                split = THREEDMATCH_SPLITS
            elif isinstance(dataset, ScannetppFaroDataset):
                split = FARO_SPLITS
            elif isinstance(dataset, ScannetppIphoneDataset):
                split = IPHONE_SPLITS
            elif isinstance(dataset, WODDataset):
                split = WOD_SPLITS
            else:
                continue
            # 해당 데이터셋에 대해 split_index를 업데이트
            self.split_indices[i] = (self.split_indices[i] + 1) % split

    def reset_seed(self, seed=0):
        self.randg.seed(seed)