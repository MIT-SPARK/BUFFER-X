import torch.utils.data as Data
import os
import open3d as o3d
import glob
from utils.SE3 import *
from utils.common import make_open3d_point_cloud
from KITTI.dataset import *
from ThreeDMatch.dataset import *

THREEDMATCH_SPLITS = 10

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
        self.threedmatch_splits = THREEDMATCH_SPLITS
        self.split_index = 0
        current_length = 0
        
        for subsetdataset in config.data.subsetdatasets:
            if subsetdataset == "KITTI":
                DatasetClass = KITTIDataset
            elif subsetdataset == "3DMatch":
                DatasetClass = ThreeDMatchDataset
            else:
                raise ValueError("Unsupported dataset class has been given")
            dataset = DatasetClass(split=split,
                                   config=config[subsetdataset])
            self.datasets.append(dataset)
            dataset_length = len(dataset)
            
            if subsetdataset == "3DMatch":
                dataset_length = dataset_length // self.threedmatch_splits
                
            self.lengths.append(dataset_length)
            current_length += dataset_length

        self.total_len = current_length
        self.dataset_intervals = np.cumsum([0] + self.lengths)
        self.randg = np.random.RandomState()

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        for i, (start, end) in enumerate(zip(self.dataset_intervals[:-1], self.dataset_intervals[1:])):
            if start <= index < end:
                if i == 1:
                    start_index = self.split_index * self.lengths[i]
                    return self.datasets[i][start_index + index - start]
                else: 
                    return self.datasets[i][index - start]
    
    def next_split(self):
        self.split_index = (self.split_index + 1) % self.threedmatch_splits

    def reset_seed(self, seed=0):
        self.randg.seed(seed)