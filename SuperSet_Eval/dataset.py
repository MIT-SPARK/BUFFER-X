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
from NewerCollege.dataset import *
from KimeraMulti.dataset import *
from Tiers.dataset import *
from KAIST.dataset import *

class SupersetDataset(Data.Dataset):
    def __init__(self, split, config=None, subsetdataset=None):
        self.datasets = []
        self.lengths = []
        self.split_indices = []
        current_length = 0
        
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
        elif subsetdataset == "NewerCollege":
            DatasetClass = NewerCollegeDataset
        elif subsetdataset == "KimeraMulti":
            DatasetClass = KimeraMultiDataset
        elif subsetdataset == "Tiers":
            DatasetClass = TiersDataset
        elif subsetdataset == "KAIST":
            DatasetClass = KAISTDataset
        else:
            raise ValueError("Unsupported dataset class has been given")
        dataset = DatasetClass(split=split,
                                config=config[subsetdataset])
        self.datasets.append(dataset)
        dataset_length = len(dataset)

        self.lengths.append(dataset_length)
        self.split_indices.append(0)
        current_length += dataset_length
            
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

    def reset_seed(self, seed=0):
        self.randg.seed(seed)