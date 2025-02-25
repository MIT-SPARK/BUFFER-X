import sys

sys.path.append('../')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import math
import time
import argparse
import copy
import numpy as np
from KITTI.config import make_cfg as kitti_make_cfg
from SuperSet_Eval.dataloader import get_dataloader
from ThreeDMatch.config import make_cfg as threedmatch_make_cfg
from Scannetpp_faro.config import make_cfg as scannetpp_faro_make_cfg
from Scannetpp_iphone.config import make_cfg as scannetpp_iphone_make_cfg
from WOD.config import make_cfg as wod_make_cfg
from NewerCollege.config import make_cfg as newercollege_make_cfg
from KimeraMulti.config import make_cfg as kimeramulti_make_cfg
from Tiers.config import make_cfg as tiers_make_cfg
from KAIST.config import make_cfg as kaist_make_cfg
from ETH.config import make_cfg as eth_make_cfg

from SuperSet_Eval.config import make_cfg
from utils.timer import Timer
from models.BUFFER import buffer
from utils.SE3 import *
from models.BUFFER import buffer

from torch import optim

from ThreeDMatch.test import evaluate_registration, read_trajectory, read_trajectory_info

arg_lists = []
parser = argparse.ArgumentParser()

if __name__ == '__main__':
    print("Start testing...")
    cfg = make_cfg()
    indoor_datasets = ['3DMatch', 'Scannetpp_faro', 'Scannetpp_iphone']
    outdoor_datasets = ['KITTI', 'WOD', 'NewerCollege', 'KimeraMulti', 'Tiers', 'KAIST', 'ETH']
    for subsetdataset in cfg.data.subsetdatasets:
        if (subsetdataset == 'KITTI'):
            cfg_tmp = kitti_make_cfg()
        elif (subsetdataset == '3DMatch'):
            cfg_tmp = threedmatch_make_cfg()
        elif (subsetdataset == 'Scannetpp_faro'):
            cfg_tmp = scannetpp_faro_make_cfg()
        elif (subsetdataset == 'Scannetpp_iphone'):
            cfg_tmp = scannetpp_iphone_make_cfg()
        elif (subsetdataset == 'WOD'):
            cfg_tmp = wod_make_cfg()
        elif (subsetdataset == 'NewerCollege'):
            cfg_tmp = newercollege_make_cfg()
        elif (subsetdataset == 'KimeraMulti'):
            cfg_tmp = kimeramulti_make_cfg()
        elif (subsetdataset == 'Tiers'):
            cfg_tmp = tiers_make_cfg()
        elif (subsetdataset == 'KAIST'):
            cfg_tmp = kaist_make_cfg()
        elif (subsetdataset == 'ETH'):
            cfg_tmp = kaist_make_cfg()
        else:
            raise ValueError("Unsupported dataset name has been given")
        cfg_tmp.stage = 'test'
        cfg[subsetdataset] = cfg_tmp
        print(f"Dataset: {subsetdataset} has been loaded")

    # snapshot
    cfg.stage = 'test'
    model = buffer(cfg)
    experiment_id = cfg.test.experiment_id

    # load the weight
    for stage in cfg.train.all_stage:
        model_path = 'snapshot/%s/%s/best.pth' % (experiment_id, stage)
        state_dict = torch.load(model_path)
        new_dict = {k: v for k, v in state_dict.items() if stage in k}
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print(f"Load {stage} model from {model_path}")
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    model = nn.DataParallel(model, device_ids=[0])
    model.eval()
    
    data_list = []
    
    timestr = time.strftime('%m%d%H%M')
    sphericity_file = f'sphericity.txt'
    with open(sphericity_file, 'w') as f:
        f.write(f"Dataset, Mean, Min, Max\n")
        for subsetdataset in cfg.data.subsetdatasets:
            f.write(f"{subsetdataset}, ")
            print(f"Start testing {subsetdataset}...")
            cfg[subsetdataset].stage = 'test'
            test_loader = get_dataloader(split='test',
                                        config=cfg,
                                        subsetdataset=subsetdataset,
                                        shuffle=False,
                                        num_workers=cfg.train.num_workers,
                                        )
            print(f"{subsetdataset} Test set size:", test_loader.dataset.__len__())
            
            sphericity_list = []
            states = []
            num_batch = len(test_loader)
            data_iter = iter(test_loader)
            for i in range(num_batch):
                data_source = data_iter.__next__()
                sphericity = data_source['sphericity']
                sphericity_list.append(sphericity)
            mean_sphericity = np.mean(sphericity_list)
            min_sphericity = np.min(sphericity_list)
            max_sphericity = np.max(sphericity_list)
            f.write(f"{mean_sphericity}, {min_sphericity}, {max_sphericity}\n")
            data_list.append([subsetdataset, mean_sphericity, min_sphericity, max_sphericity])
            print(f"{subsetdataset} Mean Sphericity: {mean_sphericity}")
            print(f"{subsetdataset} Min Sphericity: {min_sphericity}")
            print(f"{subsetdataset} Max Sphericity: {max_sphericity}")
            
                
                    
      
            
     