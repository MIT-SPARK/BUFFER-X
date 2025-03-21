import sys
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import time
import argparse
import copy
import numpy as np
from torch import optim

from utils.SE3 import *
from dataset.dataloader import get_dataloader
from models.BUFFER import buffer
from trainer import Trainer

# Import dataset-specific config
from config import make_cfg

# Argument parser
parser = argparse.ArgumentParser(description="Generalized Training Script for Registration Models")
parser.add_argument("--dataset", type=str, required=True, choices=["3DMatch", "KITTI"],
                    help="Dataset to train on (3DMatch or KITTI)")
args = parser.parse_args()


class Args(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load model
        self.model = buffer(cfg)
        self.parameter = self.model.get_parameter()

        # Load pre-trained weights and freeze unnecessary modules
        left_stage = copy.deepcopy(cfg.train.all_stage)
        left_stage.remove(cfg.stage)

        if cfg.train.pretrain_model:
            state_dict = torch.load(cfg.train.pretrain_model)
            self.model.load_state_dict(state_dict)
            print(f"Loaded pretrained model from {cfg.train.pretrain_model}\n")

        for modname in left_stage:
            weight_path = os.path.join(cfg.snapshot_root, modname, 'best.pth')
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path)
                new_dict = {k: v for k, v in state_dict.items() if modname in k}
                model_dict = self.model.state_dict()
                model_dict.update(new_dict)
                self.model.load_state_dict(model_dict)
                print(f"Loaded {modname} from {weight_path}\n")

            # Freeze parameters of the loaded module
            for p in getattr(self.model, modname).parameters():
                p.requires_grad = False

        # Optimizer
        self.optimizer = optim.Adam(self.parameter, lr=cfg.optim.lr[cfg.stage], weight_decay=cfg.optim.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.optim.lr_decay)
        self.scheduler_interval = cfg.optim.scheduler_interval[cfg.stage]

        # GPU setup
        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])

        # Dataloader
        self.train_loader = get_dataloader(dataset=cfg.data.dataset, split='train', config=cfg, shuffle=True,
                                           num_workers=cfg.train.num_workers)
        self.val_loader = get_dataloader(dataset=cfg.data.dataset, split='val', config=cfg, shuffle=False,
                                         num_workers=cfg.train.num_workers)

        print(f"Training set size: {len(self.train_loader.dataset)}")
        print(f"Validation set size: {len(self.val_loader.dataset)}")

        # Snapshot paths
        self.save_dir = os.path.join(cfg.snapshot_root, cfg.stage)
        self.tboard_dir = cfg.tensorboard_root

        # Evaluation interval
        self.evaluate_interval = 1


if __name__ == '__main__':
    print(f"Starting training on {args.dataset}...")

    # Load dataset-specific config
    cfg = make_cfg(args.dataset)
    cfg[cfg.data.dataset] = cfg.copy()
    cfg.stage = 'train'

    # Generate experiment ID
    if cfg.train.pretrain_model:
        experiment_id = cfg.train.pretrain_model.split('/')[1]
    else:
        experiment_id = time.strftime('%m%d%H%M')

    # Set seed
    if cfg.data.manual_seed is not None:
        np.random.seed(cfg.data.manual_seed)
        torch.manual_seed(cfg.data.manual_seed)
        torch.cuda.manual_seed_all(cfg.data.manual_seed)
    else:
        print("Warning: No seed setting!!!")

    dataset = args.dataset
    # Training loop
    for stage in cfg.train.all_stage:
        cfg.stage = stage
        cfg.snapshot_root = f'snapshot/{dataset}/{experiment_id}'
        cfg.tensorboard_root = f'tensorboard/{experiment_id}/{cfg.stage}'

        args = Args(cfg)
        trainer = Trainer(args)
        trainer.train()