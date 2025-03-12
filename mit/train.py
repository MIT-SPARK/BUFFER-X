import sys

sys.path.append('../')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import time
import argparse
import copy
import numpy as np
from mit.config import make_cfg
from mit.dataloader import get_dataloader
from mit.trainer import Trainer
from models.BUFFER import buffer
from torch import optim

arg_lists = []
parser = argparse.ArgumentParser()


class Args(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # model
        self.model = buffer(cfg)
        self.parameter = self.model.get_parameter()

        # load pre-trained weights and freeze irrelevant modules
        left_stage = copy.deepcopy(cfg.train.all_stage)
        left_stage.remove(cfg.stage)
        if cfg.train.pretrain_model != '':
            state_dict = torch.load(cfg.train.pretrain_model)
            self.model.load_state_dict(state_dict)
            print(f"Load pretrained model from {cfg.train.pretrain_model}\n")
        for modname in left_stage:
            weight_path = cfg.snapshot_root + f'/{modname}/best.pth'
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path)
                new_dict = {k: v for k, v in state_dict.items() if modname in k}
                model_dict = self.model.state_dict()
                model_dict.update(new_dict)
                self.model.load_state_dict(model_dict)
                print(f"Load {modname} from {weight_path}\n")
            for p in getattr(self.model, modname).parameters():
                p.requires_grad = False

        # optimizer
        self.optimizer = optim.Adam(self.parameter, lr=cfg.optim.lr[cfg.stage], weight_decay=cfg.optim.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=cfg.optim.lr_decay) # training speed related to gamma
        self.scheduler_interval = cfg.optim.scheduler_interval[cfg.stage]

        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])

        # dataloader
        self.train_loader = get_dataloader(split='train',
                                           config=cfg,
                                           shuffle=True,
                                           num_workers=cfg.train.num_workers,
                                           )
        self.val_loader = get_dataloader(split='val',
                                         config=cfg,
                                         shuffle=False,
                                         num_workers=cfg.train.num_workers,
                                         )
        print("Training set size:", self.train_loader.dataset.__len__())
        print("Validate set size:", self.val_loader.dataset.__len__())

        # snapshot
        self.save_dir = os.path.join(cfg.snapshot_root, f'{cfg.stage}/')
        self.tboard_dir = cfg.tensorboard_root

        # evaluate
        self.evaluate_interval = 1


if __name__ == '__main__':
    cfg = make_cfg()
    cfg[cfg.data.dataset] = cfg.copy()

    # snapshot
    if cfg.train.pretrain_model == '':
        experiment_id = time.strftime('%m%d%H%M')
    else:
        experiment_id = cfg.train.pretrain_model.split('/')[1]

    # set seed
    if cfg.data.manual_seed is not None:
        np.random.seed(cfg.data.manual_seed)
        torch.manual_seed(cfg.data.manual_seed)
        torch.cuda.manual_seed_all(cfg.data.manual_seed)
    else:
        print("no seed setting!!!")

    # training
    for stage in cfg.train.all_stage:
        cfg.stage = stage
        snapshot_root = 'snapshot/%s' % experiment_id
        tensorboard_root = 'tensorboard/%s/%s' % (experiment_id, cfg.stage)
        cfg.snapshot_root = snapshot_root
        cfg.tensorboard_root = tensorboard_root

        # pre-training each module
        args = Args(cfg)
        trainer = Trainer(args)
        trainer.train()
