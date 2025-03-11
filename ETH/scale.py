import sys

sys.path.append('../')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import time
import torch.nn as nn
from utils.timer import Timer
from ETH.config import make_cfg
from models.BUFFER import buffer
from utils.SE3 import *
from ETH.dataloader import get_dataloader


if __name__ == '__main__':
    cfg = make_cfg()
    cfg.stage = 'test'
    timestr = time.strftime('%m%d%H%M')
    model = buffer(cfg)

    experiment_id = cfg.test.experiment_id

    for stage in cfg.test.all_stage:
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

    test_loader = get_dataloader(split='test',
                                 config=cfg,
                                 shuffle=False,
                                 num_workers=16,
                                 )
    print("Test set size:", test_loader.dataset.__len__())
    data_timer, model_timer = Timer(), Timer()
    scale_file = f'scale.txt'
    with open(scale_file, 'w') as f:
        with torch.no_grad():
            states = []
            num_batch = len(test_loader)
            data_iter = iter(test_loader)
            scale_list = []
            for i in range(num_batch):
                data_timer.tic()
                data_source = data_iter.__next__()
                src_pts = data_source['src_pcd']
                tgt_pts = data_source['tgt_pcd']
                
                # sample points
                src_max_range = np.max(np.linalg.norm(src_pts, axis=1))
                tgt_max_range = np.max(np.linalg.norm(tgt_pts, axis=1))
                scale_list.append((src_max_range+tgt_max_range)/2)
            mean_scale = np.mean(scale_list)
            f.write(f"Mean scale: {mean_scale}\n")
            print(f"Mean scale: {mean_scale}")
                
           
       
          