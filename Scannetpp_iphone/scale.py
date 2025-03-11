import sys

sys.path.append('../')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import time
import torch.nn as nn
import nibabel.quaternions as nq
from utils.timer import Timer
from Scannetpp_iphone.config import make_cfg
from models.BUFFER import buffer
from Scannetpp_iphone.dataloader import get_dataloader
from utils.SE3 import *
import open3d as o3d

from tqdm import tqdm


def read_trajectory(filename, dim=4):
    """
    Function that reads a trajectory saved in the 3DMatch/Redwood format to a numpy array.
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html

    Args:
    filename (str): path to the '.txt' file containing the trajectory data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)

    Returns:
    final_keys (dict): indices of pairs with more than 30% overlap (only this ones are included in the gt file)
    traj (numpy array): gt pairwise transformation matrices for n pairs[n,dim, dim]
    """

    with open(filename) as f:
        lines = f.readlines()

        # Extract the point cloud pairs
        keys = lines[0::(dim + 1)]
        temp_keys = []
        for i in range(len(keys)):
            temp_keys.append(keys[i].split('\t')[0:3])

        final_keys = []
        for i in range(len(temp_keys)):
            final_keys.append(
                [temp_keys[i][0].strip(), temp_keys[i][1].strip(), temp_keys[i][2].strip()])

        traj = []
        for i in range(len(lines)):
            if i % 5 != 0:
                traj.append(lines[i].split('\t')[0:dim])

        traj = np.asarray(traj, dtype=np.float32).reshape(-1, dim, dim)

        final_keys = np.asarray(final_keys)

        return final_keys, traj


if __name__ == '__main__':
    print("Start testing...")
    cfg = make_cfg()
    cfg[cfg.data.dataset] = cfg.copy()
    cfg.stage = 'test'
    timestr = time.strftime('%m%d%H%M')
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

    test_loader = get_dataloader(split='test',
                                 config=cfg,
                                 shuffle=False,
                                 num_workers=cfg.train.num_workers,
                                 )
    print("Test set size:", test_loader.dataset.__len__())
    data_timer, model_timer = Timer(), Timer()

    overall_time = np.zeros(7)
    scene_scale = []
    with torch.no_grad():
        states = []
        num_batch = len(test_loader)
        data_iter = iter(test_loader)
        for i in tqdm(range(num_batch)):
            data_timer.tic()
            data_source = data_iter.__next__()
            
            src_pts = data_source['src_pcd_raw']
            tgt_pts = data_source['tgt_pcd_raw']

            # sample points
            src_max_range = np.max(np.linalg.norm(src_pts, axis=1))
            tgt_max_range = np.max(np.linalg.norm(tgt_pts, axis=1))
            scene_scale.append((src_max_range+tgt_max_range)/2)

        mean_scale = np.mean(scene_scale)
        scene_scale.append(mean_scale)
        save_scale_path = f'scale_analysis_Scannetpp_iphone.txt'
        with open(save_scale_path, 'w') as f:
            f.write(f"Scannetpp_iphone: {mean_scale}\n")
        print(f"Scale analysis has been saved to {save_scale_path}")
            