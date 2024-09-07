import sys

sys.path.append('../')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import math
import time
import torch.nn as nn
import nibabel.quaternions as nq
import open3d as o3d
from utils.timer import Timer
from NSS.config import make_cfg
from models.BUFFER import buffer
from NSS.dataloader import get_dataloader
from utils.SE3 import *

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

    overall_time = np.zeros(10)
    with torch.no_grad():
        states = []
        num_batch = len(test_loader)
        data_iter = iter(test_loader)
        for i in range(num_batch):
            data_timer.tic()
            data_source = data_iter.__next__()

            data_timer.toc()
            model_timer.tic()
            trans_est, src_axis, tgt_axis, times = model(data_source)
            model_timer.toc()

            if trans_est is not None:
                trans_est = trans_est
            else:
                trans_est = np.eye(4, 4)
            ####### calculate the recall of DGR #######
            rte_thresh = 0.1
            rre_thresh = 10
            trans = data_source['relt_pose'].numpy()
            rte = np.linalg.norm(trans_est[:3, 3] - trans[:3, 3])
            rre = np.arccos(
                np.clip((np.trace(trans_est[:3, :3].T @ trans[:3, :3]) - 1) / 2, -1 + 1e-16, 1 - 1e-16)) * 180 / math.pi
            states.append(np.array([rte < rte_thresh and rre < rre_thresh, rte, rre]))
            flag = True
            if rte > rte_thresh or rre > rre_thresh:
                print(f"{i}th fragment fails, RRE：{rre}, RTE：{rte}")
                flag = False
            overall_time += np.array([data_timer.diff, model_timer.diff, *times])
            torch.cuda.empty_cache()
            if (i + 1) % 100 == 0 or i == num_batch - 1:
                temp_states = np.array(states)
                temp_recall = temp_states[:, 0].sum() / temp_states.shape[0]
                temp_te = temp_states[temp_states[:, 0] == 1, 1].mean()
                temp_re = temp_states[temp_states[:, 0] == 1, 2].mean()
                print(f"[{i + 1}/{num_batch}] "
                      f"Registration Recall: {temp_recall:.4f} "
                      f"RTE: {temp_te:.4f} "
                      f"RRE: {temp_re:.4f} ")
            # src_pcd_name = data_source['src_id']
            # tgt_pcd_name = data_source['tgt_id']
            # src_pcd_path = os.path.join(cfg.data.root, 'point_cloud', src_pcd_name)
            # tgt_pcd_path = os.path.join(cfg.data.root, 'point_cloud', tgt_pcd_name)
            
            # src_pcd_gt = o3d.io.read_point_cloud(src_pcd_path)
            # src_pcd_est = o3d.io.read_point_cloud(src_pcd_path)
            # tgt_pcd = o3d.io.read_point_cloud(tgt_pcd_path)
            
            # src_pcd_gt.transform(trans)
            # src_pcd_est.transform(trans_est)
            # src_pcd_gt.paint_uniform_color([1, 0, 0])
            # src_pcd_est.paint_uniform_color([1, 0, 0])
            # tgt_pcd.paint_uniform_color([0, 0, 1])
            
            # gt_pcd = src_pcd_gt + tgt_pcd
            # o3d.io.write_point_cloud(f"results/{i}_gt.ply", gt_pcd)
            # est_pcd = src_pcd_est + tgt_pcd
            # o3d.io.write_point_cloud(f"results/{i}_est_{flag}.ply", est_pcd)
            
    states = np.array(states)
    Recall = states[:, 0].sum() / states.shape[0]
    TE = states[states[:, 0] == 1, 1].mean()
    RE = states[states[:, 0] == 1, 2].mean()

    print()
    print("---------------Test Result---------------")
    print(f'Registration Recall: {Recall:.4f}')
    print(f'RTE: {TE:.4f}')
    print(f'RRE: {RE:.4f}')
    
    average_times = overall_time / num_batch
    print(f"Average data_time: {average_times[0]:.4f}s "
        f"Average model_time: {average_times[1]:.4f}s ")
    print(f"ref_time: {average_times[2]:.4f}s "
        f"keypt_time: {average_times[3]:.4f}s "
        f"fps_time: {average_times[4]:.4f}s "
        f"desc_time: {average_times[5]:.4f}s "
        f"mutual_matching_time: {average_times[6]:.4f}s "
        f"inlier_time: {average_times[7]:.4f}s "
        f"correspondence_proposal_time: {average_times[8]:.4f}s "
        f"ransac_time: {average_times[9]:.4f}s ")