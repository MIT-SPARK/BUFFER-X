import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import time
import torch.nn as nn
from utils.timer import Timer
from KITTI.config import make_cfg
from models.BUFFER import buffer
from utils.SE3 import *
from utils.common import make_open3d_point_cloud
from utils.tools import find_voxel_size
import open3d as o3d

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
        model_path = 'KITTI/snapshot/%s/%s/best.pth' % (experiment_id, stage)
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

    src_path = "../datasets/250118_tiers/tiers_indoor06/vel16/scans/000008.pcd"
    tgt_path = "../datasets/250118_tiers/tiers_indoor06/vel16/vel16_map.pcd"

    trans = np.array([[0.999301, 0.0371835, -0.00398471, 0.650437],
                        [-0.0372082, 0.999288, -0.00629173, 0.679018],
                        [0.00374792, 0.00643559, 0.999972, 0.0701454],
                        [0, 0, 0, 1]])
    
    src_pcd = o3d.io.read_point_cloud(src_path)
    tgt_pcd = o3d.io.read_point_cloud(tgt_path)
    
    ds_size = find_voxel_size(src_pcd)
            
    src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=ds_size) 
    src_pts = np.array(src_pcd.points)
    np.random.shuffle(src_pts)

    tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=ds_size)
    tgt_pts = np.array(tgt_pcd.points)
    np.random.shuffle(tgt_pts)
    
    data_source = {
        'src_pcd_raw': torch.from_numpy(src_pts).float(),
        'tgt_pcd_raw': torch.from_numpy(tgt_pts).float(),
    }
    
    
    overall_time = np.zeros(7)
    with torch.no_grad():
        states = []
        # data_timer.tic()
        # data_source = data_iter.__next__()

        # data_timer.toc()
        # model_timer.tic()
        trans_est, times = model(data_source)
        # model_timer.toc()

        if trans_est is not None:
            trans_est = trans_est
        else:
            trans_est = np.eye(4, 4)

        ####### calculate the recall #######
        rte_thresh = 2.0
        rre_thresh = 5.0
        # trans = data_source['relt_pose'].numpy()
        rte = np.linalg.norm(trans_est[:3, 3] - trans[:3, 3])
        rre = np.arccos(
            np.clip((np.trace(trans_est[:3, :3].T @ trans[:3, :3]) - 1) / 2, -1 + 1e-16, 1 - 1e-16)) * 180 / math.pi
        states.append(np.array([rte < rte_thresh and rre < rre_thresh, rte, rre]))

        fail = False
        if rte > rte_thresh or rre > rre_thresh:
            print(f"Fragment fails, RRE：{rre:.4f}, RTE：{rte:.4f}")
            fail = True
            
        torch.cuda.empty_cache()
        
        src_pts, tgt_pts = data_source['src_pcd_raw'], data_source['tgt_pcd_raw']
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_pts)
        src_pcd.paint_uniform_color([1, 0.706, 0])
        
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pts)
        tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
        
        if not os.path.exists("results_ply"):
            os.makedirs("results_ply")
            
        before_matching = src_pcd + tgt_pcd
        o3d.io.write_point_cloud(f"results_ply/before_matching.ply", before_matching)
        
        src_pcd.transform(trans)
        gt_matching = src_pcd + tgt_pcd
        o3d.io.write_point_cloud(f"results_ply/gt_matching.ply", gt_matching)
        
        src_pcd.transform(np.linalg.inv(trans))
        src_pcd.transform(trans_est)
        pred_matching = src_pcd + tgt_pcd
        result = "Fail" if fail else "Success"
        o3d.io.write_point_cloud(f"results_ply/{result}_pred_matching.ply", pred_matching)
        print(f"Fragment {result}, RRE：{rre:.4f}, RTE：{rte:.4f}")
            
        # temp_states = np.array(states)
        # temp_recall = temp_states[:, 0].sum() / temp_states.shape[0]
        # temp_te = temp_states[temp_states[:, 0] == 1, 1].mean()
        # temp_re = temp_states[temp_states[:, 0] == 1, 2].mean()

# states = np.array(states)
# Recall = states[:, 0].sum() / states.shape[0]
# TE = states[states[:, 0] == 1, 1].mean()
# RE = states[states[:, 0] == 1, 2].mean()
# print()
# print("---------------Test Result---------------")
# print(f'Registration Recall: {Recall:.8f}')
# print(f'RTE: {TE*100:.8f}')
# print(f'RRE: {RE:.8f}')

# average_times = overall_time / num_batch
# print(f"Average data_time: {average_times[0]:.4f}s "
#     f"Average model_time: {average_times[1]:.4f}s ")
# print(f"desc_time: {average_times[2]:.4f}s "
#     f"mutual_matching_time: {average_times[3]:.4f}s "
#     f"inlier_time: {average_times[4]:.4f}s "
#     f"correspondence_proposal_time: {average_times[5]:.4f}s "
#     f"ransac_time: {average_times[6]:.4f}s ")

