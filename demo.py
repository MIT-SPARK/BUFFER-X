import os
import math
import time
import torch.nn as nn
from config import make_cfg
from utils.timer import Timer
from models.BUFFERX import BufferX
from utils.SE3 import *
from utils.tools import sphericity_based_voxel_analysis
import open3d as o3d

if __name__ == '__main__':
    print("Start testing...")
    cfg = make_cfg("3DMatch")
    cfg[cfg.data.dataset] = cfg.copy()
    cfg.stage = 'test'
    timestr = time.strftime('%m%d%H%M')
    model = BufferX(cfg)
    model_timer = Timer()

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

    # src_path = "/root/dataset/modelnet_ply/cup/cup_0080.ply"
    # tgt_path = "/root/dataset/modelnet_ply/cup/cup_0080.ply"
    
    pcd_data = o3d.data.DemoICPPointClouds()
    src_path = pcd_data.paths[0]
    tgt_path = pcd_data.paths[1]

    trans = np.array([[0.0, -1.0, 0.0, 0.3],
                   [1.0, 0.0, 0.0, 0.5],
                   [0.0, 0.0, 1.0, 0.1],
                   [0, 0, 0, 1]])
    
    src_pcd = o3d.io.read_point_cloud(src_path)
    tgt_pcd = o3d.io.read_point_cloud(tgt_path)
    tgt_pcd = tgt_pcd.transform(trans)
    
    ds_size, sphericity, is_aligned_to_global_z = sphericity_based_voxel_analysis(src_pcd, tgt_pcd)
            
    src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=ds_size) 
    src_pts = np.array(src_pcd.points)
    np.random.shuffle(src_pts)

    tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=ds_size)
    tgt_pts = np.array(tgt_pcd.points)
    np.random.shuffle(tgt_pts)
    
    data_source = {
        'src_fds_pcd': torch.from_numpy(src_pts).float(),
        'tgt_fds_pcd': torch.from_numpy(tgt_pts).float(),
        'is_aligned_to_global_z': is_aligned_to_global_z,
    }
    
    overall_time = np.zeros(7)
    with torch.no_grad():
        states = []
        model_timer.tic()
        trans_est, times = model(data_source)
        model_timer.toc()

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
            fail = True
            
        torch.cuda.empty_cache()
        
        src_pts, tgt_pts = data_source['src_fds_pcd'], data_source['tgt_fds_pcd']
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

