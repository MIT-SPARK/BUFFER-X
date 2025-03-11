import sys

sys.path.append('../')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import math
import time
import torch.nn as nn
from utils.timer import Timer
from ETH.config import make_cfg
from models.BUFFER import buffer
from utils.SE3 import *
from ETH.dataloader import get_dataloader
import open3d as o3d


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

    with torch.no_grad():
        states = []
        num_batch = len(test_loader)
        data_iter = iter(test_loader)
        for i in range(num_batch):
            data_timer.tic()
            data_source = data_iter.__next__()

            data_timer.toc()
            model_timer.tic()
            trans_est, _ = model(data_source)
            model_timer.toc()

            if trans_est is not None:
                trans_est = trans_est
            else:
                trans_est = np.eye(4, 4)

            ####### calculate the recall of DGR #######
            rte_thresh = 0.3
            rre_thresh = 2
            trans = data_source['relt_pose'].numpy()
            rte = np.linalg.norm(trans_est[:3, 3] - trans[:3, 3])
            rre = np.arccos(
                np.clip((np.trace(trans_est[:3, :3].T @ trans[:3, :3]) - 1) / 2, -1 + 1e-16, 1 - 1e-16)) * 180 / math.pi
            states.append(np.array([rte < rte_thresh and rre < rre_thresh, rte, rre]))
            fail = False
            if rte > rte_thresh or rre > rre_thresh:
                print(f"{i}th fragment fails, RRE：{rre}, RTE：{rte}")
                fail = True

            torch.cuda.empty_cache()
            
            scene = data_source['src_id'].split('_')[0]
            src_id = data_source['src_id'].split('/')[-1].split('_')[-1]
            tgt_id = data_source['tgt_id'].split('/')[-1].split('_')[-1]
            if i % 10 == 0:
                src_pts, tgt_pts = data_source['src_pcd_real_raw'], data_source['tgt_pcd_real_raw']
                src_pcd = o3d.geometry.PointCloud()
                src_pcd.points = o3d.utility.Vector3dVector(src_pts)
                src_pcd.paint_uniform_color([1, 0.706, 0])
                
                src_gray_pts = data_source['src_pcd_real_raw']
                src_gray_pcd = o3d.geometry.PointCloud()
                src_gray_pcd.paint_uniform_color([0.5, 0.5, 0.5])
                
                tgt_pcd = o3d.geometry.PointCloud()
                tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pts)
                tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
                
                pair_name = f"{scene}_{src_id}_{tgt_id}"
                ply_path = f"../results_ply/ETH/"
                if not os.path.exists(ply_path):
                    os.makedirs(ply_path)
                
                before_matching = src_pcd + tgt_pcd
                o3d.io.write_point_cloud(f"../results_ply/ETH/{pair_name}_before_matching.ply", before_matching)
                
                before_matching_gray = src_gray_pcd + tgt_pcd
                o3d.io.write_point_cloud(f"../results_ply/ETH/{pair_name}_before_matching_gray.ply", before_matching_gray)
                
                src_pcd.transform(trans)
                gt_matching = src_pcd + tgt_pcd
                o3d.io.write_point_cloud(f"../results_ply/ETH/{pair_name}_gt_matching.ply", gt_matching)
                
                src_pcd.transform(np.linalg.inv(trans))
                src_pcd.transform(trans_est)
                pred_matching = src_pcd + tgt_pcd
                result = "Fail" if fail else "Success"
                o3d.io.write_point_cloud(f"../results_ply/ETH/{pair_name}_{result}_pred_matching.ply", pred_matching)
            
            if (i + 1) % 100 == 0 or i == num_batch - 1:
                temp_states = np.array(states)
                temp_recall = temp_states[:, 0].sum() / temp_states.shape[0]
                temp_te = temp_states[temp_states[:, 0] == 1, 1].mean()
                temp_re = temp_states[temp_states[:, 0] == 1, 2].mean()
                print(f"[{i + 1}/{num_batch}] "
                      f"Registration Recall: {temp_recall:.4f} "
                      f"RTE: {temp_te:.4f} "
                      f"RRE: {temp_re:.4f} "
                      f"data_time: {data_timer.diff:.4f}s "
                      f"model_time: {model_timer.diff:.4f}s ")

    states = np.array(states)
    Recall = states[:, 0].sum() / states.shape[0]
    TE = states[states[:, 0] == 1, 1].mean()
    RE = states[states[:, 0] == 1, 2].mean()
    
    print("---------------Test Result---------------")
    print(f'Registration Recall: {Recall:.8f}')
    print(f'RTE: {TE*100:.8f}')
    print(f'RRE: {RE:.8f}')
