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

    overall_time = np.zeros(10)
    with torch.no_grad():
        states = []
        num_batch = len(test_loader)
        data_iter = iter(test_loader)
        for i in range(num_batch):
            fail = False
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

            scene = data_source['src_id'].split('/')[-2]
            src_id = data_source['src_id'].split('/')[-1].split('_')[-1]
            tgt_id = data_source['tgt_id'].split('/')[-1].split('_')[-1]

            ####### calculate the recall of DGR #######
            rte_thresh = 0.3
            rre_thresh = 15
            trans = data_source['relt_pose'].numpy()
            rte = np.linalg.norm(trans_est[:3, 3] - trans[:3, 3])
            rre = np.arccos(
                np.clip((np.trace(trans_est[:3, :3].T @ trans[:3, :3]) - 1) / 2, -1 + 1e-16, 1 - 1e-16)) * 180 / math.pi
            states.append(np.array([rte < rte_thresh and rre < rre_thresh, rte, rre]))

            if rte > rte_thresh or rre > rre_thresh:
                print(f"{i}th fragment fails, RRE：{rre}, RTE：{rte}")
                fail = True
            overall_time += np.array([data_timer.diff, model_timer.diff, *times])
            torch.cuda.empty_cache()
            
            if fail == False:
                src_pts, tgt_pts = data_source['src_pcd_raw'], data_source['tgt_pcd_raw']
                src_pcd = o3d.geometry.PointCloud()
                src_pcd.points = o3d.utility.Vector3dVector(src_pts)
                src_pcd.paint_uniform_color([1, 0.706, 0])
                
                tgt_pcd = o3d.geometry.PointCloud()
                tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pts)
                tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
                
                before_matching = src_pcd + tgt_pcd
                o3d.io.write_point_cloud(f"results/{i}_before_matching.ply", before_matching)
                
                src_pcd.transform(trans)
                gt_matching = src_pcd + tgt_pcd
                o3d.io.write_point_cloud(f"results/{i}_gt_matching.ply", gt_matching)
                
                src_pcd.transform(np.linalg.inv(trans))
                src_pcd.transform(trans_est)
                pred_matching = src_pcd + tgt_pcd
                result = "Fail" if fail else "Success"
                o3d.io.write_point_cloud(f"results/{i}_{result}_pred_matching.ply", pred_matching)
            
            if (i + 1) % 100 == 0 or i == num_batch - 1:
                temp_states = np.array(states)
                temp_recall = temp_states[:, 0].sum() / temp_states.shape[0]
                temp_te = temp_states[temp_states[:, 0] == 1, 1].mean()
                temp_re = temp_states[temp_states[:, 0] == 1, 2].mean()
                print(f"[{i + 1}/{num_batch}] "
                      f"Registration Recall: {temp_recall:.4f} "
                      f"RTE: {temp_te:.4f} "
                      f"RRE: {temp_re:.4f} ")

    states = np.array(states)
    Recall = states[:, 0].sum() / states.shape[0]
    TE = states[states[:, 0] == 1, 1].mean()
    RE = states[states[:, 0] == 1, 2].mean()

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