import sys
import copy
sys.path.append('../')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import time
import torch.nn as nn
import nibabel.quaternions as nq
from utils.timer import Timer
# from ThreeDMatch.config import make_cfg
from config.threedmatch_config import make_cfg
from models.BUFFER import buffer
# from ThreeDMatch.dataloader import get_dataloader
from dataset.dataloader import get_dataloader
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


def read_trajectory_info(filename, dim=6):
    """
    Function that reads the trajectory information saved in the 3DMatch/Redwood format to a numpy array.
    Information file contains the variance-covariance matrix of the transformation paramaters.
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html

    Args:
    filename (str): path to the '.txt' file containing the trajectory information data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)

    Returns:
    n_frame (int): number of fragments in the scene
    cov_matrix (numpy array): covariance matrix of the transformation matrices for n pairs[n,dim, dim]
    """

    with open(filename) as fid:
        contents = fid.readlines()
    n_pairs = len(contents) // 7
    assert (len(contents) == 7 * n_pairs)
    info_list = []
    n_frame = 0

    for i in range(n_pairs):
        frame_idx0, frame_idx1, n_frame = [int(item) for item in contents[i * 7].strip().split()]
        info_matrix = np.concatenate(
            [np.fromstring(item, sep='\t').reshape(1, -1) for item in
             contents[i * 7 + 1:i * 7 + 7]], axis=0)
        info_list.append(info_matrix)

    cov_matrix = np.asarray(info_list, dtype=np.float32).reshape(-1, dim, dim)

    return n_frame, cov_matrix


def computeTransformationErr(trans, info):
    """
    Computer the transformation error as an approximation of the RMSE of corresponding points.
    More informaiton at http://redwood-data.org/indoor/registration.html

    Args:
    trans (numpy array): transformation matrices [n,4,4]
    info (numpy array): covariance matrices of the gt transformation paramaters [n,4,4]

    Returns:
    p (float): transformation error
    """

    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]

    return p.item()


def evaluate_registration(num_fragment, result, result_pairs, gt_pairs, gt, gt_info, err2=0.2):
    """
    Evaluates the performance of the registration algorithm according to the evaluation protocol defined
    by the 3DMatch/Redwood datasets. The evaluation protocol can be found at http://redwood-data.org/indoor/registration.html

    Args:
    num_fragment (int): path to the '.txt' file containing the trajectory information data
    result (numpy array): estimated transformation matrices [n,4,4]
    result_pairs (numpy array): indices of the point cloud for which the transformation matrix was estimated (m,3)
    gt_pairs (numpy array): indices of the ground truth overlapping point cloud pairs (n,3)
    gt (numpy array): ground truth transformation matrices [n,4,4]
    gt_cov (numpy array): covariance matrix of the ground truth transfromation parameters [n,6,6]
    err2 (float): threshold for the RMSE of the gt correspondences (default: 0.2m)

    Returns:
    precision (float): mean registration precision over the scene (not so important because it can be increased see papers)
    recall (float): mean registration recall over the scene (deciding parameter for the performance of the algorithm)
    """

    err2 = err2 ** 2
    gt_mask = np.zeros((num_fragment, num_fragment), dtype=np.int64)
    flags = []

    for idx in range(gt_pairs.shape[0]):
        i = int(gt_pairs[idx, 0])
        j = int(gt_pairs[idx, 1])

        # Only non consecutive pairs are tested
        if j - i > 1:
            gt_mask[i, j] = idx

    n_gt = np.sum(gt_mask > 0)
    transformation_errors = np.full(result_pairs.shape[0], np.nan)

    good = 0
    n_res = 0
    for idx in range(result_pairs.shape[0]):
        i = int(result_pairs[idx, 0])
        j = int(result_pairs[idx, 1])
        pose = result[idx, :, :]

        if gt_mask[i, j] > 0:
            n_res += 1
            gt_idx = gt_mask[i, j]
            p = computeTransformationErr(np.linalg.inv(gt[gt_idx, :, :]) @ pose,
                                         gt_info[gt_idx, :, :])
            transformation_errors[idx] = p
            if p <= err2:
                good += 1
                flags.append(0)
            else:
                flags.append(1)
        else:
            flags.append(2)
    if n_res == 0:
        n_res += 1e6
    precision = good * 1.0 / n_res
    recall = good * 1.0 / n_gt

    return precision, recall, flags, transformation_errors


def extract_corresponding_trajectors(est_pairs, gt_pairs, gt_traj):
    """
    Extract only those transformation matrices from the ground truth trajectory that are also in the estimated trajectory.

    Args:
    est_pairs (numpy array): indices of point cloud pairs with enough estimated overlap [m, 3]
    gt_pairs (numpy array): indices of gt overlaping point cloud pairs [n,3]
    gt_traj (numpy array): 3d array of the gt transformation parameters [n,4,4]

    Returns:
    ext_traj (numpy array): gt transformation parameters for the point cloud pairs from est_pairs [m,4,4]
    """
    ext_traj = np.zeros((len(est_pairs), 4, 4))

    for est_idx, pair in enumerate(est_pairs):
        pair[2] = gt_pairs[0][2]
        gt_idx = np.where((gt_pairs == pair).all(axis=1))[0]

        ext_traj[est_idx, :, :] = gt_traj[gt_idx, :, :]

    return ext_traj


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
        model_path = '../snapshot/%s/%s/best.pth' % (experiment_id, stage)
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

    test_loader = get_dataloader(dataset="3DMatch",
                                 split='test',
                                 config=cfg,
                                 shuffle=False,
                                 num_workers=cfg.train.num_workers,
                                 )
    print("Test set size:", test_loader.dataset.__len__())
    data_timer, model_timer = Timer(), Timer()

    overall_time = np.zeros(7)
    fail_list = []
    with torch.no_grad():
        states = []
        num_batch = len(test_loader)
        data_iter = iter(test_loader)
        for i in range(num_batch):
            data_timer.tic()
            data_source = data_iter.__next__()

            data_timer.toc()
            model_timer.tic()
            
            trans_est, times = model(data_source)
            model_timer.toc()

            if trans_est is not None:
                trans_est = trans_est
            else:
                trans_est = np.eye(4, 4)

            scene = data_source['src_id'].split('/')[-2]
            src_id = data_source['src_id'].split('/')[-1].split('_')[-1]
            tgt_id = data_source['tgt_id'].split('/')[-1].split('_')[-1]
            logpath = f"log_{cfg.data.benchmark}/{scene}"
            if not os.path.exists(logpath):
                os.makedirs(logpath)
            # write the transformation matrix into .log file for evaluation.
            with open(os.path.join(logpath, f'{timestr}.log'), 'a+') as f:
                trans = np.linalg.inv(trans_est)
                s1 = f'{src_id}\t {tgt_id}\t  1\n'
                f.write(s1)
                f.write(f"{trans[0, 0]}\t {trans[0, 1]}\t {trans[0, 2]}\t {trans[0, 3]}\t \n")
                f.write(f"{trans[1, 0]}\t {trans[1, 1]}\t {trans[1, 2]}\t {trans[1, 3]}\t \n")
                f.write(f"{trans[2, 0]}\t {trans[2, 1]}\t {trans[2, 2]}\t {trans[2, 3]}\t \n")
                f.write(f"{trans[3, 0]}\t {trans[3, 1]}\t {trans[3, 2]}\t {trans[3, 3]}\t \n")

            ####### calculate the recall of DGR #######
            rte_thresh = 0.3
            rre_thresh = 15
            trans = data_source['relt_pose'].numpy()
            rte = np.linalg.norm(trans_est[:3, 3] - trans[:3, 3])
            rre = np.arccos(
                np.clip((np.trace(trans_est[:3, :3].T @ trans[:3, :3]) - 1) / 2, -1 + 1e-16, 1 - 1e-16)) * 180 / math.pi
            states.append(np.array([rte < rte_thresh and rre < rre_thresh, rte, rre]))
            
            fail = False
            if rte > rte_thresh or rre > rre_thresh:
                print(f"{i}th fragment fails, RRE：{rre}, RTE：{rte}")
                fail_list.append([scene, src_id, tgt_id])
                fail = True
            overall_time += np.array([data_timer.diff, model_timer.diff, *times])
            torch.cuda.empty_cache()
            
            # if i % 10 == 0:
            #     src_pts, tgt_pts = data_source['src_pcd_real_raw'], data_source['tgt_pcd_real_raw']
            #     src_pcd = o3d.geometry.PointCloud()
            #     src_pcd.points = o3d.utility.Vector3dVector(src_pts)
            #     src_pcd.paint_uniform_color([1, 0.706, 0])
                
            #     src_gray_pts = data_source['src_pcd_real_raw']
            #     src_gray_pcd = o3d.geometry.PointCloud()
            #     src_gray_pcd.paint_uniform_color([0.5, 0.5, 0.5])
                
            #     tgt_pcd = o3d.geometry.PointCloud()
            #     tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pts)
            #     tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
                
            #     pair_name = f"{scene}_{src_id}_{tgt_id}"
            #     ply_path = f"../results_ply/3DLoMatch/"
            #     if not os.path.exists(ply_path):
            #         os.makedirs(ply_path)
                
            #     before_matching = src_pcd + tgt_pcd
            #     o3d.io.write_point_cloud(f"../results_ply/3DLoMatch/{pair_name}_before_matching.ply", before_matching)
                
            #     before_matching_gray = src_gray_pcd + tgt_pcd
            #     o3d.io.write_point_cloud(f"../results_ply/3DLoMatch/{pair_name}_before_matching_gray.ply", before_matching_gray)
                
            #     src_pcd.transform(trans)
            #     gt_matching = src_pcd + tgt_pcd
            #     o3d.io.write_point_cloud(f"../results_ply/3DLoMatch/{pair_name}_gt_matching.ply", gt_matching)
                
            #     src_pcd.transform(np.linalg.inv(trans))
            #     src_pcd.transform(trans_est)
            #     pred_matching = src_pcd + tgt_pcd
            #     result = "Fail" if fail else "Success"
            #     o3d.io.write_point_cloud(f"../results_ply/3DLoMatch/{pair_name}_{result}_pred_matching.ply", pred_matching)
            
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
    fail_pairs_save_path = "fail_pairs.txt"
    with open(fail_pairs_save_path, 'w') as f:
        for fail_pair in fail_list:
            f.write(f"{fail_pair[0]}: {fail_pair[1]} {fail_pair[2]}\n")
    states = np.array(states)
    Recall = states[:, 0].sum() / states.shape[0]
    TE = states[states[:, 0] == 1, 1].mean()
    RE = states[states[:, 0] == 1, 2].mean()

    # calculate Registration Recall
    if cfg.data.benchmark == '3DMatch':
        gtpath = cfg.data.root + f'/test/{cfg.data.benchmark}/gt_result'
    elif cfg.data.benchmark == '3DLoMatch':
        gtpath = cfg.data.root + f'/test/{cfg.data.benchmark}'
    scenes = sorted(os.listdir(gtpath))
    scene_names = [os.path.join(gtpath, ele) for ele in scenes]
    recall = []
    
    scene_recall_path = f"scene_recall/{experiment_id}/{timestr}.txt"
    if not os.path.exists(f"scene_recall/{experiment_id}"):
        os.makedirs(f"scene_recall/{experiment_id}")
    with open(scene_recall_path, 'w') as f:
        for idx, scene in enumerate(scene_names):
            scene_name = scene.split('/')[-1]
            # ground truth info
            gt_pairs, gt_traj = read_trajectory(os.path.join(scene, "gt.log"))
            n_fragments, gt_traj_cov = read_trajectory_info(os.path.join(scene, "gt.info"))

            # estimated info
            est_pairs, est_traj = read_trajectory(os.path.join(f"log_{cfg.data.benchmark}", scenes[idx], f'{timestr}.log'))
            # est_pairs, est_traj = read_trajectory(os.path.join(f"3DMatch_5000_prob", scenes[idx], f'est.log'))
            temp_precision, temp_recall, c_flag, errors = evaluate_registration(n_fragments, est_traj,
                                                                                est_pairs, gt_pairs,
                                                                                    gt_traj, gt_traj_cov)
            recall.append(temp_recall)
            f.write(f"{scene_name}: {temp_recall:.8f}\n")
        f.write(f"Average Recall: {np.array(recall).mean():.8f}")
    print()
    print("---------------Test Result---------------")
    print(f'Registration Recall: {Recall:.8f}')
    print(f'Registration Recall (3DMatch setting): {np.array(recall).mean():.8f}')
    print(f'RTE: {TE*100:.8f}')
    print(f'RRE: {RE:.8f}')
    
    average_times = overall_time / num_batch
    print(f"Average data_time: {average_times[0]:.4f}s "
        f"Average model_time: {average_times[1]:.4f}s ")
    print(f"desc_time: {average_times[2]:.4f}s "
        f"mutual_matching_time: {average_times[3]:.4f}s "
        f"inlier_time: {average_times[4]:.4f}s "
        f"correspondence_proposal_time: {average_times[5]:.4f}s "
        f"ransac_time: {average_times[6]:.4f}s ")