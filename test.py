import argparse
import sys
import os
# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import time
import torch
import torch.nn as nn
import numpy as np
from utils.timer import Timer
from utils.SE3 import *
from utils.tools import *
from config import make_cfg
from dataset.dataloader import get_dataloader
from models.BUFFER import buffer
import open3d as o3d

# Argument parser
parser = argparse.ArgumentParser(description="Generalized Testing Script for Registration Models")
parser.add_argument("--dataset", type=str, required=True, choices=[
    "3DMatch", "3DLoMatch", "Scannetpp_iphone", "Scannetpp_faro", "Tiers",
    "KITTI", "WOD", "MIT", "KAIST", "ETH", "Oxford"
], help="Dataset to test on")
parser.add_argument("--experiment_id", type=str, default=None,
                    help="Optional experiment ID (default: uses config.test.experiment_id)")
args = parser.parse_args()

if __name__ == '__main__':
    print(f"Start testing on {args.dataset}...")

    # Load dataset-specific config
    cfg = make_cfg(args.dataset)
    cfg[cfg.data.dataset] = cfg.copy()
    cfg.stage = 'test'
    timestr = time.strftime('%m%d%H%M')

    # Initialize model
    model = buffer(cfg)
    experiment_id = args.experiment_id if args.experiment_id else cfg.test.experiment_id

    # Load model weights
    for stage in cfg.train.all_stage:
        model_path = f'snapshot/{experiment_id}/{stage}/best.pth'
        state_dict = torch.load(model_path, map_location='cuda')
        new_dict = {k: v for k, v in state_dict.items() if stage in k}
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {stage} model from {model_path}")

    # Model Parameter Info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params / 1e6:.2f}M")

    model = nn.DataParallel(model, device_ids=[0])
    model.eval()

    # Load test dataset
    load_dataset = "3DMatch" if args.dataset == "3DLoMatch" else args.dataset
    test_loader = get_dataloader(dataset=load_dataset,
                                 split='test',
                                 config=cfg,
                                 shuffle=False,
                                 num_workers=cfg.train.num_workers)

    print(f"Test set size: {len(test_loader.dataset)}")
    data_timer, model_timer = Timer(), Timer()

    # Run test
    overall_time = np.zeros(7)
    with torch.no_grad():
        states = []
        num_batch = len(test_loader)
        data_iter = iter(test_loader)
        est_poses = {}

        for i in range(num_batch):
            data_timer.tic()
            data_source = next(data_iter)
            data_timer.toc()

            model_timer.tic()
            trans_est, times = model(data_source)
            model_timer.toc()

            trans_est = trans_est if trans_est is not None else np.eye(4)
            
            if cfg.data.dataset == "3DMatch":
                scene = data_source['src_id'].split('/')[-2]
                src_id = data_source['src_id'].split('/')[-1].split('_')[-1]
                tgt_id = data_source['tgt_id'].split('/')[-1].split('_')[-1]
                logpath = f"logs/log_{cfg.data.benchmark}/{scene}"
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

            ####### Evaluation #######
            rte_thresh, rre_thresh = cfg.test.rte_thresh, cfg.test.rre_thresh
            trans = data_source['relt_pose'].numpy()
            rte = np.linalg.norm(trans_est[:3, 3] - trans[:3, 3])
            rre = np.arccos(np.clip((np.trace(trans_est[:3, :3].T @ trans[:3, :3]) - 1) / 2, -1 + 1e-16, 1 - 1e-16)) * 180 / math.pi
            states.append([rte < rte_thresh and rre < rre_thresh, rte, rre])

            if rte > rte_thresh or rre > rre_thresh:
                print(f"{i}th fragment failed, RRE: {rre:.4f}, RTE: {rte:.4f}")

            overall_time += np.array([data_timer.diff, model_timer.diff, *times])
            torch.cuda.empty_cache()

            # Print progress every 100 iterations
            if (i + 1) % 100 == 0 or i == num_batch - 1:
                temp_states = np.array(states)
                temp_recall = temp_states[:, 0].sum() / temp_states.shape[0]
                temp_te = temp_states[temp_states[:, 0] == 1, 1].mean()
                temp_re = temp_states[temp_states[:, 0] == 1, 2].mean()
                print(f"[{i + 1}/{num_batch}] Recall: {temp_recall:.4f} RTE: {temp_te:.4f} RRE: {temp_re:.4f}"
                      f" Data time: {data_timer.diff:.4f}s Model time: {model_timer.diff:.4f}s")

    states = np.array(states)
    recall = states[:, 0].sum() / states.shape[0]
    rte_mean = states[states[:, 0] == 1, 1].mean()
    rre_mean = states[states[:, 0] == 1, 2].mean()
    
    if cfg.data.dataset == '3DMatch':
        if cfg.data.benchmark == '3DMatch':
            gtpath = cfg.data.root + f'/test/{cfg.data.benchmark}/gt_result'
        elif cfg.data.benchmark == '3DLoMatch':
            gtpath = cfg.data.root + f'/test/{cfg.data.benchmark}'
        scenes = sorted(os.listdir(gtpath))
        scene_names = [os.path.join(gtpath, ele) for ele in scenes]
        rmse_recall = []
        
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
                est_pairs, est_traj = read_trajectory(os.path.join(f"logs/log_{cfg.data.benchmark}", scenes[idx], f'{timestr}.log'))
                temp_precision, temp_recall, c_flag, errors = evaluate_registration(n_fragments, est_traj,
                                                                                    est_pairs, gt_pairs,
                                                                                    gt_traj, gt_traj_cov)
                rmse_recall.append(temp_recall)
            

    # Print summary
    print("\n---------------Test Results---------------")
    print(f"Recall: {recall:.8f}")
    if cfg.data.dataset == '3DMatch':
        print(f'Registration Recall (3DMatch setting): {np.array(rmse_recall).mean():.8f}')
    print(f"RTE: {rte_mean*100:.8f}")
    print(f"RRE: {rre_mean:.8f}")

    average_times = overall_time / num_batch
    print(f"Average data_time: {average_times[0]:.4f}s "
          f"Average model_time: {average_times[1]:.4f}s ")