import sys

sys.path.append('../')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import math
import time
import argparse
import copy
import numpy as np
from KITTI.config import make_cfg as kitti_make_cfg
from SuperSet_Eval.dataloader import get_dataloader
from ThreeDMatch.config import make_cfg as threedmatch_make_cfg
from Scannetpp_faro.config import make_cfg as scannetpp_faro_make_cfg
from Scannetpp_iphone.config import make_cfg as scannetpp_iphone_make_cfg
from WOD.config import make_cfg as wod_make_cfg
from NewerCollege.config import make_cfg as newercollege_make_cfg
from KimeraMulti.config import make_cfg as kimeramulti_make_cfg
from Tiers.config import make_cfg as tiers_make_cfg
from KAIST.config import make_cfg as kaist_make_cfg
from ETH.config import make_cfg as eth_make_cfg

from SuperSet_Eval.config import make_cfg
from utils.timer import Timer
from models.BUFFER import buffer
from utils.SE3 import *
from models.BUFFER import buffer

from torch import optim

from ThreeDMatch.test import evaluate_registration, read_trajectory, read_trajectory_info

arg_lists = []
parser = argparse.ArgumentParser()

if __name__ == '__main__':
    print("Start testing...")
    cfg = make_cfg()
    indoor_datasets = ['3DMatch', 'Scannetpp_faro', 'Scannetpp_iphone']
    outdoor_datasets = ['KITTI', 'WOD', 'NewerCollege', 'KimeraMulti', 'Tiers', 'KAIST', 'ETH']
    for subsetdataset in cfg.data.subsetdatasets:
        if (subsetdataset == 'KITTI'):
            cfg_tmp = kitti_make_cfg()
        elif (subsetdataset == '3DMatch'):
            cfg_tmp = threedmatch_make_cfg()
        elif (subsetdataset == 'Scannetpp_faro'):
            cfg_tmp = scannetpp_faro_make_cfg()
        elif (subsetdataset == 'Scannetpp_iphone'):
            cfg_tmp = scannetpp_iphone_make_cfg()
        elif (subsetdataset == 'WOD'):
            cfg_tmp = wod_make_cfg()
        elif (subsetdataset == 'NewerCollege'):
            cfg_tmp = newercollege_make_cfg()
        elif (subsetdataset == 'KimeraMulti'):
            cfg_tmp = kimeramulti_make_cfg()
        elif (subsetdataset == 'Tiers'):
            cfg_tmp = tiers_make_cfg()
        elif (subsetdataset == 'KAIST'):
            cfg_tmp = kaist_make_cfg()
        elif (subsetdataset == 'ETH'):
            cfg_tmp = kaist_make_cfg()
        else:
            raise ValueError("Unsupported dataset name has been given")
        cfg_tmp.stage = 'test'
        cfg[subsetdataset] = cfg_tmp
        print(f"Dataset: {subsetdataset} has been loaded")

    # snapshot
    cfg.stage = 'test'
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
    
    recall_list = []
    rte_list = []
    rre_list = []
    
    timestr = time.strftime('%m%d%H%M')
    
    for subsetdataset in cfg.data.subsetdatasets:
        print(f"Start testing {subsetdataset}...")
        cfg[subsetdataset].stage = 'test'
        test_loader = get_dataloader(split='test',
                                     config=cfg,
                                     subsetdataset=subsetdataset,
                                     shuffle=False,
                                     num_workers=cfg.train.num_workers,
                                     )
        print(f"{subsetdataset} Test set size:", test_loader.dataset.__len__())
        data_timer, model_timer = Timer(), Timer()
        
        if subsetdataset in indoor_datasets:
            rte_thresh = 0.3
            rre_thresh = 15
        elif subsetdataset in outdoor_datasets:
            rte_thresh = 2.0
            rre_thresh = 5.0

        overall_time = np.zeros(7)
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
                if subsetdataset == '3DMatch':
                    scene = data_source['src_id'].split('/')[-2]
                    src_id = data_source['src_id'].split('/')[-1].split('_')[-1]
                    tgt_id = data_source['tgt_id'].split('/')[-1].split('_')[-1]
                    logpath = f"log_{subsetdataset}/{scene}"
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
                            
                ####### calculate the recall #######
                # rte_thresh = 2.0
                # rre_thresh = 5.0
                trans = data_source['relt_pose'].numpy()
                rte = np.linalg.norm(trans_est[:3, 3] - trans[:3, 3])
                rre = np.arccos(
                    np.clip((np.trace(trans_est[:3, :3].T @ trans[:3, :3]) - 1) / 2, -1 + 1e-16, 1 - 1e-16)) * 180 / math.pi
                states.append(np.array([rte < rte_thresh and rre < rre_thresh, rte, rre]))

                if rte > rte_thresh or rre > rre_thresh:
                    print(f"{i}th fragment fails, RRE：{rre:.4f}, RTE：{rte:.4f}")
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
                        f"RRE: {temp_re:.4f} "
                        f"data_time: {overall_time[0] / (i + 1):.4f}s "
                        f"model_time: {overall_time[1] / (i + 1):.4f}s ")
        states = np.array(states)
        Recall = states[:, 0].sum() / states.shape[0]
        TE = states[states[:, 0] == 1, 1].mean()
        RE = states[states[:, 0] == 1, 2].mean()
        recall_list.append(Recall)
        rte_list.append(TE)
        rre_list.append(RE)
        if subsetdataset == '3DMatch':
            gtpath = cfg[subsetdataset].data.root + f'/test/{subsetdataset}/gt_result'
            scenes = sorted(os.listdir(gtpath))
            scene_names = [os.path.join(gtpath, ele) for ele in scenes]
            recall_3DMatch = []
            for idx, scene in enumerate(scene_names):
                # ground truth info
                gt_pairs, gt_traj = read_trajectory(os.path.join(scene, "gt.log"))
                n_fragments, gt_traj_cov = read_trajectory_info(os.path.join(scene, "gt.info"))

                # estimated info
                est_pairs, est_traj = read_trajectory(os.path.join(f"log_{subsetdataset}", scenes[idx], f'{timestr}.log'))

                temp_precision, temp_recall, c_flag, errors = evaluate_registration(n_fragments, est_traj,
                                                                                    est_pairs, gt_pairs,
                                                                                    gt_traj, gt_traj_cov)
                recall_3DMatch.append(temp_recall)
            
        
        print()
        print(f"---------------{subsetdataset} Test Result---------------")
        print(f"Registration Recall: {Recall:.8f}")
        if subsetdataset == '3DMatch':
            print(f'Registration Recall (3DMatch setting): {np.array(recall_3DMatch).mean():.8f}')
        print(f"RTE: {TE*100:.8f}")
        print(f"RRE: {RE:.4f}")
       
        
        average_times = overall_time / num_batch
        print(f"Average data_time: {average_times[0]:.4f}s "
            f"Average model_time: {average_times[1]:.4f}s ")
        print(f"desc_time: {average_times[2]:.4f}s "
            f"mutual_matching_time: {average_times[3]:.4f}s "
            f"inlier_time: {average_times[4]:.4f}s "
            f"correspondence_proposal_time: {average_times[5]:.4f}s "
            f"ransac_time: {average_times[6]:.4f}s ")
    
    max_length = max(len(subsetdataset) for subsetdataset in cfg.data.subsetdatasets) + 30  # Adjust padding
    print(f"{' Overall Test Result '.center(max_length, '-')}")
    for i, subsetdataset in enumerate(cfg.data.subsetdatasets):
        title = f" {subsetdataset} Test Result "
        print(f"{title.center(max_length, '-')}")
        print(f'Registration Recall: {recall_list[i]:.8f}')
        if subsetdataset == '3DMatch':
            print(f'Registration Recall (3DMatch setting): {np.array(recall_3DMatch).mean():.8f}')
        print(f'RTE: {rte_list[i]*100:.8f}')
        print(f'RRE: {rre_list[i]:.8f}')
    print('-' * max_length)
    print(f'Overall Registration Recall: {np.mean(recall_list):.8f}')
    
    result_path = f"results/{experiment_id}_{timestr}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(f"{result_path}/result.txt", 'w') as f:
        # Overall Test Result
        f.write(f"{' Overall Test Result '.center(max_length, '-')}\n")
        
        # Subset Test Results
        for i, subsetdataset in enumerate(cfg.data.subsetdatasets):
            title = f" {subsetdataset} Test Result "
            f.write(f"{title.center(max_length, '-')}\n")
            f.write(f'Registration Recall: {recall_list[i]:.8f}\n')
            if subsetdataset == '3DMatch':
                f.write(f'Registration Recall (3DMatch setting): {np.array(recall_3DMatch).mean():.8f}\n')
            f.write(f'RTE: {rte_list[i]*100:.8f}\n')
            f.write(f'RRE: {rre_list[i]:.8f}\n')

        # Overall Registration Recall
        # fill only - until max length
        f.write('-' * max_length + '\n')
        f.write(f"{' Overall Registration Recall '} {np.mean(recall_list):.8f}\n")
        

        


