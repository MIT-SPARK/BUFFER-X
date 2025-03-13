import sys

sys.path.append('../')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import time
import torch.nn as nn
from utils.timer import Timer
# from Tiers.config import make_cfg
from config.tiers_config import make_cfg
from models.BUFFER import buffer
from utils.SE3 import *
# from Tiers.dataloader import get_dataloader
from dataset.dataloader import get_dataloader
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

    # pdist_list = np.arange(1, 6) * 0.5
    pdist_list = [2]
    for pdist in pdist_list:
        print (f"Testing with pdist: {pdist}m")
        cfg.data.pdist = pdist
        test_loader = get_dataloader(dataset = "Tiers",
                                    split='test',
                                    config=cfg,
                                    shuffle=False,
                                    num_workers=cfg.train.num_workers,
                                    )
        print("Test set size:", test_loader.dataset.__len__())
        data_timer, model_timer = Timer(), Timer()

        overall_time = np.zeros(7)
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
                trans_est, times = model(data_source)
                model_timer.toc()

                if trans_est is not None:
                    trans_est = trans_est
                else:
                    trans_est = np.eye(4, 4)

                ####### calculate the recall #######
                rte_thresh = 2.0
                rre_thresh = 5.0
                trans = data_source['relt_pose'].numpy()
                rte = np.linalg.norm(trans_est[:3, 3] - trans[:3, 3])
                rre = np.arccos(
                    np.clip((np.trace(trans_est[:3, :3].T @ trans[:3, :3]) - 1) / 2, -1 + 1e-16, 1 - 1e-16)) * 180 / math.pi
                states.append(np.array([rte < rte_thresh and rre < rre_thresh, rte, rre]))

                fail = False
                if rte > rte_thresh or rre > rre_thresh:
                    print(f"{i}th fragment fails, RRE：{rre:.4f}, RTE：{rte:.4f}")
                    fail = True
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
        states = np.array(states)
        Recall = states[:, 0].sum() / states.shape[0]
        TE = states[states[:, 0] == 1, 1].mean()
        RE = states[states[:, 0] == 1, 2].mean()
        print()
        print("---------------Test Result---------------")
        print(f'Registration Recall: {Recall:.8f}')
        print(f'RTE: {TE*100:.8f} (cm)') ## cm
        print(f'RRE: {RE:.8f}')
        
        average_times = overall_time / num_batch
        print(f"Average data_time: {average_times[0]:.4f}s "
            f"Average model_time: {average_times[1]:.4f}s ")
        print(f"desc_time: {average_times[2]:.4f}s "
            f"mutual_matching_time: {average_times[3]:.4f}s "
            f"inlier_time: {average_times[4]:.4f}s "
            f"correspondence_proposal_time: {average_times[5]:.4f}s "
            f"ransac_time: {average_times[6]:.4f}s ")

        result_path = f"results/pdist/{pdist}m.txt"
        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))
        with open(result_path, 'w') as f:
            f.write("---------------Test Result---------------\n")
            f.write(f'Registration Recall: {Recall:.8f}\n')
            f.write(f'RTE: {TE*100:.8f}\n')
            f.write(f'RRE: {RE:.8f}\n')
            f.write(f"Average data_time: {average_times[0]:.4f}s\n")
            f.write(f"Average model_time: {average_times[1]:.4f}s\n")
            f.write(f"desc_time: {average_times[2]:.4f}s\n")
            f.write(f"mutual_matching_time: {average_times[3]:.4f}s\n")
            f.write(f"inlier_time: {average_times[4]:.4f}s\n")
            f.write(f"correspondence_proposal_time: {average_times[5]:.4f}s\n")
            f.write(f"ransac_time: {average_times[6]:.4f}s\n")