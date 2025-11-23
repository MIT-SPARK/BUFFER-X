import argparse
import os

import time
import torch
import torch.nn as nn
import numpy as np
from scipy.io import savemat
from utils.timer import Timer
from utils.SE3 import compute_rte, compute_rre
from utils.tools import evaluate_registration, read_trajectory, read_trajectory_info, setup_logger
from config import make_cfg
from dataset.dataloader import get_dataloader
from models.BUFFERX import BufferX
from tabulate import tabulate


def run(args, timestr, experiment_id, dataset_name):
    log_file = f"logs/test/{experiment_id}/{dataset_name}_{timestr}.log"
    os.makedirs(f"logs/test/{experiment_id}", exist_ok=True)
    logger = setup_logger(log_file)
    logger.info(f"Start testing on {dataset_name}...")

    # Load dataset-specific config
    cfg = make_cfg(dataset_name, args.root_dir)
    cfg[cfg.data.dataset] = cfg.copy()
    cfg.stage = "test"

    if dataset_name.endswith("_hetero"):
        logger.info(f"Heterogeneous evaluation: {cfg.data.src_sensor} -> {cfg.data.tgt_sensor}")

    # Overwrite config with command-line arguments if provided
    if args.num_points_per_patch is not None:
        cfg.patch.num_points_per_patch = args.num_points_per_patch
        logger.info(f"Overwriting num_points_per_patch: {args.num_points_per_patch}")
    if args.num_scales is not None:
        cfg.patch.num_scales = args.num_scales
        logger.info(f"Overwriting num_scales: {args.num_scales}")
    if args.num_fps is not None:
        cfg.patch.num_fps = args.num_fps
        logger.info(f"Overwriting num_fps: {args.num_fps}")
    if args.search_radius_thresholds is not None:
        cfg.patch.search_radius_thresholds = args.search_radius_thresholds
        logger.info(f"Overwriting search_radius_thresholds: {args.search_radius_thresholds}")
    if args.pose_estimator is not None:
        cfg.match.pose_estimator = args.pose_estimator
        logger.info(f"Overwriting pose_estimator: {args.pose_estimator}")

    # Initialize model
    # TODO(hlim): If `cfg` specifies a different model, the model can be changed.
    # We might need an option to fix the model across all scenes.
    model = BufferX(cfg)

    # Load model weights
    for stage in cfg.train.all_stage:
        model_path = f"snapshot/{experiment_id}/{stage}/best.pth"
        state_dict = torch.load(model_path, map_location="cuda")
        new_dict = {k: v for k, v in state_dict.items() if stage in k}
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded {stage} model from {model_path}")

    # Model Parameter Info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of trainable parameters: {total_params / 1e6:.2f}M")

    model = nn.DataParallel(model, device_ids=[0])
    model.eval()

    # Load test dataset
    load_dataset = "3DMatch" if dataset_name == "3DLoMatch" else dataset_name
    test_loader = get_dataloader(
        dataset=load_dataset,
        split="test",
        config=cfg,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    logger.info(f"Test set size: {len(test_loader.dataset)}")
    data_timer, model_timer = Timer(), Timer()

    # Create directory for per-sample results
    results_dir = f"per_sample_results/{experiment_id}"
    os.makedirs(results_dir, exist_ok=True)

    # Run test
    overall_time = None
    all_times = []
    with torch.no_grad():
        states = []
        num_batch = len(test_loader)
        data_iter = iter(test_loader)

        for i in range(num_batch):
            data_timer.tic()
            data_source = next(data_iter)
            data_timer.toc()

            model_timer.tic()
            trans_est, times, num_inliers = model(data_source)
            model_timer.toc()

            trans_est = trans_est if trans_est is not None else np.eye(4)

            if cfg.data.dataset == "3DMatch":
                scene = data_source["src_id"].split("/")[-2]
                src_id = data_source["src_id"].split("/")[-1].split("_")[-1]
                tgt_id = data_source["tgt_id"].split("/")[-1].split("_")[-1]
                logpath = f"logs/log_{cfg.data.benchmark}/{scene}"
                if not os.path.exists(logpath):
                    os.makedirs(logpath)
                # write the transformation matrix into .log file for evaluation.
                with open(os.path.join(logpath, f"{timestr}.log"), "a+") as f:
                    trans = np.linalg.inv(trans_est)
                    s1 = f"{src_id}\t {tgt_id}\t  1\n"
                    f.write(s1)
                    f.write(f"{trans[0, 0]}\t {trans[0, 1]}\t {trans[0, 2]}\t {trans[0, 3]}\t \n")
                    f.write(f"{trans[1, 0]}\t {trans[1, 1]}\t {trans[1, 2]}\t {trans[1, 3]}\t \n")
                    f.write(f"{trans[2, 0]}\t {trans[2, 1]}\t {trans[2, 2]}\t {trans[2, 3]}\t \n")
                    f.write(f"{trans[3, 0]}\t {trans[3, 1]}\t {trans[3, 2]}\t {trans[3, 3]}\t \n")

            ####### Evaluation #######
            rte_thresh, rre_thresh = cfg.test.rte_thresh, cfg.test.rre_thresh
            trans = data_source["relt_pose"].numpy()
            rte = compute_rte(trans_est, trans)
            rre = compute_rre(trans_est, trans)
            success = rte < rte_thresh and rre < rre_thresh

            # Store per-sample results with timing
            states.append(
                [success, rte, rre, num_inliers, data_timer.diff, model_timer.diff, *times]
            )

            if (rte > rte_thresh or rre > rre_thresh) and args.verbose:
                logger.info(f"{i}th fragment failed, RRE: {rre:.4f}, RTE: {rte:.4f}")

            curr_time = np.array([data_timer.diff, model_timer.diff, *times])
            if overall_time is None:
                overall_time = curr_time
            else:
                overall_time += curr_time
            all_times.append(curr_time)
            torch.cuda.empty_cache()

            # logger.info progress every 100 iterations
            if ((i + 1) % 100 == 0 or i == num_batch - 1) and args.verbose:
                temp_states = np.array(states)
                temp_recall = temp_states[:, 0].sum() / temp_states.shape[0]
                temp_te = temp_states[temp_states[:, 0] == 1, 1].mean()
                temp_re = temp_states[temp_states[:, 0] == 1, 2].mean()

                log_prefix = f"[{i + 1}/{num_batch}]"
                log_metrics = f"Recall: {temp_recall:.4f} RTE: {temp_te:.4f} RRE: {temp_re:.4f}"
                log_timing = (
                    f"Data time: {data_timer.diff:.4f}s Model time: {model_timer.diff:.4f}s"
                )

                logger.info(f"{log_prefix} {log_metrics} {log_timing}")

    states = np.array(states)
    recall = states[:, 0].sum() / states.shape[0]
    rte_mean = states[states[:, 0] == 1, 1].mean()
    rre_mean = states[states[:, 0] == 1, 2].mean()
    rte_std = states[states[:, 0] == 1, 1].std()
    rre_std = states[states[:, 0] == 1, 2].std()
    inliers_mean = states[:, 3].mean()
    inliers_std = states[:, 3].std()

    # Save per-sample results to txt file (using parameters for ablation studies)
    per_sample_file = (
        f"{results_dir}/{experiment_id}_{timestr}_{dataset_name}_"
        f"{cfg.patch.num_points_per_patch}_{cfg.patch.num_scales}_{cfg.patch.num_fps}.txt"
    )
    with open(per_sample_file, "w") as f:
        pose_method = cfg.match.pose_estimator.upper()
        header = (
            f"# Pose Estimator: {pose_method}\n"
            "# Sample_ID\tSuccess\tRTE(m)\tRRE(deg)\tNum_Inliers\t"
            "Data_time(s)\tModel_time(s)\tDesc_time(s)\tPose_time(s)\tPoseEst_time(s)\n"
        )
        f.write(header)
        for idx, state in enumerate(states):
            success_flag = int(state[0])
            f.write(
                f"{idx}\t{success_flag}\t{state[1]:.6f}\t{state[2]:.6f}\t{int(state[3])}\t"
                f"{state[4]:.6f}\t{state[5]:.6f}\t{state[6]:.6f}\t{state[7]:.6f}\t{state[8]:.6f}\n"
            )
    logger.info(f"Per-sample results saved to {per_sample_file}")

    if cfg.data.dataset == "3DMatch":
        if cfg.data.benchmark == "3DMatch":
            gtpath = cfg.data.root / "test" / cfg.data.benchmark / "gt_result"
        elif cfg.data.benchmark == "3DLoMatch":
            gtpath = cfg.data.root / "test" / cfg.data.benchmark
        scenes = sorted(os.listdir(gtpath))
        scene_names = [os.path.join(gtpath, ele) for ele in scenes]
        rmse_recall = []

        scene_recall_path = f"scene_recall/{experiment_id}/{timestr}.txt"
        if not os.path.exists(f"scene_recall/{experiment_id}"):
            os.makedirs(f"scene_recall/{experiment_id}")
        with open(scene_recall_path, "w") as f:
            for idx, scene in enumerate(scene_names):
                # ground truth info
                gt_pairs, gt_traj = read_trajectory(os.path.join(scene, "gt.log"))
                n_fragments, gt_traj_cov = read_trajectory_info(os.path.join(scene, "gt.info"))

                # estimated info
                est_path = os.path.join(
                    f"logs/log_{cfg.data.benchmark}", scenes[idx], f"{timestr}.log"
                )
                est_pairs, est_traj = read_trajectory(est_path)
                temp_precision, temp_recall, c_flag, errors = evaluate_registration(
                    n_fragments, est_traj, est_pairs, gt_pairs, gt_traj, gt_traj_cov
                )
                rmse_recall.append(temp_recall)

    # logger.info summary
    logger.info(f"\n---------------Results for {dataset_name}---------------")
    logger.info(f"Recall: {recall:.8f}")
    if cfg.data.dataset == "3DMatch":
        logger.info(f"RMSE Recall (3DMatch Evaluation): {np.array(rmse_recall).mean():.8f}")
        # For 3DMatch evaluation, replace the recall with RMSE-based recall
        recall = np.array(rmse_recall).mean()
    logger.info(f"RTE: {rte_mean * 100:.8f} ± {rte_std * 100:.8f}")
    logger.info(f"RRE: {rre_mean:.8f} ± {rre_std:.8f}")
    logger.info(f"Inliers: {inliers_mean:.2f} ± {inliers_std:.2f}")
    logger.info(f"Pose Estimator: {cfg.match.pose_estimator}")

    all_times = np.array(all_times)
    average_times = overall_time / num_batch

    # Exclude first few iterations (warmup) from std calculation to avoid inflation
    warmup_iters = min(
        5, num_batch // 10
    )  # Skip first 5 iters or 10% of batches, whichever is smaller
    if len(all_times) > warmup_iters:
        std_times = all_times[warmup_iters:].std(axis=0)
    else:
        std_times = all_times.std(axis=0)

    logger.info(f"Average data_time: {average_times[0]:.4f}s ± {std_times[0]:.4f}s")
    logger.info(f"Average model_time: {average_times[1]:.4f}s ± {std_times[1]:.4f}s")

    return (
        recall,
        rte_mean,
        rre_mean,
        rte_std,
        rre_std,
        inliers_mean,
        inliers_std,
        average_times[0],
        average_times[1],
        std_times[0],
        std_times[1],
        cfg.patch.num_points_per_patch,
        cfg.patch.num_scales,
        cfg.patch.num_fps,
    )


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Generalized Testing Script for Registration Models"
    )
    parser.add_argument(
        "--root_dir", type=str, default="../datasets", help="Root directory for all datasets"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        nargs="+",
        choices=[
            "3DMatch",
            "3DLoMatch",
            "Scannetpp_iphone",
            "Scannetpp_faro",
            "TIERS",
            "TIERS_hetero",
            "KITTI",
            "WOD",
            "MIT",
            "KAIST",
            "KAIST_hetero",
            "ETH",
            "Oxford",
            "ModelNet40",
        ],
        help="Dataset to test on",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print detailed progress messages during testing",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="Optional experiment ID (default: uses config.test.experiment_id)",
    )
    parser.add_argument(
        "--num_points_per_patch",
        type=int,
        default=None,
        help="Number of points per patch (default: uses config value)",
    )
    parser.add_argument(
        "--num_scales",
        type=int,
        default=None,
        help="Number of scales for multi-scale patch embedder (default: uses config value)",
    )
    parser.add_argument(
        "--num_fps",
        type=int,
        default=None,
        help="Number of FPS (Farthest Point Sampling) points (default: uses config value)",
    )
    parser.add_argument(
        "--search_radius_thresholds",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Search radius thresholds in decreasing order "
            "(e.g., --search_radius_thresholds 5 2 0.5)"
        ),
    )
    parser.add_argument(
        "--pose_estimator",
        type=str,
        default=None,
        choices=["ransac", "kiss_matcher"],
        help='Pose estimation method: "ransac" or "kiss_matcher" (default: uses config value)',
    )
    args = parser.parse_args()

    timestr = time.strftime("%m%d%H%M")
    # NOTE(hlim): We employ the model trained 3DMatch as a default mode.
    experiment_id = args.experiment_id if args.experiment_id else "threedmatch"
    results = []
    num_points_per_patch = None
    num_scales = None
    num_fps = None

    for dataset_name in args.dataset:
        (
            recall,
            rte_mean,
            rre_mean,
            rte_std,
            rre_std,
            inliers_mean,
            inliers_std,
            avg_data_time,
            avg_model_time,
            std_data_time,
            std_model_time,
            npp,
            ns,
            nfps,
        ) = run(args, timestr, experiment_id, dataset_name)

        # Store config values from first dataset for filename
        if num_points_per_patch is None:
            num_points_per_patch = npp
            num_scales = ns
            num_fps = nfps

        results.append(
            [
                dataset_name,
                f"{recall:.4f}",
                f"{rte_mean * 100:.4f}",
                f"{rte_std * 100:.4f}",
                f"{rre_mean:.4f}",
                f"{rre_std:.4f}",
                f"{inliers_mean:.2f}",
                f"{inliers_std:.2f}",
                f"{avg_data_time:.4f}s",
                f"{std_data_time:.4f}s",
                f"{avg_model_time:.4f}s",
                f"{std_model_time:.4f}s",
            ]
        )

    print("\n\033[1;32m========== Final Results Summary ==========")
    headers = [
        "Scene",
        "Recall",
        "RTE (cm)",
        "RTE std (cm)",
        "RRE (deg)",
        "RRE std (deg)",
        "Inliers",
        "Inliers std",
        "Avg data t",
        "Std data t",
        "Avg model t",
        "Std model t",
    ]
    print(tabulate(results, headers=headers, tablefmt="grid"), "\033[0m")

    # Save results to .mat file for Matlab
    matlab_results = {
        "datasets": [r[0] for r in results],
        "recall": np.array([float(r[1]) for r in results]),
        "rte_mean_cm": np.array([float(r[2]) for r in results]),
        "rte_std_cm": np.array([float(r[3]) for r in results]),
        "rre_mean_deg": np.array([float(r[4]) for r in results]),
        "rre_std_deg": np.array([float(r[5]) for r in results]),
        "inliers_mean": np.array([float(r[6]) for r in results]),
        "inliers_std": np.array([float(r[7]) for r in results]),
        "avg_data_time_s": np.array([float(r[8].replace("s", "")) for r in results]),
        "std_data_time_s": np.array([float(r[9].replace("s", "")) for r in results]),
        "avg_model_time_s": np.array([float(r[10].replace("s", "")) for r in results]),
        "std_model_time_s": np.array([float(r[11].replace("s", "")) for r in results]),
        "experiment_id": experiment_id,
        "timestamp": timestr,
    }

    mat_file_path = (
        f"results_{experiment_id}_{timestr}_{num_points_per_patch}_{num_scales}_{num_fps}.mat"
    )
    savemat(mat_file_path, matlab_results)
    print(f"\n\033[1;34mResults saved to {mat_file_path} for Matlab\033[0m")
