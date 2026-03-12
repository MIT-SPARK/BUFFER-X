import os
import shutil
import sys

import time
import torch
import torch.nn as nn
import numpy as np
from utils.timer import Timer
from utils.gpu_timer import GPUTimer
from utils.test_args import build_eval_targets, parse_test_args
from utils.result_io import (
    print_final_results_summary,
    save_full_results_csv,
    save_per_sample_results,
)
from utils.SE3 import compute_rte, compute_rre
from utils.progress_format import resolve_sample_display_fields
from utils.tools import evaluate_registration, read_trajectory, read_trajectory_info, setup_logger
from config import make_cfg
from dataset.dataloader import get_dataloader
from models.BUFFERX import BufferX

FIRST_A_FEW_FRAMES = 5


def run(
    args,
    timestr,
    experiment_id,
    dataset_name,
    src_sensor_override=None,
    tgt_sensor_override=None,
):
    # Set default CUDA device
    torch.cuda.set_device(args.gpu)

    exp_name = experiment_id.rsplit("/", 1)[-1]

    log_file = f"logs/test/{exp_name}/{dataset_name}_{timestr}.log"
    os.makedirs(f"logs/test/{exp_name}", exist_ok=True)
    logger = setup_logger(log_file)
    logger.info(f"Start testing on {dataset_name}...")

    # Load dataset-specific config
    cfg = make_cfg(dataset_name, args.root_dir)
    cfg[cfg.data.dataset] = cfg.copy()
    cfg.stage = "test"

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
        logger.info(f"\033[1;32mOverwriting pose_estimator: {args.pose_estimator}\033[0m")
    if dataset_name.endswith("_hetero"):
        src_sensor = src_sensor_override if src_sensor_override is not None else args.src_sensor
        tgt_sensor = tgt_sensor_override if tgt_sensor_override is not None else args.tgt_sensor
        if src_sensor is not None:
            cfg.data.src_sensor = src_sensor
            logger.info(f"Overwriting src_sensor: {src_sensor}")
        if tgt_sensor is not None:
            cfg.data.tgt_sensor = tgt_sensor
            logger.info(f"Overwriting tgt_sensor: {tgt_sensor}")
        logger.info(f"Heterogeneous evaluation: {cfg.data.src_sensor} -> {cfg.data.tgt_sensor}")
    scene_label = dataset_name
    if dataset_name.endswith("_hetero"):
        scene_label = f"{dataset_name}:{cfg.data.src_sensor}->{cfg.data.tgt_sensor}"

    # Initialize model
    # TODO(hlim): If `cfg` specifies a different model, the model can be changed.
    # We might need an option to fix the model across all scenes.
    model = BufferX(cfg)

    # Load model weights
    device = f"cuda:{args.gpu}"
    for stage in cfg.train.all_stage:
        model_path = f"snapshot/{experiment_id}/{stage}/best.pth"
        state_dict = torch.load(model_path, map_location=device)
        new_dict = {k: v for k, v in state_dict.items() if stage in k}
        model_dict = model.state_dict()
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded {stage} model from {model_path}")

    logger.info(f"Using GPU: {args.gpu}")

    # Move model to the specified GPU
    model = model.to(device)

    # Model Parameter Info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of trainable parameters: {total_params / 1e6:.2f}M")

    model = nn.DataParallel(model, device_ids=[args.gpu])
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
    data_timer, model_timer = Timer(), GPUTimer()  # CPU timer for data, GPU timer for model

    # Create directory for per-sample results
    results_dir = f"per_sample_results/{exp_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Run test
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
            (
                trans_est,
                times,
                num_inliers,
                num_mutual_inliers,
                num_inlier_ind,
                scales_used,
            ) = model(data_source)
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
                [
                    success,
                    rte,
                    rre,
                    num_inliers,
                    num_mutual_inliers,
                    num_inlier_ind,
                    scales_used,
                    data_timer.diff,
                    model_timer.diff / 1000.0,  # Convert ms to seconds
                    *times,
                ]
            )

            curr_time = np.array([data_timer.diff, model_timer.diff / 1000.0, *times])
            all_times.append(curr_time)
            torch.cuda.empty_cache()

            # tqdm-like single-line verbose progress (updated every iteration)
            if args.verbose:
                temp_states = np.array(states)
                temp_recall = temp_states[:, 0].sum() / temp_states.shape[0]
                success_mask = temp_states[:, 0] == 1
                if success_mask.any():
                    temp_te = temp_states[success_mask, 1].mean()
                    temp_re = temp_states[success_mask, 2].mean()
                else:
                    temp_te = float("nan")
                    temp_re = float("nan")

                sample_fields = resolve_sample_display_fields(
                    data_source, fallback_dataset_name=dataset_name
                )
                scene_name = sample_fields["scene_name"]
                sensor_name = sample_fields["sensor_name"]
                src_disp = sample_fields["src_disp"]
                tgt_disp = sample_fields["tgt_disp"]
                fail_src = sample_fields["fail_src"]
                fail_tgt = sample_fields["fail_tgt"]
                sensor_prefix = f"Sensor: {sensor_name} " if sensor_name else ""
                sensor_log = f" | Sensor: {sensor_name}" if sensor_name else ""

                if rte > rte_thresh or rre > rre_thresh:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                    logger.info(
                        f"[FAIL {i + 1}/{num_batch}] Scene: {scene_name}{sensor_log} | "
                        f"Src: {fail_src} | Tgt: {fail_tgt} | "
                        f"RRE: {rre:.4f} (th: {rre_thresh:.4f}), "
                        f"RTE: {rte:.4f} (th: {rte_thresh:.4f})"
                    )

                progress_line = (
                    f"[{i + 1}/{num_batch}] "
                    f"Scene: {scene_name} "
                    f"{sensor_prefix}"
                    f"Src: {src_disp} "
                    f"Tgt: {tgt_disp} "
                    f"Recall: {temp_recall:.4f} "
                    f"RTE: {temp_te:.4f} "
                    f"RRE: {temp_re:.4f} "
                    f"Data t: {data_timer.diff:.4f}s "
                    f"Model t: {model_timer.diff / 1000.0:.4f}s"
                )

                # Keep progress on one terminal row: truncate to width and clear remainder.
                term_width = shutil.get_terminal_size(fallback=(120, 20)).columns
                max_len = max(1, term_width - 1)
                if len(progress_line) > max_len:
                    progress_line = (
                        f"{progress_line[: max_len - 3]}..." if max_len > 3 else progress_line[:max_len]
                    )
                sys.stdout.write("\r\033[K" + progress_line)
                sys.stdout.flush()

    if args.verbose:
        # Move to next line after in-place progress output.
        print()

    states = np.array(states)
    recall = states[:, 0].sum() / states.shape[0]
    rte_mean = states[states[:, 0] == 1, 1].mean()
    rre_mean = states[states[:, 0] == 1, 2].mean()
    rte_std = states[states[:, 0] == 1, 1].std()
    rre_std = states[states[:, 0] == 1, 2].std()
    inliers_mean = states[:, 3].mean()
    inliers_std = states[:, 3].std()
    mutual_inliers_mean = states[:, 4].mean()
    mutual_inliers_std = states[:, 4].std()
    inlier_ind_mean = states[:, 5].mean()
    inlier_ind_std = states[:, 5].std()
    scales_used_mean = states[:, 6].mean()
    scales_used_std = states[:, 6].std()

    # Save per-sample results to csv file (using parameters for ablation studies)
    per_sample_file = (
        f"{results_dir}/{exp_name}_{dataset_name}_"
        f"{cfg.patch.num_points_per_patch}_{cfg.patch.num_scales}_{cfg.patch.num_fps}_{timestr}.csv"
    )
    pose_method = cfg.match.pose_estimator.upper()
    early_exit_status = "ON" if cfg.match.get("enable_early_exit", True) else "OFF"
    save_per_sample_results(states, per_sample_file, pose_method, early_exit_status)
    logger.info(f"Per-sample results saved to {per_sample_file}")

    if cfg.data.dataset == "3DMatch":
        if cfg.data.benchmark == "3DMatch":
            gtpath = cfg.data.root / "test" / cfg.data.benchmark / "gt_result"
        elif cfg.data.benchmark == "3DLoMatch":
            gtpath = cfg.data.root / "test" / cfg.data.benchmark
        scenes = sorted(os.listdir(gtpath))
        scene_names = [os.path.join(gtpath, ele) for ele in scenes]
        rmse_recall = []

        scene_recall_path = f"scene_recall/{exp_name}/{timestr}.txt"
        if not os.path.exists(f"scene_recall/{exp_name}"):
            os.makedirs(f"scene_recall/{exp_name}")
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
    logger.info(f"\n---------------Results for {scene_label}---------------")
    logger.info(f"Recall: {recall:.8f}")
    if cfg.data.dataset == "3DMatch":
        logger.info(f"RMSE Recall (3DMatch Evaluation): {np.array(rmse_recall).mean():.8f}")
        # For 3DMatch evaluation, replace the recall with RMSE-based recall
        recall = np.array(rmse_recall).mean()
    logger.info(f"RTE: {rte_mean * 100:.8f} ± {rte_std * 100:.8f}")
    logger.info(f"RRE: {rre_mean:.8f} ± {rre_std:.8f}")
    logger.info(f"Inliers: {inliers_mean:.2f} ± {inliers_std:.2f}")
    logger.info(f"Mutual Inliers: {mutual_inliers_mean:.2f} ± {mutual_inliers_std:.2f}")
    logger.info(f"Inlier Ind: {inlier_ind_mean:.2f} ± {inlier_ind_std:.2f}")
    logger.info(f"Scales Used: {scales_used_mean:.2f} ± {scales_used_std:.2f}")
    logger.info(f"Pose Estimator: {cfg.match.pose_estimator}")
    early_exit_status = "ON" if cfg.match.get("enable_early_exit", True) else "OFF"
    logger.info(f"Early Exit: {early_exit_status}")

    all_times = np.array(all_times)
    # Exclude first FIRST_A_FEW_FRAMES iterations (warmup) from both mean and std.
    if len(all_times) > FIRST_A_FEW_FRAMES:
        effective_times = all_times[FIRST_A_FEW_FRAMES:]
    else:
        effective_times = all_times
    average_times = effective_times.mean(axis=0)
    std_times = effective_times.std(axis=0)

    logger.info(f"Average data_time: {average_times[0]:.4f}s ± {std_times[0]:.4f}s")
    logger.info(f"Average model_time: {average_times[1]:.4f}s ± {std_times[1]:.4f}s")
    logger.info(f"Average desc_time: {average_times[2]:.4f}s ± {std_times[2]:.4f}s")
    logger.info(f"Average pose_time: {average_times[3]:.4f}s ± {std_times[3]:.4f}s")
    logger.info(f"Average pose_optim_time: {average_times[4]:.4f}s ± {std_times[4]:.4f}s")

    return (
        scene_label,
        recall,
        rte_mean,
        rre_mean,
        rte_std,
        rre_std,
        inliers_mean,
        inliers_std,
        mutual_inliers_mean,
        mutual_inliers_std,
        inlier_ind_mean,
        inlier_ind_std,
        scales_used_mean,
        scales_used_std,
        average_times[0],
        average_times[1],
        std_times[0],
        std_times[1],
        cfg.patch.num_points_per_patch,
        cfg.patch.num_scales,
        cfg.patch.num_fps,
    )


if __name__ == "__main__":
    args = parse_test_args()

    timestr = time.strftime("%m%d%H%M")
    # NOTE(hlim): We employ the model trained 3DMatch as a default mode.
    experiment_id = args.experiment_id if args.experiment_id else "threedmatch"
    results = []
    num_points_per_patch = None
    num_scales = None
    num_fps = None

    eval_targets = build_eval_targets(args)
    for dataset_name, src_sensor_override, tgt_sensor_override in eval_targets:
        (
            scene_label,
            recall,
            rte_mean,
            rre_mean,
            rte_std,
            rre_std,
            inliers_mean,
            inliers_std,
            mutual_inliers_mean,
            mutual_inliers_std,
            inlier_ind_mean,
            inlier_ind_std,
            scales_used_mean,
            scales_used_std,
            avg_data_time,
            avg_model_time,
            std_data_time,
            std_model_time,
            npp,
            ns,
            nfps,
        ) = run(
            args,
            timestr,
            experiment_id,
            dataset_name,
            src_sensor_override,
            tgt_sensor_override,
        )

        # Store config values from first dataset for filename
        if num_points_per_patch is None:
            num_points_per_patch = npp
            num_scales = ns
            num_fps = nfps

        results.append(
            {
                "dataset": scene_label,
                "recall": recall,
                "rte_mean_cm": rte_mean * 100,
                "rte_std_cm": rte_std * 100,
                "rre_mean_deg": rre_mean,
                "rre_std_deg": rre_std,
                "inliers_mean": inliers_mean,
                "inliers_std": inliers_std,
                "mutual_inliers_mean": mutual_inliers_mean,
                "mutual_inliers_std": mutual_inliers_std,
                "inlier_ind_mean": inlier_ind_mean,
                "inlier_ind_std": inlier_ind_std,
                "scales_used_mean": scales_used_mean,
                "scales_used_std": scales_used_std,
                "avg_data_time_s": avg_data_time,
                "std_data_time_s": std_data_time,
                "avg_model_time_s": avg_model_time,
                "std_model_time_s": std_model_time,
            }
        )

    print_final_results_summary(results)

    csv_file_path = save_full_results_csv(
        results,
        experiment_id,
        timestr,
        num_points_per_patch,
        num_scales,
        num_fps,
    )
    print(f"\n\033[1;34mResults saved to {csv_file_path}\033[0m")
