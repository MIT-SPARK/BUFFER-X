import csv
import os

from tabulate import tabulate


def save_per_sample_results(states, per_sample_file, pose_method, early_exit_status):
    os.makedirs(os.path.dirname(per_sample_file), exist_ok=True)
    with open(per_sample_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "success",
                "rte_m",
                "rre_deg",
                "num_inliers",
                "num_mutual_inliers",
                "num_inlier_ind",
                "scales_used",
                "data_time_s",
                "model_time_s",
                "desc_time_s",
                "pose_time_s",
                "poseest_time_s",
                "pose_estimator",
                "early_exit",
            ]
        )
        for idx, state in enumerate(states):
            writer.writerow(
                [
                    idx,
                    int(state[0]),
                    f"{state[1]:.6f}",
                    f"{state[2]:.6f}",
                    int(state[3]),
                    int(state[4]),
                    int(state[5]),
                    int(state[6]),
                    f"{state[7]:.6f}",
                    f"{state[8]:.6f}",
                    f"{state[9]:.6f}",
                    f"{state[10]:.6f}",
                    f"{state[11]:.6f}",
                    pose_method,
                    early_exit_status,
                ]
            )


def print_final_results_summary(results):
    print("\n\033[1;32m========== Final Results Summary ==========")
    headers = [
        "Scene",
        "Recall",
        "RTE mean (cm)",
        "RTE std (cm)",
        "RRE mean (deg)",
        "RRE std (deg)",
        "Avg data t (s)",
        "Avg model t (s)",
    ]
    rows = [
        [
            r["dataset"],
            f"{r['recall']:.4f}",
            f"{r['rte_mean_cm']:.4f}",
            f"{r['rte_std_cm']:.4f}",
            f"{r['rre_mean_deg']:.4f}",
            f"{r['rre_std_deg']:.4f}",
            f"{r['avg_data_time_s']:.4f}",
            f"{r['avg_model_time_s']:.4f}",
        ]
        for r in results
    ]
    print(tabulate(rows, headers=headers, tablefmt="grid"), "\033[0m")


def save_full_results_csv(
    results,
    experiment_id,
    timestr,
    num_points_per_patch,
    num_scales,
    num_fps,
):
    full_results_dir = f"full_results"
    os.makedirs(full_results_dir, exist_ok=True)
    exp_name = experiment_id.rsplit("/", 1)[-1]
    csv_file_path = (
        f"{full_results_dir}/results_{exp_name}_{num_points_per_patch}_{num_scales}_{num_fps}_{timestr}.csv"
    )
    csv_headers = [
        "dataset",
        "recall",
        "rte_mean_cm",
        "rte_std_cm",
        "rre_mean_deg",
        "rre_std_deg",
        "inliers_mean",
        "inliers_std",
        "mutual_inliers_mean",
        "mutual_inliers_std",
        "inlier_ind_mean",
        "inlier_ind_std",
        "scales_used_mean",
        "scales_used_std",
        "avg_data_time_s",
        "std_data_time_s",
        "avg_model_time_s",
        "std_model_time_s",
        "experiment_id",
        "timestamp",
    ]
    with open(csv_file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for row in results:
            row_with_meta = row.copy()
            row_with_meta["experiment_id"] = experiment_id
            row_with_meta["timestamp"] = timestr
            writer.writerow(row_with_meta)
    return csv_file_path
