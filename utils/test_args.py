import argparse


TEST_DATASET_CHOICES = [
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
]


def build_test_arg_parser():
    parser = argparse.ArgumentParser(description="Generalized Testing Script for Registration Models")
    parser.add_argument(
        "--root_dir", type=str, default="../datasets", help="Root directory for all datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        nargs="+",
        choices=TEST_DATASET_CHOICES,
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
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        choices=[0, 1],
        help="GPU device to use: 0 or 1 (default: 0)",
    )
    parser.add_argument(
        "--src_sensor",
        type=str,
        default=None,
        help="Source sensor for heterogeneous datasets (e.g., Aeva, os0_128)",
    )
    parser.add_argument(
        "--tgt_sensor",
        type=str,
        default=None,
        help="Target sensor for heterogeneous datasets (e.g., Avia, os1_64)",
    )
    parser.add_argument(
        "--hetero_pairs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional heterogeneous eval specs in the form "
            "'<dataset>:<src_sensor>-><tgt_sensor>' "
            "(e.g., 'KAIST_hetero:Aeva->Avia')."
        ),
    )
    return parser


def parse_test_args():
    return build_test_arg_parser().parse_args()


def parse_hetero_pair_specs(hetero_pairs):
    eval_targets = []
    for spec in hetero_pairs:
        if ":" not in spec or "->" not in spec:
            raise ValueError(
                f"Invalid --hetero_pairs spec '{spec}'. "
                "Expected format: <dataset>:<src_sensor>-><tgt_sensor>"
            )
        dataset_name, sensor_spec = spec.split(":", 1)
        src_sensor, tgt_sensor = sensor_spec.split("->", 1)
        dataset_name = dataset_name.strip()
        src_sensor = src_sensor.strip()
        tgt_sensor = tgt_sensor.strip()
        if not dataset_name.endswith("_hetero"):
            raise ValueError(
                f"Invalid hetero dataset '{dataset_name}' in spec '{spec}'. "
                "Dataset must end with '_hetero'."
            )
        if not src_sensor or not tgt_sensor:
            raise ValueError(
                f"Invalid sensor direction in spec '{spec}'. "
                "Both source and target sensors are required."
            )
        eval_targets.append((dataset_name, src_sensor, tgt_sensor))
    return eval_targets


def build_eval_targets(args):
    # If explicit hetero directions are provided, evaluate those exactly as listed.
    if args.hetero_pairs:
        eval_targets = [(d, None, None) for d in args.dataset if not d.endswith("_hetero")]
        eval_targets.extend(parse_hetero_pair_specs(args.hetero_pairs))
        return eval_targets

    # Default mode: one target per dataset with optional global hetero override.
    eval_targets = []
    for dataset_name in args.dataset:
        if dataset_name.endswith("_hetero"):
            eval_targets.append((dataset_name, args.src_sensor, args.tgt_sensor))
        else:
            eval_targets.append((dataset_name, None, None))
    return eval_targets
