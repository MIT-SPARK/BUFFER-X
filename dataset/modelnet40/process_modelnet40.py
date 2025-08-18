import os
import glob
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from tqdm import tqdm
import copy


class RandomTransformSE3_euler:
    """Generate a random SE(3) transformation using Euler angles."""

    def __init__(self, rot_mag=45.0, trans_mag=0.5):
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag

    def generate_transform(self):
        anglex = np.random.uniform() * np.pi * self._rot_mag / 180.0
        angley = np.random.uniform() * np.pi * self._rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * self._rot_mag / 180.0

        cosx, cosy, cosz = np.cos([anglex, angley, anglez])
        sinx, siny, sinz = np.sin([anglex, angley, anglez])

        Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

        R = Rx @ Ry @ Rz
        t = np.random.uniform(-self._trans_mag, self._trans_mag, 3)
        return R, t


def random_crop_halfspace(pcd, keep_ratio=0.7):
    """Randomly crop a point cloud using a random half-space."""
    pts = np.asarray(pcd.points)
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)
    centered = pts - np.mean(pts, axis=0)
    dist = np.dot(centered, direction)
    threshold = np.percentile(dist, (1.0 - keep_ratio) * 100)
    mask = dist > threshold
    cropped_pts = pts[mask]
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(cropped_pts)
    return cropped_pcd


def jitter(pcd, std=0.01, clip=0.05):
    """Apply Gaussian noise to point positions."""
    pts = np.array(pcd.points)
    noise = np.clip(np.random.normal(0.0, std, size=pts.shape), -clip, clip)
    pts += noise
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate partial point clouds from ModelNet and gt.log in 3DMatch format"
    )
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path("../../../datasets/modelnet40_ply"),
        help="Root directory containing ModelNet .ply files",
    )
    parser.add_argument(
        "--dst_root",
        type=Path,
        default=Path("../../../datasets/processed_modelnet40"),
        help="Output directory for processed point clouds and gt.log",
    )
    args = parser.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)
    pose_gen = RandomTransformSE3_euler()

    gt_log_path = args.dst_root / "gt.log"
    curr_path = os.path.dirname(os.path.realpath(__file__))
    split_path = curr_path + "/../../config/splits/modelnet40_half2.txt"

    with open(split_path, "r") as f:
        categories = f.read().splitlines()

    with open(gt_log_path, "w") as log_f:
        for cat in categories:
            cat_in = args.src_root / cat
            if not os.path.isdir(cat_in):
                continue

            cat_out = args.dst_root / cat
            os.makedirs(cat_out, exist_ok=True)

            for ply_file in tqdm(glob.glob(str(cat_in / "*.ply"))):
                name = os.path.splitext(os.path.basename(ply_file))[0]

                # Load original full point cloud
                pcd = o3d.io.read_point_cloud(ply_file)
                o3d.io.write_point_cloud(str(cat_out / f"{name}.ply"), pcd)

                # Generate random SE(3) transformation
                R, t = pose_gen.generate_transform()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t

                # Apply inverse transformation to create target
                tgt_pcd = copy.deepcopy(pcd)
                tgt_pcd.transform(T)

                # Apply random cropping
                src_part = random_crop_halfspace(pcd)
                tgt_part = random_crop_halfspace(tgt_pcd)

                # Save source and target point clouds
                o3d.io.write_point_cloud(str(cat_out / f"{name}_src.ply"), src_part)
                o3d.io.write_point_cloud(str(cat_out / f"{name}_tgt.ply"), tgt_part)

                # Flatten transformation and write log entry
                log_f.write(f"{name}_src\t{name}_tgt\t{cat}\n")
                for row in T:
                    log_f.write(f" {row[0]: .8e}\t{row[1]: .8e}\t{row[2]: .8e}\t{row[3]: .8e}\n")
