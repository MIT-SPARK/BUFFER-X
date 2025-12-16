"""
Pose Estimation Module for BUFFER-X++
Supports RANSAC and KISS-Matcher with unified interface
"""

import numpy as np
import open3d as o3d
from utils.common import make_open3d_point_cloud


class PoseEstimator:
    """Unified pose estimator supporting RANSAC and KISS-Matcher"""

    def __init__(self, cfg):
        """
        Args:
            cfg: Configuration object with match parameters
        """
        self.cfg = cfg
        self.pose_estimator = cfg.match.pose_estimator

    def estimate_pose(self, src_kpts, tgt_kpts, inlier_ind):
        """
        Estimate pose from keypoint correspondences

        Args:
            src_kpts: Source keypoints [N, 3] (torch tensor or numpy)
            tgt_kpts: Target keypoints [N, 3] (torch tensor or numpy)
            inlier_ind: Inlier indices [M] (numpy array)

        Returns:
            init_pose: 4x4 transformation matrix (numpy)
            num_inliers: Number of final inliers (int)
        """
        # Convert to numpy if needed
        if hasattr(src_kpts, "detach"):
            src_kpts_np = src_kpts.detach().cpu().numpy()
            tgt_kpts_np = tgt_kpts.detach().cpu().numpy()
        else:
            src_kpts_np = src_kpts
            tgt_kpts_np = tgt_kpts

        if self.pose_estimator == "kiss_matcher":
            return self._estimate_kiss_matcher(src_kpts_np, tgt_kpts_np, inlier_ind)
        elif self.pose_estimator == "ransac":
            return self._estimate_ransac(src_kpts_np, tgt_kpts_np, inlier_ind)
        else:
            raise ValueError(f"Unknown pose estimator: {self.pose_estimator}")

    def _estimate_kiss_matcher(self, src_kpts, tgt_kpts, inlier_ind):
        """Estimate pose using KISS-Matcher"""
        try:
            from kiss_matcher import KISSMatcherConfig, KISSMatcher

            # Prepare point clouds (use inlier correspondences)
            src_pts = src_kpts[inlier_ind]
            tgt_pts = tgt_kpts[inlier_ind]

            # Configure and run KISS-Matcher
            kiss_config = KISSMatcherConfig(self.cfg.match.kiss_resolution)
            matcher = KISSMatcher(kiss_config)
            result = matcher.solve(src_pts.transpose(), tgt_pts.transpose())

            # Build transformation matrix
            init_pose = np.eye(4)
            init_pose[:3, :3] = result.rotation
            init_pose[:3, 3] = result.translation

            # Get inlier count
            num_inliers = matcher.get_num_final_inliers()

            return init_pose, num_inliers

        except ImportError:
            print(
                "Warning: KISS-Matcher not installed. "
                "Falling back to RANSAC. "
                "Install with: pip install kiss-matcher"
            )
            # Fall back to RANSAC
            self.pose_estimator = "ransac"
            return self._estimate_ransac(src_kpts, tgt_kpts, inlier_ind)

    def _estimate_ransac(self, src_kpts, tgt_kpts, inlier_ind):
        """Estimate pose using RANSAC"""
        # Create Open3D point clouds
        pcd0 = make_open3d_point_cloud(src_kpts, [1, 0.706, 0])
        pcd1 = make_open3d_point_cloud(tgt_kpts, [0, 0.651, 0.929])

        # Create correspondences
        corr = o3d.utility.Vector2iVector(np.array([inlier_ind, inlier_ind]).T)

        # Run RANSAC
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcd0,
            pcd1,
            corr,
            self.cfg.match.dist_th,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    self.cfg.match.similar_th
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    self.cfg.match.dist_th
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.cfg.match.iter_n, self.cfg.match.confidence
            ),
        )

        init_pose = result.transformation
        num_inliers = len(result.correspondence_set)

        return init_pose, num_inliers

    def compute_confidence_score(self, num_inliers):
        # Absolute inlier count threshold
        min_inliers = self.cfg.match.get("early_exit_min_inliers", 15)

        # Decision logic
        should_exit = num_inliers >= min_inliers

        return should_exit
