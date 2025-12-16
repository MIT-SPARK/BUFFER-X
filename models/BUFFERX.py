import torch
import torch.nn as nn
import torch.nn.functional as F
import models.patchnet as pn
from models.patch_embedder import MiniSpinNet
from models.pose_estimator import PoseEstimator
import pointnet2_ops.pointnet2_utils as pnt2
from knn_cuda import KNN
from utils.SE3 import transform, integrate_trans
from einops import rearrange
import kornia.geometry.conversions as Convert
from utils.gpu_timer import GPUTimer
import numpy as np


class EquiMatch(nn.Module):
    def __init__(self, config):
        super(EquiMatch, self).__init__()
        self.azi_n = config.patch.azi_n
        init_index = np.arange(self.azi_n)
        index_list = []
        for i in range(self.azi_n):
            cur_index = np.concatenate([init_index[self.azi_n - i :], init_index[: self.azi_n - i]])
            index_list.append(cur_index)
        self.index_list = np.array(index_list)

    def forward(self, Des1, Des2):
        [B, C, K, L] = Des1.shape
        index_list = torch.from_numpy(self.index_list).to(Des1.device)
        Des1 = Des1[:, :, :, torch.reshape(index_list, [-1])].reshape(
            [Des1.shape[0], Des1.shape[1], Des1.shape[2], self.azi_n, self.azi_n]
        )
        Des1 = rearrange(Des1, "b c k n l -> b c n k l").reshape([B, C, -1, K * L])
        Des2 = Des2.reshape([B, C, K * L])
        cor = torch.einsum("bfag,bfg->ba", Des1, Des2)
        return cor


class CostVolume(nn.Module):
    def __init__(self, config):
        super(CostVolume, self).__init__()
        self.azi_n = config.patch.azi_n
        init_index = np.arange(self.azi_n)
        index_list = []
        for i in range(self.azi_n):
            cur_index = np.concatenate([init_index[self.azi_n - i :], init_index[: self.azi_n - i]])
            index_list.append(cur_index)
        self.index_list = np.array(index_list)
        self.conv = pn.CostNet(inchan=32, dim=20)

    def forward(self, Des1, Des2):
        """
        Input
            - Des1: [B, C, K, L]
            - Des2: [B, C, K, L]
        Output:
            -
        """
        index_list = torch.from_numpy(self.index_list).to(Des1.device)
        Des1 = Des1[:, :, :, torch.reshape(index_list, [-1])].reshape(
            [Des1.shape[0], Des1.shape[1], Des1.shape[2], self.azi_n, self.azi_n]
        )
        Des1 = rearrange(Des1, "b c k n l -> b c n k l")
        Des2 = Des2.unsqueeze(2)
        cost = Des1 - Des2
        cost = self.conv(cost).squeeze()
        prob = F.softmax(cost, dim=-1)
        ind = torch.sum(prob * torch.arange(0, self.azi_n)[None].to(prob.device), dim=-1)
        return ind


class BufferX(nn.Module):
    def __init__(self, config):
        super(BufferX, self).__init__()
        self.config = config
        self.config.stage = config.stage

        self.Desc = MiniSpinNet(config)
        self.Pose = CostVolume(config)
        # equivariant feature matching
        self.equi_match = EquiMatch(config)
        # pose estimator (RANSAC or KISS-Matcher)
        if config.stage == "test":
            self.pose_estimator = PoseEstimator(config)

    def cal_so2_gt(self, src, tgt, gt_trans, integer=True, aug_rotation=None):
        _, _, s_rand_axis, s_R, _ = (
            src["desc"],
            src["equi"],
            src["rand_axis"],
            src["R"],
            src["patches"],
        )
        _, _, _, t_R, _ = (
            tgt["desc"],
            tgt["equi"],
            tgt["rand_axis"],
            tgt["R"],
            tgt["patches"],
        )
        # calculate gt lable in SO(2)
        t_rand_axis = torch.matmul(s_rand_axis[:, None], gt_trans[:3, :3].transpose(-1, -2))
        s_rand_axis = torch.matmul(s_rand_axis[:, None], s_R)
        t_rand_axis = torch.matmul(t_rand_axis, t_R)
        if aug_rotation is not None:
            t_rand_axis = t_rand_axis @ aug_rotation.transpose(-1, -2)
        z_axis = torch.zeros_like(t_rand_axis)
        z_axis[:, :, -1] = 1
        proj_t = F.normalize(
            t_rand_axis - torch.sum(t_rand_axis * z_axis, dim=-1, keepdim=True) * z_axis,
            p=2,
            dim=-1,
        )
        s_rand_axis = s_rand_axis.squeeze()
        proj_t = proj_t.squeeze()
        z_axis = z_axis.squeeze()
        dev_angle = torch.acos(F.cosine_similarity(s_rand_axis, proj_t).clamp(min=-1, max=1))
        sign = torch.sum(torch.cross(s_rand_axis, proj_t) * z_axis, dim=-1) < 0
        dev_angle[sign] = 2 * np.pi - dev_angle[sign]
        if integer:
            lable = torch.round(dev_angle * self.config.patch.azi_n / (2 * np.pi))
            lable[lable == self.config.patch.azi_n] = 0
            lable = lable.type(torch.int64).detach()
        else:
            lable = dev_angle * self.config.patch.azi_n / (2 * np.pi)
            lable[lable == self.config.patch.azi_n] = 0
            lable = lable.detach()
        return lable

    def forward(self, data_source):
        """
        src_fds_pcd / tgt_fds_pcd:
        - First-level downsampled point clouds via voxelization.
        - Farthest Point Sampling (FPS) is applied on these points to obtain keypoints.
        - Patch descriptors are then computed by sampling neighborhoods from these fds points.
        - During training: downsampled using config-specified voxel size.
        - During testing: voxel size is automatically estimated for each sample.

        src_sds_pcd / tgt_sds_pcd:
        - Second-level downsampled point clouds via voxelization.
        - Used only during training as sampled keypoints for patch-based supervision.
        (e.g., loss, correspondence).
        - Always downsampled using config-specified voxel size.
        """

        src_fds_pcd, tgt_fds_pcd = data_source["src_fds_pcd"], data_source["tgt_fds_pcd"]

        if self.config.stage != "test":
            src_sds_pts, tgt_sds_pts = data_source["src_sds_pcd"], data_source["tgt_sds_pcd"]
            # find positive correspondences
            gt_trans = data_source["relt_pose"]
            match_inds = self.get_matching_indices(
                src_sds_pts, tgt_sds_pts, gt_trans, data_source["voxel_sizes"][0]
            )

            dataset_name = self.config["data"]["dataset"]
            cfg = self.config

            # randomly sample some positive pairs to speed up the training
            if match_inds.shape[0] > self.config.train.pos_num:
                rand_ind = np.random.choice(
                    range(match_inds.shape[0]), self.config.train.pos_num, replace=False
                )
                match_inds = match_inds[rand_ind]
            src_kpt = src_sds_pts[match_inds[:, 0]]
            tgt_kpt = tgt_sds_pts[match_inds[:, 1]]

            if src_kpt.shape[0] == 0 or tgt_kpt.shape[0] == 0:
                print(f"{data_source['src_id']} {data_source['tgt_id']} has no keypts")
                return None
            #######################
            # training descriptor
            #######################
            # calculate feature descriptor
            if dataset_name == "3DMatch":
                center = cfg.patch.des_r
                lower_bound = center * 0.5
                upper_bound = center * 1.5
                std_dev = (upper_bound - lower_bound) / 6
                des_r = np.round(
                    np.clip(np.random.normal(center, std_dev, 1), lower_bound, upper_bound), 2
                )[0]

            elif dataset_name == "KITTI":
                # Define the range of possible values based on the center
                center = cfg.patch.des_r
                if center == 3.0:
                    possible_values = [2.0, 2.5, 3.0, 3.5, 4.0]
                elif center == 0.3:
                    possible_values = [0.2, 0.25, 0.3, 0.35, 0.4]

                # Assign probabilities (optional: uniform probabilities)
                probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]  # Adjust probabilities as needed

                # Select a value from possible_values based on the probabilities
                des_r = np.random.choice(possible_values, p=probabilities)
            else:
                des_r = cfg.patch.des_r

            is_aligned_to_global_z = data_source["is_aligned_to_global_z"]
            src = self.Desc(src_fds_pcd[None], src_kpt[None], des_r, is_aligned_to_global_z)
            if self.config.stage == "Pose":
                # SO(2) augmentation
                tgt = self.Desc(
                    tgt_fds_pcd[None], tgt_kpt[None], des_r, is_aligned_to_global_z, None, True
                )
            else:
                tgt = self.Desc(
                    tgt_fds_pcd[None], tgt_kpt[None], des_r, is_aligned_to_global_z, None
                )

            if self.config.stage == "Desc":
                # calc matching score of equivariant feature maps
                equi_score = self.equi_match(src["equi"], tgt["equi"])

                # if number of patches is less than 2, return None
                if src["rand_axis"].shape[0] < 2 or tgt["rand_axis"].shape[0] < 2:
                    print(
                        f"{data_source['src_id']} {data_source['tgt_id']} don't have enough patches"
                    )
                    return None

                # calculate gt lable in SO(2)
                lable = self.cal_so2_gt(src, tgt, gt_trans)

                return {
                    "src_kpt": src_kpt,
                    "tgt_kpt": tgt_kpt,
                    "src_des": src["desc"],
                    "tgt_des": tgt["desc"],
                    "equi_score": equi_score,
                    "gt_label": lable,
                }

            #######################
            # training matching
            #######################
            # predict index of SO(2) rotation
            # only consider part of elements along the elevation to speed up
            if src["rand_axis"].shape[0] < 2 or tgt["rand_axis"].shape[0] < 2:
                print(f"{data_source['src_id']} {data_source['tgt_id']} don't have enough patches")
                return None

            pred_ind = self.Pose(
                src["equi"][:, :, 1 : self.config.patch.ele_n - 1],
                tgt["equi"][:, :, 1 : self.config.patch.ele_n - 1],
            )
            # calculate gt lable in SO(2)
            lable = self.cal_so2_gt(src, tgt, gt_trans, False, aug_rotation=tgt["aug_rotation"])

            if self.config.stage == "Pose":
                return {
                    "pred_ind": pred_ind,
                    "gt_ind": lable,
                }

        else:
            ##################
            # inference
            ##################
            cfg = self.config
            #######################################
            # Geometric bootstrapping
            #######################################

            # Sphericity-based voxelization
            # Note: Sphericity-based voxelization has already been applied in the dataloader.
            src_fds_pcd, tgt_fds_pcd = data_source["src_fds_pcd"], data_source["tgt_fds_pcd"]

            # Density-aware radius estimation
            num_radius_estimation_points = cfg.patch.num_points_radius_estimate
            search_radius_thresholds = cfg.patch.search_radius_thresholds
            num_fps = cfg.patch.num_fps
            num_scales = cfg.patch.num_scales

            assert num_scales == len(
                search_radius_thresholds
            ), f"num_scales {num_scales} != num_thresholds {len(search_radius_thresholds)}"

            # NOTE(hlim): It only takes < ~ 0.0002 sec, which is negligible
            s_pts_flipped, t_pts_flipped = (
                src_fds_pcd[None].transpose(1, 2).contiguous(),
                tgt_fds_pcd[None].transpose(1, 2).contiguous(),
            )
            s_fps_idx = pnt2.furthest_point_sample(src_fds_pcd[None], num_radius_estimation_points)
            t_fps_idx = pnt2.furthest_point_sample(tgt_fds_pcd[None], num_radius_estimation_points)

            kpts1 = pnt2.gather_operation(s_pts_flipped, s_fps_idx).transpose(1, 2).contiguous()
            kpts2 = pnt2.gather_operation(t_pts_flipped, t_fps_idx).transpose(1, 2).contiguous()

            des_r_list = density_aware_radius_estimation(
                src_fds_pcd, kpts1, tgt_fds_pcd, kpts2, thresholds=search_radius_thresholds
            )

            #################################
            # BUFFER-X++: Incremental multi-scale processing with early exit
            #################################
            is_aligned_to_global_z = data_source["is_aligned_to_global_z"]
            enable_early_exit = cfg.match.get("enable_early_exit", True)
            enable_timing = cfg.test.get("enable_timing", False)

            # Accumulated results across scales
            R_accum = []
            t_accum = []
            ss_kpts_accum = []
            tt_kpts_accum = []

            init_pose = None
            num_inliers = 0
            num_inlier_ind = 0
            scales_used = 0

            desc_time_total = 0
            pose_time_total = 0
            pose_optim_total = 0

            if enable_timing:
                desc_timer = GPUTimer()
                pose_timer = GPUTimer()

            # Process scales incrementally (only extract keypoints/descriptors when needed)
            should_exit = False
            for i, des_r in enumerate(des_r_list):
                if enable_timing:
                    desc_timer.tic()
                #################################
                # Extract keypoints for this scale
                #################################
                s_pts_flipped, t_pts_flipped = (
                    src_fds_pcd[None].transpose(1, 2).contiguous(),
                    tgt_fds_pcd[None].transpose(1, 2).contiguous(),
                )
                s_fps_idx = pnt2.furthest_point_sample(src_fds_pcd[None], num_fps)
                t_fps_idx = pnt2.furthest_point_sample(tgt_fds_pcd[None], num_fps)

                src_kpts = (
                    pnt2.gather_operation(s_pts_flipped, s_fps_idx).transpose(1, 2).contiguous()
                )
                tgt_kpts = (
                    pnt2.gather_operation(t_pts_flipped, t_fps_idx).transpose(1, 2).contiguous()
                )

                #################################
                # Extract descriptors for this scale
                #################################
                src = self.Desc(src_fds_pcd[None], src_kpts, des_r, is_aligned_to_global_z)
                tgt = self.Desc(tgt_fds_pcd[None], tgt_kpts, des_r, is_aligned_to_global_z)

                src_des, src_equi, s_R = src["desc"], src["equi"], src["R"]
                tgt_des, tgt_equi, t_R = tgt["desc"], tgt["equi"], tgt["R"]

                #################################
                # Intra-scale matching
                #################################
                s_mids, t_mids = self.mutual_matching(src_des, tgt_des)

                ss_kpts = src_kpts[0, s_mids]
                ss_equi = src_equi[s_mids]
                ss_R = s_R[s_mids]
                tt_kpts = tgt_kpts[0, t_mids]
                tt_equi = tgt_equi[t_mids]
                tt_R = t_R[t_mids]

                if enable_timing:
                    desc_timer.toc()
                    desc_time_total += desc_timer.diff / 1000.0  # Convert ms to seconds

                if enable_timing:
                    pose_timer.tic()

                # Pairwise transformation estimation
                ind = self.Pose(
                    ss_equi[:, :, 1 : cfg.patch.ele_n - 1], tt_equi[:, :, 1 : cfg.patch.ele_n - 1]
                )

                # Recover pose
                angle = ind * 2 * np.pi / cfg.patch.azi_n + 1e-6
                angle_axis = torch.zeros_like(ss_kpts)
                angle_axis[:, -1] = 1
                angle_axis = angle_axis * angle[:, None]
                azi_R = Convert.axis_angle_to_rotation_matrix(angle_axis)

                R = tt_R @ azi_R @ ss_R.transpose(-1, -2)
                t = tt_kpts - (R @ ss_kpts.unsqueeze(-1)).squeeze()

                # Accumulate results from current scale
                R_accum.append(R)
                t_accum.append(t)
                ss_kpts_accum.append(ss_kpts)
                tt_kpts_accum.append(tt_kpts)
                scales_used = i + 1

                # Concatenate accumulated results
                R_cat = torch.cat(R_accum, dim=0)
                t_cat = torch.cat(t_accum, dim=0)
                ss_kpts_cat = torch.cat(ss_kpts_accum, dim=0)
                tt_kpts_cat = torch.cat(tt_kpts_accum, dim=0)

                # Cross-scale consensus maximization on accumulated results
                tss_kpts = ss_kpts_cat[None] @ R_cat.transpose(-1, -2) + t_cat[:, None]
                diffs = torch.sqrt(torch.sum((tss_kpts - tt_kpts_cat[None]) ** 2, dim=-1))
                thr = (
                    torch.sqrt(torch.sum(ss_kpts_cat**2, dim=-1))
                    * np.pi
                    / cfg.patch.azi_n
                    * cfg.match.inlier_th
                )
                sign = diffs < thr[None]
                inlier_num = torch.sum(sign, dim=-1)
                best_ind = torch.argmax(inlier_num)
                inlier_ind = torch.where(sign[best_ind])[0].detach().cpu().numpy()
                num_inlier_ind = len(inlier_ind)

                if enable_timing:
                    pose_timer.toc()
                    pose_time_total += pose_timer.diff / 1000.0  # Convert ms to seconds

                # Estimate pose for early exit mode (check confidence after each scale)
                if enable_early_exit and i == 0:
                    if enable_timing:
                        pose_timer.tic()
                    init_pose, num_inliers = self.pose_estimator.estimate_pose(
                        ss_kpts_cat, tt_kpts_cat, inlier_ind
                    )
                    if enable_timing:
                        pose_timer.toc()
                        pose_optim_total += pose_timer.diff / 1000.0  # Convert ms to seconds

                    # Check confidence for early exit
                    should_exit = self.pose_estimator.compute_confidence_score(num_inliers)

                    if should_exit:
                        # Early exit: confident enough with current scales
                        break

            # Use final accumulated results
            ss_kpts = ss_kpts_cat
            tt_kpts = tt_kpts_cat

            # Number of mutual inliers (after mutual matching across used scales)
            num_mutual_inliers = ss_kpts.shape[0]

            # Estimate pose once at the end for non-early-exit mode
            if not enable_early_exit or (enable_early_exit and not should_exit):
                if enable_timing:
                    pose_timer.tic()
                init_pose, num_inliers = self.pose_estimator.estimate_pose(
                    ss_kpts_cat, tt_kpts_cat, inlier_ind
                )
                if enable_timing:
                    pose_timer.toc()
                    pose_optim_total += pose_timer.diff / 1000.0  # Convert ms to seconds

            if cfg.test.pose_refine is True:
                device = ss_kpts.device
                init_pose_tensor = torch.FloatTensor(init_pose.copy()[None]).to(device)
                pose = self.post_refinement(init_pose_tensor, ss_kpts[None], tt_kpts[None])
                pose = pose[0].detach().cpu().numpy()
            else:
                pose = init_pose
            times = [desc_time_total, pose_time_total, pose_optim_total]
            return pose, times, num_inliers, num_mutual_inliers, num_inlier_ind, scales_used

    def mutual_matching(self, src_des, tgt_des):
        """
        Input
            - src_des:    [M, C]
            - tgt_des:    [N, C]
        Output:
            - s_mids:    [A]
            - t_mids:    [B]
        """
        # mutual knn
        ref = tgt_des.unsqueeze(0)
        query = src_des.unsqueeze(0)
        s_dis, s_idx = KNN(k=1, transpose_mode=True)(ref, query)
        sourceNNidx = s_idx[0, :, 0]

        ref = src_des.unsqueeze(0)
        query = tgt_des.unsqueeze(0)
        t_dis, t_idx = KNN(k=1, transpose_mode=True)(ref, query)
        targetNNidx = t_idx[0, :, 0]

        # find mutual correspondences (keep on GPU)
        mutual_mask = targetNNidx[sourceNNidx] == torch.arange(
            sourceNNidx.shape[0], device=sourceNNidx.device
        )
        s_mids = torch.where(mutual_mask)[0]
        t_mids = sourceNNidx[s_mids]

        return s_mids, t_mids

    def get_matching_indices(self, source, target, relt_pose, search_voxel_size):
        """
        Input
            - source:     [N, 3]
            - target:     [M, 3]
            - relt_pose:  [4, 4]
        Output:
            - match_inds: [C, 2]
        """
        source = transform(source, relt_pose)
        # knn
        ref = target.unsqueeze(0)
        query = source.unsqueeze(0)
        s_dis, s_idx = KNN(k=1, transpose_mode=True)(ref, query)
        sourceNNidx = s_idx[0]
        device = source.device
        min_ind = torch.cat(
            [torch.arange(source.shape[0])[:, None].to(device), sourceNNidx], dim=-1
        )
        min_val = s_dis.view(-1)
        match_inds = min_ind[min_val < search_voxel_size]

        return match_inds

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, weights=None):
        """
        [CVPR'21 PointDSC] (https://github.com/XuyangBai/PointDSC)
        Post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 4, 4]
        """
        assert initial_trans.shape[0] == 1

        inlier_threshold_list = [self.config.match.dist_th] * 20

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = transform(src_keypts, initial_trans)
            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            inlier_num = torch.sum(pred_inlier)
            if abs(int(inlier_num - previous_inlier_num)) < 1:
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = rigid_transform_3d(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                ## https://link.springer.com/article/10.1007/s10589-014-9643-2
                # weights=None,
                weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
                # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
            )
        return initial_trans

    def get_parameter(self):
        return list(self.parameters())


def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (
        torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6
    )
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
        torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6
    )

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    # find rotation (keep on GPU)
    U, S, Vt = torch.svd(H)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3, device=A.device, dtype=A.dtype)[None].repeat(bs, 1, 1)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0, 2, 1) - R @ centroid_A.permute(0, 2, 1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)


# TODO
# Modify this functions for calculating des_r


def squared_cdist(x, y):
    """
    Computes the squared Euclidean distance between two sets of points.

    Args:
        x (torch.Tensor): Tensor of shape (N, D)
        y (torch.Tensor): Tensor of shape (M, D)

    Returns:
        torch.Tensor: Squared Euclidean distance matrix of shape (N, M)
    """
    x2 = x.pow(2).sum(dim=-1, keepdim=True)  # (N, 1)
    y2 = y.pow(2).sum(dim=-1, keepdim=True).T  # (1, M)
    xy = torch.matmul(x, y.T)  # (N, M)
    return x2 + y2 - 2 * xy  # Squared Euclidean distance


def density_aware_radius_estimation(
    src_fds_pts,
    src_kpts,
    tgt_fds_pts,
    tgt_kpts,
    min_r=0.0,
    max_r=5.0,
    tolerance=0.01,
    thresholds=[5, 2, 0.5],
):
    """
    This function calculates the radius (des_r) for each keypoint such that the percentage of points
    within that radius matches the specified thresholds.

    Args:
        src_fds_pts (torch.Tensor): Source points of shape (N, 3).
        src_kpts (torch.Tensor): Keypoints of shape (b, num_keypts, 3).
        tgt_fds_pts (torch.Tensor): Target points of shape (M, 3).
        tgt_kpts (torch.Tensor): Keypoints of shape (b, num_keypts, 3).
        min_r (float): Minimum radius threshold.
        max_r (float): Maximum radius threshold.

    Returns:
        list: The calculated des_r values for the given percentages.
    """
    des_r_values = []

    if src_fds_pts.shape[0] > tgt_fds_pts.shape[0]:
        pts = src_fds_pts
        kpts = src_kpts
    else:
        pts = tgt_fds_pts
        kpts = tgt_kpts

    num_pts = pts.shape[0]
    num_kpts = kpts.shape[1]

    if pts.shape[0] > 200000:
        pts = pts[torch.randint(0, pts.shape[0], (200000,))]

    dists_sqr = squared_cdist(kpts, pts)

    # NOTE(hlim): This filtering reduces unnecessary computations by
    # pre-dividing with the maximum possible radius

    dists_sqr = dists_sqr[dists_sqr <= max_r * max_r]

    for threshold in thresholds:
        low, high = min_r, max_r  # Start with a wide search range for des_r
        des_r = 0.0
        while high - low > 1e-3:  # Precision threshold
            des_r = (low + high) / 2.0
            points_within_radius = (
                dists_sqr < des_r * des_r
            ).int()  # Binary mask for points within radius
            percentage = (
                points_within_radius.sum().float() / (num_pts * num_kpts) * 100
            )  # Percentage per keypoint
            percentage = percentage.item()

            if percentage < threshold - tolerance:
                low = des_r  # Increase des_r to capture more points
            elif percentage > threshold + tolerance:
                high = des_r  # Decrease des_r to capture fewer points
            else:
                break  # Close enough to the percentage

        des_r_values.append(round(des_r, 2))  # Round to 2 decimal places for consistency

    return des_r_values
