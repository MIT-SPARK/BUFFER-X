import torch.nn as nn
import torch.nn.functional as F
import models.patchnet as pn
from models.patch_embedder import MiniSpinNet
import pointnet2_ops.pointnet2_utils as pnt2
from knn_cuda import KNN
from utils.SE3 import *
from einops import rearrange
import kornia.geometry.conversions as Convert
import open3d as o3d
from utils.common import make_open3d_point_cloud
from utils.timer import Timer
import numpy as np


class EquiMatch(nn.Module):
    def __init__(self, config):
        super(EquiMatch, self).__init__()
        self.azi_n = config.patch.azi_n
        init_index = np.arange(self.azi_n)
        index_list = []
        for i in range(self.azi_n):
            cur_index = np.concatenate([init_index[self.azi_n - i:], init_index[:self.azi_n - i]])
            index_list.append(cur_index)
        self.index_list = np.array(index_list)

    def forward(self, Des1, Des2):
        [B, C, K, L] = Des1.shape
        index_list = torch.from_numpy(self.index_list).to(Des1.device)
        Des1 = Des1[:, :, :, torch.reshape(index_list, [-1])].reshape(
            [Des1.shape[0], Des1.shape[1], Des1.shape[2], self.azi_n, self.azi_n])
        Des1 = rearrange(Des1, 'b c k n l -> b c n k l').reshape([B, C, -1, K * L])
        Des2 = Des2.reshape([B, C, K * L])
        cor = torch.einsum('bfag,bfg->ba', Des1, Des2)
        return cor


class CostVolume(nn.Module):
    def __init__(self, config):
        super(CostVolume, self).__init__()
        self.azi_n = config.patch.azi_n
        init_index = np.arange(self.azi_n)
        index_list = []
        for i in range(self.azi_n):
            cur_index = np.concatenate([init_index[self.azi_n - i:], init_index[:self.azi_n - i]])
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
            [Des1.shape[0], Des1.shape[1], Des1.shape[2], self.azi_n, self.azi_n])
        Des1 = rearrange(Des1, 'b c k n l -> b c n k l')
        Des2 = Des2.unsqueeze(2)
        cost = Des1 - Des2
        cost = self.conv(cost).squeeze()
        prob = F.softmax(cost, dim=-1)
        ind = torch.sum(prob * torch.arange(0, self.azi_n)[None].to(prob.device), dim=-1)
        return ind


class buffer(nn.Module):
    def __init__(self, config):
        super(buffer, self).__init__()
        self.config = config
        self.config.stage = config.stage

        self.Desc = MiniSpinNet(config)
        self.Inlier = CostVolume(config)
        # equivariant feature matching
        self.equi_match = EquiMatch(config)

    def cal_so2_gt(self, src, tgt, gt_trans, integer=True, aug_rotation=None):
        src_des, src_equi, s_rand_axis, s_R, s_patches = src['desc'], src['equi'], src['rand_axis'], src['R'], src[
            'patches']
        tgt_des, tgt_equi, _, t_R, t_patches = tgt['desc'], tgt['equi'], tgt['rand_axis'], tgt['R'], tgt[
            'patches']
        # calculate gt lable in SO(2)
        t_rand_axis = torch.matmul(s_rand_axis[:, None], gt_trans[:3, :3].transpose(-1, -2))
        s_rand_axis = torch.matmul(s_rand_axis[:, None], s_R)
        t_rand_axis = torch.matmul(t_rand_axis, t_R)
        if aug_rotation is not None:
            t_rand_axis = t_rand_axis @ aug_rotation.transpose(-1, -2)
        z_axis = torch.zeros_like(t_rand_axis)
        z_axis[:, :, -1] = 1
        proj_t = F.normalize(t_rand_axis - torch.sum(t_rand_axis * z_axis, dim=-1, keepdim=True) * z_axis, p=2,
                             dim=-1)
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
        '''
        src_fds_pcd / tgt_fds_pcd:
        - First-level downsampled point clouds via voxelization.
        - Farthest Point Sampling (FPS) is applied on these points to obtain keypoints.
        - Patch descriptors are then computed by sampling neighborhoods from these fds points centered at each keypoint.
        - During training: downsampled using config-specified voxel size.
        - During testing: voxel size is automatically estimated for each sample.

        src_sds_pcd / tgt_sds_pcd:
        - Second-level downsampled point clouds via voxelization.
        - Used only during training as sampled keypoints for patch-based supervision. (e.g., loss, correspondence).
        - Always downsampled using config-specified voxel size.
        '''
        
        src_fds_pcd, tgt_fds_pcd = data_source['src_fds_pcd'], data_source['tgt_fds_pcd']

        if self.config.stage != 'test':
            
            src_sds_pts, tgt_sds_pts = data_source['src_sds_pcd'], data_source['tgt_sds_pcd']
            # find positive correspondences
            gt_trans = data_source['relt_pose']   
            match_inds = self.get_matching_indices(src_sds_pts, tgt_sds_pts, gt_trans, data_source['voxel_sizes'][0])

            dataset_name = self.config['data']['dataset']
            cfg = self.config

            # randomly sample some positive pairs to speed up the training
            if match_inds.shape[0] > self.config.train.pos_num:
                rand_ind = np.random.choice(range(match_inds.shape[0]), self.config.train.pos_num, replace=False)
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
            if dataset_name == '3DMatch':
                center = cfg.patch.des_r
                lower_bound = center * 0.5
                upper_bound = center * 1.5
                std_dev = (upper_bound - lower_bound) / 6
                des_r = np.round(np.clip(np.random.normal(center, std_dev, 1), lower_bound, upper_bound), 2)[0]
                
            elif dataset_name == 'KITTI':
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

            src = self.Desc(src_fds_pcd[None], src_kpt[None], des_r, dataset_name)
            if self.config.stage == 'Inlier':
                # SO(2) augmentation
                tgt = self.Desc(tgt_fds_pcd[None], tgt_kpt[None], des_r, dataset_name, None, True)
            else:
                tgt = self.Desc(tgt_fds_pcd[None], tgt_kpt[None], des_r, dataset_name, None)

            if self.config.stage == 'Desc':
                # calc matching score of equivariant feature maps
                equi_score = self.equi_match(src['equi'], tgt['equi'])

                # if number of patches is less than 2, return None
                if src['rand_axis'].shape[0] < 2 or tgt['rand_axis'].shape[0] < 2:
                    print(f"{data_source['src_id']} {data_source['tgt_id']} don't have enough patches")
                    return None
                
                # calculate gt lable in SO(2)
                lable = self.cal_so2_gt(src, tgt, gt_trans)

                return {'src_kpt': src_kpt,
                        'tgt_kpt': tgt_kpt,
                        'src_des': src['desc'],
                        'tgt_des': tgt['desc'],
                        'equi_score': equi_score,
                        'gt_label': lable,
                        }

            #######################
            # training matching
            #######################
            # predict index of SO(2) rotation
            # only consider part of elements along the elevation to speed up
            if src['rand_axis'].shape[0] < 2 or tgt['rand_axis'].shape[0] < 2:
                    print(f"{data_source['src_id']} {data_source['tgt_id']} don't have enough patches")
                    return None
                
            pred_ind = self.Inlier(src['equi'][:, :, 1:self.config.patch.ele_n - 1],
                                   tgt['equi'][:, :, 1:self.config.patch.ele_n - 1])
            # calculate gt lable in SO(2)
            lable = self.cal_so2_gt(src, tgt, gt_trans, False, aug_rotation=tgt['aug_rotation'])

            if self.config.stage == 'Inlier':
                return {'pred_ind': pred_ind,
                        'gt_ind': lable,
                        }

        else:
            #######################
            # inference
            ######################
            src_fds_pcd, tgt_fds_pcd = data_source['src_fds_pcd'], data_source['tgt_fds_pcd']

            dataset_name = self.config['data']['dataset']
            cfg = self.config
                    
            num_radius_estimation_points = cfg.patch.num_points_radius_estimate
            search_radius_thresholds = cfg.patch.search_radius_thresholds
                        
            num_fps = cfg.patch.num_fps
            num_scales = cfg.patch.num_scales
            
            assert num_scales == len(search_radius_thresholds), \
                f"num_scales {num_scales} != len(search_radius_thresholds) {len(search_radius_thresholds)}"
            
            # NOTE(hlim): It only takes < ~ 0.0002 sec, which is negligible
            s_pts_flipped, t_pts_flipped = src_fds_pcd[None].transpose(1, 2).contiguous(), tgt_fds_pcd[None].transpose(1,2).contiguous()
            s_fps_idx = pnt2.furthest_point_sample(src_fds_pcd[None], num_radius_estimation_points)
            t_fps_idx = pnt2.furthest_point_sample(tgt_fds_pcd[None], num_radius_estimation_points)

            kpts1 = pnt2.gather_operation(s_pts_flipped, s_fps_idx).transpose(1, 2).contiguous()
            kpts2 = pnt2.gather_operation(t_pts_flipped, t_fps_idx).transpose(1, 2).contiguous()
 
            des_r_list = find_des_r(src_fds_pcd, kpts1, tgt_fds_pcd, kpts2, thresholds = search_radius_thresholds)
                        
            ss_kpts_raw_list = [None] * num_scales
            tt_kpts_raw_list = [None] * num_scales

            desc_timer = Timer()
            desc_timer.tic()
            mutual_matching_total = 0
            inlier_total = 0
            correspondence_proposal_total = 0
            
            # Furthest point sampling 
            for i, des_r in enumerate(des_r_list):
                s_pts_flipped, t_pts_flipped = src_fds_pcd[None].transpose(1, 2).contiguous(), tgt_fds_pcd[None].transpose(1,2).contiguous()
                s_fps_idx = pnt2.furthest_point_sample(src_fds_pcd[None], num_fps)
                t_fps_idx = pnt2.furthest_point_sample(tgt_fds_pcd[None], num_fps)
                kpts1 = pnt2.gather_operation(s_pts_flipped, s_fps_idx).transpose(1, 2).contiguous()
                kpts2 = pnt2.gather_operation(t_pts_flipped, t_fps_idx).transpose(1, 2).contiguous()

                ss_kpts_raw_list[i] = kpts1
                tt_kpts_raw_list[i] = kpts2

            ss_des_list = [None] * num_scales
            tt_des_list = [None] * num_scales        
            # Compute descriptors
            for i, des_r in enumerate(des_r_list):
                src = self.Desc(src_fds_pcd[None], ss_kpts_raw_list[i], des_r, dataset_name)
                tgt = self.Desc(tgt_fds_pcd[None], tt_kpts_raw_list[i], des_r, dataset_name)
                ss_des_list[i] = src
                tt_des_list[i] = tgt

            R_list = [None] * num_scales
            t_list = [None] * num_scales
            ss_kpts_list = [None] * num_scales
            tt_kpts_list = [None] * num_scales
            # Match
            for i, (src_kpts, src, tgt_kpts, tgt) in enumerate(zip(ss_kpts_raw_list, 
                                                    ss_des_list,
                                                    tt_kpts_raw_list,
                                                    tt_des_list)):
                src_des, src_equi, s_R = src['desc'], src['equi'], src['R']
                tgt_des, tgt_equi, t_R = tgt['desc'], tgt['equi'], tgt['R']

                mutual_matching_timer = Timer()
                mutual_matching_timer.tic()
                # Perform mutual matching using equivariant feature maps
                s_mids, t_mids = self.mutual_matching(src_des, tgt_des)
                                
                ss_kpts = src_kpts[0, s_mids]
                ss_equi = src_equi[s_mids]
                ss_R = s_R[s_mids]
                tt_kpts = tgt_kpts[0, t_mids]
                tt_equi = tgt_equi[t_mids]
                tt_R = t_R[t_mids]
                
                mutual_matching_timer.toc()   
                mutual_matching_total += mutual_matching_timer.diff
                
                inlier_timer = Timer()
                inlier_timer.tic()
                # Calculate inliers
                ind = self.Inlier(ss_equi[:, :, 1:cfg.patch.ele_n - 1],
                                tt_equi[:, :, 1:cfg.patch.ele_n - 1])
                inlier_timer.toc()
                inlier_total += inlier_timer.diff

                correspondence_proposal_timer = Timer()
                correspondence_proposal_timer.tic()
                # Recover pose
                angle = ind * 2 * np.pi / cfg.patch.azi_n + 1e-6
                angle_axis = torch.zeros_like(ss_kpts)
                angle_axis[:, -1] = 1
                angle_axis = angle_axis * angle[:, None]
                azi_R = Convert.axis_angle_to_rotation_matrix(angle_axis)
                
                R = tt_R @ azi_R @ ss_R.transpose(-1, -2)
                t = tt_kpts - (R @ ss_kpts.unsqueeze(-1)).squeeze()
                R_list[i] = R
                t_list[i] = t
                ss_kpts_list[i] = ss_kpts
                tt_kpts_list[i] = tt_kpts
                correspondence_proposal_timer.toc()
                correspondence_proposal_total += correspondence_proposal_timer.diff
                    
            desc_timer.toc()
            R = torch.cat(R_list, dim=0)
            t = torch.cat(t_list, dim=0)
            ss_kpts = torch.cat(ss_kpts_list, dim=0)
            tt_kpts = torch.cat(tt_kpts_list, dim=0)

            correspondence_proposal_timer = Timer()
            correspondence_proposal_timer.tic()     
            
            # Find the best R and t
            tss_kpts = ss_kpts[None] @ R.transpose(-1, -2) + t[:, None]
            diffs = torch.sqrt(torch.sum((tss_kpts - tt_kpts[None]) ** 2, dim=-1))
            thr = torch.sqrt(torch.sum(ss_kpts ** 2, dim=-1)) * np.pi / cfg.patch.azi_n * cfg.match.inlier_th
            sign = diffs < thr[None]
            inlier_num = torch.sum(sign, dim=-1)  # Number of inliers for the current des_r (instead inlier num maximum, ratio maximum?)
            best_ind = torch.argmax(inlier_num)
            inlier_ind = torch.where(sign[best_ind] == True)[0].detach().cpu().numpy()
            
            correspondence_proposal_timer.toc()
            correspondence_proposal_total += correspondence_proposal_timer.diff

            # use RANSAC to calculate pose
            pcd0 = make_open3d_point_cloud(ss_kpts.detach().cpu().numpy(), [1, 0.706, 0])
            pcd1 = make_open3d_point_cloud(tt_kpts.detach().cpu().numpy(), [0, 0.651, 0.929])
            corr = o3d.utility.Vector2iVector(np.array([inlier_ind, inlier_ind]).T)
            ransac_timer = Timer()
            ransac_timer.tic()
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                pcd0, pcd1, corr, cfg.match.dist_th,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(cfg.match.similar_th),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(cfg.match.dist_th)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(cfg.match.iter_n,
                                                                     cfg.match.confidence))

            init_pose = result.transformation
            ransac_timer.toc()
            
            if cfg.test.pose_refine is True:
                pose = self.post_refinement(torch.FloatTensor(init_pose[None]).cuda(), ss_kpts[None], tt_kpts[None], dataset_name)
                pose = pose[0].detach().cpu().numpy()
            else:
                pose = init_pose
            times = [desc_timer.diff, mutual_matching_total,
                        inlier_total, correspondence_proposal_total, ransac_timer.diff]
            return pose, times

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
        sourceNNidx = s_idx[0, :, 0].detach().cpu().numpy()

        ref = src_des.unsqueeze(0)
        query = tgt_des.unsqueeze(0)
        t_dis, t_idx = KNN(k=1, transpose_mode=True)(ref, query)
        targetNNidx = t_idx[0, :, 0].detach().cpu().numpy()

        # find mutual correspondences
        s_mids = np.where((targetNNidx[sourceNNidx] - np.arange(sourceNNidx.shape[0])) == 0)[0]
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
        min_ind = torch.cat([torch.arange(source.shape[0])[:, None].cuda(), sourceNNidx], dim=-1)
        min_val = s_dis.view(-1)
        match_inds = min_ind[min_val < search_voxel_size]

        return match_inds

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, dataset_name=None, weights=None):
        """
        [CVPR'21 PointDSC] (https://github.com/XuyangBai/PointDSC)
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 4, 4]
        """
        assert initial_trans.shape[0] == 1
        if dataset_name in ['3DMatch', '3DLoMatch', 'ETH']:
            inlier_threshold_list = [0.10] * 20
        else:  # for KITTI
            inlier_threshold_list = [1.2] * 20

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
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (
            torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
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

def find_des_r(src_fds_pts, src_kpts, tgt_fds_pts, tgt_kpts, min_r=0.0, max_r=5.0, tolerance=0.01, thresholds =[5, 2, 0.5]):
    """
    Finds the des_r values corresponding to the given target percentages of points within the radius.
    
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

    # NOTE(hlim): This filtering reduces unnecessary computations by pre-dividing with the maximum possible radius 

    dists_sqr = dists_sqr[dists_sqr <= max_r * max_r]
    
    for threshold in thresholds:
        low, high = min_r, max_r  # Start with a wide search range for des_r
        des_r = 0.0
        while high - low > 1e-3:  # Precision threshold
            des_r = (low + high) / 2.0
            points_within_radius = (dists_sqr < des_r * des_r).int()  # Binary mask for points within radius
            percentage = points_within_radius.sum().float() / (num_pts * num_kpts) * 100  # Percentage per keypoint
            percentage = percentage.item()
            
            if percentage < threshold - tolerance:
                low = des_r  # Increase des_r to capture more points
            elif percentage > threshold + tolerance:
                high = des_r  # Decrease des_r to capture fewer points
            else:
                break  # Close enough to the percentage

        des_r_values.append(round(des_r, 2))  # Round to 2 decimal places for consistency
        
    return des_r_values