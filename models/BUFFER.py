import torch.nn as nn
import torch.nn.functional as F
import models.patchnet as pn
from models.point_learner import EFCNN, DetNet
from models.patch_embedder import MiniSpinNet
import pointnet2_ops.pointnet2_utils as pnt2
from knn_cuda import KNN
from utils.SE3 import *
from einops import rearrange
import kornia.geometry.conversions as Convert
import open3d as o3d
from ThreeDMatch.dataset import make_open3d_point_cloud
from utils.timer import Timer


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

        self.Ref = EFCNN(config)
        self.Desc = MiniSpinNet(config)
        self.Keypt = DetNet(config)
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
        """
        Input
            - src_pts:    [bs, N, 3]
            - src_kpt:    [bs, M, 3]
            - tgt_pts:    [bs, N, 3]
            - tgt_kpt:    [bs, M, 3]
            - gt_trans:   [bs, 4, 4]
        Output:
            -
        """

        src_pts, tgt_pts = data_source['src_pcd'], data_source['tgt_pcd']
        src_pcd_raw, tgt_pcd_raw = data_source['src_pcd_raw'], data_source['tgt_pcd_raw']
        len_src_f = data_source['stack_lengths'][0][0]

        if self.config.stage != 'test':

            # find positive correspondences
            gt_trans = data_source['relt_pose']
            match_inds = self.get_matching_indices(src_pts, tgt_pts, gt_trans, data_source['voxel_sizes'][0])
            if self.config['data']['dataset'] == 'SuperSet':
                dataset_name = data_source["dataset_names"][0]
                cfg = self.config[dataset_name]
            else: 
                dataset_name = self.config['data']['dataset']
                cfg = self.config

            #######################
            # training ref axis
            #######################
            axis, eps, branch = self.Ref(data_source)

            # split into src and tgt
            src_axis = axis[:len_src_f]
            tgt_axis = axis[len_src_f:]
            src_s = eps[:len_src_f]
            tgt_s = eps[len_src_f:]

            # normalized and oriented axis
            src_axis = F.normalize(src_axis, p=2, dim=1)
            tgt_axis = F.normalize(tgt_axis, p=2, dim=1)
            mask = (torch.sum(-src_axis * src_pts, dim=1) < 0).float().unsqueeze(1)
            src_axis = src_axis * (1 - mask) - src_axis * mask
            mask = (torch.sum(-tgt_axis * tgt_pts, dim=1) < 0).float().unsqueeze(1)
            tgt_axis = tgt_axis * (1 - mask) - tgt_axis * mask

            src_axis = src_axis[match_inds[:, 0]]
            src_s = src_s[match_inds[:, 0]]
            tgt_axis = tgt_axis[match_inds[:, 1]]
            tgt_s = tgt_s[match_inds[:, 1]]
            
            if src_axis.shape[0] == 0 or tgt_axis.shape[0] == 0:
                print(f"{data_source['src_id']} {data_source['tgt_id']} has no axis")
                return None
            if self.config.stage == 'Ref':
                return {'src_ref': src_axis,
                        'tgt_ref': tgt_axis,
                        'src_s': src_s,
                        'tgt_s': tgt_s,
                        }
            
            if src_s is None or tgt_s is None:
                print(f"{data_source['src_id']} {data_source['tgt_id']} has no keypts")
                return None

            # randomly sample some positive pairs to speed up the training
            if match_inds.shape[0] > self.config.train.pos_num:
                rand_ind = np.random.choice(range(match_inds.shape[0]), self.config.train.pos_num, replace=False)
                match_inds = match_inds[rand_ind]
                src_axis = src_axis[rand_ind]
                tgt_axis = tgt_axis[rand_ind]
            src_kpt = src_pts[match_inds[:, 0]]
            tgt_kpt = tgt_pts[match_inds[:, 1]]
            
            if src_kpt.shape[0] == 0 or tgt_kpt.shape[0] == 0:
                print(f"{data_source['src_id']} {data_source['tgt_id']} has no keypts")
                return None
            #######################
            # training descriptor
            #######################
            # calculate feature descriptor
            src = self.Desc(src_pcd_raw[None], src_kpt[None], dataset_name, src_axis[None])
            if self.config.stage == 'Inlier':
                # SO(2) augmentation
                tgt = self.Desc(tgt_pcd_raw[None], tgt_kpt[None], dataset_name, tgt_axis[None], True)
            else:
                tgt = self.Desc(tgt_pcd_raw[None], tgt_kpt[None], dataset_name, tgt_axis[None])

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
            # training detector
            #######################
            if self.config.stage == 'Keypt':
                det_score = self.Keypt(data_source, branch)

                src_s, tgt_s = det_score[:len_src_f], det_score[len_src_f:]
                src_s, tgt_s = src_s[match_inds[:, 0]], tgt_s[match_inds[:, 1]]

                return {'src_kpt': src_kpt,
                        'src_s': src_s,
                        'tgt_s': tgt_s,
                        'src_des': src['desc'],
                        'tgt_des': tgt['desc'],
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
            src_pts, tgt_pts = data_source['src_pcd'], data_source['tgt_pcd']
            src_pcd_raw, tgt_pcd_raw = data_source['src_pcd_raw'], data_source['tgt_pcd_raw']
            len_src_f = data_source['stack_lengths'][0][0]
            gt_trans = data_source['relt_pose']
            
            if self.config['data']['dataset'] == 'SuperSet':
                dataset_name = data_source["dataset_names"][0]
                cfg = self.config[dataset_name]
            else: 
                dataset_name = self.config['data']['dataset']
                cfg = self.config

            ref_timer = Timer()
            ref_timer.tic()
            axis, eps, branch = self.Ref(data_source)
            src_axis, tgt_axis = axis[:len_src_f], axis[len_src_f:]

            # normalized and oriented axis
            src_axis = F.normalize(src_axis, p=2, dim=1)
            tgt_axis = F.normalize(tgt_axis, p=2, dim=1)
            mask = (torch.sum(-src_axis * src_pts, dim=1) < 0).float().unsqueeze(1)
            src_axis = src_axis * (1 - mask) - src_axis * mask
            mask = (torch.sum(-tgt_axis * tgt_pts, dim=1) < 0).float().unsqueeze(1)
            tgt_axis = tgt_axis * (1 - mask) - tgt_axis * mask
            ref_timer.toc()

            keypt_timer = Timer()
            keypt_timer.tic()
            det_score = self.Keypt(data_source, branch)
            src_s, tgt_s = det_score[:len_src_f], det_score[len_src_f:]

            # select keypts by detection scores
            src_s, tgt_s = src_s[:, 0], tgt_s[:, 0]
            s_det_idx, t_det_idx = torch.where(src_s > cfg.point.keypts_th), torch.where(
                tgt_s > cfg.point.keypts_th)
            src_pts, tgt_pts = src_pts[s_det_idx[0]], tgt_pts[t_det_idx[0]]
            s_axis, t_axis = src_axis[s_det_idx[0]], tgt_axis[t_det_idx[0]]
            keypt_timer.toc()
            
            fps_timer = Timer()
            fps_timer.tic()
            # fps
            s_pts_flipped, t_pts_flipped = src_pts[None].transpose(1, 2).contiguous(), tgt_pts[None].transpose(1,
                                                                                                               2).contiguous()
            s_axis_flipped, t_axis_flipped = s_axis[None].transpose(1, 2).contiguous(), t_axis[None].transpose(1,
                                                                                                               2).contiguous()
            s_fps_idx = pnt2.furthest_point_sample(src_pts[None], cfg.point.num_keypts)
            t_fps_idx = pnt2.furthest_point_sample(tgt_pts[None], cfg.point.num_keypts)
            kpts1 = pnt2.gather_operation(s_pts_flipped, s_fps_idx).transpose(1, 2).contiguous()
            kpts2 = pnt2.gather_operation(t_pts_flipped, t_fps_idx).transpose(1, 2).contiguous()
            k_axis1 = pnt2.gather_operation(s_axis_flipped, s_fps_idx).transpose(1, 2).contiguous()
            k_axis2 = pnt2.gather_operation(t_axis_flipped, t_fps_idx).transpose(1, 2).contiguous()
            fps_timer.toc()
            
            desc_timer = Timer()
            desc_timer.tic()
            # calculate descriptor
            src = self.Desc(src_pcd_raw[None], kpts1, dataset_name, k_axis1)
            tgt = self.Desc(tgt_pcd_raw[None], kpts2, dataset_name, k_axis2)
            src_des, src_equi, s_rand_axis, s_R, s_patches = src['desc'], src['equi'], src['rand_axis'], src['R'], src[
                'patches']
            tgt_des, tgt_equi, t_rand_axis, t_R, t_patches = tgt['desc'], tgt['equi'], tgt['rand_axis'], tgt['R'], tgt[
                'patches']
            desc_timer.toc()
            
            mutual_matching_timer = Timer()
            mutual_matching_timer.tic()
            # use equivariant feature maps
            # mutual_matching
            s_mids, t_mids = self.mutual_matching(src_des, tgt_des)
            ss_kpts = kpts1[0, s_mids]
            ss_equi = src_equi[s_mids]
            ss_R = s_R[s_mids]
            tt_kpts = kpts2[0, t_mids]
            tt_equi = tgt_equi[t_mids]
            tt_R = t_R[t_mids]
            mutual_matching_timer.toc()
            
            inlier_timer = Timer()
            inlier_timer.tic()
            ind = self.Inlier(ss_equi[:, :, 1:cfg.patch.ele_n - 1],
                              tt_equi[:, :, 1:cfg.patch.ele_n - 1])
            inlier_timer.toc()
            
            correspondence_proposal_timer = Timer()
            correspondence_proposal_timer.tic()
            # recover pose
            angle = ind * 2 * np.pi / cfg.patch.azi_n + 1e-6
            angle_axis = torch.zeros_like(ss_kpts)
            angle_axis[:, -1] = 1
            angle_axis = angle_axis * angle[:, None]
            # By Hyungtae: To resolve warning message about kornia 0.7.0
            # azi_R = Convert.angle_axis_to_rotation_matrix(angle_axis)
            azi_R = Convert.axis_angle_to_rotation_matrix(angle_axis)
            R = tt_R @ azi_R @ ss_R.transpose(-1, -2)
            t = tt_kpts - (R @ ss_kpts.unsqueeze(-1)).squeeze()

            # find the best R, t
            tss_kpts = ss_kpts[None] @ R.transpose(-1, -2) + t[:, None]
            diffs = torch.sqrt(torch.sum((tss_kpts - tt_kpts[None]) ** 2, dim=-1))
            thr = torch.sqrt(
                torch.sum(ss_kpts ** 2, dim=-1)) * np.pi / cfg.patch.azi_n * cfg.match.inlier_th
            sign = diffs < thr[None]
            inlier_num = torch.sum(sign, dim=-1)
            best_ind = torch.argmax(inlier_num)
            inlier_ind = torch.where(sign[best_ind] == True)[0].detach().cpu().numpy()
            correspondence_proposal_timer.toc()

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
                pose = self.post_refinement(torch.FloatTensor(init_pose[None]).cuda(), ss_kpts[None], tt_kpts[None])
                pose = pose[0].detach().cpu().numpy()
            else:
                pose = init_pose
            # print(f"ref_time: {ref_timer.diff:.2f}s "
            #       f"keypt_time: {keypt_timer.diff:.2f}s "
            #       f"fps_time: {fps_timer.diff:.2f}s "
            #       f"desc_time: {desc_timer.diff:.2f}s "
            #       f"mutual_matching_time: {mutual_matching_timer.diff:.2f}s "
            #       f"inlier_time: {inlier_timer.diff:.2f}s "
            #       f"correspondence_proposal_time: {correspondence_proposal_timer.diff:.2f}s "
            #       f"ransac_time: {ransac_timer.diff:.2f}s ")
            times = [ref_timer.diff, keypt_timer.diff, fps_timer.diff, desc_timer.diff, mutual_matching_timer.diff,
                        inlier_timer.diff, correspondence_proposal_timer.diff, ransac_timer.diff]
            return pose, src_axis, tgt_axis, times

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

    def post_refinement(self, initial_trans, src_keypts, tgt_keypts, weights=None):
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
        if self.config.data.dataset in ['3DMatch', '3DLoMatch', 'ETH']:
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
