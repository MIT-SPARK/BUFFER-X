import torch.nn as nn
import torch.nn.functional as F
import models.patchnet as pn
import utils.common
import pointnet2_ops.pointnet2_utils as pnt2
from knn_cuda import KNN
from loss.desc_loss import cdist
from utils.SE3 import *
from einops import repeat, rearrange
import kornia.geometry.conversions as Convert
import open3d as o3d
from ThreeDMatch.dataset import make_open3d_point_cloud
import copy
from utils.timer import Timer


class MiniSpinNet(nn.Module):
    def __init__(self, config):
        super(MiniSpinNet, self).__init__()
        self.config = config
        # Data-dependent param: `des_r`
        self.des_r = config.patch.des_r
        self.patch_sample = config.patch.num_points_per_patch
        self.rad_n = config.patch.rad_n
        self.azi_n = config.patch.azi_n
        self.ele_n = config.patch.ele_n
        self.delta = config.patch.delta
        self.voxel_sample = config.patch.voxel_sample
        self.pnt_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )

        self.pool_layer = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )

        self.conv_net = pn.Cylindrical_Net(inchan=16, dim=32)
        # self.conv_net = pn.Cylindrical_UNet(inchan=16, dim=32)

    def forward(self, pts, kpts, des_r, dataset_name, z_axis=None, is_aug=False):
        
        # if superset dataset
        # des_r = self.config[dataset_name].patch.des_r
        
        # if single dataset
        # des_r = self.des_rs
        
        # extract patches
        init_patch = self.select_patches(pts, kpts, vicinity=des_r, patch_sample=self.patch_sample)
        init_patch = init_patch.squeeze(0)

        # align with reference axis
        patches, rand_axis, R = self.axis_align(init_patch, dataset_name, z_axis)
        patches = self.normalize(patches, des_r)

        # SO(2) augmentation
        if is_aug is True:
            angles = np.zeros([patches.shape[0], 3])
            angles[:, -1] = 1
            angles = angles * np.random.random([patches.shape[0], 1]) * 2 * np.pi
            angles = torch.FloatTensor(angles).to(patches.device)
            # By Hyungtae: To resolve warning message about kornia 0.7.0
            # aug_rotation = Convert.angle_axis_to_rotation_matrix(angles)
            aug_rotation = Convert.axis_angle_to_rotation_matrix(angles)
        else:
            aug_rotation = np.eye(3)[None].repeat(patches.shape[0], axis=0)
            aug_rotation = torch.FloatTensor(aug_rotation).to(patches.device)
        patches = patches @ aug_rotation.transpose(-1, -2)
        rand_axis = (rand_axis.unsqueeze(1) @ aug_rotation.transpose(-1, -2)).squeeze(1)

        # spatial point transformer
        inv_patches = self.SPT(patches, 1, self.delta / self.rad_n)

        # Vanilla SpinNet
        new_points = inv_patches.permute(0, 3, 1, 2)  # (B, C_in, npoint, nsample+1), input features
        x = self.pnt_layer(new_points)
        x = F.max_pool2d(x, kernel_size=(1, x.shape[-1])).squeeze(3)  # (B, C_in, npoint)
        del new_points
        x = x.view(x.shape[0], x.shape[1], self.rad_n, self.ele_n, self.azi_n)
        x, mid = self.conv_net(x)

        w = self.pool_layer(x)
        f = F.avg_pool2d(x * w, kernel_size=(x.shape[2], x.shape[3]))
        f = F.normalize(f.view(f.shape[0], -1), p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        return {'desc': f,
                'equi': x,
                'rand_axis': rand_axis,
                'R': R,
                'patches': patches,
                'aug_rotation': aug_rotation}

    def select_patches(self, pts, refer_pts, vicinity, patch_sample=1024):
        B, N, C = pts.shape

        # shuffle pts if pts is not orderless
        index = np.random.choice(N, N, replace=False)
        pts = pts[:, index]

        group_idx = pnt2.ball_query(vicinity, patch_sample, pts, refer_pts)
        pts_trans = pts.transpose(1, 2).contiguous()
        new_points = pnt2.grouping_operation(
            pts_trans, group_idx
        )
        new_points = new_points.permute([0, 2, 3, 1])
        mask = group_idx[:, :, 0].unsqueeze(2).repeat(1, 1, patch_sample)
        mask = (group_idx == mask).float()
        mask[:, :, 0] = 0
        mask[:, :, patch_sample - 1] = 1
        mask = mask.unsqueeze(3).repeat([1, 1, 1, C])
        new_pts = refer_pts.unsqueeze(2).repeat([1, 1, patch_sample, 1])
        local_patches = new_points * (1 - mask).float() + new_pts * mask.float()
        
        # For calculating the average number of unique points per patch
        
        # vicinities = [0.005, 0.01, 0.025, 0.05, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]

        # for vicinity in vicinities:
        #     print(f"\nRunning patch selection with vicinity: {vicinity}")

        #     group_idx = pnt2.ball_query(vicinity, patch_sample, pts, refer_pts)
        #     pts_trans = pts.transpose(1, 2).contiguous()
        #     new_points = pnt2.grouping_operation(pts_trans, group_idx)
        #     new_points = new_points.permute([0, 2, 3, 1])
            
        #     mask = group_idx[:, :, 0].unsqueeze(2).repeat(1, 1, patch_sample)
        #     mask = (group_idx == mask).float()
        #     mask[:, :, 0] = 0
        #     mask[:, :, patch_sample - 1] = 1
        #     mask = mask.unsqueeze(3).repeat([1, 1, 1, C])
            
        #     new_pts = refer_pts.unsqueeze(2).repeat([1, 1, patch_sample, 1])
        #     local_patches = new_points * (1 - mask).float() + new_pts * mask.float()
            
        #     unique_points_per_patch = []
        #     # Iterate over each patch
        #     for patch in local_patches[0]:  # Ignore batch dimension (assuming batch size is 1)
        #         # Use torch.unique with dim=0 to remove duplicate 3D points directly on the GPU
        #         unique_points = torch.unique(patch, dim=0)
                
        #         # Store the number of unique points in the list
        #         unique_points_per_patch.append(unique_points.shape[0])

        #     # Calculate the average number of unique points per patch
        #     average_unique_points = sum(unique_points_per_patch) / len(unique_points_per_patch)
        #     print(f"Average number of unique points per patch: {average_unique_points:.2f}")

        del mask
        del new_points
        del group_idx
        del new_pts
        del pts
        del pts_trans

        return local_patches

    def axis_align(self, input, dataset, z_axis=None):
        center = input[:, -1, :3]
        delta_x = input[:, :, :3] - center.unsqueeze(1)  # (B, npoint, 3), normalized coordinates
        
        z_axis_datasets = {'3DMatch', '3DLoMatch', 'NSS', 'Scannetpp_iphone', 'Scannetpp_faro'}
        rand_axis_datasets = {'KITTI', 'ETH', 'WOD', 'NewerCollege', 'KimeraMulti'}
        
        if dataset in z_axis_datasets:
            if z_axis is None:
                z_axis = utils.common.cal_Z_axis(delta_x, ref_point=center)
                z_axis = utils.common.l2_norm(z_axis, axis=1)
            else:
                z_axis = z_axis[0]
            R = utils.common.RodsRotatFormula(z_axis,
                                              torch.FloatTensor([0, 0, 1]).expand_as(z_axis))
            delta_x = torch.matmul(delta_x, R)

            # for calculate gt lable
            rand_axis = torch.zeros_like(center)
            rand_axis[:, -1] = 1
            rand_axis = torch.cross(z_axis, rand_axis)
            rand_axis = F.normalize(rand_axis, p=2, dim=-1)

        else:
            rand_axis = torch.zeros_like(center)
            rand_axis[:, 0] = 1
            R = torch.eye(3).to(center.device)
            R = R[None].repeat([center.shape[0], 1, 1])

        return delta_x, rand_axis, R

    def SPT(self, delta_x, des_r, voxel_r):

        # partition the local surface along elevator, azimuth, radial dimensions
        S2_xyz = torch.FloatTensor(utils.common.get_voxel_coordinate(radius=des_r,
                                                                     rad_n=self.rad_n,
                                                                     azi_n=self.azi_n,
                                                                     ele_n=self.ele_n))

        pts_xyz = S2_xyz.view(1, -1, 3).repeat([delta_x.shape[0], 1, 1]).cuda()
        # query points in sphere
        new_points = utils.common.sphere_query(delta_x, pts_xyz, radius=voxel_r,
                                               nsample=self.voxel_sample)
        # transform rotation-variant coords into rotation-invariant coords
        new_points = utils.common.var_to_invar(new_points, self.rad_n, self.azi_n, self.ele_n)

        return new_points

    def normalize(self, pts, radius):
        delta_x = pts / (torch.ones_like(pts).to(pts.device) * radius)

        return delta_x

    def get_parameter(self):
        return list(self.parameters())
