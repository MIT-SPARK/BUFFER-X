from models.KPConv.blocks import *
from models.vn_layers import *

architecture = [
    'VNN_first',
    'VNN_resnetb_strided',
    'VNN_resnetb',
    'VNN_resnetb_strided',
    'VNN_resnetb',
    'nearest_upsample',
    'VN',
    'nearest_upsample',
    'VN',
]


class Encoder(nn.Module):

    def __init__(self, config, need_param=True):
        super(Encoder, self).__init__()

        ############
        # Parameters
        ############
        # Current radius of convolution and feature dimension
        self.layer = 0

        # ToDo. Currently, just fix the reference voxel size
        self.reference_voxel_size = config.data.voxel_size_0
        self.r = self.reference_voxel_size * config.point.conv_radius

        # Data-agnostic parameters
        self.in_dim = config.point.in_feats_dim // 3
        self.out_dim = config.point.first_feats_dim // 3
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))

        #####################
        # List Encoder blocks
        #####################
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(architecture):

            # Check equivariance
            if ('equivariant' in block) and (not self.out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(self.in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            if need_param:
                # Apply the good block function defining tf ops
                self.encoder_blocks.append(block_decider(block,
                                                         self.r,
                                                         self.in_dim,
                                                         self.out_dim,
                                                         self.layer,
                                                         self.reference_voxel_size))

            # Update dimension of input from output
            self.in_dim = self.out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                self.layer += 1
                self.r *= 2
                self.out_dim *= 2


class Decoder(Encoder):

    def __init__(self, config, need_encoder=True):
        Encoder.__init__(self, config, need_encoder)
        #####################
        # List Decoder blocks
        #####################
        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in architecture[start_i + block_i - 1]:
                self.in_dim += self.encoder_skip_dims[self.layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     self.r,
                                                     self.in_dim,
                                                     self.out_dim,
                                                     self.layer,
                                                     self.reference_voxel_size))

            # Update dimension of input from output
            self.in_dim = self.out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                self.layer -= 1
                self.r *= 0.5
                self.out_dim = self.out_dim // 2


class DetNet(Decoder):

    def __init__(self, config):
        Decoder.__init__(self, config, need_encoder=False)
        self.config = config

        self.invar_layer = nn.Sequential(
            VNStdFeature(self.out_dim, dim=4, normalize_frame=False, negative_slope=0.0),
            nn.Conv1d(self.out_dim * 3, self.out_dim * 2, kernel_size=1),
            nn.InstanceNorm1d(self.out_dim * 2),
            nn.Conv1d(self.out_dim * 2, self.out_dim, kernel_size=1),
            nn.InstanceNorm1d(self.out_dim),
            nn.Conv1d(self.out_dim, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, batch, branch):
        x = batch['features'].clone().detach()
        [N, C] = x.shape

        # another decoder branch
        x, skip_x = branch['bottle_feature'], branch['skip_feature']
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        input = x[None].permute(0, 2, 1).view(1, -1, 3, N)
        score = self.invar_layer(input)

        return score[0].transpose(-1, -2)


class EFCNN(Decoder):

    def __init__(self, config):
        Decoder.__init__(self, config, need_encoder=True)

        self.fc_layer = nn.Sequential(
            VNLinearLeakyReLU(self.out_dim, self.out_dim // 2, dim=4),
            VNLinearLeakyReLU(self.out_dim // 2, 1, dim=4),
        )
        self.inv_layer = nn.Sequential(
            VNStdFeature(self.out_dim, dim=4, normalize_frame=False, negative_slope=0.0),
            nn.Conv1d(self.out_dim * 3, self.out_dim * 2, kernel_size=1),
            nn.InstanceNorm1d(self.out_dim * 2),
            nn.Conv1d(self.out_dim * 2, self.out_dim, kernel_size=1),
            nn.InstanceNorm1d(self.out_dim),
            nn.Conv1d(self.out_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        # Get input features
        x = batch['features'].clone().detach()

        #################################
        # 1. joint encoder part
        skip_x, skip_x_copy = [], []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
                skip_x_copy.append(x)
            x = block_op(x, batch)
        bottle_feature = x
        skip_feature = skip_x_copy
        #################################
        # 2. decoder part
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        #################################
        # 3. regress axis
        N, C = x.shape
        input = x[None].permute(0, 2, 1).view(1, -1, 3, N)
        # VN conv
        output_features = self.fc_layer(input)
        est_axis = output_features.reshape(1, -1, N)
        eps = self.inv_layer(input)
        #################################

        return est_axis[0].transpose(-1, -2), eps[0].transpose(-1, -2), {'skip_feature': skip_feature,
                                                                         'bottle_feature': bottle_feature}


# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#

def block_decider(block_name,
                  radius,
                  in_dim,
                  out_dim,
                  layer_ind,
                  reference_voxel_size,
                  scale=1.0):
    if block_name == 'VN':
        return VNBlock(in_dim, out_dim)

    elif block_name in 'VNN_first':
        return VNNBlock(block_name, in_dim, out_dim, radius, reference_voxel_size, layer_ind, pooling='mean', mode='6')

    elif block_name in ['VNN',
                        'VNN_strided']:
        return VNNBlock(block_name, in_dim, out_dim, radius, reference_voxel_size, layer_ind, pooling='mean', mode='1')

    elif block_name in ['VNN_resnetb',
                        'VNN_resnetb_strided']:
        return VNNResnetBlock(block_name, in_dim, out_dim, radius, reference_voxel_size, layer_ind, pooling='mean', mode='1')

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)

    elif block_name == 'global_average':
        return GlobalAverageBlock()

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


class VNBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        """
        Initialize a standard VN block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        """

        super(VNBlock, self).__init__()
        self.mlp = VNLinearLeakyReLU(in_dim, out_dim, dim=4)
        return

    def forward(self, x, batch=None):
        N, C = x.shape
        input = x[None].permute(0, 2, 1).view(1, -1, 3, N)
        # VN conv
        output_features = self.mlp(input)
        output_features = output_features.reshape(1, -1, N)
        return output_features[0].transpose(-1, -2)


class VNNBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, reference_voxel_size, layer_ind, pooling='mean', mode='0'):
        """
        Initialize the first VNN block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param mode: '0' -- feature,
                     '1' -- feature, xyz,
                     '2' -- feature, xyz, mean,
                     '3' -- feature, xyz, proj_xyz,
                     '4' -- feature, xyz, mean, proj_xyz
                     '5' -- feature, xyz, feature X xyz
                     '6' -- feature, xyz, feature X xyz, mean,
                     '7' -- feature, xyz, feature X xyz, mean, proj_xyz
        """
        super(VNNBlock, self).__init__()

        # Get other parameters
        self.radius = radius
        self.reference_voxel_size = reference_voxel_size
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.mode = mode

        if pooling == 'max':
            self.pool = VNMaxPool(out_dim)
        elif pooling == 'mean':
            self.pool = mean_pool
        elif pooling == 'atten':
            self.pool = VNAttenPool(out_dim)

        if mode == '0':
            in_dim = in_dim
        elif mode == '1':
            in_dim = in_dim + 1
        elif mode == '2' or mode == '3' or mode == '5':
            in_dim = in_dim + 2
        elif mode == '4' or mode == '6':
            in_dim = in_dim + 3
        elif mode == '7':
            in_dim = in_dim + 4

        self.conv = VNLinearLeakyReLU(in_dim, out_dim)
        return

    def forward(self, x, batch):
        if 'strided' in self.block_name:
            q_pts = batch['points'][self.layer_ind + 1]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['pools'][self.layer_ind]
        else:
            q_pts = batch['points'][self.layer_ind]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['neighbors'][self.layer_ind]

        N, K = neighb_inds.shape

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Replace the fake points by the corresponding query points
        mask = (neighbors == 1e6)
        neighbors = mask * q_pts[:, None] + neighbors * (~mask)

        # Center every neighborhood
        eqv_neighbors = neighbors - q_pts.unsqueeze(1)

        #########################
        # scale normalization
        scale = self.reference_voxel_size / batch["voxel_sizes"][0]
        eqv_neighbors = eqv_neighbors / scale

        if self.mode == '0' and min(x.shape) == 0:
            raise ValueError('Features can not be empty!')

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, neighb_inds)

        if self.mode == '0':
            input = neighb_x
        elif self.mode == '1':
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors], dim=-1)
        elif self.mode == '2':
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            mean_cor = q_pts.unsqueeze(1) - mean  # neighbors - q_pts.unsqueeze(1).mean(0, keepdim=True) #
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors, mean], dim=-1)
        elif self.mode == '3':
            # calculate projection
            d = torch.sum(eqv_neighbors ** 2, dim=-1, keepdim=True).sqrt()
            proj_xyz = self.radius / d * eqv_neighbors
            # replace the nan element by zero
            proj_xyz = torch.nan_to_num(proj_xyz)
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors, proj_xyz], dim=-1)
        elif self.mode == '4':
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            mean_cor = neighbors - mean
            # calculate projection
            d = torch.sum(eqv_neighbors ** 2, dim=-1, keepdim=True).sqrt()
            proj_xyz = self.radius / d * eqv_neighbors
            # replace the nan element by zero
            proj_xyz = torch.nan_to_num(proj_xyz)
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors, mean_cor, proj_xyz], dim=-1)
        elif self.mode == '5':
            cros = torch.cross(neighb_x, eqv_neighbors)
            input = torch.cat([neighb_x, eqv_neighbors, cros], dim=-1)
        elif self.mode == '6':
            cros = torch.cross(neighb_x, eqv_neighbors)
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            input = torch.cat([neighb_x, eqv_neighbors, cros, mean], dim=-1)
        elif self.mode == '7':
            cros = torch.cross(neighb_x, eqv_neighbors)
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            # calculate projection
            d = torch.sum(eqv_neighbors ** 2, dim=-1, keepdim=True).sqrt()
            proj_xyz = self.radius / d * eqv_neighbors
            # replace the nan element by zero
            proj_xyz = torch.nan_to_num(proj_xyz)
            input = torch.cat([neighb_x, eqv_neighbors, cros, mean, proj_xyz], dim=-1)

        input = input[None].permute(0, 3, 1, 2).view(1, -1, 3, N, K)

        # VN conv
        input = self.conv(input)

        # pooling
        output_features = self.pool(input)
        output_features = output_features.view(1, -1, N)

        return output_features[0].transpose(-1, -2)


class VNNResnetBlock(nn.Module):

    def __init__(self, block_name, in_dim, out_dim, radius, reference_voxel_size, layer_ind, pooling='mean', mode='0'):
        """
        Initialize the first VNN block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param radius: current radius of convolution
        :param mode: '0' -- feature,
                     '1' -- feature, xyz,
                     '2' -- feature, xyz, mean,
                     '3' -- xyz, mean, feature
                     '4' -- xyz, mean, proj_xyz
                     '5' -- xyz, mean, proj_xyz, feature
        """
        super(VNNResnetBlock, self).__init__()

        # Get other parameters
        self.radius = radius
        self.reference_voxel_size = reference_voxel_size
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.mode = mode

        if pooling == 'max':
            self.pool = VNMaxPool(out_dim // 2)
        elif pooling == 'mean':
            self.pool = mean_pool
        elif pooling == 'atten':
            self.pool = VNAttenPool(out_dim // 2)

        if mode == '0':
            in_dim_ = in_dim
        elif mode == '1':
            in_dim_ = in_dim + 1
        elif mode == '2' or mode == '3' or mode == '5':
            in_dim_ = in_dim + 2
        elif mode == '4' or mode == '6':
            in_dim_ = in_dim + 3
        elif mode == '7':
            in_dim_ = in_dim + 4

        self.conv = VNLinearLeakyReLU(in_dim_, out_dim // 2)
        self.unary = VNLinearLeakyReLU(out_dim // 2, out_dim, dim=4)

        self.unary_shortcut = VNLinearLeakyReLU(in_dim, out_dim, dim=4)
        return

    def forward(self, features, batch):

        if 'strided' in self.block_name:
            q_pts = batch['points'][self.layer_ind + 1]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['pools'][self.layer_ind]
        else:
            q_pts = batch['points'][self.layer_ind]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['neighbors'][self.layer_ind]

        N, K = neighb_inds.shape

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]

        # Replace the fake points by the corresponding query points
        mask = (neighbors == 1e6)
        neighbors = mask * q_pts[:, None] + neighbors * (~mask)

        # Center every neighborhood
        eqv_neighbors = neighbors - q_pts.unsqueeze(1)

        #########################
        # scale normalization
        scale = self.reference_voxel_size / batch["voxel_sizes"][0]
        eqv_neighbors = eqv_neighbors / scale

        if self.mode == '0' and min(features.shape) == 0:
            raise ValueError('Features can not be empty!')

        # Add a zero feature for shadow neighbors
        x = torch.cat((features, torch.zeros_like(features[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, neighb_inds)

        if self.mode == '0':
            input = neighb_x
        elif self.mode == '1':
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors], dim=-1)
        elif self.mode == '2':
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            mean_cor = q_pts.unsqueeze(1) - mean  # neighbors - q_pts.unsqueeze(1).mean(0, keepdim=True) #
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors, mean], dim=-1)
        elif self.mode == '3':
            # calculate projection
            d = torch.sum(eqv_neighbors ** 2, dim=-1, keepdim=True).sqrt()
            proj_xyz = self.radius / d * eqv_neighbors
            # replace the nan element by zero
            proj_xyz = torch.nan_to_num(proj_xyz)
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors, proj_xyz], dim=-1)
        elif self.mode == '4':
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            mean_cor = neighbors - mean
            # calculate projection
            d = torch.sum(eqv_neighbors ** 2, dim=-1, keepdim=True).sqrt()
            proj_xyz = self.radius / d * eqv_neighbors
            # replace the nan element by zero
            proj_xyz = torch.nan_to_num(proj_xyz)
            # concatenate
            input = torch.cat([neighb_x, eqv_neighbors, mean_cor, proj_xyz], dim=-1)
        elif self.mode == '5':
            cros = torch.cross(neighb_x, eqv_neighbors)
            input = torch.cat([neighb_x, eqv_neighbors, cros], dim=-1)
        elif self.mode == '6':
            cros = torch.cross(neighb_x, eqv_neighbors)
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            input = torch.cat([neighb_x, eqv_neighbors, cros, mean], dim=-1)
        elif self.mode == '7':
            cros = torch.cross(neighb_x, eqv_neighbors)
            N, K, C = neighbors.shape
            # calculate mean
            mean = eqv_neighbors.mean(-2, keepdim=True).repeat([1, K, 1])
            # calculate projection
            d = torch.sum(eqv_neighbors ** 2, dim=-1, keepdim=True).sqrt()
            proj_xyz = self.radius / d * eqv_neighbors
            # replace the nan element by zero
            proj_xyz = torch.nan_to_num(proj_xyz)
            input = torch.cat([neighb_x, eqv_neighbors, cros, mean, proj_xyz], dim=-1)

        input = input[None].permute(0, 3, 1, 2).view(1, -1, 3, N, K)

        # VN conv
        input = self.conv(input)

        # pooling
        x = self.pool(input)

        # Second upscaling mlp
        x = self.unary(x)

        # Shortcut
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)
        else:
            shortcut = features
        N, C = shortcut.shape
        shortcut = shortcut[None].permute(0, 2, 1).view(1, -1, 3, N)
        shortcut = self.unary_shortcut(shortcut)

        output_features = x + shortcut
        output_features = output_features.reshape(1, -1, N)

        return output_features[0].transpose(-1, -2)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \********************/
#

def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig ** 2 + eps))


def closest_pool(x, inds):
    """
    Pools features from the closest neighbors. WARNING: this function assumes the neighbors are ordered.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] Only the first column is used for pooling
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get features for each pooling location [n2, d]
    return gather(x, inds[:, 0])


def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features


def global_average(x, batch_lengths):
    """
    Block performing a global average over batch pooling
    :param x: [N, D] input features
    :param batch_lengths: [B] list of batch lengths
    :return: [B, D] averaged features
    """

    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):
        # Average features for each batch cloud
        averaged_features.append(torch.mean(x[i0:i0 + length], dim=0))

        # Increment for next cloud
        i0 += length

    # Average features in each batch
    return torch.stack(averaged_features)
