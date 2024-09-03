from functools import partial
from KITTI.dataset import KITTIDataset
import torch
import numpy as np
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from models.point_learner import architecture
from models.KPConv.lib.timer import Timer


num_layer = 1
for block_i, block in enumerate(architecture):
    if ('pool' in block or 'strided' in block):
        num_layer = num_layer + 1


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):
    timer = Timer()
    last_display = timer.total_time

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.point.conv_radius) ** 3))
    neighb_hists = np.zeros((num_layer, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        timer.tic()
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in
                  batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)
        timer.toc()

        if timer.total_time - last_display > 0.1:
            last_display = timer.total_time
            print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print(f'neighborhood_limits: {neighborhood_limits}\n')

    return neighborhood_limits


def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0,
                                  random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                                batches_len,
                                                                                features=features,
                                                                                classes=labels,
                                                                                sampleDl=sampleDl,
                                                                                max_p=max_p,
                                                                                verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(
            s_labels)

def batch_grid_subsampling_and_searching_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, max_neighbors=15,
                                  random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    For faster speed, finding `up_i` runs together
    """
    if (features is None) and (labels is None):
        s_points, s_len, indices_for_upconv = cpp_subsampling.subsample_and_search_batch(points,
                                                                     batches_len,
                                                                     sampleDl=sampleDl,
                                                                     max_p=max_p,
                                                                     max_neighbors=max_neighbors,
                                                                     verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(indices_for_upconv)

        # s_points, s_len = cpp_subsampling.subsample_batch(points,
        #                                                   batches_len,
        #                                                   sampleDl=sampleDl,
        #                                                   max_p=max_p,
        #                                                   verbose=verbose)
        # return torch.from_numpy(s_points), torch.from_numpy(s_len)


    else:
        raise NotImplementedError("Currently, other options are not supported")


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def collate_fn_descriptor(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_lengths_list = []
    batched_features_list = []
    batched_voxel_size_list = []
    batched_dataset_names = []

    assert len(list_data) == 1
    list_data = list_data[0]

    s_pts, t_pts = list_data['src_fds_pts'], list_data['tgt_fds_pts']
    relt_pose = list_data['relt_pose']
    s_kpt, t_kpt = list_data['src_sds_pts'], list_data['tgt_sds_pts']
    src_kpt = s_kpt[:, :3]
    tgt_kpt = t_kpt[:, :3]
    src_f = s_kpt[:, 3:]
    tgt_f = t_kpt[:, 3:]
    batched_points_list.append(src_kpt)
    batched_points_list.append(tgt_kpt)
    batched_features_list.append(src_f)
    batched_features_list.append(tgt_f)
    batched_lengths_list.append(len(src_kpt))
    batched_lengths_list.append(len(tgt_kpt))
    batched_voxel_size_list.append(list_data['voxel_size'])
    batched_dataset_names.append(list_data['dataset_name'])

    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0))
    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()
    batched_voxel_sizes = torch.from_numpy(np.array(batched_voxel_size_list))

    # Starting radius of convolutions
    r_normal = config.data.voxel_size_0 * config.point.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(architecture) - 1 and not ('upsample' in architecture[block_i + 1]):
                continue

        # print("\033[1;32m" ,block_i, block, "radius: ", r_normal, ", layer: " , layer, "layer_blocks", layer_blocks, "\033[0m")
        # Convolution neighbors indices
        # *****************************
        # last_display = timer.total_time

        # HT: More tightly optimized              
        if layer_blocks and ('pool' in block or 'strided' in block):
            # Convolutions are done in this layer, compute the neighbors with the good radius
            # New subsampling length
            dl = 2 * r_normal / config.point.conv_radius

            # Subsampled points
            # pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)
            pool_p, pool_b, up_i = batch_grid_subsampling_and_searching_kpconv(batched_points, batched_lengths, sampleDl=dl, max_neighbors=neighborhood_limits[layer])

            r = r_normal
            # print("\033[1;32mdl:" , dl, "r: ", r, "\033[0m")

            # Originally, it was decoupled
            # conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
            #                                 neighborhood_limits[layer])
            # 
            # # Subsample indices
            # pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
            #                                 neighborhood_limits[layer])

            # What if fetching neighbors simultaneously? (and It worked)
            concat_queries= torch.cat((batched_points, pool_p), dim=0)
            concat_batches= torch.cat((batched_lengths, pool_b), dim=0)

            conv_and_pool_i = batch_neighbors_kpconv(concat_queries, batched_points, concat_batches, batched_lengths, r,
                                            neighborhood_limits[layer])

            len_conv_i = batched_lengths[0] + batched_lengths[1]

            conv_i = conv_and_pool_i[:len_conv_i, :]
            pool_i = conv_and_pool_i[len_conv_i:, :]

            # Upsample indices (with the radius of the next layer to keep wanted density)
            # up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
            #                               neighborhood_limits[layer])
            # print(neighborhood_limits[layer])
            # print("up_i shape:", up_i.shape)
            # print("batch_points shape:", batched_points.shape)

        elif layer_blocks:
            r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])

            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)


        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2.0
        layer += 1
        layer_blocks = []

    ###############
    # Return inputs
    ###############
    dict_inputs = {
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'src_pcd_raw': torch.from_numpy(s_pts).float(),
        'tgt_pcd_raw': torch.from_numpy(t_pts).float(),
        'src_pcd': torch.from_numpy(src_kpt).float(),
        'tgt_pcd': torch.from_numpy(tgt_kpt).float(),
        'relt_pose': torch.from_numpy(relt_pose).float(),
        'voxel_sizes': batched_voxel_sizes,
        'dataset_names': batched_dataset_names,
    }

    return dict_inputs


def get_dataloader(split, config, num_workers=16, shuffle=True, drop_last=True):
    dataset = KITTIDataset(
        split=split,
        config=config
    )

    # calibrate the number of neighborhood
    neighborhood_limits = calibrate_neighbors(dataset, config, collate_fn=collate_fn_descriptor)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn_descriptor, config=config, neighborhood_limits=neighborhood_limits),
        drop_last=drop_last,
    )

    return dataloader
