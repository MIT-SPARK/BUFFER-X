from functools import partial
from ThreeDMatch.dataset import ThreeDMatchDataset
import torch
import numpy as np

def collate_fn_descriptor(list_data, config):
    batched_points_list = []
    batched_lengths_list = []
    batched_features_list = []# = np.ones_like(input_points[0][:, :0]).astype(np.float32)
    batched_voxel_size_list = []
    batched_dataset_names = []
    batched_sphericity = []

    assert len(list_data) == 1
    list_data = list_data[0]
    
    s_pts, t_pts = list_data['src_fds_pts'], list_data['tgt_fds_pts']
    relt_pose = list_data['relt_pose']
    s_kpt, t_kpt = list_data['src_sds_pts'], list_data['tgt_sds_pts']
    src_id, tgt_id = list_data['src_id'], list_data['tgt_id']
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
    batched_voxel_sizes = torch.from_numpy(np.array(batched_voxel_size_list))
    batched_sphericity.append(list_data['sphericity'])
    batched_sphericity = torch.from_numpy(np.array(batched_sphericity)).float()

    ###############
    # Return inputs
    ###############
    dict_inputs = {
        'src_pcd_raw': torch.from_numpy(s_pts).float(),
        'tgt_pcd_raw': torch.from_numpy(t_pts).float(),
        'src_pcd': torch.from_numpy(src_kpt).float(),
        'tgt_pcd': torch.from_numpy(tgt_kpt).float(),
        'relt_pose': torch.from_numpy(relt_pose).float(),
        'src_id': src_id,
        'tgt_id': tgt_id,
        'voxel_sizes': batched_voxel_sizes,
        'dataset_names': batched_dataset_names,
        'sphericity': batched_sphericity,
        'src_pcd_real_raw': torch.from_numpy(list_data['src_pcd_raw']).float(),
        'tgt_pcd_real_raw': torch.from_numpy(list_data['tgt_pcd_raw']).float()
    }

    return dict_inputs


def get_dataloader(split, config, num_workers=16, shuffle=True, drop_last=True):
    dataset = ThreeDMatchDataset(
        split=split,
        config=config
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn_descriptor, config=config),
        drop_last=drop_last,
    )

    return dataloader