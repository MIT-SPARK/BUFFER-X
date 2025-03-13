from functools import partial
import torch

def collate_fn_descriptor(list_data, config):
    """
    Generic collate function for dataset processing.
    """
    batched_lengths_list = []
    batched_voxel_size_list = []
    batched_dataset_names = []
    batched_sphericity = []

    assert len(list_data) == 1
    list_data = list_data[0]
    
    src_kpt, tgt_kpt = list_data['src_sds_pts'], list_data['tgt_sds_pts']
    src_id, tgt_id = list_data['src_id'], list_data['tgt_id']

    batched_lengths_list.append(len(src_kpt))
    batched_lengths_list.append(len(tgt_kpt))
    batched_voxel_size_list.append(list_data['voxel_size'])
    batched_dataset_names.append(list_data['dataset_name'])
    batched_sphericity.append(list_data['sphericity'])

    batched_voxel_sizes = torch.tensor(batched_voxel_size_list)
    batched_sphericity = torch.tensor(batched_sphericity, dtype=torch.float32)

    dict_inputs = {
        'src_pcd_raw': torch.tensor(list_data['src_fds_pts'], dtype=torch.float32),
        'tgt_pcd_raw': torch.tensor(list_data['tgt_fds_pts'], dtype=torch.float32),
        'src_pcd': torch.tensor(src_kpt[:, :3], dtype=torch.float32),
        'tgt_pcd': torch.tensor(tgt_kpt[:, :3], dtype=torch.float32),
        'relt_pose': torch.tensor(list_data['relt_pose'], dtype=torch.float32),
        'src_id': src_id,
        'tgt_id': tgt_id,
        'voxel_sizes': batched_voxel_sizes,
        'dataset_names': batched_dataset_names,
        'sphericity': batched_sphericity
    }
    
    return dict_inputs

def get_dataloader(dataset, split, config, num_workers=16, shuffle=True, drop_last=True):
    """
    Generalized function to get dataloader for different datasets.
    """
    if dataset == "3DMatch":
        from .threedmatch import ThreeDMatchDataset as Dataset
    elif dataset == "KITTI":
        from .kitti import KITTIDataset as Dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dataset = Dataset(split=split, config=config)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_fn_descriptor, config=config),
        drop_last=drop_last,
    )

    return dataloader