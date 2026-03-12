from functools import partial
import os
import re
import torch


def _extract_scene_name(list_data):
    """
    Infer a stable scene identifier across datasets from the sample metadata.
    """
    scene_name = list_data.get("scene_name")
    if scene_name is not None and str(scene_name) != "":
        return str(scene_name)

    dataset_name = str(list_data.get("dataset_name", ""))
    src_id = str(list_data.get("src_id", ""))
    sensor = str(list_data.get("sensor", ""))
    src_id = src_id.replace("\\", "/")
    parts = [p for p in src_id.split("/") if p]

    if dataset_name in {"3DMatch", "3DLoMatch"}:
        if "fragments" in parts:
            frag_idx = parts.index("fragments")
            if frag_idx + 1 < len(parts):
                return parts[frag_idx + 1]
        if parts:
            return parts[0]

    if dataset_name == "Scannetpp_iphone":
        if parts:
            return parts[0]

    if dataset_name == "Scannetpp_faro":
        if len(parts) >= 2 and parts[0] == "data":
            return parts[1]
        if parts:
            return parts[0]

    if dataset_name == "ETH":
        if parts:
            return parts[0]

    if dataset_name in {"KITTI", "WOD", "MIT", "Oxford", "KAIST", "TIERS", "TIERS_hetero"}:
        src_base = os.path.basename(src_id)
        drive_match = re.match(r"^(.*)_(\d+)$", src_base)
        drive = drive_match.group(1) if drive_match else src_base
        if dataset_name == "KITTI" and str(drive).isdigit():
            drive = f"{int(drive):02d}"
        return drive

    if dataset_name == "KAIST_hetero":
        return "KAIST"

    if dataset_name == "ModelNet40":
        # src_id/tgt_id are object-local frame ids (e.g., "airplane_0123")
        # so keep dataset-level grouping if object id is unavailable.
        return "ModelNet40"

    if sensor:
        return sensor

    if len(parts) >= 2:
        return parts[-2]
    if parts:
        return parts[0]
    if dataset_name:
        return dataset_name
    return "unknown"


def collate_fn_descriptor(list_data, config):
    """
    Generic collate function for dataset processing.
    """

    batched_voxel_size_list = []
    batched_dataset_names = []
    batched_sphericity = []

    assert len(list_data) == 1
    list_data = list_data[0]

    src_sds, tgt_sds = list_data["src_sds_pts"], list_data["tgt_sds_pts"]
    src_id, tgt_id = list_data["src_id"], list_data["tgt_id"]
    scene_name = _extract_scene_name(list_data)
    sensor = list_data.get("sensor", "")

    batched_voxel_size_list.append(list_data["voxel_size"])
    batched_dataset_names.append(list_data["dataset_name"])
    batched_sphericity.append(list_data["sphericity"])

    batched_voxel_sizes = torch.tensor(batched_voxel_size_list)
    batched_sphericity = torch.tensor(batched_sphericity, dtype=torch.float32)

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
    - Always downsampled using config-specified voxel size.
    """
    dict_inputs = {
        "src_fds_pcd": torch.tensor(list_data["src_fds_pts"], dtype=torch.float32),
        "tgt_fds_pcd": torch.tensor(list_data["tgt_fds_pts"], dtype=torch.float32),
        "src_sds_pcd": torch.tensor(src_sds[:, :3], dtype=torch.float32),
        "tgt_sds_pcd": torch.tensor(tgt_sds[:, :3], dtype=torch.float32),
        "relt_pose": torch.tensor(list_data["relt_pose"], dtype=torch.float32),
        "src_id": src_id,
        "tgt_id": tgt_id,
        "scene_name": scene_name,
        "sensor": sensor,
        "voxel_sizes": batched_voxel_sizes,
        "dataset_names": batched_dataset_names,
        "sphericity": batched_sphericity,
        "is_aligned_to_global_z": list_data["is_aligned_to_global_z"],
    }

    return dict_inputs


def get_dataloader(dataset, split, config, num_workers=16, shuffle=True, drop_last=True):
    """
    Generalized function to get dataloader for different datasets.
    """
    if dataset == "3DMatch":
        from .threedmatch import ThreeDMatchDataset as Dataset
    elif dataset == "Scannetpp_iphone":
        from .scannetpp_iphone import ScannetppIphoneDataset as Dataset
    elif dataset == "Scannetpp_faro":
        from .scannetpp_faro import ScannetppFaroDataset as Dataset
    elif dataset == "TIERS":
        from .tiers import TIERSDataset as Dataset
    elif dataset == "TIERS_hetero":
        from .tiers import TIERSHeteroDataset as Dataset
    elif dataset == "KITTI":
        from .kitti import KITTIDataset as Dataset
    elif dataset == "WOD":
        from .wod import WODDataset as Dataset
    elif dataset == "MIT":
        from .mit import MITDataset as Dataset
    elif dataset == "KAIST":
        from .kaist import KAISTDataset as Dataset
    elif dataset == "KAIST_hetero":
        from .kaist import KAISTHeteroDataset as Dataset
    elif dataset == "ETH":
        from .eth import ETHDataset as Dataset
    elif dataset == "Oxford":
        from .oxford import OxfordDataset as Dataset
    elif dataset == "ModelNet40":
        from .modelnet40 import ModelNet40Dataset as Dataset
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
