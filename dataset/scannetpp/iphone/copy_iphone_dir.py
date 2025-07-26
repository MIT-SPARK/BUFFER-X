import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

def copy_iphone_dirs(source_root: Path, target_root: Path, split_file_path: str):
    """
    Copies the 'iphone' directory from selected ScanNet++ scenes to a target directory.

    Args:
        source_root (Path): Root directory containing ScanNet++ scene folders.
        target_root (Path): Destination directory to copy iphone folders into.
        split_file_path (str): Path to the file listing scene IDs to process.
    """
    target_root.mkdir(parents=True, exist_ok=True)

    # Load scene IDs from the split file
    with open(split_file_path, "r") as f:
        scene_ids = [line.strip() for line in f if line.strip()]

    for scene_id in tqdm(scene_ids, desc="Copying selected iphone dirs"):
        scene_dir = source_root / scene_id
        iphone_dir = scene_dir / "iphone"

        if iphone_dir.exists():
            target_scene_dir = target_root / scene_id
            target_scene_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(iphone_dir, target_scene_dir / "iphone", dirs_exist_ok=True)
        else:
            print(f"[Warning] 'iphone' folder not found for scene ID: {scene_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy 'iphone' folders from ScanNet++ scenes based on split file")
    parser.add_argument(
        "--source_root",
        type=Path,
        default=Path("../../../datasets/scannetpp/scannet-plusplus/data"),
        help="Root path containing source ScanNet++ scene folders"
    )
    parser.add_argument(
        "--target_root",
        type=Path,
        default=Path("../../../datasets/Scannetpp_iphone/test"),
        help="Destination root where 'iphone' folders will be copied"
    )
    parser.add_argument(
        "--split_file_path",
        type=str,
        default=os.path.join("..", "..", "config", "splits", "test_scannetpp_iphone.txt"),
        help="Path to split file listing scene IDs"
    )
    args = parser.parse_args()

    copy_iphone_dirs(args.source_root, args.target_root, args.split_file_path)
