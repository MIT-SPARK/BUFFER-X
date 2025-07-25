import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Path to the root directory containing all scene ID folders
source_root = Path("../../../datasets/scannetpp/scannet-plusplus/data")

# Path to the destination directory where iphone/ folders will be copied
target_root = Path("../../../datasets/Scannetpp_iphone/test")
target_root.mkdir(parents=True, exist_ok=True)

# Path to the split file containing the list of scene IDs to process
split_file_path = os.path.join("..", "..", "config", "splits", "test_scannetpp_iphone.txt")

# Load scene IDs from the split file (one ID per line)
with open(split_file_path, "r") as f:
    scene_ids = [line.strip() for line in f if line.strip()]

# Loop through the scene IDs specified in the split file
for scene_id in tqdm(scene_ids, desc="Copying selected iphone dirs"):
    scene_dir = source_root / scene_id
    iphone_dir = scene_dir / "iphone"

    # Check if the 'iphone' directory exists for the scene
    if iphone_dir.exists():
        # Create the corresponding scene directory in the target location
        target_scene_dir = target_root / scene_id
        target_scene_dir.mkdir(parents=True, exist_ok=True)

        # Copy 'iphone' directory contents to target
        shutil.copytree(iphone_dir, target_scene_dir / "iphone", dirs_exist_ok=True)
    else:
        print(f"[Warning] 'iphone' folder not found for scene ID: {scene_id}")
