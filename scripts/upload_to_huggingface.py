#!/usr/bin/env python3
"""Create/update Hugging Face model and Space repos for BUFFER-X."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_CARD_DIR = ROOT_DIR / "huggingface" / "model_card"
SPACE_DIR = ROOT_DIR / "huggingface" / "space"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-repo-id",
        default=os.environ.get("BUFFERX_HF_MODEL_REPO", ""),
        help="Hugging Face model repo id, e.g. <org-or-user>/BUFFER-X.",
    )
    parser.add_argument(
        "--space-repo-id",
        default=os.environ.get("BUFFERX_HF_SPACE_REPO", ""),
        help="Optional Hugging Face Space repo id, e.g. <org-or-user>/buffer-x-demo.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="snapshot",
        help="Local pretrained snapshot directory to upload into the model repo.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repos as private when they do not already exist.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN when set.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without creating repos or uploading files.",
    )
    return parser.parse_args()


def require_huggingface_hub():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it with "
            "`pip install -e '.[hub]'`. "
            "For a full CUDA runtime install, use `./scripts/install.sh --with-hub`."
        ) from exc
    return HfApi


def iter_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return (
        p
        for p in path.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and p.name != ".DS_Store"
    )


def print_tree(label: str, path: Path) -> None:
    print(f"{label}: {path}")
    if not path.exists():
        print("  - missing")
        return
    for file_path in sorted(iter_files(path)):
        print(f"  - {file_path.relative_to(path)}")


def create_repo(
    api, repo_id: str, repo_type: str, private: bool, token: Optional[str]
) -> None:
    if repo_type == "space":
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                private=private,
                exist_ok=True,
                space_sdk="docker",
                token=token,
            )
            return
        except TypeError:
            pass

    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
        token=token,
    )


def upload_model_repo(api, args: argparse.Namespace) -> None:
    if not args.model_repo_id:
        raise ValueError("--model-repo-id is required to upload the model repo.")

    snapshot_dir = (ROOT_DIR / args.snapshot_dir).resolve()
    create_repo(api, args.model_repo_id, "model", args.private, args.token)

    api.upload_file(
        path_or_fileobj=str(MODEL_CARD_DIR / "README.md"),
        path_in_repo="README.md",
        repo_id=args.model_repo_id,
        repo_type="model",
        token=args.token,
    )

    license_path = ROOT_DIR / "LICENSE"
    if license_path.exists():
        api.upload_file(
            path_or_fileobj=str(license_path),
            path_in_repo="LICENSE",
            repo_id=args.model_repo_id,
            repo_type="model",
            token=args.token,
        )

    if snapshot_dir.exists():
        api.upload_folder(
            folder_path=str(snapshot_dir),
            path_in_repo="snapshot",
            repo_id=args.model_repo_id,
            repo_type="model",
            token=args.token,
            ignore_patterns=[".DS_Store", "**/.DS_Store"],
            commit_message="Upload BUFFER-X pretrained snapshots",
        )
    else:
        print(f"Skipping snapshot upload because {snapshot_dir} does not exist.")


def upload_space_repo(api, args: argparse.Namespace) -> None:
    if not args.space_repo_id:
        return
    create_repo(api, args.space_repo_id, "space", args.private, args.token)
    api.upload_folder(
        folder_path=str(SPACE_DIR),
        path_in_repo=".",
        repo_id=args.space_repo_id,
        repo_type="space",
        token=args.token,
        ignore_patterns=["__pycache__/**", "*.pyc", ".DS_Store", "**/.DS_Store"],
        commit_message="Upload BUFFER-X Space scaffold",
    )


def main() -> int:
    args = parse_args()
    snapshot_dir = (ROOT_DIR / args.snapshot_dir).resolve()

    if args.dry_run:
        print(f"Model repo: {args.model_repo_id or '(not set)'}")
        print(f"Space repo: {args.space_repo_id or '(not set)'}")
        print_tree("Model card files", MODEL_CARD_DIR)
        print_tree("Space files", SPACE_DIR)
        print_tree("Snapshot files", snapshot_dir)
        return 0

    HfApi = require_huggingface_hub()
    api = HfApi()

    upload_model_repo(api, args)
    upload_space_repo(api, args)

    print("Hugging Face upload completed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
