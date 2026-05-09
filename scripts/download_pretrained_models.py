#!/usr/bin/env python3
"""Download BUFFER-X pretrained snapshots from Hugging Face or Dropbox."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DROPBOX_SNAPSHOT_URL = (
    "https://www.dropbox.com/scl/fi/jmqcrngnul2kfyw8iqsi1/snapshot.zip"
    "?rlkey=k0sy6v1s6a57p7rqzygdmnkb6&st=h4m9j1x6&dl=1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["auto", "hf", "dropbox"],
        default="auto",
        help="Download source. 'auto' uses Hugging Face when --repo-id is set.",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("BUFFERX_HF_MODEL_REPO", ""),
        help="Hugging Face model repo id, e.g. <org-or-user>/BUFFER-X.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face branch, tag, or commit.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Repository root where the snapshot/ directory should be placed.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN when set.",
    )
    parser.add_argument(
        "--dropbox-url",
        default=DROPBOX_SNAPSHOT_URL,
        help="Override the fallback Dropbox snapshot URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing snapshot/ directory.",
    )
    return parser.parse_args()


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def prepare_snapshot_dir(output_dir: Path, force: bool) -> None:
    snapshot_dir = output_dir / "snapshot"
    if snapshot_dir.exists():
        if not force:
            raise FileExistsError(
                f"{snapshot_dir} already exists. Pass --force to replace it."
            )
        shutil.rmtree(snapshot_dir)


def download_from_hugging_face(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.repo_id:
        raise ValueError("--repo-id is required when --source hf is used.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face downloads. "
            "Install it with `pip install -e '.[hub]'`. "
            "For a full CUDA runtime install, use `./scripts/install.sh --with-hub`."
        ) from exc

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=str(output_dir),
        allow_patterns=["snapshot/**"],
        token=args.token,
    )


def download_from_dropbox(args: argparse.Namespace, output_dir: Path) -> None:
    zip_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="bufferx_snapshot_", suffix=".zip", delete=False
        ) as f:
            zip_path = Path(f.name)

        print(f"Downloading pretrained snapshots from {args.dropbox_url}")
        urllib.request.urlretrieve(args.dropbox_url, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)
    finally:
        if zip_path is not None and zip_path.exists():
            zip_path.unlink()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)

    source = args.source
    if source == "auto":
        source = "hf" if args.repo_id else "dropbox"

    prepare_snapshot_dir(output_dir, args.force)

    if source == "hf":
        download_from_hugging_face(args, output_dir)
    else:
        download_from_dropbox(args, output_dir)

    snapshot_dir = output_dir / "snapshot"
    if not snapshot_dir.exists():
        raise RuntimeError(f"Download finished, but {snapshot_dir} was not created.")

    print(f"Pretrained snapshots are ready at {snapshot_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
