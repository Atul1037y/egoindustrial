#!/usr/bin/env python3
"""Download and prepare datasets for EgoIndustrial training."""

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

DATASETS = {
    "epic_kitchens": {
        "name": "EPIC-KITCHENS-100",
        "annotations_url": "https://github.com/epic-kitchens/epic-kitchens-100-annotations/archive/refs/heads/master.zip",
        "annotations_dir": "epic_kitchens/annotations",
        "videos_url": "https://epic-kitchens.github.io/2021/downloads.html",
        "videos_dir": "epic_kitchens/videos",
        "requires_login": True,
    },
    "assembly101": {
        "name": "Assembly101",
        "annotations_url": "https://assembly-101.s3.amazonaws.com/annotations.zip",
        "annotations_dir": "assembly101/annotations",
        "videos_url": "https://assembly-101.s3.amazonaws.com/videos.zip",
        "videos_dir": "assembly101/videos",
        "requires_login": False,
    },
    "holoassist": {
        "name": "HoloAssist",
        "annotations_url": "https://holoassist.s3.amazonaws.com/annotations.zip",
        "annotations_dir": "holoassist/annotations",
        "videos_url": "https://holoassist.s3.amazonaws.com/videos.zip",
        "videos_dir": "holoassist/videos",
        "requires_login": False,
    },
}


def download_file(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=desc
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def unzip_file(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip file."""
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def download_epic_kitchens(data_root: Path, annotations_only: bool = False) -> bool:
    """Download EPIC-KITCHENS-100 dataset."""
    print("\n" + "="*60)
    print("EPIC-KITCHENS-100 Dataset")
    print("="*60)

    dataset_root = Path(data_root) / "epic_kitchens"
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Download annotations
    annot_dir = dataset_root / "annotations"
    annot_dir.mkdir(parents=True, exist_ok=True)

    annot_zip = dataset_root / "annotations.zip"
    if not annot_dir.exists() or not any(annot_dir.iterdir()):
        print("Downloading EPIC-KITCHENS-100 annotations...")
        if download_file(DATASETS["epic_kitchens"]["annotations_url"], annot_zip, "Annotations"):
            print("Extracting annotations...")
            if unzip_file(annot_zip, dataset_root):
                # Move extracted files to correct location
                extracted = dataset_root / "epic-kitchens-100-annotations-master"
                if extracted.exists():
                    import shutil
                    for item in extracted.iterdir():
                        shutil.move(str(item), str(annot_dir))
                    shutil.rmtree(extracted)
                annot_zip.unlink(missing_ok=True)
                print("Annotations downloaded successfully!")
            else:
                return False
        else:
            print("Annotations already exist, skipping...")

    if annotations_only:
        return True

    # Videos require manual download
    videos_dir = dataset_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "!"*60)
    print("EPIC-KITCHENS-100 VIDEOS REQUIRE MANUAL DOWNLOAD")
    print("!"*60)
    print(f"Please download videos from: {DATASETS['epic_kitchens']['videos_url']}")
    print(f"Extract to: {videos_dir}")
    print("Expected structure:")
    print(f"  {videos_dir}/train/P01/..., P02/..., etc.")
    print(f"  {videos_dir}/val/P01/..., P02/..., etc.")

    if not any(videos_dir.iterdir()):
        print("\n⚠️  Videos directory is empty. Please download and extract videos.")
        return False

    return True


def download_assembly101(data_root: Path) -> bool:
    """Download Assembly101 dataset."""
    print("\n" + "="*60)
    print("Assembly101 Dataset")
    print("="*60)

    dataset_root = Path(data_root) / "assembly101"
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Download annotations
    annot_zip = dataset_root / "annotations.zip"
    annot_dir = dataset_root / "annotations"

    if not annot_dir.exists() or not any(annot_dir.iterdir()):
        print("Downloading Assembly101 annotations...")
        if download_file(DATASETS["assembly101"]["annotations_url"], annot_zip, "Annotations"):
            if unzip_file(annot_zip, dataset_root):
                annot_zip.unlink(missing_ok=True)
                print("Annotations downloaded successfully!")
            else:
                return False
        else:
            return False
    else:
        print("Annotations already exist, skipping...")

    # Download videos
    videos_zip = dataset_root / "videos.zip"
    videos_dir = dataset_root / "videos"

    if not videos_dir.exists() or not any(videos_dir.iterdir()):
        print("Downloading Assembly101 videos (this may take a while)...")
        if download_file(DATASETS["assembly101"]["videos_url"], videos_zip, "Videos"):
            if unzip_file(videos_zip, dataset_root):
                videos_zip.unlink(missing_ok=True)
                print("Videos downloaded successfully!")
            else:
                return False
        else:
            return False
    else:
        print("Videos already exist, skipping...")

    return True


def download_holoassist(data_root: Path) -> bool:
    """Download HoloAssist dataset."""
    print("\n" + "="*60)
    print("HoloAssist Dataset")
    print("="*60)

    dataset_root = Path(data_root) / "holoassist"
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Download annotations
    annot_zip = dataset_root / "annotations.zip"
    annot_dir = dataset_root / "annotations"

    if not annot_dir.exists() or not any(annot_dir.iterdir()):
        print("Downloading HoloAssist annotations...")
        if download_file(DATASETS["holoassist"]["annotations_url"], annot_zip, "Annotations"):
            if unzip_file(annot_zip, dataset_root):
                annot_zip.unlink(missing_ok=True)
                print("Annotations downloaded successfully!")
            else:
                return False
        else:
            return False
    else:
        print("Annotations already exist, skipping...")

    # Download videos
    videos_zip = dataset_root / "videos.zip"
    videos_dir = dataset_root / "videos"

    if not videos_dir.exists() or not any(videos_dir.iterdir()):
        print("Downloading HoloAssist videos...")
        if download_file(DATASETS["holoassist"]["videos_url"], videos_zip, "Videos"):
            if unzip_file(videos_zip, dataset_root):
                videos_zip.unlink(missing_ok=True)
                print("Videos downloaded successfully!")
            else:
                return False
        else:
            return False
    else:
        print("Videos already exist, skipping...")

    return True


def verify_dataset_structure(data_root: Path, dataset_name: str) -> bool:
    """Verify dataset directory structure."""
    Path(data_root) / dataset_name

    required = {
        "epic_kitchens": ["annotations/EPIC_100_train.csv", "annotations/EPIC_100_val.csv", "videos/train", "videos/val"],
        "assembly101": ["annotations/train.csv", "annotations/val.csv", "videos"],
        "holoassist": ["annotations/train.csv", "annotations/val.csv", "videos"],
    }

    if dataset_name not in required:
        return True

    missing = []
    for path in required[dataset_name]:
        if not (Path(data_root) / dataset_name / path).exists():
            missing.append(path)

    if missing:
        print(f"⚠️  Missing paths in {dataset_name}: {missing}")
        return False

    print(f"✅ {dataset_name} structure verified!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download datasets for EgoIndustrial")
    parser.add_argument("--data-root", default="/data", help="Root data directory")
    parser.add_argument("--datasets", nargs="+", default=["epic_kitchens", "assembly101", "holoassist"],
                        choices=["epic_kitchens", "assembly101", "holoassist", "all"],
                        help="Datasets to download")
    parser.add_argument("--annotations-only", action="store_true", help="Download only annotations")
    parser.add_argument("--verify", action="store_true", help="Verify dataset structure")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["epic_kitchens", "assembly101", "holoassist"]

    success = True

    if args.verify:
        for ds in datasets:
            verify_dataset_structure(args.data_root, ds)
        return

    for ds in datasets:
        if ds == "epic_kitchens":
            success &= download_epic_kitchens(args.data_root, args.annotations_only)
        elif ds == "assembly101":
            success &= download_assembly101(args.data_root)
        elif ds == "holoassist":
            success &= download_holoassist(args.data_root)

    if args.verify:
        for ds in datasets:
            verify_dataset_structure(args.data_root, ds)

    if success:
        print("\n" + "="*60)
        print("✅ All datasets downloaded successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Some downloads failed. Check errors above.")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
