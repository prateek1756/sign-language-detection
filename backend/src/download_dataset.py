"""
download_dataset.py — Phase 2: Public ASL Dataset Downloader
=============================================================
Skills Applied:
  - ml-pipeline-workflow : Idempotent download stage with verification
  - python-pro           : Typed, clean, resumable download with progress
  - data-scientist       : Dataset validation and integrity checks

Downloads the public ASL Alphabet dataset from Kaggle (via kaggle CLI)
and organizes it into our data/raw/ASL/<class>/ structure.

Usage:
    python src/download_dataset.py
    python src/download_dataset.py --source kaggle
    python src/download_dataset.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_dataset")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "ASL"
DOWNLOAD_DIR = BASE_DIR / "data" / "downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

ASL_CLASSES: list[str] = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "space", "del", "nothing",
]

# Kaggle dataset reference
KAGGLE_DATASET = "grassknoted/asl-alphabet"
"""
Public ASgle ASL Alphabet dataset on Kaggle:
  - 87,000 images (200x200 pixels)
  - 29 classes: A-Z + space, del, nothing
  - ~1GB compressed
  https://www.kaggle.com/datasets/grassknoted/asl-alphabet
"""


# ── Kaggle Download ───────────────────────────────────────────────────────────
def check_kaggle_cli() -> bool:
    """Check if kaggle CLI is installed and configured."""
    result = shutil.which("kaggle")
    if not result:
        log.error(
            "Kaggle CLI not found. Install it with:\n"
            "  pip install kaggle\n"
            "  Then set up ~/.kaggle/kaggle.json with your API key.\n"
            "  Get your API key at: https://www.kaggle.com/account"
        )
        return False
    log.info("✅ Kaggle CLI found at: %s", result)
    return True


def download_kaggle_dataset() -> Path:
    """
    Download ASL Alphabet dataset from Kaggle.

    ml-pipeline-workflow pattern:
      - Idempotent: skips download if zip already exists
      - Shows download progress
      - Returns path to downloaded zip

    Returns:
        Path to the downloaded zip file.
    """
    zip_path = DOWNLOAD_DIR / "asl-alphabet.zip"

    if zip_path.exists():
        log.info("Dataset already downloaded: %s. Skipping.", zip_path)
        return zip_path

    log.info("Downloading ASL Alphabet dataset from Kaggle...")
    log.info("Dataset: %s (~1GB)", KAGGLE_DATASET)

    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "--dataset", KAGGLE_DATASET,
            "--path", str(DOWNLOAD_DIR),
        ],
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed (exit code {result.returncode}).\n"
            "Ensure kaggle.json is configured correctly."
        )

    log.info("✅ Download complete: %s", zip_path)
    return zip_path


def extract_and_organize(zip_path: Path) -> None:
    """
    Extract the Kaggle ASL dataset and organize into our directory structure.

    The Kaggle dataset has structure:
      asl_alphabet_train/asl_alphabet_train/<class>/<image>.jpg

    We reorganize to:
      data/raw/ASL/<class>/<image>.jpg

    ml-pipeline-workflow idempotent pattern:
      - Skips classes that already have images
      - Reports progress per class
    """
    log.info("Extracting dataset from: %s", zip_path)
    extract_dir = DOWNLOAD_DIR / "extracted"
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        total = len(zf.namelist())
        log.info("Extracting %d files...", total)
        zf.extractall(str(extract_dir))

    log.info("✅ Extraction complete. Organizing into ASL class structure...")

    # Find the train directory (may vary by dataset version)
    possible_roots = [
        extract_dir / "asl_alphabet_train" / "asl_alphabet_train",
        extract_dir / "asl_alphabet_train",
        extract_dir,
    ]
    train_root = next((p for p in possible_roots if p.is_dir()), None)

    if train_root is None:
        log.error("Could not find train directory in extracted files.")
        log.error("Contents: %s", list(extract_dir.iterdir()))
        return

    # Organize files
    moved_total = 0
    for class_label in ASL_CLASSES:
        src_dir = train_root / class_label
        dst_dir = RAW_DIR / class_label
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            # Try case-insensitive match
            matches = [d for d in train_root.iterdir() if d.name.lower() == class_label.lower()]
            src_dir = matches[0] if matches else None

        if src_dir is None or not src_dir.exists():
            log.warning("Source dir not found for class '%s'. Skipping.", class_label)
            continue

        images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
        existing = len(list(dst_dir.glob("*.jpg")))

        if existing >= len(images):
            log.info("  %-10s already has %d images. Skipping.", class_label, existing)
            continue

        for img_path in images:
            shutil.copy2(str(img_path), str(dst_dir / img_path.name))
            moved_total += 1

        log.info("  %-10s → %d images copied", class_label, len(images))

    log.info("✅ Organization complete. Total files moved: %d", moved_total)


def verify_dataset(min_per_class: int = 100) -> bool:
    """
    Verify that the dataset has sufficient images per class.

    data-scientist pattern:
      - Always validate data before training
      - Report per-class counts with visual bar chart

    Args:
        min_per_class: Minimum acceptable images per class.

    Returns:
        True if all classes meet minimum requirement.
    """
    print("\n" + "═" * 60)
    print("  ASL Raw Dataset Verification")
    print("═" * 60)

    all_ok = True
    for label in ASL_CLASSES:
        images = list((RAW_DIR / label).glob("*.jpg"))
        count = len(images)
        ok = count >= min_per_class
        if not ok:
            all_ok = False
        bar = "█" * min(int(count / 3000 * 20), 20)
        status = "✅" if ok else "❌"
        print(f"  {status} {label:<8} {bar:<20} {count:>6} images")

    print("═" * 60)
    if all_ok:
        print("  ✅ All classes have sufficient images!")
    else:
        print(f"  ❌ Some classes have fewer than {min_per_class} images.")
    print("═" * 60 + "\n")
    return all_ok


# ── Manual Download Instructions ──────────────────────────────────────────────
def print_manual_instructions() -> None:
    """Print manual download instructions if Kaggle CLI is unavailable."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Manual Dataset Download Instructions               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Option 1: Kaggle CLI (Recommended)                          ║
║  ─────────────────────────────────                           ║
║  1. pip install kaggle                                       ║
║  2. Get API key from https://www.kaggle.com/account         ║
║  3. Save kaggle.json to ~/.kaggle/kaggle.json               ║
║  4. Run: python src/download_dataset.py                      ║
║                                                              ║
║  Option 2: Direct Download                                   ║
║  ─────────────────────────                                   ║
║  URL: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
║  1. Download asl-alphabet.zip                                ║
║  2. Extract to: data/downloads/extracted/                    ║
║  3. Run: python src/download_dataset.py --organize_only      ║
║                                                              ║
║  Option 3: Collect Your Own Data                             ║
║  ─────────────────────────────                               ║
║  Run: python src/collect_data.py --all --samples 300        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and organize public ASL Alphabet dataset",
    )
    parser.add_argument("--source", choices=["kaggle"], default="kaggle",
                        help="Dataset source (default: kaggle)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing dataset and exit")
    parser.add_argument("--organize_only", action="store_true",
                        help="Skip download, only organize already-extracted files")
    parser.add_argument("--min_per_class", type=int, default=100,
                        help="Minimum images per class for verification (default: 100)")
    parser.add_argument("--instructions", action="store_true",
                        help="Print manual download instructions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.instructions:
        print_manual_instructions()
        return

    if args.verify:
        verify_dataset(min_per_class=args.min_per_class)
        return

    if args.organize_only:
        zip_path = DOWNLOAD_DIR / "asl-alphabet.zip"
        if not zip_path.exists():
            log.error("No zip found at %s. Download first.", zip_path)
            return
        extract_and_organize(zip_path)
        verify_dataset(min_per_class=args.min_per_class)
        return

    # Full pipeline: download + extract + organize + verify
    if not check_kaggle_cli():
        print_manual_instructions()
        return

    zip_path = download_kaggle_dataset()
    extract_and_organize(zip_path)
    verify_dataset(min_per_class=args.min_per_class)


if __name__ == "__main__":
    main()
