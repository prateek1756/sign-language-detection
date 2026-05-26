"""
preprocess.py — Phase 2: Landmark Extraction & Preprocessing Pipeline
======================================================================
Skills Applied:
  - computer-vision-expert : MediaPipe landmark extraction, image crop pipeline
  - ml-pipeline-workflow   : Modular, idempotent preprocessing stages
  - data-scientist         : Normalization theory, augmentation strategies
  - ml-engineer            : NumPy-efficient processing, tf.data-ready output
  - python-pro             : Typed, documented, production-grade Python

Pipeline:
  Raw Image → MediaPipe Hands → 21 Landmarks
    → Normalize (wrist anchor + bounding box scale)
    → Augment (flip, rotate, brightness)
    → Save as .npy landmark arrays + image crops

Usage:
    python src/preprocess.py --class_label A
    python src/preprocess.py --all
    python src/preprocess.py --all --augment --aug_factor 3
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_LANDMARKS = 21
"""MediaPipe Hands produces 21 landmarks per hand."""

LANDMARK_DIM = 3
"""Each landmark has x, y, z coordinates."""

FEATURE_DIM = NUM_LANDMARKS * LANDMARK_DIM  # 63 features per sample

ASL_CLASSES: list[str] = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "space", "del", "nothing",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "ASL"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "ASL"
CROPS_DIR = PROCESSED_DIR / "crops"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""

    class_label: Optional[str] = None
    """Target class. None = process all classes."""

    augment: bool = True
    """Whether to apply data augmentation."""

    aug_factor: int = 3
    """How many augmented copies to generate per original sample."""

    crop_size: int = 224
    """Output crop size for CNN model input (MobileNetV3)."""

    min_detection_confidence: float = 0.5
    """MediaPipe detection confidence (lower = more permissive for offline processing)."""

    padding_ratio: float = 0.2
    """Padding around hand bounding box for image crops (fraction of bbox size)."""

    output_landmarks: bool = True
    """Save normalized landmark arrays as .npy."""

    output_crops: bool = True
    """Save image crops for CNN model."""


# ── MediaPipe ─────────────────────────────────────────────────────────────────
def build_static_hands() -> mp.solutions.hands.Hands:
    """
    Build MediaPipe Hands in static image mode for offline preprocessing.

    computer-vision-expert note:
      static_image_mode=True treats each image independently (no tracking),
      which is correct for offline batch processing.
    """
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


# ── Landmark Extraction ───────────────────────────────────────────────────────
def extract_landmarks(
    image_bgr: np.ndarray,
    hands: mp.solutions.hands.Hands,
) -> Optional[np.ndarray]:
    """
    Extract 21 hand landmarks from an image.

    Args:
        image_bgr: BGR image from OpenCV.
        hands:     MediaPipe Hands processor.

    Returns:
        Flat float32 array of shape (63,) [x,y,z × 21 landmarks],
        or None if no hand is detected.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_lm = results.multi_hand_landmarks[0]
    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
        dtype=np.float32,
    )  # shape: (21, 3)

    return landmarks.flatten()  # shape: (63,)


# ── Normalization ─────────────────────────────────────────────────────────────
def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    """
    Normalize raw landmark coordinates for translation + scale invariance.

    Method (data-scientist pattern — makes model location/size invariant):
      1. Translation invariance: subtract wrist position (landmark 0)
         → hand can be anywhere in frame
      2. Scale invariance: divide by bounding box diagonal
         → hand can be any size / distance from camera

    Args:
        raw: Flat array of shape (63,) = 21 landmarks × (x, y, z).

    Returns:
        Normalized flat array of shape (63,).
    """
    landmarks = raw.reshape(NUM_LANDMARKS, LANDMARK_DIM)  # (21, 3)

    # Step 1: Translation — anchor to wrist (landmark 0)
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist  # all relative to wrist

    # Step 2: Scale — normalize by bounding box diagonal (xy only)
    xy = landmarks[:, :2]  # (21, 2)
    bbox_min = xy.min(axis=0)
    bbox_max = xy.max(axis=0)
    bbox_size = bbox_max - bbox_min
    diagonal = float(np.linalg.norm(bbox_size))

    if diagonal < 1e-6:
        # Degenerate case: all landmarks at same point
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    landmarks = landmarks / diagonal

    return landmarks.flatten().astype(np.float32)


# ── Image Crop Extraction ─────────────────────────────────────────────────────
def extract_hand_crop(
    image_bgr: np.ndarray,
    hands_result: mp.solutions.hands.Hands,
    crop_size: int = 224,
    padding_ratio: float = 0.2,
) -> Optional[np.ndarray]:
    """
    Crop the hand region from an image with padding.

    Used for CNN (MobileNetV3) secondary model input.
    (computer-vision-expert: bounding box + padding crop pattern)

    Args:
        image_bgr:     BGR source image.
        hands_result:  MediaPipe process() result.
        crop_size:     Output square size (224 for MobileNetV3).
        padding_ratio: Relative padding around the bounding box.

    Returns:
        Cropped, resized BGR image of shape (crop_size, crop_size, 3)
        or None if no hand detected.
    """
    if not hands_result.multi_hand_landmarks:
        return None

    h, w = image_bgr.shape[:2]
    hand_lm = hands_result.multi_hand_landmarks[0]

    xs = [lm.x * w for lm in hand_lm.landmark]
    ys = [lm.y * h for lm in hand_lm.landmark]

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    # Add padding
    pad_x = int((x_max - x_min) * padding_ratio)
    pad_y = int((y_max - y_min) * padding_ratio)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)


# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_image(image: np.ndarray) -> list[np.ndarray]:
    """
    Generate augmented versions of an image.

    Augmentation strategy (data-scientist + ml-engineer pattern):
      - Horizontal flip (mirror gesture — increases variation)
      - Random rotation ±15° (angle invariance)
      - Brightness/contrast jitter (lighting invariance)
      - Gaussian noise (robustness to camera noise)

    Args:
        image: BGR image.

    Returns:
        List of augmented BGR images.
    """
    augmented: list[np.ndarray] = []
    h, w = image.shape[:2]

    # 1. Horizontal flip
    augmented.append(cv2.flip(image, 1))

    # 2. Random rotation ±15°
    angle = random.uniform(-15.0, 15.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    augmented.append(rotated)

    # 3. Brightness jitter (alpha=contrast, beta=brightness)
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    bright = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    augmented.append(bright)

    # 4. Gaussian noise
    noise = np.random.randn(*image.shape).astype(np.float32) * 8.0
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy)

    return augmented


def augment_landmarks(landmarks: np.ndarray) -> list[np.ndarray]:
    """
    Augment normalized landmarks with small perturbations.

    Landmark augmentation strategy (ml-engineer pattern):
      - Mirror flip: negate x-coordinates
      - Small random noise (simulates tracking jitter)
      - Small scale variation

    Args:
        landmarks: Normalized flat array of shape (63,).

    Returns:
        List of augmented landmark arrays.
    """
    augmented: list[np.ndarray] = []
    lm = landmarks.reshape(NUM_LANDMARKS, LANDMARK_DIM)

    # 1. Mirror flip (negate x)
    flipped = lm.copy()
    flipped[:, 0] = -flipped[:, 0]
    augmented.append(flipped.flatten())

    # 2. Small Gaussian jitter
    jitter = lm + np.random.randn(*lm.shape).astype(np.float32) * 0.02
    augmented.append(jitter.flatten())

    # 3. Small scale variation
    scale = random.uniform(0.85, 1.15)
    scaled = lm * scale
    augmented.append(scaled.flatten())

    return augmented


# ── Per-Class Processing ──────────────────────────────────────────────────────
def process_class(
    class_label: str,
    cfg: PreprocessConfig,
    hands: mp.solutions.hands.Hands,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Process all images for a single ASL class.

    ml-pipeline-workflow pattern:
      Each stage is independently runnable and reports counts.

    Args:
        class_label: ASL class name.
        cfg:         PreprocessConfig.
        hands:       MediaPipe Hands processor.

    Returns:
        Tuple of (landmark_vectors, labels) lists.
    """
    class_index = ASL_CLASSES.index(class_label)
    raw_images = sorted((RAW_DIR / class_label).glob("*.jpg"))

    if not raw_images:
        log.warning("No raw images found for class '%s'. Skipping.", class_label)
        return [], []

    landmark_vectors: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    for img_path in tqdm(raw_images, desc=f"  {class_label}", leave=False):
        image = cv2.imread(str(img_path))
        if image is None:
            skipped += 1
            continue

        # Extract raw landmarks
        raw_lm = extract_landmarks(image, hands)
        if raw_lm is None:
            skipped += 1
            continue

        # Normalize
        norm_lm = normalize_landmarks(raw_lm)
        landmark_vectors.append(norm_lm)
        labels.append(class_index)

        # Save image crop for CNN model
        if cfg.output_crops:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=1
            ).process(rgb)
            crop = extract_hand_crop(image, results, cfg.crop_size, cfg.padding_ratio)
            if crop is not None:
                crop_dir = CROPS_DIR / class_label
                crop_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(crop_dir / img_path.name), crop)

        # Augmentation
        if cfg.augment:
            aug_lms = augment_landmarks(norm_lm)
            for aug in aug_lms[: cfg.aug_factor]:
                landmark_vectors.append(aug)
                labels.append(class_index)

    log.info(
        "  %-8s → %4d samples (augmented: %4d) | skipped: %d",
        class_label,
        len(raw_images) - skipped,
        len(landmark_vectors),
        skipped,
    )
    return landmark_vectors, labels


# ── Full Pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(cfg: PreprocessConfig) -> dict[str, int]:
    """
    Run the full preprocessing pipeline.

    ml-pipeline-workflow stages:
      1. Load raw images
      2. Extract MediaPipe landmarks
      3. Normalize landmarks
      4. Augment data
      5. Save processed arrays

    Args:
        cfg: PreprocessConfig.

    Returns:
        Dict with stats: total samples per split.
    """
    classes_to_process = (
        [cfg.class_label] if cfg.class_label else ASL_CLASSES
    )

    all_landmarks: list[np.ndarray] = []
    all_labels: list[int] = []

    hands = build_static_hands()
    log.info("═" * 60)
    log.info("  Starting preprocessing pipeline for %d class(es)", len(classes_to_process))
    log.info("  Augmentation: %s (×%d per sample)", cfg.augment, cfg.aug_factor)
    log.info("═" * 60)

    for label in classes_to_process:
        lms, lbls = process_class(label, cfg, hands)
        all_landmarks.extend(lms)
        all_labels.extend(lbls)

    hands.close()

    if not all_landmarks:
        log.warning("No data was processed. Check that raw images exist.")
        return {}

    X = np.array(all_landmarks, dtype=np.float32)  # (N, 63)
    y = np.array(all_labels, dtype=np.int32)        # (N,)

    # Save arrays
    out_X = PROCESSED_DIR / "landmarks_all.npy"
    out_y = PROCESSED_DIR / "labels_all.npy"
    np.save(str(out_X), X)
    np.save(str(out_y), y)

    # Save class mapping
    class_map = {i: label for i, label in enumerate(ASL_CLASSES)}
    import json
    with open(PROCESSED_DIR / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    log.info("═" * 60)
    log.info("  ✅ Saved X: %s  shape=%s", out_X.name, X.shape)
    log.info("  ✅ Saved y: %s  shape=%s", out_y.name, y.shape)
    log.info("  ✅ Class map: class_map.json")
    log.info("  Feature dim: %d | Classes: %d | Total samples: %d",
             FEATURE_DIM, len(set(all_labels)), len(all_labels))
    log.info("═" * 60)

    return {"total_samples": len(all_labels), "feature_dim": FEATURE_DIM}


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASL Landmark Extraction & Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/preprocess.py --all
  python src/preprocess.py --class_label A
  python src/preprocess.py --all --augment --aug_factor 3
  python src/preprocess.py --all --no_crops
        """,
    )
    parser.add_argument("--class_label", type=str, default=None, help="Process single class (default: all)")
    parser.add_argument("--all", action="store_true", help="Process all 29 ASL classes")
    parser.add_argument("--augment", action="store_true", default=True, help="Enable data augmentation (default: on)")
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--aug_factor", type=int, default=3, help="Augmentation copies per sample (default: 3)")
    parser.add_argument("--no_crops", action="store_true", help="Skip saving image crops for CNN model")
    parser.add_argument("--crop_size", type=int, default=224, help="Image crop size in pixels (default: 224)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(
        class_label=args.class_label,
        augment=not args.no_augment,
        aug_factor=args.aug_factor,
        output_crops=not args.no_crops,
        crop_size=args.crop_size,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
