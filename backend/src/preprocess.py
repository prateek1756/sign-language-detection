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

# ── MediaPipe API compatibility ───────────────────────────────────────────────
# mediapipe >= 0.10.18 removed mp.solutions in favour of mp.tasks.
# We detect which API is available and provide a unified Hands wrapper.
_MP_VERSION = tuple(int(x) for x in mp.__version__.split(".")[:3])
_USE_LEGACY_API = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

if not _USE_LEGACY_API:
    # New Tasks API (mediapipe >= 0.10.18)
    from mediapipe.tasks import python as _mp_tasks
    from mediapipe.tasks.python import vision as _mp_vision
    import urllib.request as _urllib_request
    import tempfile as _tempfile
    import os as _os

    # Download the hand landmarker model if not cached
    _MODEL_PATH = _os.path.join(_tempfile.gettempdir(), "hand_landmarker.task")
    if not _os.path.exists(_MODEL_PATH):
        _MODEL_URL = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        _urllib_request.urlretrieve(_MODEL_URL, _MODEL_PATH)

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

    align_rotation: bool = False
    """Rotate landmarks relative to the palm plane (rotation invariant)."""

    get_geometric: bool = False
    """Compute and append 14 scale/rotation invariant geometric features (results in 77 features)."""



# ── MediaPipe ─────────────────────────────────────────────────────────────────
def build_static_hands():
    """
    Build a MediaPipe Hands processor compatible with both old and new APIs.

    - mediapipe < 0.10.18  : uses mp.solutions.hands.Hands (legacy)
    - mediapipe >= 0.10.18 : uses mp.tasks HandLandmarker (new Tasks API)

    Returns an object with a .process(rgb_image) method that returns
    a result with .multi_hand_landmarks compatible with the rest of the code.
    """
    if _USE_LEGACY_API:
        return mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        # New Tasks API wrapper that mimics the legacy interface
        options = _mp_vision.HandLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = _mp_vision.HandLandmarker.create_from_options(options)

        class _LandmarkResult:
            """Mimics mp.solutions.hands result for backward compatibility."""
            def __init__(self, landmarks_list):
                self.multi_hand_landmarks = landmarks_list

        class _LandmarkPoint:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        class _HandGroup:
            def __init__(self, landmarks):
                self.landmark = landmarks

        class _HandsWrapper:
            def __init__(self, lm):
                self._lm = lm

            def process(self, rgb_image):
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_image,
                )
                result = self._lm.detect(mp_image)
                if not result.hand_landmarks:
                    return _LandmarkResult(None)
                # Convert NormalizedLandmark list to legacy format
                hand_group = _HandGroup([
                    _LandmarkPoint(lm.x, lm.y, lm.z)
                    for lm in result.hand_landmarks[0]
                ])
                return _LandmarkResult([hand_group])

            def close(self):
                self._lm.close()

        return _HandsWrapper(landmarker)


# ── Landmark Extraction ───────────────────────────────────────────────────────
def extract_landmarks(
    image_bgr: np.ndarray,
    hands,
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
def align_palm_plane(landmarks: np.ndarray) -> np.ndarray:
    """
    Rotate 3D landmarks relative to the palm plane to achieve rotation invariance.
    Origin is wrist (0), Index MCP (5) defines Y-axis,
    Pinky MCP (17) defines the plane.
    landmarks shape: (21, 3), already translated relative to wrist.
    """
    p5 = landmarks[5]
    p17 = landmarks[17]

    # Define Y-axis (from wrist to index MCP)
    norm_p5 = np.linalg.norm(p5)
    y_axis = p5 / (norm_p5 + 1e-8)

    # Define temp vector for Pinky MCP
    norm_p17 = np.linalg.norm(p17)
    v_tmp = p17 / (norm_p17 + 1e-8)

    # Z-axis (palm normal) is perpendicular to Y-axis and v_tmp
    z_raw = np.cross(y_axis, v_tmp)
    z_norm = np.linalg.norm(z_raw)
    z_axis = z_raw / (z_norm + 1e-8)

    # X-axis (transverse) is perpendicular to Y and Z
    x_raw = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_raw)
    x_axis = x_raw / (x_norm + 1e-8)

    # Rotation matrix R with columns [x_axis, y_axis, z_axis]
    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape: (3, 3)

    # Rotate landmarks: landmarks_rotated = landmarks @ R
    rotated = landmarks @ R
    return rotated


def compute_geometric_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Compute 14 scale- and rotation-invariant geometric features:
      - 5 Fingertip-to-Wrist normalized distances
      - 4 Tip-to-Tip normalized distances
      - 5 Joint Bending Angles (in radians)
    landmarks shape: (21, 3), translated and normalized.
    """
    features = []
    
    # 1. Fingertip-to-wrist normalized distances
    tips = [4, 8, 12, 16, 20]
    for tip in tips:
        dist = np.linalg.norm(landmarks[tip])
        features.append(dist)
        
    # 2. Tip-to-Tip normalized distances
    for i in range(len(tips) - 1):
        dist = np.linalg.norm(landmarks[tips[i]] - landmarks[tips[i+1]])
        features.append(dist)
        
    # 3. Joint Bending Angles (5 features)
    finger_vectors = [
        (landmarks[2] - landmarks[1], landmarks[4] - landmarks[3]),  # Thumb
        (landmarks[6] - landmarks[5], landmarks[8] - landmarks[7]),  # Index
        (landmarks[10] - landmarks[9], landmarks[12] - landmarks[11]),  # Middle
        (landmarks[14] - landmarks[13], landmarks[16] - landmarks[15]),  # Ring
        (landmarks[18] - landmarks[17], landmarks[20] - landmarks[19]),  # Pinky
    ]
    
    for v1, v2 in finger_vectors:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            angle = 0.0
        else:
            cosine = np.dot(v1, v2) / (norm1 * norm2)
            cosine = np.clip(cosine, -1.0, 1.0)
            angle = np.arccos(cosine)
        features.append(angle)
        
    return np.array(features, dtype=np.float32)


def normalize_landmarks(
    raw: np.ndarray,
    width: int = 1,
    height: int = 1,
    align_rotation: bool = False,
    get_geometric: bool = False,
) -> np.ndarray:
    """
    Normalize raw landmark coordinates for translation + scale invariance.

    Method (data-scientist pattern — makes model location/size invariant):
      1. Aspect ratio correction: scale x-coordinates by width/height so
         spatial dimensions are physical and isotropic.
      2. Translation invariance: subtract wrist position (landmark 0)
         → hand can be anywhere in frame
      3. (Optional) Rotate to canonical palm plane alignment (FreiHAND style)
      4. Scale invariance: divide by bounding box diagonal (xy only)
         → hand can be any size / distance from camera

    Args:
        raw: Flat array of shape (63,) = 21 landmarks × (x, y, z).
        width: Width of the source image.
        height: Height of the source image.
        align_rotation: Whether to rotate the landmarks to a canonical palm plane orientation.
        get_geometric: Whether to compute and append geometric features.

    Returns:
        Normalized flat array of shape (63,) or (77,) if get_geometric is True.
    """
    landmarks = raw.reshape(NUM_LANDMARKS, LANDMARK_DIM)  # (21, 3)

    # Step 1: Aspect ratio correction (x_adjusted = x * width / height)
    aspect_ratio = width / height
    landmarks[:, 0] = landmarks[:, 0] * aspect_ratio

    # Step 2: Translation — anchor to wrist (landmark 0)
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist  # all relative to wrist

    # Step 3: (Optional) Rotation alignment (FreiHAND style)
    if align_rotation:
        landmarks = align_palm_plane(landmarks)

    # Step 4: Scale — normalize by bounding box diagonal (xy only)
    xy = landmarks[:, :2]  # (21, 2)
    bbox_min = xy.min(axis=0)
    bbox_max = xy.max(axis=0)
    bbox_size = bbox_max - bbox_min
    diagonal = float(np.linalg.norm(bbox_size))

    if diagonal < 1e-6:
        # Degenerate case: all landmarks at same point
        base_norm = np.zeros(FEATURE_DIM, dtype=np.float32)
    else:
        landmarks = landmarks / diagonal
        base_norm = landmarks.flatten().astype(np.float32)

    if get_geometric:
        geom = compute_geometric_features(base_norm.reshape(NUM_LANDMARKS, LANDMARK_DIM))
        return np.concatenate([base_norm, geom]).astype(np.float32)

    return base_norm




# ── Image Crop Extraction ─────────────────────────────────────────────────────
def extract_hand_crop(
    image_bgr: np.ndarray,
    hands_result,
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
    hands,
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

        # Base normalization (optionally aligned)
        h, w = image.shape[:2]
        base_norm = normalize_landmarks(
            raw_lm,
            width=w,
            height=h,
            align_rotation=cfg.align_rotation,
            get_geometric=False,
        )

        # Append geometric features if enabled
        if cfg.get_geometric:
            norm_lm = np.concatenate([
                base_norm,
                compute_geometric_features(base_norm.reshape(NUM_LANDMARKS, LANDMARK_DIM))
            ]).astype(np.float32)
        else:
            norm_lm = base_norm

        landmark_vectors.append(norm_lm)
        labels.append(class_index)

        # Save image crop for CNN model
        if cfg.output_crops:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Reuse the shared hands processor for crop extraction
            crop_result = hands.process(rgb)
            crop = extract_hand_crop(image, crop_result, cfg.crop_size, cfg.padding_ratio)
            if crop is not None:
                crop_dir = CROPS_DIR / class_label
                crop_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(crop_dir / img_path.name), crop)

        # Augmentation
        if cfg.augment:
            aug_bases = augment_landmarks(base_norm)
            for aug_base in aug_bases[: cfg.aug_factor]:
                if cfg.get_geometric:
                    aug_lm = np.concatenate([
                        aug_base,
                        compute_geometric_features(aug_base.reshape(NUM_LANDMARKS, LANDMARK_DIM))
                    ]).astype(np.float32)
                else:
                    aug_lm = aug_base
                landmark_vectors.append(aug_lm)
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
    log.info("  Rotation Alignment: %s", cfg.align_rotation)
    log.info("  Geometric Features: %s", cfg.get_geometric)
    log.info("═" * 60)

    for label in classes_to_process:
        lms, lbls = process_class(label, cfg, hands)
        all_landmarks.extend(lms)
        all_labels.extend(lbls)

    hands.close()

    if not all_landmarks:
        log.warning("No data was processed. Check that raw images exist.")
        return {}

    X = np.array(all_landmarks, dtype=np.float32)  # (N, D)
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
             X.shape[1], len(set(all_labels)), len(all_labels))
    log.info("═" * 60)

    return {"total_samples": len(all_labels), "feature_dim": X.shape[1]}


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
  python src/preprocess.py --all --align_rotation --get_geometric
        """,
    )
    parser.add_argument("--class_label", type=str, default=None, help="Process single class (default: all)")
    parser.add_argument("--all", action="store_true", help="Process all 29 ASL classes")
    parser.add_argument("--augment", action="store_true", default=True, help="Enable data augmentation (default: on)")
    parser.add_argument("--no_augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--aug_factor", type=int, default=3, help="Augmentation copies per sample (default: 3)")
    parser.add_argument("--no_crops", action="store_true", help="Skip saving image crops for CNN model")
    parser.add_argument("--crop_size", type=int, default=224, help="Image crop size in pixels (default: 224)")
    parser.add_argument("--align_rotation", action="store_true", help="Enable palm-plane rotation alignment")
    parser.add_argument("--get_geometric", action="store_true", help="Include 14 scale/rotation invariant geometric features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(
        class_label=args.class_label,
        augment=not args.no_augment,
        aug_factor=args.aug_factor,
        output_crops=not args.no_crops,
        crop_size=args.crop_size,
        align_rotation=args.align_rotation,
        get_geometric=args.get_geometric,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
