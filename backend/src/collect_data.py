"""
collect_data.py — Phase 2: ASL Data Collection Pipeline
========================================================
Skills Applied:
  - computer-vision-expert : OpenCV webcam + MediaPipe Hands real-time pipeline
  - python-pro             : Type hints, dataclasses, structured logging, clean patterns
  - ml-pipeline-workflow   : Idempotent collection stages, structured data layout

Usage:
    python src/collect_data.py --class_label A --samples 300
    python src/collect_data.py --class_label A --samples 300 --show_landmarks
    python src/collect_data.py --list_classes
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect_data")

# ── ASL Class Registry ────────────────────────────────────────────────────────
ASL_CLASSES: list[str] = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "space",
    "del",
    "nothing",
]
"""All 29 ASL gesture classes: A–Z alphabet + space, del, nothing."""

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "ASL"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed" / "ASL"


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class CollectionConfig:
    """Configuration for a data collection session."""

    class_label: str
    """Target ASL gesture class (e.g., 'A', 'space')."""

    num_samples: int = 300
    """Number of frames to capture per class."""

    camera_index: int = 0
    """Webcam device index."""

    frame_width: int = 640
    """Capture resolution width."""

    frame_height: int = 480
    """Capture resolution height."""

    show_landmarks: bool = True
    """Overlay MediaPipe hand landmarks on live preview."""

    countdown_seconds: int = 3
    """Countdown before capture begins."""

    capture_delay_ms: int = 50
    """Delay between captured frames in ms (controls capture FPS)."""

    min_detection_confidence: float = 0.7
    """MediaPipe minimum hand detection confidence threshold."""

    min_tracking_confidence: float = 0.5
    """MediaPipe minimum hand tracking confidence threshold."""

    save_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        if self.class_label not in ASL_CLASSES:
            raise ValueError(
                f"Unknown class '{self.class_label}'. "
                f"Valid classes: {ASL_CLASSES}"
            )
        self.save_dir = RAW_DATA_DIR / self.class_label
        self.save_dir.mkdir(parents=True, exist_ok=True)


# ── MediaPipe Setup ───────────────────────────────────────────────────────────
def build_hands_detector(cfg: CollectionConfig) -> mp.solutions.hands.Hands:
    """
    Build a MediaPipe Hands detector.

    Uses computer-vision-expert pattern:
      - High confidence thresholds for clean training data
      - max_num_hands=1 for single-hand ASL gestures
    """
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )


# ── Drawing Utilities ─────────────────────────────────────────────────────────
def draw_landmarks_on_frame(
    frame: np.ndarray,
    hand_landmarks: mp.solutions.hands.HandLandmark,
) -> np.ndarray:
    """
    Draw 21 MediaPipe hand landmarks and connections on a BGR frame.

    Applies computer-vision-expert real-time overlay pattern:
      - Landmark dots in green
      - Connection lines in white
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    annotated = frame.copy()
    mp_drawing.draw_landmarks(
        annotated,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
    return annotated


def draw_hud(
    frame: np.ndarray,
    label: str,
    captured: int,
    total: int,
    hand_detected: bool,
    collecting: bool,
) -> np.ndarray:
    """Render heads-up display on the preview frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Class label
    cv2.putText(
        frame,
        f"Class: {label}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Progress bar
    progress = captured / total if total > 0 else 0
    bar_x, bar_y, bar_w, bar_h = 10, 65, w - 20, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + int(bar_w * progress), bar_y + bar_h),
        (0, 220, 100) if collecting else (100, 100, 100),
        -1,
    )
    cv2.putText(
        frame,
        f"{captured}/{total}",
        (bar_x + bar_w + 5, bar_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Hand detection status
    status_color = (0, 220, 100) if hand_detected else (0, 80, 220)
    status_text = "HAND ✓" if hand_detected else "NO HAND"
    cv2.putText(
        frame,
        status_text,
        (w - 160, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # Collecting indicator
    if collecting:
        cv2.circle(frame, (w - 20, 70), 8, (0, 0, 255), -1)  # Red dot = recording

    return frame


# ── Countdown ─────────────────────────────────────────────────────────────────
def run_countdown(cap: cv2.VideoCapture, seconds: int, label: str) -> None:
    """Display live countdown before capture starts."""
    start = time.time()
    while True:
        elapsed = time.time() - start
        remaining = seconds - int(elapsed)
        if remaining <= 0:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        cv2.putText(
            frame,
            f"Get ready: {label}",
            (w // 2 - 160, h // 2 - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(remaining),
            (w // 2 - 30, h // 2 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (0, 220, 100),
            4,
            cv2.LINE_AA,
        )
        cv2.imshow("ASL Data Collection", frame)
        cv2.waitKey(1)


# ── Core Collection Loop ──────────────────────────────────────────────────────
def collect_samples(cfg: CollectionConfig) -> int:
    """
    Run the interactive webcam capture loop.

    Pipeline (computer-vision-expert pattern):
      Frame → Flip → RGB convert → MediaPipe detect
        → Draw landmarks (optional) → HUD overlay → Save on key press

    Args:
        cfg: CollectionConfig instance.

    Returns:
        Number of samples actually captured.
    """
    # Count existing samples to support resuming (idempotent — ml-pipeline-workflow)
    existing = list(cfg.save_dir.glob("*.jpg"))
    start_index = len(existing)
    if start_index >= cfg.num_samples:
        log.info(
            "Class '%s' already has %d/%d samples. Skipping.",
            cfg.class_label,
            start_index,
            cfg.num_samples,
        )
        return start_index

    log.info(
        "Starting collection for class '%s'. Need %d more samples (have %d).",
        cfg.class_label,
        cfg.num_samples - start_index,
        start_index,
    )

    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {cfg.camera_index}.")

    hands = build_hands_detector(cfg)
    captured_count = start_index
    collecting = False

    # Countdown
    run_countdown(cap, cfg.countdown_seconds, cfg.class_label)
    collecting = True
    log.info("Capture started. Press 'q' to quit early, 's' to pause/resume.")

    try:
        while captured_count < cfg.num_samples:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read frame from camera.")
                break

            # Mirror flip (computer-vision-expert: natural for selfie-style)
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe inference
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            hand_detected = results.multi_hand_landmarks is not None
            display_frame = frame.copy()

            # Draw landmarks if detected
            if hand_detected and cfg.show_landmarks:
                for hand_lm in results.multi_hand_landmarks:
                    display_frame = draw_landmarks_on_frame(display_frame, hand_lm)

            # HUD overlay
            display_frame = draw_hud(
                display_frame,
                cfg.class_label,
                captured_count,
                cfg.num_samples,
                hand_detected,
                collecting,
            )

            cv2.imshow("ASL Data Collection", display_frame)

            # Save frame when collecting and hand is detected
            if collecting and hand_detected:
                filename = cfg.save_dir / f"{cfg.class_label}_{captured_count:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                captured_count += 1
                cv2.waitKey(cfg.capture_delay_ms)
            else:
                key = cv2.waitKey(1) & 0xFF

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("Quit requested by user.")
                break
            elif key == ord("s"):
                collecting = not collecting
                log.info("Capture %s.", "resumed" if collecting else "paused")

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    log.info(
        "Collection complete for '%s': %d/%d samples saved to %s",
        cfg.class_label,
        captured_count,
        cfg.num_samples,
        cfg.save_dir,
    )
    return captured_count


# ── Batch Collection ──────────────────────────────────────────────────────────
def collect_all_classes(num_samples: int = 300) -> dict[str, int]:
    """
    Collect samples for all 29 ASL classes sequentially.

    Implements ml-pipeline-workflow idempotent pattern:
      - Skips classes that already have enough samples
      - Resumes from where it left off

    Args:
        num_samples: Target samples per class.

    Returns:
        Dict mapping class label → number of samples collected.
    """
    results: dict[str, int] = {}
    for label in ASL_CLASSES:
        log.info("═" * 50)
        log.info("Class %d/%d: %s", ASL_CLASSES.index(label) + 1, len(ASL_CLASSES), label)
        cfg = CollectionConfig(class_label=label, num_samples=num_samples)
        results[label] = collect_samples(cfg)
    return results


# ── Dataset Status Report ─────────────────────────────────────────────────────
def print_dataset_status(target: int = 300) -> None:
    """
    Print a summary table of how many samples exist per class.
    (data-scientist pattern: always inspect your data before training)
    """
    print("\n" + "═" * 55)
    print(f"  ASL Dataset Status  (target: {target} samples/class)")
    print("═" * 55)
    total = 0
    for label in ASL_CLASSES:
        count = len(list((RAW_DATA_DIR / label).glob("*.jpg")))
        total += count
        bar = "█" * min(int(count / target * 20), 20)
        status = "✅" if count >= target else "⚠️ "
        print(f"  {status} {label:<8} {bar:<20} {count:>4}/{target}")
    print("═" * 55)
    print(f"  Total samples: {total}/{target * len(ASL_CLASSES)}")
    print("═" * 55 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASL Sign Language Data Collection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/collect_data.py --class_label A --samples 300
  python src/collect_data.py --class_label space --samples 200 --no_landmarks
  python src/collect_data.py --all --samples 300
  python src/collect_data.py --status
  python src/collect_data.py --list_classes
        """,
    )
    parser.add_argument("--class_label", type=str, help="ASL class to collect (e.g., A, B, space)")
    parser.add_argument("--samples", type=int, default=300, help="Number of samples to capture (default: 300)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--no_landmarks", action="store_true", help="Disable landmark overlay on preview")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before capture (default: 3)")
    parser.add_argument("--all", action="store_true", help="Collect samples for ALL 29 classes sequentially")
    parser.add_argument("--status", action="store_true", help="Show dataset collection status and exit")
    parser.add_argument("--list_classes", action="store_true", help="List all valid class names and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_classes:
        print("Valid ASL classes:", ", ".join(ASL_CLASSES))
        return

    if args.status:
        print_dataset_status(target=args.samples)
        return

    if args.all:
        results = collect_all_classes(num_samples=args.samples)
        print_dataset_status(target=args.samples)
        return

    if not args.class_label:
        print("Error: Provide --class_label or use --all / --status / --list_classes")
        return

    cfg = CollectionConfig(
        class_label=args.class_label.upper() if args.class_label not in ("space", "del", "nothing") else args.class_label,
        num_samples=args.samples,
        camera_index=args.camera,
        show_landmarks=not args.no_landmarks,
        countdown_seconds=args.countdown,
    )
    collect_samples(cfg)
    print_dataset_status(target=args.samples)


if __name__ == "__main__":
    main()
