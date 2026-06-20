"""
inference.py — Phase 4: Real-Time Inference Engine
====================================================
InferenceEngine:
  - Loads MLP + LSTM + CNN models at startup (lazy, thread-safe)
  - Extracts MediaPipe landmarks from base64 frame
  - Applies confidence threshold (70%) filtering
  - Temporal smoothing via rolling majority vote (last 10 frames)
  - Mode detector: static frame → Letter | N-frame buffer → Word
  - GPU auto-detect with CPU fallback

Usage (from FastAPI):
    engine = InferenceEngine()
    result = engine.predict_frame(base64_image_str)
    result = engine.predict_sequence(list_of_base64_frames)
"""

from __future__ import annotations

import base64
import collections
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Any

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.training_config import (
    ASL_CLASSES,
    FEATURE_DIM,
    MODELS_DIR,
    NUM_CLASSES,
)
from src.preprocess import normalize_landmarks

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55   # suppress predictions below this (tuned for better responsiveness)
SMOOTHING_WINDOW     = 4      # rolling majority-vote window size (tuned from 10 to 4 to reduce lag)
SEQUENCE_LEN         = 30     # frames for LSTM word mode



# ── GPU Setup ─────────────────────────────────────────────────────────────────────────
# B2 FIX: idempotent guard — prevents double-call with model.py's configure_gpu.
# set_memory_growth raises RuntimeError if called after GPU is already initialized.
_GPU_CONFIGURED: bool = False


def _configure_gpu() -> str:
    global _GPU_CONFIGURED
    gpus = tf.config.list_physical_devices("GPU")
    device = "GPU" if gpus else "CPU"
    if not _GPU_CONFIGURED:
        _GPU_CONFIGURED = True
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    # Already initialized by another module (e.g., model.py) — safe to ignore
                    log.debug("GPU memory growth already set; skipping.")
            log.info("✅ GPU detected: %s", [g.name for g in gpus])
        else:
            log.info("ℹ️  No GPU — inference will run on CPU.")
    return device


DEVICE = _configure_gpu()


# ── MediaPipe Hands (singleton) ────────────────────────────────────────────────
class _HandsPool:
    """Thread-local MediaPipe Hands instances (MediaPipe is not thread-safe)."""

    def __init__(self) -> None:
        self._local = threading.local()

    def get(self) -> mp.solutions.hands.Hands:
        if not hasattr(self._local, "hands"):
            self._local.hands = mp.solutions.hands.Hands(
                static_image_mode=True,       # stateless server mode: process frames independently
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self._local.hands



_hands_pool = _HandsPool()


# ── Base64 → OpenCV ───────────────────────────────────────────────────────────
def decode_base64_image(b64_str: str) -> Optional[np.ndarray]:
    """Decode a base64-encoded JPEG/PNG string to a BGR OpenCV image."""
    try:
        # Strip data-URL prefix if present (e.g., "data:image/jpeg;base64,...")
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        log.warning("Failed to decode image: %s", exc)
        return None


# ── Landmark Extraction ───────────────────────────────────────────────────────
def extract_landmarks_from_image(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Run MediaPipe Hands on a BGR frame and return normalized (63,) landmark vector.
    Returns None if no hand is detected.
    """
    hands = _hands_pool.get()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_lm = results.multi_hand_landmarks[0]
    raw = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
        dtype=np.float32,
    ).flatten()  # (63,)

    h, w = image_bgr.shape[:2]
    return normalize_landmarks(raw, width=w, height=h)


def extract_landmarks_with_viz(
    image_bgr: np.ndarray,
) -> tuple[Optional[np.ndarray], list[dict]]:
    """
    Extract landmarks AND return raw landmark list for front-end overlay rendering.

    Returns:
        (normalized_landmarks, raw_landmark_list)
        raw_landmark_list: [{"x": float, "y": float, "z": float}, ...] × 21
    """
    hands = _hands_pool.get()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None, []

    hand_lm = results.multi_hand_landmarks[0]
    raw_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_lm.landmark]

    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32).flatten()
    h, w = image_bgr.shape[:2]
    normalized = normalize_landmarks(raw, width=w, height=h)

    return normalized, raw_list



# ── Temporal Smoother ─────────────────────────────────────────────────────────
class TemporalSmoother:
    """
    Rolling majority-vote smoother over recent predictions.

    Filters out single-frame flicker and transient misclassifications.
    """

    def __init__(self, window: int = SMOOTHING_WINDOW) -> None:
        self._window = window
        self._history: collections.deque[int] = collections.deque(maxlen=window)

    def push(self, label_idx: int) -> int:
        """Add a prediction and return the majority vote."""
        self._history.append(label_idx)
        return collections.Counter(self._history).most_common(1)[0][0]

    def reset(self) -> None:
        self._history.clear()

    @property
    def filled(self) -> bool:
        return len(self._history) == self._window


# ── Mode Detector ─────────────────────────────────────────────────────────────
class ModeDetector:
    """
    Detects whether the current hand gesture is static (Letter) or dynamic (Word).
    Uses a dual-threshold hysteresis based on fingertip displacement.
    """

    START_THRESHOLD = 0.035
    END_THRESHOLD = 0.015
    WINDOW = 8
    CONSECUTIVE_FRAMES = 3

    def __init__(self) -> None:
        self.state = "letter"  # "letter" (static) or "word" (moving)
        self._motion: collections.deque[float] = collections.deque(maxlen=self.WINDOW)
        self._prev_tips: Optional[np.ndarray] = None
        self._stop_counter = 0

    def update(self, landmarks: np.ndarray) -> str:
        """Returns 'letter' or 'word'."""
        # landmarks is shape (63,)
        lms = landmarks.reshape(21, 3)
        # Slices coordinates of 5 fingertips: 4, 8, 12, 16, 20
        tips = lms[[4, 8, 12, 16, 20]]

        if self._prev_tips is not None:
            # Average displacement of fingertips
            disp = float(np.mean(np.linalg.norm(tips - self._prev_tips, axis=1)))
            self._motion.append(disp)
        self._prev_tips = tips.copy()

        if len(self._motion) < 3:
            return self.state

        mean_motion = float(np.mean(self._motion))

        if self.state == "letter":
            if mean_motion > self.START_THRESHOLD:
                self.state = "word"
                self._stop_counter = 0
        else:  # state is "word"
            if mean_motion < self.END_THRESHOLD:
                self._stop_counter += 1
                if self._stop_counter >= self.CONSECUTIVE_FRAMES:
                    self.state = "letter"
            else:
                self._stop_counter = 0

        return self.state

    def reset(self) -> None:
        self.state = "letter"
        self._motion.clear()
        self._prev_tips = None
        self._stop_counter = 0


# ── Inference Engine (main class) ─────────────────────────────────────────────
class InferenceEngine:
    """
    Central inference engine — load once, call many times.

    Thread-safe: model.predict() with TF2 is safe under the GIL.
    Models are loaded lazily on first use to avoid startup delay.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        smoothing_window: int = SMOOTHING_WINDOW,
        sequence_len: int = SEQUENCE_LEN,
        dialect: str = "ASL",
    ) -> None:
        self._threshold = confidence_threshold
        self._dialect   = dialect
        self._seq_len   = sequence_len

        self._mlp:  Optional[Any] = None
        self._lstm: Optional[Any] = None
        self._cnn:  Optional[Any] = None

        self._lock        = threading.Lock()
        self._smoother    = TemporalSmoother(smoothing_window)
        self._mode_det    = ModeDetector()
        self._seq_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=sequence_len)

    # ── Lazy model loaders ────────────────────────────────────────────────

    def _load_mlp(self) -> Optional[Any]:
        try:
            import onnxruntime as ort
            path = MODELS_DIR / "asl_mlp.onnx"
            if path.exists():
                session = ort.InferenceSession(str(path))
                log.info("✅ MLP loaded from ONNX: %s", path.name)
                return session
        except Exception as exc:
            log.debug("ONNX MLP load failed, falling back to Keras: %s", exc)

        path = MODELS_DIR / "asl_mlp.keras"
        if not path.exists():
            path = MODELS_DIR / "asl_mlp_best.keras"
        if not path.exists():
            log.warning("MLP model not found at %s", MODELS_DIR)
            return None
        model = tf.keras.models.load_model(str(path))
        log.info("✅ MLP loaded from Keras: %s", path.name)
        return model

    def _load_lstm(self) -> Optional[Any]:
        try:
            import onnxruntime as ort
            path = MODELS_DIR / "asl_lstm.onnx"
            if path.exists():
                session = ort.InferenceSession(str(path))
                log.info("✅ LSTM loaded from ONNX: %s", path.name)
                return session
        except Exception as exc:
            log.debug("ONNX LSTM load failed, falling back to Keras: %s", exc)

        path = MODELS_DIR / "asl_lstm.keras"
        if not path.exists():
            path = MODELS_DIR / "asl_lstm_best.keras"
        if not path.exists():
            log.warning("LSTM model not found at %s", MODELS_DIR)
            return None
        model = tf.keras.models.load_model(str(path))
        log.info("✅ LSTM loaded from Keras: %s", path.name)
        return model

    def _load_cnn(self) -> Optional[Any]:
        try:
            import onnxruntime as ort
            path = MODELS_DIR / "asl_mobilenet.onnx"
            if path.exists():
                session = ort.InferenceSession(str(path))
                log.info("✅ CNN loaded from ONNX: %s", path.name)
                return session
        except Exception as exc:
            log.debug("ONNX CNN load failed, falling back to Keras: %s", exc)

        path = MODELS_DIR / "asl_mobilenet.keras"
        if not path.exists():
            log.warning("CNN model not found at %s", MODELS_DIR)
            return None
        model = tf.keras.models.load_model(str(path))
        log.info("✅ CNN loaded from Keras: %s", path.name)
        return model

    def ensure_models_loaded(self) -> dict[str, bool]:
        """Load all available models. Called at API startup."""
        with self._lock:
            if self._mlp is None:
                self._mlp = self._load_mlp()
            if self._lstm is None:
                self._lstm = self._load_lstm()
            if self._cnn is None:
                self._cnn = self._load_cnn()

        return {
            "mlp":  self._mlp is not None,
            "lstm": self._lstm is not None,
            "cnn":  self._cnn is not None,
        }

    # ── Single-Frame Prediction (Letter Mode) ────────────────────────────

    def predict_frame(
        self, b64_image: str
    ) -> dict:
        """
        Predict ASL letter from a single base64-encoded frame.

        Pipeline:
            decode → MediaPipe → normalize → MLP inference
            → confidence filter → temporal smoothing → response

        Returns:
            {
                "letter":     str | None,
                "confidence": float,
                "mode":       "letter" | "word",
                "landmarks":  list[dict],   # for front-end overlay
                "latency_ms": float,
            }
        """
        t0 = time.perf_counter()

        # B4 FIX: lock guard on lazy load — prevents TOCTOU race condition
        if self._mlp is None:
            with self._lock:
                if self._mlp is None:  # double-checked locking
                    self._mlp = self._load_mlp()

        if not hasattr(self, "_frame_count"):
            self._frame_count = 0
        self._frame_count += 1

        image = decode_base64_image(b64_image)
        if image is None:
            if self._frame_count % 30 == 0:
                log.warning("[DEBUG] Failed to decode base64 image")
            return self._error_response("Invalid image data", t0)

        landmarks, raw_lm = extract_landmarks_with_viz(image)
        if landmarks is None:
            if self._frame_count % 30 == 0:
                log.info(f"[DEBUG] Frame processed. Image shape: {image.shape if image is not None else 'None'}. Hand detected: False")
            return self._no_hand_response(t0)

        if self._frame_count % 30 == 0:
            log.info(f"[DEBUG] Frame processed. Image shape: {image.shape}. Hand detected: True")

        # Update mode detector
        prev_mode = self._mode_det.state
        mode = self._mode_det.update(landmarks)
        gesture_ended = (prev_mode == "word" and mode == "letter")

        # MLP inference
        x = landmarks[np.newaxis, :]               # (1, 63)
        if self._mlp is not None:
            if hasattr(self._mlp, "run"):
                input_name = self._mlp.get_inputs()[0].name
                proba = self._mlp.run(None, {input_name: x.astype(np.float32)})[0][0]
            else:
                proba = self._mlp.predict(x, verbose=0)[0]  # (29,)
        else:
            # Fallback: uniform distribution (no model trained yet)
            proba = np.ones(NUM_CLASSES) / NUM_CLASSES

        best_idx  = int(np.argmax(proba))
        best_conf = float(proba[best_idx])

        # Confidence gate
        if best_conf < self._threshold:
            # DO NOT reset smoother on transient dropouts
            return {
                "letter":     None,
                "confidence": round(best_conf, 4),
                "mode":       mode,
                "gesture_ended": gesture_ended,
                "landmarks":  raw_lm,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "below_threshold": True,
            }


        # Temporal smoothing
        smoothed_idx  = self._smoother.push(best_idx)
        smoothed_conf = float(proba[smoothed_idx])

        # Update sequence buffer (for word mode)
        self._seq_buffer.append(landmarks)

        return {
            "letter":          ASL_CLASSES[smoothed_idx],
            "raw_letter":      ASL_CLASSES[best_idx],
            "confidence":      round(smoothed_conf, 4),
            "top3":            self._top3(proba),
            "mode":            mode,
            "gesture_ended":   gesture_ended,
            "landmarks":       raw_lm,
            "smoothed":        self._smoother.filled,
            "below_threshold": False,
            "latency_ms":      round((time.perf_counter() - t0) * 1000, 1),
        }

    # ── Sequence Prediction (Word Mode) ──────────────────────────────────

    def predict_sequence(
        self, b64_frames: list[str]
    ) -> dict:
        """
        Predict ASL word from a sequence of base64-encoded frames.

        Args:
            b64_frames: List of base64 image strings (target: 30 frames).

        Returns:
            {
                "word":       str | None,
                "confidence": float,
                "latency_ms": float,
            }
        """
        t0 = time.perf_counter()

        # B4 FIX: double-checked locking for LSTM lazy load
        if self._lstm is None:
            with self._lock:
                if self._lstm is None:
                    self._lstm = self._load_lstm()

        # Extract landmarks from each frame
        seq_landmarks: list[np.ndarray] = []
        for b64 in b64_frames:
            image = decode_base64_image(b64)
            if image is None:
                continue
            lm = extract_landmarks_from_image(image)
            if lm is not None:
                seq_landmarks.append(lm)

        if len(seq_landmarks) < 5:
            return {
                "word":       None,
                "confidence": 0.0,
                "error":      "Too few valid frames with detected hands.",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }

        # Pad or truncate to SEQUENCE_LEN
        if len(seq_landmarks) < self._seq_len:
            # Repeat last frame to fill
            pad = [seq_landmarks[-1]] * (self._seq_len - len(seq_landmarks))
            seq_landmarks = seq_landmarks + pad
        else:
            seq_landmarks = seq_landmarks[: self._seq_len]

        seq = np.stack(seq_landmarks, axis=0)[np.newaxis, :, :]  # (1, T, 63)

        if self._lstm is not None:
            if hasattr(self._lstm, "run"):
                input_name = self._lstm.get_inputs()[0].name
                proba = self._lstm.run(None, {input_name: seq.astype(np.float32)})[0][0]
            else:
                proba = self._lstm.predict(seq, verbose=0)[0]
        else:
            proba = np.ones(NUM_CLASSES) / NUM_CLASSES

        best_idx  = int(np.argmax(proba))
        best_conf = float(proba[best_idx])

        return {
            "word":       ASL_CLASSES[best_idx] if best_conf >= self._threshold else None,
            "confidence": round(best_conf, 4),
            "top3":       self._top3(proba),
            "frames_used": len(seq_landmarks),
            "below_threshold": best_conf < self._threshold,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ── CNN Fallback Prediction ───────────────────────────────────────────

    def predict_frame_cnn(self, b64_image: str) -> dict:
        """Predict using CNN (MobileNetV3) on raw image crop — fallback for low-landmark confidence."""
        t0 = time.perf_counter()

        # B4 FIX: double-checked locking for CNN lazy load
        if self._cnn is None:
            with self._lock:
                if self._cnn is None:
                    self._cnn = self._load_cnn()

        image = decode_base64_image(b64_image)
        if image is None:
            return self._error_response("Invalid image", t0)

        # Resize to 224×224 for MobileNetV3
        crop = cv2.resize(image, (224, 224))
        x = np.expand_dims(crop, 0).astype(np.float32)  # (1, 224, 224, 3)

        if self._cnn is not None:
            if hasattr(self._cnn, "run"):
                input_name = self._cnn.get_inputs()[0].name
                proba = self._cnn.run(None, {input_name: x.astype(np.float32)})[0][0]
            else:
                proba = self._cnn.predict(x, verbose=0)[0]
        else:
            proba = np.ones(NUM_CLASSES) / NUM_CLASSES

        best_idx  = int(np.argmax(proba))
        best_conf = float(proba[best_idx])

        return {
            "letter":     ASL_CLASSES[best_idx] if best_conf >= self._threshold else None,
            "confidence": round(best_conf, 4),
            "model":      "cnn_fallback",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _top3(self, proba: np.ndarray) -> list[dict]:
        top_idx = np.argsort(proba)[-3:][::-1]
        return [
            {"label": ASL_CLASSES[i], "confidence": round(float(proba[i]), 4)}
            for i in top_idx
        ]

    def _error_response(self, msg: str, t0: float) -> dict:
        return {"error": msg, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def _no_hand_response(self, t0: float) -> dict:
        return {
            "letter":     None,
            "confidence": 0.0,
            "mode":       "letter",
            "landmarks":  [],
            "hand_detected": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def reset_state(self) -> None:
        """Reset smoother and mode detector (call between sessions)."""
        self._smoother.reset()
        self._mode_det.reset()
        self._seq_buffer.clear()

    @property
    def status(self) -> dict:
        return {
            "device":  DEVICE,
            "dialect": self._dialect,
            "models": {
                "mlp":  self._mlp is not None,
                "lstm": self._lstm is not None,
                "cnn":  self._cnn is not None,
            },
            "confidence_threshold": self._threshold,
            "smoothing_window":     self._smoother._window,
        }


# ── Module-level singleton (imported by main.py) ──────────────────────────────
engine = InferenceEngine()
