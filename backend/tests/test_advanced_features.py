"""
test_advanced_features.py — Unit tests for palm alignment, geometric features, and hysteresis ModeDetector.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.preprocess import align_palm_plane, compute_geometric_features, normalize_landmarks
from src.inference import ModeDetector, InferenceEngine


def test_palm_plane_alignment():
    """Verify that palm-plane rotation alignment projects landmarks into a canonical orientation."""
    # Create a random hand landmarks array translated to wrist
    raw_lms = np.random.rand(21, 3).astype(np.float32)
    raw_lms[0] = [0.0, 0.0, 0.0]  # Wrist at origin

    rotated = align_palm_plane(raw_lms)

    assert rotated.shape == (21, 3)
    # The wrist (0) must remain at the origin after rotation
    np.testing.assert_allclose(rotated[0], [0.0, 0.0, 0.0], atol=1e-5)

    # In the canonical coordinate system:
    # 1. Index MCP (5) should be aligned along the positive Y-axis.
    # Therefore, its X and Z coordinates must be near 0.0.
    assert abs(rotated[5, 0]) < 1e-4
    assert abs(rotated[5, 2]) < 1e-4
    assert rotated[5, 1] > 0.0  # Must be positive along Y-axis

    # 2. Pinky MCP (17) lies in the XY plane, so its Z coordinate must be near 0.0.
    assert abs(rotated[17, 2]) < 1e-4


def test_compute_geometric_features():
    """Verify that geometric feature engineering extracts 14 valid scale and rotation invariant features."""
    raw_lms = np.random.rand(21, 3).astype(np.float32)
    raw_lms[0] = [0.0, 0.0, 0.0]

    features = compute_geometric_features(raw_lms)
    assert features.shape == (14,)
    
    # All distance features (first 9 elements: 5 tips-to-wrist + 4 tip-to-tip) must be non-negative
    assert np.all(features[:9] >= 0.0)
    
    # All angle features (last 5 elements) must be in range [0, pi]
    assert np.all(features[9:] >= 0.0)
    assert np.all(features[9:] <= np.pi + 1e-5)


def test_normalize_landmarks_advanced_options():
    """Verify normalize_landmarks handles advanced options and returns the expected dimensions."""
    raw = np.random.rand(63).astype(np.float32)
    
    # Option 1: Rotation alignment enabled
    norm_rot = normalize_landmarks(raw, align_rotation=True)
    assert norm_rot.shape == (63,)
    
    # Option 2: Geometric features enabled (63 + 14 = 77)
    norm_geom = normalize_landmarks(raw, get_geometric=True)
    assert norm_geom.shape == (77,)

    # Option 3: Both enabled
    norm_both = normalize_landmarks(raw, align_rotation=True, get_geometric=True)
    assert norm_both.shape == (77,)


def test_hysteresis_mode_detector():
    """Verify that the dual-threshold hysteresis ModeDetector correctly handles active gesture boundaries."""
    detector = ModeDetector()
    assert detector.state == "letter"

    # Simulate static state (all landmarks are unchanged, motion = 0)
    static_lms = np.zeros(63, dtype=np.float32)
    for _ in range(10):
        state = detector.update(static_lms)
    assert state == "letter"

    # Simulate hand moving (large displacement on fingertips)
    # We alter fingertips (indices 4, 8, 12, 16, 20 corresponding to flat coords 12-14, 24-26, etc)
    moving_lms_1 = np.zeros(63, dtype=np.float32)
    moving_lms_2 = np.zeros(63, dtype=np.float32)
    
    # Apply a shift of 0.25 to the fingertips to exceed the START_THRESHOLD of 0.035
    tips = [4, 8, 12, 16, 20]
    for tip in tips:
        moving_lms_2[tip * 3 : tip * 3 + 3] = 0.25

    # Alternating feed to simulate high motion velocity
    for i in range(10):
        lms = moving_lms_2 if i % 2 == 0 else moving_lms_1
        state = detector.update(lms)
    
    assert state == "word"

    # Simulate hand slowing down (feed 12 slow frames to empty the motion window and trigger transition)
    slow_lms = np.zeros(63, dtype=np.float32)
    for _ in range(12):
        state = detector.update(slow_lms)
    
    # After consecutive slow frames, it should transition back to letter state
    assert detector.state == "letter"


def test_inference_engine_onnx_fallback():
    """Verify that InferenceEngine correctly initializes and exposes loaded models status."""
    engine = InferenceEngine()
    status = engine.ensure_models_loaded()
    assert isinstance(status, dict)
    assert "mlp" in status
    assert "lstm" in status
    assert "cnn" in status
