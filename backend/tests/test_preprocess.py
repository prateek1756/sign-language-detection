"""
test_preprocess.py — Unit tests for landmark normalization.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.preprocess import normalize_landmarks


def test_output_shape():
    """Verify that normalize_landmarks returns a flat array of shape (63,)."""
    raw = np.random.rand(63).astype(np.float32)
    result = normalize_landmarks(raw)
    assert result.shape == (63,)


def test_wrist_at_origin():
    """Verify that the wrist landmark (index 0) is translated to [0.0, 0.0, 0.0]."""
    raw = np.random.rand(63).astype(np.float32)
    result = normalize_landmarks(raw)
    # The first 3 dimensions of the normalized array correspond to landmark 0 (wrist)
    np.testing.assert_allclose(result[:3], [0.0, 0.0, 0.0], atol=1e-6)


def test_translation_invariance():
    """Verify that shifting all landmarks by a constant vector yields the same normalized output."""
    raw = np.random.rand(21, 3).astype(np.float32)
    # Create shifted copy
    shift = np.array([1.2, -3.4, 0.5], dtype=np.float32)
    raw_shifted = raw + shift

    norm_original = normalize_landmarks(raw.flatten())
    norm_shifted = normalize_landmarks(raw_shifted.flatten())

    np.testing.assert_allclose(norm_original, norm_shifted, atol=1e-5)


def test_scale_invariance():
    """Verify that scaling the size of the hand bounding box yields the same normalized output."""
    raw = np.random.rand(21, 3).astype(np.float32)
    # Create scaled copy of all dimensions (x, y, z)
    scale_factor = 2.5
    raw_scaled = raw * scale_factor

    norm_original = normalize_landmarks(raw.flatten())
    norm_scaled = normalize_landmarks(raw_scaled.flatten())

    np.testing.assert_allclose(norm_original, norm_scaled, atol=1e-5)
