"""
test_preprocess.py — Unit tests for landmark normalization.

Tests:
    - normalize_landmarks returns shape (63,)
    - wrist anchor is at origin after normalization
    - scale invariance: scaled input → same normalized output
    - translation invariance: shifted input → same normalized output

TODO (tomorrow): implement tests.
"""

# TODO (tomorrow): implement
# import numpy as np
# import pytest
# from src.preprocess import normalize_landmarks
#
# def test_output_shape():
#     raw = np.random.rand(63).astype(np.float32)
#     result = normalize_landmarks(raw)
#     assert result.shape == (63,)
#
# def test_wrist_at_origin():
#     ...
#
# def test_translation_invariance():
#     ...
#
# def test_scale_invariance():
#     ...
