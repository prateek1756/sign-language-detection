"""
test_model_shapes.py — Unit tests for model input/output shapes.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from src.model import build_mlp, build_lstm, build_cnn
from configs.training_config import MLPConfig, LSTMConfig, CNNConfig


def test_mlp_output_shape():
    """Verify MLP inputs landmarks shape (1, 63) and outputs classification probabilities shape (1, 29)."""
    cfg = MLPConfig()
    model = build_mlp(cfg)
    x = np.zeros((1, cfg.input_dim), dtype=np.float32)
    out = model.predict(x, verbose=0)
    assert out.shape == (1, cfg.num_classes)


def test_lstm_output_shape():
    """Verify LSTM inputs sequence shape (1, 30, 63) and outputs shape (1, 29)."""
    cfg = LSTMConfig()
    model = build_lstm(cfg)
    x = np.zeros((1, cfg.sequence_len, cfg.input_dim), dtype=np.float32)
    out = model.predict(x, verbose=0)
    assert out.shape == (1, cfg.num_classes)


def test_cnn_output_shape():
    """Verify CNN inputs image crop shape (1, 224, 224, 3) and outputs shape (1, 29)."""
    cfg = CNNConfig()
    model = build_cnn(cfg)
    x = np.zeros((1,) + cfg.input_shape, dtype=np.float32)
    out = model.predict(x, verbose=0)
    assert out.shape == (1, cfg.num_classes)
