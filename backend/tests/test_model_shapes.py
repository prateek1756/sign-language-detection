"""
test_model_shapes.py — Unit tests for model input/output shapes.

Tests:
    - MLP: input (1, 63) → output (1, 29)
    - LSTM: input (1, 30, 63) → output (1, 29)
    - CNN: input (1, 224, 224, 3) → output (1, 29)

These tests build models from config and run a single forward pass.
No trained weights needed — just architecture validation.

TODO (tomorrow): implement tests.
"""

# TODO (tomorrow): implement
# import numpy as np
# import pytest
# from src.model import build_mlp, build_lstm, build_cnn
# from configs.training_config import MLPConfig, LSTMConfig, CNNConfig
#
# def test_mlp_output_shape():
#     model = build_mlp(MLPConfig())
#     x = np.zeros((1, 63), dtype=np.float32)
#     out = model.predict(x, verbose=0)
#     assert out.shape == (1, 29)
#
# def test_lstm_output_shape():
#     ...
#
# def test_cnn_output_shape():
#     ...
