"""
test_onnx_export.py — Validates the exported ONNX models.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import onnxruntime as ort
import keras
from configs.training_config import MODELS_DIR, FEATURE_DIM


def test_mlp_onnx_loaded():
    """Load exported MLP ONNX model, run prediction, and check shapes against Keras model."""
    onnx_path = MODELS_DIR / "asl_mlp.onnx"
    assert onnx_path.exists(), "asl_mlp.onnx does not exist. Run export_onnx.py first."
    
    keras_path = MODELS_DIR / "asl_mlp.keras"
    if not keras_path.exists():
        keras_path = MODELS_DIR / "asl_mlp_best.keras"
    keras_model = keras.models.load_model(str(keras_path))
    expected_classes = keras_model.output_shape[-1]
    
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    dummy_input = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    outputs = session.run([output_name], {input_name: dummy_input})
    
    prediction = outputs[0]
    assert prediction.shape == (1, expected_classes), f"Expected shape (1, {expected_classes}), got {prediction.shape}"


def test_lstm_onnx_loaded():
    """Load exported LSTM ONNX model, run prediction, and check shapes against Keras model."""
    onnx_path = MODELS_DIR / "asl_lstm.onnx"
    assert onnx_path.exists(), "asl_lstm.onnx does not exist. Run export_onnx.py first."
    
    keras_path = MODELS_DIR / "asl_lstm.keras"
    if not keras_path.exists():
        keras_path = MODELS_DIR / "asl_lstm_best.keras"
    keras_model = keras.models.load_model(str(keras_path))
    expected_classes = keras_model.output_shape[-1]
    
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    dummy_input = np.zeros((1, 30, FEATURE_DIM), dtype=np.float32)
    outputs = session.run([output_name], {input_name: dummy_input})
    
    prediction = outputs[0]
    assert prediction.shape == (1, expected_classes), f"Expected shape (1, {expected_classes}), got {prediction.shape}"


def test_cnn_onnx_loaded():
    """Load exported CNN ONNX model, run prediction, and check shapes against Keras model."""
    onnx_path = MODELS_DIR / "asl_mobilenet.onnx"
    assert onnx_path.exists(), "asl_mobilenet.onnx does not exist. Run export_onnx.py first."
    
    keras_path = MODELS_DIR / "asl_mobilenet.keras"
    keras_model = keras.models.load_model(str(keras_path))
    expected_classes = keras_model.output_shape[-1]
    
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    outputs = session.run([output_name], {input_name: dummy_input})
    
    prediction = outputs[0]
    assert prediction.shape == (1, expected_classes), f"Expected shape (1, {expected_classes}), got {prediction.shape}"

