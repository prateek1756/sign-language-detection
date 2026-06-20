"""
export_onnx.py — Converts Keras (.keras) models to ONNX format for browser execution.
======================================================================================
Usage:
    python src/export_onnx.py --model mlp
    python src/export_onnx.py --model all
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import tensorflow as tf
import tf2onnx
import keras

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.training_config import MODELS_DIR, FEATURE_DIM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_onnx")

# Target directory in the web app's public folder for ONNX assets
WEB_PUBLIC_DIR = BASE_DIR.parent / "web" / "public" / "models"


def get_keras_model_path(name: str) -> Path:
    """Finds the Keras model, checking both standard and '_best' naming variations."""
    path = MODELS_DIR / f"{name}.keras"
    if not path.exists():
        path = MODELS_DIR / f"{name}_best.keras"
    if not path.exists():
        raise FileNotFoundError(f"Keras model '{name}' not found in {MODELS_DIR}")
    return path


def convert_mlp() -> Path:
    """Converts MLP model (landmarks -> letter) to ONNX."""
    keras_path = get_keras_model_path("asl_mlp")
    onnx_path = MODELS_DIR / "asl_mlp.onnx"
    
    log.info("Loading MLP from %s...", keras_path.name)
    model = keras.models.load_model(str(keras_path))
    
    log.info("Converting MLP to ONNX...")
    spec = (tf.TensorSpec((None, FEATURE_DIM), tf.float32, name="landmarks"),)
    
    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=str(onnx_path)
    )
    log.info("✅ MLP converted: %s", onnx_path.name)
    return onnx_path


def convert_lstm() -> Path:
    """Converts LSTM model (landmark sequences -> word) to ONNX."""
    keras_path = get_keras_model_path("asl_lstm")
    onnx_path = MODELS_DIR / "asl_lstm.onnx"
    
    log.info("Loading LSTM from %s...", keras_path.name)
    model = keras.models.load_model(str(keras_path))
    
    from configs.training_config import LSTMConfig
    cfg = LSTMConfig()
    
    log.info("Converting LSTM to ONNX...")
    spec = (tf.TensorSpec((None, cfg.sequence_len, FEATURE_DIM), tf.float32, name="landmark_sequence"),)
    
    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=str(onnx_path)
    )
    log.info("✅ LSTM converted: %s", onnx_path.name)
    return onnx_path


def convert_cnn() -> Path:
    """Converts MobileNetV3 CNN model (image crop -> letter) to ONNX."""
    keras_path = get_keras_model_path("asl_mobilenet")
    onnx_path = MODELS_DIR / "asl_mobilenet.onnx"
    
    log.info("Loading MobileNetV3 CNN from %s...", keras_path.name)
    model = keras.models.load_model(str(keras_path))
    
    from configs.training_config import CNNConfig
    cfg = CNNConfig()
    
    log.info("Converting CNN to ONNX...")
    spec = (tf.TensorSpec((None, *cfg.input_shape), tf.float32, name="hand_crop"),)
    
    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=str(onnx_path)
    )
    log.info("✅ CNN converted: %s", onnx_path.name)
    return onnx_path


def copy_to_web_public(onnx_path: Path) -> None:
    """Copies exported ONNX model to the React app's public assets folder."""
    if not onnx_path.exists():
        return
    WEB_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEB_PUBLIC_DIR / onnx_path.name
    shutil.copy2(onnx_path, dest)
    log.info("🚀 Copied %s → web/public/models/", onnx_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Keras models to ONNX")
    parser.add_argument(
        "--model",
        choices=["mlp", "lstm", "cnn", "all"],
        default="all",
        help="Specify which model to export (default: all)",
    )
    args = parser.parse_args()
    
    targets = ["mlp", "lstm", "cnn"] if args.model == "all" else [args.model]
    
    for t in targets:
        try:
            log.info("━" * 50)
            log.info("Exporting model: %s", t.upper())
            log.info("━" * 50)
            
            if t == "mlp":
                path = convert_mlp()
            elif t == "lstm":
                path = convert_lstm()
            elif t == "cnn":
                path = convert_cnn()
                
            copy_to_web_public(path)
        except Exception as e:
            log.error("Failed to convert %s: %s", t.upper(), e, exc_info=True)


if __name__ == "__main__":
    main()
