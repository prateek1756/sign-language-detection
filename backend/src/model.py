"""
model.py — Phase 3A/3B/3C: Model Architectures
================================================
Three model families, all driven by training_config dataclasses:

  build_mlp()   → MLP landmark classifier  (fast, CPU-friendly)
  build_lstm()  → BiLSTM sequence classifier (word/phrase mode)
  build_cnn()   → MobileNetV3 fallback      (robustness in bad lighting)

Usage:
    from src.model import build_mlp, build_lstm, build_cnn
    from configs.training_config import MLPConfig, LSTMConfig, CNNConfig

    model = build_mlp(MLPConfig())
    model.summary()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

if TYPE_CHECKING:
    from configs.training_config import CNNConfig, LSTMConfig, MLPConfig

log = logging.getLogger(__name__)

# ── GPU / Mixed-Precision Setup ───────────────────────────────────────────────
# Module-level state — prevents re-running expensive tf.config calls on every
# model build (calling tf.config.experimental.set_memory_growth twice raises).
_GPU_CHECKED: bool = False
_DEVICE: str = "CPU"


def configure_gpu(mixed_precision: bool = False) -> str:
    """
    Auto-detect GPU and enable memory growth.

    Mixed precision via set_global_policy() is intentionally disabled —
    it corrupts Functional model internals in Keras 3 (missing _dtype_policy,
    _parent_path attributes). The output layers already use dtype='float32'
    which is the only requirement for numerical stability with float16 compute.
    GPU memory growth is the only session-level side-effect applied here.
    """
    global _GPU_CHECKED, _DEVICE

    gpus = tf.config.list_physical_devices("GPU")
    _DEVICE = "GPU" if gpus else "CPU"

    if not _GPU_CHECKED:
        _GPU_CHECKED = True
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass  # already set — safe to ignore
            log.info("✅ GPU detected: %s", [g.name for g in gpus])
        else:
            log.info("ℹ️  No GPU found — running on CPU.")

    if mixed_precision:
        log.info("ℹ️  mixed_precision=True noted but global policy not set (Keras 3 compatibility).")

    return _DEVICE


# ── 3A: MLP Landmark Classifier ───────────────────────────────────────────────

def build_mlp(cfg: "MLPConfig") -> keras.Model:
    """
    Build an MLP classifier on 63 normalized MediaPipe landmarks.

    Architecture:
        Input(63) → [Dense → BatchNorm → ReLU → Dropout] × N → Dense(N_classes, softmax)

    Why BatchNorm before activation (not after):
        Avoids internal covariate shift before the ReLU non-linearity,
        which is the standard Keras/TF convention.
    """
    configure_gpu(cfg.mixed_precision)

    inp = keras.Input(shape=(cfg.input_dim,), name="landmarks")
    x = inp

    for i, units in enumerate(cfg.hidden_dims):
        x = layers.Dense(
            units,
            kernel_regularizer=regularizers.L2(cfg.weight_decay),
            name=f"dense_{i}",
        )(x)
        if cfg.use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.Dropout(cfg.dropout_rate, name=f"drop_{i}")(x)

    # Output — float32 always (required when mixed_precision=True)
    out = layers.Dense(cfg.num_classes, name="logits")(x)
    out = layers.Activation("softmax", dtype="float32", name="predictions")(out)

    model = keras.Model(inputs=inp, outputs=out, name="ASL_MLP")
    try:
        log.info("MLP built — params: %s", model.count_params())
    except Exception:
        log.info("MLP built — (param count unavailable on this Keras version)")
    return model


# ── 3B: BiLSTM Sequence Classifier ────────────────────────────────────────────

def build_lstm(cfg: "LSTMConfig") -> keras.Model:
    """
    Build a Bidirectional LSTM for dynamic gesture (word/phrase) recognition.

    Architecture:
        Input(T=30, 63) → BiLSTM(128) → Dropout → BiLSTM(64) → Dropout
        → GlobalAvgPool → Dense(N_classes, softmax)

    GlobalAvgPooling over time is more robust than taking only the last state
    when sequence lengths vary slightly.
    """
    configure_gpu(cfg.mixed_precision)

    inp = keras.Input(
        shape=(cfg.sequence_len, cfg.input_dim), name="landmark_sequence"
    )
    x = inp

    for i, units in enumerate(cfg.lstm_units):
        # Always return_sequences=True — GlobalAveragePooling1D pools over the full sequence
        lstm_layer = layers.LSTM(
            units,
            return_sequences=True,
            dropout=cfg.dropout_rate,
            recurrent_dropout=0.0,  # keep 0 — non-zero disables cuDNN LSTM kernel
            name=f"lstm_{i}",
        )
        if cfg.bidirectional:
            x = layers.Bidirectional(lstm_layer, name=f"bilstm_{i}")(x)
        else:
            x = lstm_layer(x)

        x = layers.Dropout(cfg.dropout_rate, name=f"drop_{i}")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    out = layers.Dense(cfg.num_classes, name="logits")(x)
    out = layers.Activation("softmax", dtype="float32", name="predictions")(out)

    model = keras.Model(inputs=inp, outputs=out, name="ASL_LSTM")
    try:
        log.info("LSTM built — params: %s", model.count_params())
    except Exception:
        log.info("LSTM built — (param count unavailable on this Keras version)")
    return model


# ── 3C: MobileNetV3 CNN Fallback ──────────────────────────────────────────────

def build_cnn(cfg: "CNNConfig") -> keras.Model:
    """
    Build a MobileNetV3Small transfer-learning model on 224×224 hand crop images.

    Two-phase strategy:
        Phase 1 — Frozen base: train only the new classification head.
        Phase 2 — Fine-tune: unfreeze layers from `fine_tune_from` onward.

    Call freeze_cnn_base() / unfreeze_cnn_top() to switch between phases.
    """
    configure_gpu(cfg.mixed_precision)

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=cfg.input_shape,
        include_top=False,
        weights="imagenet",
        pooling=cfg.pooling,
    )
    base.trainable = False  # Phase 1: frozen

    inp = keras.Input(shape=cfg.input_shape, name="hand_crop")
    # MobileNetV3 expects values in [-1, 1]
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inp)
    x = base(x, training=False)
    x = layers.Dropout(cfg.dropout_rate, name="drop_head")(x)
    out = layers.Dense(cfg.num_classes, name="logits")(x)
    out = layers.Activation("softmax", dtype="float32", name="predictions")(out)

    model = keras.Model(inputs=inp, outputs=out, name="ASL_MobileNetV3")
    try:
        log.info(
            "CNN built — base params: %s  head params: %s",
            base.count_params(),
            model.count_params() - base.count_params(),
        )
    except Exception:
        log.info("CNN built — (param count unavailable on this Keras version)")
    return model


def unfreeze_cnn_top(model: keras.Model, fine_tune_from: int) -> None:
    """
    Unlock top layers of the MobileNetV3 base for Phase 2 fine-tuning.

    Args:
        model:          The CNN model returned by build_cnn().
        fine_tune_from: Unfreeze all layers with index >= this value.
    """
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    base.trainable = True
    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False
    trainable = sum(1 for l in base.layers if l.trainable)
    log.info("CNN fine-tune: %d/%d base layers unfrozen.", trainable, len(base.layers))


# ── Compiler Helpers ──────────────────────────────────────────────────────────

def compile_model(
    model: keras.Model,
    learning_rate: float,
    label_smoothing: float = 0.0,
    optimizer_name: str = "adam",
    gradient_clip: float | None = None,
) -> None:
    """Compile a model with the given optimizer and loss settings.

    Note: AdamW is intentionally avoided — it triggers _dtype_policy AttributeError
    on Functional models in Keras 3.x. Use Adam with L2 regularization in layers instead.
    """
    opt_kwargs: dict = {"learning_rate": learning_rate}
    if gradient_clip is not None:
        opt_kwargs["clipnorm"] = gradient_clip

    # Use Adam for all cases — AdamW breaks Functional models in Keras 3
    if optimizer_name == "sgd":
        optimizer = keras.optimizers.SGD(**opt_kwargs)
    else:
        optimizer = keras.optimizers.Adam(**opt_kwargs)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")],
    )
    log.info("Model compiled — optimizer=%s  lr=%.1e", optimizer_name, learning_rate)


# ── Model I/O ─────────────────────────────────────────────────────────────────

def save_model(model: keras.Model, path: Path) -> None:
    """Save model in .keras format (recommended over HDF5)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    log.info("✅ Model saved → %s", path)


def load_model(path: Path) -> keras.Model:
    """Load a saved .keras model."""
    model = keras.models.load_model(str(path))
    log.info("✅ Model loaded ← %s", path)
    return model


def export_onnx(model: keras.Model, path: Path) -> None:
    """
    Export model to ONNX for deployment (requires tf2onnx).
    Gracefully skips if tf2onnx is not installed.
    """
    try:
        import tf2onnx  # noqa: F401
        import tf2onnx.convert as converter

        path = Path(path).with_suffix(".onnx")
        converter.from_keras(model, output_path=str(path))
        log.info("✅ ONNX model exported → %s", path)
    except ImportError:
        log.warning("tf2onnx not installed — skipping ONNX export. Run: pip install tf2onnx")
    except Exception as exc:
        log.error("ONNX export failed: %s", exc)
