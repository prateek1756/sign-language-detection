"""
model.py — Model Architectures (Keras 3 compatible)
====================================================
Three model families for ASL sign language detection:

  build_mlp()   → MLP landmark classifier       (fast, ~10 min local)
  build_lstm()  → BiLSTM sequence classifier    (word/phrase mode, ~45 min local)
  build_cnn()   → MobileNetV3 image classifier  (robustness, ~2h local)

All models use pure keras imports to avoid tensorflow.keras namespace conflicts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import keras
from keras import layers

if TYPE_CHECKING:
    from configs.training_config import CNNConfig, LSTMConfig, MLPConfig

log = logging.getLogger(__name__)


def configure_gpu() -> str:
    """Auto-detect GPU and enable memory growth. Returns 'GPU' or 'CPU'."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        log.info("✅ GPU detected: %s", [g.name for g in gpus])
        return "GPU"
    log.info("ℹ️  No GPU found — running on CPU.")
    return "CPU"


# ── MLP ───────────────────────────────────────────────────────────────────────

def build_mlp(cfg: "MLPConfig") -> keras.Model:
    """
    MLP classifier on 63 normalized MediaPipe landmarks.
    Input(63) → [Dense → BN → ReLU → Dropout] × N → Softmax(N_classes)
    """
    inp = keras.Input(shape=(cfg.input_dim,), name="landmarks")
    x = inp
    for i, units in enumerate(cfg.hidden_dims):
        x = layers.Dense(units, name=f"dense_{i}")(x)
        if cfg.use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.Dropout(cfg.dropout_rate, name=f"drop_{i}")(x)

    x = layers.Dense(cfg.num_classes, name="logits")(x)
    out = layers.Softmax(name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="ASL_MLP")
    log.info("MLP built — %d classes", cfg.num_classes)
    return model


# ── BiLSTM ────────────────────────────────────────────────────────────────────

def build_lstm(cfg: "LSTMConfig") -> keras.Model:
    """
    Bidirectional LSTM for dynamic gesture recognition.
    Input(T, 63) → BiLSTM(128) → Dropout → BiLSTM(64) → GAP → Softmax
    """
    inp = keras.Input(shape=(cfg.sequence_len, cfg.input_dim), name="landmark_sequence")
    x = inp
    for i, units in enumerate(cfg.lstm_units):
        lstm = layers.LSTM(
            units,
            return_sequences=True,
            dropout=cfg.dropout_rate,
            recurrent_dropout=0.0,
            name=f"lstm_{i}",
        )
        if cfg.bidirectional:
            x = layers.Bidirectional(lstm, name=f"bilstm_{i}")(x)
        else:
            x = lstm(x)
        x = layers.Dropout(cfg.dropout_rate, name=f"drop_{i}")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(cfg.num_classes, name="logits")(x)
    out = layers.Softmax(name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="ASL_LSTM")
    log.info("LSTM built — %d classes", cfg.num_classes)
    return model


# ── MobileNetV3 CNN ───────────────────────────────────────────────────────────

def build_cnn(cfg: "CNNConfig") -> keras.Model:
    """
    MobileNetV3Small transfer learning on 224×224 hand crops.
    Phase 1: frozen base → train head only.
    Phase 2: unfreeze top layers for fine-tuning.
    """
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=cfg.input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inp = keras.Input(shape=cfg.input_shape, name="hand_crop")
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inp)
    x = base(x, training=False)
    x = layers.Dropout(cfg.dropout_rate, name="drop_head")(x)
    x = layers.Dense(cfg.num_classes, name="logits")(x)
    out = layers.Softmax(name="predictions")(x)

    model = keras.Model(inputs=inp, outputs=out, name="ASL_MobileNetV3")
    log.info("CNN built — %d classes", cfg.num_classes)
    return model


def unfreeze_cnn_top(model: keras.Model, fine_tune_from: int) -> None:
    """Unfreeze top layers of MobileNetV3 base for Phase 2 fine-tuning."""
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    base.trainable = True
    for layer in base.layers[:fine_tune_from]:
        layer.trainable = False
    trainable = sum(1 for l in base.layers if l.trainable)
    log.info("CNN fine-tune: %d/%d base layers unfrozen.", trainable, len(base.layers))


# ── Compile ───────────────────────────────────────────────────────────────────

def compile_model(
    model: keras.Model,
    learning_rate: float,
    label_smoothing: float = 0.0,
    gradient_clip: float | None = None,
) -> None:
    """Compile model with Adam optimizer. AdamW avoided — Keras 3 compat."""
    opt_kwargs: dict = {"learning_rate": learning_rate}
    if gradient_clip is not None:
        opt_kwargs["clipnorm"] = gradient_clip

    model.compile(
        optimizer=keras.optimizers.Adam(**opt_kwargs),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")],
    )
    log.info("Compiled — lr=%.1e", learning_rate)


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model: keras.Model, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    log.info("✅ Saved → %s", path)


def load_model(path: Path) -> keras.Model:
    model = keras.models.load_model(str(path))
    log.info("✅ Loaded ← %s", path)
    return model
