"""
train.py — Training Pipeline (local-first, Keras 3 compatible)
===============================================================
Usage:
    python src/train.py --model mlp
    python src/train.py --model lstm
    python src/train.py --model cnn
    python src/train.py --model all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.training_config import (
    ASL_CLASSES, CNNConfig, LSTMConfig, MLPConfig,
    NUM_CLASSES, PROCESSED_DIR,
)
from src.model import (
    build_cnn, build_lstm, build_mlp, compile_model,
    configure_gpu, save_model, unfreeze_cnn_top,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_landmark_data(
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple:
    X_path = PROCESSED_DIR / "landmarks_all.npy"
    y_path = PROCESSED_DIR / "labels_all.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {PROCESSED_DIR}.\n"
            "Run: python src/preprocess.py --all"
        )

    X = np.load(str(X_path), allow_pickle=False)
    y = np.load(str(y_path), allow_pickle=False)
    log.info("Loaded — X: %s  y: %s", X.shape, y.shape)

    # Remove classes with too few samples to stratify-split
    min_samples = max(3, int(1 / min(val_size, test_size)) + 1)
    unique, counts = np.unique(y, return_counts=True)
    valid = unique[counts >= min_samples]
    removed = unique[counts < min_samples]
    if len(removed):
        log.warning(
            "Removing %d class(es) with fewer than %d samples: %s",
            len(removed), min_samples,
            [ASL_CLASSES[i] for i in removed if i < len(ASL_CLASSES)],
        )
        mask = np.isin(y, valid)
        X, y = X[mask], y[mask]
        label_map = {old: new for new, old in enumerate(sorted(valid))}
        y = np.array([label_map[lbl] for lbl in y], dtype=np.int32)

    actual_classes = len(np.unique(y))
    log.info("Classes for training: %d", actual_classes)

    lb = LabelBinarizer()
    y_ohe = lb.fit_transform(y)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y_ohe, test_size=val_size + test_size,
        random_state=seed, stratify=y,
    )
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test,
        random_state=seed, stratify=np.argmax(y_tmp, axis=1),
    )
    log.info("Split — train: %d  val: %d  test: %d", len(X_tr), len(X_val), len(X_test))
    return X_tr, X_val, X_test, y_tr, y_val, y_test, actual_classes


def build_sequence_dataset(
    X: np.ndarray, y: np.ndarray,
    seq_len: int, batch_size: int, shuffle: bool = True,
) -> tf.data.Dataset:
    N, feature_dim = len(X), X.shape[1]
    num_classes = y.shape[1]

    def generator():
        indices = np.random.permutation(N) if shuffle else np.arange(N)
        for i in indices:
            base = X[i]
            seq = np.stack([
                base + np.random.randn(feature_dim).astype(np.float32) * 0.005
                for _ in range(seq_len)
            ], axis=0)
            yield seq, y[i]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(N, 2000), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_image_dataset(split: str, cfg: CNNConfig) -> tf.data.Dataset:
    crops_dir = PROCESSED_DIR / "crops"
    if not crops_dir.exists():
        raise FileNotFoundError(f"Crops not found: {crops_dir}")

    class_names = sorted(d.name for d in crops_dir.iterdir() if d.is_dir())
    subset_map = {"train": "training", "val": "validation"}
    keras_subset = subset_map.get(split, split)
    use_split = keras_subset in ("training", "validation")

    ds = tf.keras.utils.image_dataset_from_directory(
        crops_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(cfg.input_shape[0], cfg.input_shape[1]),
        batch_size=cfg.phase1_batch_size,
        shuffle=(split == "train"),
        seed=42,
        validation_split=0.2 if use_split else None,
        subset=keras_subset if use_split else None,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def make_callbacks(
    log_dir: Path,
    checkpoint_path: Path,
    early_stop_patience: int,
    lr_patience: int = 8,
    lr_factor: float = 0.5,
) -> list:
    return [
        keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(log_dir / "log.csv"), append=True),
    ]


# ── Train MLP ─────────────────────────────────────────────────────────────────

def train_mlp(cfg: MLPConfig | None = None) -> keras.Model:
    cfg = cfg or MLPConfig()
    configure_gpu()

    X_tr, X_val, X_test, y_tr, y_val, y_test, actual_classes = load_landmark_data()
    cfg.num_classes = actual_classes

    model = build_mlp(cfg)
    compile_model(model, cfg.learning_rate, cfg.label_smoothing)
    model.summary()

    ckpt = cfg.save_dir / f"{cfg.model_name}_best.keras"
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=make_callbacks(cfg.log_dir, ckpt, cfg.early_stop_patience, cfg.lr_patience, cfg.lr_decay_factor),
        verbose=1,
    )

    save_model(model, cfg.save_dir / f"{cfg.model_name}.keras")
    loss, acc, top5 = model.evaluate(X_test, y_test, verbose=0)
    log.info("✅ MLP — loss=%.4f  acc=%.4f  top5=%.4f", loss, acc, top5)
    return model


# ── Train LSTM ────────────────────────────────────────────────────────────────

def train_lstm(cfg: LSTMConfig | None = None) -> keras.Model:
    cfg = cfg or LSTMConfig()
    configure_gpu()

    X_tr, X_val, X_test, y_tr, y_val, y_test, actual_classes = load_landmark_data()
    cfg.num_classes = actual_classes

    train_ds = build_sequence_dataset(X_tr, y_tr, cfg.sequence_len, cfg.batch_size)
    val_ds   = build_sequence_dataset(X_val, y_val, cfg.sequence_len, cfg.batch_size, shuffle=False)
    test_ds  = build_sequence_dataset(X_test, y_test, cfg.sequence_len, cfg.batch_size, shuffle=False)

    model = build_lstm(cfg)
    compile_model(model, cfg.learning_rate, cfg.label_smoothing, cfg.gradient_clip)
    model.summary()

    ckpt = cfg.save_dir / f"{cfg.model_name}_best.keras"
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=make_callbacks(cfg.log_dir, ckpt, cfg.early_stop_patience, cfg.lr_patience, cfg.lr_decay_factor),
        verbose=1,
    )

    save_model(model, cfg.save_dir / f"{cfg.model_name}.keras")
    loss, acc, top5 = model.evaluate(test_ds, verbose=0)
    log.info("✅ LSTM — loss=%.4f  acc=%.4f  top5=%.4f", loss, acc, top5)
    return model


# ── Train CNN ─────────────────────────────────────────────────────────────────

def train_cnn(cfg: CNNConfig | None = None) -> keras.Model | None:
    cfg = cfg or CNNConfig()
    configure_gpu()

    try:
        train_ds = build_image_dataset("train", cfg)
        val_ds   = build_image_dataset("val", cfg)
    except FileNotFoundError as e:
        log.error("CNN requires image crops: %s", e)
        return None

    model = build_cnn(cfg)
    ckpt  = cfg.save_dir / f"{cfg.model_name}_best.keras"

    # Phase 1 — frozen base
    log.info("CNN Phase 1 — frozen base")
    compile_model(model, cfg.phase1_lr, cfg.label_smoothing)
    model.fit(
        train_ds, validation_data=val_ds, epochs=cfg.phase1_epochs,
        callbacks=make_callbacks(cfg.log_dir / "p1", ckpt, cfg.early_stop_patience),
        verbose=1,
    )

    # Phase 2 — fine-tune
    log.info("CNN Phase 2 — fine-tuning from layer %d", cfg.fine_tune_from)
    unfreeze_cnn_top(model, cfg.fine_tune_from)
    compile_model(model, cfg.phase2_lr, cfg.label_smoothing)
    model.fit(
        train_ds, validation_data=val_ds, epochs=cfg.phase2_epochs,
        callbacks=make_callbacks(cfg.log_dir / "p2", ckpt, cfg.early_stop_patience),
        verbose=1,
    )

    save_model(model, cfg.save_dir / f"{cfg.model_name}.keras")
    log.info("✅ CNN training complete")
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL sign language models")
    parser.add_argument("--model", choices=["mlp", "lstm", "cnn", "all"], default="mlp")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    args = parser.parse_args()

    mlp_cfg  = MLPConfig()
    lstm_cfg = LSTMConfig()
    cnn_cfg  = CNNConfig()

    if args.epochs:
        mlp_cfg.epochs  = args.epochs
        lstm_cfg.epochs = args.epochs
    if args.lr:
        mlp_cfg.learning_rate  = args.lr
        lstm_cfg.learning_rate = args.lr
    if args.batch_size:
        mlp_cfg.batch_size  = args.batch_size
        lstm_cfg.batch_size = args.batch_size

    targets = ["mlp", "lstm", "cnn"] if args.model == "all" else [args.model]
    for t in targets:
        log.info("━" * 50)
        log.info("Training: %s", t.upper())
        log.info("━" * 50)
        if t == "mlp":
            train_mlp(mlp_cfg)
        elif t == "lstm":
            train_lstm(lstm_cfg)
        elif t == "cnn":
            train_cnn(cnn_cfg)

    log.info("🎉 Done! Models saved to backend/models/")
    log.info("   TensorBoard: tensorboard --logdir backend/logs/")


if __name__ == "__main__":
    main()
