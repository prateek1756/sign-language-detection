"""
train.py — Phase 3: Full Training Pipeline
==========================================
Trains MLP, LSTM, and CNN models with:
  - Early stopping + model checkpointing
  - ReduceLROnPlateau scheduler
  - TensorBoard logging
  - Model export (.keras + ONNX)

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.training_config import (
    ASL_CLASSES,
    CNNConfig,
    FEATURE_DIM,
    LSTMConfig,
    MLPConfig,
    NUM_CLASSES,
    PROCESSED_DIR,
)
from src.model import (
    build_cnn,
    build_lstm,
    build_mlp,
    compile_model,
    export_onnx,
    save_model,
    unfreeze_cnn_top,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ── Data Loaders ──────────────────────────────────────────────────────────────

def load_landmark_data(
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, ...]:
    """
    Load preprocessed landmark arrays and return train/val/test splits.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
        where y_* are one-hot encoded arrays of shape (N, NUM_CLASSES).
    """
    X_path = PROCESSED_DIR / "landmarks_all.npy"
    y_path = PROCESSED_DIR / "labels_all.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {PROCESSED_DIR}.\n"
            "Run: python src/preprocess.py --all"
        )

    X = np.load(str(X_path), allow_pickle=False)   # (N, 63)
    y = np.load(str(y_path), allow_pickle=False)    # (N,)

    log.info("Loaded data — X: %s  y: %s  classes: %d", X.shape, y.shape, NUM_CLASSES)

    # One-hot encode
    lb = LabelBinarizer()
    y_ohe = lb.fit_transform(y)                     # (N, 29)

    # Stratified split: train / (val + test)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y_ohe, test_size=val_size + test_size,
        random_state=seed, stratify=y,
    )
    rel_test = test_size / (val_size + test_size)
    # Bug fix: also stratify the val/test split to preserve class balance
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test,
        random_state=seed,
        stratify=np.argmax(y_tmp, axis=1),
    )

    log.info(
        "Split — train: %d  val: %d  test: %d",
        len(X_tr), len(X_val), len(X_test),
    )
    return X_tr, X_val, X_test, y_tr, y_val, y_test


def build_sequence_dataset(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Build a tf.data sequence dataset for LSTM training without OOM risk.

    Uses from_generator instead of pre-allocating a (N, T, 63) numpy array,
    which would be ~660MB for 87k samples (87k × 30 × 63 × 4 bytes).

    Each sample is tiled into a synthetic sequence of `seq_len` frames
    with small Gaussian jitter (std=0.005) to add temporal diversity
    when only static images (not video sequences) are available.
    """
    N = len(X)
    feature_dim = X.shape[1]
    num_classes = y.shape[1]

    def generator():
        indices = np.random.permutation(N) if shuffle else np.arange(N)
        for i in indices:
            base = X[i]
            seq = np.stack(
                [base + np.random.randn(feature_dim).astype(np.float32) * 0.005
                 for _ in range(seq_len)],
                axis=0,
            )  # (seq_len, feature_dim)
            yield seq, y[i]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,),         dtype=tf.float32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(N, 2000), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)



def build_image_dataset(
    split: str,
    cfg: CNNConfig,
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline for CNN training from image crops.

    Args:
        split: "train", "val", or "test"
        cfg:   CNNConfig
    """
    crops_dir = PROCESSED_DIR / "crops"
    if not crops_dir.exists():
        raise FileNotFoundError(f"Crops directory not found: {crops_dir}")

    class_names = sorted(
        [d.name for d in crops_dir.iterdir() if d.is_dir()],
    )

    # Bug fix: image_dataset_from_directory subset requires "training"/"validation"
    # Map friendly names to Keras names
    subset_map = {"train": "training", "val": "validation", "training": "training", "validation": "validation"}
    keras_subset = subset_map.get(split)
    use_validation_split = keras_subset in ("training", "validation")

    ds = tf.keras.utils.image_dataset_from_directory(
        crops_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(cfg.input_shape[0], cfg.input_shape[1]),
        batch_size=cfg.phase1_batch_size,
        shuffle=(split in ("train", "training")),
        seed=42,
        validation_split=0.2 if use_validation_split else None,
        subset=keras_subset if use_validation_split else None,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


# ── Callbacks ─────────────────────────────────────────────────────────────────

def make_callbacks(
    log_dir: Path,
    checkpoint_path: Path,
    early_stop_patience: int,
    lr_patience: int,
    lr_factor: float,
) -> list[keras.callbacks.Callback]:
    """Standard callback set: TensorBoard + checkpointing + early stop + LR decay."""
    return [
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            update_freq="epoch",
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
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
        keras.callbacks.CSVLogger(
            str(log_dir / "training_log.csv"),
            append=True,
        ),
    ]


# ── Phase 3A: Train MLP ───────────────────────────────────────────────────────

def train_mlp(cfg: MLPConfig | None = None) -> keras.Model:
    """Train the MLP landmark classifier and save best checkpoint."""
    cfg = cfg or MLPConfig()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(cfg.log_dir / "config.json")

    X_tr, X_val, X_test, y_tr, y_val, y_test = load_landmark_data()

    model = build_mlp(cfg)
    compile_model(
        model,
        learning_rate=cfg.learning_rate,
        label_smoothing=cfg.label_smoothing,
        optimizer_name=cfg.optimizer,
    )
    model.summary(print_fn=log.info)

    checkpoint_path = cfg.save_dir / f"{cfg.model_name}_best.keras"
    callbacks = make_callbacks(
        log_dir=cfg.log_dir,
        checkpoint_path=checkpoint_path,
        early_stop_patience=cfg.early_stop_patience,
        lr_patience=cfg.lr_patience,
        lr_factor=cfg.lr_decay_factor,
    )

    log.info("═" * 60)
    log.info("Training MLP — epochs=%d  batch=%d", cfg.epochs, cfg.batch_size)
    log.info("═" * 60)

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Final save + ONNX export
    final_path = cfg.save_dir / f"{cfg.model_name}.keras"
    save_model(model, final_path)
    export_onnx(model, cfg.save_dir / cfg.model_name)

    # Quick test evaluation
    test_loss, test_acc, test_top5 = model.evaluate(X_test, y_test, verbose=0)
    log.info("✅ MLP Test — loss=%.4f  acc=%.4f  top5=%.4f", test_loss, test_acc, test_top5)

    return model


# ── Phase 3B: Train LSTM ──────────────────────────────────────────────────────

def train_lstm(cfg: LSTMConfig | None = None) -> keras.Model:
    """Train the BiLSTM sequence classifier."""
    cfg = cfg or LSTMConfig()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(cfg.log_dir / "config.json")

    X_tr, X_val, X_test, y_tr, y_val, y_test = load_landmark_data()

    train_ds = build_sequence_dataset(X_tr, y_tr, cfg.sequence_len, cfg.batch_size)
    val_ds   = build_sequence_dataset(X_val, y_val, cfg.sequence_len, cfg.batch_size, shuffle=False)
    test_ds  = build_sequence_dataset(X_test, y_test, cfg.sequence_len, cfg.batch_size, shuffle=False)

    model = build_lstm(cfg)
    compile_model(
        model,
        learning_rate=cfg.learning_rate,
        label_smoothing=cfg.label_smoothing,
        optimizer_name=cfg.optimizer,
        gradient_clip=cfg.gradient_clip,
    )
    model.summary(print_fn=log.info)

    checkpoint_path = cfg.save_dir / f"{cfg.model_name}_best.keras"
    callbacks = make_callbacks(
        log_dir=cfg.log_dir,
        checkpoint_path=checkpoint_path,
        early_stop_patience=cfg.early_stop_patience,
        lr_patience=cfg.lr_patience,
        lr_factor=cfg.lr_decay_factor,
    )

    log.info("═" * 60)
    log.info("Training LSTM — epochs=%d  seq_len=%d", cfg.epochs, cfg.sequence_len)
    log.info("═" * 60)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = cfg.save_dir / f"{cfg.model_name}.keras"
    save_model(model, final_path)
    export_onnx(model, cfg.save_dir / cfg.model_name)

    test_loss, test_acc, test_top5 = model.evaluate(test_ds, verbose=0)
    log.info("✅ LSTM Test — loss=%.4f  acc=%.4f  top5=%.4f", test_loss, test_acc, test_top5)

    return model


# ── Phase 3C: Train CNN ───────────────────────────────────────────────────────

def train_cnn(cfg: CNNConfig | None = None) -> keras.Model | None:
    """
    Train MobileNetV3 in two phases:
        Phase 1 — Frozen base: train head only (fast convergence)
        Phase 2 — Fine-tune:  unfreeze top layers with very low LR

    Returns None if image crops are not available (graceful degradation).
    """
    cfg = cfg or CNNConfig()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(cfg.log_dir / "config.json")

    try:
        train_ds = build_image_dataset("training", cfg)
        val_ds   = build_image_dataset("validation", cfg)
    except FileNotFoundError as e:
        log.error("CNN training requires image crops: %s", e)
        log.error("Run: python src/preprocess.py --all")
        return None

    model = build_cnn(cfg)
    checkpoint_path = cfg.save_dir / f"{cfg.model_name}_best.keras"

    # ── Phase 1: Frozen base ───────────────────────────────────────────────
    log.info("═" * 60)
    log.info("CNN Phase 1 — Feature extraction (frozen base)")
    log.info("═" * 60)

    compile_model(
        model,
        learning_rate=cfg.phase1_lr,
        label_smoothing=cfg.label_smoothing,
        optimizer_name="adam",
    )

    callbacks_p1 = make_callbacks(
        log_dir=cfg.log_dir / "phase1",
        checkpoint_path=checkpoint_path,
        early_stop_patience=cfg.early_stop_patience,
        lr_patience=5,
        lr_factor=0.5,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.phase1_epochs,
        callbacks=callbacks_p1,
        verbose=1,
    )

    # ── Phase 2: Fine-tune ─────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("CNN Phase 2 — Fine-tuning (unfrozen from layer %d)", cfg.fine_tune_from)
    log.info("═" * 60)

    unfreeze_cnn_top(model, cfg.fine_tune_from)
    compile_model(
        model,
        learning_rate=cfg.phase2_lr,
        label_smoothing=cfg.label_smoothing,
        optimizer_name="adam",
    )

    callbacks_p2 = make_callbacks(
        log_dir=cfg.log_dir / "phase2",
        checkpoint_path=checkpoint_path,
        early_stop_patience=cfg.early_stop_patience,
        lr_patience=5,
        lr_factor=0.5,
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.phase2_epochs,
        callbacks=callbacks_p2,
        verbose=1,
    )

    final_path = cfg.save_dir / f"{cfg.model_name}.keras"
    save_model(model, final_path)
    export_onnx(model, cfg.save_dir / cfg.model_name)
    log.info("✅ CNN training complete.")

    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign Language Detection — Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/train.py --model mlp
  python src/train.py --model lstm
  python src/train.py --model cnn
  python src/train.py --model all
  python src/train.py --model mlp --epochs 50 --lr 5e-4
        """,
    )
    parser.add_argument(
        "--model", choices=["mlp", "lstm", "cnn", "all"], default="mlp",
        help="Which model to train (default: mlp)",
    )
    parser.add_argument("--epochs",     type=int,   default=None,  help="Override epoch count")
    parser.add_argument("--lr",         type=float, default=None,  help="Override learning rate")
    parser.add_argument("--batch_size", type=int,   default=None,  help="Override batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mlp_cfg  = MLPConfig()
    lstm_cfg = LSTMConfig()
    cnn_cfg  = CNNConfig()

    # Apply CLI overrides
    if args.epochs:
        mlp_cfg.epochs  = args.epochs
        lstm_cfg.epochs = args.epochs
        cnn_cfg.phase1_epochs = args.epochs // 3
        cnn_cfg.phase2_epochs = args.epochs - cnn_cfg.phase1_epochs
    if args.lr:
        mlp_cfg.learning_rate  = args.lr
        lstm_cfg.learning_rate = args.lr
    if args.batch_size:
        mlp_cfg.batch_size  = args.batch_size
        lstm_cfg.batch_size = args.batch_size

    targets = ["mlp", "lstm", "cnn"] if args.model == "all" else [args.model]

    for target in targets:
        log.info("━" * 60)
        log.info("  Starting: %s", target.upper())
        log.info("━" * 60)
        if target == "mlp":
            train_mlp(mlp_cfg)
        elif target == "lstm":
            train_lstm(lstm_cfg)
        elif target == "cnn":
            train_cnn(cnn_cfg)

    log.info("🎉 All training complete! Check backend/logs/ for TensorBoard.")
    log.info("   Run: tensorboard --logdir backend/logs/")


if __name__ == "__main__":
    main()
