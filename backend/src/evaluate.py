"""
evaluate.py — Phase 3: Model Evaluation & Reporting
====================================================
Loads a saved model and runs a full evaluation:
  - Accuracy, Top-5 accuracy, Loss
  - Per-class accuracy table
  - Confusion matrix (saved as PNG)
  - Full sklearn classification report

Usage:
    python src/evaluate.py --model mlp
    python src/evaluate.py --model lstm
    python src/evaluate.py --model cnn
    python src/evaluate.py --model mlp --split val
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelBinarizer

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from configs.training_config import ASL_CLASSES, MODELS_DIR, NUM_CLASSES, PROCESSED_DIR
from src.model import load_model
from src.train import build_sequence_dataset, load_landmark_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_model_path(model_name: str) -> Path:
    # Use module-level MODELS_DIR so callers can patch it at runtime
    models_dir = MODELS_DIR
    path = models_dir / f"{model_name}.keras"
    if not path.exists():
        path = models_dir / f"{model_name}_best.keras"
    if not path.exists():
        cli_map = {"asl_mlp": "mlp", "asl_lstm": "lstm", "asl_mobilenet": "cnn"}
        cli_arg = cli_map.get(model_name, model_name)
        raise FileNotFoundError(
            f"No saved model found for '{model_name}' in {models_dir}.\n"
            f"Run: python src/train.py --model {cli_arg}"
        )
    return path


def _load_test_data(
    model_name: str, split: str = "test"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, y_int, y_ohe) for the requested split.

    Shapes:
        MLP  : X=(N, 63),       y_int=(N,), y_ohe=(N, 29)
        LSTM : X=(N, T, 63),    y_int=(N,), y_ohe=(N, 29)
        CNN  : X=(N, 63) — CNN evaluation loads landmark data; image crops
               would require a separate image loader (not yet implemented).

    Note (LSTM):
        Evaluation sequences are created by tiling a single frame T times
        (no jitter). This produces a slight optimistic bias because the model
        never sees 'perfect' static sequences during training (training uses
        jittered tiles). Acceptable for validation; real accuracy requires
        genuine video sequences.
    """
    X_tr, X_val, X_test, y_tr_ohe, y_val_ohe, y_test_ohe, _ = load_landmark_data()

    split_map = {
        "train": (X_tr, y_tr_ohe),
        "val":   (X_val, y_val_ohe),
        "test":  (X_test, y_test_ohe),
    }
    if split not in split_map:
        raise ValueError(f"Invalid split '{split}'. Choose from: {list(split_map)}.")

    X, y_ohe = split_map[split]
    y_int = np.argmax(y_ohe, axis=1)

    # LSTM needs sequences — tile single frame without jitter for deterministic eval
    if "lstm" in model_name:
        from configs.training_config import LSTMConfig
        cfg = LSTMConfig()
        X = np.stack(
            [np.stack([X[i]] * cfg.sequence_len, axis=0) for i in range(len(X))],
            axis=0,
        ).astype(np.float32)  # (N, T, 63) — built without Python loop over seq_len

    return X, y_int, y_ohe


# ── Core Evaluation ───────────────────────────────────────────────────────────

def evaluate(model_name: str, split: str = "test") -> dict:
    """
    Run full evaluation on a saved model.

    Returns:
        Dict with accuracy, top5, loss, and per-class results.
    """
    import tensorflow as tf

    model_path = _get_model_path(model_name)
    log.info("Loading model: %s", model_path)
    model = load_model(model_path)

    X, y_int, y_ohe = _load_test_data(model_name, split)
    log.info("Evaluating on %d samples (%s split)...", len(X), split)

    # Keras evaluation
    loss, acc, top5 = model.evaluate(X, y_ohe, batch_size=128, verbose=0)

    # Raw predictions
    y_pred_proba = model.predict(X, batch_size=128, verbose=0)  # (N, 29)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # scikit-learn top-5 (double check)
    sk_top5 = top_k_accuracy_score(y_int, y_pred_proba, k=5)

    results = {
        "model":    model_name,
        "split":    split,
        "samples":  len(X),
        "loss":     round(float(loss), 4),
        "accuracy": round(float(acc), 4),
        "top5":     round(float(sk_top5), 4),
    }

    # ── Console Summary ────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(f"  Model    : {model_name}")
    print(f"  Split    : {split}  ({len(X)} samples)")
    print(f"  Loss     : {loss:.4f}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Top-5    : {sk_top5:.4f}  ({sk_top5*100:.1f}%)")
    print("═" * 65)

    # ── Per-Class Accuracy Table ───────────────────────────────────────────
    print("\n  Per-Class Accuracy:")
    print(f"  {'Class':<10} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("  " + "-" * 38)
    for cls_idx, cls_name in enumerate(ASL_CLASSES):
        mask = y_int == cls_idx
        if mask.sum() == 0:
            continue
        cls_acc = (y_pred[mask] == cls_idx).mean()
        total   = mask.sum()
        correct = int(cls_acc * total)
        bar = "█" * int(cls_acc * 10)
        print(f"  {cls_name:<10} {correct:>8} {total:>8} {cls_acc:>7.1%}  {bar}")

    # ── Classification Report ──────────────────────────────────────────────
    report = classification_report(
        y_int, y_pred,
        target_names=[ASL_CLASSES[i] for i in sorted(set(y_int.tolist()))],
        zero_division=0,
    )
    print("\n  Classification Report:")
    print(report)

    # Save report to file
    report_path = REPORTS_DIR / f"{model_name}_{split}_report.txt"
    report_path.write_text(
        f"Model: {model_name}\nSplit: {split}\n"
        f"Loss: {loss:.4f}  Accuracy: {acc:.4f}  Top-5: {sk_top5:.4f}\n\n"
        + report
    )
    log.info("Report saved → %s", report_path)

    # ── Confusion Matrix ───────────────────────────────────────────────────
    _plot_confusion_matrix(y_int, y_pred, model_name, split)

    return results


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split: str,
) -> None:
    """Save a normalized confusion matrix heatmap as PNG."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")  # row-normalized

    actual_class_names = [ASL_CLASSES[i] for i in sorted(set(y_true.tolist()))]
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=actual_class_names,
        yticklabels=actual_class_names,
        linewidths=0.4,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title(
        f"{model_name.upper()} — Confusion Matrix ({split} split)",
        fontsize=14, pad=15,
    )
    plt.tight_layout()

    out_path = REPORTS_DIR / f"{model_name}_{split}_confusion.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix saved → %s", out_path)


# ── Benchmark: compare all saved models ──────────────────────────────────────

def benchmark_all(split: str = "test") -> None:
    """Compare all saved models side-by-side."""
    model_names = ["asl_mlp", "asl_lstm", "asl_mobilenet"]
    results = []

    for name in model_names:
        path = MODELS_DIR / f"{name}.keras"
        if not path.exists():
            log.warning("Skipping %s — not found.", name)
            continue
        try:
            r = evaluate(name, split)
            results.append(r)
        except Exception as e:
            log.error("Failed to evaluate %s: %s", name, e)

    if not results:
        log.warning("No models found to benchmark.")
        return

    print("\n" + "═" * 55)
    print("  BENCHMARK SUMMARY")
    print("═" * 55)
    print(f"  {'Model':<22} {'Accuracy':>10} {'Top-5':>10} {'Loss':>8}")
    print("  " + "-" * 55)
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(
            f"  {r['model']:<22} {r['accuracy']:>9.1%} "
            f"{r['top5']:>9.1%} {r['loss']:>8.4f}"
        )
    print("═" * 55 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sign Language Detection — Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/evaluate.py --model mlp
  python src/evaluate.py --model lstm --split val
  python src/evaluate.py --benchmark
        """,
    )
    parser.add_argument(
        "--model", choices=["mlp", "lstm", "cnn", "all"],
        default="mlp", help="Model to evaluate",
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"],
        default="test", help="Data split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Compare all saved models side-by-side",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.benchmark or args.model == "all":
        benchmark_all(args.split)
        return

    name_map = {"mlp": "asl_mlp", "lstm": "asl_lstm", "cnn": "asl_mobilenet"}
    model_name = name_map[args.model]
    evaluate(model_name, args.split)


if __name__ == "__main__":
    main()
