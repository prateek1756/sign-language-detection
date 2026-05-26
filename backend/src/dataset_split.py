"""
dataset_split.py — Phase 2: Stratified Train/Val/Test Split
============================================================
Skills Applied:
  - ml-engineer    : Stratified splitting, reproducible seeds, tf.data-ready output
  - data-scientist : Class balance verification, split distribution analysis
  - python-pro     : Typed, documented, production-ready

Splits the processed landmark arrays into stratified train/val/test sets:
  - Train: 70%
  - Validation: 15%
  - Test: 15%

Usage:
    python src/dataset_split.py
    python src/dataset_split.py --seed 42 --val_ratio 0.15 --test_ratio 0.15
    python src/dataset_split.py --verify
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dataset_split")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "ASL"

ASL_CLASSES: list[str] = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "space", "del", "nothing",
]


# ── Split Config ──────────────────────────────────────────────────────────────
def run_split(
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """
    Perform stratified train/val/test split on preprocessed landmark data.

    ml-engineer best practices applied:
      - Stratified splitting: preserves class distribution in all splits
      - Fixed random seed: reproducible experiments
      - Separate val and test sets: val for tuning, test for final eval

    data-scientist pattern:
      - Verify class distributions after splitting
      - Report per-split sizes and class balance

    Args:
        val_ratio:  Fraction of total data for validation set.
        test_ratio: Fraction of total data for test set.
        seed:       Random seed for reproducibility.

    Returns:
        Dict with split statistics.
    """
    # Load processed arrays
    X_path = PROCESSED_DIR / "landmarks_all.npy"
    y_path = PROCESSED_DIR / "labels_all.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Processed data not found. Run preprocess.py first.\n"
            f"Expected: {X_path}\n         {y_path}"
        )

    X = np.load(str(X_path))
    y = np.load(str(y_path))
    log.info("Loaded dataset: X=%s y=%s", X.shape, y.shape)

    # ── Stage 1: Split off test set ───────────────────────────────────────────
    test_size_from_total = test_ratio
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size_from_total,
        random_state=seed,
        stratify=y,
    )

    # ── Stage 2: Split val from train+val ─────────────────────────────────────
    # val_ratio is relative to original; adjust for remaining data
    val_size_adjusted = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_adjusted,
        random_state=seed,
        stratify=y_trainval,
    )

    # ── Save splits ───────────────────────────────────────────────────────────
    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }
    for name, array in splits.items():
        path = PROCESSED_DIR / f"{name}.npy"
        np.save(str(path), array)
        log.info("  Saved %-12s shape=%s", name, array.shape)

    # ── Save split metadata ───────────────────────────────────────────────────
    metadata = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "total_samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "feature_dim": int(X.shape[1]),
        "num_classes": len(ASL_CLASSES),
        "class_names": ASL_CLASSES,
    }
    with open(PROCESSED_DIR / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Distribution Report ───────────────────────────────────────────────────
    _print_distribution(y_train, y_val, y_test)

    return {
        "train": {"samples": len(X_train)},
        "val": {"samples": len(X_val)},
        "test": {"samples": len(X_test)},
    }


def _print_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Print class distribution table for all splits.
    (data-scientist pattern: always verify split balance)
    """
    total_train = len(y_train)
    total_val = len(y_val)
    total_test = len(y_test)

    print("\n" + "═" * 72)
    print(f"  Dataset Split Summary")
    print(f"  {'Split':<10} {'Samples':>8} {'%':>6}")
    print("─" * 72)
    total = total_train + total_val + total_test
    for name, count in [("Train", total_train), ("Val", total_val), ("Test", total_test)]:
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<10} {count:>8,}  {pct:>5.1f}%  {bar}")
    print(f"  {'TOTAL':<10} {total:>8,}")
    print("═" * 72)

    # Per-class balance check
    print(f"\n  Per-Class Distribution (top 5 most imbalanced):")
    print(f"  {'Class':<10} {'Train':>7} {'Val':>7} {'Test':>7}")
    print("─" * 40)
    class_ratios = []
    for idx, label in enumerate(ASL_CLASSES):
        tr = int(np.sum(y_train == idx))
        va = int(np.sum(y_val == idx))
        te = int(np.sum(y_test == idx))
        if tr + va + te > 0:
            class_ratios.append((label, tr, va, te))

    # Sort by training count (ascending = most imbalanced first)
    class_ratios.sort(key=lambda x: x[1])
    for label, tr, va, te in class_ratios[:5]:
        print(f"  {label:<10} {tr:>7} {va:>7} {te:>7}")
    print("═" * 72 + "\n")


# ── Verification ──────────────────────────────────────────────────────────────
def verify_splits() -> None:
    """
    Verify that split files exist and are consistent.
    (ml-engineer: always validate pipeline outputs)
    """
    required = ["X_train.npy", "y_train.npy", "X_val.npy",
                "y_val.npy", "X_test.npy", "y_test.npy"]
    print("\n🔍 Verifying split files...")
    all_ok = True
    for fname in required:
        path = PROCESSED_DIR / fname
        if path.exists():
            arr = np.load(str(path))
            print(f"  ✅ {fname:<20} shape={arr.shape}  dtype={arr.dtype}")
        else:
            print(f"  ❌ MISSING: {fname}")
            all_ok = False

    if (PROCESSED_DIR / "split_metadata.json").exists():
        with open(PROCESSED_DIR / "split_metadata.json") as f:
            meta = json.load(f)
        print(f"\n  📋 Metadata: {meta['train_samples']} train | "
              f"{meta['val_samples']} val | {meta['test_samples']} test")
        print(f"  📋 Feature dim: {meta['feature_dim']} | "
              f"Classes: {meta['num_classes']}")

    print("\n  " + ("✅ All splits verified!" if all_ok else "❌ Some files missing. Run split first."))


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified Train/Val/Test Split for ASL Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/dataset_split.py
  python src/dataset_split.py --val_ratio 0.15 --test_ratio 0.15 --seed 42
  python src/dataset_split.py --verify
        """,
    )
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verify", action="store_true", help="Verify existing split files and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        verify_splits()
        return
    run_split(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
