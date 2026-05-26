"""
eda_analysis.py — Phase 2: Exploratory Data Analysis
=====================================================
Skills Applied:
  - data-scientist : EDA patterns — distributions, correlations, visualizations
  - python-pro     : Clean, typed, self-contained analysis script
  - ml-engineer    : Data quality checks before training

Generates visual reports on the ASL dataset:
  1. Class distribution histogram
  2. Sample images per class (grid)
  3. Landmark scatter plots
  4. Feature correlation heatmap
  5. PCA 2D projection of landmark space
  6. Dataset completeness report

Usage:
    python notebooks/eda_analysis.py
    python notebooks/eda_analysis.py --output_dir notebooks/eda_outputs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eda_analysis")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "backend" / "data" / "processed" / "ASL"
RAW_DIR = BASE_DIR / "backend" / "data" / "raw" / "ASL"
DEFAULT_OUTPUT = Path(__file__).parent / "eda_outputs"

ASL_CLASSES: list[str] = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "space", "del", "nothing",
]

# ── Plot Style ────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
COLORS = plt.cm.Set2(np.linspace(0, 1, 29))


# ── 1. Class Distribution ─────────────────────────────────────────────────────
def plot_class_distribution(output_dir: Path) -> None:
    """
    Bar chart of sample counts per ASL class in raw data.
    (data-scientist: always check class balance before training)
    """
    counts = []
    for label in ASL_CLASSES:
        n = len(list((RAW_DIR / label).glob("*.jpg")))
        counts.append(n)

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(ASL_CLASSES, counts, color=COLORS, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title("ASL Dataset — Class Distribution", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Gesture Class", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.axhline(y=np.mean(counts), color="orange", linestyle="--", alpha=0.8, label=f"Mean: {np.mean(counts):.0f}")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=7, color="white")

    ax.legend(fontsize=11)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")
    plt.tight_layout()

    out = output_dir / "01_class_distribution.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✅ Saved: %s", out.name)


# ── 2. Sample Image Grid ──────────────────────────────────────────────────────
def plot_sample_images(output_dir: Path, n_per_class: int = 3) -> None:
    """
    Grid of sample images for each ASL class.
    (data-scientist: visual inspection of raw data quality)
    """
    n_classes = len(ASL_CLASSES)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 2.5, n_classes * 2.5))
    fig.suptitle("ASL Dataset — Sample Images per Class", fontsize=16, fontweight="bold", y=1.002)
    fig.patch.set_facecolor("#0f0f1a")

    for row, label in enumerate(ASL_CLASSES):
        images = sorted((RAW_DIR / label).glob("*.jpg"))[:n_per_class]
        for col in range(n_per_class):
            ax = axes[row][col] if n_per_class > 1 else axes[row]
            ax.axis("off")
            if col < len(images):
                img = cv2.imread(str(images[col]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (100, 100))
                    ax.imshow(img)
            if col == 0:
                ax.set_ylabel(label, fontsize=8, color="white", labelpad=3)

    plt.tight_layout()
    out = output_dir / "02_sample_images.png"
    plt.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close()
    log.info("  ✅ Saved: %s", out.name)


# ── 3. Landmark Distributions ─────────────────────────────────────────────────
def plot_landmark_distributions(output_dir: Path) -> None:
    """
    Distribution of normalized landmark x/y values across all samples.
    (data-scientist: verify normalization effectiveness)
    """
    X_path = PROCESSED_DIR / "landmarks_all.npy"
    if not X_path.exists():
        log.warning("landmarks_all.npy not found. Run preprocess.py first.")
        return

    X = np.load(str(X_path))
    landmarks = X.reshape(-1, 21, 3)  # (N, 21, 3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("Normalized Landmark Distributions (x, y, z)", fontsize=15, fontweight="bold")

    dim_names = ["X (horizontal)", "Y (vertical)", "Z (depth)"]
    dim_colors = ["#00d4ff", "#ff6b6b", "#51cf66"]

    for dim, (name, color) in enumerate(zip(dim_names, dim_colors)):
        ax = axes[dim]
        values = landmarks[:, :, dim].flatten()
        ax.hist(values, bins=60, color=color, alpha=0.8, edgecolor="none")
        ax.set_title(f"Landmark {name}", fontsize=12)
        ax.set_xlabel("Normalized Value", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_facecolor("#1a1a2e")
        ax.axvline(x=0, color="white", linestyle="--", alpha=0.5, label="Origin (wrist)")
        ax.legend(fontsize=9)
        ax.text(0.98, 0.95, f"μ={values.mean():.3f}\nσ={values.std():.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round", facecolor="#333", alpha=0.6))

    plt.tight_layout()
    out = output_dir / "03_landmark_distributions.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✅ Saved: %s", out.name)


# ── 4. Feature Correlation Heatmap ────────────────────────────────────────────
def plot_feature_correlation(output_dir: Path, n_features: int = 21) -> None:
    """
    Correlation heatmap of landmark x-coordinates across classes.
    (data-scientist: understand feature redundancy before model design)
    """
    X_path = PROCESSED_DIR / "landmarks_all.npy"
    if not X_path.exists():
        log.warning("landmarks_all.npy not found. Skipping correlation plot.")
        return

    X = np.load(str(X_path))
    # Use only x,y landmarks (first 42 features) for readability
    X_xy = X[:, :42:3]  # x coords of 21 landmarks
    labels = [f"LM{i}_x" for i in range(21)]

    # Sub-sample for speed
    if len(X_xy) > 5000:
        idx = np.random.choice(len(X_xy), 5000, replace=False)
        X_xy = X_xy[idx]

    corr = np.corrcoef(X_xy.T)

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#0f0f1a")
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(21))
    ax.set_yticks(range(21))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Landmark X-Coordinate Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.colorbar(im, ax=ax, label="Pearson Correlation")
    ax.set_facecolor("#1a1a2e")
    plt.tight_layout()

    out = output_dir / "04_feature_correlation.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✅ Saved: %s", out.name)


# ── 5. PCA Projection ─────────────────────────────────────────────────────────
def plot_pca_projection(output_dir: Path, max_samples: int = 3000) -> None:
    """
    2D PCA projection of the 63-dimensional landmark space colored by class.
    (data-scientist: assess class separability before model training)
    """
    X_path = PROCESSED_DIR / "landmarks_all.npy"
    y_path = PROCESSED_DIR / "labels_all.npy"
    if not X_path.exists() or not y_path.exists():
        log.warning("Processed data not found. Skipping PCA plot.")
        return

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.warning("scikit-learn not installed. Skipping PCA plot.")
        return

    X = np.load(str(X_path))
    y = np.load(str(y_path))

    # Sub-sample for speed
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    cmap = plt.cm.get_cmap("tab20", len(ASL_CLASSES))
    for i, label in enumerate(ASL_CLASSES):
        mask = y == i
        if mask.sum() > 0:
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                c=[cmap(i)], label=label, alpha=0.6, s=10, edgecolors="none"
            )

    ax.set_title("PCA 2D Projection — ASL Landmark Space", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.legend(
        loc="upper right", fontsize=7, markerscale=2,
        ncol=3, framealpha=0.3,
        facecolor="#0f0f1a", edgecolor="white"
    )
    plt.tight_layout()

    out = output_dir / "05_pca_projection.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  ✅ Saved: %s", out.name)


# ── 6. Dataset Summary Report ─────────────────────────────────────────────────
def print_summary_report() -> None:
    """
    Print a comprehensive text-based dataset summary.
    (data-scientist: always document dataset characteristics)
    """
    print("\n" + "═" * 65)
    print("  ASL Dataset — EDA Summary Report")
    print("═" * 65)

    # Raw data
    print("\n  📁 RAW DATA (per class):")
    total_raw = 0
    for label in ASL_CLASSES:
        n = len(list((RAW_DIR / label).glob("*.jpg")))
        total_raw += n
        status = "✅" if n > 0 else "⚠️ "
        print(f"    {status} {label:<8} {n:>6} images")
    print(f"\n  Total raw images: {total_raw:,}")

    # Processed data
    X_path = PROCESSED_DIR / "landmarks_all.npy"
    y_path = PROCESSED_DIR / "labels_all.npy"
    if X_path.exists() and y_path.exists():
        X = np.load(str(X_path))
        y = np.load(str(y_path))
        print(f"\n  📊 PROCESSED DATA:")
        print(f"    Feature matrix X: {X.shape}  dtype={X.dtype}")
        print(f"    Labels y:         {y.shape}  dtype={y.dtype}")
        print(f"    Feature dim:      63 (21 landmarks × 3 coords)")
        print(f"    Classes:          {len(np.unique(y))}")
        print(f"\n  📈 FEATURE STATISTICS:")
        print(f"    X mean:   {X.mean():.4f}")
        print(f"    X std:    {X.std():.4f}")
        print(f"    X range:  [{X.min():.4f}, {X.max():.4f}]")
        print(f"    NaN:      {np.isnan(X).sum()} (should be 0)")
        print(f"    Inf:      {np.isinf(X).sum()} (should be 0)")

    # Split info
    meta_path = PROCESSED_DIR / "split_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n  ✂️  SPLITS:")
        print(f"    Train:  {meta['train_samples']:>6,} samples  ({meta['train_samples']/meta['total_samples']*100:.1f}%)")
        print(f"    Val:    {meta['val_samples']:>6,} samples  ({meta['val_samples']/meta['total_samples']*100:.1f}%)")
        print(f"    Test:   {meta['test_samples']:>6,} samples  ({meta['test_samples']/meta['total_samples']*100:.1f}%)")

    print("\n" + "═" * 65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_eda(output_dir: Path) -> None:
    """Run the full EDA pipeline and generate all plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("═" * 55)
    log.info("  Starting EDA Analysis")
    log.info("  Output: %s", output_dir)
    log.info("═" * 55)

    print_summary_report()

    log.info("Generating plots...")
    plot_class_distribution(output_dir)
    plot_sample_images(output_dir)
    plot_landmark_distributions(output_dir)
    plot_feature_correlation(output_dir)
    plot_pca_projection(output_dir)

    log.info("═" * 55)
    log.info("  ✅ EDA complete. %d plots saved to: %s", 5, output_dir)
    log.info("═" * 55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASL Dataset EDA Analysis")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output directory for plots (default: {DEFAULT_OUTPUT})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda(args.output_dir)
