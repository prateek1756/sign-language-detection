"""
training_config.py — Phase 3: Typed Training Configuration
============================================================
Skills Applied:
  - python-pro     : Dataclasses, type hints, validated configs, serialization
  - ml-engineer    : Production training hyperparameters, mixed precision, LR schedules
  - mlops-engineer : Reproducible experiment configs with version stamps

All model training hyperparameters live here — one place to change, everything updates.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "ASL"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── ASL Class Registry ────────────────────────────────────────────────────────
ASL_CLASSES: list[str] = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "space", "del", "nothing",
]
NUM_CLASSES = len(ASL_CLASSES)   # 29
FEATURE_DIM = 63                 # 21 landmarks × 3 (x, y, z)
CROP_SIZE   = 224                # MobileNetV3 input size


# ── MLP Config ────────────────────────────────────────────────────────────────
@dataclass
class MLPConfig:
    """
    Configuration for the MLP landmark classifier.

    ml-engineer design rationale:
      - 3-layer MLP on 63 normalized landmarks — fast, CPU-friendly
      - BatchNorm + Dropout for regularization without data augmentation dependence
      - Label smoothing handles near-identical gesture pairs (e.g., M/N)
    """

    # Architecture
    input_dim:    int        = FEATURE_DIM        # 63
    hidden_dims:  list[int]  = field(default_factory=lambda: [512, 256, 128])
    num_classes:  int        = NUM_CLASSES         # 29
    dropout_rate: float      = 0.4
    use_batch_norm: bool     = True

    # Training
    epochs:            int   = 100
    batch_size:        int   = 64
    learning_rate:     float = 1e-3
    lr_decay_factor:   float = 0.5
    lr_patience:       int   = 8
    early_stop_patience: int = 15
    label_smoothing:   float = 0.1
    weight_decay:      float = 1e-4               # L2 regularization

    # Optimizer
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"

    # Mixed precision (ml-engineer: free ~30% speedup on GPU)
    mixed_precision: bool = False                  # Enable if NVIDIA GPU available

    # Output
    model_name:  str  = "asl_mlp"
    save_dir:    Path = field(default_factory=lambda: MODELS_DIR)
    log_dir:     Path = field(default_factory=lambda: LOGS_DIR / "mlp")

    def __post_init__(self) -> None:
        self.save_dir = Path(self.save_dir)
        self.log_dir  = Path(self.log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialize config to JSON for experiment tracking (mlops-engineer pattern)."""
        d = asdict(self)
        d["save_dir"] = str(self.save_dir)
        d["log_dir"]  = str(self.log_dir)
        d["timestamp"] = datetime.now().isoformat()
        s = json.dumps(d, indent=2)
        if path:
            Path(path).write_text(s)
        return s


# ── LSTM Config ───────────────────────────────────────────────────────────────
@dataclass
class LSTMConfig:
    """
    Configuration for the LSTM sequence classifier.

    ml-engineer design rationale:
      - 2-layer bidirectional LSTM captures temporal gesture dynamics
      - Sequence length of 30 frames = ~1 second at 30fps
      - Dropout between LSTM layers prevents co-adaptation
    """

    # Architecture
    input_dim:     int       = FEATURE_DIM         # 63 per frame
    sequence_len:  int       = 30                  # frames per gesture window
    lstm_units:    list[int] = field(default_factory=lambda: [128, 64])
    num_classes:   int       = NUM_CLASSES
    dropout_rate:  float     = 0.3
    bidirectional: bool      = True                # BiLSTM for better context

    # Training
    epochs:              int   = 80
    batch_size:          int   = 32
    learning_rate:       float = 5e-4
    lr_decay_factor:     float = 0.5
    lr_patience:         int   = 6
    early_stop_patience: int   = 12
    label_smoothing:     float = 0.05
    gradient_clip:       float = 1.0               # Prevent LSTM gradient explosion

    # Optimizer
    optimizer: Literal["adam", "adamw"] = "adam"
    mixed_precision: bool = False

    # Output
    model_name: str  = "asl_lstm"
    save_dir:   Path = field(default_factory=lambda: MODELS_DIR)
    log_dir:    Path = field(default_factory=lambda: LOGS_DIR / "lstm")

    def __post_init__(self) -> None:
        self.save_dir = Path(self.save_dir)
        self.log_dir  = Path(self.log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def to_json(self, path: Optional[Path] = None) -> str:
        d = asdict(self)
        d["save_dir"]  = str(self.save_dir)
        d["log_dir"]   = str(self.log_dir)
        d["timestamp"] = datetime.now().isoformat()
        s = json.dumps(d, indent=2)
        if path:
            Path(path).write_text(s)
        return s


# ── CNN Config ────────────────────────────────────────────────────────────────
@dataclass
class CNNConfig:
    """
    Configuration for the MobileNetV3 CNN fallback model.

    ml-engineer design rationale:
      - MobileNetV3Small: 2.5M params, fast on mobile/CPU
      - Transfer learning from ImageNet: fast convergence, ~90% accuracy
      - Two-phase training: freeze base → fine-tune top layers
      - Input: 224×224 hand crops from preprocess.py
    """

    # Architecture
    input_shape:      tuple[int, int, int] = (224, 224, 3)
    num_classes:      int                  = NUM_CLASSES
    base_model:       str                  = "MobileNetV3Small"
    fine_tune_from:   int                  = 80                 # Unfreeze from layer 80
    dropout_rate:     float                = 0.3
    pooling:          str                  = "avg"              # GlobalAveragePooling

    # Phase 1 — Feature extraction (frozen base)
    phase1_epochs:        int   = 20
    phase1_lr:            float = 1e-3
    phase1_batch_size:    int   = 32

    # Phase 2 — Fine-tuning (unfrozen top layers)
    phase2_epochs:        int   = 40
    phase2_lr:            float = 1e-5                          # Very low LR for fine-tune
    phase2_batch_size:    int   = 16

    # Shared training settings
    label_smoothing:      float = 0.1
    early_stop_patience:  int   = 10
    mixed_precision:      bool  = False

    # Output
    model_name: str  = "asl_mobilenet"
    save_dir:   Path = field(default_factory=lambda: MODELS_DIR)
    log_dir:    Path = field(default_factory=lambda: LOGS_DIR / "cnn")

    def __post_init__(self) -> None:
        self.save_dir = Path(self.save_dir)
        self.log_dir  = Path(self.log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def to_json(self, path: Optional[Path] = None) -> str:
        d = asdict(self)
        d["save_dir"]     = str(self.save_dir)
        d["log_dir"]      = str(self.log_dir)
        d["input_shape"]  = list(self.input_shape)
        d["timestamp"]    = datetime.now().isoformat()
        s = json.dumps(d, indent=2)
        if path:
            Path(path).write_text(s)
        return s
