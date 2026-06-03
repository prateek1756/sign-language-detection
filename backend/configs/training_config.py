"""
training_config.py — Training Configuration
============================================
All hyperparameters in one place. Edit here, everything updates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
MODELS_DIR    = BASE_DIR / "models"
LOGS_DIR      = BASE_DIR / "logs"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "ASL"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Class Registry ────────────────────────────────────────────────────────────
ASL_CLASSES: list[str] = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "space", "del", "nothing",
]
NUM_CLASSES = len(ASL_CLASSES)   # 29
FEATURE_DIM = 63                 # 21 landmarks × 3


# ── MLP ───────────────────────────────────────────────────────────────────────
@dataclass
class MLPConfig:
    input_dim:           int        = FEATURE_DIM
    hidden_dims:         list[int]  = field(default_factory=lambda: [512, 256, 128])
    num_classes:         int        = NUM_CLASSES
    dropout_rate:        float      = 0.4
    use_batch_norm:      bool       = True

    epochs:              int        = 100
    batch_size:          int        = 64
    learning_rate:       float      = 1e-3
    lr_decay_factor:     float      = 0.5
    lr_patience:         int        = 8
    early_stop_patience: int        = 15
    label_smoothing:     float      = 0.1

    model_name: str  = "asl_mlp"
    save_dir:   Path = field(default_factory=lambda: MODELS_DIR)
    log_dir:    Path = field(default_factory=lambda: LOGS_DIR / "mlp")

    def __post_init__(self) -> None:
        self.save_dir = Path(self.save_dir)
        self.log_dir  = Path(self.log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def to_json(self, path: Optional[Path] = None) -> str:
        d = asdict(self)
        d["save_dir"] = str(self.save_dir)
        d["log_dir"]  = str(self.log_dir)
        d["timestamp"] = datetime.now().isoformat()
        s = json.dumps(d, indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(s)
        return s


# ── LSTM ──────────────────────────────────────────────────────────────────────
@dataclass
class LSTMConfig:
    input_dim:           int        = FEATURE_DIM
    sequence_len:        int        = 30
    lstm_units:          list[int]  = field(default_factory=lambda: [128, 64])
    num_classes:         int        = NUM_CLASSES
    dropout_rate:        float      = 0.3
    bidirectional:       bool       = True

    epochs:              int        = 80
    batch_size:          int        = 32
    learning_rate:       float      = 5e-4
    lr_decay_factor:     float      = 0.5
    lr_patience:         int        = 6
    early_stop_patience: int        = 12
    label_smoothing:     float      = 0.05
    gradient_clip:       float      = 1.0

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
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(s)
        return s


# ── CNN ───────────────────────────────────────────────────────────────────────
@dataclass
class CNNConfig:
    input_shape:          tuple     = (224, 224, 3)
    num_classes:          int       = NUM_CLASSES
    base_model:           str       = "MobileNetV3Small"
    fine_tune_from:       int       = 80
    dropout_rate:         float     = 0.3

    phase1_epochs:        int       = 20
    phase1_lr:            float     = 1e-3
    phase1_batch_size:    int       = 32

    phase2_epochs:        int       = 40
    phase2_lr:            float     = 1e-5
    phase2_batch_size:    int       = 16

    label_smoothing:      float     = 0.1
    early_stop_patience:  int       = 10

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
        d["save_dir"]    = str(self.save_dir)
        d["log_dir"]     = str(self.log_dir)
        d["input_shape"] = list(self.input_shape)
        d["timestamp"]   = datetime.now().isoformat()
        s = json.dumps(d, indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(s)
        return s
