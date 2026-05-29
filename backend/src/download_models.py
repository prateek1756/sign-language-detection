"""
download_models.py — Download trained .keras models from Google Drive at startup.

Called from main.py lifespan() before engine.ensure_models_loaded().
Model URLs are read from environment variables so they can be set per-environment
(local dev, Railway, etc.) without hardcoding.

Environment variables (set in .env or Railway dashboard):
    GDRIVE_MODEL_URL_MLP      — shareable Drive link for asl_mlp.keras
    GDRIVE_MODEL_URL_LSTM     — shareable Drive link for asl_lstm.keras
    GDRIVE_MODEL_URL_CNN      — shareable Drive link for asl_mobilenet.keras

Usage:
    from src.download_models import download_models_if_missing
    download_models_if_missing()

TODO (tomorrow): implement download logic using gdown or requests.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# ── Model file paths ──────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_FILES = {
    "asl_mlp.keras":       "GDRIVE_MODEL_URL_MLP",
    "asl_lstm.keras":      "GDRIVE_MODEL_URL_LSTM",
    "asl_mobilenet.keras": "GDRIVE_MODEL_URL_CNN",
}


def download_models_if_missing() -> dict[str, bool]:
    """
    Download each model file from Google Drive if it doesn't exist locally.

    Returns:
        dict mapping filename → True (already existed or downloaded) / False (failed)

    TODO (tomorrow): implement with gdown:
        import gdown
        gdown.download(url, str(dest_path), quiet=False, fuzzy=True)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    for filename, env_var in MODEL_FILES.items():
        dest = MODELS_DIR / filename

        if dest.exists():
            log.info("✅ Model already present: %s", filename)
            results[filename] = True
            continue

        url = os.getenv(env_var)
        if not url:
            log.warning(
                "⚠️  %s not set — skipping download of %s. "
                "Set this env var to enable automatic model download.",
                env_var, filename,
            )
            results[filename] = False
            continue

        log.info("⬇️  Downloading %s from Drive...", filename)
        # TODO (tomorrow): implement download
        # try:
        #     import gdown
        #     gdown.download(url, str(dest), quiet=False, fuzzy=True)
        #     log.info("✅ Downloaded: %s", filename)
        #     results[filename] = True
        # except Exception as exc:
        #     log.error("❌ Failed to download %s: %s", filename, exc)
        #     results[filename] = False
        results[filename] = False

    return results
