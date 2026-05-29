"""
download_models.py — Download trained .keras models from Google Drive at startup.

Called from main.py lifespan() before engine.ensure_models_loaded().
Model URLs are read from environment variables so they can be set per-environment
(local dev, Railway, etc.) without hardcoding.

Environment variables (set in .env or Railway dashboard):
    GDRIVE_MODEL_URL_MLP      — shareable Drive link for asl_mlp.keras
    GDRIVE_MODEL_URL_LSTM     — shareable Drive link for asl_lstm.keras
    GDRIVE_MODEL_URL_CNN      — shareable Drive link for asl_mobilenet.keras

How to get a shareable Drive link:
    1. Right-click the .keras file in Google Drive
    2. Share → Anyone with the link → Copy link
    3. Paste the full URL as the env var value
    Example: https://drive.google.com/file/d/1ABC.../view?usp=sharing
    gdown handles both /view and /uc?id= formats via fuzzy=True.

Usage:
    from src.download_models import download_models_if_missing
    results = download_models_if_missing()
    # results = {"asl_mlp.keras": True, "asl_lstm.keras": False, ...}
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# ── Model file paths ──────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Maps filename → environment variable name
MODEL_FILES: dict[str, str] = {
    "asl_mlp.keras":       "GDRIVE_MODEL_URL_MLP",
    "asl_lstm.keras":      "GDRIVE_MODEL_URL_LSTM",
    "asl_mobilenet.keras": "GDRIVE_MODEL_URL_CNN",
}


def _ensure_gdown() -> bool:
    """
    Check gdown is installed. Attempt pip install if missing.
    Returns True if gdown is available after the check.
    """
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        log.warning("gdown not installed — attempting: pip install gdown")
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "gdown", "--quiet"],
            capture_output=True,
        )
        if result.returncode == 0:
            log.info("gdown installed successfully.")
            return True
        log.error("Failed to install gdown: %s", result.stderr.decode())
        return False


def download_models_if_missing() -> dict[str, bool]:
    """
    Download each model file from Google Drive if it doesn't exist locally.

    Skips files that already exist (idempotent — safe to call on every startup).
    Skips files whose env var is not set (graceful degradation — backend still
    starts, just returns 503 for prediction endpoints until models are present).

    Returns:
        dict mapping filename → True (present or downloaded) / False (missing/failed)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}
    needs_download = False

    # First pass: check what's missing
    for filename, env_var in MODEL_FILES.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            log.info("✅ Model already present: %s  (%.1f MB)", filename, dest.stat().st_size / 1e6)
            results[filename] = True
        elif os.getenv(env_var):
            needs_download = True
            results[filename] = False
        else:
            log.warning(
                "⚠️  %s not set — %s will not be downloaded. "
                "Add this env var to enable automatic model download.",
                env_var, filename,
            )
            results[filename] = False

    if not needs_download:
        return results

    # Ensure gdown is available before attempting downloads
    if not _ensure_gdown():
        log.error("Cannot download models: gdown unavailable.")
        return results

    import gdown

    # Second pass: download missing files
    for filename, env_var in MODEL_FILES.items():
        if results[filename]:
            continue  # already present

        url = os.getenv(env_var)
        if not url:
            continue  # no URL configured

        dest = MODELS_DIR / filename
        log.info("⬇️  Downloading %s from Google Drive...", filename)

        try:
            # fuzzy=True handles both /view and /uc?id= URL formats
            output = gdown.download(url, str(dest), quiet=False, fuzzy=True)
            if output and dest.exists():
                size_mb = dest.stat().st_size / 1e6
                log.info("✅ Downloaded: %s  (%.1f MB)", filename, size_mb)
                results[filename] = True
            else:
                log.error(
                    "❌ Download returned no output for %s. "
                    "Check that the Drive link is publicly accessible.",
                    filename,
                )
                results[filename] = False
        except Exception as exc:
            log.error("❌ Failed to download %s: %s", filename, exc)
            # Clean up partial download
            if dest.exists() and dest.stat().st_size < 1000:
                dest.unlink(missing_ok=True)
            results[filename] = False

    # Summary
    downloaded = sum(1 for v in results.values() if v)
    total = len(results)
    log.info("Model download summary: %d/%d available", downloaded, total)

    return results
