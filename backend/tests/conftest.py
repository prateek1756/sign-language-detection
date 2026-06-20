"""
conftest.py — Shared pytest fixtures for Sign Language Detection backend tests.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path so modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import base64
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="session")
def client():
    """Session-scoped FastAPI TestClient fixture."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def sample_frame():
    """640x480 black JPEG as base64 data URL — valid image format with no hand detected."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    b64 = base64.b64encode(buf).decode()
    return f"data:image/jpeg;base64,{b64}"
