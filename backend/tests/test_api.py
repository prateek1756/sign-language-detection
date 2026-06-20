"""
test_api.py — API endpoint schema and contract tests.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient


def test_health_returns_200(client):
    """Test health check route returns valid JSON and has correct keys."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "uptime_s" in data
    assert "models" in data
    assert "device" in data


def test_dialects_contains_asl(client):
    """Test dialects list contains ASL dialect."""
    resp = client.get("/dialects")
    assert resp.status_code == 200
    data = resp.json()
    assert "ASL" in data["supported"]


def test_predict_frame_rejects_short_image(client):
    """Test validation reject on short/malformed base64 image strings."""
    resp = client.post("/predict/frame", json={"image": "tooshort"})
    assert resp.status_code == 422


def test_session_reset(client):
    """Test session reset route returns 200 (or 503 if inference engine is not ready)."""
    resp = client.post("/session/reset")
    assert resp.status_code in (200, 503)
