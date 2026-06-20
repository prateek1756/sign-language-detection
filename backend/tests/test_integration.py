"""
test_integration.py — End-to-end integration tests.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


def test_predict_frame_schema(client, sample_frame):
    """Test schema and response of predict route on a black frame (no hand detected)."""
    resp = client.post("/predict/frame", json={"image": sample_frame})
    if resp.status_code == 503:
        pytest.skip("Inference engine not available — models not loaded/trained yet")

    assert resp.status_code == 200
    data = resp.json()
    assert "confidence" in data
    assert "latency_ms" in data
    assert "hand_detected" in data
    assert data["hand_detected"] is False  # A black frame contains no hand landmarks


def test_websocket_stream(client, sample_frame):
    """Test WebSocket connection, frame sending, and result reception."""
    try:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"image": sample_frame, "mode": "letter"})
            data = ws.receive_json()
            
            # If engine is not available, WebSocket sends an error and closes
            if "error" in data:
                assert "engine not available" in data["error"].lower()
            else:
                assert "latency_ms" in data
                assert data["hand_detected"] is False
    except Exception as exc:
        # If connection was refused or closed with 1011 (Service Unavailable)
        if "Inference engine not available" in str(exc) or "1011" in str(exc):
            pytest.skip("Inference engine not available on WebSocket — models not loaded/trained yet")
        else:
            raise exc
