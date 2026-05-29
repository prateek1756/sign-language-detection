"""
test_integration.py — End-to-end integration test.

Tests:
    - Send a real base64 frame to POST /predict/frame
    - Verify response has correct schema (letter/None, confidence, latency_ms)
    - Send frame with no hand → hand_detected=False
    - WebSocket: connect to /ws/stream, send frame, receive valid JSON response

Requires: backend running with at least the MLP model loaded.
Skip gracefully if models not available (503 response).

TODO (tomorrow): implement tests.
"""

# TODO (tomorrow): implement
# import pytest
#
# def test_predict_frame_schema(client, sample_frame):
#     resp = client.post("/predict/frame", json={"image": sample_frame})
#     if resp.status_code == 503:
#         pytest.skip("Inference engine not available — models not trained yet")
#     assert resp.status_code == 200
#     data = resp.json()
#     assert "confidence" in data
#     assert "latency_ms" in data
#     assert data["hand_detected"] is False  # black frame has no hand
#
# def test_websocket_stream(client, sample_frame):
#     with client.websocket_connect("/ws/stream") as ws:
#         ws.send_json({"image": sample_frame, "mode": "letter"})
#         data = ws.receive_json()
#         assert "confidence" in data or "error" in data
