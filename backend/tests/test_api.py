"""
test_api.py — API endpoint schema and contract tests.

Tests:
    GET  /health      → 200, status="ok", has uptime_s + models + device
    GET  /dialects    → 200, has "supported" list containing "ASL"
    POST /predict/frame  → 400 when image is too short (validation)
    POST /predict/frame  → 503 when engine not loaded
    POST /session/reset  → 200, status="reset"

TODO (tomorrow): implement tests.
"""

# TODO (tomorrow): implement
# import pytest
# from fastapi.testclient import TestClient
#
# def test_health_returns_200(client):
#     resp = client.get("/health")
#     assert resp.status_code == 200
#     data = resp.json()
#     assert data["status"] == "ok"
#     assert "uptime_s" in data
#     assert "models" in data
#     assert "device" in data
#
# def test_dialects_contains_asl(client):
#     resp = client.get("/dialects")
#     assert resp.status_code == 200
#     assert "ASL" in resp.json()["supported"]
#
# def test_predict_frame_rejects_short_image(client):
#     resp = client.post("/predict/frame", json={"image": "tooshort"})
#     assert resp.status_code == 422
#
# def test_session_reset(client):
#     resp = client.post("/session/reset")
#     assert resp.status_code in (200, 503)  # 503 if no engine
