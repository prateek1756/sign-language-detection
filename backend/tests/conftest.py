"""
conftest.py — Shared pytest fixtures for SignSense AI backend tests.

Fixtures:
    client          — FastAPI TestClient (no real models needed)
    sample_frame    — base64-encoded 640×480 black JPEG (valid image, no hand)
    sample_frame_with_hand — TODO: real frame with a hand for integration tests

TODO (tomorrow): implement fixtures.
"""

# TODO (tomorrow): implement
# import pytest
# from fastapi.testclient import TestClient
# from main import app
#
# @pytest.fixture(scope="session")
# def client():
#     with TestClient(app) as c:
#         yield c
#
# @pytest.fixture(scope="session")
# def sample_frame():
#     """640×480 black JPEG as base64 data URL — valid image, no hand detected."""
#     import numpy as np, cv2, base64
#     img = np.zeros((480, 640, 3), dtype=np.uint8)
#     _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 72])
#     b64 = base64.b64encode(buf).decode()
#     return f"data:image/jpeg;base64,{b64}"
