"""
test_performance.py — Latency benchmark tests.

Tests:
    - POST /predict/frame: mean latency < 100ms over 10 consecutive frames
    - WebSocket /ws/stream: round-trip latency < 100ms per frame

Target from implementation_plan.md:
    - Inference latency: < 100ms per frame
    - Web app detection rate: >= 10 FPS

Skip gracefully if models not available (503 response).

TODO (tomorrow): implement tests.
"""

# TODO (tomorrow): implement
# import time
# import pytest
#
# LATENCY_TARGET_MS = 100
# BENCHMARK_FRAMES  = 10
#
# def test_predict_frame_latency(client, sample_frame):
#     resp = client.post("/predict/frame", json={"image": sample_frame})
#     if resp.status_code == 503:
#         pytest.skip("Inference engine not available — models not trained yet")
#
#     latencies = []
#     for _ in range(BENCHMARK_FRAMES):
#         t0 = time.perf_counter()
#         r  = client.post("/predict/frame", json={"image": sample_frame})
#         latencies.append((time.perf_counter() - t0) * 1000)
#
#     mean_ms = sum(latencies) / len(latencies)
#     assert mean_ms < LATENCY_TARGET_MS, (
#         f"Mean latency {mean_ms:.1f}ms exceeds {LATENCY_TARGET_MS}ms target"
#     )
