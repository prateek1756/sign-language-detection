"""
test_performance.py — Latency benchmark tests.
"""

import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import pytest

LATENCY_TARGET_MS = 100
BENCHMARK_FRAMES = 10


def test_predict_frame_latency(client, sample_frame):
    """Benchmark the average latency of the /predict/frame endpoint over multiple calls."""
    resp = client.post("/predict/frame", json={"image": sample_frame})
    if resp.status_code == 503:
        pytest.skip("Inference engine not available — models not loaded/trained yet")

    latencies = []
    for _ in range(BENCHMARK_FRAMES):
        t0 = time.perf_counter()
        r = client.post("/predict/frame", json={"image": sample_frame})
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(latencies) / len(latencies)
    assert mean_ms < LATENCY_TARGET_MS, (
        f"Mean latency {mean_ms:.1f}ms exceeds target of {LATENCY_TARGET_MS}ms"
    )
    print(f"\n  [Performance] Mean latency over {BENCHMARK_FRAMES} frames: {mean_ms:.2f}ms")
