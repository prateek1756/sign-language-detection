"""
main.py — Phase 4: FastAPI Inference Backend
=============================================
Endpoints:
  GET  /health                 — server + model status
  GET  /dialects               — supported sign language dialects
  POST /predict/frame          — single frame → letter prediction
  POST /predict/sequence       — frame sequence → word prediction
  POST /predict/frame/cnn      — CNN fallback prediction
  POST /session/reset          — reset temporal smoother state
  WS   /ws/stream              — real-time WebSocket streaming (low-latency)

Features:
  - Lifespan model loading (no cold-start on first request)
  - SlowAPI rate limiting (60 req/min default)
  - CORS for React web + React Native mobile
  - Pydantic v2 request/response schemas
  - Structured JSON error responses
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ── Lifespan: load models once at startup ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup sequence:
      1. Download any missing .keras models from Google Drive (if URLs configured)
      2. Load models into the InferenceEngine
      3. Attach engine to app.state for use by route handlers
    """
    log.info("🚀 Starting Sign Language Detection API...")

    # Step 1: Download missing models from Google Drive
    try:
        from src.download_models import download_models_if_missing
        download_results = download_models_if_missing()
        available = [f for f, ok in download_results.items() if ok]
        missing   = [f for f, ok in download_results.items() if not ok]
        if available:
            log.info("📦 Models available: %s", available)
        if missing:
            log.warning("⚠️  Models not available (predictions will return 503): %s", missing)
    except Exception as exc:
        log.warning("⚠️  Model download step failed (continuing without): %s", exc)

    # Step 2: Load models into InferenceEngine
    try:
        from src.inference import engine
        loaded = engine.ensure_models_loaded()
        log.info("Models loaded: %s", loaded)
        app.state.engine = engine
    except Exception as exc:
        log.warning("⚠️  Model loading failed (running without models): %s", exc)
        app.state.engine = None

    yield
    log.info("👋 Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sign Language Detection API",
    description=(
        "Real-time ASL sign language recognition using MediaPipe landmarks + "
        "TensorFlow MLP/LSTM/MobileNetV3 models.\n\n"
        "**WebSocket** `/ws/stream` provides the lowest-latency path for live camera streams."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# B1 FIX: Cannot mix allow_credentials=True with allow_origins=["*"].
# Browsers reject wildcard origin with credentials per CORS spec.
# List explicit dev origins; add your production domain here.
_CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:19006",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class FrameRequest(BaseModel):
    """Single base64-encoded frame for letter prediction."""
    image: str = Field(..., description="Base64-encoded JPEG/PNG frame (data-URL prefix optional)")
    dialect: str = Field(default="ASL", description="Sign language dialect (ASL / ISL)")

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str) -> str:
        if not v or len(v) < 100:
            raise ValueError("image field is too short to be a valid base64 image")
        return v


class SequenceRequest(BaseModel):
    """List of base64-encoded frames for word/phrase prediction."""
    frames: list[str] = Field(
        ..., min_length=5, max_length=60,
        description="Ordered list of base64 frames (5–60 frames)"
    )
    dialect: str = Field(default="ASL")


class PredictionResponse(BaseModel):
    letter: Optional[str]
    confidence: float
    top3: Optional[list[dict]] = None
    mode: Optional[str] = None
    landmarks: Optional[list[dict]] = None
    smoothed: Optional[bool] = None
    below_threshold: Optional[bool] = None
    hand_detected: Optional[bool] = None
    latency_ms: float


class WordResponse(BaseModel):
    word: Optional[str]
    confidence: float
    top3: Optional[list[dict]] = None
    frames_used: Optional[int] = None
    below_threshold: Optional[bool] = None
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_s: float
    models: dict
    device: str


# ── Startup time ──────────────────────────────────────────────────────────────
_START_TIME = time.time()


def _get_engine(request: Request):
    """Get the inference engine from app state, raising 503 if unavailable."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available. Models may not be trained yet.",
        )
    return engine


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
@limiter.limit("120/minute")
async def health(request: Request):
    """Health check — returns server status and loaded model inventory."""
    engine = getattr(request.app.state, "engine", None)
    if engine:
        engine_status = engine.status
    else:
        engine_status = {"device": "unknown", "models": {"mlp": False, "lstm": False, "cnn": False}}

    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_s=round(time.time() - _START_TIME, 1),
        models=engine_status.get("models", {}),
        device=engine_status.get("device", "unknown"),
    )


@app.get("/dialects", tags=["System"])
@limiter.limit("60/minute")
async def dialects(request: Request):
    """List supported sign language dialects and their model status."""
    return {
        "supported": ["ASL"],
        "coming_soon": ["ISL", "BSL", "PSL"],
        "current": "ASL",
        "note": "ISL multi-head expansion planned in Phase 3D.",
    }


@app.post("/predict/frame", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("120/minute")
async def predict_frame(request: Request, body: FrameRequest):
    """
    Predict ASL letter from a single webcam frame.

    - Extracts MediaPipe hand landmarks (21 × 3D points)
    - Runs MLP classifier on normalized 63-dim feature vector
    - Applies 70% confidence threshold
    - Returns smoothed prediction via rolling majority vote (last 10 frames)
    """
    engine = _get_engine(request)
    result = await asyncio.to_thread(engine.predict_frame, body.image)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return PredictionResponse(
        letter=result.get("letter"),
        confidence=result.get("confidence", 0.0),
        top3=result.get("top3"),
        mode=result.get("mode"),
        landmarks=result.get("landmarks", []),
        smoothed=result.get("smoothed"),
        below_threshold=result.get("below_threshold"),
        hand_detected=result.get("hand_detected", True),
        latency_ms=result.get("latency_ms", 0.0),
    )


@app.post("/predict/sequence", response_model=WordResponse, tags=["Prediction"])
@limiter.limit("30/minute")
async def predict_sequence(request: Request, body: SequenceRequest):
    """
    Predict ASL word/phrase from a sequence of consecutive frames.

    - Extracts landmarks from each frame
    - Pads/truncates to 30-frame window
    - Runs BiLSTM classifier on temporal sequence
    - Returns word prediction with confidence
    """
    engine = _get_engine(request)
    result = await asyncio.to_thread(engine.predict_sequence, body.frames)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return WordResponse(
        word=result.get("word"),
        confidence=result.get("confidence", 0.0),
        top3=result.get("top3"),
        frames_used=result.get("frames_used"),
        below_threshold=result.get("below_threshold"),
        latency_ms=result.get("latency_ms", 0.0),
    )


@app.post("/predict/frame/cnn", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("60/minute")
async def predict_frame_cnn(request: Request, body: FrameRequest):
    """
    CNN fallback prediction using MobileNetV3 on raw image crops.

    Use when MediaPipe landmark confidence is low (e.g., bad lighting, 
    partial hand occlusion). Slower than MLP but more robust.
    """
    engine = _get_engine(request)
    result = await asyncio.to_thread(engine.predict_frame_cnn, body.image)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return PredictionResponse(
        letter=result.get("letter"),
        confidence=result.get("confidence", 0.0),
        latency_ms=result.get("latency_ms", 0.0),
    )


@app.post("/session/reset", tags=["Session"])
@limiter.limit("30/minute")
async def session_reset(request: Request):
    """
    Reset the temporal smoother and mode detector state.

    Call this when:
    - Starting a new signing session
    - Switching between Letter / Word modes
    - After a long pause in signing
    """
    engine = _get_engine(request)
    engine.reset_state()
    return {"status": "reset", "message": "Temporal smoother and mode detector cleared."}


@app.get("/model/status", tags=["System"])
@limiter.limit("60/minute")
async def model_status(request: Request):
    """Detailed model and engine configuration status."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return {"available": False, "reason": "Engine not initialized"}
    return {"available": True, **engine.status}


# ── WebSocket: Real-Time Streaming ────────────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sign language detection.

    Protocol:
        Client → Server: JSON { "image": "<base64>", "mode": "letter"|"word" }
        Server → Client: JSON { "letter": "A", "confidence": 0.94, ... }

    Advantages over HTTP polling:
        - Persistent connection (no HTTP handshake per frame)
        - ~50ms latency vs ~100ms for HTTP
        - Server can push notifications (e.g., "hand lost")

    Example JS client:
        const ws = new WebSocket("ws://localhost:8000/ws/stream");
        ws.onmessage = (e) => console.log(JSON.parse(e.data));
        ws.send(JSON.stringify({ image: canvas.toDataURL("image/jpeg", 0.7) }));
    """
    await websocket.accept()
    log.info("🔌 WebSocket connected: %s", websocket.client)

    engine = getattr(websocket.app.state, "engine", None)
    if engine is None:
        await websocket.send_json({"error": "Inference engine not available."})
        await websocket.close(code=1011)
        return

    # Per-connection frame buffer for LSTM word mode
    # B3 FIX: word mode must accumulate frames then call predict_sequence (LSTM)
    word_frame_buffer: list[str] = []

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON payload"})
                continue

            b64_image = payload.get("image", "")
            mode      = payload.get("mode", "letter")

            if not b64_image:
                await websocket.send_json({"error": "Missing 'image' field"})
                continue

            if mode == "letter":
                word_frame_buffer.clear()  # reset buffer when switching modes
                result = await asyncio.to_thread(engine.predict_frame, b64_image)
            else:
                # Word mode:
                frame_res = await asyncio.to_thread(engine.predict_frame, b64_image)
                if frame_res.get("gesture_ended"):
                    # Hand slowed down -> Trigger sequence prediction on collected buffer
                    result = await asyncio.to_thread(engine.predict_sequence, word_frame_buffer[:])
                    word_frame_buffer.clear()
                    result["gesture_ended"] = True
                elif frame_res.get("mode") == "word":
                    # Hand is moving -> Keep buffering
                    word_frame_buffer.append(b64_image)
                    if len(word_frame_buffer) > engine._seq_len:
                        word_frame_buffer.pop(0)
                    result = {
                        "buffering": True,
                        "frames_buffered": len(word_frame_buffer),
                        "mode": "word",
                        "latency_ms": frame_res.get("latency_ms", 0.0),
                    }
                else:
                    # Hand is static in word mode -> Return frame prediction
                    result = frame_res

            if "mode" not in result:
                result["mode"] = mode

            await websocket.send_json(result)

    except WebSocketDisconnect:
        log.info("🔌 WebSocket disconnected: %s", websocket.client)
        if engine:
            engine.reset_state()
    except Exception as exc:
        log.error("WebSocket error: %s", exc)
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass


# ── Global Exception Handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Dev Server Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
