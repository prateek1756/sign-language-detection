# FastAPI Backend Entry Point
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Sign Language Detection API",
    description="Real-time sign language recognition using MediaPipe + TensorFlow",
    version="0.1.0",
)

# CORS — allow React web app and React Native mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/dialects")
def dialects():
    return {"supported": ["ASL"], "coming_soon": ["ISL", "BSL"]}


# Routers will be added in Phase 4
# from src.routes import predict
# app.include_router(predict.router, prefix="/predict")
