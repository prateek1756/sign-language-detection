# Sign Language Detection System — Task Tracker

## Phase 1 — Environment & Project Scaffold
- [x] Create directory structure (backend/, web/, mobile/, notebooks/)
- [x] Write `backend/requirements.txt`
- [x] Create virtual environment setup script (`setup.ps1`)
- [x] Create `backend/main.py` (FastAPI entry point + CORS)
- [x] Create `backend/.env.example`
- [x] Create all `backend/src/` placeholder modules
- [x] Scaffold React web app (`web/`) via Vite
- [x] Add `.gitignore`
- [x] Add `README.md`
- [x] Add `LICENSE`
- [x] Git init + first commit (29 files) ✅

## Phase 2 — Data Collection & Preprocessing (ASL MVP)
- [x] Write `backend/src/collect_data.py` — OpenCV + MediaPipe webcam capture with HUD
- [x] Integrate MediaPipe Hands 21-landmark extraction
- [x] Write `backend/src/preprocess.py` — normalize + augment + crop pipeline
- [x] Write `backend/src/dataset_split.py` — stratified 70/15/15 split
- [x] Create ASL class directory structure (A-Z + space/del/nothing = 29 classes)
- [x] Write `backend/src/download_dataset.py` — Kaggle ASL dataset downloader
- [x] Write `notebooks/eda_analysis.py` — 5 EDA plots (distribution, PCA, correlation, etc.)
- [x] Git commit Phase 2 (5 files, 1873 insertions) ✅

## Phase 3 — Model Development ✅
- [x] Write `backend/src/model.py` — MLP + LSTM + MobileNetV3 architectures
- [x] Write `backend/src/train.py` — full training pipeline with checkpointing
- [x] Write `backend/src/evaluate.py` — confusion matrix + classification report
- [x] Write `backend/configs/training_config.py` — typed config dataclasses
- [x] TensorBoard logging + experiment tracking (CSVLogger + TensorBoard callbacks)
- [x] Model export (.keras + ONNX)
- [x] 3C: MobileNetV3 CNN fallback (build_cnn in model.py)
- [ ] 3D: ISL expansion — deferred to later phase

## Phase 4 — FastAPI Inference Backend ✅
- [x] `backend/src/inference.py` — InferenceEngine with GPU/CPU auto-detect
- [x] `backend/main.py` — all endpoints: /predict/frame, /predict/sequence, /health, /dialects
- [x] Confidence threshold (70%) + temporal smoothing (rolling majority vote)
- [x] Mode detector (static vs. dynamic gesture)
- [x] CORS + rate limiting (SlowAPI)
- [x] Bug audit: 5 bugs found and fixed (B1–B5)

## Phase 5 — Web App (React) ✅
- [x] Live camera feed component (CameraFeed.jsx — mirrored + MediaPipe overlay)
- [x] Frame → FastAPI WebSocket streaming (useWebSocket.js — exponential backoff)
- [x] Landmark overlay on canvas (21-point hand skeleton, glow effects)
- [x] Predicted label + confidence bar (colour-coded: emerald/violet/amber)
- [x] Accumulated text panel (SentenceBuilder + 1.5s auto-commit)
- [x] Mode toggle + dialect selector (ControlBar.jsx — ARIA-complete)
- [x] TTS via SpeechSynthesis API (Speak/Stop buttons)
- [x] Polished dark UI (glassmorphism + electric violet + Inter font)
- [x] Bug audit: B1 CORS, B2 GPU, B3 WS word mode, B4 race, B5 import — all fixed

### Phase 5 Quality Audit (concise-planning + lint-and-validate + systematic-debugging + kaizen) ✅
- [x] ESLint config fixed — eslint-plugin-react-hooks flat config API corrected (reactHooks.configs.flat.recommended → manual plugin + rules registration)
- [x] Lint: 3 errors fixed — `animRef` unused (CameraFeed), `isConnected` unused (ControlBar), `i` unused in top3.map (PredictionPanel)
- [x] B-WS1: useWebSocket intentional-disconnect bug — onclose was always scheduling reconnect even after user clicked Stop; fixed with `intentionalRef` flag
- [x] B-CAM1: CameraFeed canvas hardcoded 640×480 — landmark overlay misaligned at non-standard resolutions; fixed with loadedmetadata listener syncing canvas to actual video dimensions
- [x] B-CAM2: No camera loading state — blank area between Start click and first frame; fixed with `isLoading` state in useCamera (finally block), spinner in CameraFeed, wired through App.jsx
- [x] B-SB1: SentenceBuilder unstable index key — chars stored as [{id, char}] array with monotonic ID counter; prevents reconciliation bugs on backspace
- [x] Dead CSS removed — App.css cleared of ~120 lines of Vite scaffold boilerplate (conflicting --accent vs --clr-accent tokens)
- [x] @keyframes spin added to index.css for camera loading spinner
- [x] Vite dev proxy added — /ws and /api proxied to localhost:8000; eliminates CORS issues in development
- [x] ErrorBoundary.jsx added — wraps App in main.jsx; shows friendly fallback + Reload button instead of blank screen on unhandled render errors
- [x] Build gate: npm run build ✅ — 23 modules, 65.72 kB gzip, 491ms
- [x] Lint gate: npx eslint . ✅ — 0 errors, 0 warnings

## Phase 3D-train — Model Training (Colab)
- [x] Implement `notebooks/train_mlp.ipynb` — MLP training on Colab T4 GPU
- [x] Implement `notebooks/train_lstm.ipynb` — BiLSTM training on Colab T4 GPU
- [x] Implement `notebooks/train_cnn.ipynb` — MobileNetV3 fine-tune on Colab T4 GPU
- [ ] Implement `backend/src/download_models.py` — gdown from Google Drive at startup
- [ ] Wire `download_models_if_missing()` into `main.py` lifespan()
- [ ] **ACTION REQUIRED:** Run notebooks on Colab, save trained models to Google Drive
- [ ] Set GDRIVE_MODEL_URL_* env vars in .env

## Phase 6 — Mobile App (React Native)
- [ ] Camera setup (expo-camera)
- [ ] Shared FastAPI backend integration
- [ ] Prediction overlay UI
- [ ] Native TTS (react-native-tts)
- [ ] Android + iOS support

## Phase 7 — Testing & Validation
- [ ] Implement `backend/tests/conftest.py` — TestClient + sample_frame fixtures
- [ ] Implement `backend/tests/test_preprocess.py` — landmark normalization unit tests
- [ ] Implement `backend/tests/test_model_shapes.py` — MLP/LSTM/CNN shape validation
- [ ] Implement `backend/tests/test_api.py` — /health, /dialects, /predict/frame schema
- [ ] Implement `backend/tests/test_integration.py` — frame → /ws/stream → label
- [ ] Implement `backend/tests/test_performance.py` — latency < 100ms benchmark
- [ ] Run `pytest tests/ -v` — all green ✅

## Phase 8 — Documentation & Deployment
- [ ] Write `backend/Dockerfile` (python:3.11-slim, install deps, download models at build)
- [ ] Write `docker-compose.yml` (local smoke test)
- [ ] Local `docker build` + smoke test
- [ ] Deploy backend to Railway (set CORS_ORIGINS + GDRIVE_MODEL_URL_* env vars)
- [ ] Set `VITE_WS_URL=wss://your-app.railway.app/ws/stream` in web/.env.production
- [ ] Deploy web app to Vercel
- [ ] End-to-end smoke test on live URLs
- [ ] Update README.md with live URLs + demo instructions
- [ ] (Optional) App Store / Play Store
