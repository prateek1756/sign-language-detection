# Sign Language Detection System — Implementation Plan (v2)

A real-time sign language recognition system supporting **ASL → ISL** dialect expansion,
with an adaptive 3-mode output pipeline, served via a **React + FastAPI** web app and
**React Native** mobile app sharing one inference API.

---

## ✅ Design Decisions (Locked In)

| Question | Decision |
|---|---|
| **Sign language dialect** | Start with **ASL** (MVP). Add **ISL** in Phase 3 via multi-head architecture. Expand to others later. |
| **Output target** | **Adaptive 3-mode pipeline**: static gesture → Letter, gesture sequence → Word, accumulated words → Sentence (LLM-assisted grammar) |
| **Platform** | **React** (web) + **React Native** (mobile), sharing one **FastAPI** backend |
| **Model approach** | **Primary**: MediaPipe 21-landmark → MLP/LSTM (fast, CPU-friendly). **Secondary**: MobileNetV3 CNN on image crops (robustness in bad lighting) |
| **Hardware** | **Natural load distribution**: MediaPipe on CPU, model inference on GPU with CPU fallback via TF/PyTorch device selection |

---

## System Architecture

```
Webcam / Mobile Camera
        ↓
MediaPipe Hands (CPU) — 21 landmarks per hand
        ↓
Mode Detector (static vs. dynamic gesture sequence)
        ↓
┌─────────────────┬──────────────────────┐
│   MLP Model     │   LSTM/GRU Model     │
│  (A–Z alphabet) │  (words / phrases)   │
└─────────────────┴──────────────────────┘
        ↓  (GPU if available, CPU fallback)
  FastAPI Inference API
        ↓
┌─────────────────┬──────────────────────┐
│  React Web App  │  React Native App    │
│  (browser cam)  │  (native camera)     │
└─────────────────┴──────────────────────┘
        ↓
TTS Output + Text Display Panel
```

---

## Phase 1 — Environment & Project Scaffold

- `[x]` Create directory structure:
  ```
  sign-language-detection/
  ├── backend/          # FastAPI server + ML models
  │   ├── src/
  │   │   ├── collect_data.py
  │   │   ├── preprocess.py
  │   │   ├── model.py
  │   │   ├── train.py
  │   │   └── inference.py
  │   ├── models/       # saved .h5 / .pt files
  │   ├── data/
  │   │   ├── raw/
  │   │   └── processed/
  │   ├── tests/
  │   ├── main.py       # FastAPI entry point
  │   └── requirements.txt
  ├── web/              # React web app
  ├── mobile/           # React Native app
  ├── notebooks/        # EDA and training experiments
  └── README.md
  ```
- `[x]` Set up `backend/requirements.txt`: `fastapi`, `uvicorn`, `opencv-python`, `mediapipe`, `tensorflow` / `torch`, `numpy`, `scikit-learn`, `pyttsx3`, `gTTS`
- `[x]` Set up Python virtual environment (`venv` or `conda`)
- `[x]` Scaffold React web app (`npx create-react-app web/` or Vite)
- `[x]` Scaffold React Native app (`npx react-native init mobile/`)
- `[x]` Add `.gitignore`, `README.md`, `LICENSE`

---

## Phase 2 — Data Collection & Preprocessing (ASL MVP)

- `[x]` Write `src/collect_data.py` — OpenCV webcam capture
  - `[x]` Capture N frames per gesture class (target: 200–500 images/class)
  - `[x]` Save raw frames to `data/raw/ASL/<class_label>/`
- `[x]` Integrate **MediaPipe Hands** to extract 21 3D landmarks per frame
- `[x]` Write `src/preprocess.py`
  - `[x]` Normalize landmarks relative to wrist anchor point (translation invariance)
  - `[x]` Scale-normalize relative to hand bounding box (size invariance)
  - `[x]` Augment: horizontal flip, small rotation (±15°), brightness jitter
  - `[x]` Save processed landmark arrays to `data/processed/ASL/landmarks.npy`
  - `[x]` Save image crops to `data/processed/ASL/images/` (for CNN model)
- `[x]` Download & integrate public ASL datasets (e.g., Kaggle ASL Alphabet dataset)
- `[x]` Write train/val/test split script (70% / 15% / 15%)
- `[x]` EDA notebook: class distribution, sample visualization, landmark plots

---

## Phase 3 — Model Development

### 3A — Primary Model: MediaPipe Landmarks → MLP/LSTM (ASL)
- `[/]` Implement MLP classifier in `src/model.py`
  - Architecture: `Input(63) → Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dense(N_classes, Softmax)`
- `[/]` Write `src/train.py` with early stopping, checkpointing, LR scheduler
- `[/]` Log metrics to TensorBoard (`logs/`)
- `[/]` Evaluate: confusion matrix, per-class accuracy, top-5 accuracy
- `[/]` Save best model to `models/asl_mlp.h5`

### 3B — Dynamic Gesture Model: LSTM for Word/Phrase Mode
- `[ ]` Collect **sequence data**: N consecutive frames per gesture word
- `[ ]` Build LSTM model: `Input(T, 63) → LSTM(128) → Dense(N_words, Softmax)`
- `[ ]` Train and evaluate; save to `models/asl_lstm.h5`

### 3C — Secondary Model: MobileNetV3 CNN (robustness fallback)
- `[ ]` Fine-tune MobileNetV3 on image crops via transfer learning
- `[ ]` Use as fallback when MediaPipe landmark confidence is low
- `[ ]` Save to `models/asl_mobilenet.h5`

### 3D — ISL Expansion (after ASL MVP ships)
- `[ ]` Collect ISL gesture dataset
- `[ ]` Add ISL output head to existing model (multi-head architecture)
- `[ ]` Add dialect selector to API and UI

---

## Phase 4 — FastAPI Inference Backend

- `[ ]` Build `backend/main.py` — FastAPI app with endpoints:
  - `POST /predict/frame` — accepts base64 image, returns predicted label + confidence
  - `POST /predict/sequence` — accepts frame sequence, returns word prediction
  - `GET /health` — health check
  - `GET /dialects` — list supported sign language dialects
- `[ ]` Implement `src/inference.py`
  - `[ ]` Load MLP + LSTM + CNN models at startup
  - `[ ]` Auto-detect GPU via `tf.config.list_physical_devices('GPU')`, fallback to CPU
  - `[ ]` Per-request: MediaPipe landmark extraction → model inference → response
- `[ ]` Add **confidence threshold** filtering (suppress predictions < 70%)
- `[ ]` Add **temporal smoothing** (rolling majority vote over last 10 frames)
- `[ ]` Add **mode detector**: static single frame → Letter mode; N-frame sequence → Word mode
- `[ ]` Enable CORS for React web and React Native clients
- `[ ]` Add request rate limiting and input validation

---

## Phase 5 — Web App (React)

- `[ ]` Build live camera feed component using `getUserMedia` + `<canvas>`
- `[ ]` Send frames to FastAPI at ~10 FPS via `fetch` / WebSocket
- `[ ]` Display:
  - `[ ]` Live video with hand landmark overlay (drawn on canvas)
  - `[ ]` Predicted letter / word with confidence bar
  - `[ ]` Accumulated text output panel (letter → word → sentence)
- `[ ]` Mode toggle: **Letter / Word / Sentence**
- `[ ]` Language/dialect selector: **ASL / ISL**
- `[ ]` Controls: Start, Stop, Clear, Speak (TTS)
- `[ ]` TTS: call browser `SpeechSynthesis` API for text-to-speech
- `[ ]` Polished UI: dark mode, smooth animations, responsive layout

---

## Phase 6 — Mobile App (React Native)

- `[ ]` Set up `react-native-camera` or `expo-camera` for native camera access
- `[ ]` Reuse same FastAPI endpoints as web app (shared backend)
- `[ ]` Build camera view with real-time prediction overlay
- `[ ]` Shared components with web where possible (business logic, API calls)
- `[ ]` Native TTS via `react-native-tts`
- `[ ]` Support both Android and iOS

---

## Phase 7 — Testing & Validation

- `[ ]` Unit tests (`tests/`):
  - `[ ]` Landmark normalization correctness
  - `[ ]` Model input/output shape validation
  - `[ ]` API endpoint response format
- `[ ]` Integration test: frame → API → predicted label (end-to-end)
- `[ ]` Performance tests:
  - `[ ]` Inference latency target: **< 100ms per frame**
  - `[ ]` Web app target: **≥ 10 FPS** detection rate
- `[ ]` UAT: test with multiple users, lighting conditions, skin tones, hand sizes
- `[ ]` Run `pytest tests/` and ensure all pass

---

## Phase 8 — Documentation & Deployment

- `[ ]` Write full `README.md` (setup, usage, dataset, demo GIF/video)
- `[ ]` Document model architecture, training results, accuracy metrics
- `[ ]` Dockerize backend: `Dockerfile` + `docker-compose.yml`
- `[ ]` Deploy backend to **Cloud Run** or **Railway**
- `[ ]` Deploy web app to **Vercel** or **Netlify**
- `[ ]` (Optional) Publish mobile app to Play Store / App Store

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| Hand Tracking | MediaPipe Hands |
| ML — Static Gestures | TensorFlow/Keras MLP |
| ML — Dynamic Gestures | TensorFlow/Keras LSTM |
| ML — Robustness Fallback | MobileNetV3 (fine-tuned) |
| Backend API | FastAPI + Uvicorn |
| Web Frontend | React + Canvas API |
| Mobile App | React Native |
| TTS (Web) | Browser SpeechSynthesis API |
| TTS (Mobile) | react-native-tts |
| Data | NumPy, Pandas, scikit-learn |
| Experiment Tracking | TensorBoard |
| Deployment | Docker, Cloud Run / Railway, Vercel |

---

## Verification Plan

### Automated
- `pytest tests/` — unit + integration tests
- TensorBoard training metrics at `tensorboard --logdir backend/logs/`
- FastAPI auto-docs at `/docs` (Swagger UI)

### Manual
- Live webcam demo: verify all 26 ASL letters detected correctly
- Word mode: verify 5+ common words recognized in sequence
- TTS output matches predicted gesture text
- Web + mobile UI functional on Chrome, Safari, Android, iOS
- Inference latency < 100ms measured via browser DevTools Network tab
