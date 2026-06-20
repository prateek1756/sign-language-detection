# 🤟 Sign Language Detection System

A real-time sign language recognition system that detects and translates hand gestures into text and speech using **MediaPipe**, **TensorFlow**, **FastAPI**, and **React**.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)

---

## ✨ Features

- 🖐️ **Real-time hand gesture detection** via webcam (web & mobile)
- 🔤 **3-mode adaptive output**: Letter → Word → Sentence
- 🌍 **Multi-dialect support**: ASL (live), ISL (coming soon)
- ⚡ **Adaptive hardware**: GPU inference with CPU fallback
- 🔊 **Text-to-Speech** output (browser API + native mobile)
- 📱 **Cross-platform**: React web app + React Native mobile

---

## 🏗️ Architecture

```
Camera Feed → MediaPipe Hands (CPU) → MLP/LSTM Model (GPU/CPU)
           → FastAPI Backend → React Web App / React Native App
           → TTS Output
```

---

## 🚀 Quick Start

### Backend

```powershell
cd backend
.\setup.ps1         # Creates venv & installs dependencies
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

API docs available at: `http://localhost:8000/docs`

### Web App

```bash
cd web
npm install
npm run dev         # Starts at http://localhost:5173
```

### Mobile App

```bash
cd mobile
npm install
npx expo start
```

---

## 🌐 Deployment

### Backend (Railway)
1. Register/Login to [Railway](https://railway.app).
2. Create a new project and connect this GitHub repository.
3. Configure the Railway service root directory to `backend/` (Railway will automatically build using the `backend/Dockerfile`).
4. Set the following environment variables under **Settings**:
   - `PORT`: `8000`
   - `CORS_ORIGINS`: `http://localhost:5173,https://your-frontend.vercel.app`
   - `GDRIVE_MODEL_URL_MLP`: Direct download link for `asl_mlp.onnx` / `asl_mlp.keras` (if downloading on lifespan startup).
   - `GDRIVE_MODEL_URL_LSTM`: Direct download link for `asl_lstm.onnx` / `asl_lstm.keras`.
   - `GDRIVE_MODEL_URL_CNN`: Direct download link for `asl_mobilenet.keras`.

### Web Frontend (Vercel)
1. Register/Login to [Vercel](https://vercel.com).
2. Connect your GitHub repository.
3. Set the **Root Directory** to `web`.
4. Configure the build settings:
   - Build Command: `npm run build`
   - Output Directory: `dist`
5. Under environment variables, add:
   - `VITE_WS_URL`: `wss://your-backend.railway.app/ws/stream`
6. Click **Deploy**. SPA routing fallback is handled automatically via `web/vercel.json`.

### Mobile App (Expo Go)
1. Install **Expo Go** on your physical iOS or Android device.
2. In the `mobile/` directory, start the development server: `npm run start` (or `npx expo start`).
3. Scan the terminal's QR code with your phone.
4. Once loaded, click the **Settings (⚙️)** button in the top right.
5. Update the WebSocket URL (e.g. `ws://192.168.1.X:8000/ws/stream` for local testing on the same Wi-Fi, or `wss://your-backend.railway.app/ws/stream` for production) and tap **Connect**.

---

## 📁 Project Structure

```
sign-language-detection/
├── backend/                # FastAPI server + ML models
│   ├── src/
│   │   ├── collect_data.py     # Phase 2: Data collection
│   │   ├── preprocess.py       # Phase 2: MediaPipe + augmentation
│   │   ├── model.py            # Phase 3: MLP / LSTM / CNN
│   │   ├── train.py            # Phase 3: Training pipeline
│   │   └── inference.py        # Phase 4: Real-time inference
│   ├── models/                 # Saved .keras / .pt model files
│   ├── data/
│   │   ├── raw/ASL/            # Raw captured images by class
│   │   └── processed/ASL/      # Normalized landmark arrays
│   ├── tests/                  # pytest unit + integration tests
│   ├── main.py                 # FastAPI entry point
│   ├── requirements.txt
│   └── setup.ps1               # Windows setup script
├── web/                    # React + Vite web application
├── mobile/                 # React Native mobile app
├── notebooks/              # EDA and training experiments
└── README.md
```

---

## 🧠 Models

| Model | Input | Use Case | Target Accuracy |
|---|---|---|---|
| MLP | 63 landmarks | ASL A–Z alphabet | > 95% |
| LSTM | T×63 sequences | Words & phrases | > 85% |
| MobileNetV3 | Image crops | Low-confidence fallback | > 90% |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Hand Tracking | MediaPipe Hands |
| ML Framework | TensorFlow / Keras |
| Backend API | FastAPI + Uvicorn |
| Web Frontend | React + Vite |
| Mobile App | React Native + Expo |
| TTS (Web) | Browser SpeechSynthesis |
| TTS (Mobile) | react-native-tts |

---

## 📊 Roadmap

- [x] Phase 1: Project scaffold
- [ ] Phase 2: Data collection & preprocessing (ASL)
- [ ] Phase 3: Model training (MLP + LSTM + CNN)
- [ ] Phase 4: FastAPI inference backend
- [ ] Phase 5: React web app
- [ ] Phase 6: React Native mobile app
- [ ] Phase 7: Testing & validation
- [ ] Phase 8: Deployment

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
