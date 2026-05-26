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
