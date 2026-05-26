# Sign Language Detection System — Backend
# PowerShell setup script for Windows

Write-Host "🔧 Setting up Sign Language Detection System backend..." -ForegroundColor Cyan

# 1. Create virtual environment
Write-Host "`n📦 Creating Python virtual environment..." -ForegroundColor Yellow
python -m venv .venv

# 2. Activate venv
Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip
Write-Host "`n⬆️  Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 4. Install dependencies
Write-Host "`n📥 Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# 5. Verify key packages
Write-Host "`n🔍 Verifying installation..." -ForegroundColor Yellow
python -c "import cv2; print(f'  ✅ OpenCV {cv2.__version__}')"
python -c "import mediapipe; print(f'  ✅ MediaPipe {mediapipe.__version__}')"
python -c "import tensorflow as tf; print(f'  ✅ TensorFlow {tf.__version__}')"
python -c "import fastapi; print(f'  ✅ FastAPI {fastapi.__version__}')"

# 6. Create .env file if not exists
if (-Not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "`n📝 Created .env from .env.example" -ForegroundColor Green
}

Write-Host "`n🎉 Setup complete! Run the server with:" -ForegroundColor Green
Write-Host "   .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   uvicorn main:app --reload" -ForegroundColor White
