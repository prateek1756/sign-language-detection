# ── GPU-enabled base image with CUDA 12.3 + cuDNN 9 ──────────────────────────
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Install Python 3.11 and system dependencies for OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install GPU-enabled TensorFlow (overwrites CPU-only version from requirements.txt)
RUN pip install --no-cache-dir tensorflow[and-cuda]==2.16.1 || true

# Install ONNX-related packages with --no-deps to bypass Protobuf version conflicts
RUN pip install --no-cache-dir --no-deps tf2onnx==1.17.0 onnx==1.22.0 onnxruntime==1.27.0 ml_dtypes==0.5.4

# Copy project files
COPY . .

# Expose uvicorn port (default: 8000)
EXPOSE 8000

# Set default port environment variable
ENV PORT=8000

# Allow TensorFlow to see the GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Start FastAPI application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
