# Multi-stage Dockerfile for StreamProcess

# Base stage with common dependencies
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    wget \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY protos/ ./protos/
COPY scripts/ ./scripts/

# Compile protobuf files
RUN python scripts/compile_protos.py

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data

# gRPC Server stage
FROM base as grpc-server
ENV SERVICE_TYPE=grpc
CMD ["python", "-m", "src.grpc_server.server"]

# REST API stage
FROM base as rest-api
ENV SERVICE_TYPE=rest
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# STT Worker stage
FROM base as stt-worker
ENV SERVICE_TYPE=stt-worker
# Download Whisper model during build (optional, can be done at runtime)
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu')"
CMD ["python", "-m", "src.workers.stt_worker"]

# OCR Worker stage
FROM base as ocr-worker
ENV SERVICE_TYPE=ocr-worker
# Download PaddleOCR models during build (optional)
# RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"
CMD ["python", "-m", "src.workers.ocr_worker"]

# Development stage with additional tools
FROM base as development
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy
CMD ["/bin/bash"]