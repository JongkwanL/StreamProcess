# Containerfile for Podman - Multi-stage build with security best practices
# Podman-specific optimizations and rootless container support

# Base stage with minimal attack surface
FROM registry.access.redhat.com/ubi9/python-311:latest as base

# Switch to root for package installation
USER 0

# Install only essential system dependencies
RUN dnf update -y --security && \
    dnf install -y \
    gcc \
    gcc-c++ \
    git \
    wget \
    libgomp \
    libsndfile \
    ffmpeg-free \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Create non-root user for running application
RUN useradd -m -u 1001 -s /bin/bash appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Switch to app directory
WORKDIR /app

# Copy requirements as root (for pip install)
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser protos/ ./protos/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Compile protobuf files
RUN python3 scripts/compile_protos.py

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Security: Drop all capabilities by default
# These will be overridden in podman-compose.yml as needed

# gRPC Server stage
FROM base as grpc-server
USER appuser
ENV SERVICE_TYPE=grpc
EXPOSE 50051 9090
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python3 -c "import grpc; channel=grpc.insecure_channel('localhost:50051'); channel.close()" || exit 1
CMD ["python3", "-m", "src.grpc_server.server"]

# REST API stage
FROM base as rest-api
USER appuser
ENV SERVICE_TYPE=rest
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# STT Worker stage
FROM base as stt-worker
USER appuser
ENV SERVICE_TYPE=stt-worker
# Pre-download model during build (optional)
# RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', download_root='/app/models')"
CMD ["python3", "-m", "src.workers.stt_worker"]

# OCR Worker stage
FROM base as ocr-worker
USER appuser
ENV SERVICE_TYPE=ocr-worker
# Pre-download OCR models during build (optional)
# RUN python3 -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', show_log=False)"
CMD ["python3", "-m", "src.workers.ocr_worker"]

# Development stage with additional tools
FROM base as development
USER 0
RUN pip3 install --no-cache-dir \
    ipython \
    jupyter \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    debugpy
USER appuser
CMD ["/bin/bash"]

# Production hardened stage (example)
FROM base as production
USER appuser
# Remove unnecessary files
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -delete && \
    rm -rf /app/tests /app/benchmarks /app/scripts

# Set read-only root filesystem flag (will be enforced by Podman)
# Additional security settings in podman-compose.yml