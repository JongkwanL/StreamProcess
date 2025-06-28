# Multi-stage build for Go services
# Podman/Buildah optimized with security best practices

# Build stage
FROM docker.io/golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Create non-root user for build
RUN adduser -D -s /bin/sh appuser

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build services
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o bin/grpc-server ./cmd/grpc-server
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o bin/stt-worker ./cmd/stt-worker
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o bin/ocr-worker ./cmd/ocr-worker
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o bin/autoscaler ./cmd/autoscaler

# gRPC Server
FROM scratch AS grpc-server

# Copy certificates for HTTPS
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/passwd /etc/passwd

# Copy binary
COPY --from=builder /app/bin/grpc-server /grpc-server

# Use non-root user
USER appuser

# Expose ports
EXPOSE 50051 8080 9092

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["/grpc-server", "--health-check"]

ENTRYPOINT ["/grpc-server"]

# STT Worker
FROM docker.io/alpine:3.18 AS stt-worker

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata

# Create non-root user
RUN adduser -D -s /bin/sh appuser

# Copy binary
COPY --from=builder /app/bin/stt-worker /usr/local/bin/stt-worker

# Use non-root user
USER appuser

# Create temp directory for worker
RUN mkdir -p /tmp/stt-worker
VOLUME ["/tmp/stt-worker"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD ["wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9090/health", "||", "exit", "1"]

ENTRYPOINT ["stt-worker"]

# OCR Worker
FROM docker.io/alpine:3.18 AS ocr-worker

# Install Tesseract and runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    tesseract-ocr \
    tesseract-ocr-data-eng \
    tesseract-ocr-data-chi_sim \
    tesseract-ocr-data-jpn \
    tesseract-ocr-data-kor \
    && rm -rf /var/cache/apk/*

# Create non-root user
RUN adduser -D -s /bin/sh appuser

# Copy binary
COPY --from=builder /app/bin/ocr-worker /usr/local/bin/ocr-worker

# Use non-root user
USER appuser

# Create temp directory for worker
RUN mkdir -p /tmp/ocr-worker
VOLUME ["/tmp/ocr-worker"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD ["wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9091/health", "||", "exit", "1"]

ENTRYPOINT ["ocr-worker"]

# Autoscaler Controller
FROM scratch AS autoscaler

# Copy certificates and timezone data
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/passwd /etc/passwd

# Copy binary
COPY --from=builder /app/bin/autoscaler /autoscaler

# Use non-root user
USER appuser

# Health check
HEALTHCHECK --interval=60s --timeout=5s --start-period=10s --retries=3 \
  CMD ["/autoscaler", "--health-check"]

ENTRYPOINT ["/autoscaler"]

# Development stage with all tools
FROM docker.io/golang:1.21-alpine AS development

# Install development dependencies
RUN apk add --no-cache \
    git \
    ca-certificates \
    tzdata \
    curl \
    wget \
    bash \
    tesseract-ocr \
    && rm -rf /var/cache/apk/*

# Install Go tools
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest && \
    go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest && \
    go install github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-grpc-gateway@latest

# Set working directory
WORKDIR /app

# Copy source
COPY . .

# Build all binaries
RUN make build

ENTRYPOINT ["bash"]