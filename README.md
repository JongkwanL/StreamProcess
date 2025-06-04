# StreamProcess - Real-time Multimodal Processing Pipeline

## 🎯 프로젝트 개요
StreamProcess는 실시간 음성 인식(STT)과 문서 광학 문자 인식(OCR)을 통합한 멀티모달 처리 파이프라인입니다. 스트리밍 gRPC와 큐 기반 아키텍처로 대용량 미디어 데이터를 효율적으로 처리합니다.

## 🚀 핵심 기능
- **실시간 음성 인식**: Faster-Whisper 기반 저지연 STT
- **문서 OCR 처리**: PaddleOCR을 활용한 다국어 문서 인식
- **스트리밍 처리**: gRPC 스트리밍으로 실시간 데이터 전송
- **배치 최적화**: 동적 배칭으로 처리량 극대화
- **오토스케일링**: 워크로드 기반 자동 스케일링

## 🛠️ 기술 스택
### ML/AI
- **STT Engine**: Faster-Whisper
- **OCR Engine**: PaddleOCR, Tesseract
- **Inference Server**: Triton Inference Server
- **Model Format**: ONNX

### Backend
- **Language**: Python, Golang (workers)
- **API**: gRPC (streaming), REST
- **Queue**: Redis Streams, SQS
- **Storage**: S3/MinIO

### Infrastructure
- **Container**: Docker
- **Orchestration**: Kubernetes (Worker autoscaling)
- **GPU Support**: NVIDIA Device Plugin

## 📊 성능 목표
- **STT 지연시간**: < 300ms
- **OCR 처리시간**: p99 < 2초
- **동시 스트림**: 100+ concurrent
- **처리 실패율**: < 1%
- **GPU 활용률**: > 60%

## 🏗️ 아키텍처
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Audio/Image │────▶│  gRPC Server │────▶│    Queue     │
│    Client    │     │  (Streaming) │     │ (Redis/SQS)  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                          ┌───────▼────────┐
                                          │    Workers     │
                                          │  (Autoscale)   │
                                          └───────┬────────┘
                                                  │
                    ┌──────────────┬──────────────┴──────────────┐
                    │              │                              │
            ┌───────▼──────┐ ┌────▼─────┐           ┌────────────▼──────────┐
            │ Faster-      │ │ Paddle   │           │  Triton Inference     │
            │ Whisper      │ │   OCR    │           │      Server           │
            └──────────────┘ └──────────┘           └───────────────────────┘
```

## 📁 프로젝트 구조
```
StreamProcess/
├── src/
│   ├── grpc_server/   # gRPC streaming server
│   ├── stt/           # STT service logic
│   ├── ocr/           # OCR service logic
│   ├── workers/       # Queue workers
│   └── preprocessing/ # Audio/Image preprocessing
├── models/            # Model files and configs
├── k8s/               # Kubernetes manifests
├── triton/            # Triton model repository
├── tests/             # Test suites
└── benchmarks/        # Performance benchmarks
```

## 🚦 API Endpoints

### gRPC Services
```protobuf
service STTService {
  rpc StreamingRecognize(stream AudioChunk) returns (stream TranscriptEvent);
}

service OCRService {
  rpc ProcessDocument(DocumentRequest) returns (OCRResponse);
  rpc BatchProcess(stream DocumentRequest) returns (stream OCRResponse);
}
```

### REST API
- `POST /stt/upload` - 오디오 파일 업로드 처리
- `POST /ocr/process` - 이미지/문서 OCR 처리
- `GET /status/{job_id}` - 작업 상태 조회
- `GET /health` - 서비스 헬스체크

## 🔧 설치 및 실행
```bash
# 모델 다운로드
python scripts/download_models.py

# 로컬 개발
docker-compose up -d

# Kubernetes 배포
kubectl apply -f k8s/

# 성능 테스트
python benchmarks/run_benchmarks.py
```

## 📈 개발 로드맵
- [x] Week 4: STT 서비스 구현
- [x] Week 5: OCR 서비스 구현
- [x] Week 6: 큐 시스템 및 Triton 통합
- [ ] 모델 최적화 (양자화, pruning)
- [ ] 다국어 지원 확대
- [ ] 실시간 번역 기능 추가

## 🎯 지원 기능
### STT
- **언어**: 한국어, 영어, 일본어, 중국어
- **형식**: WAV, MP3, M4A, FLAC
- **실시간**: WebSocket 스트리밍 지원

### OCR
- **언어**: 80+ languages
- **형식**: PNG, JPG, PDF, TIFF
- **레이아웃**: 표, 양식, 다단 문서 지원

## 📚 문서
- [gRPC API Reference](./docs/grpc_api.md)
- [Model Optimization Guide](./docs/optimization.md)
- [Deployment Guide](./docs/deployment.md)
- [Performance Tuning](./docs/performance.md)

## 📄 라이선스
MIT License