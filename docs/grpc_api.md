# gRPC API Reference

## Overview
StreamProcess provides gRPC APIs for real-time speech-to-text (STT) and optical character recognition (OCR) processing.

## STT Service

### StreamingRecognize (Bidirectional Streaming)
Real-time speech recognition with bidirectional streaming.

**Request Stream**: `StreamingRecognizeRequest`
- First message must contain `StreamingRecognitionConfig`
- Subsequent messages contain `AudioChunk` data

**Response Stream**: `StreamingRecognizeResponse`
- Returns transcript events as they become available
- Includes partial and final transcripts

**Example Usage (Python)**:
```python
import grpc
from src.generated import stream_process_pb2, stream_process_pb2_grpc

async def stream_audio():
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = stream_process_pb2_grpc.STTServiceStub(channel)
    
    # Configure stream
    config = stream_process_pb2.StreamingRecognitionConfig(
        audio_config=stream_process_pb2.AudioConfig(
            encoding=stream_process_pb2.AudioConfig.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        ),
        recognition_config=stream_process_pb2.RecognitionConfig(
            model="base",
            beam_size=1,
            compute_type="int8"
        ),
        enable_partial_results=True,
        partial_results_interval_ms=150
    )
    
    # Stream audio chunks
    async def request_generator():
        # Send config first
        yield stream_process_pb2.StreamingRecognizeRequest(config=config)
        
        # Stream audio chunks
        with open("audio.wav", "rb") as f:
            while chunk := f.read(8192):
                yield stream_process_pb2.StreamingRecognizeRequest(
                    audio_chunk=stream_process_pb2.AudioChunk(
                        content=chunk,
                        offset_ms=0,
                        duration_ms=100
                    )
                )
    
    # Process responses
    async for response in stub.StreamingRecognize(request_generator()):
        for result in response.results:
            if result.event_type == stream_process_pb2.TranscriptEvent.FINAL_TRANSCRIPT:
                print(f"Final: {result.alternatives[0].transcript}")
            else:
                print(f"Partial: {result.alternatives[0].transcript}")
```

### Recognize (Unary)
Batch audio processing for complete audio files.

**Request**: `RecognizeRequest`
- `audio_config`: Audio format configuration
- `recognition_config`: Recognition parameters
- `audio_content`: Complete audio data

**Response**: `RecognizeResponse`
- `results`: Array of recognition results
- `metrics`: Processing metrics

### GetCapabilities
Query service capabilities and supported features.

**Response**: `STTCapabilities`
- `available_models`: List of available Whisper models
- `supported_languages`: Supported language codes
- `supported_encodings`: Audio format support
- `max_concurrent_streams`: Concurrency limit

## OCR Service

### ProcessDocument (Unary)
Process a single document for OCR.

**Request**: `DocumentRequest`
- `config`: OCR configuration
- `image_content` or `document_url`: Image data or URL

**Response**: `OCRResponse`
- `result`: OCR results with text blocks
- `metrics`: Processing metrics

**Example Usage**:
```python
async def process_document():
    channel = grpc.aio.insecure_channel('localhost:50051')
    stub = stream_process_pb2_grpc.OCRServiceStub(channel)
    
    with open("document.png", "rb") as f:
        image_data = f.read()
    
    request = stream_process_pb2.DocumentRequest(
        config=stream_process_pb2.DocumentConfig(
            languages=["en", "ch_sim"],
            detect_layout=True,
            detect_tables=True,
            output_format=stream_process_pb2.DocumentConfig.FORMAT_JSON
        ),
        image_content=image_data
    )
    
    response = await stub.ProcessDocument(request)
    print(f"Text: {response.result.full_text}")
    print(f"Confidence: {response.result.confidence}")
```

### BatchProcess (Server Streaming)
Process multiple documents with streaming responses.

**Request Stream**: `DocumentRequest`
- Stream of document requests

**Response Stream**: `OCRResponse`
- Results streamed as processing completes

### GetCapabilities
Query OCR service capabilities.

**Response**: `OCRCapabilities`
- `supported_languages`: Available OCR languages
- `supported_formats`: Image format support
- `gpu_available`: GPU acceleration status

## Common Types

### Priority Levels
- `PRIORITY_LOW`: Background processing
- `PRIORITY_NORMAL`: Standard processing
- `PRIORITY_HIGH`: Elevated priority
- `PRIORITY_REALTIME`: Real-time processing

### Processing Status
- `STATUS_PENDING`: Queued for processing
- `STATUS_PROCESSING`: Currently processing
- `STATUS_COMPLETED`: Successfully completed
- `STATUS_FAILED`: Processing failed
- `STATUS_CANCELLED`: Cancelled by client

### Error Handling
All responses include an optional `ErrorDetail` with:
- `code`: Error code
- `message`: Human-readable message
- `details`: Additional error context

## Performance Considerations

### Streaming Best Practices
1. Use appropriate chunk sizes (320-640ms for audio)
2. Enable partial results for real-time feedback
3. Implement client-side buffering
4. Handle backpressure signals

### Batching Guidelines
1. Group similar requests for better throughput
2. Use priority levels appropriately
3. Monitor queue depth metrics
4. Implement retry logic with exponential backoff

## Metrics and Monitoring

All services expose Prometheus metrics on port 9090:
- Request latency histograms
- Active stream gauges
- Queue depth metrics
- GPU utilization
- Error rates

## Rate Limiting

Default limits:
- Max concurrent streams: 100
- Max message size: 100MB
- Max queue depth: 10,000 jobs
- Backpressure activation: 80% utilization