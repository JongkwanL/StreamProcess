"""FastAPI REST API for StreamProcess."""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import grpc
import asyncio
import uuid
import time
from loguru import logger
import json

from ..config import settings
from ..generated import stream_process_pb2, stream_process_pb2_grpc
from ..queue.redis_queue import RedisQueue


app = FastAPI(
    title="StreamProcess API",
    description="Real-time multimodal processing pipeline",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connections
grpc_channel = None
stt_stub = None
ocr_stub = None
queue_stub = None
redis_queue = None


class STTRequest(BaseModel):
    """STT request model."""
    language: str = Field(default="en", description="Language code")
    model: str = Field(default="base", description="Whisper model size")
    enable_vad: bool = Field(default=True, description="Enable VAD")
    enable_punctuation: bool = Field(default=True, description="Enable punctuation")


class OCRRequest(BaseModel):
    """OCR request model."""
    languages: List[str] = Field(default=["en"], description="OCR languages")
    detect_layout: bool = Field(default=True, description="Detect document layout")
    detect_tables: bool = Field(default=True, description="Detect tables")
    output_format: str = Field(default="json", description="Output format")


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str
    created_at: float
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup."""
    global grpc_channel, stt_stub, ocr_stub, queue_stub, redis_queue
    
    # Initialize gRPC channel
    grpc_channel = grpc.aio.insecure_channel(
        f"{settings.grpc_server_host}:{settings.grpc_server_port}",
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )
    
    # Create stubs
    stt_stub = stream_process_pb2_grpc.STTServiceStub(grpc_channel)
    ocr_stub = stream_process_pb2_grpc.OCRServiceStub(grpc_channel)
    queue_stub = stream_process_pb2_grpc.QueueServiceStub(grpc_channel)
    
    # Initialize Redis queue for job tracking
    redis_queue = RedisQueue(settings)
    await redis_queue.initialize()
    
    logger.info("API server initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if grpc_channel:
        await grpc_channel.close()
    if redis_queue:
        await redis_queue.close()
    logger.info("API server shutdown")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "StreamProcess API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {}
    }
    
    # Check gRPC connection
    try:
        capabilities = await stt_stub.GetCapabilities(
            stream_process_pb2.GetCapabilitiesRequest()
        )
        health_status["services"]["grpc"] = "healthy"
    except Exception as e:
        health_status["services"]["grpc"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        depth = await redis_queue.get_depth()
        health_status["services"]["redis"] = "healthy"
        health_status["queue_depth"] = depth
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


@app.post("/stt/upload")
async def stt_upload(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    language: str = "en",
    model: str = "base"
):
    """Upload audio file for STT processing."""
    job_id = str(uuid.uuid4())
    
    try:
        # Read audio file
        audio_content = await file.read()
        
        # Validate file size
        if len(audio_content) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=413, detail="File too large")
        
        # Create gRPC request
        request = stream_process_pb2.RecognizeRequest(
            metadata=stream_process_pb2.ProcessingMetadata(
                session_id=job_id,
                priority=stream_process_pb2.PRIORITY_NORMAL
            ),
            audio_config=stream_process_pb2.AudioConfig(
                encoding=stream_process_pb2.AudioConfig.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language
            ),
            recognition_config=stream_process_pb2.RecognitionConfig(
                model=model,
                max_alternatives=1,
                enable_automatic_punctuation=True
            ),
            audio_content=audio_content
        )
        
        # Submit to gRPC service (async)
        background_tasks.add_task(process_stt_request, job_id, request)
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Audio file submitted for processing"
        }
    
    except Exception as e:
        logger.error(f"STT upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_stt_request(job_id: str, request: stream_process_pb2.RecognizeRequest):
    """Process STT request in background."""
    try:
        response = await stt_stub.Recognize(request)
        
        # Store result
        result = {
            "transcript": response.results[0].alternatives[0].transcript if response.results else "",
            "confidence": response.results[0].alternatives[0].confidence if response.results else 0.0,
            "language": response.results[0].language_code if response.results else "unknown",
            "metrics": {
                "processing_time_ms": response.metrics.total_processing_time_ms,
                "audio_duration_ms": response.metrics.audio_duration_ms,
                "real_time_factor": response.metrics.real_time_factor
            }
        }
        
        await redis_queue.store_result(job_id, result)
        
    except Exception as e:
        logger.error(f"STT processing error: {e}")
        await redis_queue.store_result(job_id, {"error": str(e)})


@app.post("/ocr/process")
async def ocr_process(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    languages: str = "en,ch_sim",
    detect_layout: bool = True,
    output_format: str = "json"
):
    """Process document for OCR."""
    job_id = str(uuid.uuid4())
    
    try:
        # Read image file
        image_content = await file.read()
        
        # Validate file size
        if len(image_content) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="File too large")
        
        # Create gRPC request
        request = stream_process_pb2.DocumentRequest(
            metadata=stream_process_pb2.ProcessingMetadata(
                session_id=job_id,
                priority=stream_process_pb2.PRIORITY_NORMAL
            ),
            config=stream_process_pb2.DocumentConfig(
                languages=languages.split(","),
                detect_layout=detect_layout,
                detect_tables=True,
                output_format=getattr(
                    stream_process_pb2.DocumentConfig,
                    f"FORMAT_{output_format.upper()}",
                    stream_process_pb2.DocumentConfig.FORMAT_JSON
                )
            ),
            image_content=image_content
        )
        
        # Submit to gRPC service
        background_tasks.add_task(process_ocr_request, job_id, request)
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Document submitted for OCR processing"
        }
    
    except Exception as e:
        logger.error(f"OCR upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_ocr_request(job_id: str, request: stream_process_pb2.DocumentRequest):
    """Process OCR request in background."""
    try:
        response = await ocr_stub.ProcessDocument(request)
        
        # Store result
        result = {
            "text": response.result.full_text if response.result else "",
            "confidence": response.result.confidence if response.result else 0.0,
            "blocks": [
                {
                    "text": block.text,
                    "confidence": block.confidence,
                    "type": block.block_type
                }
                for block in (response.result.blocks if response.result else [])
            ],
            "metrics": {
                "processing_time_ms": response.metrics.processing_time_ms if response.metrics else 0,
                "total_characters": response.metrics.total_characters if response.metrics else 0,
                "total_words": response.metrics.total_words if response.metrics else 0
            }
        }
        
        await redis_queue.store_result(job_id, result)
        
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        await redis_queue.store_result(job_id, {"error": str(e)})


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job processing status."""
    # Check for result
    result = await redis_queue.get_result(job_id)
    
    if result:
        if "error" in result:
            return JobStatus(
                job_id=job_id,
                status="failed",
                created_at=time.time(),
                error=result["error"]
            )
        else:
            return JobStatus(
                job_id=job_id,
                status="completed",
                created_at=time.time(),
                completed_at=time.time(),
                result=result
            )
    else:
        # Check if job exists in queue
        # For now, assume it's still processing
        return JobStatus(
            job_id=job_id,
            status="processing",
            created_at=time.time()
        )


@app.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket):
    """WebSocket endpoint for real-time STT."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    
    try:
        # Create bidirectional stream
        async def request_generator():
            # Send initial config
            config = stream_process_pb2.StreamingRecognitionConfig(
                metadata=stream_process_pb2.ProcessingMetadata(
                    session_id=session_id,
                    priority=stream_process_pb2.PRIORITY_REALTIME
                ),
                audio_config=stream_process_pb2.AudioConfig(
                    encoding=stream_process_pb2.AudioConfig.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en"
                ),
                recognition_config=stream_process_pb2.RecognitionConfig(
                    model="base",
                    beam_size=1,
                    compute_type="int8"
                ),
                enable_vad=True,
                enable_partial_results=True,
                partial_results_interval_ms=150
            )
            
            yield stream_process_pb2.StreamingRecognizeRequest(config=config)
            
            # Stream audio chunks from WebSocket
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
                    
                    yield stream_process_pb2.StreamingRecognizeRequest(
                        audio_chunk=stream_process_pb2.AudioChunk(
                            content=data,
                            offset_ms=0,
                            duration_ms=100
                        )
                    )
                except asyncio.TimeoutError:
                    # Send keepalive
                    await websocket.send_json({"type": "keepalive"})
                except WebSocketDisconnect:
                    break
        
        # Process responses
        async for response in stt_stub.StreamingRecognize(request_generator()):
            for result in response.results:
                await websocket.send_json({
                    "type": "partial" if result.event_type == stream_process_pb2.TranscriptEvent.PARTIAL_TRANSCRIPT else "final",
                    "transcript": result.alternatives[0].transcript if result.alternatives else "",
                    "confidence": result.alternatives[0].confidence if result.alternatives else 0.0,
                    "stability": result.stability
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.get("/queue/status")
async def queue_status():
    """Get queue status."""
    try:
        response = await queue_stub.GetQueueStatus(
            stream_process_pb2.QueueStatusRequest(queue_name="default")
        )
        
        return {
            "pending_jobs": response.pending_jobs,
            "processing_jobs": response.processing_jobs,
            "completed_jobs": response.completed_jobs,
            "failed_jobs": response.failed_jobs,
            "average_wait_time_ms": response.average_wait_time_ms,
            "average_processing_time_ms": response.average_processing_time_ms
        }
    
    except Exception as e:
        logger.error(f"Queue status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/capabilities/stt")
async def stt_capabilities():
    """Get STT service capabilities."""
    try:
        response = await stt_stub.GetCapabilities(
            stream_process_pb2.GetCapabilitiesRequest()
        )
        
        return {
            "models": list(response.available_models),
            "languages": list(response.supported_languages),
            "encodings": list(response.supported_encodings),
            "max_stream_duration": response.max_stream_duration_seconds,
            "max_concurrent_streams": response.max_concurrent_streams
        }
    
    except Exception as e:
        logger.error(f"Capabilities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/capabilities/ocr")
async def ocr_capabilities():
    """Get OCR service capabilities."""
    try:
        response = await ocr_stub.GetCapabilities(
            stream_process_pb2.GetCapabilitiesRequest()
        )
        
        return {
            "languages": list(response.supported_languages),
            "formats": list(response.supported_formats),
            "max_image_size_mb": response.max_image_size_mb,
            "max_concurrent_requests": response.max_concurrent_requests,
            "gpu_available": response.gpu_available
        }
    
    except Exception as e:
        logger.error(f"Capabilities error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.rest_server_host,
        port=settings.rest_server_port,
        log_level=settings.log_level.value.lower()
    )