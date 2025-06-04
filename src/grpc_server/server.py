"""gRPC streaming server implementation."""

import asyncio
import grpc
from concurrent import futures
from typing import AsyncIterator, Optional, Dict, Any
import time
import uuid
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from ..config import settings
from ..generated import stream_process_pb2, stream_process_pb2_grpc
from ..queue.redis_queue import RedisQueue
from .backpressure import BackpressureManager
from .session_manager import SessionManager


# Metrics
grpc_requests_total = Counter(
    "grpc_requests_total", 
    "Total gRPC requests",
    ["service", "method", "status"]
)
grpc_request_duration = Histogram(
    "grpc_request_duration_seconds",
    "gRPC request duration",
    ["service", "method"]
)
active_streams = Gauge(
    "active_streams",
    "Number of active streaming connections",
    ["service"]
)
queue_depth = Gauge(
    "queue_depth",
    "Current queue depth",
    ["queue_name"]
)


class STTServicer(stream_process_pb2_grpc.STTServiceServicer):
    """Speech-to-Text service implementation."""
    
    def __init__(self, queue: RedisQueue, session_manager: SessionManager):
        self.queue = queue
        self.session_manager = session_manager
        self.backpressure = BackpressureManager(
            high_watermark=settings.backpressure_high_watermark,
            low_watermark=settings.backpressure_low_watermark
        )
    
    async def StreamingRecognize(
        self,
        request_iterator: AsyncIterator[stream_process_pb2.StreamingRecognizeRequest],
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[stream_process_pb2.StreamingRecognizeResponse]:
        """Handle bidirectional streaming for real-time STT."""
        session_id = str(uuid.uuid4())
        session = None
        active_streams.labels(service="stt").inc()
        
        try:
            async for request in request_iterator:
                # Check backpressure
                if await self.backpressure.should_pause():
                    logger.warning(f"Backpressure activated for session {session_id}")
                    await self.backpressure.wait_for_capacity()
                
                # Handle config or audio chunk
                if request.HasField("config"):
                    # Initialize session with config
                    config = request.config
                    session = await self.session_manager.create_session(
                        session_id=session_id,
                        config=config,
                        context=context
                    )
                    logger.info(f"Initialized STT session {session_id}")
                    
                    # Send initial response
                    yield stream_process_pb2.StreamingRecognizeResponse(
                        status=stream_process_pb2.STATUS_PROCESSING,
                        metrics=stream_process_pb2.StreamingMetrics(
                            queue_depth=await self.queue.get_depth()
                        )
                    )
                
                elif request.HasField("audio_chunk"):
                    if not session:
                        context.abort(
                            grpc.StatusCode.FAILED_PRECONDITION,
                            "Audio chunk received before configuration"
                        )
                    
                    chunk = request.audio_chunk
                    
                    # Queue audio chunk for processing
                    job_data = {
                        "type": "stt_chunk",
                        "session_id": session_id,
                        "audio": chunk.content,
                        "offset_ms": chunk.offset_ms,
                        "duration_ms": chunk.duration_ms,
                        "is_final": chunk.is_final,
                        "config": session.config_dict,
                        "timestamp": time.time()
                    }
                    
                    # Add to queue with priority
                    priority = session.config.metadata.priority if session.config.metadata else 1
                    await self.queue.add_job(job_data, priority=priority)
                    
                    # Check for results from worker
                    results = await session.get_results()
                    for result in results:
                        yield result
                    
                    if chunk.is_final:
                        logger.info(f"Final chunk received for session {session_id}")
                        break
        
        except asyncio.CancelledError:
            logger.info(f"STT stream cancelled for session {session_id}")
            raise
        except Exception as e:
            logger.error(f"Error in STT stream {session_id}: {e}")
            yield stream_process_pb2.StreamingRecognizeResponse(
                status=stream_process_pb2.STATUS_FAILED,
                error=stream_process_pb2.ErrorDetail(
                    code="INTERNAL_ERROR",
                    message=str(e)
                )
            )
        finally:
            active_streams.labels(service="stt").dec()
            if session:
                await self.session_manager.close_session(session_id)
    
    async def Recognize(
        self,
        request: stream_process_pb2.RecognizeRequest,
        context: grpc.aio.ServicerContext
    ) -> stream_process_pb2.RecognizeResponse:
        """Handle batch audio processing."""
        job_id = str(uuid.uuid4())
        
        try:
            # Queue job for processing
            job_data = {
                "type": "stt_batch",
                "job_id": job_id,
                "audio": request.audio_content,
                "audio_config": MessageToDict(request.audio_config),
                "recognition_config": MessageToDict(request.recognition_config),
                "metadata": MessageToDict(request.metadata) if request.metadata else {},
                "timestamp": time.time()
            }
            
            priority = request.metadata.priority if request.metadata else 1
            await self.queue.add_job(job_data, priority=priority)
            
            # Wait for result (with timeout)
            result = await self.queue.wait_for_result(job_id, timeout=30)
            
            if result:
                return stream_process_pb2.RecognizeResponse(
                    results=result.get("results", []),
                    status=stream_process_pb2.STATUS_COMPLETED,
                    metrics=stream_process_pb2.BatchMetrics(
                        total_processing_time_ms=result.get("processing_time_ms", 0),
                        audio_duration_ms=result.get("audio_duration_ms", 0),
                        real_time_factor=result.get("real_time_factor", 0)
                    )
                )
            else:
                return stream_process_pb2.RecognizeResponse(
                    status=stream_process_pb2.STATUS_FAILED,
                    error=stream_process_pb2.ErrorDetail(
                        code="TIMEOUT",
                        message="Processing timeout"
                    )
                )
        
        except Exception as e:
            logger.error(f"Error in batch STT processing: {e}")
            return stream_process_pb2.RecognizeResponse(
                status=stream_process_pb2.STATUS_FAILED,
                error=stream_process_pb2.ErrorDetail(
                    code="INTERNAL_ERROR",
                    message=str(e)
                )
            )
    
    async def GetCapabilities(
        self,
        request: stream_process_pb2.GetCapabilitiesRequest,
        context: grpc.aio.ServicerContext
    ) -> stream_process_pb2.STTCapabilities:
        """Return STT service capabilities."""
        return stream_process_pb2.STTCapabilities(
            available_models=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
            supported_languages=["en", "ko", "ja", "zh", "es", "fr", "de", "ru", "ar", "hi"],
            supported_encodings=["LINEAR16", "FLAC", "MP3", "OPUS"],
            max_stream_duration_seconds=3600,
            max_concurrent_streams=settings.max_concurrent_streams
        )


class OCRServicer(stream_process_pb2_grpc.OCRServiceServicer):
    """OCR service implementation."""
    
    def __init__(self, queue: RedisQueue):
        self.queue = queue
        self.backpressure = BackpressureManager(
            high_watermark=settings.backpressure_high_watermark,
            low_watermark=settings.backpressure_low_watermark
        )
    
    async def ProcessDocument(
        self,
        request: stream_process_pb2.DocumentRequest,
        context: grpc.aio.ServicerContext
    ) -> stream_process_pb2.OCRResponse:
        """Process a single document."""
        job_id = str(uuid.uuid4())
        
        try:
            # Check backpressure
            if await self.backpressure.should_pause():
                logger.warning(f"OCR backpressure activated")
                await self.backpressure.wait_for_capacity()
            
            # Prepare job data
            job_data = {
                "type": "ocr_single",
                "job_id": job_id,
                "config": MessageToDict(request.config),
                "metadata": MessageToDict(request.metadata) if request.metadata else {},
                "timestamp": time.time()
            }
            
            # Add image content or URL
            if request.HasField("image_content"):
                job_data["image_content"] = request.image_content
            elif request.HasField("document_url"):
                job_data["document_url"] = request.document_url
            
            # Queue job
            priority = request.metadata.priority if request.metadata else 1
            await self.queue.add_job(job_data, priority=priority)
            
            # Wait for result
            result = await self.queue.wait_for_result(job_id, timeout=30)
            
            if result:
                return stream_process_pb2.OCRResponse(
                    session_id=job_id,
                    status=stream_process_pb2.STATUS_COMPLETED,
                    result=result.get("ocr_result"),
                    metrics=result.get("metrics")
                )
            else:
                return stream_process_pb2.OCRResponse(
                    session_id=job_id,
                    status=stream_process_pb2.STATUS_FAILED,
                    error=stream_process_pb2.ErrorDetail(
                        code="TIMEOUT",
                        message="OCR processing timeout"
                    )
                )
        
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return stream_process_pb2.OCRResponse(
                session_id=job_id,
                status=stream_process_pb2.STATUS_FAILED,
                error=stream_process_pb2.ErrorDetail(
                    code="INTERNAL_ERROR",
                    message=str(e)
                )
            )
    
    async def BatchProcess(
        self,
        request_iterator: AsyncIterator[stream_process_pb2.DocumentRequest],
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[stream_process_pb2.OCRResponse]:
        """Handle batch OCR processing with streaming."""
        batch_id = str(uuid.uuid4())
        active_streams.labels(service="ocr").inc()
        
        try:
            async for request in request_iterator:
                # Check backpressure
                if await self.backpressure.should_pause():
                    await self.backpressure.wait_for_capacity()
                
                # Process each document
                job_id = f"{batch_id}_{uuid.uuid4()}"
                
                job_data = {
                    "type": "ocr_batch",
                    "job_id": job_id,
                    "batch_id": batch_id,
                    "config": MessageToDict(request.config),
                    "metadata": MessageToDict(request.metadata) if request.metadata else {},
                    "timestamp": time.time()
                }
                
                if request.HasField("image_content"):
                    job_data["image_content"] = request.image_content
                elif request.HasField("document_url"):
                    job_data["document_url"] = request.document_url
                
                # Queue job
                priority = request.metadata.priority if request.metadata else 1
                await self.queue.add_job(job_data, priority=priority)
                
                # Yield results as they become available
                result = await self.queue.wait_for_result(job_id, timeout=30)
                if result:
                    yield stream_process_pb2.OCRResponse(
                        session_id=job_id,
                        status=stream_process_pb2.STATUS_COMPLETED,
                        result=result.get("ocr_result"),
                        metrics=result.get("metrics")
                    )
        
        except Exception as e:
            logger.error(f"Error in batch OCR processing: {e}")
            yield stream_process_pb2.OCRResponse(
                session_id=batch_id,
                status=stream_process_pb2.STATUS_FAILED,
                error=stream_process_pb2.ErrorDetail(
                    code="INTERNAL_ERROR",
                    message=str(e)
                )
            )
        finally:
            active_streams.labels(service="ocr").dec()
    
    async def GetCapabilities(
        self,
        request: stream_process_pb2.GetCapabilitiesRequest,
        context: grpc.aio.ServicerContext
    ) -> stream_process_pb2.OCRCapabilities:
        """Return OCR service capabilities."""
        return stream_process_pb2.OCRCapabilities(
            supported_languages=[
                "ch_sim", "en", "ja", "ko", "es", "fr", "de", "ru", "ar", "hi",
                "pt", "it", "tr", "vi", "th", "id", "ms", "fa", "ur", "sw"
            ],
            supported_formats=["PNG", "JPG", "JPEG", "BMP", "TIFF", "PDF"],
            max_image_size_mb=50,
            max_concurrent_requests=100,
            gpu_available=settings.ocr_use_gpu
        )


async def serve():
    """Start the gRPC server."""
    # Initialize queue
    queue = RedisQueue(settings)
    await queue.initialize()
    
    # Initialize session manager
    session_manager = SessionManager(queue)
    
    # Start metrics server
    if settings.enable_metrics:
        start_http_server(settings.metrics_port)
        logger.info(f"Metrics server started on port {settings.metrics_port}")
    
    # Create gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=settings.worker_concurrency),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ('grpc.max_concurrent_streams', settings.max_concurrent_streams),
        ]
    )
    
    # Add servicers
    stream_process_pb2_grpc.add_STTServiceServicer_to_server(
        STTServicer(queue, session_manager), server
    )
    stream_process_pb2_grpc.add_OCRServiceServicer_to_server(
        OCRServicer(queue), server
    )
    
    # Start server
    listen_addr = f"{settings.grpc_server_host}:{settings.grpc_server_port}"
    server.add_insecure_port(listen_addr)
    
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        await server.stop(5)


def MessageToDict(message) -> Dict[str, Any]:
    """Convert protobuf message to dictionary."""
    from google.protobuf.json_format import MessageToDict as _MessageToDict
    return _MessageToDict(message, preserving_proto_field_name=True)


if __name__ == "__main__":
    asyncio.run(serve())