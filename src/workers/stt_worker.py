"""STT Worker implementation with Faster-Whisper."""

import asyncio
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from io import BytesIO
import soundfile as sf
from faster_whisper import WhisperModel
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge
import torch

from ..config import settings
from ..queue.redis_queue import RedisQueue
from ..preprocessing.audio_processor import AudioProcessor
from .batch_aggregator import BatchAggregator, EarliestDeadlineFirst


# Metrics
stt_processed = Counter(
    "stt_processed_total",
    "Total STT jobs processed",
    ["status", "model"]
)
stt_processing_time = Histogram(
    "stt_processing_seconds",
    "STT processing time",
    ["model", "batch_size"]
)
stt_rtf = Histogram(
    "stt_real_time_factor",
    "Real-time factor (processing time / audio duration)",
    ["model"]
)
active_stt_workers = Gauge(
    "active_stt_workers",
    "Number of active STT workers"
)


class STTWorker:
    """Worker for processing STT jobs."""
    
    def __init__(self, worker_id: str, queue: RedisQueue):
        self.worker_id = worker_id
        self.queue = queue
        self.model: Optional[WhisperModel] = None
        self.audio_processor = AudioProcessor()
        self.batch_aggregator: Optional[BatchAggregator] = None
        
        # Session state for streaming
        self.sessions: Dict[str, SessionState] = {}
        
        # Performance tracking
        self._jobs_processed = 0
        self._total_audio_ms = 0
        self._total_processing_ms = 0
        
        active_stt_workers.inc()
    
    async def initialize(self):
        """Initialize the worker."""
        try:
            # Load Whisper model
            logger.info(f"Loading Whisper model: {settings.whisper_model_size}")
            
            self.model = WhisperModel(
                settings.get_whisper_model_path(),
                device=settings.whisper_device.value,
                compute_type=settings.whisper_compute_type.value,
                cpu_threads=4 if settings.whisper_device.value == "cpu" else 0,
                num_workers=2
            )
            
            logger.info(f"Model loaded on {settings.whisper_device.value}")
            
            # Initialize batch aggregator
            self.batch_aggregator = BatchAggregator(
                max_batch_size=settings.stt_batch_size,
                max_wait_ms=settings.stt_batch_timeout_ms,
                scheduler=EarliestDeadlineFirst()
            )
            
            # Start batch processor
            asyncio.create_task(self._batch_processor())
            
        except Exception as e:
            logger.error(f"Failed to initialize STT worker: {e}")
            raise
    
    async def run(self):
        """Main worker loop."""
        logger.info(f"STT Worker {self.worker_id} started")
        
        try:
            while True:
                # Get job from queue
                job = await self.queue.get_job(timeout=1000)
                
                if job:
                    await self._process_job(job)
                
                # Periodic cleanup
                if self._jobs_processed % 100 == 0:
                    await self._cleanup_sessions()
        
        except KeyboardInterrupt:
            logger.info(f"STT Worker {self.worker_id} shutting down")
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            active_stt_workers.dec()
    
    async def _process_job(self, job: Dict[str, Any]):
        """Process a single job."""
        job_type = job.get("type")
        
        try:
            if job_type == "stt_chunk":
                await self._process_streaming_chunk(job)
            elif job_type == "stt_batch":
                await self._process_batch_audio(job)
            else:
                logger.warning(f"Unknown job type: {job_type}")
                await self.queue.nack_job(job, retry=False)
                return
            
            # Acknowledge job
            await self.queue.ack_job(job)
            self._jobs_processed += 1
            stt_processed.labels(status="success", model=settings.whisper_model_size.value).inc()
            
        except Exception as e:
            logger.error(f"Failed to process job: {e}")
            await self.queue.nack_job(job, retry=True)
            stt_processed.labels(status="failure", model=settings.whisper_model_size.value).inc()
    
    async def _process_streaming_chunk(self, job: Dict[str, Any]):
        """Process a streaming audio chunk."""
        session_id = job.get("session_id")
        audio_data = job.get("audio")
        offset_ms = job.get("offset_ms", 0)
        is_final = job.get("is_final", False)
        config = job.get("config", {})
        
        # Get or create session state
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(
                session_id=session_id,
                config=config,
                audio_processor=self.audio_processor
            )
        
        session = self.sessions[session_id]
        
        # Add audio chunk to session buffer
        session.add_audio_chunk(audio_data, offset_ms)
        
        # Check if we should process (VAD or chunk size threshold)
        if session.should_process() or is_final:
            # Get audio for processing
            audio_segment = session.get_processing_segment()
            
            if audio_segment is not None and len(audio_segment) > 0:
                # Add to batch aggregator
                await self.batch_aggregator.add_item({
                    "session_id": session_id,
                    "audio": audio_segment,
                    "offset_ms": session.last_processed_offset,
                    "is_partial": not is_final,
                    "config": config,
                    "deadline": time.time() + 0.3  # 300ms deadline for partials
                })
        
        # Clean up if final
        if is_final:
            await self._finalize_session(session_id)
    
    async def _process_batch_audio(self, job: Dict[str, Any]):
        """Process a complete audio file."""
        job_id = job.get("job_id")
        audio_data = job.get("audio")
        audio_config = job.get("audio_config", {})
        recognition_config = job.get("recognition_config", {})
        
        start_time = time.time()
        
        try:
            # Decode audio
            audio_array, sample_rate = self._decode_audio(audio_data, audio_config)
            
            # Preprocess
            processed_audio = self.audio_processor.preprocess(
                audio_array,
                sample_rate,
                target_sample_rate=16000
            )
            
            # Transcribe
            segments, info = self.model.transcribe(
                processed_audio,
                beam_size=recognition_config.get("beam_size", settings.whisper_beam_size),
                language=audio_config.get("language_code", None),
                temperature=recognition_config.get("temperature", settings.whisper_temperature),
                vad_filter=settings.whisper_vad_filter,
                vad_parameters={
                    "threshold": settings.whisper_vad_threshold,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": settings.vad_silence_duration_ms
                }
            )
            
            # Format results
            results = []
            full_transcript = []
            
            for segment in segments:
                results.append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": getattr(segment, "confidence", 0.0)
                })
                full_transcript.append(segment.text)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            audio_duration = len(processed_audio) / 16000
            rtf = processing_time / audio_duration if audio_duration > 0 else 0
            
            # Store result
            await self.queue.store_result(job_id, {
                "results": [{
                    "alternatives": [{
                        "transcript": " ".join(full_transcript),
                        "confidence": info.language_probability if info else 0.0
                    }],
                    "language_code": info.language if info else audio_config.get("language_code", "unknown")
                }],
                "processing_time_ms": int(processing_time * 1000),
                "audio_duration_ms": int(audio_duration * 1000),
                "real_time_factor": rtf
            })
            
            # Update metrics
            stt_processing_time.labels(
                model=settings.whisper_model_size.value,
                batch_size=1
            ).observe(processing_time)
            stt_rtf.labels(model=settings.whisper_model_size.value).observe(rtf)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            raise
    
    async def _batch_processor(self):
        """Process batched items."""
        while True:
            try:
                # Get batch from aggregator
                batch = await self.batch_aggregator.get_batch()
                
                if batch:
                    await self._process_batch(batch)
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy loop
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of audio segments."""
        if not batch:
            return
        
        start_time = time.time()
        batch_size = len(batch)
        
        try:
            # Group by similar lengths for better batching
            buckets = self._bucket_by_length(batch)
            
            for bucket_items in buckets.values():
                # Process bucket
                results = await self._process_bucket(bucket_items)
                
                # Store results
                for item, result in zip(bucket_items, results):
                    session_id = item["session_id"]
                    
                    # Store result for session
                    if session_id in self.sessions:
                        self.sessions[session_id].add_result(result)
                    
                    # Store in Redis for retrieval
                    await self.queue.store_result(
                        f"{session_id}_{item['offset_ms']}",
                        result
                    )
            
            # Update metrics
            processing_time = time.time() - start_time
            stt_processing_time.labels(
                model=settings.whisper_model_size.value,
                batch_size=batch_size
            ).observe(processing_time)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    async def _process_bucket(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a bucket of similar-length audio."""
        results = []
        
        # Pad audio to same length
        max_length = max(len(item["audio"]) for item in items)
        padded_batch = []
        
        for item in items:
            audio = item["audio"]
            if len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
            padded_batch.append(audio)
        
        # Stack for batch processing
        batch_audio = np.stack(padded_batch)
        
        # Process batch through encoder (if model supports it)
        # Note: Faster-Whisper doesn't natively support batching,
        # so we process sequentially but could optimize with custom implementation
        for i, item in enumerate(items):
            audio = item["audio"]
            is_partial = item.get("is_partial", False)
            
            # Transcribe with appropriate settings for partial/final
            segments, info = self.model.transcribe(
                audio,
                beam_size=1 if is_partial else settings.whisper_beam_size,
                language=item.get("config", {}).get("language_code"),
                temperature=0.0 if is_partial else settings.whisper_temperature,
                vad_filter=False if is_partial else settings.whisper_vad_filter,
                without_timestamps=is_partial
            )
            
            # Format result
            transcript = " ".join(segment.text for segment in segments)
            
            results.append({
                "transcript": transcript,
                "is_partial": is_partial,
                "offset_ms": item["offset_ms"],
                "confidence": info.language_probability if info else 0.0,
                "language": info.language if info else "unknown"
            })
        
        return results
    
    def _bucket_by_length(
        self,
        items: List[Dict[str, Any]],
        bucket_boundaries: List[int] = [5120, 10240, 20480, 40960]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Bucket items by audio length for efficient batching."""
        buckets = {boundary: [] for boundary in bucket_boundaries}
        buckets[float('inf')] = []  # Catch-all bucket
        
        for item in items:
            audio_length = len(item["audio"])
            
            for boundary in bucket_boundaries:
                if audio_length <= boundary:
                    buckets[boundary].append(item)
                    break
            else:
                buckets[float('inf')].append(item)
        
        # Remove empty buckets
        return {k: v for k, v in buckets.items() if v}
    
    def _decode_audio(self, audio_data: bytes, config: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """Decode audio from various formats."""
        encoding = config.get("encoding", "LINEAR16")
        
        if encoding == "LINEAR16":
            # Raw PCM audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = config.get("sample_rate_hertz", 16000)
        else:
            # Use soundfile for other formats
            audio_file = BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_file)
        
        return audio_array, sample_rate
    
    async def _finalize_session(self, session_id: str):
        """Finalize and clean up a session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Process any remaining audio
            if session.has_unprocessed_audio():
                audio_segment = session.get_processing_segment()
                if audio_segment is not None:
                    await self.batch_aggregator.add_item({
                        "session_id": session_id,
                        "audio": audio_segment,
                        "offset_ms": session.last_processed_offset,
                        "is_partial": False,
                        "config": session.config,
                        "deadline": time.time() + 1.0  # 1s deadline for final
                    })
            
            # Clean up session after a delay
            await asyncio.sleep(5)  # Wait for final processing
            del self.sessions[session_id]
            logger.debug(f"Session {session_id} finalized")
    
    async def _cleanup_sessions(self):
        """Clean up old sessions."""
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_activity > 300:  # 5 minutes timeout
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            await self._finalize_session(session_id)
            logger.info(f"Cleaned up inactive session: {session_id}")


class SessionState:
    """Maintains state for a streaming session."""
    
    def __init__(self, session_id: str, config: Dict[str, Any], audio_processor: AudioProcessor):
        self.session_id = session_id
        self.config = config
        self.audio_processor = audio_processor
        
        # Audio buffer
        self.audio_buffer = []
        self.total_samples = 0
        self.last_processed_offset = 0
        self.last_activity = time.time()
        
        # VAD state
        self.vad_active = False
        self.silence_samples = 0
        self.speech_samples = 0
        
        # Results
        self.results = []
        self.transcript_buffer = []
    
    def add_audio_chunk(self, audio_data: bytes, offset_ms: int):
        """Add audio chunk to buffer."""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        self.audio_buffer.append(audio_array)
        self.total_samples += len(audio_array)
        self.last_activity = time.time()
        
        # Update VAD state
        if settings.whisper_vad_filter:
            self._update_vad_state(audio_array)
    
    def should_process(self) -> bool:
        """Check if we should process the current buffer."""
        # Check chunk size threshold
        if self.total_samples >= settings.audio_chunk_size_ms * 16:  # 16 samples per ms at 16kHz
            return True
        
        # Check VAD-based endpoint
        if self.vad_active and self.silence_samples > settings.vad_silence_duration_ms * 16:
            return True
        
        return False
    
    def get_processing_segment(self) -> Optional[np.ndarray]:
        """Get audio segment for processing."""
        if not self.audio_buffer:
            return None
        
        # Concatenate buffer
        audio = np.concatenate(self.audio_buffer)
        
        # Keep overlap for context
        overlap_samples = settings.audio_chunk_overlap_ms * 16
        
        if len(audio) > overlap_samples:
            # Process most of the audio, keep overlap
            process_length = len(audio) - overlap_samples
            segment = audio[:process_length]
            
            # Keep overlap in buffer
            self.audio_buffer = [audio[process_length:]]
            self.total_samples = overlap_samples
            self.last_processed_offset += process_length // 16  # Convert to ms
            
            return segment
        
        return audio
    
    def has_unprocessed_audio(self) -> bool:
        """Check if there's unprocessed audio in buffer."""
        return self.total_samples > 0
    
    def add_result(self, result: Dict[str, Any]):
        """Add transcription result."""
        self.results.append(result)
        
        # Update transcript buffer for context
        if result.get("transcript"):
            self.transcript_buffer.append(result["transcript"])
            
            # Keep only recent context
            if len(self.transcript_buffer) > 10:
                self.transcript_buffer = self.transcript_buffer[-10:]
    
    def _update_vad_state(self, audio: np.ndarray):
        """Update VAD state based on audio energy."""
        # Simple energy-based VAD
        energy = np.sqrt(np.mean(audio ** 2))
        
        if energy > settings.whisper_vad_threshold:
            self.vad_active = True
            self.speech_samples += len(audio)
            self.silence_samples = 0
        else:
            self.silence_samples += len(audio)
            
            # Deactivate VAD after sufficient silence
            if self.silence_samples > settings.vad_silence_duration_ms * 16:
                self.vad_active = False
                self.speech_samples = 0


async def main():
    """Main entry point for STT worker."""
    # Initialize queue
    queue = RedisQueue(settings)
    await queue.initialize()
    
    # Create and initialize worker
    worker = STTWorker(f"stt_worker_{time.time()}", queue)
    await worker.initialize()
    
    # Run worker
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())