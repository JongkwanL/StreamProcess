"""STT service implementation with Faster-Whisper."""

import asyncio
import io
import time
from typing import List, Optional, Dict, Any, AsyncGenerator
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel
from loguru import logger
import torch
import torchaudio
from prometheus_client import Counter, Histogram, Gauge

from ..config import settings


# Metrics
stt_requests_total = Counter('stt_requests_total', 'Total STT requests', ['model', 'language'])
stt_processing_time = Histogram('stt_processing_duration_seconds', 'STT processing time')
stt_active_sessions = Gauge('stt_active_sessions', 'Active STT streaming sessions')
stt_model_load_time = Histogram('stt_model_load_duration_seconds', 'Model loading time')


@dataclass
class STTConfig:
    """STT configuration."""
    model_size: str = "base"
    device: str = "auto"
    compute_type: str = "float16"
    cpu_threads: int = 4
    beam_size: int = 5
    language: Optional[str] = None
    task: str = "transcribe"
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: Optional[str] = None


@dataclass
class STTResult:
    """STT processing result."""
    text: str
    language: str
    language_probability: float
    segments: List[Dict[str, Any]]
    duration: float
    processing_time: float
    model_size: str
    confidence: float


class STTService:
    """Speech-to-Text service using Faster-Whisper."""
    
    def __init__(self, config: STTConfig = None):
        self.config = config or STTConfig()
        self.model: Optional[WhisperModel] = None
        self.model_lock = asyncio.Lock()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize the STT service."""
        logger.info("Initializing STT service...")
        
        with stt_model_load_time.time():
            # Determine device
            if self.config.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.config.device
                
            # Load model
            model_path = Path(settings.whisper_model_path) / self.config.model_size
            if model_path.exists():
                # Use local model
                self.model = WhisperModel(
                    str(model_path),
                    device=device,
                    compute_type=self.config.compute_type,
                    cpu_threads=self.config.cpu_threads
                )
            else:
                # Download model
                self.model = WhisperModel(
                    self.config.model_size,
                    device=device,
                    compute_type=self.config.compute_type,
                    cpu_threads=self.config.cpu_threads,
                    download_root=settings.whisper_model_path
                )
        
        logger.info(f"STT service initialized with model: {self.config.model_size}, device: {device}")
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> STTResult:
        """Transcribe audio data."""
        start_time = time.time()
        
        with stt_processing_time.time():
            # Update metrics
            stt_requests_total.labels(
                model=self.config.model_size,
                language=language or "auto"
            ).inc()
            
            # Ensure model is loaded
            if self.model is None:
                await self.initialize()
            
            # Convert audio data to numpy array
            audio_array = await self._prepare_audio(audio_data)
            
            # Apply configuration overrides
            transcribe_config = self._build_transcribe_config(language, config_override)
            
            # Perform transcription
            segments, info = self.model.transcribe(
                audio_array,
                **transcribe_config
            )
            
            # Process segments
            segments_list = []
            full_text = ""
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                segment_dict = {
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob
                }
                segments_list.append(segment_dict)
                full_text += segment.text
                
                # Calculate confidence (1 - avg_logprob normalized)
                confidence = max(0.0, min(1.0, 1.0 + segment.avg_logprob))
                total_confidence += confidence
                segment_count += 1
            
            # Calculate overall confidence
            avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0
            
            processing_time = time.time() - start_time
            
            return STTResult(
                text=full_text.strip(),
                language=info.language,
                language_probability=info.language_probability,
                segments=segments_list,
                duration=info.duration,
                processing_time=processing_time,
                model_size=self.config.model_size,
                confidence=avg_confidence
            )
    
    async def start_streaming_session(
        self,
        session_id: str,
        language: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a streaming transcription session."""
        stt_active_sessions.inc()
        
        session_config = self._build_transcribe_config(language, config_override)
        
        self.active_sessions[session_id] = {
            "config": session_config,
            "audio_buffer": [],
            "last_transcription": "",
            "start_time": time.time(),
            "total_duration": 0.0
        }
        
        logger.info(f"Started streaming session: {session_id}")
        return session_id
    
    async def process_audio_chunk(
        self,
        session_id: str,
        audio_chunk: bytes,
        is_final: bool = False
    ) -> Optional[STTResult]:
        """Process audio chunk for streaming session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session["audio_buffer"].append(audio_chunk)
        
        # Only transcribe on final chunk or when buffer is large enough
        if not is_final and len(session["audio_buffer"]) < 10:  # ~1 second at 100ms chunks
            return None
        
        # Combine audio chunks
        combined_audio = b"".join(session["audio_buffer"])
        
        try:
            # Transcribe combined audio
            result = await self.transcribe_audio(
                combined_audio,
                language=session["config"].get("language"),
                config_override=session["config"]
            )
            
            # Update session
            session["last_transcription"] = result.text
            session["total_duration"] += result.duration
            
            # Clear buffer for partial results
            if not is_final:
                session["audio_buffer"] = session["audio_buffer"][-5:]  # Keep some overlap
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for session {session_id}: {e}")
            return None
    
    async def end_streaming_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End streaming session and return final statistics."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions.pop(session_id)
        stt_active_sessions.dec()
        
        session_stats = {
            "session_id": session_id,
            "total_duration": session["total_duration"],
            "session_time": time.time() - session["start_time"],
            "final_transcription": session["last_transcription"]
        }
        
        logger.info(f"Ended streaming session: {session_id}")
        return session_stats
    
    async def _prepare_audio(self, audio_data: bytes) -> np.ndarray:
        """Prepare audio data for transcription."""
        try:
            # Try to load as WAV first
            audio_io = io.BytesIO(audio_data)
            waveform, sample_rate = torchaudio.load(audio_io)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to numpy
            audio_array = waveform.squeeze().numpy()
            
            # Normalize
            audio_array = audio_array / np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else audio_array
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error preparing audio: {e}")
            # Fallback: assume raw 16kHz mono float32
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            return audio_array / np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else audio_array
    
    def _build_transcribe_config(
        self,
        language: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build transcription configuration."""
        config = {
            "beam_size": self.config.beam_size,
            "language": language or self.config.language,
            "task": self.config.task,
            "temperature": self.config.temperature,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "log_prob_threshold": self.config.log_prob_threshold,
            "no_speech_threshold": self.config.no_speech_threshold,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "initial_prompt": self.config.initial_prompt,
            "word_timestamps": True,  # Enable word-level timestamps
        }
        
        # Apply overrides
        if config_override:
            config.update(config_override)
        
        return config
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            await self.initialize()
        
        return {
            "model_size": self.config.model_size,
            "device": self.model.device,
            "compute_type": self.config.compute_type,
            "languages": ["en", "ko", "ja", "zh", "es", "fr", "de", "it", "pt", "ru"],  # Supported languages
            "active_sessions": len(self.active_sessions)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy" if self.model is not None else "initializing",
            "model_loaded": self.model is not None,
            "active_sessions": len(self.active_sessions),
            "model_size": self.config.model_size
        }