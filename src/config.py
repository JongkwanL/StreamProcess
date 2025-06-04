"""Configuration management for StreamProcess."""

from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from enum import Enum


class StorageType(str, Enum):
    MINIO = "minio"
    S3 = "s3"
    LOCAL = "local"


class WhisperModelSize(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class ComputeType(str, Enum):
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server Configuration
    grpc_server_host: str = Field(default="0.0.0.0", env="GRPC_SERVER_HOST")
    grpc_server_port: int = Field(default=50051, env="GRPC_SERVER_PORT")
    rest_server_host: str = Field(default="0.0.0.0", env="REST_SERVER_HOST")
    rest_server_port: int = Field(default=8000, env="REST_SERVER_PORT")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_stream_key: str = Field(default="stream_process_queue", env="REDIS_STREAM_KEY")
    redis_consumer_group: str = Field(default="workers", env="REDIS_CONSUMER_GROUP")
    redis_max_stream_length: int = Field(default=10000, env="REDIS_MAX_STREAM_LENGTH")
    
    # AWS/SQS Configuration
    use_sqs: bool = Field(default=False, env="USE_SQS")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    sqs_queue_url: Optional[str] = Field(default=None, env="SQS_QUEUE_URL")
    
    # Storage Configuration
    storage_type: StorageType = Field(default=StorageType.MINIO, env="STORAGE_TYPE")
    minio_endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="stream-process", env="MINIO_BUCKET")
    use_ssl: bool = Field(default=False, env="USE_SSL")
    local_storage_path: str = Field(default="/tmp/stream_process", env="LOCAL_STORAGE_PATH")
    
    # Whisper Model Configuration
    whisper_model_size: WhisperModelSize = Field(default=WhisperModelSize.BASE, env="WHISPER_MODEL_SIZE")
    whisper_device: DeviceType = Field(default=DeviceType.CPU, env="WHISPER_DEVICE")
    whisper_compute_type: ComputeType = Field(default=ComputeType.INT8, env="WHISPER_COMPUTE_TYPE")
    whisper_model_path: Optional[str] = Field(default=None, env="WHISPER_MODEL_PATH")
    whisper_beam_size: int = Field(default=1, env="WHISPER_BEAM_SIZE")  # 1 for streaming
    whisper_temperature: float = Field(default=0.0, env="WHISPER_TEMPERATURE")
    whisper_vad_filter: bool = Field(default=True, env="WHISPER_VAD_FILTER")
    whisper_vad_threshold: float = Field(default=0.5, env="WHISPER_VAD_THRESHOLD")
    
    # OCR Configuration
    ocr_languages: List[str] = Field(default=["ch_sim", "en"], env="OCR_LANG")
    ocr_use_gpu: bool = Field(default=False, env="OCR_USE_GPU")
    ocr_det_model_dir: Optional[str] = Field(default=None, env="OCR_DET_MODEL_DIR")
    ocr_rec_model_dir: Optional[str] = Field(default=None, env="OCR_REC_MODEL_DIR")
    ocr_cls_model_dir: Optional[str] = Field(default=None, env="OCR_CLS_MODEL_DIR")
    ocr_use_angle_cls: bool = Field(default=True, env="OCR_USE_ANGLE_CLS")
    ocr_det_db_thresh: float = Field(default=0.3, env="OCR_DET_DB_THRESH")
    ocr_det_db_box_thresh: float = Field(default=0.6, env="OCR_DET_DB_BOX_THRESH")
    
    # Worker Configuration
    worker_concurrency: int = Field(default=4, env="WORKER_CONCURRENCY")
    worker_max_retries: int = Field(default=3, env="WORKER_MAX_RETRIES")
    worker_retry_delay: int = Field(default=5, env="WORKER_RETRY_DELAY")
    worker_heartbeat_interval: int = Field(default=30, env="WORKER_HEARTBEAT_INTERVAL")
    worker_prefetch_count: int = Field(default=2, env="WORKER_PREFETCH_COUNT")
    
    # Batching Configuration
    batch_size: int = Field(default=8, env="BATCH_SIZE")
    batch_timeout: float = Field(default=1.0, env="BATCH_TIMEOUT")
    stt_batch_size: int = Field(default=16, env="STT_BATCH_SIZE")
    stt_batch_timeout_ms: int = Field(default=20, env="STT_BATCH_TIMEOUT_MS")
    ocr_batch_size: int = Field(default=32, env="OCR_BATCH_SIZE")
    ocr_batch_timeout_ms: int = Field(default=50, env="OCR_BATCH_TIMEOUT_MS")
    
    # Performance Settings
    max_concurrent_streams: int = Field(default=100, env="MAX_CONCURRENT_STREAMS")
    stream_buffer_size: int = Field(default=8192, env="STREAM_BUFFER_SIZE")
    chunk_duration_ms: int = Field(default=100, env="CHUNK_DURATION_MS")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    
    # Audio Streaming Configuration
    audio_chunk_size_ms: int = Field(default=320, env="AUDIO_CHUNK_SIZE_MS")
    audio_chunk_overlap_ms: int = Field(default=150, env="AUDIO_CHUNK_OVERLAP_MS")
    vad_silence_duration_ms: int = Field(default=300, env="VAD_SILENCE_DURATION_MS")
    partial_results_interval_ms: int = Field(default=150, env="PARTIAL_RESULTS_INTERVAL_MS")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    
    # Triton Inference Server
    use_triton: bool = Field(default=False, env="USE_TRITON")
    triton_server_url: str = Field(default="localhost:8001", env="TRITON_SERVER_URL")
    triton_model_name: str = Field(default="whisper_onnx", env="TRITON_MODEL_NAME")
    triton_model_version: str = Field(default="1", env="TRITON_MODEL_VERSION")
    
    # Autoscaling Configuration
    autoscale_enabled: bool = Field(default=True, env="AUTOSCALE_ENABLED")
    autoscale_min_workers: int = Field(default=1, env="AUTOSCALE_MIN_WORKERS")
    autoscale_max_workers: int = Field(default=10, env="AUTOSCALE_MAX_WORKERS")
    autoscale_target_utilization: float = Field(default=0.7, env="AUTOSCALE_TARGET_UTILIZATION")
    autoscale_scale_up_threshold: float = Field(default=0.8, env="AUTOSCALE_SCALE_UP_THRESHOLD")
    autoscale_scale_down_threshold: float = Field(default=0.3, env="AUTOSCALE_SCALE_DOWN_THRESHOLD")
    autoscale_warmup_time_seconds: int = Field(default=90, env="AUTOSCALE_WARMUP_TIME_SECONDS")
    autoscale_cooldown_seconds: int = Field(default=300, env="AUTOSCALE_COOLDOWN_SECONDS")
    
    # Backpressure Configuration
    backpressure_high_watermark: float = Field(default=0.8, env="BACKPRESSURE_HIGH_WATERMARK")
    backpressure_low_watermark: float = Field(default=0.6, env="BACKPRESSURE_LOW_WATERMARK")
    max_queue_lag_ms: int = Field(default=5000, env="MAX_QUEUE_LAG_MS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @validator("ocr_languages", pre=True)
    def parse_ocr_languages(cls, v):
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v
    
    @validator("whisper_compute_type")
    def validate_compute_type(cls, v, values):
        device = values.get("whisper_device")
        if device == DeviceType.CPU and v in [ComputeType.FLOAT16, ComputeType.INT8_FLOAT16]:
            raise ValueError(f"Compute type {v} is not supported on CPU")
        return v
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @property
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for processing."""
        return self.whisper_device == DeviceType.CUDA or self.ocr_use_gpu
    
    def get_whisper_model_path(self) -> str:
        """Get the full path to the Whisper model."""
        if self.whisper_model_path:
            return self.whisper_model_path
        return f"models/whisper/{self.whisper_model_size.value}"
    
    def get_ocr_model_paths(self) -> dict:
        """Get paths to OCR models."""
        base_path = "models/paddleocr"
        return {
            "det": self.ocr_det_model_dir or f"{base_path}/det",
            "rec": self.ocr_rec_model_dir or f"{base_path}/rec",
            "cls": self.ocr_cls_model_dir or f"{base_path}/cls"
        }


# Global settings instance
settings = Settings()