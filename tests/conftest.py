"""Test configuration and fixtures."""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
import numpy as np
from PIL import Image
import redis.asyncio as redis
from unittest.mock import MagicMock

from src.config import settings
from src.queue.redis_queue import RedisQueue
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.image_processor import ImageProcessor


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def redis_client():
    """Create Redis client for testing."""
    client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db + 1,  # Use different DB for tests
        decode_responses=False
    )
    
    try:
        await client.ping()
        yield client
    except redis.ConnectionError:
        # Mock Redis if not available
        yield MagicMock()
    finally:
        try:
            await client.flushdb()
            await client.close()
        except:
            pass


@pytest.fixture
async def queue(redis_client):
    """Create queue instance for testing."""
    # Use test settings
    test_settings = settings.copy()
    test_settings.redis_db = settings.redis_db + 1
    
    queue = RedisQueue(test_settings)
    if not isinstance(redis_client, MagicMock):
        await queue.initialize()
    
    yield queue
    
    try:
        await queue.close()
    except:
        pass


@pytest.fixture
def audio_processor():
    """Create audio processor instance."""
    return AudioProcessor()


@pytest.fixture
def image_processor():
    """Create image processor instance."""
    return ImageProcessor()


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    # Generate 1 second of sine wave at 440Hz
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return audio, sample_rate


@pytest.fixture
def sample_image():
    """Generate sample image for testing."""
    # Create a simple test image with text
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='white')
    
    # Convert to numpy array
    return np.array(image)


@pytest.fixture
def temp_audio_file(sample_audio):
    """Create temporary audio file."""
    audio, sample_rate = sample_audio
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Simple WAV header (44 bytes)
        import struct
        
        # WAV header
        header = struct.pack('<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + len(audio_int16) * 2,  # File size - 8
            b'WAVE',
            b'fmt ',
            16,  # PCM format chunk size
            1,   # PCM format
            1,   # Mono
            sample_rate,
            sample_rate * 2,  # Byte rate
            2,   # Block align
            16,  # Bits per sample
            b'data',
            len(audio_int16) * 2  # Data size
        )
        
        f.write(header)
        f.write(audio_int16.tobytes())
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass


@pytest.fixture
def temp_image_file(sample_image):
    """Create temporary image file."""
    image = Image.fromarray(sample_image.astype(np.uint8))
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        image.save(f.name, 'PNG')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass


@pytest.fixture
def mock_whisper_model():
    """Mock Faster-Whisper model."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [
            MagicMock(
                text="Hello world",
                start=0.0,
                end=1.0,
                confidence=0.95
            )
        ],
        MagicMock(
            language="en",
            language_probability=0.95
        )
    )
    return mock_model


@pytest.fixture
def mock_ocr_engine():
    """Mock PaddleOCR engine."""
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [
        [
            [
                [[100, 100], [200, 100], [200, 150], [100, 150]],  # Bounding box
                ["Hello world", 0.95]  # Text and confidence
            ]
        ]
    ]
    return mock_ocr


@pytest.fixture(autouse=True)
def cleanup_logs():
    """Cleanup test logs after each test."""
    yield
    
    # Remove test log files
    log_dir = Path("logs")
    if log_dir.exists():
        for log_file in log_dir.glob("test_*.log"):
            try:
                log_file.unlink()
            except:
                pass


@pytest.fixture
def grpc_channel():
    """Mock gRPC channel for testing."""
    return MagicMock()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    test_settings = settings.copy()
    test_settings.redis_db = settings.redis_db + 1
    test_settings.log_level = "DEBUG"
    test_settings.enable_metrics = False
    return test_settings


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test location
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)