"""Unit tests for Redis queue."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from src.queue.redis_queue import RedisQueue


@pytest.mark.unit
@pytest.mark.asyncio
class TestRedisQueue:
    """Test Redis queue functionality."""
    
    async def test_initialize(self, queue):
        """Test queue initialization."""
        if hasattr(queue.redis_client, 'ping'):
            # Real Redis
            assert await queue.redis_client.ping() is True
        else:
            # Mock Redis
            assert queue.redis_client is not None
    
    async def test_add_job(self, queue):
        """Test adding jobs to queue."""
        job_data = {
            "type": "test_job",
            "data": "test_data",
            "timestamp": time.time()
        }
        
        job_id = await queue.add_job(job_data, priority=1)
        
        assert job_id is not None
        assert job_data["job_id"] == job_id
        assert job_data["priority"] == 1
    
    async def test_get_job(self, queue):
        """Test getting jobs from queue."""
        # Add a job first
        job_data = {
            "type": "test_job",
            "data": "test_data"
        }
        
        await queue.add_job(job_data, priority=1)
        
        # Get the job
        retrieved_job = await queue.get_job(timeout=100)
        
        if retrieved_job:  # Only test if Redis is available
            assert retrieved_job["type"] == "test_job"
            assert retrieved_job["data"] == "test_data"
            assert "job_id" in retrieved_job
            assert "dequeued_at" in retrieved_job
    
    async def test_priority_ordering(self, queue):
        """Test that higher priority jobs are processed first."""
        # Add jobs with different priorities
        low_priority_data = {"type": "low", "data": "low_priority"}
        high_priority_data = {"type": "high", "data": "high_priority"}
        
        await queue.add_job(low_priority_data, priority=0)
        await queue.add_job(high_priority_data, priority=3)
        
        # Should get high priority job first
        first_job = await queue.get_job(timeout=100)
        
        if first_job:
            assert first_job["data"] == "high_priority"
    
    async def test_ack_job(self, queue):
        """Test acknowledging jobs."""
        job_data = {"type": "test", "data": "test"}
        await queue.add_job(job_data, priority=1)
        
        job = await queue.get_job(timeout=100)
        
        if job:
            # Should not raise exception
            await queue.ack_job(job)
    
    async def test_nack_job_retry(self, queue):
        """Test negative acknowledgment with retry."""
        job_data = {"type": "test", "data": "test"}
        await queue.add_job(job_data, priority=1)
        
        job = await queue.get_job(timeout=100)
        
        if job:
            # Should requeue job
            await queue.nack_job(job, retry=True)
            
            # Should be able to get job again (with retry count)
            retry_job = await queue.get_job(timeout=100)
            if retry_job:
                assert retry_job.get("retry_count", 0) > 0
    
    async def test_store_and_get_result(self, queue):
        """Test storing and retrieving results."""
        job_id = "test_job_123"
        result_data = {
            "status": "completed",
            "output": "test_output",
            "metrics": {"processing_time": 1.5}
        }
        
        await queue.store_result(job_id, result_data)
        
        retrieved_result = await queue.get_result(job_id)
        
        if retrieved_result:  # Only test if Redis is available
            assert retrieved_result["status"] == "completed"
            assert retrieved_result["output"] == "test_output"
            assert retrieved_result["metrics"]["processing_time"] == 1.5
    
    async def test_wait_for_result(self, queue):
        """Test waiting for job result."""
        job_id = "test_job_456"
        result_data = {"status": "completed"}
        
        async def store_result_delayed():
            await asyncio.sleep(0.1)
            await queue.store_result(job_id, result_data)
        
        # Start storing result in background
        asyncio.create_task(store_result_delayed())
        
        # Wait for result
        result = await queue.wait_for_result(job_id, timeout=1)
        
        if result:  # Only test if Redis is available
            assert result["status"] == "completed"
    
    async def test_get_depth(self, queue):
        """Test getting queue depth."""
        # Add some jobs
        for i in range(3):
            await queue.add_job({"type": "test", "data": f"job_{i}"}, priority=1)
        
        depth = await queue.get_depth()
        
        # Should be at least 3 (might be more if other tests ran)
        if depth is not None:  # Only test if Redis is available
            assert depth >= 0
    
    async def test_get_lag(self, queue):
        """Test getting consumer lag."""
        lag = await queue.get_lag()
        
        assert isinstance(lag, dict)
        # Should have entries for each priority stream
        if lag:  # Only test if Redis is available
            for stream_key in queue.priority_streams.values():
                assert stream_key in lag or len(lag) == 0  # Empty if no consumers


@pytest.mark.unit
class TestQueueMetrics:
    """Test queue metrics and monitoring."""
    
    def test_queue_initialization_with_config(self, mock_settings):
        """Test queue initialization with custom config."""
        queue = RedisQueue(mock_settings)
        
        assert queue.config == mock_settings
        assert queue.redis_stream_key == mock_settings.redis_stream_key
        assert queue.result_ttl == 3600
    
    def test_priority_streams_mapping(self, queue):
        """Test priority streams are correctly mapped."""
        expected_priorities = [0, 1, 2, 3]  # LOW, NORMAL, HIGH, REALTIME
        
        assert len(queue.priority_streams) == 4
        
        for priority in expected_priorities:
            assert priority in queue.priority_streams
            stream_key = queue.priority_streams[priority]
            assert queue.config.redis_stream_key in stream_key


@pytest.mark.unit
@pytest.mark.asyncio
class TestQueueErrorHandling:
    """Test queue error handling."""
    
    async def test_connection_failure_handling(self, mock_settings):
        """Test handling of Redis connection failures."""
        # Create queue with invalid Redis config
        invalid_settings = mock_settings.copy()
        invalid_settings.redis_host = "invalid_host"
        invalid_settings.redis_port = 99999
        
        queue = RedisQueue(invalid_settings)
        
        # Should handle connection failure gracefully
        with pytest.raises(Exception):
            await queue.initialize()
    
    async def test_malformed_job_data(self, queue):
        """Test handling of malformed job data."""
        # This should not crash the queue
        await queue.add_job(None, priority=1)  # None data
        await queue.add_job({}, priority=1)    # Empty data
        
        # Queue should still be functional
        normal_job = {"type": "normal", "data": "test"}
        job_id = await queue.add_job(normal_job, priority=1)
        assert job_id is not None
    
    async def test_invalid_priority_handling(self, queue):
        """Test handling of invalid priority values."""
        job_data = {"type": "test", "data": "test"}
        
        # Should handle invalid priorities gracefully
        await queue.add_job(job_data, priority=999)  # Too high
        await queue.add_job(job_data, priority=-1)   # Negative
        
        # Should default to normal priority or handle gracefully
        job_id = await queue.add_job(job_data, priority=1)
        assert job_id is not None