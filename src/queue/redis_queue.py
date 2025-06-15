"""Redis Streams based queue implementation."""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
import redis.asyncio as redis
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram

from ..config import settings


# Metrics
queue_operations = Counter(
    "queue_operations_total",
    "Total queue operations",
    ["operation", "status"]
)
queue_depth_gauge = Gauge(
    "queue_depth",
    "Current queue depth",
    ["queue_name", "priority"]
)
queue_latency = Histogram(
    "queue_latency_seconds",
    "Queue operation latency",
    ["operation"]
)


class RedisQueue:
    """Redis Streams based queue with priority support."""
    
    def __init__(self, config: settings):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.consumer_name = f"worker_{datetime.now().timestamp()}"
        
        # Priority queues
        self.priority_streams = {
            3: f"{config.redis_stream_key}:realtime",  # PRIORITY_REALTIME
            2: f"{config.redis_stream_key}:high",      # PRIORITY_HIGH
            1: f"{config.redis_stream_key}:normal",    # PRIORITY_NORMAL
            0: f"{config.redis_stream_key}:low"        # PRIORITY_LOW
        }
        
        # Result storage
        self.result_key_prefix = "result:"
        self.result_ttl = 3600  # 1 hour
        
        # Statistics
        self._total_enqueued = 0
        self._total_processed = 0
        self._total_failed = 0
    
    async def initialize(self):
        """Initialize Redis connection and consumer groups."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False  # Handle binary data
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Create consumer groups for each priority stream
            for priority, stream_key in self.priority_streams.items():
                try:
                    await self.redis_client.xgroup_create(
                        stream_key,
                        self.config.redis_consumer_group,
                        id="0"
                    )
                    logger.info(f"Created consumer group for {stream_key}")
                except redis.ResponseError as e:
                    if "BUSYGROUP" in str(e):
                        logger.debug(f"Consumer group already exists for {stream_key}")
                    else:
                        raise
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis queue: {e}")
            raise
    
    async def add_job(
        self,
        job_data: Dict[str, Any],
        priority: int = 1,
        job_id: Optional[str] = None
    ) -> str:
        """
        Add a job to the queue.
        
        Args:
            job_data: Job data dictionary
            priority: Job priority (0-3)
            job_id: Optional job ID
        
        Returns:
            Job ID
        """
        if not job_id:
            job_id = f"job_{time.time()}_{self._total_enqueued}"
        
        # Add metadata
        job_data["job_id"] = job_id
        job_data["priority"] = priority
        job_data["enqueued_at"] = time.time()
        
        # Select stream based on priority
        stream_key = self.priority_streams.get(priority, self.priority_streams[1])
        
        try:
            # Serialize job data
            serialized = json.dumps(job_data).encode('utf-8')
            
            # Add to stream with max length limit
            message_id = await self.redis_client.xadd(
                stream_key,
                {"data": serialized},
                maxlen=self.config.redis_max_stream_length,
                approximate=True
            )
            
            self._total_enqueued += 1
            queue_operations.labels(operation="enqueue", status="success").inc()
            
            # Update depth metric
            depth = await self.get_depth(stream_key)
            queue_depth_gauge.labels(queue_name=stream_key, priority=priority).set(depth)
            
            logger.debug(f"Job {job_id} added to queue with priority {priority}")
            return job_id
            
        except Exception as e:
            queue_operations.labels(operation="enqueue", status="failure").inc()
            logger.error(f"Failed to add job to queue: {e}")
            raise
    
    async def get_job(self, timeout: int = 1000) -> Optional[Dict[str, Any]]:
        """
        Get a job from the queue (highest priority first).
        
        Args:
            timeout: Block timeout in milliseconds
        
        Returns:
            Job data or None
        """
        # Check queues in priority order
        for priority in sorted(self.priority_streams.keys(), reverse=True):
            stream_key = self.priority_streams[priority]
            
            try:
                # Try to claim pending messages first (for reliability)
                pending = await self.redis_client.xpending_range(
                    stream_key,
                    self.config.redis_consumer_group,
                    min="-",
                    max="+",
                    count=1
                )
                
                if pending:
                    # Claim the pending message
                    message_id = pending[0]["message_id"]
                    claimed = await self.redis_client.xclaim(
                        stream_key,
                        self.config.redis_consumer_group,
                        self.consumer_name,
                        min_idle_time=30000,  # 30 seconds
                        message_ids=[message_id]
                    )
                    
                    if claimed:
                        return await self._process_message(stream_key, claimed[0])
                
                # Read new messages
                messages = await self.redis_client.xreadgroup(
                    self.config.redis_consumer_group,
                    self.consumer_name,
                    {stream_key: ">"},
                    count=1,
                    block=0  # Non-blocking for priority iteration
                )
                
                if messages:
                    stream, stream_messages = messages[0]
                    if stream_messages:
                        return await self._process_message(stream_key, stream_messages[0])
            
            except Exception as e:
                logger.error(f"Error getting job from {stream_key}: {e}")
                continue
        
        # If no jobs in any priority queue, block on normal priority
        try:
            messages = await self.redis_client.xreadgroup(
                self.config.redis_consumer_group,
                self.consumer_name,
                {self.priority_streams[1]: ">"},
                count=1,
                block=timeout
            )
            
            if messages:
                stream, stream_messages = messages[0]
                if stream_messages:
                    return await self._process_message(
                        self.priority_streams[1],
                        stream_messages[0]
                    )
        
        except Exception as e:
            logger.error(f"Error in blocking read: {e}")
        
        return None
    
    async def _process_message(
        self,
        stream_key: str,
        message: tuple
    ) -> Dict[str, Any]:
        """Process a message from the stream."""
        message_id, data = message
        
        try:
            # Deserialize job data
            job_data = json.loads(data[b"data"])
            job_data["message_id"] = message_id.decode() if isinstance(message_id, bytes) else message_id
            job_data["stream_key"] = stream_key
            job_data["dequeued_at"] = time.time()
            
            # Calculate queue latency
            if "enqueued_at" in job_data:
                latency = job_data["dequeued_at"] - job_data["enqueued_at"]
                queue_latency.labels(operation="dequeue").observe(latency)
            
            queue_operations.labels(operation="dequeue", status="success").inc()
            return job_data
            
        except Exception as e:
            queue_operations.labels(operation="dequeue", status="failure").inc()
            logger.error(f"Failed to process message: {e}")
            raise
    
    async def ack_job(self, job_data: Dict[str, Any]):
        """Acknowledge job completion."""
        try:
            stream_key = job_data.get("stream_key")
            message_id = job_data.get("message_id")
            
            if stream_key and message_id:
                await self.redis_client.xack(
                    stream_key,
                    self.config.redis_consumer_group,
                    message_id
                )
                
                # Delete the message to keep stream size manageable
                await self.redis_client.xdel(stream_key, message_id)
                
                self._total_processed += 1
                queue_operations.labels(operation="ack", status="success").inc()
                logger.debug(f"Job acknowledged: {job_data.get('job_id')}")
        
        except Exception as e:
            queue_operations.labels(operation="ack", status="failure").inc()
            logger.error(f"Failed to acknowledge job: {e}")
    
    async def nack_job(self, job_data: Dict[str, Any], retry: bool = True):
        """Negative acknowledge (requeue or discard job)."""
        try:
            if retry and job_data.get("retry_count", 0) < self.config.worker_max_retries:
                # Increment retry count and requeue
                job_data["retry_count"] = job_data.get("retry_count", 0) + 1
                job_data["retry_at"] = time.time() + self.config.worker_retry_delay
                
                # Remove Redis-specific fields
                job_data.pop("message_id", None)
                job_data.pop("stream_key", None)
                job_data.pop("dequeued_at", None)
                
                # Requeue with lower priority
                priority = max(0, job_data.get("priority", 1) - 1)
                await self.add_job(job_data, priority=priority)
                
                logger.info(f"Job requeued: {job_data.get('job_id')} (retry {job_data['retry_count']})")
            else:
                # Max retries exceeded or retry disabled
                await self.ack_job(job_data)  # Remove from queue
                self._total_failed += 1
                logger.error(f"Job failed permanently: {job_data.get('job_id')}")
        
        except Exception as e:
            logger.error(f"Failed to nack job: {e}")
    
    async def store_result(self, job_id: str, result: Dict[str, Any]):
        """Store job result for retrieval."""
        try:
            key = f"{self.result_key_prefix}{job_id}"
            serialized = json.dumps(result)
            await self.redis_client.setex(key, self.result_ttl, serialized)
            logger.debug(f"Result stored for job {job_id}")
        
        except Exception as e:
            logger.error(f"Failed to store result: {e}")
    
    async def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job result."""
        try:
            key = f"{self.result_key_prefix}{job_id}"
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        
        except Exception as e:
            logger.error(f"Failed to get result: {e}")
            return None
    
    async def wait_for_result(
        self,
        job_id: str,
        timeout: int = 30,
        poll_interval: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """Wait for job result with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = await self.get_result(job_id)
            if result:
                return result
            await asyncio.sleep(poll_interval)
        
        return None
    
    async def get_depth(self, stream_key: Optional[str] = None) -> int:
        """Get queue depth."""
        try:
            if stream_key:
                info = await self.redis_client.xinfo_stream(stream_key)
                return info.get("length", 0)
            
            # Get total depth across all priority queues
            total = 0
            for stream in self.priority_streams.values():
                info = await self.redis_client.xinfo_stream(stream)
                total += info.get("length", 0)
            return total
        
        except Exception as e:
            logger.error(f"Failed to get queue depth: {e}")
            return 0
    
    async def get_lag(self) -> Dict[str, int]:
        """Get consumer lag for each priority queue."""
        lag = {}
        
        try:
            for priority, stream_key in self.priority_streams.items():
                pending = await self.redis_client.xpending(
                    stream_key,
                    self.config.redis_consumer_group
                )
                lag[stream_key] = pending.get("pending", 0) if pending else 0
            
            return lag
        
        except Exception as e:
            logger.error(f"Failed to get queue lag: {e}")
            return {}
    
    async def cleanup_old_messages(self, max_age_seconds: int = 3600):
        """Clean up old messages from streams."""
        try:
            cutoff_time = int((time.time() - max_age_seconds) * 1000)
            
            for stream_key in self.priority_streams.values():
                # Trim stream to remove old messages
                await self.redis_client.xtrim(
                    stream_key,
                    minid=f"{cutoff_time}-0",
                    approximate=True
                )
                logger.debug(f"Cleaned old messages from {stream_key}")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old messages: {e}")
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")