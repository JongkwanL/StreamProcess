"""Backpressure management for streaming pipeline."""

import asyncio
from typing import Optional
from loguru import logger
from prometheus_client import Gauge, Counter


# Metrics
backpressure_active = Gauge(
    "backpressure_active",
    "Whether backpressure is currently active",
    ["component"]
)
backpressure_events = Counter(
    "backpressure_events_total",
    "Total backpressure activation events",
    ["component", "action"]
)


class BackpressureManager:
    """Manages backpressure in the streaming pipeline."""
    
    def __init__(
        self,
        high_watermark: float = 0.8,
        low_watermark: float = 0.6,
        max_queue_depth: int = 1000,
        component_name: str = "default"
    ):
        """
        Initialize backpressure manager.
        
        Args:
            high_watermark: Threshold to activate backpressure (0.0-1.0)
            low_watermark: Threshold to deactivate backpressure (0.0-1.0)
            max_queue_depth: Maximum queue depth
            component_name: Name for metrics labeling
        """
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.max_queue_depth = max_queue_depth
        self.component_name = component_name
        
        self._is_paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Initially not paused
        
        # Current metrics
        self._current_depth = 0
        self._current_lag_ms = 0
        self._current_memory_usage = 0.0
        
        # Statistics
        self._total_pauses = 0
        self._total_pause_duration_ms = 0
        self._last_pause_time = None
    
    async def update_metrics(
        self,
        queue_depth: Optional[int] = None,
        lag_ms: Optional[int] = None,
        memory_usage: Optional[float] = None
    ):
        """Update current system metrics."""
        if queue_depth is not None:
            self._current_depth = queue_depth
        if lag_ms is not None:
            self._current_lag_ms = lag_ms
        if memory_usage is not None:
            self._current_memory_usage = memory_usage
        
        # Check if we should activate/deactivate backpressure
        await self._check_thresholds()
    
    async def _check_thresholds(self):
        """Check if backpressure should be activated or deactivated."""
        # Calculate utilization ratio
        utilization = self._current_depth / self.max_queue_depth if self.max_queue_depth > 0 else 0
        
        # Check high watermark
        if not self._is_paused and utilization >= self.high_watermark:
            await self._activate_backpressure()
        
        # Check low watermark
        elif self._is_paused and utilization <= self.low_watermark:
            await self._deactivate_backpressure()
    
    async def _activate_backpressure(self):
        """Activate backpressure."""
        self._is_paused = True
        self._pause_event.clear()
        self._total_pauses += 1
        self._last_pause_time = asyncio.get_event_loop().time()
        
        backpressure_active.labels(component=self.component_name).set(1)
        backpressure_events.labels(
            component=self.component_name,
            action="activate"
        ).inc()
        
        logger.warning(
            f"Backpressure activated for {self.component_name}: "
            f"depth={self._current_depth}/{self.max_queue_depth}, "
            f"lag={self._current_lag_ms}ms"
        )
    
    async def _deactivate_backpressure(self):
        """Deactivate backpressure."""
        if self._last_pause_time:
            pause_duration = (asyncio.get_event_loop().time() - self._last_pause_time) * 1000
            self._total_pause_duration_ms += pause_duration
        
        self._is_paused = False
        self._pause_event.set()
        
        backpressure_active.labels(component=self.component_name).set(0)
        backpressure_events.labels(
            component=self.component_name,
            action="deactivate"
        ).inc()
        
        logger.info(
            f"Backpressure deactivated for {self.component_name}: "
            f"depth={self._current_depth}/{self.max_queue_depth}"
        )
    
    async def should_pause(self) -> bool:
        """Check if processing should be paused due to backpressure."""
        return self._is_paused
    
    async def wait_for_capacity(self, timeout: Optional[float] = None):
        """
        Wait until backpressure is released.
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Raises:
            asyncio.TimeoutError: If timeout is exceeded
        """
        if not self._is_paused:
            return
        
        logger.debug(f"Waiting for backpressure release in {self.component_name}")
        
        try:
            await asyncio.wait_for(self._pause_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                f"Backpressure wait timeout in {self.component_name} after {timeout}s"
            )
            raise
    
    def get_statistics(self) -> dict:
        """Get backpressure statistics."""
        return {
            "is_paused": self._is_paused,
            "current_depth": self._current_depth,
            "max_depth": self.max_queue_depth,
            "utilization": self._current_depth / self.max_queue_depth if self.max_queue_depth > 0 else 0,
            "current_lag_ms": self._current_lag_ms,
            "total_pauses": self._total_pauses,
            "total_pause_duration_ms": self._total_pause_duration_ms,
            "high_watermark": self.high_watermark,
            "low_watermark": self.low_watermark
        }


class DegradationStrategy:
    """Implements degradation strategies under backpressure."""
    
    def __init__(self):
        self.degradation_level = 0  # 0: normal, 1: light, 2: moderate, 3: severe
        self._strategies = {
            "stt": {
                1: {  # Light degradation
                    "partial_interval_ms": 300,  # Reduce partial emission frequency
                    "chunk_size_ms": 500,  # Increase chunk size
                    "beam_size": 1,
                    "skip_punctuation": False
                },
                2: {  # Moderate degradation
                    "partial_interval_ms": 500,
                    "chunk_size_ms": 1000,
                    "sample_rate": 16000,  # Downsample
                    "beam_size": 1,
                    "skip_punctuation": True
                },
                3: {  # Severe degradation
                    "partial_interval_ms": 1000,
                    "chunk_size_ms": 2000,
                    "sample_rate": 8000,
                    "beam_size": 1,
                    "skip_punctuation": True,
                    "skip_language_detection": True
                }
            },
            "ocr": {
                1: {  # Light degradation
                    "max_resolution": 1920,
                    "skip_orientation_detection": False
                },
                2: {  # Moderate degradation  
                    "max_resolution": 1280,
                    "skip_orientation_detection": True,
                    "skip_layout_analysis": True
                },
                3: {  # Severe degradation
                    "max_resolution": 800,
                    "skip_orientation_detection": True,
                    "skip_layout_analysis": True,
                    "text_only": True  # Skip image regions
                }
            }
        }
    
    def get_degradation_params(self, service: str, utilization: float) -> dict:
        """
        Get degradation parameters based on utilization.
        
        Args:
            service: Service type ("stt" or "ocr")
            utilization: Current utilization ratio (0.0-1.0)
        
        Returns:
            Dictionary of degradation parameters
        """
        # Determine degradation level
        if utilization < 0.7:
            level = 0
        elif utilization < 0.8:
            level = 1
        elif utilization < 0.9:
            level = 2
        else:
            level = 3
        
        if level != self.degradation_level:
            logger.info(f"Degradation level changed: {self.degradation_level} -> {level}")
            self.degradation_level = level
        
        if level == 0:
            return {}
        
        return self._strategies.get(service, {}).get(level, {})