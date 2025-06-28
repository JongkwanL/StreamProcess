"""Autoscaling controller with PID control and predictive scaling."""

import asyncio
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from prometheus_client import Gauge, Counter
import json

from ..config import settings
from ..queue.redis_queue import RedisQueue


# Metrics
autoscaler_decisions = Counter(
    "autoscaler_decisions_total",
    "Total autoscaling decisions",
    ["action", "worker_type"]
)
current_replicas = Gauge(
    "current_replicas",
    "Current number of replicas",
    ["worker_type"]
)
target_replicas = Gauge(
    "target_replicas", 
    "Target number of replicas",
    ["worker_type"]
)
queue_utilization = Gauge(
    "queue_utilization",
    "Queue utilization ratio",
    ["worker_type"]
)


@dataclass
class WorkerMetrics:
    """Worker performance metrics."""
    worker_type: str
    current_replicas: int
    target_replicas: int
    queue_depth: int
    processing_rate: float  # jobs/second
    arrival_rate: float    # jobs/second
    avg_processing_time: float  # seconds
    utilization: float     # 0.0-1.0
    lag_ms: float
    error_rate: float
    timestamp: float


@dataclass
class ScalingDecision:
    """Scaling decision result."""
    worker_type: str
    current_replicas: int
    target_replicas: int
    action: str  # "scale_up", "scale_down", "no_change"
    reason: str
    confidence: float
    timestamp: float


class PIDController:
    """PID controller for autoscaling."""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
    
    def update(self, error: float, current_time: float) -> float:
        """Update PID controller with new error."""
        dt = current_time - self.last_time
        
        if dt <= 0:
            return 0.0
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.last_error) / dt
        
        # PID output
        output = proportional + integral + derivative
        
        # Update for next iteration
        self.last_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller state."""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


class AutoscalerController:
    """Main autoscaling controller."""
    
    def __init__(self, queue: RedisQueue):
        self.queue = queue
        self.enabled = settings.autoscale_enabled
        
        # PID controllers for each worker type
        self.pid_controllers = {
            "stt": PIDController(kp=2.0, ki=0.1, kd=0.05),
            "ocr": PIDController(kp=1.5, ki=0.08, kd=0.03)
        }
        
        # Historical data for prediction
        self.metrics_history: Dict[str, List[WorkerMetrics]] = {
            "stt": [],
            "ocr": []
        }
        
        # Service rates (estimated jobs/second per worker)
        self.service_rates = {
            "stt": 2.0,  # ~2 STT jobs per second per worker
            "ocr": 1.0   # ~1 OCR job per second per worker
        }
        
        # Current state
        self.current_state = {
            "stt": {"replicas": settings.autoscale_min_workers, "last_scale": 0},
            "ocr": {"replicas": settings.autoscale_min_workers, "last_scale": 0}
        }
        
        # Scaling parameters
        self.min_workers = settings.autoscale_min_workers
        self.max_workers = settings.autoscale_max_workers
        self.target_utilization = settings.autoscale_target_utilization
        self.scale_up_threshold = settings.autoscale_scale_up_threshold
        self.scale_down_threshold = settings.autoscale_scale_down_threshold
        self.warmup_time = settings.autoscale_warmup_time_seconds
        self.cooldown_time = settings.autoscale_cooldown_seconds
    
    async def run(self):
        """Main controller loop."""
        logger.info("Autoscaler controller started")
        
        while True:
            try:
                if self.enabled:
                    # Collect metrics
                    stt_metrics = await self._collect_metrics("stt")
                    ocr_metrics = await self._collect_metrics("ocr")
                    
                    # Make scaling decisions
                    stt_decision = await self._make_scaling_decision(stt_metrics)
                    ocr_decision = await self._make_scaling_decision(ocr_metrics)
                    
                    # Execute scaling decisions
                    if stt_decision.action != "no_change":
                        await self._execute_scaling(stt_decision)
                    
                    if ocr_decision.action != "no_change":
                        await self._execute_scaling(ocr_decision)
                    
                    # Update metrics
                    self._update_prometheus_metrics(stt_metrics, stt_decision)
                    self._update_prometheus_metrics(ocr_metrics, ocr_decision)
                
                # Sleep between checks
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Autoscaler error: {e}")
                await asyncio.sleep(60)  # Longer sleep on error
    
    async def _collect_metrics(self, worker_type: str) -> WorkerMetrics:
        """Collect metrics for a worker type."""
        try:
            # Get queue depths for each priority
            queue_depths = await self.queue.get_lag()
            total_depth = sum(queue_depths.values())
            
            # Estimate current metrics (in production, these would come from monitoring)
            current_time = time.time()
            
            # Get recent metrics from history
            recent_metrics = self._get_recent_metrics(worker_type, window_seconds=300)
            
            # Calculate rates
            arrival_rate = self._estimate_arrival_rate(worker_type, recent_metrics)
            processing_rate = self._estimate_processing_rate(worker_type, recent_metrics)
            
            # Current replicas (would come from container orchestrator)
            current_replicas = self.current_state[worker_type]["replicas"]
            
            # Calculate utilization
            capacity = current_replicas * self.service_rates[worker_type]
            utilization = arrival_rate / capacity if capacity > 0 else 1.0
            
            # Calculate average processing time
            avg_processing_time = 1.0 / self.service_rates[worker_type]
            
            # Queue lag (simplified)
            lag_ms = (total_depth / processing_rate * 1000) if processing_rate > 0 else 0
            
            metrics = WorkerMetrics(
                worker_type=worker_type,
                current_replicas=current_replicas,
                target_replicas=current_replicas,
                queue_depth=total_depth,
                processing_rate=processing_rate,
                arrival_rate=arrival_rate,
                avg_processing_time=avg_processing_time,
                utilization=utilization,
                lag_ms=lag_ms,
                error_rate=0.01,  # Placeholder
                timestamp=current_time
            )
            
            # Store in history
            self.metrics_history[worker_type].append(metrics)
            if len(self.metrics_history[worker_type]) > 100:  # Keep last 100 samples
                self.metrics_history[worker_type] = self.metrics_history[worker_type][-100:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for {worker_type}: {e}")
            # Return default metrics
            return WorkerMetrics(
                worker_type=worker_type,
                current_replicas=self.current_state[worker_type]["replicas"],
                target_replicas=self.current_state[worker_type]["replicas"],
                queue_depth=0,
                processing_rate=0.0,
                arrival_rate=0.0,
                avg_processing_time=1.0,
                utilization=0.0,
                lag_ms=0.0,
                error_rate=0.0,
                timestamp=time.time()
            )
    
    async def _make_scaling_decision(self, metrics: WorkerMetrics) -> ScalingDecision:
        """Make scaling decision based on metrics."""
        worker_type = metrics.worker_type
        current_time = time.time()
        
        # Check cooldown period
        last_scale_time = self.current_state[worker_type]["last_scale"]
        if current_time - last_scale_time < self.cooldown_time:
            return ScalingDecision(
                worker_type=worker_type,
                current_replicas=metrics.current_replicas,
                target_replicas=metrics.current_replicas,
                action="no_change",
                reason="cooldown_period",
                confidence=1.0,
                timestamp=current_time
            )
        
        # Capacity-based calculation
        target_from_capacity = self._calculate_capacity_based_target(metrics)
        
        # PID-based calculation
        target_from_pid = self._calculate_pid_based_target(metrics)
        
        # Predictive calculation
        target_from_prediction = self._calculate_predictive_target(metrics)
        
        # Combine targets (weighted average)
        target_replicas = int(
            0.4 * target_from_capacity +
            0.3 * target_from_pid +
            0.3 * target_from_prediction
        )
        
        # Apply constraints
        target_replicas = max(self.min_workers, min(self.max_workers, target_replicas))
        
        # Determine action
        if target_replicas > metrics.current_replicas:
            action = "scale_up"
            reason = f"target={target_replicas}, utilization={metrics.utilization:.2f}"
        elif target_replicas < metrics.current_replicas:
            action = "scale_down"
            reason = f"target={target_replicas}, utilization={metrics.utilization:.2f}"
        else:
            action = "no_change"
            reason = "at_target"
        
        # Calculate confidence
        confidence = self._calculate_confidence(metrics, target_replicas)
        
        return ScalingDecision(
            worker_type=worker_type,
            current_replicas=metrics.current_replicas,
            target_replicas=target_replicas,
            action=action,
            reason=reason,
            confidence=confidence,
            timestamp=current_time
        )
    
    def _calculate_capacity_based_target(self, metrics: WorkerMetrics) -> float:
        """Calculate target replicas based on capacity model."""
        if metrics.arrival_rate <= 0:
            return metrics.current_replicas
        
        # Service rate per worker
        service_rate = self.service_rates[metrics.worker_type]
        
        # Required capacity
        required_capacity = metrics.arrival_rate / self.target_utilization
        
        # Target replicas
        target = math.ceil(required_capacity / service_rate)
        
        return float(target)
    
    def _calculate_pid_based_target(self, metrics: WorkerMetrics) -> float:
        """Calculate target replicas using PID controller."""
        # Error is the difference between target and actual utilization
        target_util = self.target_utilization
        error = target_util - metrics.utilization
        
        # PID output
        pid_controller = self.pid_controllers[metrics.worker_type]
        pid_output = pid_controller.update(error, metrics.timestamp)
        
        # Convert PID output to replica adjustment
        adjustment = pid_output * 2  # Scale factor
        target = metrics.current_replicas + adjustment
        
        return max(1.0, target)
    
    def _calculate_predictive_target(self, metrics: WorkerMetrics) -> float:
        """Calculate target replicas with predictive scaling."""
        # Get trend in arrival rate
        recent_metrics = self._get_recent_metrics(metrics.worker_type, window_seconds=600)
        
        if len(recent_metrics) < 2:
            return metrics.current_replicas
        
        # Calculate trend
        times = [m.timestamp for m in recent_metrics]
        rates = [m.arrival_rate for m in recent_metrics]
        
        if len(times) >= 2:
            # Simple linear regression for trend
            n = len(times)
            sum_t = sum(times)
            sum_r = sum(rates)
            sum_tr = sum(t * r for t, r in zip(times, rates))
            sum_tt = sum(t * t for t in times)
            
            if n * sum_tt - sum_t * sum_t != 0:
                slope = (n * sum_tr - sum_t * sum_r) / (n * sum_tt - sum_t * sum_t)
                
                # Predict arrival rate after warmup time
                future_time = metrics.timestamp + self.warmup_time
                predicted_rate = metrics.arrival_rate + slope * self.warmup_time
                
                # Calculate target for predicted rate
                if predicted_rate > 0:
                    service_rate = self.service_rates[metrics.worker_type]
                    required_capacity = predicted_rate / self.target_utilization
                    target = math.ceil(required_capacity / service_rate)
                    return float(target)
        
        return metrics.current_replicas
    
    def _calculate_confidence(self, metrics: WorkerMetrics, target_replicas: int) -> float:
        """Calculate confidence in scaling decision."""
        confidence = 1.0
        
        # Reduce confidence if utilization is near target
        util_diff = abs(metrics.utilization - self.target_utilization)
        if util_diff < 0.1:
            confidence *= 0.5
        
        # Reduce confidence if queue is empty
        if metrics.queue_depth == 0 and target_replicas > metrics.current_replicas:
            confidence *= 0.3
        
        # Reduce confidence if error rate is high
        if metrics.error_rate > 0.05:
            confidence *= 0.7
        
        return max(0.0, min(1.0, confidence))
    
    def _get_recent_metrics(self, worker_type: str, window_seconds: int) -> List[WorkerMetrics]:
        """Get recent metrics within time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        return [
            m for m in self.metrics_history[worker_type]
            if m.timestamp >= cutoff_time
        ]
    
    def _estimate_arrival_rate(self, worker_type: str, recent_metrics: List[WorkerMetrics]) -> float:
        """Estimate current arrival rate."""
        if not recent_metrics:
            return 0.0
        
        # Use latest metrics or calculate from queue growth
        if len(recent_metrics) >= 2:
            latest = recent_metrics[-1]
            previous = recent_metrics[-2]
            time_diff = latest.timestamp - previous.timestamp
            queue_growth = latest.queue_depth - previous.queue_depth
            
            if time_diff > 0:
                return max(0.0, queue_growth / time_diff)
        
        # Fallback to estimated rate based on utilization
        latest = recent_metrics[-1]
        return latest.utilization * latest.current_replicas * self.service_rates[worker_type]
    
    def _estimate_processing_rate(self, worker_type: str, recent_metrics: List[WorkerMetrics]) -> float:
        """Estimate current processing rate."""
        if not recent_metrics:
            return 0.0
        
        latest = recent_metrics[-1]
        return latest.current_replicas * self.service_rates[worker_type] * latest.utilization
    
    async def _execute_scaling(self, decision: ScalingDecision):
        """Execute scaling decision."""
        try:
            logger.info(
                f"Scaling {decision.worker_type}: "
                f"{decision.current_replicas} -> {decision.target_replicas} "
                f"({decision.action}) - {decision.reason}"
            )
            
            # In a real implementation, this would call the container orchestrator
            # For now, we just update our state
            self.current_state[decision.worker_type]["replicas"] = decision.target_replicas
            self.current_state[decision.worker_type]["last_scale"] = decision.timestamp
            
            # Record metric
            autoscaler_decisions.labels(
                action=decision.action,
                worker_type=decision.worker_type
            ).inc()
            
            # TODO: Implement actual scaling via:
            # - Kubernetes API (kubectl scale)
            # - Podman API
            # - Docker Swarm API
            # - Custom orchestrator
            
        except Exception as e:
            logger.error(f"Failed to execute scaling: {e}")
    
    def _update_prometheus_metrics(self, metrics: WorkerMetrics, decision: ScalingDecision):
        """Update Prometheus metrics."""
        current_replicas.labels(worker_type=metrics.worker_type).set(metrics.current_replicas)
        target_replicas.labels(worker_type=metrics.worker_type).set(decision.target_replicas)
        queue_utilization.labels(worker_type=metrics.worker_type).set(metrics.utilization)


async def main():
    """Main entry point for autoscaler."""
    # Initialize queue
    queue = RedisQueue(settings)
    await queue.initialize()
    
    # Create and run controller
    controller = AutoscalerController(queue)
    await controller.run()


if __name__ == "__main__":
    asyncio.run(main())