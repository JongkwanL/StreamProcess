"""OCR Worker implementation with PaddleOCR."""

import asyncio
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from io import BytesIO
from PIL import Image
import cv2
from paddleocr import PaddleOCR
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge
import base64

from ..config import settings
from ..queue.redis_queue import RedisQueue
from ..preprocessing.image_processor import ImageProcessor
from .batch_aggregator import BatchAggregator, EarliestDeadlineFirst


# Metrics
ocr_processed = Counter(
    "ocr_processed_total",
    "Total OCR jobs processed",
    ["status", "language"]
)
ocr_processing_time = Histogram(
    "ocr_processing_seconds",
    "OCR processing time",
    ["stage", "batch_size"]
)
ocr_characters = Counter(
    "ocr_characters_total",
    "Total characters recognized"
)
active_ocr_workers = Gauge(
    "active_ocr_workers",
    "Number of active OCR workers"
)


class OCRWorker:
    """Worker for processing OCR jobs."""
    
    def __init__(self, worker_id: str, queue: RedisQueue):
        self.worker_id = worker_id
        self.queue = queue
        self.ocr_engine: Optional[PaddleOCR] = None
        self.image_processor = ImageProcessor()
        self.batch_aggregator: Optional[BatchAggregator] = None
        
        # Performance tracking
        self._jobs_processed = 0
        self._total_characters = 0
        self._total_processing_ms = 0
        
        active_ocr_workers.inc()
    
    async def initialize(self):
        """Initialize the worker."""
        try:
            logger.info(f"Initializing OCR engine with languages: {settings.ocr_languages}")
            
            # Initialize PaddleOCR
            self.ocr_engine = PaddleOCR(
                use_angle_cls=settings.ocr_use_angle_cls,
                lang=",".join(settings.ocr_languages),
                use_gpu=settings.ocr_use_gpu,
                det_model_dir=settings.get_ocr_model_paths()["det"],
                rec_model_dir=settings.get_ocr_model_paths()["rec"],
                cls_model_dir=settings.get_ocr_model_paths()["cls"],
                det_db_thresh=settings.ocr_det_db_thresh,
                det_db_box_thresh=settings.ocr_det_db_box_thresh,
                show_log=False,
                use_mp=True,  # Multi-process
                total_process_num=2
            )
            
            logger.info(f"OCR engine initialized (GPU: {settings.ocr_use_gpu})")
            
            # Initialize batch aggregator
            self.batch_aggregator = BatchAggregator(
                max_batch_size=settings.ocr_batch_size,
                max_wait_ms=settings.ocr_batch_timeout_ms,
                scheduler=EarliestDeadlineFirst()
            )
            
            # Start batch processor
            asyncio.create_task(self._batch_processor())
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR worker: {e}")
            raise
    
    async def run(self):
        """Main worker loop."""
        logger.info(f"OCR Worker {self.worker_id} started")
        
        try:
            while True:
                # Get job from queue
                job = await self.queue.get_job(timeout=1000)
                
                if job:
                    await self._process_job(job)
                
                # Periodic cleanup
                if self._jobs_processed % 50 == 0:
                    await self._cleanup_cache()
        
        except KeyboardInterrupt:
            logger.info(f"OCR Worker {self.worker_id} shutting down")
        except Exception as e:
            logger.error(f"Worker error: {e}")
        finally:
            active_ocr_workers.dec()
    
    async def _process_job(self, job: Dict[str, Any]):
        """Process a single job."""
        job_type = job.get("type")
        
        try:
            if job_type == "ocr_single":
                await self._process_single_document(job)
            elif job_type == "ocr_batch":
                await self._add_to_batch(job)
            else:
                logger.warning(f"Unknown job type: {job_type}")
                await self.queue.nack_job(job, retry=False)
                return
            
            # Acknowledge job
            await self.queue.ack_job(job)
            self._jobs_processed += 1
            ocr_processed.labels(
                status="success",
                language=",".join(settings.ocr_languages)
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to process job: {e}")
            await self.queue.nack_job(job, retry=True)
            ocr_processed.labels(
                status="failure",
                language=",".join(settings.ocr_languages)
            ).inc()
    
    async def _process_single_document(self, job: Dict[str, Any]):
        """Process a single document."""
        job_id = job.get("job_id")
        config = job.get("config", {})
        
        start_time = time.time()
        
        try:
            # Get image data
            image = await self._load_image(job)
            
            if image is None:
                raise ValueError("Failed to load image")
            
            # Preprocess image
            processed_image = await self._preprocess_image(image, config)
            
            # Perform OCR
            result = await self._perform_ocr(processed_image, config)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            
            # Store result
            await self.queue.store_result(job_id, {
                "ocr_result": result,
                "metrics": {
                    "processing_time_ms": int(processing_time * 1000),
                    "total_characters": result.get("total_characters", 0),
                    "total_words": result.get("total_words", 0),
                    "confidence": result.get("confidence", 0.0)
                }
            })
            
            # Update metrics
            ocr_processing_time.labels(
                stage="complete",
                batch_size=1
            ).observe(processing_time)
            ocr_characters.inc(result.get("total_characters", 0))
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise
    
    async def _add_to_batch(self, job: Dict[str, Any]):
        """Add job to batch aggregator."""
        job_id = job.get("job_id")
        config = job.get("config", {})
        
        # Load image
        image = await self._load_image(job)
        
        if image is not None:
            # Add to batch
            await self.batch_aggregator.add_item({
                "job_id": job_id,
                "batch_id": job.get("batch_id"),
                "image": image,
                "config": config,
                "deadline": time.time() + 2.0  # 2s deadline
            })
    
    async def _batch_processor(self):
        """Process batched items."""
        while True:
            try:
                # Get batch from aggregator
                batch = await self.batch_aggregator.get_batch()
                
                if batch:
                    await self._process_batch(batch)
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of images."""
        if not batch:
            return
        
        start_time = time.time()
        batch_size = len(batch)
        
        try:
            # Group by similar image sizes for better batching
            buckets = self._bucket_by_size(batch)
            
            for bucket_items in buckets.values():
                # Process bucket
                results = await self._process_bucket(bucket_items)
                
                # Store results
                for item, result in zip(bucket_items, results):
                    await self.queue.store_result(
                        item["job_id"],
                        {
                            "ocr_result": result,
                            "metrics": {
                                "processing_time_ms": int((time.time() - start_time) * 1000),
                                "total_characters": result.get("total_characters", 0),
                                "total_words": result.get("total_words", 0)
                            }
                        }
                    )
            
            # Update metrics
            processing_time = time.time() - start_time
            ocr_processing_time.labels(
                stage="batch",
                batch_size=batch_size
            ).observe(processing_time)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    async def _process_bucket(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a bucket of similar-sized images."""
        results = []
        
        for item in items:
            image = item["image"]
            config = item.get("config", {})
            
            # Preprocess
            processed_image = await self._preprocess_image(image, config)
            
            # OCR
            result = await self._perform_ocr(processed_image, config)
            results.append(result)
        
        return results
    
    async def _load_image(self, job: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load image from job data."""
        try:
            if "image_content" in job:
                # Load from bytes
                image_bytes = job["image_content"]
                if isinstance(image_bytes, str):
                    # Base64 encoded
                    image_bytes = base64.b64decode(image_bytes)
                
                image = Image.open(BytesIO(image_bytes))
                return np.array(image)
            
            elif "document_url" in job:
                # Load from URL (S3/HTTP)
                # This would need implementation for S3/HTTP download
                logger.warning("URL loading not yet implemented")
                return None
            
            else:
                logger.error("No image data in job")
                return None
        
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    async def _preprocess_image(
        self,
        image: np.ndarray,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Preprocess image for OCR."""
        try:
            preprocessing = config.get("preprocessing", {})
            
            # Apply preprocessing steps
            if preprocessing.get("auto_rotate", False):
                image = self.image_processor.auto_rotate(image)
            
            if preprocessing.get("deskew", False):
                image = self.image_processor.deskew(image)
            
            if preprocessing.get("denoise", False):
                image = self.image_processor.denoise(image)
            
            if preprocessing.get("binarize", False):
                image = self.image_processor.binarize(image)
            
            # Resize if needed
            target_dpi = preprocessing.get("target_dpi")
            if target_dpi:
                image = self.image_processor.adjust_dpi(image, target_dpi)
            
            return image
        
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return image
    
    async def _perform_ocr(
        self,
        image: np.ndarray,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform OCR on image."""
        try:
            start_time = time.time()
            
            # Run PaddleOCR
            result = self.ocr_engine.ocr(image, cls=settings.ocr_use_angle_cls)
            
            # Process results
            full_text = []
            blocks = []
            total_confidence = 0
            total_chars = 0
            
            if result and len(result) > 0:
                for line in result[0]:  # First image in batch
                    if line:
                        box = line[0]
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        full_text.append(text)
                        total_chars += len(text)
                        total_confidence += confidence
                        
                        blocks.append({
                            "text": text,
                            "bounding_box": {
                                "x": int(min(p[0] for p in box)),
                                "y": int(min(p[1] for p in box)),
                                "width": int(max(p[0] for p in box) - min(p[0] for p in box)),
                                "height": int(max(p[1] for p in box) - min(p[1] for p in box))
                            },
                            "confidence": confidence
                        })
            
            # Calculate overall confidence
            avg_confidence = total_confidence / len(blocks) if blocks else 0
            
            # Format output based on config
            output_format = config.get("output_format", "json")
            
            processing_time = time.time() - start_time
            ocr_processing_time.labels(stage="ocr", batch_size=1).observe(processing_time)
            
            return {
                "full_text": " ".join(full_text),
                "blocks": blocks,
                "confidence": avg_confidence,
                "total_characters": total_chars,
                "total_words": len(" ".join(full_text).split()),
                "processing_time_ms": int(processing_time * 1000)
            }
        
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return {
                "full_text": "",
                "blocks": [],
                "confidence": 0.0,
                "total_characters": 0,
                "total_words": 0,
                "error": str(e)
            }
    
    def _bucket_by_size(
        self,
        items: List[Dict[str, Any]],
        size_buckets: List[Tuple[int, int]] = [(640, 480), (1280, 720), (1920, 1080)]
    ) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
        """Bucket items by image size for efficient batching."""
        buckets = {size: [] for size in size_buckets}
        buckets[(float('inf'), float('inf'))] = []  # Catch-all
        
        for item in items:
            image = item["image"]
            h, w = image.shape[:2]
            
            # Find appropriate bucket
            for bucket_w, bucket_h in size_buckets:
                if w <= bucket_w and h <= bucket_h:
                    buckets[(bucket_w, bucket_h)].append(item)
                    break
            else:
                buckets[(float('inf'), float('inf'))].append(item)
        
        # Remove empty buckets
        return {k: v for k, v in buckets.items() if v}
    
    async def _cleanup_cache(self):
        """Clean up any cached data."""
        # PaddleOCR doesn't have explicit cache cleanup
        # This is a placeholder for future cache management
        pass


async def main():
    """Main entry point for OCR worker."""
    # Initialize queue
    queue = RedisQueue(settings)
    await queue.initialize()
    
    # Create and initialize worker
    worker = OCRWorker(f"ocr_worker_{time.time()}", queue)
    await worker.initialize()
    
    # Run worker
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())