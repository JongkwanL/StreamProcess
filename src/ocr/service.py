"""OCR service implementation with PaddleOCR and Tesseract."""

import asyncio
import io
import time
from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import cv2
from PIL import Image
import subprocess
import tempfile

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

from loguru import logger
from prometheus_client import Counter, Histogram, Gauge

from ..config import settings


# Metrics
ocr_requests_total = Counter('ocr_requests_total', 'Total OCR requests', ['engine', 'language'])
ocr_processing_time = Histogram('ocr_processing_duration_seconds', 'OCR processing time')
ocr_active_sessions = Gauge('ocr_active_sessions', 'Active OCR sessions')
ocr_model_load_time = Histogram('ocr_model_load_duration_seconds', 'OCR model loading time')


@dataclass
class OCRConfig:
    """OCR configuration."""
    engine: str = "paddleocr"  # "paddleocr" or "tesseract"
    languages: List[str] = None
    use_gpu: bool = False
    use_angle_cls: bool = True
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.6
    det_db_unclip_ratio: float = 1.5
    rec_batch_num: int = 6
    max_text_length: int = 25
    drop_score: float = 0.5
    use_space_char: bool = True
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]


@dataclass
class OCRBoundingBox:
    """OCR bounding box."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    

@dataclass
class OCRBlock:
    """OCR text block."""
    text: str
    bounding_box: OCRBoundingBox
    confidence: float
    language: Optional[str] = None


@dataclass
class OCRResult:
    """OCR processing result."""
    text: str
    blocks: List[OCRBlock]
    language: str
    confidence: float
    processing_time: float
    engine: str
    image_dimensions: Tuple[int, int]
    metadata: Dict[str, Any]


class OCRService:
    """Optical Character Recognition service."""
    
    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()
        self.paddleocr_model: Optional[PaddleOCR] = None
        self.model_lock = asyncio.Lock()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize the OCR service."""
        logger.info("Initializing OCR service...")
        
        if self.config.engine == "paddleocr" and PADDLEOCR_AVAILABLE:
            with ocr_model_load_time.time():
                # Initialize PaddleOCR
                self.paddleocr_model = PaddleOCR(
                    use_angle_cls=self.config.use_angle_cls,
                    lang=self.config.languages[0] if self.config.languages else "en",
                    use_gpu=self.config.use_gpu,
                    det_db_thresh=self.config.det_db_thresh,
                    det_db_box_thresh=self.config.det_db_box_thresh,
                    det_db_unclip_ratio=self.config.det_db_unclip_ratio,
                    rec_batch_num=self.config.rec_batch_num,
                    max_text_length=self.config.max_text_length,
                    drop_score=self.config.drop_score,
                    use_space_char=self.config.use_space_char,
                    show_log=False
                )
        
        elif self.config.engine == "tesseract" and TESSERACT_AVAILABLE:
            # Check if Tesseract is installed
            try:
                version = pytesseract.get_tesseract_version()
                logger.info(f"Tesseract version: {version}")
            except Exception as e:
                logger.error(f"Tesseract not available: {e}")
                raise
        
        else:
            raise ValueError(f"OCR engine '{self.config.engine}' not available or supported")
        
        logger.info(f"OCR service initialized with engine: {self.config.engine}")
    
    async def process_image(
        self,
        image_data: Union[bytes, np.ndarray],
        language: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> OCRResult:
        """Process image for OCR."""
        start_time = time.time()
        
        with ocr_processing_time.time():
            # Update metrics
            ocr_requests_total.labels(
                engine=self.config.engine,
                language=language or "auto"
            ).inc()
            
            # Ensure model is loaded
            if self.config.engine == "paddleocr" and self.paddleocr_model is None:
                await self.initialize()
            
            # Prepare image
            image_array = await self._prepare_image(image_data)
            image_height, image_width = image_array.shape[:2]
            
            # Process with selected engine
            if self.config.engine == "paddleocr":
                result = await self._process_with_paddleocr(
                    image_array, language, config_override
                )
            elif self.config.engine == "tesseract":
                result = await self._process_with_tesseract(
                    image_array, language, config_override
                )
            else:
                raise ValueError(f"Unsupported engine: {self.config.engine}")
            
            # Add metadata
            result.processing_time = time.time() - start_time
            result.image_dimensions = (image_width, image_height)
            result.engine = self.config.engine
            
            return result
    
    async def process_batch(
        self,
        images: List[Union[bytes, np.ndarray]],
        language: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> List[OCRResult]:
        """Process multiple images in batch."""
        if self.config.engine == "paddleocr" and len(images) > 1:
            # PaddleOCR supports batch processing
            return await self._process_batch_paddleocr(images, language, config_override)
        else:
            # Process individually
            tasks = [
                self.process_image(image, language, config_override)
                for image in images
            ]
            return await asyncio.gather(*tasks)
    
    async def _process_with_paddleocr(
        self,
        image: np.ndarray,
        language: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> OCRResult:
        """Process image with PaddleOCR."""
        if not PADDLEOCR_AVAILABLE:
            raise RuntimeError("PaddleOCR not available")
        
        # Run OCR
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self.paddleocr_model.ocr, image
        )
        
        # Parse results
        blocks = []
        full_text = ""
        total_confidence = 0.0
        block_count = 0
        
        if results and results[0]:
            for line in results[0]:
                if len(line) >= 2:
                    bbox_points = line[0]
                    text_info = line[1]
                    
                    text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                    confidence = text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 1.0
                    
                    # Calculate bounding box
                    x_coords = [point[0] for point in bbox_points]
                    y_coords = [point[1] for point in bbox_points]
                    
                    bbox = OCRBoundingBox(
                        x=int(min(x_coords)),
                        y=int(min(y_coords)),
                        width=int(max(x_coords) - min(x_coords)),
                        height=int(max(y_coords) - min(y_coords)),
                        confidence=confidence
                    )
                    
                    block = OCRBlock(
                        text=text,
                        bounding_box=bbox,
                        confidence=confidence,
                        language=language or self.config.languages[0]
                    )
                    
                    blocks.append(block)
                    full_text += text + " "
                    total_confidence += confidence
                    block_count += 1
        
        # Calculate overall confidence
        avg_confidence = total_confidence / block_count if block_count > 0 else 0.0
        
        return OCRResult(
            text=full_text.strip(),
            blocks=blocks,
            language=language or self.config.languages[0],
            confidence=avg_confidence,
            processing_time=0.0,  # Will be set by caller
            engine="paddleocr",
            image_dimensions=(0, 0),  # Will be set by caller
            metadata={
                "block_count": block_count,
                "paddleocr_config": {
                    "use_angle_cls": self.config.use_angle_cls,
                    "det_db_thresh": self.config.det_db_thresh,
                    "rec_batch_num": self.config.rec_batch_num
                }
            }
        )
    
    async def _process_with_tesseract(
        self,
        image: np.ndarray,
        language: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> OCRResult:
        """Process image with Tesseract."""
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract not available")
        
        # Convert numpy array to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Configure Tesseract
        lang = language or "+".join(self.config.languages)
        config = "--oem 3 --psm 6"  # LSTM + uniform text block
        
        if config_override and "tesseract_config" in config_override:
            config = config_override["tesseract_config"]
        
        # Run OCR
        loop = asyncio.get_event_loop()
        
        # Get text
        text = await loop.run_in_executor(
            None, 
            lambda: pytesseract.image_to_string(pil_image, lang=lang, config=config)
        )
        
        # Get detailed data with bounding boxes
        data = await loop.run_in_executor(
            None,
            lambda: pytesseract.image_to_data(
                pil_image, lang=lang, config=config, output_type=pytesseract.Output.DICT
            )
        )
        
        # Parse detailed results
        blocks = []
        total_confidence = 0.0
        block_count = 0
        
        for i in range(len(data['text'])):
            text_content = data['text'][i].strip()
            if text_content:
                confidence = int(data['conf'][i]) / 100.0  # Convert to 0-1 range
                
                bbox = OCRBoundingBox(
                    x=data['left'][i],
                    y=data['top'][i],
                    width=data['width'][i],
                    height=data['height'][i],
                    confidence=confidence
                )
                
                block = OCRBlock(
                    text=text_content,
                    bounding_box=bbox,
                    confidence=confidence,
                    language=language or self.config.languages[0]
                )
                
                blocks.append(block)
                total_confidence += confidence
                block_count += 1
        
        # Calculate overall confidence
        avg_confidence = total_confidence / block_count if block_count > 0 else 0.0
        
        return OCRResult(
            text=text.strip(),
            blocks=blocks,
            language=language or self.config.languages[0],
            confidence=avg_confidence,
            processing_time=0.0,  # Will be set by caller
            engine="tesseract",
            image_dimensions=(0, 0),  # Will be set by caller
            metadata={
                "block_count": block_count,
                "tesseract_config": config,
                "language": lang
            }
        )
    
    async def _process_batch_paddleocr(
        self,
        images: List[Union[bytes, np.ndarray]],
        language: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ) -> List[OCRResult]:
        """Process batch of images with PaddleOCR."""
        # Prepare all images
        image_arrays = []
        for image_data in images:
            image_array = await self._prepare_image(image_data)
            image_arrays.append(image_array)
        
        # Process batch
        loop = asyncio.get_event_loop()
        batch_results = await loop.run_in_executor(
            None, self.paddleocr_model.ocr, image_arrays
        )
        
        # Parse results for each image
        results = []
        for i, image_results in enumerate(batch_results):
            # Create result for this image (similar to _process_with_paddleocr)
            blocks = []
            full_text = ""
            total_confidence = 0.0
            block_count = 0
            
            if image_results and image_results[0]:
                for line in image_results[0]:
                    if len(line) >= 2:
                        bbox_points = line[0]
                        text_info = line[1]
                        
                        text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                        confidence = text_info[1] if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 1.0
                        
                        # Calculate bounding box
                        x_coords = [point[0] for point in bbox_points]
                        y_coords = [point[1] for point in bbox_points]
                        
                        bbox = OCRBoundingBox(
                            x=int(min(x_coords)),
                            y=int(min(y_coords)),
                            width=int(max(x_coords) - min(x_coords)),
                            height=int(max(y_coords) - min(y_coords)),
                            confidence=confidence
                        )
                        
                        block = OCRBlock(
                            text=text,
                            bounding_box=bbox,
                            confidence=confidence,
                            language=language or self.config.languages[0]
                        )
                        
                        blocks.append(block)
                        full_text += text + " "
                        total_confidence += confidence
                        block_count += 1
            
            # Calculate overall confidence
            avg_confidence = total_confidence / block_count if block_count > 0 else 0.0
            
            image_height, image_width = image_arrays[i].shape[:2]
            
            result = OCRResult(
                text=full_text.strip(),
                blocks=blocks,
                language=language or self.config.languages[0],
                confidence=avg_confidence,
                processing_time=0.0,  # Batch processing time will be distributed
                engine="paddleocr",
                image_dimensions=(image_width, image_height),
                metadata={
                    "block_count": block_count,
                    "batch_index": i,
                    "batch_size": len(images)
                }
            )
            
            results.append(result)
        
        return results
    
    async def _prepare_image(self, image_data: Union[bytes, np.ndarray]) -> np.ndarray:
        """Prepare image data for OCR."""
        if isinstance(image_data, np.ndarray):
            return image_data
        
        try:
            # Convert bytes to image
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
            
            return image
            
        except Exception as e:
            logger.error(f"Error preparing image: {e}")
            raise
    
    async def get_engine_info(self) -> Dict[str, Any]:
        """Get OCR engine information."""
        info = {
            "engine": self.config.engine,
            "languages": self.config.languages,
            "active_sessions": len(self.active_sessions)
        }
        
        if self.config.engine == "paddleocr":
            info.update({
                "paddleocr_available": PADDLEOCR_AVAILABLE,
                "use_gpu": self.config.use_gpu,
                "model_loaded": self.paddleocr_model is not None
            })
        elif self.config.engine == "tesseract":
            info.update({
                "tesseract_available": TESSERACT_AVAILABLE,
            })
            
            if TESSERACT_AVAILABLE:
                try:
                    info["tesseract_version"] = str(pytesseract.get_tesseract_version())
                    info["available_languages"] = pytesseract.get_languages()
                except:
                    pass
        
        return info
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        engine_ready = False
        
        if self.config.engine == "paddleocr":
            engine_ready = PADDLEOCR_AVAILABLE and self.paddleocr_model is not None
        elif self.config.engine == "tesseract":
            engine_ready = TESSERACT_AVAILABLE
        
        return {
            "status": "healthy" if engine_ready else "initializing",
            "engine": self.config.engine,
            "engine_ready": engine_ready,
            "active_sessions": len(self.active_sessions)
        }