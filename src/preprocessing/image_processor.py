"""Image preprocessing utilities for OCR."""

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional, List
import scipy.ndimage
from loguru import logger
import pytesseract


class ImageProcessor:
    """Image preprocessing and enhancement utilities for OCR."""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    def preprocess(
        self,
        image: np.ndarray,
        auto_rotate: bool = True,
        deskew: bool = True,
        denoise: bool = True,
        enhance_contrast: bool = True,
        binarize: bool = False
    ) -> np.ndarray:
        """
        Comprehensive image preprocessing for OCR.
        
        Args:
            image: Input image as numpy array
            auto_rotate: Automatically detect and correct rotation
            deskew: Correct skew/tilt
            denoise: Apply denoising
            enhance_contrast: Enhance contrast and brightness
            binarize: Convert to binary image
        
        Returns:
            Preprocessed image
        """
        try:
            # Convert to PIL for some operations
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                # Convert float to uint8
                image_uint8 = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_uint8)
            
            # Auto-rotate if needed
            if auto_rotate:
                pil_image = self.auto_rotate(pil_image)
                image = np.array(pil_image)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Deskew
            if deskew:
                image = self.deskew(image)
            
            # Denoise
            if denoise:
                image = self.denoise(image)
            
            # Enhance contrast
            if enhance_contrast:
                image = self.enhance_contrast(image)
            
            # Binarize
            if binarize:
                image = self.binarize(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return image
    
    def auto_rotate(self, image: Image.Image) -> Image.Image:
        """Automatically detect and correct image rotation using EXIF data and content analysis."""
        try:
            # First, try EXIF orientation
            try:
                exif = image._getexif()
                if exif:
                    orientation = exif.get(274)  # Orientation tag
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(-90, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
            except:
                pass
            
            # Convert to numpy for content-based rotation detection
            img_array = np.array(image.convert('L'))
            
            # Detect text orientation using projection profiles
            best_angle = self._detect_text_angle(img_array)
            
            if abs(best_angle) > 1:  # Only rotate if significant angle detected
                image = image.rotate(best_angle, expand=True, fillcolor='white')
            
            return image
            
        except Exception as e:
            logger.warning(f"Auto-rotation failed: {e}")
            return image
    
    def _detect_text_angle(self, image: np.ndarray) -> float:
        """Detect text skew angle using Hough line transform."""
        try:
            # Edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Hough line transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return 0.0
            
            # Calculate angles
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi - 90
                # Consider only nearly horizontal lines
                if -45 <= angle <= 45:
                    angles.append(angle)
            
            if not angles:
                return 0.0
            
            # Return median angle
            return np.median(angles)
            
        except Exception as e:
            logger.warning(f"Angle detection failed: {e}")
            return 0.0
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew/tilt."""
        try:
            # Find contours
            contours, _ = cv2.findContours(
                cv2.bitwise_not(image), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return image
            
            # Find the largest contour (likely the text block)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            
            # Adjust angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Only correct if angle is significant
            if abs(angle) > 0.5:
                h, w = image.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return image
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to image."""
        try:
            # Non-local means denoising
            denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Morphological opening to remove small noise
            kernel = np.ones((2, 2), np.uint8)
            denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPENING, kernel)
            
            return denoised
            
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast and brightness."""
        try:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
            # Gamma correction
            gamma = 1.2
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image
    
    def binarize(self, image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
        """Convert image to binary."""
        try:
            if method == 'adaptive':
                # Adaptive thresholding
                binary = cv2.adaptiveThreshold(
                    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            elif method == 'otsu':
                # Otsu's thresholding
                _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                # Simple thresholding
                _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            
            return binary
            
        except Exception as e:
            logger.warning(f"Binarization failed: {e}")
            return image
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using morphological operations."""
        try:
            # Remove small noise
            kernel = np.ones((1, 1), np.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
            # Remove horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            # Detect lines
            horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Remove lines from image
            image = cv2.subtract(image, horizontal_lines)
            image = cv2.subtract(image, vertical_lines)
            
            return image
            
        except Exception as e:
            logger.warning(f"Noise removal failed: {e}")
            return image
    
    def adjust_dpi(self, image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        """Resize image to target DPI."""
        try:
            # Calculate current DPI (estimate)
            h, w = image.shape[:2]
            
            # Assume document width is ~8.5 inches for estimation
            estimated_dpi = w / 8.5
            
            if estimated_dpi < target_dpi * 0.8:  # Only upscale if significantly lower
                scale_factor = target_dpi / estimated_dpi
                new_width = int(w * scale_factor)
                new_height = int(h * scale_factor)
                
                # Use cubic interpolation for upscaling
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                return resized
            
            return image
            
        except Exception as e:
            logger.warning(f"DPI adjustment failed: {e}")
            return image
    
    def detect_layout(self, image: np.ndarray) -> List[Dict]:
        """Detect document layout elements."""
        try:
            # Find contours
            contours, _ = cv2.findContours(
                cv2.bitwise_not(image), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            elements = []
            
            for contour in contours:
                # Filter small contours
                area = cv2.contourArea(contour)
                if area < 100:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Classify element type based on aspect ratio
                aspect_ratio = w / h
                
                if aspect_ratio > 5:
                    element_type = "line"
                elif aspect_ratio > 2:
                    element_type = "paragraph"
                elif 0.8 <= aspect_ratio <= 1.2:
                    element_type = "square"
                else:
                    element_type = "text_block"
                
                elements.append({
                    "type": element_type,
                    "bbox": (x, y, w, h),
                    "area": area,
                    "aspect_ratio": aspect_ratio
                })
            
            # Sort by position (top to bottom, left to right)
            elements.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
            
            return elements
            
        except Exception as e:
            logger.error(f"Layout detection failed: {e}")
            return []
    
    def detect_tables(self, image: np.ndarray) -> List[Dict]:
        """Detect table structures in image."""
        try:
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Find horizontal and vertical lines
            horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combine lines
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find table contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small areas
                    x, y, w, h = cv2.boundingRect(contour)
                    tables.append({
                        "bbox": (x, y, w, h),
                        "area": area,
                        "type": "table"
                    })
            
            return tables
            
        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            return []
    
    def crop_to_content(self, image: np.ndarray, padding: int = 10) -> np.ndarray:
        """Crop image to content boundaries."""
        try:
            # Find non-white pixels
            mask = image < 240  # Adjust threshold as needed
            
            # Find bounding box of content
            coords = np.column_stack(np.where(mask))
            if len(coords) == 0:
                return image
            
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            
            # Add padding
            h, w = image.shape
            y0 = max(0, y0 - padding)
            x0 = max(0, x0 - padding)
            y1 = min(h, y1 + padding)
            x1 = min(w, x1 + padding)
            
            # Crop
            cropped = image[y0:y1, x0:x1]
            return cropped
            
        except Exception as e:
            logger.warning(f"Cropping failed: {e}")
            return image
    
    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Apply OCR-specific enhancements."""
        try:
            # Crop to content
            image = self.crop_to_content(image)
            
            # Enhance contrast
            image = self.enhance_contrast(image)
            
            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend original and sharpened
            enhanced = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"OCR enhancement failed: {e}")
            return image
    
    def validate_image(self, image: np.ndarray) -> Tuple[bool, str]:
        """Validate image for OCR processing."""
        try:
            h, w = image.shape[:2]
            
            # Check minimum size
            if w < 100 or h < 100:
                return False, "Image too small"
            
            # Check maximum size
            if w > 10000 or h > 10000:
                return False, "Image too large"
            
            # Check if image is mostly blank
            non_white_pixels = np.sum(image < 240)
            total_pixels = w * h
            content_ratio = non_white_pixels / total_pixels
            
            if content_ratio < 0.01:
                return False, "Image appears to be blank"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {e}"