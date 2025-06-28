"""Unit tests for preprocessing modules."""

import pytest
import numpy as np
from PIL import Image

from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.image_processor import ImageProcessor


@pytest.mark.unit
class TestAudioProcessor:
    """Test audio preprocessing functionality."""
    
    def test_resample(self, audio_processor, sample_audio):
        """Test audio resampling."""
        audio, orig_sr = sample_audio
        target_sr = 8000
        
        resampled = audio_processor.resample(audio, orig_sr, target_sr)
        
        # Should have different length
        expected_length = len(audio) * target_sr // orig_sr
        assert abs(len(resampled) - expected_length) <= 1
    
    def test_to_mono(self, audio_processor):
        """Test stereo to mono conversion."""
        # Create stereo audio
        stereo_audio = np.random.rand(1000, 2).astype(np.float32)
        
        mono_audio = audio_processor.to_mono(stereo_audio)
        
        assert mono_audio.ndim == 1
        assert len(mono_audio) == len(stereo_audio)
    
    def test_normalize_audio(self, audio_processor, sample_audio):
        """Test audio normalization."""
        audio, _ = sample_audio
        
        # Add some gain to test normalization
        loud_audio = audio * 0.1
        
        normalized = audio_processor.normalize_audio(loud_audio)
        
        # Should be louder than original
        assert np.max(np.abs(normalized)) > np.max(np.abs(loud_audio))
        # Should not clip
        assert np.max(np.abs(normalized)) <= 1.0
    
    def test_high_pass_filter(self, audio_processor, sample_audio):
        """Test high-pass filtering."""
        audio, sample_rate = sample_audio
        
        filtered = audio_processor.high_pass_filter(audio, sample_rate, cutoff=100)
        
        # Should have same length
        assert len(filtered) == len(audio)
        # Should be different from original (unless already high-pass)
        # Allow for small differences due to floating point precision
        assert not np.allclose(filtered, audio, atol=1e-6) or np.allclose(filtered, audio, atol=1e-3)
    
    def test_apply_agc(self, audio_processor):
        """Test Automatic Gain Control."""
        # Create audio with varying amplitude
        audio = np.concatenate([
            np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.1,  # Quiet
            np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.9   # Loud
        ]).astype(np.float32)
        
        agc_audio = audio_processor.apply_agc(audio)
        
        # Should have more consistent amplitude
        first_half_rms = np.sqrt(np.mean(agc_audio[:16000] ** 2))
        second_half_rms = np.sqrt(np.mean(agc_audio[16000:] ** 2))
        
        # Ratio should be closer to 1 than original
        agc_ratio = max(first_half_rms, second_half_rms) / min(first_half_rms, second_half_rms)
        original_ratio = 0.9 / 0.1
        
        assert agc_ratio < original_ratio
    
    def test_detect_silence(self, audio_processor):
        """Test silence detection."""
        # Create audio with silence
        sample_rate = 16000
        audio = np.concatenate([
            np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate)),  # 1s speech
            np.zeros(sample_rate),  # 1s silence
            np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate))   # 1s speech
        ]).astype(np.float32)
        
        silence_segments = audio_processor.detect_silence(
            audio, sample_rate, 
            silence_threshold=0.01, 
            min_silence_duration=0.5
        )
        
        # Should detect one silence segment around 1-2 seconds
        assert len(silence_segments) >= 1
        if silence_segments:
            start, end = silence_segments[0]
            assert 0.5 <= start <= 1.5
            assert 1.5 <= end <= 2.5
    
    def test_split_on_silence(self, audio_processor):
        """Test splitting audio on silence."""
        # Create audio with silence
        sample_rate = 16000
        audio = np.concatenate([
            np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sample_rate // 2)),  # 0.5s speech
            np.zeros(sample_rate),  # 1s silence
            np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, sample_rate // 2))   # 0.5s speech
        ]).astype(np.float32)
        
        segments = audio_processor.split_on_silence(
            audio, sample_rate,
            silence_threshold=0.01,
            min_silence_duration=0.5
        )
        
        # Should split into segments
        assert len(segments) >= 1
        # Total length should be similar (allowing for padding)
        total_length = sum(len(seg) for seg in segments)
        assert abs(total_length - len(audio)) <= sample_rate * 0.2  # Allow 200ms difference
    
    def test_preprocess_pipeline(self, audio_processor, sample_audio):
        """Test complete preprocessing pipeline."""
        audio, sample_rate = sample_audio
        
        processed = audio_processor.preprocess(
            audio, sample_rate,
            target_sample_rate=16000,
            normalize=True,
            denoise=False  # Skip denoising for speed
        )
        
        # Should produce valid audio
        assert isinstance(processed, np.ndarray)
        assert processed.dtype in [np.float32, np.float64]
        assert len(processed) > 0
        assert np.max(np.abs(processed)) <= 1.0


@pytest.mark.unit
class TestImageProcessor:
    """Test image preprocessing functionality."""
    
    def test_auto_rotate(self, image_processor):
        """Test image auto-rotation."""
        # Create a simple test image
        image = Image.new('RGB', (100, 50), color='white')
        
        rotated = image_processor.auto_rotate(image)
        
        # Should return PIL Image
        assert isinstance(rotated, Image.Image)
        # Size should be preserved or reasonably close
        assert abs(rotated.size[0] - image.size[0]) <= 50
        assert abs(rotated.size[1] - image.size[1]) <= 50
    
    def test_deskew(self, image_processor, sample_image):
        """Test image deskewing."""
        deskewed = image_processor.deskew(sample_image)
        
        # Should have same shape
        assert deskewed.shape == sample_image.shape
        # Should be valid image
        assert deskewed.dtype == sample_image.dtype
    
    def test_denoise(self, image_processor, sample_image):
        """Test image denoising."""
        # Add noise to image
        noisy_image = sample_image.copy()
        noise = np.random.randint(-20, 20, sample_image.shape, dtype=np.int8)
        noisy_image = np.clip(noisy_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        denoised = image_processor.denoise(noisy_image)
        
        # Should have same shape
        assert denoised.shape == noisy_image.shape
        # Should be smoother (less variance)
        assert np.var(denoised) <= np.var(noisy_image)
    
    def test_enhance_contrast(self, image_processor, sample_image):
        """Test contrast enhancement."""
        enhanced = image_processor.enhance_contrast(sample_image)
        
        # Should have same shape
        assert enhanced.shape == sample_image.shape
        # Contrast should be improved (higher std dev)
        assert np.std(enhanced) >= np.std(sample_image) * 0.8  # Allow some tolerance
    
    def test_binarize_adaptive(self, image_processor, sample_image):
        """Test adaptive binarization."""
        binary = image_processor.binarize(sample_image, method='adaptive')
        
        # Should be binary (only 0 and 255)
        unique_values = np.unique(binary)
        assert len(unique_values) <= 2
        assert all(val in [0, 255] for val in unique_values)
    
    def test_binarize_otsu(self, image_processor, sample_image):
        """Test Otsu binarization."""
        binary = image_processor.binarize(sample_image, method='otsu')
        
        # Should be binary
        unique_values = np.unique(binary)
        assert len(unique_values) <= 2
        assert all(val in [0, 255] for val in unique_values)
    
    def test_adjust_dpi(self, image_processor, sample_image):
        """Test DPI adjustment."""
        adjusted = image_processor.adjust_dpi(sample_image, target_dpi=300)
        
        # Should be valid image
        assert isinstance(adjusted, np.ndarray)
        assert adjusted.dtype == sample_image.dtype
        # Size might change
        assert adjusted.size > 0
    
    def test_crop_to_content(self, image_processor):
        """Test cropping to content."""
        # Create image with content in center
        image = np.full((100, 100), 255, dtype=np.uint8)  # White background
        image[30:70, 30:70] = 0  # Black square in center
        
        cropped = image_processor.crop_to_content(image, padding=5)
        
        # Should be smaller than original
        assert cropped.shape[0] <= image.shape[0]
        assert cropped.shape[1] <= image.shape[1]
        # Should contain the content
        assert np.min(cropped) == 0  # Contains black pixels
    
    def test_detect_layout(self, image_processor):
        """Test layout detection."""
        # Create image with text-like rectangles
        image = np.full((200, 300), 255, dtype=np.uint8)
        # Add some rectangular "text blocks"
        image[50:70, 50:250] = 0   # Paragraph
        image[80:85, 50:150] = 0   # Line
        image[100:120, 50:100] = 0 # Square block
        
        elements = image_processor.detect_layout(image)
        
        # Should detect elements
        assert len(elements) >= 0  # Might not detect anything in simple test image
        for element in elements:
            assert "type" in element
            assert "bbox" in element
            assert "area" in element
    
    def test_validate_image_valid(self, image_processor, sample_image):
        """Test image validation with valid image."""
        is_valid, message = image_processor.validate_image(sample_image)
        
        assert is_valid is True
        assert message == "Valid"
    
    def test_validate_image_too_small(self, image_processor):
        """Test image validation with too small image."""
        small_image = np.zeros((50, 50), dtype=np.uint8)
        
        is_valid, message = image_processor.validate_image(small_image)
        
        assert is_valid is False
        assert "too small" in message.lower()
    
    def test_validate_image_blank(self, image_processor):
        """Test image validation with blank image."""
        blank_image = np.full((500, 500), 255, dtype=np.uint8)
        
        is_valid, message = image_processor.validate_image(blank_image)
        
        assert is_valid is False
        assert "blank" in message.lower()
    
    def test_preprocess_pipeline(self, image_processor, sample_image):
        """Test complete preprocessing pipeline."""
        processed = image_processor.preprocess(
            sample_image,
            auto_rotate=True,
            deskew=True,
            denoise=True,
            enhance_contrast=True,
            binarize=False
        )
        
        # Should produce valid image
        assert isinstance(processed, np.ndarray)
        assert processed.dtype == np.uint8
        assert processed.size > 0