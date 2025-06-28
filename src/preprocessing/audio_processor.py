"""Audio preprocessing utilities for STT."""

import numpy as np
import librosa
import scipy.signal
from typing import Tuple, Optional
import webrtcvad
from loguru import logger
import torch
import torchaudio


class AudioProcessor:
    """Audio preprocessing and enhancement utilities."""
    
    def __init__(self):
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        self.target_sample_rate = 16000
        
    def preprocess(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_sample_rate: int = 16000,
        normalize: bool = True,
        denoise: bool = False
    ) -> np.ndarray:
        """
        Preprocess audio for STT.
        
        Args:
            audio: Audio signal
            sample_rate: Original sample rate
            target_sample_rate: Target sample rate
            normalize: Whether to normalize audio
            denoise: Whether to apply denoising
        
        Returns:
            Preprocessed audio
        """
        try:
            # Resample if needed
            if sample_rate != target_sample_rate:
                audio = self.resample(audio, sample_rate, target_sample_rate)
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = self.to_mono(audio)
            
            # Normalize
            if normalize:
                audio = self.normalize_audio(audio)
            
            # Denoise if requested
            if denoise:
                audio = self.denoise(audio, target_sample_rate)
            
            # Apply high-pass filter to remove low-frequency noise
            audio = self.high_pass_filter(audio, target_sample_rate, cutoff=80)
            
            # Apply AGC (Automatic Gain Control)
            audio = self.apply_agc(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing error: {e}")
            return audio
    
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        try:
            # Use librosa for high-quality resampling
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except Exception as e:
            logger.warning(f"Resampling error, using scipy: {e}")
            # Fallback to scipy
            num_samples = int(len(audio) * target_sr / orig_sr)
            return scipy.signal.resample(audio, num_samples)
    
    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert stereo/multi-channel audio to mono."""
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            return np.mean(audio, axis=-1)
        else:
            return np.mean(audio, axis=tuple(range(1, audio.ndim)))
    
    def normalize_audio(self, audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
        """Normalize audio to target dBFS."""
        if len(audio) == 0:
            return audio
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return audio
        
        # Convert to dBFS
        current_dbfs = 20 * np.log10(rms)
        
        # Calculate gain needed
        gain_db = target_dbfs - current_dbfs
        gain_linear = 10 ** (gain_db / 20)
        
        # Apply gain with clipping protection
        normalized = audio * gain_linear
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized
    
    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply spectral subtraction denoising."""
        try:
            # Estimate noise from first 0.5 seconds
            noise_duration = min(int(0.5 * sample_rate), len(audio) // 4)
            if noise_duration < 1024:
                return audio  # Too short for denoising
            
            noise_sample = audio[:noise_duration]
            
            # STFT
            hop_length = 512
            win_length = 2048
            
            stft = librosa.stft(audio, hop_length=hop_length, win_length=win_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise spectrum
            noise_stft = librosa.stft(noise_sample, hop_length=hop_length, win_length=win_length)
            noise_magnitude = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01  # Spectral floor
            
            denoised_magnitude = magnitude - alpha * noise_magnitude
            denoised_magnitude = np.maximum(denoised_magnitude, beta * magnitude)
            
            # Reconstruct signal
            denoised_stft = denoised_magnitude * np.exp(1j * phase)
            denoised_audio = librosa.istft(denoised_stft, hop_length=hop_length, win_length=win_length)
            
            # Ensure same length as input
            if len(denoised_audio) != len(audio):
                denoised_audio = np.resize(denoised_audio, len(audio))
            
            return denoised_audio
            
        except Exception as e:
            logger.warning(f"Denoising failed, returning original: {e}")
            return audio
    
    def high_pass_filter(
        self,
        audio: np.ndarray,
        sample_rate: int,
        cutoff: float = 80.0,
        order: int = 4
    ) -> np.ndarray:
        """Apply high-pass filter to remove low-frequency noise."""
        try:
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            
            # Design Butterworth high-pass filter
            b, a = scipy.signal.butter(order, normalized_cutoff, btype='high', analog=False)
            
            # Apply filter
            filtered = scipy.signal.filtfilt(b, a, audio)
            return filtered
            
        except Exception as e:
            logger.warning(f"High-pass filtering failed: {e}")
            return audio
    
    def apply_agc(self, audio: np.ndarray, target_level: float = 0.1, attack: float = 0.1, release: float = 0.9) -> np.ndarray:
        """Apply Automatic Gain Control."""
        if len(audio) == 0:
            return audio
        
        # Simple AGC implementation
        output = np.zeros_like(audio)
        gain = 1.0
        
        for i, sample in enumerate(audio):
            # Calculate instantaneous amplitude
            amplitude = abs(sample)
            
            # Calculate desired gain
            if amplitude > 0:
                desired_gain = target_level / amplitude
            else:
                desired_gain = 1.0
            
            # Smooth gain changes
            if desired_gain < gain:
                # Attack (fast gain reduction)
                gain = gain * (1 - attack) + desired_gain * attack
            else:
                # Release (slow gain increase)
                gain = gain * (1 - release) + desired_gain * release
            
            # Limit gain range
            gain = np.clip(gain, 0.1, 10.0)
            
            # Apply gain
            output[i] = sample * gain
        
        return output
    
    def apply_vad(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_duration: int = 30,
        aggressiveness: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Voice Activity Detection.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            frame_duration: Frame duration in ms (10, 20, or 30)
            aggressiveness: VAD aggressiveness (0-3)
        
        Returns:
            Tuple of (speech_audio, vad_flags)
        """
        try:
            # Ensure audio is 16-bit PCM
            if audio.dtype != np.int16:
                audio_pcm = (audio * 32767).astype(np.int16)
            else:
                audio_pcm = audio
            
            # Frame size in samples
            frame_size = int(sample_rate * frame_duration / 1000)
            
            # Initialize VAD
            vad = webrtcvad.Vad(aggressiveness)
            
            # Process frames
            frames = []
            vad_flags = []
            
            for start in range(0, len(audio_pcm), frame_size):
                end = min(start + frame_size, len(audio_pcm))
                frame = audio_pcm[start:end]
                
                # Pad frame if necessary
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')
                
                # VAD decision
                try:
                    is_speech = vad.is_speech(frame.tobytes(), sample_rate)
                except:
                    is_speech = True  # Default to speech if VAD fails
                
                frames.append(frame)
                vad_flags.append(is_speech)
            
            # Combine frames and flags
            speech_audio = np.concatenate(frames)[:len(audio_pcm)]
            vad_array = np.array(vad_flags)
            
            return speech_audio.astype(audio.dtype), vad_array
            
        except Exception as e:
            logger.warning(f"VAD failed, returning original: {e}")
            return audio, np.ones(len(audio) // (sample_rate * frame_duration // 1000), dtype=bool)
    
    def detect_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        silence_threshold: float = 0.01,
        min_silence_duration: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        Detect silence segments in audio.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            silence_threshold: RMS threshold for silence
            min_silence_duration: Minimum silence duration in seconds
        
        Returns:
            List of (start_time, end_time) tuples for silence segments
        """
        # Calculate frame-wise RMS
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Find silence frames
        silence_frames = rms < silence_threshold
        
        # Convert to time segments
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            time = i * hop_length / sample_rate
            
            if is_silent and not in_silence:
                # Start of silence
                silence_start = time
                in_silence = True
            elif not is_silent and in_silence:
                # End of silence
                silence_duration = time - silence_start
                if silence_duration >= min_silence_duration:
                    silence_segments.append((silence_start, time))
                in_silence = False
        
        # Handle case where audio ends in silence
        if in_silence:
            silence_duration = len(audio) / sample_rate - silence_start
            if silence_duration >= min_silence_duration:
                silence_segments.append((silence_start, len(audio) / sample_rate))
        
        return silence_segments
    
    def split_on_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        silence_threshold: float = 0.01,
        min_silence_duration: float = 0.5,
        keep_silence: bool = True
    ) -> List[np.ndarray]:
        """Split audio on silence segments."""
        silence_segments = self.detect_silence(
            audio, sample_rate, silence_threshold, min_silence_duration
        )
        
        if not silence_segments:
            return [audio]
        
        # Split audio
        segments = []
        last_end = 0
        
        for silence_start, silence_end in silence_segments:
            start_sample = int(last_end * sample_rate)
            end_sample = int(silence_start * sample_rate)
            
            if end_sample > start_sample:
                segment = audio[start_sample:end_sample]
                
                if keep_silence and len(segments) > 0:
                    # Add a bit of silence before each segment (except first)
                    silence_samples = int(0.1 * sample_rate)  # 100ms
                    silence_padding = np.zeros(silence_samples)
                    segment = np.concatenate([silence_padding, segment])
                
                segments.append(segment)
            
            last_end = silence_end
        
        # Add remaining audio after last silence
        if last_end < len(audio) / sample_rate:
            start_sample = int(last_end * sample_rate)
            segment = audio[start_sample:]
            segments.append(segment)
        
        return segments
    
    def extract_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """Extract audio features for analysis."""
        try:
            features = {}
            
            # MFCCs
            features['mfcc'] = librosa.feature.mfcc(
                y=audio, sr=sample_rate, n_mfcc=13
            )
            
            # Spectral features
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                y=audio, sr=sample_rate
            )
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
                y=audio, sr=sample_rate
            )
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
                y=audio, sr=sample_rate
            )
            
            # Zero crossing rate
            features['zcr'] = librosa.feature.zero_crossing_rate(audio)
            
            # RMS energy
            features['rms'] = librosa.feature.rms(y=audio)
            
            # Tempo and beat
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
            features['tempo'] = tempo
            features['beats'] = beats
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}