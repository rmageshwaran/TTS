"""
Audio Processor Module

Handles audio processing and format conversion for TTS output.
Includes format conversion, quality control, real-time streaming, and audio effects.
"""

import logging
import numpy as np
import io
import wave
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"


@dataclass
class AudioProcessingResult:
    """Result of audio processing operation."""
    audio_data: bytes
    sample_rate: int
    channels: int
    format: AudioFormat
    duration: float
    size_bytes: int
    success: bool = True
    error_message: Optional[str] = None


class AudioProcessor:
    """
    Handles audio processing and format conversion.
    
    Features:
    - Audio format conversion (WAV, MP3, etc.)
    - Quality control and normalization
    - Real-time audio streaming
    - Audio effects application
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AudioProcessor with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Audio settings from config
        audio_config = config.get("audio", {})
        self.sample_rate = audio_config.get("sample_rate", 24000)
        self.channels = audio_config.get("channels", 1)
        self.format = AudioFormat(audio_config.get("format", "wav"))
        self.quality = audio_config.get("quality", "high")
        self.normalization = audio_config.get("normalization", True)
        
        # Buffer settings
        self.buffer_size = audio_config.get("buffer_size", 1024)
        self.streaming_enabled = audio_config.get("streaming", True)
        
        # Effects settings
        effects_config = audio_config.get("effects", {})
        self.effects_enabled = effects_config.get("enabled", False)
        self.reverb_level = effects_config.get("reverb", 0.1)
        self.noise_gate = effects_config.get("noise_gate", True)
        
        logger.info(f"AudioProcessor initialized: {self.sample_rate}Hz, {self.channels}ch, {self.format.value}")
    
    def convert_format(self, audio_data: Union[bytes, np.ndarray], 
                      source_format: AudioFormat, target_format: AudioFormat,
                      sample_rate: Optional[int] = None) -> bytes:
        """
        Convert audio between different formats.
        
        Args:
            audio_data: Input audio data
            source_format: Source audio format
            target_format: Target audio format
            sample_rate: Sample rate (if not provided, uses config default)
            
        Returns:
            bytes: Converted audio data
        """
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = self._bytes_to_numpy(audio_data, source_format, sample_rate or self.sample_rate)
            else:
                audio_array = audio_data
            
            # Convert to target format
            if target_format == AudioFormat.WAV:
                return self._numpy_to_wav(audio_array, sample_rate or self.sample_rate)
            elif target_format == AudioFormat.MP3:
                return self._numpy_to_mp3(audio_array, sample_rate or self.sample_rate)
            elif target_format == AudioFormat.OGG:
                return self._numpy_to_ogg(audio_array, sample_rate or self.sample_rate)
            else:
                raise ValueError(f"Unsupported target format: {target_format}")
                
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            raise
    
    def _bytes_to_numpy(self, audio_bytes: bytes, format: AudioFormat, sample_rate: int) -> np.ndarray:
        """Convert bytes to numpy array."""
        if format == AudioFormat.WAV:
            # Read WAV bytes
            with io.BytesIO(audio_bytes) as audio_io:
                with wave.open(audio_io, 'rb') as wav_file:
                    frames = wav_file.readframes(-1)
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    return audio_array.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        else:
            # For other formats, would need additional libraries like pydub
            raise NotImplementedError(f"Conversion from {format} not implemented")
    
    def _numpy_to_wav(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes."""
        # Ensure audio is in the correct range
        audio_array = np.clip(audio_array, -1.0, 1.0)
        
        # Convert to 16-bit integers
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Create WAV file in memory
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 2 bytes = 16 bits
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            return wav_io.getvalue()
    
    def _numpy_to_mp3(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to MP3 bytes."""
        # This would require additional libraries like pydub or lameenc
        # For now, convert to WAV as fallback
        logger.warning("MP3 conversion not implemented, using WAV")
        return self._numpy_to_wav(audio_array, sample_rate)
    
    def _numpy_to_ogg(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to OGG bytes."""
        # This would require additional libraries like soundfile
        # For now, convert to WAV as fallback
        logger.warning("OGG conversion not implemented, using WAV")
        return self._numpy_to_wav(audio_array, sample_rate)
    
    def normalize_audio(self, audio_data: Union[bytes, np.ndarray]) -> np.ndarray:
        """
        Normalize audio levels.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            np.ndarray: Normalized audio array
        """
        # Convert to numpy array if needed
        if isinstance(audio_data, bytes):
            audio_array = self._bytes_to_numpy(audio_data, self.format, self.sample_rate)
        else:
            audio_array = audio_data.copy()
        
        if not self.normalization:
            return audio_array
        
        # Peak normalization
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val * 0.95  # Leave some headroom
        
        # Optional: RMS normalization for consistent loudness
        target_rms = 0.6  # Higher target RMS level for louder audio
        current_rms = np.sqrt(np.mean(audio_array ** 2))
        if current_rms > 0:
            rms_ratio = target_rms / current_rms
            # Apply gentle compression to avoid clipping
            if rms_ratio > 1.0:
                rms_ratio = min(rms_ratio, 3.0)  # Allow more gain
            audio_array = audio_array * rms_ratio
        
        # Final clipping protection
        audio_array = np.clip(audio_array, -1.0, 1.0)
        
        return audio_array
    
    def apply_effects(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Apply audio effects.
        
        Args:
            audio_array: Input audio array
            
        Returns:
            np.ndarray: Processed audio array
        """
        if not self.effects_enabled:
            return audio_array
        
        processed_audio = audio_array.copy()
        
        # Noise gate
        if self.noise_gate:
            processed_audio = self._apply_noise_gate(processed_audio)
        
        # Reverb
        if self.reverb_level > 0:
            processed_audio = self._apply_reverb(processed_audio, self.reverb_level)
        
        return processed_audio
    
    def _apply_noise_gate(self, audio_array: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Apply noise gate to reduce background noise."""
        # Simple noise gate implementation
        mask = np.abs(audio_array) > threshold
        return audio_array * mask
    
    def _apply_reverb(self, audio_array: np.ndarray, level: float) -> np.ndarray:
        """Apply simple reverb effect."""
        # Simple reverb using delayed and attenuated copies
        delay_samples = int(0.1 * self.sample_rate)  # 100ms delay
        
        if len(audio_array) > delay_samples:
            delayed_audio = np.zeros_like(audio_array)
            delayed_audio[delay_samples:] = audio_array[:-delay_samples] * level * 0.3
            
            # Add multiple delays for richer reverb
            delay2_samples = int(0.2 * self.sample_rate)
            if len(audio_array) > delay2_samples:
                delayed_audio[delay2_samples:] += audio_array[:-delay2_samples] * level * 0.15
            
            return audio_array + delayed_audio
        
        return audio_array
    
    def resample_audio(self, audio_array: np.ndarray, 
                      source_rate: int, target_rate: int) -> np.ndarray:
        """
        Resample audio to different sample rate.
        
        Args:
            audio_array: Input audio array
            source_rate: Source sample rate
            target_rate: Target sample rate
            
        Returns:
            np.ndarray: Resampled audio array
        """
        if source_rate == target_rate:
            return audio_array
        
        # Simple linear interpolation resampling
        # For production, consider using scipy.signal.resample or librosa
        ratio = target_rate / source_rate
        new_length = int(len(audio_array) * ratio)
        
        # Create new time indices
        old_indices = np.arange(len(audio_array))
        new_indices = np.linspace(0, len(audio_array) - 1, new_length)
        
        # Interpolate
        resampled_audio = np.interp(new_indices, old_indices, audio_array)
        
        return resampled_audio
    
    def create_stream(self, chunk_size: Optional[int] = None) -> 'AudioStream':
        """
        Create an audio stream for real-time processing.
        
        Args:
            chunk_size: Size of audio chunks for streaming
            
        Returns:
            AudioStream: Stream instance
        """
        return AudioStream(self, chunk_size or self.buffer_size)
    
    def process(self, audio_data: Union[bytes, np.ndarray], 
               target_format: Optional[AudioFormat] = None,
               target_sample_rate: Optional[int] = None) -> AudioProcessingResult:
        """
        Process audio through the complete pipeline.
        
        Args:
            audio_data: Input audio data
            target_format: Target audio format (optional)
            target_sample_rate: Target sample rate (optional)
            
        Returns:
            AudioProcessingResult: Processing result
        """
        try:
            # Convert to numpy array for processing
            if isinstance(audio_data, bytes):
                audio_array = self._bytes_to_numpy(audio_data, self.format, self.sample_rate)
            else:
                audio_array = audio_data.copy()
            
            # Resample if needed
            if target_sample_rate and target_sample_rate != self.sample_rate:
                audio_array = self.resample_audio(audio_array, self.sample_rate, target_sample_rate)
                sample_rate = target_sample_rate
            else:
                sample_rate = self.sample_rate
            
            # Normalize audio
            audio_array = self.normalize_audio(audio_array)
            
            # Apply effects
            audio_array = self.apply_effects(audio_array)
            
            # Convert to target format
            format_to_use = target_format or self.format
            processed_bytes = self.convert_format(audio_array, AudioFormat.WAV, format_to_use, sample_rate)
            
            # Calculate duration
            duration = len(audio_array) / sample_rate
            
            logger.debug(f"Processed audio: {duration:.2f}s, {len(processed_bytes)} bytes, {format_to_use.value}")
            
            return AudioProcessingResult(
                audio_data=processed_bytes,
                sample_rate=sample_rate,
                channels=self.channels,
                format=format_to_use,
                duration=duration,
                size_bytes=len(processed_bytes),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return AudioProcessingResult(
                audio_data=b"",
                sample_rate=0,
                channels=0,
                format=AudioFormat.WAV,
                duration=0.0,
                size_bytes=0,
                success=False,
                error_message=str(e)
            )
    
    def get_audio_info(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Get information about audio data.
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            Dict[str, Any]: Audio information
        """
        try:
            audio_array = self._bytes_to_numpy(audio_data, self.format, self.sample_rate)
            
            return {
                "duration": len(audio_array) / self.sample_rate,
                "samples": len(audio_array),
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "format": self.format.value,
                "size_bytes": len(audio_data),
                "max_amplitude": float(np.max(np.abs(audio_array))),
                "rms_level": float(np.sqrt(np.mean(audio_array ** 2)))
            }
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return {"error": str(e)}


class AudioStream:
    """Real-time audio stream for chunk processing."""
    
    def __init__(self, processor: AudioProcessor, chunk_size: int):
        """
        Initialize audio stream.
        
        Args:
            processor: AudioProcessor instance
            chunk_size: Size of audio chunks
        """
        self.processor = processor
        self.chunk_size = chunk_size
        self.buffer = np.array([])
        self.is_active = False
        
    def start(self):
        """Start the audio stream."""
        self.is_active = True
        self.buffer = np.array([])
        logger.debug("Audio stream started")
    
    def stop(self):
        """Stop the audio stream."""
        self.is_active = False
        logger.debug("Audio stream stopped")
    
    def write_chunk(self, audio_chunk: np.ndarray) -> Optional[bytes]:
        """
        Write audio chunk to stream.
        
        Args:
            audio_chunk: Audio chunk to write
            
        Returns:
            Optional[bytes]: Processed audio if chunk is complete
        """
        if not self.is_active:
            return None
        
        # Add chunk to buffer
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        
        # If buffer is large enough, process a chunk
        if len(self.buffer) >= self.chunk_size:
            # Extract chunk for processing
            chunk_to_process = self.buffer[:self.chunk_size]
            self.buffer = self.buffer[self.chunk_size:]
            
            # Process chunk
            processed_chunk = self.processor.normalize_audio(chunk_to_process)
            processed_chunk = self.processor.apply_effects(processed_chunk)
            
            # Convert to bytes
            return self.processor._numpy_to_wav(processed_chunk, self.processor.sample_rate)
        
        return None
    
    def flush(self) -> Optional[bytes]:
        """
        Flush remaining buffer.
        
        Returns:
            Optional[bytes]: Remaining processed audio
        """
        if len(self.buffer) > 0:
            processed_chunk = self.processor.normalize_audio(self.buffer)
            processed_chunk = self.processor.apply_effects(processed_chunk)
            result = self.processor._numpy_to_wav(processed_chunk, self.processor.sample_rate)
            self.buffer = np.array([])
            return result
        return None