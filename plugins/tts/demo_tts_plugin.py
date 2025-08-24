"""
Demo TTS Plugin for Pipeline Testing
Simple plugin that generates basic synthetic speech for demonstration purposes
"""

import numpy as np
import soundfile as sf
import io
from core.tts_engine import BaseTTSPlugin


class DemoTTSPlugin(BaseTTSPlugin):
    """Demo TTS plugin that generates basic synthetic speech."""
    
    def __init__(self, config):
        self.config = config
        self.sample_rate = 22050
        self.is_loaded = True
        
    def get_name(self) -> str:
        return "demo_tts"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def get_description(self) -> str:
        return "Demo TTS plugin for pipeline testing"
    
    def is_available(self) -> bool:
        return True
    
    def initialize(self, config=None) -> bool:
        self.is_loaded = True
        return True
    
    def generate_speech(self, text: str, speaker_id: int = None, context: list = None) -> bytes:
        """Generate synthetic speech audio."""
        try:
            # Simple synthetic speech generation
            duration = len(text) * 0.1  # ~100ms per character
            samples = int(duration * self.sample_rate)
            
            # Generate speech-like audio with multiple frequency components
            t = np.linspace(0, duration, samples, False)
            
            # Base frequency varies with text content
            base_freq = 150 + (hash(text) % 100)  # 150-250 Hz range
            
            # Create speech-like waveform with maximum amplitude
            audio = (
                1.8 * np.sin(2 * np.pi * base_freq * t) +
                0.9 * np.sin(2 * np.pi * base_freq * 2 * t) +
                0.5 * np.sin(2 * np.pi * base_freq * 3 * t)
            )
            
            # Use constant amplitude for maximum volume
            audio = audio * 0.95  # Slight headroom to prevent clipping
            
            # Add very slight noise for realism
            noise = np.random.normal(0, 0.005, samples)
            audio = audio + noise
            
            # Normalize
            audio = np.clip(audio, -1.0, 1.0)
            
            # Convert to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio, self.sample_rate, format='WAV')
            audio_bytes = buffer.getvalue()
            
            return audio_bytes
            
        except Exception as e:
            # Return empty bytes on error
            return b""
    
    def get_sample_rate(self) -> int:
        """Get the sample rate used by this TTS engine."""
        return self.sample_rate
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return ["en"]
    
    def cleanup(self):
        """Clean up resources."""
        pass