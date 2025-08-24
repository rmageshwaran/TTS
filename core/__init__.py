"""
Core modules for TTS Virtual Microphone application.

This package contains the core components that handle the main functionality:
- Text processing and validation
- TTS engine management and coordination
- Audio processing and format conversion
- Virtual audio cable interface
"""

__version__ = "1.0.0"
__author__ = "TTS Virtual Microphone Team"

# Core component imports
from .text_processor import TextProcessor
from .tts_engine import TTSEngine
from .audio_processor import AudioProcessor
from .virtual_audio import VirtualAudioInterface

__all__ = [
    "TextProcessor",
    "TTSEngine", 
    "AudioProcessor",
    "VirtualAudioInterface"
]