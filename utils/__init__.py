"""
Utility modules for TTS Virtual Microphone application.

This package contains utility classes for logging, error handling,
device management, and other common functionality.
"""

from .logger import setup_logging
from .error_handler import handle_exception, TTSError
from .device_manager import DeviceManager

__all__ = [
    "setup_logging",
    "handle_exception", 
    "TTSError",
    "DeviceManager"
]