"""
Pytest configuration and fixtures for TTS Virtual Microphone tests.

This file contains shared fixtures and configuration for all test modules.
"""

import pytest
import tempfile
import shutil
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Generator

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager, Config
from core.text_processor import TextProcessor
from core.audio_processor import AudioProcessor
from core.tts_engine import TTSEngine
from core.virtual_audio import VirtualAudioInterface
from utils.device_manager import DeviceManager


@pytest.fixture(scope="session")
def test_config() -> Config:
    """Create test configuration."""
    config_dict = {
        "application": {
            "name": "TTS Virtual Microphone Test",
            "version": "1.0.0-test",
            "log_level": "DEBUG",
            "debug": True
        },
        "text": {
            "max_length": 1000,
            "min_length": 1,
            "enable_ssml": True,
            "default_language": "en",
            "chunk_size": 100
        },
        "tts": {
            "default_engine": "mock_tts",
            "fallback_engine": "mock_fallback",
            "model_path": "./tests/mock_models",
            "context_length": 2,
            "voice_settings": {
                "speaker_id": 0,
                "temperature": 1.0,
                "max_audio_length_ms": 5000
            }
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "format": "wav",
            "quality": "medium",
            "normalization": True,
            "buffer_size": 512,
            "streaming": True
        },
        "virtual_audio": {
            "device_name": "Test Audio Device",
            "auto_detect": False,
            "fallback_device": "test_fallback",
            "buffer_size": 512
        },
        "logging": {
            "level": "DEBUG",
            "file": "./tests/logs/test.log",
            "max_size": "1MB",
            "backup_count": 2,
            "console_output": False
        }
    }
    
    config_manager = ConfigManager()
    return config_manager._dict_to_config(config_dict)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_config_file(temp_dir: Path, test_config: Config) -> Path:
    """Create temporary configuration file."""
    config_file = temp_dir / "test_config.yaml"
    
    # Create a basic YAML config
    config_content = """
application:
  name: "Test TTS App"
  version: "1.0.0-test"
  log_level: "DEBUG"

tts:
  default_engine: "mock_tts"
  fallback_engine: "mock_fallback"

audio:
  sample_rate: 16000
  channels: 1
  format: "wav"

virtual_audio:
  device_name: "Test Device"
  auto_detect: false
"""
    
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def mock_audio_data() -> np.ndarray:
    """Generate mock audio data for testing."""
    duration = 1.0  # 1 second
    sample_rate = 16000
    samples = int(duration * sample_rate)
    
    # Generate sine wave
    t = np.linspace(0, duration, samples, False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    return audio_data.astype(np.float32)


@pytest.fixture
def mock_audio_bytes(mock_audio_data: np.ndarray) -> bytes:
    """Convert mock audio data to bytes."""
    # Convert to 16-bit integer and then to bytes
    audio_int16 = (mock_audio_data * 32767).astype(np.int16)
    return audio_int16.tobytes()


@pytest.fixture
def text_processor(test_config: Config) -> TextProcessor:
    """Create TextProcessor instance for testing."""
    config_dict = {
        "text": {
            "max_length": test_config.text.max_length,
            "min_length": test_config.text.min_length,
            "enable_ssml": test_config.text.enable_ssml,
            "default_language": test_config.text.default_language,
            "chunk_size": test_config.text.chunk_size
        }
    }
    return TextProcessor(config_dict)


@pytest.fixture
def audio_processor(test_config: Config) -> AudioProcessor:
    """Create AudioProcessor instance for testing."""
    config_dict = {
        "audio": {
            "sample_rate": test_config.audio.sample_rate,
            "channels": test_config.audio.channels,
            "format": test_config.audio.format,
            "quality": test_config.audio.quality,
            "normalization": test_config.audio.normalization,
            "buffer_size": test_config.audio.buffer_size,
            "streaming": test_config.audio.streaming,
            "effects": {
                "enabled": False,
                "reverb": 0.1,
                "noise_gate": True
            }
        }
    }
    return AudioProcessor(config_dict)


@pytest.fixture
def mock_tts_engine(test_config: Config) -> TTSEngine:
    """Create mock TTS engine for testing."""
    config_dict = {
        "tts": {
            "default_engine": test_config.tts.default_engine,
            "fallback_engine": test_config.tts.fallback_engine,
            "context_length": test_config.tts.context_length
        }
    }
    
    engine = TTSEngine(config_dict)
    
    # Clear plugins and add mock
    engine.plugins.clear()
    
    # Create mock plugin
    mock_plugin = Mock()
    mock_plugin.initialize.return_value = True
    mock_plugin.is_available.return_value = True
    mock_plugin.get_supported_languages.return_value = ["en"]
    mock_plugin.get_sample_rate.return_value = 16000
    mock_plugin.generate_speech.return_value = b"mock_audio_data"
    mock_plugin.cleanup.return_value = None
    
    engine.plugins["mock_tts"] = mock_plugin
    engine.current_engine = "mock_tts"
    
    return engine


@pytest.fixture
def mock_virtual_audio(test_config: Config) -> VirtualAudioInterface:
    """Create mock virtual audio interface for testing."""
    config_dict = {
        "virtual_audio": {
            "device_name": test_config.virtual_audio.device_name,
            "auto_detect": False,  # Disable auto-detect for tests
            "fallback_device": test_config.virtual_audio.fallback_device,
            "buffer_size": test_config.virtual_audio.buffer_size
        },
        "audio": {
            "sample_rate": test_config.audio.sample_rate,
            "channels": test_config.audio.channels
        }
    }
    
    # Mock sounddevice to avoid actual hardware dependencies
    import unittest.mock
    with unittest.mock.patch('sounddevice.query_devices'):
        with unittest.mock.patch('sounddevice.default'):
            virtual_audio = VirtualAudioInterface(config_dict)
    
    return virtual_audio


@pytest.fixture
def mock_device_manager() -> DeviceManager:
    """Create mock device manager for testing."""
    with pytest.MonkeyPatch().context() as mp:
        # Mock sounddevice functions
        mock_devices = [
            {
                'name': 'Test Input Device',
                'max_input_channels': 2,
                'max_output_channels': 0,
                'default_samplerate': 44100.0,
                'hostapi': 0
            },
            {
                'name': 'Test Output Device', 
                'max_input_channels': 0,
                'max_output_channels': 2,
                'default_samplerate': 44100.0,
                'hostapi': 0
            },
            {
                'name': 'CABLE Input (VB-Audio Virtual Cable)',
                'max_input_channels': 0,
                'max_output_channels': 2,
                'default_samplerate': 44100.0,
                'hostapi': 0
            }
        ]
        
        mp.setattr('sounddevice.query_devices', lambda: mock_devices)
        mp.setattr('sounddevice.default.device', [1, 1])  # Default input/output
        
        device_manager = DeviceManager()
        yield device_manager


@pytest.fixture
def sample_texts() -> Dict[str, str]:
    """Sample texts for testing."""
    return {
        "short": "Hello world",
        "medium": "This is a medium length text for testing purposes. It contains multiple sentences.",
        "long": "This is a very long text " * 50,
        "ssml": '<speak>Hello <break time="500ms"/> world!</speak>',
        "numbers": "I have 123 apples and 456 oranges",
        "punctuation": "Hello! How are you? I'm fine... Thanks.",
        "empty": "",
        "whitespace": "   \n\t  ",
        "unicode": "Hello 世界 🌍",
        "email": "Contact us at test@example.com",
        "url": "Visit https://example.com for more info"
    }


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    import logging
    
    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress noisy loggers during tests
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


@pytest.fixture
def audio_test_files(temp_dir: Path) -> Dict[str, Path]:
    """Create test audio files."""
    import wave
    
    files = {}
    
    # Create test WAV file
    wav_file = temp_dir / "test.wav"
    with wave.open(str(wav_file), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        
        # Generate 1 second of sine wave
        samples = 16000
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, samples))
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav.writeframes(audio_int16.tobytes())
    
    files["wav"] = wav_file
    
    # Create empty file for error testing
    empty_file = temp_dir / "empty.wav"
    empty_file.write_bytes(b"")
    files["empty"] = empty_file
    
    return files


@pytest.fixture
def performance_timer():
    """Fixture for measuring test performance."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0
    
    return Timer()


# Pytest markers for test organization
pytestmark = pytest.mark.unit


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "audio: mark test as requiring audio hardware")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests in tests/unit/
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to all tests in tests/integration/
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that might take a while
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Create test directories
    test_dirs = ["tests/logs", "tests/temp", "tests/mock_models"]
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup after all tests
    for dir_path in test_dirs:
        test_dir = Path(dir_path)
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)