"""
Virtual Audio Interface Module

Handles interface with virtual audio cable software.
Includes VB-Cable integration, device detection, and audio routing management.
"""

import logging
import sounddevice as sd
import numpy as np
import threading
import time
import queue
import soundfile as sf
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO

# Import VB-Cable interface
try:
    from core.vb_cable_interface import VBCableInterface, VBCableStatus
    VB_CABLE_AVAILABLE = True
except ImportError:
    VB_CABLE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioDeviceType(Enum):
    """Types of audio devices."""
    INPUT = "input"
    OUTPUT = "output"
    VIRTUAL = "virtual"


@dataclass
class AudioDevice:
    """Audio device information."""
    id: int
    name: str
    device_type: AudioDeviceType
    channels: int
    sample_rate: float
    is_available: bool = True


@dataclass
class AudioRoutingResult:
    """Result of audio routing operation."""
    success: bool
    device_used: str
    latency: float
    bytes_processed: int = 0
    buffer_health: float = 0.0
    error_message: Optional[str] = None


@dataclass
class StreamMetrics:
    """Real-time streaming performance metrics."""
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    average_latency: float = 0.0
    buffer_underruns: int = 0
    buffer_overruns: int = 0
    last_update: float = field(default_factory=time.time)
    
    def update_success(self, latency: float):
        """Update metrics for successful chunk."""
        self.total_chunks += 1
        self.successful_chunks += 1
        self.average_latency = ((self.average_latency * (self.successful_chunks - 1)) + latency) / self.successful_chunks
        self.last_update = time.time()
    
    def update_failure(self):
        """Update metrics for failed chunk."""
        self.total_chunks += 1
        self.failed_chunks += 1
        self.last_update = time.time()
    
    def get_success_rate(self) -> float:
        """Get success rate percentage."""
        if self.total_chunks == 0:
            return 100.0
        return (self.successful_chunks / self.total_chunks) * 100.0


class VirtualAudioInterface:
    """
    Enhanced Virtual Audio Interface with real-time streaming capabilities.
    
    Features:
    - VB-Cable integration with advanced detection
    - High-performance real-time audio streaming
    - Intelligent buffer management
    - Latency optimization
    - Error recovery and fallback mechanisms
    - Performance monitoring and metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Enhanced VirtualAudioInterface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Virtual audio settings
        virtual_config = config.get("virtual_audio", {})
        self.device_name = virtual_config.get("device_name", "CABLE Input")
        self.auto_detect = virtual_config.get("auto_detect", True)
        self.fallback_device = virtual_config.get("fallback_device", "default")
        self.buffer_size = virtual_config.get("buffer_size", 1024)
        self.enable_monitoring = virtual_config.get("enable_monitoring", True)
        
        # Advanced streaming settings
        self.stream_buffer_size = virtual_config.get("stream_buffer_size", 8192)
        self.max_queue_size = virtual_config.get("max_queue_size", 50)
        self.latency_mode = virtual_config.get("latency_mode", "low")  # low, normal, high
        self.auto_recovery = virtual_config.get("auto_recovery", True)
        
        # Audio settings
        audio_config = config.get("audio", {})
        self.sample_rate = audio_config.get("sample_rate", 24000)
        self.channels = audio_config.get("channels", 1)
        
        # Internal state
        self.available_devices: List[AudioDevice] = []
        self.current_device: Optional[AudioDevice] = None
        self.stream: Optional[sd.OutputStream] = None
        
        # Enhanced streaming features
        self.audio_queue: queue.Queue = queue.Queue(maxsize=self.max_queue_size)
        self.streaming_thread: Optional[threading.Thread] = None
        self.streaming_active = False
        self.stream_stop_event = threading.Event()
        self.metrics = StreamMetrics()
        self._lock = threading.Lock()
        
        # Performance optimization
        self._setup_latency_settings()
        
        # Callbacks
        self.error_callbacks: List[Callable] = []
        self.metrics_callbacks: List[Callable] = []
        
        # VB-Cable integration
        self.vb_cable_interface: Optional[VBCableInterface] = None
        self.vb_cable_enabled = virtual_config.get("enable_vb_cable_integration", True)
        
        logger.info("Enhanced VirtualAudioInterface initialized")
        
        # Detect available devices
        self._detect_devices()
        
        # Initialize VB-Cable integration if available
        if self.vb_cable_enabled and VB_CABLE_AVAILABLE:
            self._initialize_vb_cable_integration()
    
    def _setup_latency_settings(self):
        """Setup optimal latency settings based on mode."""
        latency_configs = {
            'low': {
                'buffer_size': 512,
                'stream_buffer_size': 4096,
                'max_queue_size': 20
            },
            'normal': {
                'buffer_size': 1024,
                'stream_buffer_size': 8192,
                'max_queue_size': 50
            },
            'high': {
                'buffer_size': 2048,
                'stream_buffer_size': 16384,
                'max_queue_size': 100
            }
        }
        
        config = latency_configs.get(self.latency_mode, latency_configs['normal'])
        self.buffer_size = config['buffer_size']
        self.stream_buffer_size = config['stream_buffer_size']
        self.max_queue_size = config['max_queue_size']
        
        logger.debug(f"Latency mode '{self.latency_mode}': buffer_size={self.buffer_size}, stream_buffer={self.stream_buffer_size}")
    
    def _initialize_vb_cable_integration(self):
        """Initialize VB-Cable integration."""
        try:
            self.vb_cable_interface = VBCableInterface(self.config)
            
            # Add callbacks for VB-Cable events
            self.vb_cable_interface.add_status_change_callback(self._on_vb_cable_status_change)
            self.vb_cable_interface.add_routing_callback(self._on_vb_cable_routing_event)
            
            vb_status = self.vb_cable_interface.get_status()
            logger.info(f"VB-Cable integration initialized: {vb_status['installation_status']}")
            
        except Exception as e:
            logger.error(f"VB-Cable integration failed: {e}")
            self.vb_cable_interface = None
    
    def _on_vb_cable_status_change(self, event_type: str, data: Any):
        """Handle VB-Cable status change events."""
        logger.debug(f"VB-Cable status change: {event_type}")
        # Notify error callbacks about VB-Cable status changes
        self._notify_error(f"vb_cable_{event_type}", str(data))
    
    def _on_vb_cable_routing_event(self, event_type: str, result):
        """Handle VB-Cable routing events."""
        logger.debug(f"VB-Cable routing event: {event_type}")
        if event_type == "routing_failure":
            self._notify_error("vb_cable_routing_failed", result.error_message)
    
    def _detect_devices(self):
        """Detect available audio devices."""
        try:
            # Get all audio devices
            devices = sd.query_devices()
            self.available_devices = []
            
            for i, device_info in enumerate(devices):
                if device_info['max_output_channels'] > 0:  # Output device
                    device = AudioDevice(
                        id=i,
                        name=device_info['name'],
                        device_type=self._classify_device(device_info['name']),
                        channels=device_info['max_output_channels'],
                        sample_rate=device_info['default_samplerate']
                    )
                    self.available_devices.append(device)
            
            logger.info(f"Detected {len(self.available_devices)} audio output devices")
            
            # Auto-select device if enabled
            if self.auto_detect:
                self._auto_select_device()
                
        except Exception as e:
            logger.error(f"Failed to detect audio devices: {e}")
    
    def _classify_device(self, device_name: str) -> AudioDeviceType:
        """Classify device type based on name."""
        device_name_lower = device_name.lower()
        
        # Check for virtual audio cable devices
        virtual_keywords = ['cable', 'virtual', 'vb-audio', 'voicemeeter', 'vac']
        if any(keyword in device_name_lower for keyword in virtual_keywords):
            return AudioDeviceType.VIRTUAL
        
        # Check for input devices (microphones)
        input_keywords = ['microphone', 'mic', 'input', 'capture']
        if any(keyword in device_name_lower for keyword in input_keywords):
            return AudioDeviceType.INPUT
        
        return AudioDeviceType.OUTPUT
    
    def _auto_select_device(self):
        """Automatically select the best available device."""
        # First priority: exact match with configured device name
        for device in self.available_devices:
            if self.device_name.lower() in device.name.lower():
                self.current_device = device
                logger.info(f"Auto-selected device: {device.name}")
                return
        
        # Second priority: any virtual audio device
        virtual_devices = [d for d in self.available_devices if d.device_type == AudioDeviceType.VIRTUAL]
        if virtual_devices:
            self.current_device = virtual_devices[0]
            logger.info(f"Auto-selected virtual device: {self.current_device.name}")
            return
        
        # Third priority: default output device
        try:
            default_device_id = sd.default.device[1]  # Output device
            for device in self.available_devices:
                if device.id == default_device_id:
                    self.current_device = device
                    logger.info(f"Auto-selected default device: {device.name}")
                    return
        except:
            pass
        
        # Fallback: first available device
        if self.available_devices:
            self.current_device = self.available_devices[0]
            logger.warning(f"Fallback to first device: {self.current_device.name}")
    
    def list_devices(self) -> List[AudioDevice]:
        """
        Get list of available audio devices.
        
        Returns:
            List[AudioDevice]: Available devices
        """
        return self.available_devices.copy()
    
    def select_device(self, device_id: Optional[int] = None, device_name: Optional[str] = None) -> bool:
        """
        Select audio device for output.
        
        Args:
            device_id: Device ID to select
            device_name: Device name to select
            
        Returns:
            bool: True if device selected successfully
        """
        target_device = None
        
        if device_id is not None:
            # Select by ID
            for device in self.available_devices:
                if device.id == device_id:
                    target_device = device
                    break
        elif device_name:
            # Select by name (partial match)
            for device in self.available_devices:
                if device_name.lower() in device.name.lower():
                    target_device = device
                    break
        
        if target_device:
            self.current_device = target_device
            logger.info(f"Selected device: {target_device.name}")
            return True
        else:
            logger.error(f"Device not found: ID={device_id}, Name={device_name}")
            return False
    
    def is_vb_cable_available(self) -> bool:
        """Check if VB-Cable is available."""
        for device in self.available_devices:
            if "cable" in device.name.lower() and device.device_type == AudioDeviceType.VIRTUAL:
                return True
        return False
    
    def get_current_device(self) -> Optional[AudioDevice]:
        """Get currently selected device."""
        return self.current_device
    
    def create_stream(self) -> bool:
        """
        Create audio output stream with Windows WDM-KS compatibility.
        
        Returns:
            bool: True if stream created successfully
        """
        if not self.current_device:
            logger.error("No device selected")
            return False
        
        try:
            # Close existing stream if any
            self.close_stream()
            
            # Try different stream configurations for Windows compatibility
            stream_configs = [
                # Configuration 1: Non-blocking with callback
                {
                    'device': self.current_device.id,
                    'channels': self.channels,
                    'samplerate': self.sample_rate,
                    'blocksize': self.buffer_size,
                    'dtype': np.float32,
                    'callback': self._audio_callback,
                    'finished_callback': self._stream_finished_callback
                },
                # Configuration 2: Smaller buffer size
                {
                    'device': self.current_device.id,
                    'channels': self.channels,
                    'samplerate': self.sample_rate,
                    'blocksize': 512,  # Smaller buffer
                    'dtype': np.float32,
                    'callback': self._audio_callback,
                    'finished_callback': self._stream_finished_callback
                },
                # Configuration 3: Different sample rate
                {
                    'device': self.current_device.id,
                    'channels': self.channels,
                    'samplerate': 44100,  # Standard sample rate
                    'blocksize': self.buffer_size,
                    'dtype': np.float32,
                    'callback': self._audio_callback,
                    'finished_callback': self._stream_finished_callback
                },
                # Configuration 4: Default device fallback
                {
                    'device': None,  # Use default device
                    'channels': self.channels,
                    'samplerate': self.sample_rate,
                    'blocksize': self.buffer_size,
                    'dtype': np.float32,
                    'callback': self._audio_callback,
                    'finished_callback': self._stream_finished_callback
                }
            ]
            
            for i, config in enumerate(stream_configs):
                try:
                    logger.debug(f"Trying stream configuration {i+1}")
                    self.stream = sd.OutputStream(**config)
                    self.stream.start()
                    logger.info(f"Created audio stream with config {i+1}: {self.current_device.name}")
                    return True
                    
                except Exception as e:
                    logger.debug(f"Stream config {i+1} failed: {e}")
                    if self.stream:
                        try:
                            self.stream.close()
                        except:
                            pass
                        self.stream = None
                    continue
            
            # If all configurations fail, try file output as fallback
            logger.warning("All stream configurations failed, using file output fallback")
            return self._setup_file_fallback()
            
        except Exception as e:
            logger.error(f"Failed to create audio stream: {e}")
            return self._setup_file_fallback()
    
    def _audio_callback(self, outdata, frames, time, status):
        """Audio callback for non-blocking stream."""
        try:
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            if self.audio_queue.empty():
                # Fill with silence if no audio data
                outdata.fill(0)
                return
            
            # Get audio data from queue
            audio_chunk = self.audio_queue.get_nowait()
            
            # Ensure correct shape
            if len(audio_chunk.shape) == 1:
                audio_chunk = audio_chunk.reshape(-1, 1)
            
            # Copy data to output buffer
            if audio_chunk.shape[0] >= frames:
                outdata[:frames] = audio_chunk[:frames]
            else:
                # Pad with silence if not enough data
                outdata[:audio_chunk.shape[0]] = audio_chunk
                outdata[audio_chunk.shape[0]:] = 0
                
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
            outdata.fill(0)
    
    def _stream_finished_callback(self):
        """Callback when stream finishes."""
        logger.debug("Audio stream finished")
    
    def _setup_file_fallback(self) -> bool:
        """Setup file output as fallback when streaming fails."""
        try:
            self.file_fallback = True
            self.output_file = f"tts_output_{int(time.time())}.wav"
            logger.info(f"Using file output fallback: {self.output_file}")
            return True
        except Exception as e:
            logger.error(f"File fallback setup failed: {e}")
            return False
    
    def play_audio_file(self, audio_file: str) -> bool:
        """
        Play audio file using system default player as fallback.
        
        Args:
            audio_file: Path to audio file to play
            
        Returns:
            bool: True if playback started successfully
        """
        try:
            import subprocess
            import platform
            
            system = platform.system()
            
            if system == "Windows":
                # Use Windows Media Player or default audio player
                subprocess.Popen(["start", audio_file], shell=True)
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", audio_file])
            else:  # Linux
                subprocess.Popen(["xdg-open", audio_file])
            
            logger.info(f"Started audio playback: {audio_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio playback: {e}")
            return False
    
    def get_audio_devices_info(self) -> Dict[str, Any]:
        """
        Get detailed information about available audio devices.
        
        Returns:
            Dict with device information
        """
        try:
            devices_info = {
                "host_apis": [],
                "input_devices": [],
                "output_devices": [],
                "virtual_devices": []
            }
            
            # Get host APIs
            host_apis = sd.query_hostapis()
            for i, api in enumerate(host_apis):
                devices_info["host_apis"].append({
                    "id": i,
                    "name": api["name"],
                    "default_input": api["defaultInputDevice"],
                    "default_output": api["defaultOutputDevice"],
                    "device_count": api["deviceCount"]
                })
            
            # Get all devices
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                device_info = {
                    "id": i,
                    "name": device["name"],
                    "host_api": device["hostapi"],
                    "max_input_channels": device["maxInputChannels"],
                    "max_output_channels": device["maxOutputChannels"],
                    "default_sample_rate": device["defaultSampleRate"],
                    "is_virtual": "cable" in device["name"].lower() or "virtual" in device["name"].lower()
                }
                
                if device["maxInputChannels"] > 0:
                    devices_info["input_devices"].append(device_info)
                
                if device["maxOutputChannels"] > 0:
                    devices_info["output_devices"].append(device_info)
                    
                    if device_info["is_virtual"]:
                        devices_info["virtual_devices"].append(device_info)
            
            return devices_info
            
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {"error": str(e)}
    
    def close_stream(self):
        """Close audio output stream."""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                logger.debug("Audio stream closed")
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
    
    def route_audio(self, audio_data: bytes) -> AudioRoutingResult:
        """
        Route audio data to virtual device with Windows WDM-KS compatibility.
        
        Args:
            audio_data: Audio data to route
            
        Returns:
            AudioRoutingResult: Routing result
        """
        if not self.current_device:
            return AudioRoutingResult(
                success=False,
                device_used="none",
                latency=0.0,
                error_message="No device selected"
            )
        
        try:
            import time
            start_time = time.time()
            
            # Convert bytes to numpy array
            audio_array = self._bytes_to_numpy(audio_data)
            
            # Handle file fallback mode
            if hasattr(self, 'file_fallback') and self.file_fallback:
                return self._route_audio_to_file(audio_array, start_time)
            
            # Create stream if not exists
            if not self.stream:
                if not self.create_stream():
                    return AudioRoutingResult(
                        success=False,
                        device_used=self.current_device.name,
                        latency=0.0,
                        error_message="Failed to create audio stream"
                    )
            
            # Try different routing methods
            routing_methods = [
                self._route_audio_callback,
                self._route_audio_direct,
                self._route_audio_queue
            ]
            
            for method in routing_methods:
                try:
                    result = method(audio_array, start_time)
                    if result.success:
                        return result
                except Exception as e:
                    logger.debug(f"Routing method {method.__name__} failed: {e}")
                    continue
            
            # If all methods fail, use file fallback
            logger.warning("All routing methods failed, using file fallback")
            return self._route_audio_to_file(audio_array, start_time)
            
        except Exception as e:
            logger.error(f"Audio routing failed: {e}")
            return AudioRoutingResult(
                success=False,
                device_used=self.current_device.name if self.current_device else "unknown",
                latency=0.0,
                error_message=str(e)
            )
    
    def _route_audio_callback(self, audio_array: np.ndarray, start_time: float) -> AudioRoutingResult:
        """Route audio using callback-based streaming."""
        try:
            # Add audio to queue for callback processing
            if not self.audio_queue.full():
                self.audio_queue.put(audio_array)
                
                latency = (time.time() - start_time) * 1000
                return AudioRoutingResult(
                    success=True,
                    device_used=self.current_device.name,
                    latency=latency
                )
            else:
                raise Exception("Audio queue full")
                
        except Exception as e:
            raise Exception(f"Callback routing failed: {e}")
    
    def _route_audio_direct(self, audio_array: np.ndarray, start_time: float) -> AudioRoutingResult:
        """Route audio using direct stream write (fallback)."""
        try:
            # Try direct write (may fail on Windows WDM-KS)
            self.stream.write(audio_array)
            
            latency = (time.time() - start_time) * 1000
            return AudioRoutingResult(
                success=True,
                device_used=self.current_device.name,
                latency=latency
            )
            
        except Exception as e:
            raise Exception(f"Direct routing failed: {e}")
    
    def _route_audio_queue(self, audio_array: np.ndarray, start_time: float) -> AudioRoutingResult:
        """Route audio using queue-based streaming."""
        try:
            # Use the existing queue system
            if hasattr(self, 'audio_queue') and not self.audio_queue.full():
                self.audio_queue.put(audio_array)
                
                latency = (time.time() - start_time) * 1000
                return AudioRoutingResult(
                    success=True,
                    device_used=self.current_device.name,
                    latency=latency
                )
            else:
                raise Exception("Audio queue not available or full")
                
        except Exception as e:
            raise Exception(f"Queue routing failed: {e}")
    
    def _route_audio_to_file(self, audio_array: np.ndarray, start_time: float) -> AudioRoutingResult:
        """Route audio to file as fallback."""
        try:
            import soundfile as sf
            
            # Save audio to file
            if hasattr(self, 'output_file') and self.output_file:
                sf.write(self.output_file, audio_array, self.sample_rate)
                
                latency = (time.time() - start_time) * 1000
                return AudioRoutingResult(
                    success=True,
                    device_used=f"file:{self.output_file}",
                    latency=latency
                )
            else:
                # Create temporary file
                temp_file = f"tts_output_{int(time.time())}.wav"
                sf.write(temp_file, audio_array, self.sample_rate)
                
                latency = (time.time() - start_time) * 1000
                return AudioRoutingResult(
                    success=True,
                    device_used=f"file:{temp_file}",
                    latency=latency
                )
                
        except Exception as e:
            raise Exception(f"File routing failed: {e}")
    
    def _bytes_to_numpy(self, audio_bytes: bytes, source_sample_rate: Optional[int] = None) -> np.ndarray:
        """Enhanced audio bytes to numpy conversion with format detection."""
        try:
            # First, try to detect format using soundfile
            try:
                with BytesIO(audio_bytes) as audio_buffer:
                    audio_data, detected_sr = sf.read(audio_buffer, dtype='float32')
                    
                    # Resample if needed
                    if source_sample_rate and detected_sr != self.sample_rate:
                        audio_data = self._resample_audio(audio_data, detected_sr, self.sample_rate)
                    
                    # Handle channel conversion
                    if len(audio_data.shape) == 1:  # Mono
                        if self.channels == 1:
                            return audio_data
                        else:
                            # Convert mono to stereo by duplicating
                            return np.column_stack([audio_data] * self.channels)
                    else:  # Multi-channel
                        if self.channels == 1:
                            # Convert to mono by averaging channels
                            return np.mean(audio_data, axis=1)
                        else:
                            return audio_data
                            
            except Exception:
                # Fallback to raw format assumption
                logger.debug("Soundfile detection failed, using raw format assumption")
                
                # Try different bit depths
                for dtype, divisor in [(np.int16, 32768.0), (np.int32, 2147483648.0), (np.float32, 1.0)]:
                    try:
                        audio_array = np.frombuffer(audio_bytes, dtype=dtype)
                        
                        # Convert to float32 and normalize if needed
                        if dtype != np.float32:
                            audio_float = audio_array.astype(np.float32) / divisor
                        else:
                            audio_float = audio_array
                        
                        # Ensure values are in valid range
                        audio_float = np.clip(audio_float, -1.0, 1.0)
                        
                        # Handle channel configuration
                        if self.channels == 1:
                            return audio_float
                        else:
                            # Assume interleaved format for multi-channel
                            if len(audio_float) % self.channels == 0:
                                return audio_float.reshape(-1, self.channels)
                            else:
                                # If not evenly divisible, just return as mono
                                return audio_float
                        
                    except Exception:
                        continue
                
                # If all formats fail, create silence
                logger.warning("All audio format conversions failed, generating silence")
                silence_length = len(audio_bytes) // 2  # Assume 16-bit
                return np.zeros(silence_length, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Critical failure in audio conversion: {e}")
            # Return minimal silence to prevent crashes
            return np.zeros(1024, dtype=np.float32)
    
    def _resample_audio(self, audio_data: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if source_sr == target_sr:
            return audio_data
        
        try:
            from scipy import signal
            # Calculate resampling ratio
            num_samples = int(len(audio_data) * target_sr / source_sr)
            
            if len(audio_data.shape) == 1:  # Mono
                return signal.resample(audio_data, num_samples)
            else:  # Multi-channel
                resampled_channels = []
                for channel in range(audio_data.shape[1]):
                    resampled_channel = signal.resample(audio_data[:, channel], num_samples)
                    resampled_channels.append(resampled_channel)
                return np.column_stack(resampled_channels)
                
        except ImportError:
            logger.warning("scipy not available for resampling, using simple decimation/interpolation")
            # Simple resampling fallback
            ratio = target_sr / source_sr
            if ratio < 1:  # Downsample
                step = int(1 / ratio)
                return audio_data[::step]
            else:  # Upsample
                repeat_factor = int(ratio)
                return np.repeat(audio_data, repeat_factor, axis=0)
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            return audio_data
    
    def write_stream_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Write audio chunk to stream for real-time playback.
        
        Args:
            audio_chunk: Audio chunk to write
            
        Returns:
            bool: True if chunk written successfully
        """
        if not self.stream:
            logger.error("No active audio stream")
            return False
        
        try:
            self.stream.write(audio_chunk)
            return True
        except Exception as e:
            logger.error(f"Failed to write stream chunk: {e}")
            return False
    
    def get_stream_info(self) -> Dict[str, Any]:
        """Get information about current audio stream."""
        if not self.stream or not self.current_device:
            return {"status": "no_stream"}
        
        return {
            "status": "active" if self.stream.active else "inactive",
            "device": self.current_device.name,
            "device_id": self.current_device.id,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "buffer_size": self.buffer_size,
            "latency": getattr(self.stream, 'latency', 'unknown')
        }
    
    def test_device(self, duration: float = 1.0) -> bool:
        """
        Test current device with a sine wave.
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            bool: True if test successful
        """
        if not self.current_device:
            logger.error("No device selected for test")
            return False
        
        try:
            # Generate test tone (440 Hz sine wave)
            samples = int(duration * self.sample_rate)
            t = np.linspace(0, duration, samples, False)
            sine_wave = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz, reduced amplitude
            
            # Create temporary stream for test
            with sd.OutputStream(
                device=self.current_device.id,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=np.float32
            ) as test_stream:
                test_stream.write(sine_wave)
            
            logger.info(f"Device test successful: {self.current_device.name}")
            return True
            
        except Exception as e:
            logger.error(f"Device test failed: {e}")
            return False
    
    def get_fallback_device(self) -> Optional[AudioDevice]:
        """Get fallback device when primary device is unavailable."""
        # Try to find a suitable fallback
        fallback_options = [
            # First: other virtual devices
            [d for d in self.available_devices if d.device_type == AudioDeviceType.VIRTUAL and d != self.current_device],
            # Second: regular output devices
            [d for d in self.available_devices if d.device_type == AudioDeviceType.OUTPUT],
            # Third: any available device
            [d for d in self.available_devices if d != self.current_device]
        ]
        
        for option_group in fallback_options:
            if option_group:
                return option_group[0]
        
        return None
    
    # ===== ENHANCED STREAMING FEATURES =====
    
    def start_continuous_streaming(self) -> bool:
        """
        Start continuous streaming mode for real-time audio.
        
        Returns:
            bool: True if streaming started successfully
        """
        if self.streaming_active:
            logger.warning("Continuous streaming already active")
            return True
        
        if not self.current_device:
            logger.error("No device selected for streaming")
            return False
        
        try:
            # Create enhanced stream with callback
            self.stream = sd.OutputStream(
                device=self.current_device.id,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                dtype=np.float32,
                callback=self._stream_callback,
                latency='low' if self.latency_mode == 'low' else 'high'
            )
            
            self.stream.start()
            self.streaming_active = True
            self.stream_stop_event.clear()
            
            # Start streaming thread
            self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
            self.streaming_thread.start()
            
            logger.info(f"Started continuous streaming: {self.current_device.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start continuous streaming: {e}")
            self._notify_error("stream_start_failed", str(e))
            return False
    
    def stop_continuous_streaming(self):
        """Stop continuous streaming mode."""
        if not self.streaming_active:
            return
        
        self.streaming_active = False
        self.stream_stop_event.set()
        
        # Wait for streaming thread to finish
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2.0)
        
        # Close stream
        self.close_stream()
        
        # Clear any remaining audio in queue
        self._clear_audio_queue()
        
        logger.info("Stopped continuous streaming")
    
    def queue_audio(self, audio_data: bytes, priority: bool = False) -> bool:
        """
        Queue audio data for continuous streaming.
        
        Args:
            audio_data: Audio data to queue
            priority: Whether to add to front of queue
            
        Returns:
            bool: True if audio queued successfully
        """
        if not self.streaming_active:
            logger.error("Streaming not active - call start_continuous_streaming() first")
            return False
        
        try:
            audio_chunk = self._bytes_to_numpy(audio_data)
            
            if priority:
                # Priority audio - try to add to front
                temp_queue = queue.Queue(maxsize=self.max_queue_size)
                temp_queue.put(audio_chunk)
                
                # Move existing items behind priority item
                while not self.audio_queue.empty():
                    try:
                        temp_queue.put(self.audio_queue.get_nowait())
                    except queue.Empty:
                        break
                
                self.audio_queue = temp_queue
            else:
                # Normal priority
                try:
                    self.audio_queue.put(audio_chunk, timeout=0.1)
                except queue.Full:
                    # Queue is full, remove oldest item and add new one
                    try:
                        self.audio_queue.get_nowait()  # Remove oldest
                        self.audio_queue.put(audio_chunk, timeout=0.1)
                        self.metrics.buffer_overruns += 1
                        logger.debug("Audio queue overflow, dropped oldest chunk")
                    except (queue.Empty, queue.Full):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue audio: {e}")
            return False
    
    def _stream_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Callback for continuous audio streaming."""
        if status:
            logger.warning(f"Stream status: {status}")
        
        try:
            # Get audio chunk from queue
            try:
                audio_chunk = self.audio_queue.get_nowait()
                
                # Ensure chunk is the right size
                if len(audio_chunk) == frames:
                    outdata[:] = audio_chunk.reshape(-1, self.channels)
                elif len(audio_chunk) < frames:
                    # Pad with zeros if chunk is too small
                    outdata[:len(audio_chunk)] = audio_chunk.reshape(-1, self.channels)
                    outdata[len(audio_chunk):] = 0
                else:
                    # Truncate if chunk is too large
                    outdata[:] = audio_chunk[:frames].reshape(-1, self.channels)
                
                # Update metrics
                with self._lock:
                    self.metrics.update_success(0.0)  # Callback latency measured separately
                
            except queue.Empty:
                # No audio available, output silence
                outdata.fill(0)
                with self._lock:
                    self.metrics.buffer_underruns += 1
                
        except Exception as e:
            logger.error(f"Stream callback error: {e}")
            outdata.fill(0)
            with self._lock:
                self.metrics.update_failure()
    
    def _streaming_worker(self):
        """Background worker for streaming management."""
        logger.debug("Streaming worker started")
        
        while self.streaming_active and not self.stream_stop_event.wait(0.1):
            try:
                # Monitor queue health
                queue_size = self.audio_queue.qsize()
                queue_health = (queue_size / self.max_queue_size) * 100
                
                # Notify callbacks about metrics
                if self.enable_monitoring and self.metrics_callbacks:
                    self._notify_metrics_update(queue_health)
                
                # Auto-recovery if stream fails
                if self.auto_recovery and self.stream and not self.stream.active:
                    logger.warning("Stream failed, attempting recovery")
                    self._attempt_stream_recovery()
                
            except Exception as e:
                logger.error(f"Streaming worker error: {e}")
        
        logger.debug("Streaming worker stopped")
    
    def _attempt_stream_recovery(self) -> bool:
        """Attempt to recover failed stream."""
        try:
            logger.info("Attempting stream recovery...")
            
            # Close failed stream
            self.close_stream()
            
            # Brief pause
            time.sleep(0.1)
            
            # Try to restart stream
            self.stream = sd.OutputStream(
                device=self.current_device.id,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                dtype=np.float32,
                callback=self._stream_callback,
                latency='low' if self.latency_mode == 'low' else 'high'
            )
            
            self.stream.start()
            logger.info("Stream recovery successful")
            return True
            
        except Exception as e:
            logger.error(f"Stream recovery failed: {e}")
            self._notify_error("stream_recovery_failed", str(e))
            return False
    
    def _clear_audio_queue(self):
        """Clear all audio from the queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def add_error_callback(self, callback: Callable):
        """Add callback for error notifications."""
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)
            logger.debug(f"Added error callback: {callback.__name__}")
    
    def add_metrics_callback(self, callback: Callable):
        """Add callback for metrics updates."""
        if callback not in self.metrics_callbacks:
            self.metrics_callbacks.append(callback)
            logger.debug(f"Added metrics callback: {callback.__name__}")
    
    def _notify_error(self, error_type: str, message: str):
        """Notify error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error_type, message)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def _notify_metrics_update(self, queue_health: float):
        """Notify metrics callbacks."""
        for callback in self.metrics_callbacks:
            try:
                callback(self.metrics, queue_health)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
    
    def get_stream_metrics(self) -> Dict[str, Any]:
        """Get comprehensive streaming metrics."""
        with self._lock:
            return {
                'streaming_active': self.streaming_active,
                'total_chunks': self.metrics.total_chunks,
                'successful_chunks': self.metrics.successful_chunks,
                'failed_chunks': self.metrics.failed_chunks,
                'success_rate': self.metrics.get_success_rate(),
                'average_latency': self.metrics.average_latency,
                'buffer_underruns': self.metrics.buffer_underruns,
                'buffer_overruns': self.metrics.buffer_overruns,
                'queue_size': self.audio_queue.qsize(),
                'queue_capacity': self.max_queue_size,
                'queue_health': (self.audio_queue.qsize() / self.max_queue_size) * 100,
                'last_update': self.metrics.last_update
            }
    
    def optimize_for_tts(self, tts_engine_info: Dict[str, Any]) -> bool:
        """
        Optimize streaming settings for specific TTS engine.
        
        Args:
            tts_engine_info: Information about the TTS engine
            
        Returns:
            bool: True if optimization applied successfully
        """
        try:
            engine_sample_rate = tts_engine_info.get('sample_rate', self.sample_rate)
            engine_channels = tts_engine_info.get('channels', self.channels)
            
            # Adjust settings for TTS engine
            if engine_sample_rate != self.sample_rate:
                logger.info(f"Adjusting sample rate for TTS: {self.sample_rate} -> {engine_sample_rate}")
                self.sample_rate = engine_sample_rate
            
            if engine_channels != self.channels:
                logger.info(f"Adjusting channels for TTS: {self.channels} -> {engine_channels}")
                self.channels = engine_channels
            
            # Restart stream if active with new settings
            if self.streaming_active:
                self.stop_continuous_streaming()
                return self.start_continuous_streaming()
            
            return True
            
        except Exception as e:
            logger.error(f"TTS optimization failed: {e}")
            return False
    
    def route_via_vb_cable(self, audio_data: bytes, optimize_latency: bool = True) -> AudioRoutingResult:
        """
        Route audio directly via VB-Cable interface with optimization.
        
        Args:
            audio_data: Audio data to route
            optimize_latency: Whether to apply latency optimizations
            
        Returns:
            AudioRoutingResult: Enhanced routing result
        """
        if not self.vb_cable_interface:
            return AudioRoutingResult(
                success=False,
                device_used="vb_cable_unavailable",
                latency=0.0,
                error_message="VB-Cable interface not available"
            )
        
        try:
            # Route via VB-Cable interface
            vb_result = self.vb_cable_interface.route_audio_direct(audio_data, optimize_latency)
            
            # Convert VB-Cable result to AudioRoutingResult
            result = AudioRoutingResult(
                success=vb_result.success,
                device_used=vb_result.device_used,
                latency=vb_result.latency_ms,
                bytes_processed=vb_result.bytes_transferred,
                buffer_health=100.0,  # VB-Cable doesn't use buffer health concept
                error_message=vb_result.error_message
            )
            
            logger.debug(f"VB-Cable routing: {vb_result.success}, latency: {vb_result.latency_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"VB-Cable routing error: {e}")
            return AudioRoutingResult(
                success=False,
                device_used="vb_cable_error",
                latency=0.0,
                error_message=f"VB-Cable routing failed: {e}"
            )
    
    def is_vb_cable_available(self) -> bool:
        """Check if VB-Cable is available and active."""
        if not self.vb_cable_interface:
            return False
        
        status = self.vb_cable_interface.get_status()
        return status['installation_status'] in ['installed', 'active']
    
    def get_vb_cable_status(self) -> Optional[Dict[str, Any]]:
        """Get detailed VB-Cable status information."""
        if not self.vb_cable_interface:
            return None
        
        return self.vb_cable_interface.get_status()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive virtual audio status."""
        virtual_devices = [d for d in self.available_devices if d.device_type == AudioDeviceType.VIRTUAL]
        
        status = {
            "current_device": self.current_device.name if self.current_device else None,
            "device_type": self.current_device.device_type.value if self.current_device else None,
            "device_id": self.current_device.id if self.current_device else None,
            "vb_cable_available": self.is_vb_cable_available(),
            "stream_active": self.stream.active if self.stream else False,
            "streaming_mode": "continuous" if self.streaming_active else "manual",
            "available_devices": len(self.available_devices),
            "virtual_devices": len(virtual_devices),
            "latency_mode": self.latency_mode,
            "buffer_size": self.buffer_size,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "stream_info": self.get_stream_info(),
            "stream_metrics": self.get_stream_metrics() if self.streaming_active else None,
            "virtual_device_details": [
                {
                    "name": device.name,
                    "id": device.id,
                    "channels": device.channels,
                    "sample_rate": device.sample_rate,
                    "is_available": device.is_available
                }
                for device in virtual_devices
            ],
            "vb_cable_integration": {
                "enabled": self.vb_cable_enabled,
                "available": self.vb_cable_interface is not None,
                "status": self.get_vb_cable_status() if self.vb_cable_interface else None
            }
        }
        
        return status
    
    def cleanup(self):
        """Enhanced cleanup with proper resource management."""
        try:
            # Stop streaming
            self.stop_continuous_streaming()
            
            # Close stream
            self.close_stream()
            
            # Clear callbacks
            self.error_callbacks.clear()
            self.metrics_callbacks.clear()
            
            # Clear audio queue
            self._clear_audio_queue()
            
            # Clean up VB-Cable interface
            if self.vb_cable_interface:
                self.vb_cable_interface.cleanup()
                self.vb_cable_interface = None
            
            logger.info("Enhanced VirtualAudioInterface cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during VirtualAudioInterface cleanup: {e}")
    
    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.cleanup()
        except:
            pass