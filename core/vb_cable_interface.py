"""
VB-Cable Integration Interface
Enhanced VB-Audio Virtual Cable integration with direct routing optimization
"""

import logging
import platform
import subprocess
import time
import threading
import queue
import numpy as np
import sounddevice as sd
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class VBCableStatus(Enum):
    """VB-Cable installation and operational status."""
    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    ACTIVE = "active"
    ERROR = "error"
    UNKNOWN = "unknown"


class VBCableType(Enum):
    """VB-Cable product types."""
    VIRTUAL_CABLE = "virtual_cable"
    VOICEMEETER = "voicemeeter"
    VOICEMEETER_BANANA = "voicemeeter_banana"
    VOICEMEETER_POTATO = "voicemeeter_potato"
    UNKNOWN = "unknown"


@dataclass
class VBCableDevice:
    """VB-Cable device information."""
    name: str
    device_id: int
    device_type: str  # "input" or "output"
    channels: int
    sample_rate: int
    is_default: bool = False
    is_active: bool = False
    latency: float = 0.0
    cable_type: VBCableType = VBCableType.UNKNOWN


@dataclass
class VBCableRoutingResult:
    """Result of VB-Cable routing operation."""
    success: bool
    device_used: str
    latency_ms: float
    bytes_transferred: int = 0
    routing_time_ms: float = 0.0
    error_message: Optional[str] = None
    cable_type: VBCableType = VBCableType.UNKNOWN


@dataclass
class VBCableMetrics:
    """VB-Cable performance metrics."""
    total_routes: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    average_latency: float = 0.0
    total_bytes_transferred: int = 0
    last_routing_time: float = 0.0
    uptime_seconds: float = 0.0
    
    def update_success(self, latency: float, bytes_transferred: int):
        """Update metrics for successful routing."""
        self.total_routes += 1
        self.successful_routes += 1
        self.total_bytes_transferred += bytes_transferred
        
        # Update average latency
        if self.successful_routes == 1:
            self.average_latency = latency
        else:
            self.average_latency = ((self.average_latency * (self.successful_routes - 1)) + latency) / self.successful_routes
        
        self.last_routing_time = time.time()
    
    def update_failure(self):
        """Update metrics for failed routing."""
        self.total_routes += 1
        self.failed_routes += 1
    
    def get_success_rate(self) -> float:
        """Get routing success rate percentage."""
        if self.total_routes == 0:
            return 100.0
        return (self.successful_routes / self.total_routes) * 100.0


class VBCableInterface:
    """
    Enhanced VB-Audio Virtual Cable integration interface.
    
    Features:
    - Advanced VB-Cable detection and identification
    - Direct VB-Cable routing with optimization
    - Latency minimization and performance tuning
    - Real-time monitoring and metrics
    - Multi-VB-Cable product support
    - Error recovery and fallback mechanisms
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VB-Cable interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # VB-Cable configuration
        vb_config = config.get("vb_cable", {})
        self.auto_detect = vb_config.get("auto_detect", True)
        self.preferred_cable_type = VBCableType(vb_config.get("preferred_type", "virtual_cable"))
        self.enable_latency_optimization = vb_config.get("enable_latency_optimization", True)
        self.routing_buffer_size = vb_config.get("routing_buffer_size", 512)
        self.max_routing_attempts = vb_config.get("max_routing_attempts", 3)
        
        # Audio settings
        audio_config = config.get("audio", {})
        self.sample_rate = audio_config.get("sample_rate", 24000)
        self.channels = audio_config.get("channels", 1)
        
        # Internal state
        self.status = VBCableStatus.UNKNOWN
        self.detected_devices: List[VBCableDevice] = []
        self.active_device: Optional[VBCableDevice] = None
        self.metrics = VBCableMetrics()
        self.start_time = time.time()
        
        # Threading and monitoring
        self.monitoring_enabled = vb_config.get("enable_monitoring", True)
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Callbacks
        self.status_change_callbacks: List[Callable] = []
        self.routing_callbacks: List[Callable] = []
        
        logger.info("VB-Cable interface initialized")
        
        # Initialize VB-Cable detection and setup
        self._initialize_vb_cable()
    
    def _initialize_vb_cable(self):
        """Initialize VB-Cable detection and setup."""
        try:
            # Detect VB-Cable installation
            self._detect_vb_cable_installation()
            
            # Detect available devices
            self._detect_vb_cable_devices()
            
            # Auto-select best device
            if self.auto_detect and self.detected_devices:
                self._auto_select_device()
            
            # Start monitoring if enabled
            if self.monitoring_enabled:
                self._start_monitoring()
            
            logger.info(f"VB-Cable initialization complete: {self.status.value}")
            
        except Exception as e:
            logger.error(f"VB-Cable initialization failed: {e}")
            self.status = VBCableStatus.ERROR
    
    def _detect_vb_cable_installation(self):
        """Detect VB-Cable installation status."""
        try:
            installation_found = False
            detected_products = []
            
            # Check for VB-Cable devices in system audio devices
            devices = sd.query_devices()
            vb_cable_devices = []
            
            for i, device_info in enumerate(devices):
                device_name = device_info['name'].lower()
                
                # Check for VB-Cable patterns
                if any(pattern in device_name for pattern in ['vb-audio', 'cable', 'voicemeeter']):
                    vb_cable_devices.append((i, device_info))
                    installation_found = True
                    
                    # Identify product type
                    if 'voicemeeter' in device_name:
                        if 'potato' in device_name:
                            detected_products.append(VBCableType.VOICEMEETER_POTATO)
                        elif 'banana' in device_name:
                            detected_products.append(VBCableType.VOICEMEETER_BANANA)
                        else:
                            detected_products.append(VBCableType.VOICEMEETER)
                    elif 'cable' in device_name or 'vb-audio' in device_name:
                        detected_products.append(VBCableType.VIRTUAL_CABLE)
            
            # Windows registry check for additional verification
            if platform.system() == "Windows":
                try:
                    import winreg
                    registry_products = self._check_windows_registry()
                    detected_products.extend(registry_products)
                    if registry_products:
                        installation_found = True
                except ImportError:
                    logger.warning("Windows registry check not available")
            
            # Update status
            if installation_found:
                if vb_cable_devices:
                    self.status = VBCableStatus.ACTIVE
                else:
                    self.status = VBCableStatus.INSTALLED
                
                logger.info(f"VB-Cable detected: {len(detected_products)} products, {len(vb_cable_devices)} devices")
            else:
                self.status = VBCableStatus.NOT_INSTALLED
                logger.warning("VB-Cable not detected")
            
        except Exception as e:
            logger.error(f"VB-Cable detection failed: {e}")
            self.status = VBCableStatus.ERROR
    
    def _check_windows_registry(self) -> List[VBCableType]:
        """Check Windows registry for VB-Cable installations."""
        detected_products = []
        
        try:
            import winreg
            
            # Check uninstall registry
            uninstall_key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
            
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, uninstall_key) as key:
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        with winreg.OpenKey(key, subkey_name) as subkey:
                            display_name = winreg.QueryValueEx(subkey, "DisplayName")[0].lower()
                            
                            if "vb-audio" in display_name or "virtual cable" in display_name:
                                detected_products.append(VBCableType.VIRTUAL_CABLE)
                            elif "voicemeeter" in display_name:
                                if "potato" in display_name:
                                    detected_products.append(VBCableType.VOICEMEETER_POTATO)
                                elif "banana" in display_name:
                                    detected_products.append(VBCableType.VOICEMEETER_BANANA)
                                else:
                                    detected_products.append(VBCableType.VOICEMEETER)
                    except (FileNotFoundError, OSError):
                        continue
        
        except Exception as e:
            logger.warning(f"Registry check failed: {e}")
        
        return detected_products
    
    def _detect_vb_cable_devices(self):
        """Detect and catalog VB-Cable devices."""
        try:
            devices = sd.query_devices()
            self.detected_devices = []
            
            for i, device_info in enumerate(devices):
                device_name = device_info['name']
                device_name_lower = device_name.lower()
                
                # Check if this is a VB-Cable device
                if any(pattern in device_name_lower for pattern in ['vb-audio', 'cable', 'voicemeeter']):
                    
                    # Determine cable type
                    cable_type = VBCableType.UNKNOWN
                    if 'voicemeeter' in device_name_lower:
                        if 'potato' in device_name_lower:
                            cable_type = VBCableType.VOICEMEETER_POTATO
                        elif 'banana' in device_name_lower:
                            cable_type = VBCableType.VOICEMEETER_BANANA
                        else:
                            cable_type = VBCableType.VOICEMEETER
                    elif 'cable' in device_name_lower or 'vb-audio' in device_name_lower:
                        cable_type = VBCableType.VIRTUAL_CABLE
                    
                    # Determine device type (input/output)
                    device_type = "unknown"
                    if device_info['max_input_channels'] > 0 and device_info['max_output_channels'] > 0:
                        device_type = "bidirectional"
                    elif device_info['max_input_channels'] > 0:
                        device_type = "input"
                    elif device_info['max_output_channels'] > 0:
                        device_type = "output"
                    
                    # Create VBCableDevice
                    vb_device = VBCableDevice(
                        name=device_name,
                        device_id=i,
                        device_type=device_type,
                        channels=max(device_info['max_input_channels'], device_info['max_output_channels']),
                        sample_rate=int(device_info['default_samplerate']),
                        cable_type=cable_type
                    )
                    
                    # Test device availability
                    vb_device.is_active = self._test_device_availability(vb_device)
                    
                    self.detected_devices.append(vb_device)
            
            logger.info(f"Detected {len(self.detected_devices)} VB-Cable devices")
            
        except Exception as e:
            logger.error(f"VB-Cable device detection failed: {e}")
    
    def _test_device_availability(self, device: VBCableDevice) -> bool:
        """Test if a VB-Cable device is available and responsive."""
        try:
            # For VB-Cable devices, if they're detected in the system, they're usually available
            # The previous test was too strict and causing false negatives
            
            # Basic check: ensure device has output channels for our use case
            if device.device_type in ["output", "bidirectional"] and device.channels > 0:
                # Try a minimal device query to verify it's accessible
                devices = sd.query_devices()
                if device.device_id < len(devices):
                    device_info = devices[device.device_id]
                    # If the device still exists and has output channels, consider it active
                    return device_info['max_output_channels'] > 0
            
            return False
        except Exception as e:
            logger.debug(f"Device availability test failed for {device.name}: {e}")
            # If we can't test it, but it's a known VB-Cable device, assume it's available
            # This is more permissive but prevents false negatives
            return device.device_type in ["output", "bidirectional"]
    
    def _auto_select_device(self):
        """Automatically select the best VB-Cable device."""
        if not self.detected_devices:
            return
        
        # Priority order: active output devices of preferred type
        candidates = []
        
        # First, filter by preferred type and active status
        for device in self.detected_devices:
            if device.is_active and device.device_type in ["output", "bidirectional"]:
                if device.cable_type == self.preferred_cable_type:
                    candidates.insert(0, device)  # High priority
                else:
                    candidates.append(device)  # Lower priority
        
        # If no candidates, use any active device
        if not candidates:
            candidates = [d for d in self.detected_devices if d.is_active]
        
        # Select the first candidate
        if candidates:
            self.active_device = candidates[0]
            logger.info(f"Auto-selected VB-Cable device: {self.active_device.name}")
        else:
            # Fallback: try to select any detected VB-Cable device, even if not marked as active
            if self.detected_devices:
                # Prefer output devices
                output_devices = [d for d in self.detected_devices if d.device_type in ["output", "bidirectional"]]
                if output_devices:
                    self.active_device = output_devices[0]
                    logger.info(f"Fallback selected VB-Cable device: {self.active_device.name}")
                else:
                    self.active_device = self.detected_devices[0]
                    logger.info(f"Fallback selected any VB-Cable device: {self.active_device.name}")
            else:
                logger.warning("No suitable VB-Cable device found")
    
    def select_device(self, device_name: str = None, device_id: int = None) -> bool:
        """
        Manually select a VB-Cable device.
        
        Args:
            device_name: Name of device to select
            device_id: ID of device to select
            
        Returns:
            bool: True if device selected successfully
        """
        target_device = None
        
        for device in self.detected_devices:
            if device_id is not None and device.device_id == device_id:
                target_device = device
                break
            elif device_name and device_name.lower() in device.name.lower():
                target_device = device
                break
        
        if target_device:
            self.active_device = target_device
            logger.info(f"Selected VB-Cable device: {target_device.name}")
            self._notify_status_change("device_selected", target_device)
            return True
        else:
            logger.error(f"VB-Cable device not found: name={device_name}, id={device_id}")
            return False
    
    def force_select_vb_audio_point(self) -> bool:
        """
        Force selection of VB-Audio Point device as a fallback.
        
        Returns:
            bool: True if device found and selected
        """
        try:
            # Look for VB-Audio Point specifically
            devices = sd.query_devices()
            
            for i, device_info in enumerate(devices):
                device_name = device_info['name']
                if 'vb-audio point' in device_name.lower() and device_info['max_output_channels'] > 0:
                    # Create a VBCableDevice for this
                    vb_device = VBCableDevice(
                        name=device_name,
                        device_id=i,
                        device_type="output",
                        channels=device_info['max_output_channels'],
                        sample_rate=int(device_info['default_samplerate']),
                        is_active=True,
                        cable_type=VBCableType.VIRTUAL_CABLE
                    )
                    
                    self.active_device = vb_device
                    
                    # Add to detected devices if not already there
                    if not any(d.device_id == i for d in self.detected_devices):
                        self.detected_devices.append(vb_device)
                    
                    logger.info(f"Force-selected VB-Audio Point: {device_name}")
                    return True
            
            logger.warning("VB-Audio Point device not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to force-select VB-Audio Point: {e}")
            return False
    
    def route_audio_direct(self, audio_data: bytes, optimize_latency: bool = True) -> VBCableRoutingResult:
        """
        Route audio directly to VB-Cable with optimization.
        
        Args:
            audio_data: Audio data to route
            optimize_latency: Whether to apply latency optimizations
            
        Returns:
            VBCableRoutingResult: Routing result with metrics
        """
        if not self.active_device:
            # Try to force-select VB-Audio Point as a last resort
            if not self.force_select_vb_audio_point():
                return VBCableRoutingResult(
                    success=False,
                    device_used="none",
                    latency_ms=0.0,
                    error_message="No active VB-Cable device selected"
                )
        
        start_time = time.time()
        
        try:
            # Convert audio data to numpy array
            audio_array = self._convert_audio_for_vb_cable(audio_data)
            
            # Apply latency optimizations if enabled
            if optimize_latency and self.enable_latency_optimization:
                audio_array = self._apply_latency_optimizations(audio_array)
            
            # Create optimized stream for VB-Cable
            stream_params = self._get_optimized_stream_params()
            
            with sd.OutputStream(
                device=self.active_device.device_id,
                channels=stream_params['channels'],
                samplerate=stream_params['sample_rate'],
                blocksize=stream_params['buffer_size'],
                dtype=np.float32,
                latency=stream_params['latency']
            ) as stream:
                
                # Write audio to VB-Cable
                stream.write(audio_array)
                
                # Calculate metrics
                routing_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Update metrics
                with self._lock:
                    self.metrics.update_success(routing_time, len(audio_data))
                
                result = VBCableRoutingResult(
                    success=True,
                    device_used=self.active_device.name,
                    latency_ms=routing_time,
                    bytes_transferred=len(audio_data),
                    routing_time_ms=routing_time,
                    cable_type=self.active_device.cable_type
                )
                
                # Notify callbacks
                self._notify_routing_success(result)
                
                return result
        
        except Exception as e:
            # Update failure metrics
            with self._lock:
                self.metrics.update_failure()
            
            error_msg = f"VB-Cable routing failed: {e}"
            logger.error(error_msg)
            
            result = VBCableRoutingResult(
                success=False,
                device_used=self.active_device.name,
                latency_ms=0.0,
                error_message=error_msg,
                cable_type=self.active_device.cable_type
            )
            
            # Notify callbacks
            self._notify_routing_failure(result)
            
            return result
    
    def _convert_audio_for_vb_cable(self, audio_data: bytes) -> np.ndarray:
        """Convert audio data for optimal VB-Cable compatibility."""
        try:
            # Use the enhanced conversion from VirtualAudioInterface
            # This is a simplified version for VB-Cable specific optimization
            
            # Try different formats
            for dtype, divisor in [(np.int16, 32768.0), (np.int32, 2147483648.0), (np.float32, 1.0)]:
                try:
                    audio_array = np.frombuffer(audio_data, dtype=dtype)
                    
                    # Convert to float32 and normalize
                    if dtype != np.float32:
                        audio_float = audio_array.astype(np.float32) / divisor
                    else:
                        audio_float = audio_array
                    
                    # Ensure valid range
                    audio_float = np.clip(audio_float, -1.0, 1.0)
                    
                    # Handle channel configuration for VB-Cable
                    if self.active_device.channels == 1:
                        return audio_float
                    else:
                        # Convert mono to stereo by duplicating
                        if len(audio_float.shape) == 1:
                            return np.column_stack([audio_float, audio_float])
                        else:
                            return audio_float
                
                except Exception:
                    continue
            
            # If all conversions fail, return silence
            logger.warning("Audio conversion failed, using silence")
            return np.zeros(1024, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return np.zeros(1024, dtype=np.float32)
    
    def _apply_latency_optimizations(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply VB-Cable specific latency optimizations."""
        try:
            # Pre-emphasis filter for better VB-Cable transmission
            if len(audio_array) > 1:
                pre_emphasis = 0.97
                emphasized = np.append(audio_array[0], audio_array[1:] - pre_emphasis * audio_array[:-1])
                return emphasized
            return audio_array
        except Exception as e:
            logger.warning(f"Latency optimization failed: {e}")
            return audio_array
    
    def _get_optimized_stream_params(self) -> Dict[str, Any]:
        """Get optimized stream parameters for VB-Cable."""
        # Base parameters
        params = {
            'channels': min(self.channels, self.active_device.channels),
            'sample_rate': self.sample_rate,
            'buffer_size': self.routing_buffer_size,
            'latency': 'low'
        }
        
        # VB-Cable specific optimizations
        if self.active_device.cable_type == VBCableType.VIRTUAL_CABLE:
            # Virtual Cable works best with smaller buffers
            params['buffer_size'] = min(512, self.routing_buffer_size)
            params['latency'] = 'low'
        elif self.active_device.cable_type in [VBCableType.VOICEMEETER, VBCableType.VOICEMEETER_BANANA, VBCableType.VOICEMEETER_POTATO]:
            # VoiceMeeter can handle larger buffers
            params['buffer_size'] = min(1024, self.routing_buffer_size)
            params['latency'] = 'high'
        
        return params
    
    def _start_monitoring(self):
        """Start VB-Cable monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.monitoring_stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        logger.debug("VB-Cable monitoring started")
    
    def _stop_monitoring(self):
        """Stop VB-Cable monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_stop_event.set()
            self.monitoring_thread.join(timeout=2.0)
        logger.debug("VB-Cable monitoring stopped")
    
    def _monitoring_worker(self):
        """Background monitoring worker."""
        logger.debug("VB-Cable monitoring worker started")
        
        while not self.monitoring_stop_event.wait(5.0):  # Check every 5 seconds
            try:
                # Update uptime
                with self._lock:
                    self.metrics.uptime_seconds = time.time() - self.start_time
                
                # Check device status
                if self.active_device:
                    is_active = self._test_device_availability(self.active_device)
                    if is_active != self.active_device.is_active:
                        self.active_device.is_active = is_active
                        self._notify_status_change("device_status_changed", self.active_device)
                
                # Re-detect devices periodically (every minute)
                if int(self.metrics.uptime_seconds) % 60 == 0:
                    old_count = len(self.detected_devices)
                    self._detect_vb_cable_devices()
                    new_count = len(self.detected_devices)
                    
                    if new_count != old_count:
                        self._notify_status_change("devices_changed", {
                            'old_count': old_count,
                            'new_count': new_count
                        })
                
            except Exception as e:
                logger.warning(f"VB-Cable monitoring error: {e}")
        
        logger.debug("VB-Cable monitoring worker stopped")
    
    def add_status_change_callback(self, callback: Callable):
        """Add callback for status change notifications."""
        if callback not in self.status_change_callbacks:
            self.status_change_callbacks.append(callback)
    
    def add_routing_callback(self, callback: Callable):
        """Add callback for routing event notifications."""
        if callback not in self.routing_callbacks:
            self.routing_callbacks.append(callback)
    
    def _notify_status_change(self, event_type: str, data: Any):
        """Notify status change callbacks."""
        for callback in self.status_change_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Status change callback error: {e}")
    
    def _notify_routing_success(self, result: VBCableRoutingResult):
        """Notify routing success callbacks."""
        for callback in self.routing_callbacks:
            try:
                callback("routing_success", result)
            except Exception as e:
                logger.error(f"Routing callback error: {e}")
    
    def _notify_routing_failure(self, result: VBCableRoutingResult):
        """Notify routing failure callbacks."""
        for callback in self.routing_callbacks:
            try:
                callback("routing_failure", result)
            except Exception as e:
                logger.error(f"Routing callback error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive VB-Cable status."""
        with self._lock:
            return {
                'installation_status': self.status.value,
                'active_device': {
                    'name': self.active_device.name,
                    'id': self.active_device.device_id,
                    'type': self.active_device.device_type,
                    'cable_type': self.active_device.cable_type.value,
                    'channels': self.active_device.channels,
                    'sample_rate': self.active_device.sample_rate,
                    'is_active': self.active_device.is_active
                } if self.active_device else None,
                'detected_devices': [
                    {
                        'name': device.name,
                        'id': device.device_id,
                        'type': device.device_type,
                        'cable_type': device.cable_type.value,
                        'is_active': device.is_active
                    }
                    for device in self.detected_devices
                ],
                'metrics': {
                    'total_routes': self.metrics.total_routes,
                    'success_rate': self.metrics.get_success_rate(),
                    'average_latency_ms': self.metrics.average_latency,
                    'total_bytes_transferred': self.metrics.total_bytes_transferred,
                    'uptime_seconds': self.metrics.uptime_seconds
                },
                'configuration': {
                    'latency_optimization': self.enable_latency_optimization,
                    'buffer_size': self.routing_buffer_size,
                    'preferred_type': self.preferred_cable_type.value,
                    'monitoring_enabled': self.monitoring_enabled
                }
            }
    
    def get_metrics(self) -> VBCableMetrics:
        """Get current VB-Cable metrics."""
        with self._lock:
            return self.metrics
    
    def reset_metrics(self):
        """Reset VB-Cable metrics."""
        with self._lock:
            self.metrics = VBCableMetrics()
            self.start_time = time.time()
    
    def cleanup(self):
        """Clean up VB-Cable interface resources."""
        try:
            self._stop_monitoring()
            self.status_change_callbacks.clear()
            self.routing_callbacks.clear()
            logger.info("VB-Cable interface cleanup completed")
        except Exception as e:
            logger.error(f"VB-Cable cleanup error: {e}")
    
    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.cleanup()
        except:
            pass