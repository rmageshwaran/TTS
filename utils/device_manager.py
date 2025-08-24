"""
Device Manager Module

Handles audio device detection and management.
Provides device enumeration, status monitoring, and device information.
"""

import logging
import os
import sounddevice as sd
import threading
import time
import subprocess
import platform
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Audio device types."""
    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"
    VIRTUAL = "virtual"
    UNKNOWN = "unknown"


class VirtualAudioSoftware(Enum):
    """Known virtual audio software."""
    VB_CABLE = "VB-Audio Virtual Cable"
    VOICEMEETER = "VoiceMeeter"
    VAC = "Virtual Audio Cable"
    SOUNDFLOWER = "Soundflower"
    BLACKHOLE = "BlackHole"
    JACK = "JACK Audio"
    UNKNOWN = "Unknown Virtual Audio"


@dataclass
class DeviceInfo:
    """Enhanced audio device information."""
    id: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi: int
    is_default_input: bool = False
    is_default_output: bool = False
    # Enhanced fields
    device_type: DeviceType = DeviceType.UNKNOWN
    virtual_software: Optional[VirtualAudioSoftware] = None
    is_virtual: bool = False
    supported_sample_rates: List[int] = field(default_factory=list)
    last_tested: Optional[float] = None
    test_result: Optional[bool] = None
    
    def __post_init__(self):
        """Determine device type and virtual audio software."""
        # Determine device type
        if self.max_input_channels > 0 and self.max_output_channels > 0:
            self.device_type = DeviceType.BIDIRECTIONAL
        elif self.max_input_channels > 0:
            self.device_type = DeviceType.INPUT
        elif self.max_output_channels > 0:
            self.device_type = DeviceType.OUTPUT
        
        # Check for virtual audio devices
        name_lower = self.name.lower()
        virtual_keywords = {
            'vb-audio': VirtualAudioSoftware.VB_CABLE,
            'cable': VirtualAudioSoftware.VB_CABLE,
            'voicemeeter': VirtualAudioSoftware.VOICEMEETER,
            'vac': VirtualAudioSoftware.VAC,
            'virtual audio cable': VirtualAudioSoftware.VAC,
            'soundflower': VirtualAudioSoftware.SOUNDFLOWER,
            'blackhole': VirtualAudioSoftware.BLACKHOLE,
            'jack': VirtualAudioSoftware.JACK
        }
        
        for keyword, software in virtual_keywords.items():
            if keyword in name_lower:
                self.is_virtual = True
                self.virtual_software = software
                self.device_type = DeviceType.VIRTUAL
                break
        
        if self.is_virtual and self.virtual_software is None:
            self.virtual_software = VirtualAudioSoftware.UNKNOWN


class DeviceManager:
    """
    Enhanced audio device detection and management.
    
    Features:
    - Device enumeration and detection
    - Virtual audio device validation
    - Device capability analysis
    - Real-time status monitoring
    - Device preference management
    - Virtual audio software detection
    """
    
    def __init__(self, monitoring_enabled: bool = False):
        """
        Initialize Enhanced Device Manager.
        
        Args:
            monitoring_enabled: Enable real-time device monitoring
        """
        self.devices: List[DeviceInfo] = []
        self.default_input_device: Optional[DeviceInfo] = None
        self.default_output_device: Optional[DeviceInfo] = None
        
        # Enhanced features
        self.monitoring_enabled = monitoring_enabled
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_stop_event = threading.Event()
        self.device_change_callbacks: List[Callable] = []
        self.virtual_audio_installed: Dict[VirtualAudioSoftware, bool] = {}
        
        logger.info("Enhanced DeviceManager initialized")
        self.refresh_devices()
        self._detect_virtual_audio_software()
        
        if monitoring_enabled:
            self.start_monitoring()
    
    def refresh_devices(self):
        """Refresh the list of available devices."""
        try:
            # Get device information from sounddevice
            raw_devices = sd.query_devices()
            default_input = sd.default.device[0] if sd.default.device[0] is not None else -1
            default_output = sd.default.device[1] if sd.default.device[1] is not None else -1
            
            self.devices = []
            
            for i, device_info in enumerate(raw_devices):
                device = DeviceInfo(
                    id=i,
                    name=device_info['name'],
                    max_input_channels=device_info['max_input_channels'],
                    max_output_channels=device_info['max_output_channels'],
                    default_samplerate=device_info['default_samplerate'],
                    hostapi=device_info['hostapi'],
                    is_default_input=(i == default_input),
                    is_default_output=(i == default_output)
                )
                
                self.devices.append(device)
                
                # Set default devices
                if device.is_default_input:
                    self.default_input_device = device
                if device.is_default_output:
                    self.default_output_device = device
            
            logger.info(f"Refreshed devices: {len(self.devices)} found")
            
        except Exception as e:
            logger.error(f"Failed to refresh devices: {e}")
    
    def list_devices(self):
        """Print list of available devices to console."""
        print("\n=== Available Audio Devices ===")
        print(f"{'ID':<4} {'Type':<8} {'Name':<40} {'Channels':<12} {'Sample Rate':<12}")
        print("-" * 80)
        
        for device in self.devices:
            device_type = ""
            if device.max_input_channels > 0 and device.max_output_channels > 0:
                device_type = "I/O"
            elif device.max_input_channels > 0:
                device_type = "Input"
            elif device.max_output_channels > 0:
                device_type = "Output"
            
            channels = f"{device.max_input_channels}in/{device.max_output_channels}out"
            
            default_marker = ""
            if device.is_default_input:
                default_marker += " [Default In]"
            if device.is_default_output:
                default_marker += " [Default Out]"
            
            name_with_marker = f"{device.name}{default_marker}"
            
            print(f"{device.id:<4} {device_type:<8} {name_with_marker:<40} {channels:<12} {device.default_samplerate:<12.0f}")
        
        print("-" * 80)
        print(f"Total devices: {len(self.devices)}")
        
        # Show virtual audio devices
        virtual_devices = self.get_virtual_devices()
        if virtual_devices:
            print(f"\nVirtual Audio Devices Found: {len(virtual_devices)}")
            for device in virtual_devices:
                print(f"  - {device.name} (ID: {device.id})")
    
    def get_devices(self) -> List[DeviceInfo]:
        """Get list of all devices."""
        return self.devices.copy()
    
    def get_input_devices(self) -> List[DeviceInfo]:
        """Get list of input devices."""
        return [device for device in self.devices if device.max_input_channels > 0]
    
    def get_output_devices(self) -> List[DeviceInfo]:
        """Get list of output devices."""
        return [device for device in self.devices if device.max_output_channels > 0]
    
    def get_virtual_devices(self) -> List[DeviceInfo]:
        """Get list of virtual audio devices."""
        virtual_keywords = ['cable', 'virtual', 'vb-audio', 'voicemeeter', 'vac']
        virtual_devices = []
        
        for device in self.devices:
            device_name_lower = device.name.lower()
            if any(keyword in device_name_lower for keyword in virtual_keywords):
                virtual_devices.append(device)
        
        return virtual_devices
    
    def find_device_by_name(self, name: str, partial_match: bool = True) -> Optional[DeviceInfo]:
        """
        Find device by name.
        
        Args:
            name: Device name to search for
            partial_match: Whether to allow partial name matching
            
        Returns:
            DeviceInfo: Found device or None
        """
        name_lower = name.lower()
        
        for device in self.devices:
            device_name_lower = device.name.lower()
            
            if partial_match:
                if name_lower in device_name_lower:
                    return device
            else:
                if name_lower == device_name_lower:
                    return device
        
        return None
    
    def find_device_by_id(self, device_id: int) -> Optional[DeviceInfo]:
        """
        Find device by ID.
        
        Args:
            device_id: Device ID
            
        Returns:
            DeviceInfo: Found device or None
        """
        for device in self.devices:
            if device.id == device_id:
                return device
        return None
    
    def is_device_available(self, device_id: int) -> bool:
        """
        Check if device is available.
        
        Args:
            device_id: Device ID to check
            
        Returns:
            bool: True if device is available
        """
        try:
            # Try to query the specific device
            device_info = sd.query_devices(device_id)
            return device_info is not None
        except Exception:
            return False
    
    def test_device(self, device_id: int, duration: float = 1.0) -> bool:
        """
        Test device with a simple tone.
        
        Args:
            device_id: Device ID to test
            duration: Test duration in seconds
            
        Returns:
            bool: True if test successful
        """
        device = self.find_device_by_id(device_id)
        if not device:
            logger.error(f"Device {device_id} not found")
            return False
        
        if device.max_output_channels == 0:
            logger.error(f"Device {device.name} has no output channels")
            return False
        
        try:
            import numpy as np
            
            # Generate test tone (440 Hz sine wave)
            sample_rate = int(device.default_samplerate)
            samples = int(duration * sample_rate)
            t = np.linspace(0, duration, samples, False)
            sine_wave = 0.3 * np.sin(2 * np.pi * 440 * t)
            
            # Play test tone
            sd.play(sine_wave, samplerate=sample_rate, device=device_id)
            sd.wait()  # Wait for playback to complete
            
            logger.info(f"Device test successful: {device.name}")
            return True
            
        except Exception as e:
            logger.error(f"Device test failed for {device.name}: {e}")
            return False
    
    def get_device_capabilities(self, device_id: int) -> Dict[str, Any]:
        """
        Get detailed capabilities of a device.
        
        Args:
            device_id: Device ID
            
        Returns:
            Dict[str, Any]: Device capabilities
        """
        device = self.find_device_by_id(device_id)
        if not device:
            return {"error": "Device not found"}
        
        capabilities = {
            "id": device.id,
            "name": device.name,
            "input_channels": device.max_input_channels,
            "output_channels": device.max_output_channels,
            "default_samplerate": device.default_samplerate,
            "is_input_device": device.max_input_channels > 0,
            "is_output_device": device.max_output_channels > 0,
            "is_default_input": device.is_default_input,
            "is_default_output": device.is_default_output,
            "is_virtual": any(keyword in device.name.lower() for keyword in ['cable', 'virtual', 'vb-audio']),
            "available": self.is_device_available(device_id)
        }
        
        # Test supported sample rates
        test_rates = [8000, 16000, 22050, 44100, 48000, 96000]
        supported_rates = []
        
        for rate in test_rates:
            try:
                sd.check_output_settings(device=device_id, samplerate=rate)
                supported_rates.append(rate)
            except:
                pass
        
        capabilities["supported_sample_rates"] = supported_rates
        
        return capabilities
    
    # ===== ENHANCED FEATURES =====
    
    def _detect_virtual_audio_software(self):
        """Detect installed virtual audio software."""
        logger.info("Detecting virtual audio software...")
        
        # Check for common virtual audio software
        software_checks = {
            VirtualAudioSoftware.VB_CABLE: self._check_vb_cable,
            VirtualAudioSoftware.VOICEMEETER: self._check_voicemeeter,
            VirtualAudioSoftware.VAC: self._check_vac,
        }
        
        for software, check_func in software_checks.items():
            try:
                self.virtual_audio_installed[software] = check_func()
                if self.virtual_audio_installed[software]:
                    logger.info(f"Detected: {software.value}")
            except Exception as e:
                logger.warning(f"Error checking {software.value}: {e}")
                self.virtual_audio_installed[software] = False
    
    def _check_vb_cable(self) -> bool:
        """Check if VB-Audio Virtual Cable is installed."""
        # Check for VB-Cable devices
        for device in self.devices:
            if 'vb-audio' in device.name.lower() or 'cable' in device.name.lower():
                return True
        
        # Check Windows registry (Windows only)
        if platform.system() == "Windows":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                   r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall") as key:
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            subkey_name = winreg.EnumKey(key, i)
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                if "vb-audio" in display_name.lower() or "virtual cable" in display_name.lower():
                                    return True
                        except (FileNotFoundError, OSError):
                            continue
            except ImportError:
                pass
        
        return False
    
    def _check_voicemeeter(self) -> bool:
        """Check if VoiceMeeter is installed."""
        # Check for VoiceMeeter devices
        for device in self.devices:
            if 'voicemeeter' in device.name.lower():
                return True
        
        # Check for VoiceMeeter executable (Windows)
        if platform.system() == "Windows":
            common_paths = [
                r"C:\Program Files (x86)\VB\Voicemeeter\voicemeeter.exe",
                r"C:\Program Files\VB\Voicemeeter\voicemeeter.exe"
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return True
        
        return False
    
    def _check_vac(self) -> bool:
        """Check if Virtual Audio Cable is installed."""
        # Check for VAC devices
        for device in self.devices:
            name_lower = device.name.lower()
            if 'virtual audio cable' in name_lower or 'line 1 (virtual audio cable)' in name_lower:
                return True
        
        return False
    
    def validate_virtual_audio_setup(self) -> Dict[str, Any]:
        """
        Validate virtual audio setup for TTS functionality.
        
        Returns:
            Dict[str, Any]: Validation results
        """
        logger.info("Validating virtual audio setup...")
        
        validation = {
            "has_virtual_devices": False,
            "recommended_software": [],
            "virtual_devices": [],
            "suitable_for_tts": False,
            "issues": [],
            "recommendations": []
        }
        
        # Get virtual devices
        virtual_devices = self.get_virtual_devices()
        validation["virtual_devices"] = [
            {
                "name": device.name,
                "id": device.id,
                "software": device.virtual_software.value if device.virtual_software else "Unknown",
                "input_channels": device.max_input_channels,
                "output_channels": device.max_output_channels
            }
            for device in virtual_devices
        ]
        
        validation["has_virtual_devices"] = len(virtual_devices) > 0
        
        # Check for suitable TTS setup
        suitable_devices = []
        for device in virtual_devices:
            if device.max_input_channels > 0:  # Can act as virtual microphone
                suitable_devices.append(device)
        
        validation["suitable_for_tts"] = len(suitable_devices) > 0
        
        # Provide recommendations
        if not validation["has_virtual_devices"]:
            validation["issues"].append("No virtual audio devices detected")
            validation["recommendations"].extend([
                "Install VB-Audio Virtual Cable (free)",
                "Install VoiceMeeter (free)",
                "Install Virtual Audio Cable (commercial)"
            ])
            validation["recommended_software"] = [
                VirtualAudioSoftware.VB_CABLE.value,
                VirtualAudioSoftware.VOICEMEETER.value
            ]
        elif not validation["suitable_for_tts"]:
            validation["issues"].append("Virtual audio devices found but none suitable for TTS")
            validation["recommendations"].append("Ensure virtual device has input channels for microphone functionality")
        else:
            validation["recommendations"].append("Virtual audio setup looks good for TTS!")
        
        # Check installed software
        installed_software = [software.value for software, installed in self.virtual_audio_installed.items() if installed]
        if installed_software:
            validation["recommendations"].append(f"Detected software: {', '.join(installed_software)}")
        
        return validation
    
    def get_recommended_tts_device(self) -> Optional[DeviceInfo]:
        """
        Get the best virtual device for TTS functionality.
        
        Returns:
            DeviceInfo: Recommended device or None
        """
        virtual_devices = self.get_virtual_devices()
        
        # Prefer devices with input channels (can act as virtual microphone)
        input_devices = [device for device in virtual_devices if device.max_input_channels > 0]
        
        if input_devices:
            # Prefer VB-Cable if available
            vb_devices = [device for device in input_devices 
                         if device.virtual_software == VirtualAudioSoftware.VB_CABLE]
            if vb_devices:
                return vb_devices[0]
            
            # Return first available input device
            return input_devices[0]
        
        # If no input devices, return first virtual device
        if virtual_devices:
            return virtual_devices[0]
        
        return None
    
    def start_monitoring(self, interval: float = 5.0):
        """
        Start real-time device monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Device monitoring already running")
            return
        
        self.monitoring_enabled = True
        self.monitoring_stop_event.clear()
        
        def monitor_devices():
            logger.info(f"Device monitoring started (interval: {interval}s)")
            previous_device_count = len(self.devices)
            
            while not self.monitoring_stop_event.wait(interval):
                try:
                    # Store previous state
                    old_devices = self.devices.copy()
                    
                    # Refresh devices
                    self.refresh_devices()
                    
                    # Check for changes
                    current_device_count = len(self.devices)
                    if current_device_count != previous_device_count:
                        logger.info(f"Device count changed: {previous_device_count} -> {current_device_count}")
                        self._notify_device_change("device_count_changed", {
                            "old_count": previous_device_count,
                            "new_count": current_device_count
                        })
                        previous_device_count = current_device_count
                    
                    # Check for new virtual devices
                    new_virtual_devices = []
                    old_virtual_names = {device.name for device in old_devices if device.is_virtual}
                    
                    for device in self.devices:
                        if device.is_virtual and device.name not in old_virtual_names:
                            new_virtual_devices.append(device)
                    
                    if new_virtual_devices:
                        logger.info(f"New virtual devices detected: {[d.name for d in new_virtual_devices]}")
                        self._notify_device_change("virtual_devices_added", {
                            "devices": [d.name for d in new_virtual_devices]
                        })
                
                except Exception as e:
                    logger.error(f"Device monitoring error: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitor_devices, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop device monitoring."""
        if not self.monitoring_enabled:
            return
        
        self.monitoring_enabled = False
        self.monitoring_stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
            logger.info("Device monitoring stopped")
    
    def add_device_change_callback(self, callback: Callable):
        """
        Add callback for device changes.
        
        Args:
            callback: Function to call on device changes
        """
        if callback not in self.device_change_callbacks:
            self.device_change_callbacks.append(callback)
            logger.debug(f"Added device change callback: {callback.__name__}")
    
    def remove_device_change_callback(self, callback: Callable):
        """
        Remove device change callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self.device_change_callbacks:
            self.device_change_callbacks.remove(callback)
            logger.debug(f"Removed device change callback: {callback.__name__}")
    
    def _notify_device_change(self, change_type: str, data: Dict[str, Any]):
        """Notify callbacks of device changes."""
        for callback in self.device_change_callbacks:
            try:
                callback(change_type, data)
            except Exception as e:
                logger.error(f"Error in device change callback {callback.__name__}: {e}")
    
    def test_tts_compatibility(self, device_id: int) -> Dict[str, Any]:
        """
        Test device compatibility for TTS functionality.
        
        Args:
            device_id: Device ID to test
            
        Returns:
            Dict[str, Any]: Compatibility test results
        """
        device = self.find_device_by_id(device_id)
        if not device:
            return {"error": "Device not found"}
        
        logger.info(f"Testing TTS compatibility for: {device.name}")
        
        test_result = {
            "device_name": device.name,
            "device_id": device_id,
            "is_virtual": device.is_virtual,
            "virtual_software": device.virtual_software.value if device.virtual_software else None,
            "has_input": device.max_input_channels > 0,
            "has_output": device.max_output_channels > 0,
            "can_be_virtual_mic": False,
            "sample_rate_test": {},
            "audio_test": False,
            "recommended_for_tts": False,
            "issues": [],
            "recommendations": []
        }
        
        # Test virtual microphone capability
        if device.is_virtual and device.max_input_channels > 0:
            test_result["can_be_virtual_mic"] = True
            test_result["recommendations"].append("Can be used as virtual microphone")
        elif not device.is_virtual:
            test_result["issues"].append("Not a virtual device")
            test_result["recommendations"].append("Use virtual audio device for TTS")
        elif device.max_input_channels == 0:
            test_result["issues"].append("No input channels for virtual microphone")
        
        # Test sample rates
        common_tts_rates = [16000, 22050, 24000, 44100, 48000]
        supported_rates = []
        
        for rate in common_tts_rates:
            try:
                if device.max_output_channels > 0:
                    sd.check_output_settings(device=device_id, samplerate=rate)
                    supported_rates.append(rate)
            except:
                pass
        
        test_result["sample_rate_test"] = {
            "supported_rates": supported_rates,
            "supports_common_tts": len(supported_rates) > 0
        }
        
        # Audio playback test
        if device.max_output_channels > 0:
            test_result["audio_test"] = self.test_device(device_id, duration=0.5)
        
        # Overall recommendation
        if (test_result["can_be_virtual_mic"] and 
            test_result["sample_rate_test"]["supports_common_tts"] and
            (test_result["audio_test"] or device.max_output_channels == 0)):
            test_result["recommended_for_tts"] = True
            test_result["recommendations"].append("✅ Excellent for TTS virtual microphone")
        
        # Update device test info
        device.last_tested = time.time()
        device.test_result = test_result["recommended_for_tts"]
        
        return test_result
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive device manager status."""
        virtual_devices = self.get_virtual_devices()
        
        status = {
            "total_devices": len(self.devices),
            "input_devices": len(self.get_input_devices()),
            "output_devices": len(self.get_output_devices()),
            "virtual_devices": len(virtual_devices),
            "default_input": self.default_input_device.name if self.default_input_device else None,
            "default_output": self.default_output_device.name if self.default_output_device else None,
            "sounddevice_version": sd.__version__,
            
            # Enhanced status
            "monitoring_enabled": self.monitoring_enabled,
            "virtual_audio_software": {
                software.value: installed 
                for software, installed in self.virtual_audio_installed.items()
            },
            "recommended_tts_device": None,
            "tts_ready": False,
            "virtual_device_details": []
        }
        
        # Get recommended TTS device
        recommended_device = self.get_recommended_tts_device()
        if recommended_device:
            status["recommended_tts_device"] = {
                "name": recommended_device.name,
                "id": recommended_device.id,
                "software": recommended_device.virtual_software.value if recommended_device.virtual_software else "Unknown"
            }
            status["tts_ready"] = True
        
        # Virtual device details
        for device in virtual_devices:
            status["virtual_device_details"].append({
                "name": device.name,
                "id": device.id,
                "software": device.virtual_software.value if device.virtual_software else "Unknown",
                "input_channels": device.max_input_channels,
                "output_channels": device.max_output_channels,
                "can_be_virtual_mic": device.max_input_channels > 0,
                "last_tested": device.last_tested,
                "test_result": device.test_result
            })
        
        return status
    
    def __del__(self):
        """Cleanup when DeviceManager is destroyed."""
        try:
            self.stop_monitoring()
        except:
            pass