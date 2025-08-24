"""
TTS Engine Manager Module

Manages TTS plugins and coordinates speech generation.
Handles plugin loading, fallback mechanisms, and context management.
"""

import logging
import importlib
import time
import threading
import statistics
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class EngineStatus(Enum):
    """TTS Engine status enumeration."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


@dataclass
class EngineMetrics:
    """Performance metrics for TTS engines."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    last_used: Optional[float] = None
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0
    last_error: Optional[str] = None
    
    def update_success(self, processing_time: float):
        """Update metrics for successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.total_requests
        self.last_used = time.time()
        
        # Keep only recent response times (last 100)
        self.response_times.append(processing_time)
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def update_failure(self, error_message: str):
        """Update metrics for failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.error_count += 1
        self.last_error = error_message
    
    def get_success_rate(self) -> float:
        """Get success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    def get_median_response_time(self) -> float:
        """Get median response time."""
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)


@dataclass
class TTSResult:
    """Result of TTS generation."""
    audio_data: bytes
    sample_rate: int
    duration: float
    engine_used: str
    speaker_id: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


class BaseTTSPlugin(ABC):
    """Abstract base class for TTS plugins."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the TTS engine."""
        pass
    
    @abstractmethod
    def generate_speech(self, text: str, speaker_id: Optional[int] = None, 
                       context: Optional[List] = None) -> bytes:
        """Generate speech from text."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine is available."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get the sample rate of generated audio."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass


class TTSEngine:
    """
    Enhanced TTS Engine Manager that coordinates multiple TTS plugins.
    
    Features:
    - Plugin management system with auto-discovery
    - Model loading and caching
    - Fallback engine support with load balancing
    - Context management for conversational TTS
    - Performance monitoring and health checks
    - Engine selection optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Enhanced TTS Engine Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.plugins: Dict[str, BaseTTSPlugin] = {}
        self.current_engine = None
        self.fallback_engines = []
        self.context_history = []
        self.max_context_length = config.get("tts", {}).get("context_length", 4)
        
        # Enhanced features
        self.engine_metrics: Dict[str, EngineMetrics] = {}
        self.engine_status: Dict[str, EngineStatus] = {}
        self.health_check_interval = config.get("tts", {}).get("health_check_interval", 30.0)
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.health_monitor_stop_event = threading.Event()
        self.monitoring_enabled = config.get("tts", {}).get("monitoring_enabled", True)
        self.load_balancing_enabled = config.get("tts", {}).get("load_balancing", True)
        self._lock = threading.Lock()
        
        # Engine preferences
        self.default_engine = config.get("tts", {}).get("default_engine", "sesame_csm")
        self.fallback_engine = config.get("tts", {}).get("fallback_engine", "pyttsx3")
        self.engine_priority = config.get("tts", {}).get("engine_priority", [])
        
        # Callbacks
        self.engine_change_callbacks: List[Callable] = []
        
        logger.info("Enhanced TTSEngine initialized")
        
        # Load default plugins
        self._load_default_plugins()
        
        # Start health monitoring
        if self.monitoring_enabled:
            self.start_health_monitoring()
    
    def _load_default_plugins(self):
        """Load default TTS plugins."""
        plugin_configs = {
            "demo_tts": "plugins.tts.demo_tts_plugin",
            "sesame_csm": "plugins.tts.sesame_csm_plugin",
            "pyttsx3": "plugins.tts.fallback_tts_plugin",
            "sapi": "plugins.tts.sapi_plugin"
        }
        
        for plugin_name, module_path in plugin_configs.items():
            try:
                self.load_plugin(plugin_name, module_path)
            except Exception as e:
                logger.warning(f"Failed to load plugin {plugin_name}: {e}")
    
    def load_plugin(self, plugin_name: str, module_path: Optional[str] = None) -> bool:
        """
        Load a TTS plugin.
        
        Args:
            plugin_name: Name of the plugin
            module_path: Python module path for the plugin
            
        Returns:
            bool: True if plugin loaded successfully
        """
        try:
            if module_path:
                # Import the plugin module
                module = importlib.import_module(module_path)
                
                # Try multiple naming conventions for plugin classes
                possible_names = [
                    f"{plugin_name.title().replace('_', '')}Plugin",  # DemoTtsPlugin
                    f"{''.join(word.title() for word in plugin_name.split('_'))}Plugin",  # DemoTtsPlugin  
                    f"{''.join(word.upper() for word in plugin_name.split('_'))}Plugin",  # DEMOTTSPlugin
                    f"Demo{'TTS' if 'tts' in plugin_name else ''}Plugin",  # DemoTTSPlugin (special case)
                ]
                
                plugin_class = None
                for name in possible_names:
                    try:
                        plugin_class = getattr(module, name)
                        break
                    except AttributeError:
                        continue
                
                if plugin_class is None:
                    # Try to find any class that inherits from BaseTTSPlugin
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            hasattr(attr, '__bases__') and 
                            any('BaseTTSPlugin' in str(base) for base in attr.__bases__)):
                            plugin_class = attr
                            break
                
                if plugin_class is None:
                    raise AttributeError(f"No suitable plugin class found in {module_path}")
            else:
                # Try to import from default location
                module_path = f"plugins.tts.{plugin_name}_plugin"
                module = importlib.import_module(module_path)
                plugin_class = getattr(module, f"{plugin_name.title().replace('_', '')}Plugin")
            
            # Create plugin instance
            plugin_instance = plugin_class(self.config)
            
            # Initialize plugin
            plugin_config = self.config.get("tts", {}).get(plugin_name, {})
            if plugin_instance.initialize(plugin_config):
                self.plugins[plugin_name] = plugin_instance
                
                # Initialize metrics and status
                self.engine_metrics[plugin_name] = EngineMetrics()
                self.engine_status[plugin_name] = EngineStatus.AVAILABLE
                
                logger.info(f"Successfully loaded TTS plugin: {plugin_name}")
                
                # Set as current engine if it's the default
                if plugin_name == self.default_engine and self.current_engine is None:
                    self.current_engine = plugin_name
                
                # Notify callbacks
                self._notify_engine_change("plugin_loaded", {"name": plugin_name})
                
                return True
            else:
                logger.error(f"Failed to initialize plugin: {plugin_name}")
                self.engine_status[plugin_name] = EngineStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    def load_plugin_instance(self, plugin_instance: BaseTTSPlugin) -> bool:
        """
        Load a TTS plugin instance directly.
        
        Args:
            plugin_instance: Instance of a TTS plugin
            
        Returns:
            bool: True if plugin loaded successfully
        """
        try:
            plugin_name = plugin_instance.get_name()
            
            # Check if plugin is available
            if not plugin_instance.is_available():
                logger.error(f"Plugin {plugin_name} is not available")
                return False
            
            # Store plugin instance
            self.plugins[plugin_name] = plugin_instance
            
            # Initialize metrics and status
            self.engine_metrics[plugin_name] = EngineMetrics()
            self.engine_status[plugin_name] = EngineStatus.AVAILABLE
            
            logger.info(f"Successfully loaded TTS plugin instance: {plugin_name}")
            
            # Set as current engine if it's the default or no engine is set
            if plugin_name == self.default_engine or self.current_engine is None:
                self.current_engine = plugin_name
                logger.info(f"Set {plugin_name} as current engine")
            
            # Notify callbacks
            self._notify_engine_change(plugin_name, EngineStatus.AVAILABLE)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugin instance: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a TTS plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            bool: True if plugin unloaded successfully
        """
        if plugin_name in self.plugins:
            try:
                # Cleanup plugin resources
                self.plugins[plugin_name].cleanup()
                del self.plugins[plugin_name]
                
                # Update current engine if needed
                if self.current_engine == plugin_name:
                    self.current_engine = self._get_next_available_engine()
                
                logger.info(f"Unloaded TTS plugin: {plugin_name}")
                return True
            except Exception as e:
                logger.error(f"Error unloading plugin {plugin_name}: {e}")
                return False
        else:
            logger.warning(f"Plugin {plugin_name} not found")
            return False
    
    def _get_next_available_engine(self) -> Optional[str]:
        """Get the next available engine."""
        # Try default engine first
        if self.default_engine in self.plugins and self.plugins[self.default_engine].is_available():
            return self.default_engine
        
        # Try fallback engine
        if self.fallback_engine in self.plugins and self.plugins[self.fallback_engine].is_available():
            return self.fallback_engine
        
        # Try any available engine
        for name, plugin in self.plugins.items():
            if plugin.is_available():
                return name
        
        return None
    
    def get_active_engine(self) -> Optional[str]:
        """Get the currently active engine."""
        if self.current_engine and self.current_engine in self.plugins:
            if self.plugins[self.current_engine].is_available():
                return self.current_engine
        
        # Current engine not available, find alternative
        self.current_engine = self._get_next_available_engine()
        return self.current_engine
    
    def set_engine(self, engine_name: str) -> bool:
        """
        Set the active TTS engine.
        
        Args:
            engine_name: Name of the engine to activate
            
        Returns:
            bool: True if engine set successfully
        """
        if engine_name in self.plugins and self.plugins[engine_name].is_available():
            self.current_engine = engine_name
            logger.info(f"Switched to TTS engine: {engine_name}")
            return True
        else:
            logger.error(f"Engine {engine_name} not available")
            return False
    
    def list_engines(self) -> List[str]:
        """Get list of available engines."""
        return [name for name, plugin in self.plugins.items() if plugin.is_available()]
    
    def add_context(self, text: str, speaker_id: int, audio_data: Optional[bytes] = None):
        """
        Add context for conversational TTS.
        
        Args:
            text: Text content
            speaker_id: Speaker identifier
            audio_data: Optional audio data for the text
        """
        context_item = {
            "text": text,
            "speaker_id": speaker_id,
            "audio": audio_data,
            "timestamp": None  # Could add timestamp if needed
        }
        
        self.context_history.append(context_item)
        
        # Limit context history length
        if len(self.context_history) > self.max_context_length:
            self.context_history.pop(0)
        
        logger.debug(f"Added context: speaker {speaker_id}, {len(text)} chars")
    
    def get_context(self, max_items: Optional[int] = None) -> List[Dict]:
        """
        Get current conversation context.
        
        Args:
            max_items: Maximum number of context items to return
            
        Returns:
            List[Dict]: Context history
        """
        if max_items:
            return self.context_history[-max_items:]
        return self.context_history.copy()
    
    def clear_context(self):
        """Clear conversation context."""
        self.context_history.clear()
        logger.debug("Cleared conversation context")
    
    # ===== ENHANCED FEATURES =====
    
    def start_health_monitoring(self):
        """Start health monitoring for all engines."""
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            logger.warning("Health monitoring already running")
            return
        
        self.health_monitor_stop_event.clear()
        
        def monitor_engines():
            logger.info(f"Engine health monitoring started (interval: {self.health_check_interval}s)")
            
            while not self.health_monitor_stop_event.wait(self.health_check_interval):
                try:
                    self._perform_health_checks()
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
        
        self.health_monitor_thread = threading.Thread(target=monitor_engines, daemon=True)
        self.health_monitor_thread.start()
    
    def stop_health_monitoring(self):
        """Stop health monitoring."""
        if not self.monitoring_enabled:
            return
        
        self.monitoring_enabled = False
        self.health_monitor_stop_event.set()
        
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=2.0)
            logger.info("Engine health monitoring stopped")
    
    def _perform_health_checks(self):
        """Perform health checks on all engines."""
        with self._lock:
            for engine_name, plugin in self.plugins.items():
                try:
                    if plugin.is_available():
                        if self.engine_status[engine_name] != EngineStatus.AVAILABLE:
                            self.engine_status[engine_name] = EngineStatus.AVAILABLE
                            logger.info(f"Engine {engine_name} is now available")
                            self._notify_engine_change("engine_available", {"name": engine_name})
                    else:
                        if self.engine_status[engine_name] == EngineStatus.AVAILABLE:
                            self.engine_status[engine_name] = EngineStatus.UNAVAILABLE
                            logger.warning(f"Engine {engine_name} is now unavailable")
                            self._notify_engine_change("engine_unavailable", {"name": engine_name})
                except Exception as e:
                    logger.error(f"Health check failed for {engine_name}: {e}")
                    self.engine_status[engine_name] = EngineStatus.ERROR
                    self.engine_metrics[engine_name].update_failure(str(e))
    
    def add_engine_change_callback(self, callback: Callable):
        """Add callback for engine change events."""
        if callback not in self.engine_change_callbacks:
            self.engine_change_callbacks.append(callback)
            logger.debug(f"Added engine change callback: {callback.__name__}")
    
    def remove_engine_change_callback(self, callback: Callable):
        """Remove engine change callback."""
        if callback in self.engine_change_callbacks:
            self.engine_change_callbacks.remove(callback)
            logger.debug(f"Removed engine change callback: {callback.__name__}")
    
    def _notify_engine_change(self, change_type: str, data: Dict[str, Any]):
        """Notify callbacks of engine changes."""
        for callback in self.engine_change_callbacks:
            try:
                callback(change_type, data)
            except Exception as e:
                logger.error(f"Error in engine change callback {callback.__name__}: {e}")
    
    def get_optimal_engine(self, text_length: int = 0) -> Optional[str]:
        """
        Get optimal engine based on load balancing and performance metrics.
        
        Args:
            text_length: Length of text to be processed
            
        Returns:
            str: Name of optimal engine
        """
        if not self.load_balancing_enabled:
            return self.get_active_engine()
        
        available_engines = []
        
        # Get available engines with their metrics
        for engine_name, plugin in self.plugins.items():
            if (self.engine_status.get(engine_name) == EngineStatus.AVAILABLE and 
                plugin.is_available()):
                
                metrics = self.engine_metrics[engine_name]
                available_engines.append({
                    'name': engine_name,
                    'success_rate': metrics.get_success_rate(),
                    'avg_response_time': metrics.average_processing_time,
                    'last_used': metrics.last_used or 0,
                    'total_requests': metrics.total_requests
                })
        
        if not available_engines:
            return None
        
        # Get load balancing strategy from config
        load_strategy = self.config.get("tts", {}).get("load_balancing_strategy", "weighted")
        
        if load_strategy == "round_robin":
            return self._get_round_robin_engine(available_engines)
        elif load_strategy == "least_used":
            return self._get_least_used_engine(available_engines)
        elif load_strategy == "weighted":
            return self._get_weighted_optimal_engine(available_engines)
        else:
            # Default to weighted
            return self._get_weighted_optimal_engine(available_engines)
    
    def _get_round_robin_engine(self, available_engines: List[Dict]) -> str:
        """Simple round-robin selection."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        # Sort engines by name for consistent ordering
        sorted_engines = sorted(available_engines, key=lambda x: x['name'])
        
        if self._round_robin_index >= len(sorted_engines):
            self._round_robin_index = 0
        
        selected_engine = sorted_engines[self._round_robin_index]['name']
        self._round_robin_index += 1
        
        logger.debug(f"Round-robin selected: {selected_engine} (index: {self._round_robin_index-1})")
        return selected_engine
    
    def _get_least_used_engine(self, available_engines: List[Dict]) -> str:
        """Select engine with least total requests."""
        # Sort by total requests (ascending) then by last used (ascending)
        sorted_engines = sorted(available_engines, 
                               key=lambda x: (x['total_requests'], x['last_used']))
        
        selected_engine = sorted_engines[0]['name']
        logger.debug(f"Least used selected: {selected_engine} ({sorted_engines[0]['total_requests']} requests)")
        return selected_engine
    
    def _get_weighted_optimal_engine(self, available_engines: List[Dict]) -> str:
        """Weighted selection with aggressive load balancing."""
        current_time = time.time()
        
        # Calculate scores with MORE aggressive load balancing
        for engine in available_engines:
            # Factors: load balancing (70%), success rate (20%), response time (10%)
            success_score = engine['success_rate'] / 100.0 if engine['success_rate'] > 0 else 1.0
            time_score = 1.0 / (1.0 + engine['avg_response_time']) if engine['avg_response_time'] > 0 else 1.0
            
            # VERY aggressive load balancing
            time_since_last_use = current_time - engine['last_used']
            
            if engine['total_requests'] == 0:
                # Strongly prefer unused engines
                load_score = 2.0  
            else:
                # Calculate average requests across all engines
                total_all_requests = sum(e['total_requests'] for e in available_engines)
                avg_requests = total_all_requests / len(available_engines)
                
                # Strongly penalize engines with above-average requests
                if engine['total_requests'] > avg_requests:
                    request_penalty = engine['total_requests'] / (avg_requests + 1)
                    load_score = max(0.1, 1.0 / (1.0 + request_penalty))
                else:
                    # Reward engines with below-average requests
                    load_score = 1.0 + (avg_requests - engine['total_requests']) / (avg_requests + 1)
                
                # Time factor - prefer engines not used recently
                time_factor = min(time_since_last_use / 10.0, 1.0)  # 10 second window
                load_score = (load_score + time_factor) / 2.0
            
            # AGGRESSIVE load balancing weighting
            engine['final_score'] = (0.7 * load_score + 0.2 * success_score + 0.1 * time_score)
        
        # Sort by final score (highest first)
        available_engines.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Add some randomization to prevent deterministic behavior
        # Select from top 2 engines with slight preference for the best
        import random
        if len(available_engines) >= 2:
            # 70% chance for best engine, 30% for second best
            if random.random() < 0.7:
                selected_engine = available_engines[0]['name']
            else:
                selected_engine = available_engines[1]['name']
        else:
            selected_engine = available_engines[0]['name']
        
        logger.debug(f"Weighted selection: {selected_engine} (score: {next(e['final_score'] for e in available_engines if e['name'] == selected_engine):.3f})")
        return selected_engine
    
    def get_engine_metrics(self, engine_name: str) -> Optional[EngineMetrics]:
        """Get performance metrics for an engine."""
        return self.engine_metrics.get(engine_name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all engines."""
        metrics_data = {}
        
        for engine_name, metrics in self.engine_metrics.items():
            metrics_data[engine_name] = {
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'success_rate': metrics.get_success_rate(),
                'average_response_time': metrics.average_processing_time,
                'median_response_time': metrics.get_median_response_time(),
                'last_used': metrics.last_used,
                'error_count': metrics.error_count,
                'last_error': metrics.last_error,
                'status': self.engine_status.get(engine_name, EngineStatus.UNKNOWN).value
            }
        
        return metrics_data
    
    def reset_metrics(self, engine_name: Optional[str] = None):
        """Reset performance metrics for engines."""
        if engine_name:
            if engine_name in self.engine_metrics:
                self.engine_metrics[engine_name] = EngineMetrics()
                logger.info(f"Reset metrics for engine: {engine_name}")
        else:
            for name in self.engine_metrics:
                self.engine_metrics[name] = EngineMetrics()
            logger.info("Reset metrics for all engines")
    
    def discover_plugins(self, plugin_dir: Optional[str] = None) -> List[str]:
        """
        Discover available TTS plugins.
        
        Args:
            plugin_dir: Directory to search for plugins
            
        Returns:
            List[str]: List of discovered plugin names
        """
        discovered = []
        
        # Default plugin directory
        if not plugin_dir:
            plugin_dir = "plugins/tts"
        
        try:
            plugin_path = Path(plugin_dir)
            if plugin_path.exists():
                for plugin_file in plugin_path.glob("*_plugin.py"):
                    plugin_name = plugin_file.stem.replace("_plugin", "")
                    if plugin_name not in self.plugins:
                        discovered.append(plugin_name)
                        logger.info(f"Discovered TTS plugin: {plugin_name}")
        except Exception as e:
            logger.error(f"Plugin discovery failed: {e}")
        
        return discovered
    
    def auto_load_plugins(self, plugin_dir: Optional[str] = None) -> int:
        """
        Automatically discover and load TTS plugins.
        
        Args:
            plugin_dir: Directory to search for plugins
            
        Returns:
            int: Number of plugins loaded
        """
        discovered_plugins = self.discover_plugins(plugin_dir)
        loaded_count = 0
        
        for plugin_name in discovered_plugins:
            try:
                if self.load_plugin(plugin_name):
                    loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to auto-load plugin {plugin_name}: {e}")
        
        logger.info(f"Auto-loaded {loaded_count} TTS plugins")
        return loaded_count
    
    def generate_speech(self, text: str, speaker_id: Optional[int] = None, 
                       engine: Optional[str] = None, use_context: bool = True) -> TTSResult:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            speaker_id: Speaker identifier
            engine: Specific engine to use (optional)
            use_context: Whether to use conversation context
            
        Returns:
            TTSResult: Generation result
        """
        # Determine which engine to use (with load balancing)
        target_engine = engine or self.get_optimal_engine(len(text))
        
        if not target_engine:
            return TTSResult(
                audio_data=b"",
                sample_rate=0,
                duration=0.0,
                engine_used="none",
                success=False,
                error_message="No TTS engine available"
            )
        
        # Prepare context if requested
        context = self.get_context() if use_context else None
        
        # Track performance
        start_time = time.time()
        
        try:
            # Generate speech using the selected engine
            plugin = self.plugins[target_engine]
            audio_data = plugin.generate_speech(text, speaker_id, context)
            sample_rate = plugin.get_sample_rate()
            
            # Calculate duration (approximate)
            duration = len(audio_data) / (sample_rate * 2)  # Assuming 16-bit audio
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.engine_metrics[target_engine].update_success(processing_time)
            
            # Add to context for future reference
            if use_context and speaker_id is not None:
                self.add_context(text, speaker_id, audio_data)
            
            logger.info(f"Generated speech: {len(text)} chars, {duration:.2f}s, {processing_time:.2f}s processing, engine: {target_engine}")
            
            return TTSResult(
                audio_data=audio_data,
                sample_rate=sample_rate,
                duration=duration,
                engine_used=target_engine,
                speaker_id=speaker_id,
                success=True
            )
            
        except Exception as e:
            # Update failure metrics
            processing_time = time.time() - start_time
            self.engine_metrics[target_engine].update_failure(str(e))
            self.engine_status[target_engine] = EngineStatus.ERROR
            
            logger.error(f"Speech generation failed with {target_engine} after {processing_time:.2f}s: {e}")
            
            # Try fallback engine if primary failed
            if target_engine != self.fallback_engine and self.fallback_engine in self.plugins:
                fallback_status = self.engine_status.get(self.fallback_engine, EngineStatus.UNKNOWN)
                if fallback_status == EngineStatus.AVAILABLE:
                    logger.info(f"Trying fallback engine: {self.fallback_engine}")
                    return self.generate_speech(text, speaker_id, self.fallback_engine, use_context)
            
            return TTSResult(
                audio_data=b"",
                sample_rate=0,
                duration=0.0,
                engine_used=target_engine,
                speaker_id=speaker_id,
                success=False,
                error_message=str(e)
            )
    
    def get_engine_info(self, engine_name: str) -> Dict[str, Any]:
        """
        Get information about a specific engine.
        
        Args:
            engine_name: Name of the engine
            
        Returns:
            Dict[str, Any]: Engine information
        """
        if engine_name not in self.plugins:
            return {"error": "Engine not found"}
        
        plugin = self.plugins[engine_name]
        
        return {
            "name": engine_name,
            "available": plugin.is_available(),
            "supported_languages": plugin.get_supported_languages(),
            "sample_rate": plugin.get_sample_rate() if plugin.is_available() else 0
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive TTS engine status."""
        return {
            "current_engine": self.current_engine,
            "available_engines": self.list_engines(),
            "context_length": len(self.context_history),
            "max_context_length": self.max_context_length,
            "monitoring_enabled": self.monitoring_enabled,
            "load_balancing_enabled": self.load_balancing_enabled,
            "health_check_interval": self.health_check_interval,
            "engines_info": {name: self.get_engine_info(name) for name in self.plugins.keys()},
            "performance_metrics": self.get_all_metrics(),
            "engine_status": {name: status.value for name, status in self.engine_status.items()}
        }
    
    def cleanup(self):
        """Clean up all plugins and resources."""
        try:
            # Stop health monitoring
            self.stop_health_monitoring()
            
            # Cleanup all plugins
            for plugin in self.plugins.values():
                try:
                    plugin.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up plugin: {e}")
            
            # Clear data structures
            self.plugins.clear()
            self.context_history.clear()
            self.engine_metrics.clear()
            self.engine_status.clear()
            self.engine_change_callbacks.clear()
            
            logger.info("Enhanced TTSEngine cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during TTSEngine cleanup: {e}")
    
    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.cleanup()
        except:
            pass