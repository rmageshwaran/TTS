"""
Configuration Manager Module

Handles loading, validation, and management of application configuration.
Supports YAML configuration files, environment variable overrides, and hot-reload.
"""

import os
import yaml
import logging
import threading
import time
import shutil
from typing import Dict, Any, Optional, Union, Callable, List
from pathlib import Path
from dataclasses import dataclass, asdict
from copy import deepcopy
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration hot-reload."""
    
    def __init__(self, config_manager: 'ConfigManager'):
        """Initialize watcher with reference to ConfigManager."""
        self.config_manager = config_manager
        self.last_modified = 0
        self.debounce_seconds = 1.0  # Prevent rapid successive reloads
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        # Check if it's our config file
        if event.src_path == self.config_manager.config_path:
            # Debounce rapid changes
            current_time = time.time()
            if current_time - self.last_modified < self.debounce_seconds:
                return
            
            self.last_modified = current_time
            logger.info(f"Configuration file changed: {event.src_path}")
            
            # Trigger hot-reload
            self.config_manager._trigger_hot_reload()


@dataclass
class ApplicationConfig:
    """Application-level configuration."""
    name: str = "TTS Virtual Microphone"
    version: str = "1.0.0"
    log_level: str = "INFO"
    debug: bool = False


@dataclass
class TextConfig:
    """Enhanced text processing configuration."""
    max_length: int = 5000
    min_length: int = 1
    enable_ssml: bool = True
    default_language: str = "en"
    chunk_size: int = 1000
    enable_pronunciation: bool = True
    custom_pronunciations: Dict[str, str] = None
    
    def __post_init__(self):
        if self.custom_pronunciations is None:
            self.custom_pronunciations = {}


@dataclass
class VoiceSettings:
    """Voice generation settings."""
    speaker_id: int = 0
    temperature: float = 1.0
    max_audio_length_ms: int = 10000


@dataclass
class TTSConfig:
    """TTS engine configuration."""
    default_engine: str = "sesame_csm"
    fallback_engine: str = "pyttsx3"
    model_path: str = "./models/sesame-csm-1b"
    context_length: int = 4
    voice_settings: VoiceSettings = None
    
    def __post_init__(self):
        if self.voice_settings is None:
            self.voice_settings = VoiceSettings()


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 24000
    channels: int = 1
    format: str = "wav"
    quality: str = "high"
    normalization: bool = True
    buffer_size: int = 1024
    streaming: bool = True


@dataclass
class VirtualAudioConfig:
    """Virtual audio configuration."""
    device_name: str = "CABLE Input"
    auto_detect: bool = True
    fallback_device: str = "default"
    buffer_size: int = 1024


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "./logs/tts_app.log"
    max_size: str = "10MB"
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_output: bool = True


@dataclass
class Config:
    """Main configuration container."""
    application: ApplicationConfig = None
    text: TextConfig = None
    tts: TTSConfig = None
    audio: AudioConfig = None
    virtual_audio: VirtualAudioConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.application is None:
            self.application = ApplicationConfig()
        if self.text is None:
            self.text = TextConfig()
        if self.tts is None:
            self.tts = TTSConfig()
        if self.audio is None:
            self.audio = AudioConfig()
        if self.virtual_audio is None:
            self.virtual_audio = VirtualAudioConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


class ConfigManager:
    """
    Configuration Manager for loading and managing application settings.
    
    Features:
    - YAML configuration loading
    - Environment variable support
    - Configuration validation
    - Hot-reload capabilities
    - Configuration backup and restore
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._find_default_config()
        self.config: Config = Config()
        self.raw_config: Dict[str, Any] = {}
        self.environment_prefix = "TTS_"
        
        # Hot-reload and notification features
        self.hot_reload_enabled = False
        self.file_watcher = None
        self.observer = None
        self.change_callbacks: List[Callable[[Config], None]] = []
        self._lock = threading.Lock()
        
        # Profile management
        self.current_profile = "default"
        self.profiles: Dict[str, Config] = {}
        
        # Backup and restore
        self.backup_directory = os.path.join(os.path.dirname(self.config_path), 'backups')
        self.auto_backup = True
        self.max_backups = 10
        
        logger.info(f"ConfigManager initialized with config: {self.config_path}")
    
    def _find_default_config(self) -> str:
        """Find default configuration file."""
        possible_paths = [
            "./config/config.yaml",
            "./config.yaml",
            os.path.expanduser("~/.tts_virtual_mic/config.yaml"),
            "/etc/tts_virtual_mic/config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return first path as default (will be created if needed)
        return possible_paths[0]
    
    def load(self) -> Config:
        """
        Load configuration from file.
        
        Returns:
            Config: Loaded configuration
        """
        try:
            # Load from file if exists
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.raw_config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self.raw_config = {}
            
            # Apply environment variable overrides
            self._apply_environment_overrides()
            
            # Convert to structured config
            self.config = self._dict_to_config(self.raw_config)
            
            # Validate configuration
            self._validate_config()
            
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Return default configuration on error
            self.config = Config()
            return self.config
    
    def save(self, config: Optional[Config] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save (uses current if None)
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if config:
                self.config = config
            
            # Convert config to dictionary
            config_dict = self._config_to_dict(self.config)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Save to file
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object."""
        config = Config()
        
        # Application config
        if 'application' in config_dict:
            app_dict = config_dict['application']
            config.application = ApplicationConfig(
                name=app_dict.get('name', 'TTS Virtual Microphone'),
                version=app_dict.get('version', '1.0.0'),
                log_level=app_dict.get('log_level', 'INFO'),
                debug=app_dict.get('debug', False)
            )
        
        # Text config
        if 'text' in config_dict:
            text_dict = config_dict['text']
            config.text = TextConfig(
                max_length=text_dict.get('max_length', 5000),
                min_length=text_dict.get('min_length', 1),
                enable_ssml=text_dict.get('enable_ssml', True),
                default_language=text_dict.get('default_language', 'en'),
                chunk_size=text_dict.get('chunk_size', 1000)
            )
        
        # TTS config
        if 'tts' in config_dict:
            tts_dict = config_dict['tts']
            voice_settings_dict = tts_dict.get('voice_settings', {})
            voice_settings = VoiceSettings(
                speaker_id=voice_settings_dict.get('speaker_id', 0),
                temperature=voice_settings_dict.get('temperature', 1.0),
                max_audio_length_ms=voice_settings_dict.get('max_audio_length_ms', 10000)
            )
            
            config.tts = TTSConfig(
                default_engine=tts_dict.get('default_engine', 'sesame_csm'),
                fallback_engine=tts_dict.get('fallback_engine', 'pyttsx3'),
                model_path=tts_dict.get('model_path', './models/sesame-csm-1b'),
                context_length=tts_dict.get('context_length', 4),
                voice_settings=voice_settings
            )
        
        # Audio config
        if 'audio' in config_dict:
            audio_dict = config_dict['audio']
            config.audio = AudioConfig(
                sample_rate=audio_dict.get('sample_rate', 24000),
                channels=audio_dict.get('channels', 1),
                format=audio_dict.get('format', 'wav'),
                quality=audio_dict.get('quality', 'high'),
                normalization=audio_dict.get('normalization', True),
                buffer_size=audio_dict.get('buffer_size', 1024),
                streaming=audio_dict.get('streaming', True)
            )
        
        # Virtual audio config
        if 'virtual_audio' in config_dict:
            va_dict = config_dict['virtual_audio']
            config.virtual_audio = VirtualAudioConfig(
                device_name=va_dict.get('device_name', 'CABLE Input'),
                auto_detect=va_dict.get('auto_detect', True),
                fallback_device=va_dict.get('fallback_device', 'default'),
                buffer_size=va_dict.get('buffer_size', 1024)
            )
        
        # Logging config
        if 'logging' in config_dict:
            log_dict = config_dict['logging']
            config.logging = LoggingConfig(
                level=log_dict.get('level', 'INFO'),
                file=log_dict.get('file', './logs/tts_app.log'),
                max_size=log_dict.get('max_size', '10MB'),
                backup_count=log_dict.get('backup_count', 5),
                format=log_dict.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                console_output=log_dict.get('console_output', True)
            )
        
        return config
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary."""
        return {
            'application': asdict(config.application),
            'text': asdict(config.text),
            'tts': asdict(config.tts),
            'audio': asdict(config.audio),
            'virtual_audio': asdict(config.virtual_audio),
            'logging': asdict(config.logging)
        }
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        for env_var, value in os.environ.items():
            if env_var.startswith(self.environment_prefix):
                # Convert environment variable to config path
                config_path = env_var[len(self.environment_prefix):].lower().replace('_', '.')
                
                # Parse value
                parsed_value = self._parse_env_value(value)
                
                # Apply to config
                self._set_nested_value(self.raw_config, config_path, parsed_value)
                
                logger.debug(f"Applied environment override: {config_path} = {parsed_value}")
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value."""
        # Try boolean
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Try numeric
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config_dict: Dict[str, Any], path: str, value: Any):
        """Set nested configuration value."""
        keys = path.split('.')
        current = config_dict
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set value
        current[keys[-1]] = value
    
    def _validate_config(self):
        """Validate configuration values."""
        errors = []
        
        # Validate audio settings
        if self.config.audio.sample_rate <= 0:
            errors.append("Audio sample rate must be positive")
        
        if self.config.audio.channels < 1 or self.config.audio.channels > 2:
            errors.append("Audio channels must be 1 or 2")
        
        if self.config.audio.format not in ['wav', 'mp3', 'ogg']:
            errors.append("Audio format must be wav, mp3, or ogg")
        
        # Validate TTS settings
        if self.config.tts.context_length < 0:
            errors.append("TTS context length must be non-negative")
        
        if self.config.tts.voice_settings.temperature < 0:
            errors.append("Voice temperature must be non-negative")
        
        # Validate text settings
        if self.config.text.max_length <= self.config.text.min_length:
            errors.append("Text max_length must be greater than min_length")
        
        if errors:
            error_msg = "Configuration validation failed: " + "; ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug("Configuration validation passed")
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary."""
        return deepcopy(self.raw_config)
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
            
        Returns:
            bool: True if updated successfully
        """
        try:
            # Apply updates to raw config
            self._deep_update(self.raw_config, updates)
            
            # Reload configuration
            self.config = self._dict_to_config(self.raw_config)
            
            # Validate
            self._validate_config()
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def create_backup(self) -> bool:
        """
        Create backup of current configuration.
        
        Returns:
            bool: True if backup created successfully
        """
        try:
            if not os.path.exists(self.config_path):
                return False
            
            backup_dir = os.path.join(os.path.dirname(self.config_path), 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f'config_backup_{timestamp}.yaml')
            
            import shutil
            shutil.copy2(self.config_path, backup_path)
            
            logger.info(f"Configuration backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create configuration backup: {e}")
            return False
    
    # ===== HOT-RELOAD AND NOTIFICATION FEATURES =====
    
    def enable_hot_reload(self) -> bool:
        """
        Enable hot-reload functionality.
        
        Returns:
            bool: True if hot-reload enabled successfully
        """
        try:
            if self.hot_reload_enabled:
                logger.warning("Hot-reload already enabled")
                return True
            
            # Create file watcher
            self.file_watcher = ConfigFileWatcher(self)
            self.observer = Observer()
            
            # Watch the directory containing the config file
            watch_dir = os.path.dirname(self.config_path)
            self.observer.schedule(self.file_watcher, watch_dir, recursive=False)
            
            # Start watching
            self.observer.start()
            self.hot_reload_enabled = True
            
            logger.info(f"Hot-reload enabled for: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable hot-reload: {e}")
            return False
    
    def disable_hot_reload(self) -> bool:
        """
        Disable hot-reload functionality.
        
        Returns:
            bool: True if hot-reload disabled successfully
        """
        try:
            if not self.hot_reload_enabled:
                return True
            
            if self.observer:
                self.observer.stop()
                self.observer.join()
                self.observer = None
            
            self.file_watcher = None
            self.hot_reload_enabled = False
            
            logger.info("Hot-reload disabled")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable hot-reload: {e}")
            return False
    
    def add_change_callback(self, callback: Callable[[Config], None]) -> bool:
        """
        Add callback to be called when configuration changes.
        
        Args:
            callback: Function to call with new config
            
        Returns:
            bool: True if callback added successfully
        """
        try:
            with self._lock:
                if callback not in self.change_callbacks:
                    self.change_callbacks.append(callback)
                    logger.debug(f"Added config change callback: {callback.__name__}")
                    return True
                else:
                    logger.warning(f"Callback already registered: {callback.__name__}")
                    return False
        except Exception as e:
            logger.error(f"Failed to add change callback: {e}")
            return False
    
    def remove_change_callback(self, callback: Callable[[Config], None]) -> bool:
        """
        Remove configuration change callback.
        
        Args:
            callback: Callback function to remove
            
        Returns:
            bool: True if callback removed successfully
        """
        try:
            with self._lock:
                if callback in self.change_callbacks:
                    self.change_callbacks.remove(callback)
                    logger.debug(f"Removed config change callback: {callback.__name__}")
                    return True
                else:
                    logger.warning(f"Callback not found: {callback.__name__}")
                    return False
        except Exception as e:
            logger.error(f"Failed to remove change callback: {e}")
            return False
    
    def _trigger_hot_reload(self):
        """Internal method to trigger hot-reload."""
        try:
            with self._lock:
                # Backup current config before reload
                old_config = deepcopy(self.config)
                
                # Reload configuration
                new_config = self.load()
                
                # Notify callbacks of the change
                for callback in self.change_callbacks:
                    try:
                        callback(new_config)
                    except Exception as e:
                        logger.error(f"Error in config change callback {callback.__name__}: {e}")
                
                logger.info("Configuration hot-reloaded successfully")
                
        except Exception as e:
            logger.error(f"Hot-reload failed: {e}")
    
    # ===== CONFIGURATION PROFILES =====
    
    def create_profile(self, profile_name: str, config: Optional[Config] = None) -> bool:
        """
        Create a new configuration profile.
        
        Args:
            profile_name: Name of the profile
            config: Configuration to save (uses current if None)
            
        Returns:
            bool: True if profile created successfully
        """
        try:
            if config is None:
                config = self.config
            
            self.profiles[profile_name] = deepcopy(config)
            
            # Save profile to file
            profile_path = os.path.join(os.path.dirname(self.config_path), f'profile_{profile_name}.yaml')
            config_dict = self._config_to_dict(config)
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration profile '{profile_name}' created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create profile '{profile_name}': {e}")
            return False
    
    def load_profile(self, profile_name: str) -> bool:
        """
        Load a configuration profile.
        
        Args:
            profile_name: Name of the profile to load
            
        Returns:
            bool: True if profile loaded successfully
        """
        try:
            # Try to load from memory first
            if profile_name in self.profiles:
                self.config = deepcopy(self.profiles[profile_name])
                self.current_profile = profile_name
                logger.info(f"Loaded profile '{profile_name}' from memory")
                return True
            
            # Try to load from file
            profile_path = os.path.join(os.path.dirname(self.config_path), f'profile_{profile_name}.yaml')
            if os.path.exists(profile_path):
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile_dict = yaml.safe_load(f) or {}
                
                profile_config = self._dict_to_config(profile_dict)
                self.config = profile_config
                self.profiles[profile_name] = deepcopy(profile_config)
                self.current_profile = profile_name
                
                logger.info(f"Loaded profile '{profile_name}' from file")
                return True
            
            logger.error(f"Profile '{profile_name}' not found")
            return False
            
        except Exception as e:
            logger.error(f"Failed to load profile '{profile_name}': {e}")
            return False
    
    def list_profiles(self) -> List[str]:
        """
        List all available configuration profiles.
        
        Returns:
            List[str]: List of profile names
        """
        profiles = set(self.profiles.keys())
        
        # Add profiles from files
        config_dir = os.path.dirname(self.config_path)
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.startswith('profile_') and file.endswith('.yaml'):
                    profile_name = file[8:-5]  # Remove 'profile_' and '.yaml'
                    profiles.add(profile_name)
        
        return sorted(list(profiles))
    
    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete a configuration profile.
        
        Args:
            profile_name: Name of the profile to delete
            
        Returns:
            bool: True if profile deleted successfully
        """
        try:
            # Remove from memory
            if profile_name in self.profiles:
                del self.profiles[profile_name]
            
            # Remove file
            profile_path = os.path.join(os.path.dirname(self.config_path), f'profile_{profile_name}.yaml')
            if os.path.exists(profile_path):
                os.remove(profile_path)
            
            logger.info(f"Configuration profile '{profile_name}' deleted")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete profile '{profile_name}': {e}")
            return False
    
    # ===== ENHANCED BACKUP AND RESTORE =====
    
    def restore_from_backup(self, backup_filename: Optional[str] = None) -> bool:
        """
        Restore configuration from backup.
        
        Args:
            backup_filename: Specific backup to restore (latest if None)
            
        Returns:
            bool: True if restore successful
        """
        try:
            if not os.path.exists(self.backup_directory):
                logger.error("No backup directory found")
                return False
            
            # Find backup file
            if backup_filename:
                backup_path = os.path.join(self.backup_directory, backup_filename)
                if not os.path.exists(backup_path):
                    logger.error(f"Backup file not found: {backup_filename}")
                    return False
            else:
                # Get latest backup
                backup_files = [f for f in os.listdir(self.backup_directory) if f.startswith('config_backup_')]
                if not backup_files:
                    logger.error("No backup files found")
                    return False
                
                backup_files.sort(reverse=True)  # Most recent first
                backup_path = os.path.join(self.backup_directory, backup_files[0])
            
            # Create backup of current config before restore
            if self.auto_backup:
                self.create_backup()
            
            # Restore from backup
            shutil.copy2(backup_path, self.config_path)
            
            # Reload configuration
            self.load()
            
            logger.info(f"Configuration restored from: {os.path.basename(backup_path)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available configuration backups.
        
        Returns:
            List[Dict[str, Any]]: List of backup information
        """
        backups = []
        
        try:
            if not os.path.exists(self.backup_directory):
                return backups
            
            for filename in os.listdir(self.backup_directory):
                if filename.startswith('config_backup_') and filename.endswith('.yaml'):
                    backup_path = os.path.join(self.backup_directory, filename)
                    stat = os.stat(backup_path)
                    
                    backups.append({
                        'filename': filename,
                        'created': datetime.fromtimestamp(stat.st_ctime),
                        'modified': datetime.fromtimestamp(stat.st_mtime),
                        'size_bytes': stat.st_size
                    })
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x['created'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
        
        return backups
    
    def cleanup_old_backups(self) -> int:
        """
        Clean up old backup files keeping only max_backups.
        
        Returns:
            int: Number of backups deleted
        """
        try:
            backups = self.list_backups()
            if len(backups) <= self.max_backups:
                return 0
            
            backups_to_delete = backups[self.max_backups:]
            deleted_count = 0
            
            for backup in backups_to_delete:
                backup_path = os.path.join(self.backup_directory, backup['filename'])
                try:
                    os.remove(backup_path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete backup {backup['filename']}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old backup files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
            return 0
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            'config_path': self.config_path,
            'application': {
                'name': self.config.application.name,
                'version': self.config.application.version,
                'debug': self.config.application.debug
            },
            'tts_engine': self.config.tts.default_engine,
            'audio_format': f"{self.config.audio.format} ({self.config.audio.sample_rate}Hz)",
            'virtual_device': self.config.virtual_audio.device_name,
            'log_level': self.config.logging.level
        }