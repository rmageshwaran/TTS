"""
Error Handler Module

Provides centralized error handling and custom exception classes
for the TTS Virtual Microphone application.
"""

import logging
import traceback
import sys
from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TTSError(Exception):
    """Base exception class for TTS application errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize TTS error.
        
        Args:
            message: Error message
            severity: Error severity level
            error_code: Optional error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.error_code = error_code
        self.details = details or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "message": self.message,
            "severity": self.severity.value,
            "error_code": self.error_code,
            "details": self.details,
            "type": self.__class__.__name__
        }


class ConfigurationError(TTSError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.config_path = config_path


class ModelLoadError(TTSError):
    """Model loading errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 model_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.model_path = model_path


class AudioProcessingError(TTSError):
    """Audio processing errors."""
    
    def __init__(self, message: str, audio_format: Optional[str] = None, 
                 sample_rate: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.audio_format = audio_format
        self.sample_rate = sample_rate


class VirtualAudioError(TTSError):
    """Virtual audio device errors."""
    
    def __init__(self, message: str, device_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.device_name = device_name


class TextProcessingError(TTSError):
    """Text processing errors."""
    
    def __init__(self, message: str, text_length: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.text_length = text_length


class PluginError(TTSError):
    """Plugin-related errors."""
    
    def __init__(self, message: str, plugin_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.plugin_name = plugin_name


logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Centralized error handler for the application.
    
    Features:
    - Error logging and reporting
    - Error classification and handling
    - Recovery strategies
    - Error statistics
    """
    
    def __init__(self):
        """Initialize error handler."""
        self.error_counts: Dict[str, int] = {}
        self.error_history: list = []
        self.max_history = 100
        
    def handle_error(self, error: Exception, context: Optional[str] = None, 
                    recover: bool = True) -> bool:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            recover: Whether to attempt recovery
            
        Returns:
            bool: True if error was handled successfully
        """
        error_info = self._analyze_error(error, context)
        
        # Log the error
        self._log_error(error_info)
        
        # Update statistics
        self._update_statistics(error_info)
        
        # Add to history
        self._add_to_history(error_info)
        
        # Attempt recovery if requested
        if recover:
            return self._attempt_recovery(error_info)
        
        return False
    
    def _analyze_error(self, error: Exception, context: Optional[str] = None) -> Dict[str, Any]:
        """Analyze error and extract relevant information."""
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "severity": ErrorSeverity.MEDIUM
        }
        
        # Enhanced analysis for TTS errors
        if isinstance(error, TTSError):
            error_info.update({
                "severity": error.severity,
                "error_code": error.error_code,
                "details": error.details
            })
        
        # Classify severity based on error type
        if isinstance(error, (ConfigurationError, ModelLoadError)):
            error_info["severity"] = ErrorSeverity.HIGH
        elif isinstance(error, (KeyboardInterrupt, SystemExit)):
            error_info["severity"] = ErrorSeverity.LOW
        elif isinstance(error, (MemoryError, OSError)):
            error_info["severity"] = ErrorSeverity.CRITICAL
        
        return error_info
    
    def _log_error(self, error_info: Dict[str, Any]):
        """Log error with appropriate level."""
        severity = error_info["severity"]
        message = f"{error_info['type']}: {error_info['message']}"
        
        if error_info["context"]:
            message = f"[{error_info['context']}] {message}"
        
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(message)
            logger.critical(f"Traceback:\n{error_info['traceback']}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(message)
            logger.debug(f"Traceback:\n{error_info['traceback']}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(message)
            logger.debug(f"Traceback:\n{error_info['traceback']}")
        else:  # LOW
            logger.info(message)
    
    def _update_statistics(self, error_info: Dict[str, Any]):
        """Update error statistics."""
        error_type = error_info["type"]
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def _add_to_history(self, error_info: Dict[str, Any]):
        """Add error to history."""
        import datetime
        
        error_info["timestamp"] = datetime.datetime.now().isoformat()
        self.error_history.append(error_info)
        
        # Limit history size
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
    
    def _attempt_recovery(self, error_info: Dict[str, Any]) -> bool:
        """
        Attempt to recover from error.
        
        Args:
            error_info: Error information
            
        Returns:
            bool: True if recovery was successful
        """
        error_type = error_info["type"]
        
        # Recovery strategies based on error type
        recovery_strategies = {
            "ConfigurationError": self._recover_configuration_error,
            "ModelLoadError": self._recover_model_load_error,
            "VirtualAudioError": self._recover_virtual_audio_error,
            "AudioProcessingError": self._recover_audio_processing_error,
            "PluginError": self._recover_plugin_error
        }
        
        if error_type in recovery_strategies:
            try:
                return recovery_strategies[error_type](error_info)
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {error_type}: {recovery_error}")
        
        return False
    
    def _recover_configuration_error(self, error_info: Dict[str, Any]) -> bool:
        """Recover from configuration errors."""
        logger.info("Attempting to recover from configuration error...")
        
        # Could attempt to load default configuration
        # For now, just log the attempt
        return False
    
    def _recover_model_load_error(self, error_info: Dict[str, Any]) -> bool:
        """Recover from model loading errors."""
        logger.info("Attempting to recover from model load error...")
        
        # Could attempt to use fallback model
        return False
    
    def _recover_virtual_audio_error(self, error_info: Dict[str, Any]) -> bool:
        """Recover from virtual audio errors."""
        logger.info("Attempting to recover from virtual audio error...")
        
        # Could attempt to use different audio device
        return False
    
    def _recover_audio_processing_error(self, error_info: Dict[str, Any]) -> bool:
        """Recover from audio processing errors."""
        logger.info("Attempting to recover from audio processing error...")
        
        # Could attempt different audio format/settings
        return False
    
    def _recover_plugin_error(self, error_info: Dict[str, Any]) -> bool:
        """Recover from plugin errors."""
        logger.info("Attempting to recover from plugin error...")
        
        # Could attempt to reload plugin or use alternative
        return False
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts.copy(),
            "history_size": len(self.error_history),
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }
    
    def get_error_history(self, limit: Optional[int] = None) -> list:
        """
        Get error history.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            list: Error history
        """
        history = self.error_history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()
        logger.info("Error history cleared")
    
    def reset_statistics(self):
        """Reset error statistics."""
        self.error_counts.clear()
        logger.info("Error statistics reset")


# Global error handler instance
error_handler = ErrorHandler()


def handle_exception(error: Exception, context: Optional[str] = None, 
                    recover: bool = True) -> bool:
    """
    Global function to handle exceptions.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        recover: Whether to attempt recovery
        
    Returns:
        bool: True if error was handled successfully
    """
    return error_handler.handle_error(error, context, recover)


def get_error_stats() -> Dict[str, Any]:
    """Get error statistics."""
    return error_handler.get_error_statistics()


def setup_global_exception_handler():
    """Setup global exception handler for unhandled exceptions."""
    
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        """Handle unhandled exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Let KeyboardInterrupt pass through
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Unhandled exception occurred", exc_info=(exc_type, exc_value, exc_traceback))
        handle_exception(exc_value, "Global Exception Handler", recover=False)
    
    sys.excepthook = global_exception_handler
    logger.info("Global exception handler installed")


class ErrorContext:
    """Context manager for error handling."""
    
    def __init__(self, context: str, suppress: bool = False):
        """
        Initialize error context.
        
        Args:
            context: Context description
            suppress: Whether to suppress exceptions
        """
        self.context = context
        self.suppress = suppress
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context and handle any exception."""
        if exc_type is not None:
            handle_exception(exc_value, self.context)
            return self.suppress  # Return True to suppress exception
        return False


def safe_execute(func, *args, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (success: bool, result: Any, error: Exception)
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        handle_exception(e, f"safe_execute: {func.__name__}")
        return False, None, e


class RetryHandler:
    """Handler for retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0, 
                 backoff_factor: float = 2.0):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retries
            initial_delay: Initial delay between retries
            backoff_factor: Exponential backoff factor
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
    
    def retry(self, func, *args, **kwargs):
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        import time
        
        last_exception = None
        delay = self.initial_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} failed: {e}")
                    time.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    logger.error(f"All {self.max_retries} retries failed")
        
        # All retries failed, raise the last exception
        raise last_exception