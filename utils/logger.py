"""
Logging Utilities Module

Provides centralized logging setup and configuration for the application.
Includes log rotation, formatting, and multiple output destinations.
"""

import logging
import logging.handlers
import sys
import os
from typing import Optional
from pathlib import Path


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 max_size: str = "10MB",
                 backup_count: int = 5,
                 format_string: Optional[str] = None,
                 console_output: bool = True,
                 verbose: bool = False) -> logging.Logger:
    """
    Setup centralized logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        format_string: Custom log format string
        console_output: Whether to output to console
        verbose: Enable verbose logging
        
    Returns:
        logging.Logger: Configured root logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Default format
    if format_string is None:
        if verbose:
            format_string = "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
        else:
            format_string = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        
        # Add color support for console (if available)
        try:
            import colorlog
            color_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(color_formatter)
        except ImportError:
            # colorlog not available, use standard formatter
            pass
        
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse max_size
        max_bytes = _parse_size(max_size)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set up specific logger levels for external libraries
    _configure_external_loggers()
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {level}, File: {log_file}")
    
    return root_logger


def _parse_size(size_str: str) -> int:
    """Parse size string (e.g., '10MB') to bytes."""
    size_str = size_str.upper()
    
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        # Assume bytes
        return int(size_str)


def _configure_external_loggers():
    """Configure logging levels for external libraries."""
    # Reduce noise from external libraries
    external_loggers = {
        'urllib3': logging.WARNING,
        'requests': logging.WARNING,
        'matplotlib': logging.WARNING,
        'PIL': logging.WARNING,
        'transformers': logging.WARNING,
        'torch': logging.WARNING,
        'sounddevice': logging.WARNING,
    }
    
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


class ContextualLogger:
    """Logger wrapper that adds contextual information to log messages."""
    
    def __init__(self, logger: logging.Logger, context: str):
        """
        Initialize contextual logger.
        
        Args:
            logger: Base logger instance
            context: Context string to add to messages
        """
        self.logger = logger
        self.context = context
    
    def _format_message(self, message: str) -> str:
        """Add context to message."""
        return f"[{self.context}] {message}"
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message with context."""
        self.logger.debug(self._format_message(message), *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message with context."""
        self.logger.info(self._format_message(message), *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message with context."""
        self.logger.warning(self._format_message(message), *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message with context."""
        self.logger.error(self._format_message(message), *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message with context."""
        self.logger.critical(self._format_message(message), *args, **kwargs)


class LogCapture:
    """Utility class for capturing log messages during testing."""
    
    def __init__(self, logger_name: str = None, level: int = logging.DEBUG):
        """
        Initialize log capture.
        
        Args:
            logger_name: Name of logger to capture (None for root)
            level: Minimum level to capture
        """
        self.logger_name = logger_name
        self.level = level
        self.messages = []
        self.handler = None
    
    def __enter__(self):
        """Start capturing log messages."""
        self.handler = LoggingHandler(self.messages)
        self.handler.setLevel(self.level)
        
        logger = logging.getLogger(self.logger_name)
        logger.addHandler(self.handler)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop capturing log messages."""
        if self.handler:
            logger = logging.getLogger(self.logger_name)
            logger.removeHandler(self.handler)
    
    def get_messages(self) -> list:
        """Get captured messages."""
        return self.messages.copy()
    
    def clear(self):
        """Clear captured messages."""
        self.messages.clear()


class LoggingHandler(logging.Handler):
    """Custom logging handler for capturing messages."""
    
    def __init__(self, message_list: list):
        """
        Initialize handler.
        
        Args:
            message_list: List to store messages
        """
        super().__init__()
        self.message_list = message_list
    
    def emit(self, record):
        """Emit log record."""
        self.message_list.append(self.format(record))


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        import time
        
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {e}")
            raise
    
    return wrapper


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, name: str):
        """
        Initialize performance logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(f"performance.{name}")
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        import time
        self.metrics[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """
        End timing an operation and log the result.
        
        Args:
            operation: Operation name
            
        Returns:
            float: Elapsed time in seconds
        """
        import time
        
        if operation not in self.metrics:
            self.logger.warning(f"Timer for '{operation}' not started")
            return 0.0
        
        elapsed = time.time() - self.metrics[operation]
        self.logger.info(f"{operation}: {elapsed:.3f}s")
        del self.metrics[operation]
        
        return elapsed
    
    def log_memory_usage(self, operation: str):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"{operation} - Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")


# Create default performance logger
perf_logger = PerformanceLogger("default")