"""
Logging utilities for the People Counter application
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import os

from ..utils.config import config


def setup_logging(name: str = "people_counter",
                  log_level: Optional[str] = None,
                  log_file: Optional[str] = None,
                  enable_console: bool = True) -> logging.Logger:
    """
    Setup comprehensive logging configuration

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file path
        enable_console: Enable console logging

    Returns:
        Configured logger instance
    """
    # Use config values if not specified
    log_level = log_level or config.analytics.log_level
    log_file = log_file or config.analytics.log_file

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler for detailed logging
    if config.analytics.enable_logging:
        try:
            # Ensure log directory exists
            log_path = Path(log_file).parent
            log_path.mkdir(parents=True, exist_ok=True)

            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # Prevent duplicate messages
    logger.propagate = False

    return logger


class PerformanceLogger:
    """
    Specialized logger for performance monitoring
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}

    def start_timer(self, name: str):
        """Start a performance timer"""
        self.timers[name] = {
            'start_time': os.times().elapsed,
            'cpu_start': os.times().user + os.times().system
        }

    def end_timer(self, name: str) -> float:
        """End a performance timer and return elapsed time"""
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' not started")
            return 0.0

        timer = self.timers[name]
        elapsed = os.times().elapsed - timer['start_time']
        cpu_time = (os.times().user + os.times().system) - timer['cpu_start']

        self.logger.info(f"Performance: {name} - Elapsed: {elapsed".3f"}s, CPU: {cpu_time".3f"}s")

        del self.timers[name]
        return elapsed

    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        log_data = {
            'operation': operation,
            'duration_ms': duration * 1000,
            **kwargs
        }

        self.logger.info(f"Performance: {log_data}")


class AnalyticsLogger:
    """
    Logger for analytics and business metrics
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.session_start = os.times().elapsed

    def log_session_event(self, event: str, **kwargs):
        """Log session-related events"""
        session_duration = os.times().elapsed - self.session_start
        event_data = {
            'event': event,
            'session_duration': session_duration,
            **kwargs
        }

        self.logger.info(f"Session: {event_data}")

    def log_count_event(self, event_type: str, count: int, **kwargs):
        """Log counting-related events"""
        event_data = {
            'event_type': event_type,
            'count': count,
            **kwargs
        }

        self.logger.info(f"Count: {event_data}")


class ErrorLogger:
    """
    Specialized logger for error tracking and diagnostics
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_error(self, error: Exception, context: str = "", **kwargs):
        """Log detailed error information"""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            **kwargs
        }

        self.logger.error(f"Error: {error_data}", exc_info=True)

    def log_warning(self, message: str, **kwargs):
        """Log warning with context"""
        warning_data = {
            'message': message,
            **kwargs
        }

        self.logger.warning(f"Warning: {warning_data}")


# Global loggers
performance_logger = None
analytics_logger = None
error_logger = None

def initialize_loggers():
    """Initialize global loggers"""
    global performance_logger, analytics_logger, error_logger

    base_logger = setup_logging()

    performance_logger = PerformanceLogger(base_logger)
    analytics_logger = AnalyticsLogger(base_logger)
    error_logger = ErrorLogger(base_logger)


# Convenience functions for common logging patterns
def log_function_call(func_name: str, args: dict = None):
    """Log function call for debugging"""
    if not config.analytics.enable_logging:
        return

    call_data = {
        'function': func_name,
        'arguments': args or {}
    }

    logging.getLogger("people_counter").debug(f"Function Call: {call_data}")


def log_api_request(endpoint: str, method: str = "POST", status_code: Optional[int] = None):
    """Log API request for monitoring"""
    if not config.analytics.enable_logging:
        return

    request_data = {
        'endpoint': endpoint,
        'method': method,
        'status_code': status_code
    }

    logging.getLogger("people_counter").info(f"API Request: {request_data}")


def log_performance_metric(name: str, value: float, unit: str = "ms"):
    """Log performance metrics"""
    if not config.analytics.enable_logging:
        return

    metric_data = {
        'metric': name,
        'value': value,
        'unit': unit
    }

    logging.getLogger("people_counter").info(f"Performance Metric: {metric_data}")


# Context manager for performance logging
class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger("people_counter")
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (time.time() - self.start_time) * 1000  # Convert to ms
            self.logger.info(f"Performance: {self.operation_name} - {duration".2f"}ms")

            if exc_type:
                self.logger.error(f"Error in {self.operation_name}: {exc_val}")


# Import time for the timer context manager
import time
