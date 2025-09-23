"""
Utilities module for Advanced People Counter
Contains configuration, logging, and helper functions
"""
from .config import config, ConfigManager, ModelConfig, TrackingConfig, DetectionArea
from .logger import setup_logging, PerformanceLogger, AnalyticsLogger, ErrorLogger
from .helpers import validate_video_source, get_video_info, DataExporter

__all__ = [
    'config',
    'ConfigManager',
    'ModelConfig',
    'TrackingConfig',
    'DetectionArea',
    'setup_logging',
    'PerformanceLogger',
    'AnalyticsLogger',
    'ErrorLogger',
    'validate_video_source',
    'get_video_info',
    'DataExporter'
]