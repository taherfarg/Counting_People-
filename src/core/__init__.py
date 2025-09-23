"""
Core module for Advanced People Counter
Contains detection, tracking, and counting functionality
"""
from .detector import PersonDetector, DetectionArea
from .tracker import ObjectTracker, CentroidTracker, SORTTracker, KalmanTracker
from .counter import PeopleCounter, APIManager

__all__ = [
    'PersonDetector',
    'DetectionArea',
    'ObjectTracker',
    'CentroidTracker',
    'SORTTracker',
    'KalmanTracker',
    'PeopleCounter',
    'APIManager'
]