"""
Advanced People Counter - Professional Edition
A state-of-the-art computer vision application for real-time people counting and tracking
"""
from .core import PersonDetector, ObjectTracker, PeopleCounter
from .ui import PeopleCounterMainWindow
from .utils import config

__version__ = "1.0.0"
__author__ = "Computer Vision Solutions"
__description__ = "Advanced people counting using YOLOv8 and modern UI"

__all__ = [
    'PersonDetector',
    'ObjectTracker',
    'PeopleCounter',
    'PeopleCounterMainWindow',
    'config'
]