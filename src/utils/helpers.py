"""
Utility functions and helper classes for the People Counter application
"""
import cv2
import numpy as np
import os
import time
import json
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import logging

from ..utils.config import config


def validate_video_source(source: str) -> Tuple[bool, str]:
    """
    Validate video source (file path or camera index)

    Args:
        source: Video source (file path or camera index)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not source:
        return False, "Video source cannot be empty"

    try:
        # Try to parse as integer (camera index)
        camera_index = int(source)
        if camera_index < 0:
            return False, "Camera index must be non-negative"

        # Test camera availability
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return False, f"Camera {camera_index} is not available"

        cap.release()
        return True, "Camera available"

    except ValueError:
        # Not a number, treat as file path
        if not os.path.exists(source):
            return False, f"Video file not found: {source}"

        if not source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            return False, f"Unsupported video format: {source}"

        # Try to open video file
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return False, f"Cannot open video file: {source}"

        # Check video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()

        if frame_count <= 0:
            return False, "Invalid video file (no frames)"

        return True, f"Video ready: {frame_count} frames, {fps} FPS, {width}x{height}"


def get_video_info(source: str) -> Dict[str, Any]:
    """
    Get detailed information about a video source

    Args:
        source: Video source path or camera index

    Returns:
        Dictionary with video information
    """
    try:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            return {'error': 'Cannot open video source'}

        info = {
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'source': source
        }

        cap.release()
        return info

    except Exception as e:
        return {'error': str(e)}


def optimize_frame_size(frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Optimize frame size for processing

    Args:
        frame: Input frame
        target_size: Target size (width, height)

    Returns:
        Optimized frame
    """
    if target_size is None:
        # Auto-optimize based on frame size
        height, width = frame.shape[:2]

        if width > 1920 or height > 1080:
            # Downscale HD+ videos
            scale_factor = min(1920 / width, 1080 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        elif width < 640 or height < 480:
            # Upscale small videos for better detection
            scale_factor = max(640 / width, 480 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))

    else:
        frame = cv2.resize(frame, target_size)

    return frame


def draw_detection_area(frame: np.ndarray,
                       area_coords: Tuple[int, int, int, int],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
    """
    Draw detection area on frame

    Args:
        frame: Input frame
        area_coords: (left, top, right, bottom) coordinates
        color: Area color
        thickness: Line thickness

    Returns:
        Frame with detection area drawn
    """
    left, top, right, bottom = area_coords
    annotated_frame = frame.copy()

    # Draw rectangle
    cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, thickness)

    # Add corner markers for better visibility
    marker_size = 10
    cv2.line(annotated_frame, (left, top), (left + marker_size, top), color, thickness + 2)
    cv2.line(annotated_frame, (left, top), (left, top + marker_size), color, thickness + 2)
    cv2.line(annotated_frame, (right, top), (right - marker_size, top), color, thickness + 2)
    cv2.line(annotated_frame, (right, top), (right, top + marker_size), color, thickness + 2)
    cv2.line(annotated_frame, (left, bottom), (left + marker_size, bottom), color, thickness + 2)
    cv2.line(annotated_frame, (left, bottom), (left, bottom - marker_size), color, thickness + 2)
    cv2.line(annotated_frame, (right, bottom), (right - marker_size, bottom), color, thickness + 2)
    cv2.line(annotated_frame, (right, bottom), (right, bottom - marker_size), color, thickness + 2)

    return annotated_frame


def calculate_optimal_detection_area(frame: np.ndarray,
                                   area_percentage: float = 0.6) -> Tuple[int, int, int, int]:
    """
    Calculate optimal detection area based on frame dimensions

    Args:
        frame: Input frame
        area_percentage: Percentage of frame to use as detection area

    Returns:
        (left, top, right, bottom) coordinates
    """
    height, width = frame.shape[:2]

    # Calculate area dimensions
    area_width = int(width * np.sqrt(area_percentage))
    area_height = int(height * np.sqrt(area_percentage))

    # Center the area
    left = (width - area_width) // 2
    top = (height - area_height) // 2
    right = left + area_width
    bottom = top + area_height

    return (left, top, right, bottom)


def smooth_counts(count_history: List[int], window_size: int = 5) -> List[int]:
    """
    Smooth count values using moving average

    Args:
        count_history: List of count values
        window_size: Size of smoothing window

    Returns:
        List of smoothed count values
    """
    if len(count_history) < window_size:
        return count_history

    smoothed = []
    for i in range(len(count_history) - window_size + 1):
        window = count_history[i:i + window_size]
        smoothed.append(int(np.mean(window)))

    return smoothed


def calculate_count_statistics(count_history: List[int]) -> Dict[str, float]:
    """
    Calculate statistics from count history

    Args:
        count_history: List of count values over time

    Returns:
        Dictionary with statistics
    """
    if not count_history:
        return {'mean': 0, 'max': 0, 'min': 0, 'std': 0, 'trend': 0}

    counts = np.array(count_history, dtype=float)

    # Calculate trend (slope of linear regression)
    if len(counts) > 1:
        x = np.arange(len(counts))
        slope = np.polyfit(x, counts, 1)[0]
    else:
        slope = 0

    return {
        'mean': np.mean(counts),
        'max': np.max(counts),
        'min': np.min(counts),
        'std': np.std(counts),
        'trend': slope,
        'count': len(counts)
    }


def export_statistics_to_json(statistics: Dict[str, Any], filename: str):
    """
    Export statistics to JSON file

    Args:
        statistics: Statistics dictionary
        filename: Output filename
    """
    try:
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(statistics, f, indent=2, default=str)

    except Exception as e:
        logging.error(f"Failed to export statistics: {e}")


def load_statistics_from_json(filename: str) -> Dict[str, Any]:
    """
    Load statistics from JSON file

    Args:
        filename: Input filename

    Returns:
        Statistics dictionary
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load statistics: {e}")
        return {}


class FrameProcessor:
    """
    Utility class for frame processing operations
    """

    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = 0

    def should_process_frame(self) -> bool:
        """Check if current frame should be processed based on target FPS"""
        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_interval:
            self.last_frame_time = current_time
            return True
        return False

    def apply_preprocessing(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to improve detection quality

        Args:
            frame: Input frame

        Returns:
            Preprocessed frame
        """
        # Denoise if frame is noisy
        if self._is_noisy(frame):
            frame = cv2.fastNlMeansDenoisingColored(frame)

        # Enhance contrast for better detection
        if self._has_low_contrast(frame):
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return frame

    def _is_noisy(self, frame: np.ndarray, threshold: float = 50.0) -> bool:
        """Check if frame is noisy"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.std(gray) > threshold

    def _has_low_contrast(self, frame: np.ndarray, threshold: float = 10.0) -> bool:
        """Check if frame has low contrast"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.std(gray) < threshold


class DataExporter:
    """
    Utility class for exporting data in various formats
    """

    def __init__(self, base_filename: str = "people_counter_data"):
        self.base_filename = base_filename

    def export_to_csv(self, data: List[Dict[str, Any]], filename: Optional[str] = None):
        """Export data to CSV format"""
        if filename is None:
            filename = f"{self.base_filename}.csv"

        try:
            import csv
            with open(filename, 'w', newline='') as csvfile:
                if data:
                    writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
            return True
        except Exception as e:
            logging.error(f"CSV export failed: {e}")
            return False

    def export_to_json(self, data: Dict[str, Any], filename: Optional[str] = None):
        """Export data to JSON format"""
        if filename is None:
            filename = f"{self.base_filename}.json"

        try:
            with open(filename, 'w') as jsonfile:
                json.dump(data, jsonfile, indent=2, default=str)
            return True
        except Exception as e:
            logging.error(f"JSON export failed: {e}")
            return False


class NotificationManager:
    """
    Simple notification system for alerts and updates
    """

    def __init__(self):
        self.notifications = []
        self.max_notifications = 100

    def add_notification(self, message: str, level: str = "info"):
        """
        Add a notification

        Args:
            message: Notification message
            level: Notification level (info, warning, error, success)
        """
        notification = {
            'message': message,
            'level': level,
            'timestamp': time.time()
        }

        self.notifications.append(notification)

        # Keep only recent notifications
        if len(self.notifications) > self.max_notifications:
            self.notifications.pop(0)

    def get_notifications(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent notifications"""
        if last_n is None:
            return self.notifications.copy()
        else:
            return self.notifications[-last_n:]

    def clear_notifications(self):
        """Clear all notifications"""
        self.notifications.clear()


# Global utility instances
frame_processor = FrameProcessor()
data_exporter = DataExporter()
notification_manager = NotificationManager()
