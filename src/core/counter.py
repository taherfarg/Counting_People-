"""
Main People Counter Module
Orchestrates detection, tracking, and counting functionality
"""
import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Dict, Optional, Any
from collections import deque

from .detector import PersonDetector, DetectionArea
from .tracker import ObjectTracker
from ..utils.config import config


class PeopleCounter:
    """
    Main people counting system that combines detection and tracking

    Features:
    - Real-time people counting
    - Historical data tracking
    - Configurable counting zones
    - API integration for external systems
    - Performance monitoring
    """

    def __init__(self, detector: Optional[PersonDetector] = None,
                 tracker: Optional[ObjectTracker] = None,
                 detection_area: Optional[DetectionArea] = None):
        """
        Initialize the people counter

        Args:
            detector: PersonDetector instance (creates new if None)
            tracker: ObjectTracker instance (creates new if None)
            detection_area: DetectionArea instance (creates new if None)
        """
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.detector = detector or PersonDetector()
        self.tracker = tracker or ObjectTracker()
        self.detection_area = detection_area or DetectionArea()

        # Counting state
        self.people_detected: Dict[int, Tuple[int, int]] = {}
        self.detected = set()
        self.total_count = 0
        self.current_count = 0
        self.history: List[Tuple[float, int]] = []

        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.fps_history = deque(maxlen=100)

        # API integration
        self.api_manager = APIManager() if config.api.enable_api_integration else None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame and return annotated frame with statistics

        Args:
            frame: Input frame to process

        Returns:
            Tuple of (annotated_frame, statistics_dict)
        """
        start_time = time.time()

        try:
            # Detect persons
            detections = self.detector.detect(frame)

            # Update tracker
            tracked_objects = self.tracker.update(detections)

            # Update counting logic
            self._update_counts(tracked_objects, frame.shape[:2])

            # Draw results on frame
            annotated_frame = self._draw_results(frame, tracked_objects)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            self.processing_times.append(processing_time)

            # Update statistics
            statistics = self._get_statistics()

            return annotated_frame, statistics

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame, {'error': str(e)}

    def _update_counts(self, tracked_objects: List[Tuple[int, float, float]],
                      frame_shape: Tuple[int, int]) -> None:
        """
        Update people counts based on tracked objects

        Args:
            tracked_objects: List of (object_id, x, y) tuples
            frame_shape: (height, width) of the frame
        """
        frame_height, frame_width = frame_shape

        # Clear current detection set
        self.detected.clear()

        # Process each tracked object
        for object_id, x, y in tracked_objects:
            point = (int(x), int(y))

            # Check if point is in detection area
            if self.detection_area.is_point_inside(point):
                # Check if this is a new detection
                if object_id not in self.people_detected:
                    self.total_count += 1

                    # Send API notification if enabled
                    if self.api_manager:
                        self.api_manager.send_count_update(self.total_count)

                # Update detection record
                self.people_detected[object_id] = point
                self.detected.add(object_id)

        self.current_count = len(self.detected)

        # Update history
        current_time = time.time()
        self.history.append((current_time, self.current_count))

        # Keep only recent history
        if len(self.history) > config.analytics.history_length:
            self.history.pop(0)

    def _draw_results(self, frame: np.ndarray,
                     tracked_objects: List[Tuple[int, float, float]]) -> np.ndarray:
        """
        Draw detection results on the frame

        Args:
            frame: Input frame
            tracked_objects: List of tracked objects

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        # Draw detection area
        annotated_frame = self.detection_area.draw_area(annotated_frame)

        # Draw tracked objects
        for object_id, x, y in tracked_objects:
            point = (int(x), int(y))

            if self.detection_area.is_point_inside(point):
                # Object in detection area - highlight in green
                color = config.ui.detection_color
                cv2.rectangle(annotated_frame,
                            (point[0] - 10, point[1] - 10),
                            (point[0] + 10, point[1] + 10),
                            color, 2)
                cv2.putText(annotated_frame, str(object_id),
                           (point[0] + 15, point[1] - 10),
                           cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
            else:
                # Object outside detection area - show in different color
                color = config.ui.secondary_color
                cv2.circle(annotated_frame, point, 3, color, -1)
                cv2.putText(annotated_frame, str(object_id),
                           (point[0] + 10, point[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw count statistics
        self._draw_statistics(annotated_frame)

        return annotated_frame

    def _draw_statistics(self, frame: np.ndarray) -> None:
        """Draw count statistics on the frame"""
        # Current count
        cv2.putText(frame, f'Current: {self.current_count}',
                   (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                   config.ui.primary_color, 2)

        # Total count
        cv2.putText(frame, f'Total: {self.total_count}',
                   (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                   config.ui.primary_color, 2)

        # Processing performance
        if config.analytics.calculate_fps and len(self.processing_times) > 0:
            avg_time = np.mean(self.processing_times)
            fps = 1000 / avg_time if avg_time > 0 else 0
            cv2.putText(frame, f'FPS: {fps".1f"}',
                       (20, 120), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                       (255, 255, 0), 2)

    def _get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        current_time = time.time()

        # Calculate average processing time
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0

        # Calculate FPS
        fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0

        # Get track statistics
        active_tracks = self.tracker.get_active_tracks()
        total_tracks = len(active_tracks)

        # Historical data summary
        history_duration = 0
        if len(self.history) > 1:
            history_duration = self.history[-1][0] - self.history[0][0]

        statistics = {
            'current_count': self.current_count,
            'total_count': self.total_count,
            'active_tracks': total_tracks,
            'processing_time_ms': avg_processing_time,
            'fps': fps,
            'history_duration_seconds': history_duration,
            'history_points': len(self.history),
            'people_detected': len(self.people_detected),
            'timestamp': current_time
        }

        return statistics

    def reset_counts(self):
        """Reset all counting statistics"""
        self.people_detected.clear()
        self.detected.clear()
        self.total_count = 0
        self.current_count = 0
        self.history.clear()
        self.logger.info("All counts reset")

    def get_history_data(self, last_n_points: Optional[int] = None) -> List[Tuple[float, int]]:
        """
        Get historical count data

        Args:
            last_n_points: Number of recent points to return (None for all)

        Returns:
            List of (timestamp, count) tuples
        """
        if last_n_points is None:
            return self.history.copy()
        else:
            return self.history[-last_n_points:] if len(self.history) >= last_n_points else self.history.copy()

    def export_data(self, filename: str = "people_count_data.csv"):
        """
        Export counting data to CSV file

        Args:
            filename: Output filename
        """
        try:
            import csv
            import os

            # Ensure data directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(['timestamp', 'current_count', 'total_count', 'active_tracks'])

                # Write data
                for timestamp, current_count in self.history:
                    statistics = self._get_statistics_at_time(timestamp)
                    writer.writerow([
                        timestamp,
                        current_count,
                        statistics.get('total_count', 0),
                        statistics.get('active_tracks', 0)
                    ])

            self.logger.info(f"Data exported to {filename}")

        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")

    def _get_statistics_at_time(self, timestamp: float) -> Dict[str, Any]:
        """
        Get statistics for a specific timestamp

        Args:
            timestamp: Target timestamp

        Returns:
            Statistics dictionary
        """
        # Find the closest timestamp in history
        closest_idx = 0
        min_diff = float('inf')

        for i, (hist_timestamp, _) in enumerate(self.history):
            diff = abs(hist_timestamp - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        # Reconstruct statistics for that time
        if closest_idx < len(self.history):
            _, current_count = self.history[closest_idx]
            # Approximate total count based on position in history
            total_count = sum(count for _, count in self.history[:closest_idx + 1])

            return {
                'current_count': current_count,
                'total_count': total_count,
                'active_tracks': len(self.tracker.get_active_tracks())  # This is approximate
            }

        return {}

    def set_detection_area(self, left: int, top: int, right: int, bottom: int):
        """
        Update detection area

        Args:
            left, top, right, bottom: Area coordinates
        """
        self.detection_area.update_area(left, top, right, bottom)
        self.logger.info(f"Detection area updated to ({left}, {top}, {right}, {bottom})")

    def get_detection_area(self) -> Tuple[int, int, int, int]:
        """Get current detection area coordinates"""
        return (self.detection_area.config.left,
                self.detection_area.config.top,
                self.detection_area.config.right,
                self.detection_area.config.bottom)

    def set_confidence_threshold(self, threshold: float):
        """
        Set detection confidence threshold

        Args:
            threshold: Confidence threshold (0.0 - 1.0)
        """
        self.detector.set_confidence_threshold(threshold)
        self.logger.info(f"Confidence threshold set to {threshold}")


class APIManager:
    """
    Handles API integration for external systems
    """

    def __init__(self):
        self.config = config.api
        self.session = None
        self._setup_session()

    def _setup_session(self):
        """Setup HTTP session for API requests"""
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update(self.config.headers)
            self.session.timeout = self.config.request_timeout
        except ImportError:
            self.logger.error("Requests library not available for API integration")

    def send_count_update(self, count: int):
        """
        Send count update to external API

        Args:
            count: Current total count
        """
        if not self.session or not self.config.enable_api_integration:
            return

        try:
            payload = {
                'key': self.config.api_key,
                'count': count,
                'timestamp': time.time()
            }

            response = self.session.post(
                self.config.api_endpoint,
                data=payload,
                timeout=self.config.request_timeout
            )

            if response.status_code == 200:
                self.logger.debug(f"API update sent successfully: count={count}")
            else:
                self.logger.warning(f"API update failed: {response.status_code}")

        except Exception as e:
            self.logger.error(f"Failed to send API update: {e}")

    def test_connection(self) -> bool:
        """
        Test API connection

        Returns:
            True if connection successful
        """
        if not self.session:
            return False

        try:
            response = self.session.get(
                self.config.api_endpoint.replace('/api.php', '/health'),
                timeout=self.config.request_timeout
            )
            return response.status_code == 200
        except Exception:
            return False


# Utility functions
def create_counter_from_config(config_path: Optional[str] = None) -> PeopleCounter:
    """
    Create a PeopleCounter instance from configuration file

    Args:
        config_path: Path to configuration file

    Returns:
        Configured PeopleCounter instance
    """
    if config_path:
        config_manager = type('ConfigManager', (), {})()
        config_manager.load_config(config_path)

    return PeopleCounter()


def benchmark_counter(counter: PeopleCounter, num_frames: int = 100,
                     frame_generator=None) -> Dict[str, Any]:
    """
    Benchmark people counter performance

    Args:
        counter: PeopleCounter instance
        num_frames: Number of frames to test
        frame_generator: Function to generate test frames

    Returns:
        Benchmark results dictionary
    """
    if frame_generator is None:
        # Default frame generator
        def frame_generator():
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    processing_times = []
    detection_counts = []

    for _ in range(num_frames):
        frame = frame_generator()
        start_time = time.time()

        annotated_frame, stats = counter.process_frame(frame)

        processing_time = (time.time() - start_time) * 1000  # ms
        processing_times.append(processing_time)
        detection_counts.append(stats.get('current_count', 0))

    avg_time = np.mean(processing_times)
    fps = 1000 / avg_time if avg_time > 0 else 0

    return {
        'avg_processing_time_ms': avg_time,
        'fps': fps,
        'min_time_ms': np.min(processing_times),
        'max_time_ms': np.max(processing_times),
        'std_time_ms': np.std(processing_times),
        'avg_detections': np.mean(detection_counts),
        'total_frames': num_frames
    }
