"""
Configuration management for Advanced People Counter
Centralizes all configurable parameters for easy management and deployment
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path


@dataclass
class DetectionArea:
    """Configuration for detection area coordinates"""
    left: int = 100
    top: int = 100
    right: int = 540
    bottom: int = 380

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def is_valid(self, frame_width: int, frame_height: int) -> bool:
        return (0 <= self.left < self.right <= frame_width and
                0 <= self.top < self.bottom <= frame_height)


@dataclass
class ModelConfig:
    """YOLO model configuration"""
    model_path: str = "yolov8s.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 100
    device: str = "auto"  # auto, cpu, cuda, mps

    @property
    def model_name(self) -> str:
        return Path(self.model_path).stem


@dataclass
class TrackingConfig:
    """Object tracking configuration"""
    max_disappeared: int = 50
    disappeared_time_threshold: float = 2.0
    centroid_distance_threshold: int = 35
    tracking_algorithm: str = "centroid"  # centroid, sort, kalman

    # SORT-specific parameters
    sort_max_age: int = 30
    sort_min_hits: int = 3
    sort_iou_threshold: float = 0.3


@dataclass
class VideoConfig:
    """Video processing configuration"""
    input_source: str = "0"  # webcam, file path, or URL
    output_path: Optional[str] = None
    frame_width: int = 1020
    frame_height: int = 500
    fps: int = 30
    buffer_size: int = 32

    # Processing options
    skip_frames: int = 0
    process_every_nth_frame: int = 1
    enable_recording: bool = False


@dataclass
class UIConfig:
    """User interface configuration"""
    window_title: str = "Advanced People Counter"
    window_width: int = 1200
    window_height: int = 800
    theme: str = "dark"

    # Colors (BGR format for OpenCV)
    primary_color: Tuple[int, int, int] = (0, 168, 255)  # Blue
    secondary_color: Tuple[int, int, int] = (255, 0, 255)  # Magenta
    detection_color: Tuple[int, int, int] = (0, 0, 255)    # Red
    tracking_color: Tuple[int, int, int] = (0, 255, 0)     # Green

    # UI update intervals (ms)
    frame_update_interval: int = 33  # ~30 FPS
    stats_update_interval: int = 1000  # 1 second


@dataclass
class AnalyticsConfig:
    """Analytics and logging configuration"""
    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: str = "logs/people_counter.log"

    # Analytics storage
    enable_csv_export: bool = True
    csv_output_path: str = "data/analytics/count_data.csv"
    history_length: int = 100  # Number of data points to keep

    # Real-time statistics
    calculate_fps: bool = True
    track_processing_time: bool = True


@dataclass
class APIConfig:
    """External API configuration"""
    enable_api_integration: bool = True
    api_endpoint: str = "https://people-count.smaster.live/api.php"
    api_key: str = "47MzVlvfaUdNUDiOFlvswysdgrxW8aPeUW2oLF1NW1C92Ibfbe"
    request_timeout: float = 5.0
    max_retries: int = 3

    # Headers
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'Advanced-People-Counter/1.0'
            }


class ConfigManager:
    """Central configuration manager with environment variable support"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "data/configs/default_config.json"
        self._ensure_directories()

        # Initialize default configurations
        self.model = ModelConfig()
        self.tracking = TrackingConfig()
        self.video = VideoConfig()
        self.ui = UIConfig()
        self.analytics = AnalyticsConfig()
        self.api = APIConfig()
        self.detection_area = DetectionArea()

        # Load configuration from file if it exists
        self.load_config()

    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            "data/configs",
            "data/videos",
            "logs",
            "models"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                import json
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)

                # Update configurations
                for section_name, section_data in config_data.items():
                    if hasattr(self, section_name):
                        section = getattr(self, section_name)
                        for key, value in section_data.items():
                            if hasattr(section, key):
                                setattr(section, key, value)

                print(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")

    def save_config(self):
        """Save current configuration to file"""
        try:
            import json
            config_data = {
                'model': self.model.__dict__,
                'tracking': self.tracking.__dict__,
                'video': self.video.__dict__,
                'ui': self.ui.__dict__,
                'analytics': self.analytics.__dict__,
                'api': self.api.__dict__,
                'detection_area': self.detection_area.__dict__
            }

            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)

            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def update_from_env(self):
        """Update configuration from environment variables"""
        # Model configuration
        if os.getenv('MODEL_PATH'):
            self.model.model_path = os.getenv('MODEL_PATH')
        if os.getenv('CONFIDENCE_THRESHOLD'):
            self.model.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD'))

        # API configuration
        if os.getenv('API_ENDPOINT'):
            self.api.api_endpoint = os.getenv('API_ENDPOINT')
        if os.getenv('API_KEY'):
            self.api.api_key = os.getenv('API_KEY')

        # Video configuration
        if os.getenv('VIDEO_SOURCE'):
            self.video.input_source = os.getenv('VIDEO_SOURCE')

        # Detection area (expects comma-separated values)
        if os.getenv('DETECTION_AREA'):
            try:
                area_values = [int(x.strip()) for x in os.getenv('DETECTION_AREA').split(',')]
                if len(area_values) == 4:
                    self.detection_area = DetectionArea(*area_values)
            except ValueError:
                print("Invalid DETECTION_AREA format. Expected: left,top,right,bottom")

    def get_detection_area_polygon(self) -> List[Tuple[int, int]]:
        """Get detection area as polygon points for OpenCV"""
        return [
            (self.detection_area.left, self.detection_area.top),
            (self.detection_area.right, self.detection_area.top),
            (self.detection_area.right, self.detection_area.bottom),
            (self.detection_area.left, self.detection_area.bottom)
        ]

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Validate model configuration
        if not os.path.exists(self.model.model_path) and not self.model.model_path.endswith('.pt'):
            issues.append(f"Model file not found: {self.model.model_path}")

        if not 0 <= self.model.confidence_threshold <= 1:
            issues.append("Confidence threshold must be between 0 and 1")

        # Validate detection area
        if not self.detection_area.is_valid(self.video.frame_width, self.video.frame_height):
            issues.append("Detection area coordinates are out of frame bounds")

        # Validate tracking parameters
        if self.tracking.max_disappeared < 1:
            issues.append("max_disappeared must be positive")

        if self.tracking.disappeared_time_threshold < 0:
            issues.append("disappeared_time_threshold must be non-negative")

        return issues


# Global configuration instance
config = ConfigManager()

# Preset configurations for different use cases
PRESET_CONFIGS = {
    'high_accuracy': {
        'model': {'model_path': 'yolov8l.pt', 'confidence_threshold': 0.7},
        'tracking': {'max_disappeared': 30, 'disappeared_time_threshold': 1.0}
    },
    'high_performance': {
        'model': {'model_path': 'yolov8n.pt', 'confidence_threshold': 0.3},
        'tracking': {'max_disappeared': 60, 'disappeared_time_threshold': 3.0}
    },
    'balanced': {
        'model': {'model_path': 'yolov8s.pt', 'confidence_threshold': 0.5},
        'tracking': {'max_disappeared': 50, 'disappeared_time_threshold': 2.0}
    }
}

def load_preset(preset_name: str):
    """Load a preset configuration"""
    if preset_name in PRESET_CONFIGS:
        preset = PRESET_CONFIGS[preset_name]
        for section_name, section_updates in preset.items():
            if hasattr(config, section_name):
                section = getattr(config, section_name)
                for key, value in section_updates.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        print(f"Loaded preset configuration: {preset_name}")
    else:
        available = ', '.join(PRESET_CONFIGS.keys())
        print(f"Unknown preset '{preset_name}'. Available presets: {available}")
