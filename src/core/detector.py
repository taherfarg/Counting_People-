"""
YOLOv8 Object Detection Module
Handles person detection using YOLOv8 model
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ultralytics import YOLO
from pathlib import Path
import time
import logging

from ..utils.config import ModelConfig, config


class PersonDetector:
    """
    YOLOv8-based person detector with configurable parameters

    Features:
    - Real-time person detection
    - Configurable confidence thresholds
    - Support for multiple YOLO model sizes
    - Automatic model downloading
    """

    def __init__(self, model_config: Optional[ModelConfig] = None):
        """
        Initialize the person detector

        Args:
            model_config: Model configuration (uses global config if None)
        """
        self.model_config = model_config or config.model
        self.model: Optional[YOLO] = None
        self.class_names: List[str] = []
        self.logger = logging.getLogger(__name__)

        self._load_model()
        self._load_class_names()

    def _load_model(self):
        """Load YOLOv8 model with automatic download if needed"""
        try:
            model_path = self.model_config.model_path

            # Check if model file exists
            if not Path(model_path).exists():
                self.logger.info(f"Model file {model_path} not found. Attempting to download...")
                # Let YOLO handle the download automatically
                self.model = YOLO(model_path)
            else:
                self.model = YOLO(model_path)

            self.logger.info(f"Loaded YOLO model: {self.model_config.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Could not load YOLO model: {e}")

    def _load_class_names(self):
        """Load COCO class names for person detection"""
        try:
            # Try to load from coco.txt if it exists
            coco_file = Path("coco.txt")
            if coco_file.exists():
                with open(coco_file, 'r') as f:
                    self.class_names = [line.strip() for line in f if line.strip()]
            else:
                # Fallback to built-in COCO classes (first 80)
                self.class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ][:80]  # Limit to COCO classes

            self.person_class_id = self.class_names.index('person') if 'person' in self.class_names else 0
            self.logger.info(f"Loaded {len(self.class_names)} class names. Person class ID: {self.person_class_id}")

        except Exception as e:
            self.logger.error(f"Failed to load class names: {e}")
            # Fallback to basic setup
            self.class_names = ['person']
            self.person_class_id = 0

    def detect(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Detect persons in the given frame

        Args:
            frame: Input image/frame
            conf_threshold: Confidence threshold (overrides config value)

        Returns:
            List of detection dictionaries with keys: 'bbox', 'confidence', 'class_id'
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        confidence = conf_threshold or self.model_config.confidence_threshold

        try:
            # Run inference
            results = self.model(frame, conf=confidence, iou=self.model_config.iou_threshold)

            detections = []

            # Process results
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get class ID and check if it's a person
                        class_id = int(box.cls.item())
                        if class_id == self.person_class_id:
                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf.item()

                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                            }
                            detections.append(detection)

            return detections

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []

    def detect_and_draw(self, frame: np.ndarray,
                       conf_threshold: Optional[float] = None,
                       color: Tuple[int, int, int] = (0, 255, 0)) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detect persons and draw bounding boxes on the frame

        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            color: Bounding box color (BGR format)

        Returns:
            Tuple of (annotated_frame, detections)
        """
        detections = self.detect(frame, conf_threshold)

        # Draw detections on the frame
        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw confidence score
            label = f"Person: {confidence".2f"}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw center point
            center_x, center_y = detection['center']
            cv2.circle(annotated_frame, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)

        return annotated_frame, detections

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get model performance statistics

        Returns:
            Dictionary with performance metrics
        """
        if self.model is None:
            return {}

        try:
            # Get model info from YOLO
            model_info = {
                'model_size': Path(self.model_config.model_path).stat().st_size / (1024 * 1024),  # MB
                'input_size': self.model.model.args.get('imgsz', 640),
                'device': str(self.model.device),
                'half_precision': self.model.model.fp16 if hasattr(self.model.model, 'fp16') else False
            }

            return model_info

        except Exception as e:
            self.logger.error(f"Failed to get performance stats: {e}")
            return {}

    def set_confidence_threshold(self, threshold: float):
        """Set detection confidence threshold"""
        if 0 <= threshold <= 1:
            self.model_config.confidence_threshold = threshold
            self.logger.info(f"Confidence threshold set to {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0 and 1")

    def get_supported_models(self) -> List[str]:
        """Get list of supported YOLOv8 model sizes"""
        return ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']


class DetectionArea:
    """
    Manages detection area configuration and validation
    """

    def __init__(self, area_config=None):
        self.config = area_config or config.detection_area
        self.area_polygon = self._create_polygon()

    def _create_polygon(self) -> np.ndarray:
        """Create polygon from detection area coordinates"""
        return np.array([
            [self.config.left, self.config.top],
            [self.config.right, self.config.top],
            [self.config.right, self.config.bottom],
            [self.config.left, self.config.bottom]
        ])

    def is_point_inside(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the detection area

        Args:
            point: (x, y) coordinates

        Returns:
            True if point is inside detection area
        """
        return cv2.pointPolygonTest(self.area_polygon.astype(np.int32), point, False) >= 0

    def draw_area(self, frame: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw detection area on frame

        Args:
            frame: Input frame
            color: Area outline color
            thickness: Line thickness

        Returns:
            Frame with detection area drawn
        """
        annotated_frame = frame.copy()
        cv2.polylines(annotated_frame, [self.area_polygon.astype(np.int32)], True, color, thickness)
        return annotated_frame

    def update_area(self, left: int, top: int, right: int, bottom: int):
        """Update detection area coordinates"""
        self.config.left = left
        self.config.top = top
        self.config.right = right
        self.config.bottom = bottom
        self.area_polygon = self._create_polygon()


# Utility functions
def get_optimal_model_for_hardware() -> str:
    """
    Recommend optimal YOLO model based on available hardware

    Returns:
        Recommended model name
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if gpu_memory > 8:
                return 'yolov8l.pt'
            elif gpu_memory > 4:
                return 'yolov8m.pt'
            else:
                return 'yolov8s.pt'
        else:
            # CPU - use smaller model
            return 'yolov8n.pt'
    except ImportError:
        # Fallback if torch not available
        return 'yolov8s.pt'


def benchmark_detection(detector: PersonDetector, test_frames: int = 10) -> Dict[str, float]:
    """
    Benchmark detection performance

    Args:
        detector: PersonDetector instance
        test_frames: Number of frames to test

    Returns:
        Dictionary with benchmark results
    """
    import time

    processing_times = []

    # Create a dummy frame for testing
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    for _ in range(test_frames):
        start_time = time.time()
        detections = detector.detect(dummy_frame)
        end_time = time.time()

        processing_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    avg_time = np.mean(processing_times)
    fps = 1000 / avg_time if avg_time > 0 else 0

    return {
        'avg_detection_time_ms': avg_time,
        'fps': fps,
        'min_time_ms': np.min(processing_times),
        'max_time_ms': np.max(processing_times),
        'std_time_ms': np.std(processing_times)
    }
