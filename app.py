import sys
import cv2
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.spatial import distance as dist
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QLabel, QPushButton,
                             QSlider, QSpinBox, QFileDialog)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

@dataclass
class DetectionArea:
    left: int
    top: int
    right: int
    bottom: int

class PeopleTracker:
    def __init__(self, max_disappeared: int = 50, disappeared_time_threshold: float = 2.0):
        self.next_object_id = 0
        self.objects: Dict[int, np.ndarray] = {}
        self.disappeared: Dict[int, int] = {}
        self.last_detected: Dict[int, float] = {}
        self.max_disappeared = max_disappeared
        self.disappeared_time_threshold = disappeared_time_threshold

    def register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.last_detected[self.next_object_id] = time.time()
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.last_detected[object_id]

    def update(self, rects: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int]]:
        if not rects:
            current_time = time.time()
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if (current_time - self.last_detected[object_id]) > self.disappeared_time_threshold:
                    self.deregister(object_id)
            return [(object_id, *self.objects[object_id]) for object_id in self.objects]

        input_centroids = np.array([(int((startX + endX) / 2.0), int((startY + endY) / 2.0)) for (startX, startY, endX, endY) in rects])

        if not self.objects:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                self.last_detected[object_id] = time.time()
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if (time.time() - self.last_detected[object_id]) > self.disappeared_time_threshold:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return [(object_id, *self.objects[object_id]) for object_id in self.objects]

class PeopleCounter:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = PeopleTracker()
        self.people_detected: Dict[int, Tuple[int, int]] = {}
        self.detected = set()
        self.total_count = 0
        self.detection_area = DetectionArea(100, 100, 540, 380)
        self.confidence_threshold = 0.5
        self.history = []  # To store historical data for graphs

    def set_detection_area(self, left: int, top: int, right: int, bottom: int):
        self.detection_area = DetectionArea(left, top, right, bottom)

    def set_confidence_threshold(self, threshold: float):
        self.confidence_threshold = threshold

    def is_in_detection_area(self, point: Tuple[int, int]) -> bool:
        x, y = point
        return (self.detection_area.left <= x <= self.detection_area.right and
                self.detection_area.top <= y <= self.detection_area.bottom)

    def draw_detection(self, frame: np.ndarray, id: int, point: Tuple[int, int], color: Tuple[int, int, int] = (0, 0, 255)) -> None:
        cx, cy = point
        cv2.rectangle(frame, (cx - 10, cy - 10), (cx + 10, cy + 10), color, 2)
        cv2.circle(frame, point, 4, (255, 0, 255), -1)
        cv2.putText(frame, str(id), (cx + 15, cy - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)

    def draw_detection_area(self, frame: np.ndarray) -> None:
        cv2.rectangle(frame, (self.detection_area.left, self.detection_area.top),
                      (self.detection_area.right, self.detection_area.bottom), (255, 0, 0), 2)

    def draw_count(self, frame: np.ndarray) -> None:
        cv2.putText(frame, f'Current count: {len(self.detected)}', (20, 44),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Total count: {self.total_count}', (20, 84),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        results = self.model(frame, conf=self.confidence_threshold)
        rects = []
        if results:
            for result in results:
                for box in result.boxes:
                    if box.cls == 0:  # YOLO class 0 is for person
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        rects.append((x1, y1, x2, y2))

        bbox_id = self.tracker.update(rects)

        self.detected.clear()  # Clear the set at the start of each frame
        for bbox in bbox_id:
            id, cx, cy = bbox
            point = (int(cx), int(cy))

            if self.is_in_detection_area(point):
                if id not in self.people_detected:
                    self.total_count += 1

                self.people_detected[id] = point
                self.draw_detection(frame, id, point)
                self.detected.add(id)
            else:
                # Draw detections outside the area in a different color
                self.draw_detection(frame, id, point, color=(0, 255, 255))

        self.draw_detection_area(frame)
        self.draw_count(frame)

        current_count = len(self.detected)
        self.history.append((time.time(), current_count))
        # Keep only the last 100 data points
        if len(self.history) > 100:
            self.history.pop(0)

        return frame, current_count

class ModernButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                border: none;
                color: #e0e0e0;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)

class Dashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.current_count_label = QLabel("Current Count: 0")
        self.total_count_label = QLabel("Total Count: 0")
        self.current_count_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #e0e0e0;")
        self.total_count_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #e0e0e0;")
        self.layout.addWidget(self.current_count_label)
        self.layout.addWidget(self.total_count_label)

        self.figure, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax.set_facecolor('#2c2c2c')
        self.figure.patch.set_facecolor('#2c2c2c')
        self.ax.grid(color='#4a4a4a', linestyle='-', linewidth=0.5)
        self.ax.set_title("People Count Over Time", fontsize=10, fontweight='bold', color='#e0e0e0')
        self.ax.set_xlabel("Time (s)", fontsize=8, color='#e0e0e0')
        self.ax.set_ylabel("Count", fontsize=8, color='#e0e0e0')
        self.ax.tick_params(colors='#e0e0e0')

    def update_chart(self, history):
        self.ax.clear()
        if history:
            times, counts = zip(*history)
            start_time = times[0]
            times = [t - start_time for t in times]
            self.ax.plot(times, counts, color='#00a8ff', linewidth=2)
            self.ax.set_xlim(0, max(times))
            self.ax.set_ylim(0, max(counts) + 1)

        self.ax.set_facecolor('#2c2c2c')
        self.ax.grid(color='#4a4a4a', linestyle='-', linewidth=0.5)
        self.ax.set_title("People Count Over Time", fontsize=10, fontweight='bold', color='#e0e0e0')
        self.ax.set_xlabel("Time (s)", fontsize=8, color='#e0e0e0')
        self.ax.set_ylabel("Count", fontsize=8, color='#e0e0e0')
        self.ax.tick_params(colors='#e0e0e0')
        self.canvas.draw()

    def update_counts(self, current_count, total_count):
        self.current_count_label.setText(f"Current Count: {current_count}")
        self.total_count_label.setText(f"Total Count: {total_count}")

class PeopleCounterUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced People Counter")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #1e1e1e;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QGridLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("background-color: #2c2c2c; border: 2px solid #00a8ff; border-radius: 10px;")
        self.layout.addWidget(self.video_label, 0, 0, 3, 3)

        self.dashboard = Dashboard()
        self.dashboard.setStyleSheet("background-color: #2c2c2c; border-radius: 10px; padding: 10px;")
        self.layout.addWidget(self.dashboard, 0, 3, 1, 1)

        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.layout.addWidget(self.controls_widget, 1, 3, 2, 1)

        self.start_button = ModernButton("Start", self)
        self.start_button.clicked.connect(self.start_video)
        self.controls_layout.addWidget(self.start_button)

        self.stop_button = ModernButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_video)
        self.controls_layout.addWidget(self.stop_button)

        self.file_button = ModernButton("Open Video", self)
        self.file_button.clicked.connect(self.open_file)
        self.controls_layout.addWidget(self.file_button)

        self.confidence_label = QLabel("Confidence Threshold: 0.50", self)
        self.confidence_label.setStyleSheet("font-size: 14px; margin-top: 10px; color: #e0e0e0;")
        self.controls_layout.addWidget(self.confidence_label)

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #4a4a4a;
                height: 8px;
                background: #2c2c2c;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #00a8ff;
                border: 1px solid #00a8ff;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        self.controls_layout.addWidget(self.confidence_slider)

        self.detection_area_label = QLabel("Detection Area:", self)
        self.detection_area_label.setStyleSheet("font-size: 14px; margin-top: 10px; color: #e0e0e0;")
        self.controls_layout.addWidget(self.detection_area_label)

        self.detection_area_layout = QGridLayout()
        self.controls_layout.addLayout(self.detection_area_layout)

        self.left_spinbox = QSpinBox(self)
        self.left_spinbox.setRange(0, 640)
        self.left_spinbox.setValue(100)
        self.detection_area_layout.addWidget(QLabel("Left:"), 0, 0)
        self.detection_area_layout.addWidget(self.left_spinbox, 0, 1)

        self.top_spinbox = QSpinBox(self)
        self.top_spinbox.setRange(0, 480)
        self.top_spinbox.setValue(100)
        self.detection_area_layout.addWidget(QLabel("Top:"), 1, 0)
        self.detection_area_layout.addWidget(self.top_spinbox, 1, 1)

        self.right_spinbox = QSpinBox(self)
        self.right_spinbox.setRange(0, 640)
        self.right_spinbox.setValue(540)
        self.detection_area_layout.addWidget(QLabel("Right:"), 0, 2)
        self.detection_area_layout.addWidget(self.right_spinbox, 0, 3)

        self.bottom_spinbox = QSpinBox(self)
        self.bottom_spinbox.setRange(0, 480)
        self.bottom_spinbox.setValue(380)
        self.detection_area_layout.addWidget(QLabel("Bottom:"), 1, 2)
        self.detection_area_layout.addWidget(self.bottom_spinbox, 1, 3)

        self.update_detection_area_button = ModernButton("Update Detection Area", self)
        self.update_detection_area_button.clicked.connect(self.update_detection_area)
        self.controls_layout.addWidget(self.update_detection_area_button)

        self.video_source = 1  # Default to webcam
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.people_counter = PeopleCounter('yolov8s.pt')

    def start_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_source)
        self.timer.start(30)  # Update every 30 ms

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if filename:
            self.video_source = filename
            self.stop_video()
            self.start_video()

    def update_confidence(self):
        value = self.confidence_slider.value() / 100
        self.confidence_label.setText(f"Confidence Threshold: {value:.2f}")
        self.people_counter.set_confidence_threshold(value)

    def update_detection_area(self):
        left = self.left_spinbox.value()
        top = self.top_spinbox.value()
        right = self.right_spinbox.value()
        bottom = self.bottom_spinbox.value()
        self.people_counter.set_detection_area(left, top, right, bottom)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            processed_frame, current_count = self.people_counter.process_frame(frame)
            
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)
            
            # Update dashboard
            self.dashboard.update_counts(current_count, self.people_counter.total_count)
            self.dashboard.update_chart(self.people_counter.history)
        else:
            self.stop_video()

def main():
    app = QApplication(sys.argv)
    window = PeopleCounterUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()