"""
PyQt6 Main Window Module
Modern GUI interface for the people counter application
"""
import sys
import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QLabel, QPushButton, QSlider, QSpinBox, QFileDialog,
                             QComboBox, QGroupBox, QCheckBox, QTextEdit, QProgressBar, QStatusBar,
                             QMessageBox, QSplitter)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon

from ..core.counter import PeopleCounter
from ..utils.config import config


class VideoThread(QThread):
    """Thread for video processing to avoid blocking the UI"""

    frame_processed = pyqtSignal(np.ndarray, dict)  # frame, statistics

    def __init__(self, counter: PeopleCounter, video_source: str = "0"):
        super().__init__()
        self.counter = counter
        self.video_source = video_source
        self.cap = None
        self.running = False

    def run(self):
        """Main video processing loop"""
        self.cap = cv2.VideoCapture(self.video_source)
        self.running = True

        if not self.cap.isOpened():
            self.frame_processed.emit(np.zeros((480, 640, 3), dtype=np.uint8), {'error': 'Cannot open video source'})
            return

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Resize frame for consistent processing
                frame = cv2.resize(frame, (640, 480))

                # Process frame
                annotated_frame, statistics = self.counter.process_frame(frame)

                # Emit the processed frame
                self.frame_processed.emit(annotated_frame, statistics)

                # Small delay to prevent overwhelming the system
                self.msleep(33)  # ~30 FPS

        except Exception as e:
            self.frame_processed.emit(np.zeros((480, 640, 3), dtype=np.uint8), {'error': str(e)})
        finally:
            if self.cap:
                self.cap.release()

    def stop(self):
        """Stop the video processing thread"""
        self.running = False
        self.wait()


class ModernButton(QPushButton):
    """Modern styled button with hover effects"""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                border: none;
                color: white;
                padding: 12px 24px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                font-weight: bold;
                margin: 4px 2px;
                border-radius: 8px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2e5f8f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)


class StatisticsWidget(QWidget):
    """Widget for displaying real-time statistics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup the statistics widget UI"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("📊 Live Statistics")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # Statistics labels
        self.current_count_label = QLabel("Current Count: 0")
        self.total_count_label = QLabel("Total Count: 0")
        self.fps_label = QLabel("Processing FPS: 0.0")
        self.tracks_label = QLabel("Active Tracks: 0")

        for label in [self.current_count_label, self.total_count_label,
                     self.fps_label, self.tracks_label]:
            label.setFont(QFont("Arial", 12))
            label.setStyleSheet("color: #333; padding: 5px; background-color: #f8f9fa; border-radius: 4px; margin: 2px;")
            layout.addWidget(label)

        # Progress bar for processing load
        self.processing_bar = QProgressBar()
        self.processing_bar.setRange(0, 100)
        self.processing_bar.setValue(0)
        layout.addWidget(self.processing_bar)

        layout.addStretch()

    def update_statistics(self, statistics: Dict[str, Any]):
        """Update statistics display"""
        self.current_count_label.setText(f"Current Count: {statistics.get('current_count', 0)}")
        self.total_count_label.setText(f"Total Count: {statistics.get('total_count', 0)}")
        self.fps_label.setText(f"Processing FPS: {statistics.get('fps', 0)".1f"}")
        self.tracks_label.setText(f"Active Tracks: {statistics.get('active_tracks', 0)}")

        # Update processing load indicator
        avg_time = statistics.get('processing_time_ms', 0)
        if avg_time > 0:
            load_percentage = min(100, (avg_time / 33) * 100)  # 33ms = 30fps
            self.processing_bar.setValue(int(load_percentage))


class ControlPanel(QWidget):
    """Control panel for application settings"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
        """Setup the control panel UI"""
        layout = QVBoxLayout(self)

        # Video controls group
        video_group = QGroupBox("🎥 Video Controls")
        video_layout = QVBoxLayout(video_group)

        self.start_button = ModernButton("▶️ Start")
        self.start_button.clicked.connect(self.parent.start_video)
        video_layout.addWidget(self.start_button)

        self.stop_button = ModernButton("⏹️ Stop")
        self.stop_button.clicked.connect(self.parent.stop_video)
        self.stop_button.setEnabled(False)
        video_layout.addWidget(self.stop_button)

        self.load_video_button = ModernButton("📁 Load Video File")
        self.load_video_button.clicked.connect(self.parent.load_video_file)
        video_layout.addWidget(self.load_video_button)

        layout.addWidget(video_group)

        # Detection settings group
        detection_group = QGroupBox("🔍 Detection Settings")
        detection_layout = QVBoxLayout(detection_group)

        # Confidence threshold
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(int(config.model.confidence_threshold * 100))
        self.confidence_slider.valueChanged.connect(self.parent.update_confidence)
        confidence_layout.addWidget(self.confidence_slider)
        self.confidence_value = QLabel(f"{config.model.confidence_threshold".2f"}")
        confidence_layout.addWidget(self.confidence_value)
        detection_layout.addLayout(confidence_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'])
        self.model_combo.setCurrentText(config.model.model_path)
        self.model_combo.currentTextChanged.connect(self.parent.change_model)
        model_layout.addWidget(self.model_combo)
        detection_layout.addLayout(model_layout)

        layout.addWidget(detection_group)

        # Detection area group
        area_group = QGroupBox("📐 Detection Area")
        area_layout = QVBoxLayout(area_group)

        # Area coordinates
        self.area_inputs = {}
        coordinates = ['Left', 'Top', 'Right', 'Bottom']

        for coord in coordinates:
            coord_layout = QHBoxLayout()
            coord_layout.addWidget(QLabel(f"{coord}:"))
            spin_box = QSpinBox()
            spin_box.setRange(0, 1920)  # Support for HD video
            spin_box.setValue(getattr(config.detection_area, coord.lower()))
            spin_box.valueChanged.connect(self.parent.update_detection_area)
            coord_layout.addWidget(spin_box)
            self.area_inputs[coord.lower()] = spin_box
            area_layout.addLayout(coord_layout)

        update_area_button = ModernButton("Update Area")
        update_area_button.clicked.connect(self.parent.update_detection_area)
        area_layout.addWidget(update_area_button)

        layout.addWidget(area_group)

        # Tracking settings group
        tracking_group = QGroupBox("🎯 Tracking Settings")
        tracking_layout = QVBoxLayout(tracking_group)

        algorithm_layout = QHBoxLayout()
        algorithm_layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(['centroid', 'sort', 'kalman'])
        self.algorithm_combo.setCurrentText(config.tracking.tracking_algorithm)
        self.algorithm_combo.currentTextChanged.connect(self.parent.change_algorithm)
        algorithm_layout.addWidget(self.algorithm_combo)
        tracking_layout.addLayout(algorithm_layout)

        # Max disappeared
        max_disappeared_layout = QHBoxLayout()
        max_disappeared_layout.addWidget(QLabel("Max Disappeared:"))
        self.max_disappeared_spin = QSpinBox()
        self.max_disappeared_spin.setRange(1, 200)
        self.max_disappeared_spin.setValue(config.tracking.max_disappeared)
        max_disappeared_layout.addWidget(self.max_disappeared_spin)
        tracking_layout.addLayout(max_disappeared_layout)

        layout.addWidget(tracking_group)

        layout.addStretch()

    def update_confidence_display(self, value: float):
        """Update confidence value display"""
        self.confidence_value.setText(f"{value".2f"}")


class VideoDisplay(QWidget):
    """Widget for displaying video feed"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup the video display UI"""
        layout = QVBoxLayout(self)

        # Video label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 2px solid #4a90e2;
                border-radius: 8px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Placeholder text
        self.video_label.setText("🎥 Video Feed\nClick Start to begin")
        layout.addWidget(self.video_label)

        # Info bar
        info_layout = QHBoxLayout()
        self.info_label = QLabel("Ready")
        self.info_label.setStyleSheet("color: #666; font-style: italic;")
        info_layout.addWidget(self.info_label)

        self.status_label = QLabel("⏸️ Stopped")
        self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        info_layout.addWidget(self.status_label)
        layout.addLayout(info_layout)

    def update_frame(self, frame: np.ndarray):
        """Update video frame display"""
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create QImage
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Create pixmap and scale
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio)

        self.video_label.setPixmap(scaled_pixmap)

    def set_status(self, status: str, color: str = "#666"):
        """Set status text and color"""
        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def set_info(self, info: str):
        """Set info text"""
        self.info_label.setText(info)


class PeopleCounterMainWindow(QMainWindow):
    """Main window for the People Counter application"""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.counter = PeopleCounter()
        self.video_thread: Optional[VideoThread] = None

        # Current settings
        self.current_video_source = "0"

        self.setup_ui()
        self.setup_logging()
        self.load_settings()

    def setup_ui(self):
        """Setup the main window UI"""
        self.setWindowTitle("Advanced People Counter - Professional Edition")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(self._get_dark_theme_style())

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Video display
        self.video_display = VideoDisplay()
        splitter.addWidget(self.video_display)

        # Right panel - Controls and statistics
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Statistics widget
        self.statistics_widget = StatisticsWidget()
        right_layout.addWidget(self.statistics_widget)

        # Control panel
        self.control_panel = ControlPanel(self)
        right_layout.addWidget(self.control_panel)

        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([800, 600])

        main_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Menu bar
        self.setup_menu_bar()

    def _get_dark_theme_style(self) -> str:
        """Get dark theme CSS styling"""
        return """
        QMainWindow {
            background-color: #2c2c2c;
        }
        QWidget {
            background-color: #2c2c2c;
            color: #e0e0e0;
            font-family: Arial, sans-serif;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #4a90e2;
            border-radius: 8px;
            margin-top: 1ex;
            background-color: #3a3a3a;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 10px 0 10px;
            color: #4a90e2;
        }
        QLabel {
            color: #e0e0e0;
        }
        QSlider::groove:horizontal {
            border: 1px solid #4a4a4a;
            height: 8px;
            background: #3a3a3a;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #4a90e2;
            border: 1px solid #4a90e2;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }
        QSpinBox {
            background-color: #3a3a3a;
            color: #e0e0e0;
            border: 1px solid #4a90e2;
            border-radius: 4px;
            padding: 5px;
        }
        QComboBox {
            background-color: #3a3a3a;
            color: #e0e0e0;
            border: 1px solid #4a90e2;
            border-radius: 4px;
            padding: 5px;
        }
        QComboBox QAbstractItemView {
            background-color: #3a3a3a;
            color: #e0e0e0;
            selection-background-color: #4a90e2;
        }
        """

    def setup_menu_bar(self):
        """Setup menu bar with application menus"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        load_video_action = file_menu.addAction('Load Video')
        load_video_action.triggered.connect(self.load_video_file)

        export_data_action = file_menu.addAction('Export Data')
        export_data_action.triggered.connect(self.export_data)

        file_menu.addSeparator()

        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)

        # Settings menu
        settings_menu = menubar.addMenu('Settings')

        reset_counts_action = settings_menu.addAction('Reset Counts')
        reset_counts_action.triggered.connect(self.reset_counts)

        settings_menu.addSeparator()

        load_config_action = settings_menu.addAction('Load Configuration')
        load_config_action.triggered.connect(self.load_configuration)

        save_config_action = settings_menu.addAction('Save Configuration')
        save_config_action.triggered.connect(self.save_configuration)

        # Help menu
        help_menu = menubar.addMenu('Help')

        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/people_counter_gui.log')
            ]
        )

    def load_settings(self):
        """Load application settings"""
        try:
            # Load configuration
            self.control_panel.confidence_slider.setValue(int(config.model.confidence_threshold * 100))
            self.control_panel.confidence_value.setText(f"{config.model.confidence_threshold".2f"}")

            # Update area inputs
            self.control_panel.area_inputs['left'].setValue(config.detection_area.left)
            self.control_panel.area_inputs['top'].setValue(config.detection_area.top)
            self.control_panel.area_inputs['right'].setValue(config.detection_area.right)
            self.control_panel.area_inputs['bottom'].setValue(config.detection_area.bottom)

            self.logger.info("Settings loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load settings: {e}")
            QMessageBox.warning(self, "Settings Error", f"Failed to load settings: {e}")

    # Event handlers
    def start_video(self):
        """Start video processing"""
        try:
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.stop()

            self.video_thread = VideoThread(self.counter, self.current_video_source)
            self.video_thread.frame_processed.connect(self.on_frame_processed)
            self.video_thread.start()

            self.control_panel.start_button.setEnabled(False)
            self.control_panel.stop_button.setEnabled(True)
            self.video_display.set_status("▶️ Running", "#27ae60")

            self.status_bar.showMessage("Video processing started")

        except Exception as e:
            self.logger.error(f"Failed to start video: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start video: {e}")

    def stop_video(self):
        """Stop video processing"""
        if self.video_thread:
            self.video_thread.stop()

        self.control_panel.start_button.setEnabled(True)
        self.control_panel.stop_button.setEnabled(False)
        self.video_display.set_status("⏸️ Stopped", "#e74c3c")

        self.status_bar.showMessage("Video processing stopped")

    def load_video_file(self):
        """Load video file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )

        if filename:
            self.current_video_source = filename
            self.video_display.set_info(f"Video: {Path(filename).name}")
            self.status_bar.showMessage(f"Loaded video: {filename}")

    def update_confidence(self):
        """Update detection confidence threshold"""
        confidence = self.control_panel.confidence_slider.value() / 100.0
        self.counter.set_confidence_threshold(confidence)
        self.control_panel.update_confidence_display(confidence)

    def change_model(self, model_name: str):
        """Change YOLO model"""
        try:
            # This would require recreating the detector
            # For now, just update the config
            config.model.model_path = model_name
            self.status_bar.showMessage(f"Model changed to: {model_name}")
        except Exception as e:
            QMessageBox.warning(self, "Model Error", f"Failed to change model: {e}")

    def change_algorithm(self, algorithm: str):
        """Change tracking algorithm"""
        try:
            config.tracking.tracking_algorithm = algorithm
            # This would require recreating the tracker
            self.status_bar.showMessage(f"Algorithm changed to: {algorithm}")
        except Exception as e:
            QMessageBox.warning(self, "Algorithm Error", f"Failed to change algorithm: {e}")

    def update_detection_area(self):
        """Update detection area"""
        try:
            left = self.control_panel.area_inputs['left'].value()
            top = self.control_panel.area_inputs['top'].value()
            right = self.control_panel.area_inputs['right'].value()
            bottom = self.control_panel.area_inputs['bottom'].value()

            self.counter.set_detection_area(left, top, right, bottom)
            self.status_bar.showMessage(f"Detection area updated: ({left}, {top}, {right}, {bottom})")

        except Exception as e:
            QMessageBox.warning(self, "Area Error", f"Failed to update detection area: {e}")

    def reset_counts(self):
        """Reset all counts"""
        reply = QMessageBox.question(
            self, 'Reset Counts',
            'Are you sure you want to reset all counts?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.counter.reset_counts()
            self.status_bar.showMessage("Counts reset")

    def export_data(self):
        """Export counting data"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "people_count_data.csv", "CSV Files (*.csv)"
        )

        if filename:
            self.counter.export_data(filename)
            self.status_bar.showMessage(f"Data exported to: {filename}")

    def load_configuration(self):
        """Load configuration from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "data/configs", "JSON Files (*.json)"
        )

        if filename:
            try:
                config_manager = type('ConfigManager', (), {'config_file': filename})
                config_manager.load_config()
                self.load_settings()
                self.status_bar.showMessage(f"Configuration loaded from: {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Config Error", f"Failed to load configuration: {e}")

    def save_configuration(self):
        """Save configuration to file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "data/configs", "JSON Files (*.json)"
        )

        if filename:
            try:
                config_manager = type('ConfigManager', (), {'config_file': filename})
                config_manager.save_config()
                self.status_bar.showMessage(f"Configuration saved to: {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Config Error", f"Failed to save configuration: {e}")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About Advanced People Counter",
            "Advanced People Counter - Professional Edition\n\n"
            "A state-of-the-art computer vision application for real-time people counting "
            "and tracking using YOLOv8.\n\n"
            "Features:\n"
            "• Real-time video processing\n"
            "• Multiple tracking algorithms\n"
            "• Configurable detection areas\n"
            "• Live statistics and analytics\n"
            "• Modern PyQt6 interface\n\n"
            "Built with Python, OpenCV, and YOLOv8"
        )

    def on_frame_processed(self, frame: np.ndarray, statistics: Dict[str, Any]):
        """Handle processed frame from video thread"""
        self.video_display.update_frame(frame)
        self.statistics_widget.update_statistics(statistics)

        # Update status
        if 'error' in statistics:
            self.video_display.set_status("❌ Error", "#e74c3c")
            self.status_bar.showMessage(f"Error: {statistics['error']}")
        else:
            self.video_display.set_status("✅ Running", "#27ae60")

    def closeEvent(self, event):
        """Handle application close event"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

        # Save configuration
        try:
            config_manager = type('ConfigManager', (), {})
            config_manager.save_config()
        except Exception as e:
            self.logger.warning(f"Failed to save configuration on exit: {e}")

        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("Advanced People Counter")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Computer Vision Solutions")

    # Create and show main window
    window = PeopleCounterMainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
