#!/usr/bin/env python3
"""
Advanced configuration example for Advanced People Counter
Demonstrates custom configuration and optimization
"""
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from advanced_people_counter.core.counter import PeopleCounter
from advanced_people_counter.core.detector import PersonDetector
from advanced_people_counter.core.tracker import ObjectTracker
from advanced_people_counter.utils.config import config, ModelConfig, TrackingConfig, DetectionArea
from advanced_people_counter.utils.logger import setup_logging
import cv2
import time
import numpy as np


def create_high_accuracy_config():
    """Create configuration optimized for high accuracy"""
    print("🎯 Creating high accuracy configuration...")

    # High accuracy model settings
    model_config = ModelConfig(
        model_path="yolov8l.pt",  # Large model for better accuracy
        confidence_threshold=0.7,  # Higher confidence threshold
        iou_threshold=0.3,         # Lower IoU for better detection
        max_detections=50
    )

    # Conservative tracking settings
    tracking_config = TrackingConfig(
        max_disappeared=30,        # Shorter tracking time
        disappeared_time_threshold=1.0,  # Faster cleanup
        tracking_algorithm="sort"  # Use SORT for better tracking
    )

    # Smaller detection area for precision
    detection_area = DetectionArea(
        left=200, top=150, right=440, bottom=330
    )

    return model_config, tracking_config, detection_area


def create_high_performance_config():
    """Create configuration optimized for high performance"""
    print("⚡ Creating high performance configuration...")

    # Fast model settings
    model_config = ModelConfig(
        model_path="yolov8n.pt",  # Nano model for speed
        confidence_threshold=0.3,  # Lower confidence for speed
        iou_threshold=0.5,         # Higher IoU for fewer overlaps
        max_detections=30
    )

    # Aggressive tracking settings
    tracking_config = TrackingConfig(
        max_disappeared=60,        # Longer tracking time
        disappeared_time_threshold=3.0,  # Slower cleanup
        tracking_algorithm="centroid"    # Fast centroid tracking
    )

    # Larger detection area for coverage
    detection_area = DetectionArea(
        left=50, top=50, right=590, bottom=430
    )

    return model_config, tracking_config, detection_area


def benchmark_configurations():
    """Benchmark different configurations"""
    print("📊 Benchmarking different configurations...")

    video_path = "peoplecount1.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    configurations = [
        ("High Accuracy", create_high_accuracy_config),
        ("High Performance", create_high_performance_config),
        ("Balanced", lambda: (config.model, config.tracking, config.detection_area))
    ]

    results = []

    for config_name, config_func in configurations:
        print(f"\n🔧 Testing configuration: {config_name}")

        try:
            model_config, tracking_config, detection_area = config_func()

            # Create components with custom config
            detector = PersonDetector(model_config)
            tracker = ObjectTracker(tracking_config)
            counter = PeopleCounter(detector, tracker, DetectionArea(detection_area))

            # Benchmark processing
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            processing_times = []

            start_time = time.time()

            while frame_count < 100:  # Test first 100 frames
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start = time.time()
                counter.process_frame(frame)
                processing_times.append((time.time() - frame_start) * 1000)  # ms

                frame_count += 1

            cap.release()

            if processing_times:
                avg_time = np.mean(processing_times)
                fps = 1000 / avg_time if avg_time > 0 else 0

                result = {
                    'config': config_name,
                    'avg_time_ms': avg_time,
                    'fps': fps,
                    'frames_processed': frame_count,
                    'total_time': time.time() - start_time
                }

                results.append(result)
                print(f"   ✅ {config_name}: {fps".1f"} FPS, {avg_time".1f"}ms per frame")

        except Exception as e:
            print(f"   ❌ Error testing {config_name}: {e}")

    # Display results
    print("\n📈 Benchmark Results:")
    print("-" * 60)
    print(f"{'Configuration'"<15"} {'FPS'"<8"} {'Avg Time (ms)'"<12"} {'Frames'"<8"}")
    print("-" * 60)

    for result in results:
        print(f"{result['config']"<15"} {result['fps']"<8.1f"} {result['avg_time_ms']"<12.1f"} {result['frames_processed']"<8"}")

    print("-" * 60)
    print("💡 Recommendation: Choose configuration based on your needs:")
    print("   • High Accuracy: Better for precise counting")
    print("   • High Performance: Better for real-time applications")
    print("   • Balanced: Good compromise for most use cases")


def custom_processing_example():
    """Example of custom processing pipeline"""
    print("\n🎨 Custom Processing Example")

    # Create custom configuration
    model_config, tracking_config, detection_area = create_high_accuracy_config()

    # Create counter with custom settings
    detector = PersonDetector(model_config)
    tracker = ObjectTracker(tracking_config)
    counter = PeopleCounter(detector, tracker, DetectionArea(detection_area))

    video_path = "peoplecount1.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    print("🎬 Processing with custom configuration...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Custom preprocessing (example)
        # Apply some image enhancement
        enhanced_frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

        # Process frame
        annotated_frame, statistics = counter.process_frame(enhanced_frame)

        # Display results every 50 frames
        if frame_count % 50 == 0:
            print(f"Frame {frame_count}: Current={statistics.get('current_count', 0)}, "
                  f"Total={statistics.get('total_count', 0)}")

        # Stop after processing 300 frames for demo
        if frame_count >= 300:
            break

    cap.release()

    print(f"\n✅ Custom processing completed!")
    print(f"Final count: {counter.total_count}")
    print("Custom configuration allows for:")
    print("• Different model sizes and thresholds")
    print("• Custom tracking algorithms")
    print("• Adjustable detection areas")
    print("• Pre/post-processing hooks")


def main():
    """Main function demonstrating advanced configuration"""
    print("🚀 Advanced People Counter - Configuration Examples")
    print("=" * 60)

    # Benchmark different configurations
    benchmark_configurations()

    # Demonstrate custom processing
    custom_processing_example()

    print("\n" + "=" * 60)
    print("🎉 Advanced configuration examples completed!")
    print("\n💡 Tips for optimal configuration:")
    print("1. Use 'yolov8n.pt' for fastest processing")
    print("2. Use 'yolov8l.pt' for highest accuracy")
    print("3. Adjust confidence threshold based on lighting conditions")
    print("4. Tune detection area to focus on relevant regions")
    print("5. Choose tracking algorithm based on object density")
    print("6. Monitor performance and adjust as needed")


if __name__ == "__main__":
    main()
