#!/usr/bin/env python3
"""
Basic usage example for Advanced People Counter
Demonstrates simple command-line usage
"""
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from advanced_people_counter.core.counter import PeopleCounter
from advanced_people_counter.utils.config import config
from advanced_people_counter.utils.logger import setup_logging


def main():
    """Basic people counting example"""
    # Setup logging
    logger = setup_logging("basic_example")

    print("🚀 Advanced People Counter - Basic Example")
    print("=" * 50)

    # Create counter with default settings
    counter = PeopleCounter()

    # Load a video file (replace with your video path)
    video_path = "peoplecount1.mp4"  # Change this to your video file

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please update the video_path variable with your video file location.")
        return

    # Process video frame by frame
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 Processing video: {video_path}")
    print(f"📊 Video info: {total_frames} frames, {fps} FPS")
    print("-" * 50)

    frame_count = 0
    start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process frame
        annotated_frame, statistics = counter.process_frame(frame)

        # Display progress
        if frame_count % 100 == 0:
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress".1f"}% ({frame_count}/{total_frames}) - "
                  f"Current: {statistics.get('current_count', 0)}, "
                  f"Total: {statistics.get('total_count', 0)}")

        # Optional: Save every 100th frame
        if frame_count % 100 == 0:
            output_filename = f"output_frame_{frame_count"04d"}.jpg"
            cv2.imwrite(output_filename, annotated_frame)
            print(f"💾 Saved: {output_filename}")

    # Calculate final statistics
    total_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print("-" * 50)
    print("📈 Final Results:")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Processing time: {total_time".2f"} seconds")
    print(f"   Average FPS: {avg_fps".2f"}")
    print(f"   Final current count: {counter.current_count}")
    print(f"   Final total count: {counter.total_count}")

    # Export data
    counter.export_data("basic_usage_results.csv")
    print("💾 Results exported to: basic_usage_results.csv")

    cap.release()
    cv2.destroyAllWindows()

    print("\n✅ Basic example completed successfully!")


if __name__ == "__main__":
    main()
