#!/usr/bin/env python3
"""
GUI usage example for Advanced People Counter
Demonstrates the PyQt6 interface
"""
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from advanced_people_counter.ui.main_window import PeopleCounterMainWindow
from advanced_people_counter.utils.logger import setup_logging
import logging


def main():
    """Launch the GUI application"""
    # Setup logging
    logger = setup_logging("gui_example")
    logger.info("Starting GUI application")

    print("🚀 Advanced People Counter - GUI Example")
    print("=" * 50)
    print("🎨 Launching PyQt6 interface...")
    print("💡 Features available in GUI:")
    print("   • Real-time video processing")
    print("   • Configurable detection areas")
    print("   • Adjustable confidence thresholds")
    print("   • Multiple tracking algorithms")
    print("   • Live statistics dashboard")
    print("   • Export functionality")
    print("-" * 50)

    try:
        from PyQt6.QtWidgets import QApplication

        # Create Qt application
        app = QApplication(sys.argv)
        app.setApplicationName("Advanced People Counter")
        app.setApplicationVersion("1.0.0")

        # Create and show main window
        window = PeopleCounterMainWindow()
        window.show()

        print("✅ GUI application started successfully!")
        print("🔧 Controls:")
        print("   • Start/Stop: Control video processing")
        print("   • Load Video: Load video files")
        print("   • Detection Settings: Adjust confidence and model")
        print("   • Detection Area: Configure counting zone")
        print("   • Export Data: Save results to CSV")
        print("\nPress Ctrl+C to exit...")

        # Start event loop
        sys.exit(app.exec())

    except ImportError as e:
        logger.error(f"PyQt6 not available: {e}")
        print("❌ PyQt6 is required for GUI mode")
        print("Install with: pip install PyQt6")
        print("\n💡 Alternative: Use command-line mode")
        print("Run: python scripts/basic_usage.py")

    except Exception as e:
        logger.error(f"GUI application failed: {e}")
        print(f"❌ Failed to start GUI: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
