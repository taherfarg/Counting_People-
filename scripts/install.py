#!/usr/bin/env python3
"""
Installation script for Advanced People Counter
Handles dependency installation and setup
"""
import sys
import os
import subprocess
import platform
from pathlib import Path


def run_command(command, description=""):
    """Run a shell command with error handling"""
    print(f"🔧 {description}")
    print(f"   Command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        print(f"   ✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {description}")
        print(f"   Error: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"   ⏰ Timeout: {description}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False

    print("✅ Python version is compatible")
    return True


def install_requirements():
    """Install Python requirements"""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"

    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False

    print("📦 Installing Python dependencies...")

    # Install core requirements
    success = run_command([
        sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
    ], "Installing core dependencies")

    if not success:
        print("⚠️  Core installation failed. Trying individual packages...")

        # Fallback: install essential packages one by one
        essential_packages = [
            "opencv-python",
            "numpy",
            "scipy",
            "ultralytics",
            "PyQt6",
            "matplotlib",
            "pandas",
            "requests"
        ]

        for package in essential_packages:
            run_command([
                sys.executable, "-m", "pip", "install", package
            ], f"Installing {package}")

    return True


def download_models():
    """Download YOLOv8 model if not present"""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "yolov8s.pt"

    if model_path.exists():
        print("✅ YOLOv8 model already exists")
        return True

    print("🤖 Downloading YOLOv8 model...")

    try:
        from ultralytics import YOLO

        # This will automatically download the model
        model = YOLO('yolov8s.pt')
        print("✅ YOLOv8 model downloaded successfully")
        return True

    except Exception as e:
        print(f"❌ Failed to download YOLOv8 model: {e}")
        print("💡 You can download it manually from:")
        print("   https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "data/videos",
        "data/configs",
        "data/processed"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")


def create_config_files():
    """Create default configuration files"""
    config_dir = Path("data/configs")
    config_file = config_dir / "default_config.json"

    if not config_file.exists():
        print("⚙️  Creating default configuration...")

        # Import config to create default
        try:
            from src.utils.config import ConfigManager
            config_manager = ConfigManager(str(config_file))
            config_manager.save_config()
            print("✅ Default configuration created")
        except Exception as e:
            print(f"⚠️  Could not create default config: {e}")


def setup_environment():
    """Setup virtual environment if requested"""
    env_name = ".venv"

    if os.path.exists(env_name):
        print(f"✅ Virtual environment '{env_name}' already exists")
        return True

    use_venv = input("🤔 Create virtual environment? (y/n): ").lower().strip()

    if use_venv in ['y', 'yes']:
        print(f"🔧 Creating virtual environment: {env_name}")

        if platform.system() == "Windows":
            python_cmd = "python"
        else:
            python_cmd = "python3"

        success = run_command([
            python_cmd, "-m", "venv", env_name
        ], "Creating virtual environment")

        if success:
            print("✅ Virtual environment created")
            print(f"💡 Activate with: {'source .venv/bin/activate' if platform.system() != 'Windows' else '.venv\\Scripts\\activate'}")
            print(f"💡 Install in environment: {env_name}/bin/pip install -r requirements.txt")
        else:
            print("⚠️  Virtual environment creation failed, using system Python")

    return True


def main():
    """Main installation function"""
    print("🚀 Advanced People Counter - Installation Script")
    print("=" * 55)

    # Check Python version
    if not check_python_version():
        print("❌ Installation aborted due to incompatible Python version")
        return 1

    # Setup virtual environment
    setup_environment()

    # Create directories
    create_directories()

    # Install requirements
    if not install_requirements():
        print("❌ Installation failed")
        return 1

    # Download models
    download_models()

    # Create configuration files
    create_config_files()

    print("\n" + "=" * 55)
    print("🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate virtual environment (if created):")
    print("   Linux/Mac: source .venv/bin/activate")
    print("   Windows: .venv\\Scripts\\activate")
    print("\n2. Run the application:")
    print("   GUI mode: python -m advanced_people_counter.ui.main_window")
    print("   CLI mode: python scripts/basic_usage.py")
    print("\n3. Configuration files:")
    print("   • data/configs/default_config.json")
    print("   • logs/ (for application logs)")
    print("\n4. Documentation:")
    print("   • README.md (comprehensive guide)")
    print("   • scripts/ (example scripts)")
    print("\n💡 For GPU support, install PyTorch with CUDA:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117")

    return 0


if __name__ == "__main__":
    sys.exit(main())
