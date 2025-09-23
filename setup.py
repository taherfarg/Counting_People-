"""
Setup script for Advanced People Counter
Professional Python package configuration
"""
from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

# Development requirements
dev_requirements = [
    'pytest>=7.0.0',
    'pytest-cov>=3.0.0',
    'black>=22.0.0',
    'flake8>=4.0.0',
    'mypy>=0.990',
    'sphinx>=5.0.0',
    'sphinx-rtd-theme>=1.2.0'
]

setup(
    # Basic package information
    name="advanced-people-counter",
    version="1.0.0",
    author="Computer Vision Solutions",
    author_email="support@advancedpeoplecounter.com",
    description="A state-of-the-art computer vision application for real-time people counting and tracking using YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # URLs
    url="https://github.com/your-username/advanced-people-counter",
    project_urls={
        "Documentation": "https://github.com/your-username/advanced-people-counter/docs",
        "Source": "https://github.com/your-username/advanced-people-counter",
        "Tracker": "https://github.com/your-username/advanced-people-counter/issues",
        "Download": "https://github.com/your-username/advanced-people-counter/releases",
    },

    # License
    license="MIT",
    license_files=["LICENSE"],

    # Package configuration
    packages=find_packages(where="src", include=["advanced_people_counter*"]),
    package_dir={"": "src"},

    # Dependencies
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "gpu": [
            "torch>=1.13.0+cu117",
            "torchvision>=0.14.0+cu117",
            "torchaudio>=0.13.0+cu117"
        ],
        "all": dev_requirements + [
            "torch>=1.13.0+cu117",
            "torchvision>=0.14.0+cu117",
            "torchaudio>=0.13.0+cu117"
        ]
    },

    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "people-counter=advanced_people_counter.ui.main_window:main",
            "people-counter-cli=advanced_people_counter.core.counter:main_cli",
        ],
    },

    # Package data
    include_package_data=True,
    zip_safe=False,

    # Classifiers
    classifiers=[
        # Development status
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",

        # License
        "License :: OSI Approved :: MIT License",

        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",

        # Operating systems
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",

        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video",
        "Topic :: Security",
        "Topic :: System :: Monitoring",

        # Frameworks
        "Framework :: AsyncIO",
    ],

    # Keywords
    keywords=[
        "computer-vision",
        "yolo",
        "yolov8",
        "object-detection",
        "people-counting",
        "tracking",
        "opencv",
        "realtime",
        "surveillance",
        "analytics",
        "deep-learning",
        "machine-learning",
        "artificial-intelligence",
        "video-processing"
    ],
)
