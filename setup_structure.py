#!/usr/bin/env python3
"""
Setup script to create the professional project structure
"""
import os
import shutil

def create_directory_structure():
    """Create the professional project directory structure"""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the directory structure
    directories = [
        'src/core',
        'src/ui',
        'src/utils',
        'models',
        'data/videos',
        'data/configs',
        'tests',
        'docs',
        'scripts'
    ]

    # Create all directories
    for dir_path in directories:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create __init__.py files for Python packages
    init_files = [
        'src/__init__.py',
        'src/core/__init__.py',
        'src/ui/__init__.py',
        'src/utils/__init__.py'
    ]

    for init_file in init_files:
        full_path = os.path.join(base_dir, init_file)
        if not os.path.exists(full_path):
            with open(full_path, 'w') as f:
                f.write('"""Package initialization"""\n')
            print(f"Created file: {init_file}")

    print("Project structure created successfully!")

if __name__ == "__main__":
    create_directory_structure()
